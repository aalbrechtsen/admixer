// Kernel profiler for both data types (PLINK .bed or beagle .beagle.gz input).
//  1. Wall time per call of the Model kernels: loglik, em, sqp_P, sqp_Q.
//  2. A replica of sqp_P's tile loop with runtime tile sizes and per-phase timers (h-GEMM, elementwise,
//     Z packing, Hessian GEMM, gradient GEMM, QP), summed over threads, for a sweep of tile shapes.
// build: make tools/prof_kernels
// usage: tools/prof_kernels data.bed|data.beagle.gz K threads [TJ,TI ...]
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <random>
#include <string>

#include "../src/beagle.hpp"
#include "../src/io.hpp"
#include "../src/model.hpp"

using Vec = std::vector<double>;

template <class F>
static double timeit(F f, int reps = 3) {
  f();
  const double t0 = omp_get_wtime();
  for (int r = 0; r < reps; r++) f();
  return (omp_get_wtime() - t0) / reps;
}

enum { PH_H, PH_EW, PH_PACK, PH_HESS, PH_GRAD, PH_QP, NPH };
static const char* phname[NPH] = {"h-gemm", "elementwise", "pack Z", "hess-gemm", "grad-gemm", "QP"};

// sqp_P with tile sizes TJ x TI; ph[t * NPH + p] = seconds of thread t in phase p
template <class Data>
static double sqp_P_tiled(const Model<Data>& m, const double* Q, const double* P0, double* P1, int TJ, int TI,
                          std::vector<double>& ph) {
  const auto& D = m.D;
  const int K = m.K, N = D.N, M = D.M, kp = m.KP(), nsb = (M + TJ - 1) / TJ;
  const int nt = omp_get_max_threads();
  ph.assign((size_t)nt * NPH, 0.0);
  std::vector<double> part(nsb, 0.0);
#pragma omp parallel
  {
    double* my = ph.data() + (size_t)omp_get_thread_num() * NPH;
    Vec T((size_t)TJ * TI), W((size_t)TJ * TI), Z((size_t)TI * kp), Hp((size_t)TJ * kp), G((size_t)TJ * K);
    Vec H(K * K), lo(K), hi(K), d(K);
#pragma omp for schedule(dynamic, 1)
    for (int sb = 0; sb < nsb; sb++) {
      const int j0 = sb * TJ, jn = std::min(TJ, M - j0);
      std::fill(Hp.begin(), Hp.end(), 0.0);
      std::fill(G.begin(), G.end(), 0.0);
      for (int i0 = 0; i0 < N; i0 += TI) {
        const int in = std::min(TI, N - i0);
        const double* Qi = Q + (size_t)i0 * K;
        double t = omp_get_wtime(), u;
        m.tile_h(P0, Q, j0, jn, i0, in, T.data());
        u = omp_get_wtime(), my[PH_H] += u - t, t = u;
        part[sb] += D.tile_wd(j0, jn, i0, in, T.data(), W.data(), m.PMIN, m.PMAX);
        u = omp_get_wtime(), my[PH_EW] += u - t, t = u;
        for (int ii = 0; ii < in; ii++) m.pack_outer(Qi + (size_t)ii * K, Z.data() + (size_t)ii * kp);
        u = omp_get_wtime(), my[PH_PACK] += u - t, t = u;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, kp, in, 1.0, W.data(), in, Z.data(), kp, 1.0,
                    Hp.data(), kp);
        u = omp_get_wtime(), my[PH_HESS] += u - t, t = u;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, T.data(), in, Qi, K, 1.0, G.data(), K);
        u = omp_get_wtime(), my[PH_GRAD] += u - t, t = u;
      }
      double t = omp_get_wtime();
      for (int jj = 0; jj < jn; jj++) {
        const double* f = P0 + (size_t)(j0 + jj) * K;
        m.unpack_sym(Hp.data() + (size_t)jj * kp, H.data());
        for (int k = 0; k < K; k++) lo[k] = m.PMIN - f[k], hi[k] = m.PMAX - f[k];
        qp_active_set(K, H.data(), G.data() + (size_t)jj * K, lo.data(), hi.data(), false, d.data());
        double* fo = P1 + (size_t)(j0 + jj) * K;
        for (int k = 0; k < K; k++) fo[k] = std::min(std::max(f[k] + d[k], m.PMIN), m.PMAX);
      }
      my[PH_QP] += omp_get_wtime() - t;
    }
  }
  double s = 0;
  for (double v : part) s += v;
  return s;
}

// sqp_Q with the same phase timers (tasks: blocks of individuals x SNP splits, as Model::sqp_Q)
template <class Data>
static void sqp_Q_tiled(const Model<Data>& m, const double* P, const double* Q0, double* Q1, std::vector<double>& ph) {
  const auto& D = m.D;
  const int K = m.K, N = D.N, kp = m.KP(), nt = omp_get_max_threads(), ja = 0, jb = D.M;
  constexpr int MAX_SPLITS = 8;
  const int want_nib = (2 * nt + MAX_SPLITS - 1) / MAX_SPLITS;
  const int bi = std::max(64, std::min(1024, (N / want_nib + 7) / 8 * 8)), bj = 64, nib = (N + bi - 1) / bi;
  const int nsp = std::max(1, std::min({(jb - ja) / bj, (2 * nt + nib - 1) / nib, MAX_SPLITS}));
  ph.assign((size_t)nt * NPH, 0.0);
  std::vector<double> HpAll((size_t)nsp * N * kp, 0.0), GAll((size_t)nsp * N * K, 0.0);
#pragma omp parallel
  {
    double* my = ph.data() + (size_t)omp_get_thread_num() * NPH;
    Vec T((size_t)bj * bi), W((size_t)bj * bi), Y((size_t)bj * kp);
#pragma omp for schedule(dynamic, 1) collapse(2)
    for (int ib = 0; ib < nib; ib++)
      for (int sp = 0; sp < nsp; sp++) {
        const int i0 = ib * bi, in = std::min(bi, N - i0);
        const int sa = ja + (int)((long)(jb - ja) * sp / nsp), sz = ja + (int)((long)(jb - ja) * (sp + 1) / nsp);
        double* Hp = HpAll.data() + ((size_t)sp * N + i0) * kp;
        double* G = GAll.data() + ((size_t)sp * N + i0) * K;
        for (int j0 = sa; j0 < sz; j0 += bj) {
          const int jn = std::min(bj, sz - j0);
          const double* Pj = P + (size_t)j0 * K;
          double t = omp_get_wtime(), u;
          m.tile_h(P, Q0, j0, jn, i0, in, T.data());
          u = omp_get_wtime(), my[PH_H] += u - t, t = u;
          D.tile_wd(j0, jn, i0, in, T.data(), W.data(), m.PMIN, m.PMAX);
          u = omp_get_wtime(), my[PH_EW] += u - t, t = u;
          for (int jj = 0; jj < jn; jj++) m.pack_outer(Pj + (size_t)jj * K, Y.data() + (size_t)jj * kp);
          u = omp_get_wtime(), my[PH_PACK] += u - t, t = u;
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, kp, jn, 1.0, W.data(), in, Y.data(), kp, 1.0, Hp, kp);
          u = omp_get_wtime(), my[PH_HESS] += u - t, t = u;
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, T.data(), in, Pj, K, 1.0, G, K);
          u = omp_get_wtime(), my[PH_GRAD] += u - t, t = u;
        }
      }
  }
  const double t = omp_get_wtime();
  Vec H(K * K), hp(kp), g(K), lo(K), hi(K), d(K);
  for (int i = 0; i < N; i++) {  // serial here: N is small; reported as QP wall x threads below
    const double* q = Q0 + (size_t)i * K;
    double* qo = Q1 + (size_t)i * K;
    std::fill(hp.begin(), hp.end(), 0.0);
    std::fill(g.begin(), g.end(), 0.0);
    for (int sp = 0; sp < nsp; sp++) {
      for (int u = 0; u < kp; u++) hp[u] += HpAll[((size_t)sp * N + i) * kp + u];
      for (int k = 0; k < K; k++) g[k] += GAll[((size_t)sp * N + i) * K + k];
    }
    m.unpack_sym(hp.data(), H.data());
    for (int k = 0; k < K; k++) lo[k] = m.QMIN - q[k], hi[k] = 1 - q[k];
    qp_active_set(K, H.data(), g.data(), lo.data(), hi.data(), true, d.data());
    for (int k = 0; k < K; k++) qo[k] = q[k] + d[k];
    project_simplex(K, qo, m.QMIN);
  }
  ph[PH_QP] += omp_get_wtime() - t;  // serial time, credited to thread 0
}

template <class Data>
static void profile(const Data& D, int K, int nt, std::vector<std::string> shapes) {
  Model<Data> m(D, K);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> U(0, 1);
  Vec x(m.size()), y(m.size()), y2(m.size());
  for (size_t t = 0; t < m.nP; t++) x[t] = 0.05 + 0.9 * U(rng);
  for (size_t t = m.nP; t < x.size(); t++) x[t] = 1 + 0.01 * U(rng);
  m.project(x.data());
  for (int r = 0; r < 5; r++) m.em(x.data(), y.data()), x.swap(y);  // move off the start
  if (const char* at = getenv("PROF_AT")) {  // PROF_AT=prefix: profile at prefix.Q and prefix.P(.gz), e.g. a converged fit
    const std::string pre = at;
    read_matrix(pre + ".Q", x.data() + m.nP, D.N, K);
    std::ifstream f(pre + ".P");
    read_matrix(f.good() ? pre + ".P" : pre + ".P.gz", x.data(), D.M, K);
    m.project(x.data());
    std::printf("# at %s.Q / .P, log L %.6f\n", at, m.loglik(x.data()));
  }
  const double MN = (double)D.M * D.N;
  std::printf("# M=%d N=%d K=%d threads=%d   (ns per genotype entry = wall * threads / (M N) * 1e9)\n", D.M, D.N, K,
              nt);
  auto rep = [&](const char* name, double s) {
    std::printf("%-8s wall %.3f s   %.2f ns/entry*thread\n", name, s, s * nt / MN * 1e9);
  };
  rep("loglik", timeit([&] { m.loglik(x.data()); }));
  rep("em", timeit([&] { m.em(x.data(), y.data()); }));
  rep("sqp_P", timeit([&] { m.sqp_P(x.data() + m.nP, x.data(), y.data()); }));
  rep("sqp_Q", timeit([&] { m.sqp_Q(x.data(), x.data() + m.nP, y.data() + m.nP); }));

  if (shapes.empty()) shapes = {"128,1024", "64,512", "32,512", "64,256", "32,256", "16,256", "128,128"};
  const double ref = m.sqp_P(x.data() + m.nP, x.data(), y.data());
  std::printf("\nsqp_P replica: phase time summed over threads, ns per entry\n%-10s %8s", "TJ,TI", "wall_s");
  for (int p = 0; p < NPH; p++) std::printf(" %11s", phname[p]);
  std::printf("  check\n");
  for (const auto& s : shapes) {
    const int TJ = std::atoi(s.c_str()), TI = std::atoi(s.c_str() + s.find(',') + 1);
    std::vector<double> ph, acc(NPH, 0.0);
    double ll = 0;
    const double w = timeit([&] {
      ll = sqp_P_tiled(m, x.data() + m.nP, x.data(), y2.data(), TJ, TI, ph);
      for (int t = 0; t < nt; t++)
        for (int p = 0; p < NPH; p++) acc[p] += ph[t * NPH + p];
    });
    double dmax = 0;
    for (size_t t = 0; t < m.nP; t++) dmax = std::max(dmax, std::fabs(y[t] - y2[t]));
    std::printf("%-10s %8.3f", s.c_str(), w);
    for (int p = 0; p < NPH; p++) std::printf(" %11.2f", acc[p] / 4 / MN * 1e9);
    std::printf("  dll %.1e dP %.1e\n", ll - ref, dmax);
  }
  m.sqp_Q(x.data(), x.data() + m.nP, y.data() + m.nP);
  std::vector<double> ph, acc(NPH, 0.0);
  const double w = timeit([&] {
    sqp_Q_tiled(m, x.data(), x.data() + m.nP, y2.data() + m.nP, ph);
    for (int t = 0; t < nt; t++)
      for (int p = 0; p < NPH; p++) acc[p] += ph[t * NPH + p];
  });
  double dmax = 0;
  for (size_t t = m.nP; t < x.size(); t++) dmax = std::max(dmax, std::fabs(y[t] - y2[t]));
  std::printf("\nsqp_Q replica (QP serial, its wall time is shown): ns per entry\n%-10s %8.3f", "Q", w);
  for (int p = 0; p < NPH; p++) std::printf(" %11.2f", acc[p] / 4 / MN * 1e9);
  std::printf("  dQ %.1e\n", dmax);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s data.bed|data.beagle.gz K threads [TJ,TI ...]\n", argv[0]);
    return 1;
  }
  const std::string fn = argv[1];
  const int K = std::atoi(argv[2]), nt = std::atoi(argv[3]);
  omp_set_num_threads(nt);
  std::vector<std::string> shapes;
  for (int a = 4; a < argc; a++) shapes.push_back(argv[a]);
  if (fn.size() > 3 && fn.substr(fn.size() - 3) == ".gz") {
    GLData D = read_beagle(fn, GLFilter{});
    D.exact_hess = !getenv("PROF_EM_HESS");
    std::printf("# beagle GLs, %s curvature\n", D.exact_hess ? "exact" : "EM-type");
    profile(D, K, nt, shapes);
  } else {
    profile(read_genotypes(fn), K, nt, shapes);
  }
}
