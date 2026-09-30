// admixer: the ADMIXTURE model and its block-relaxation Newton steps, evaluated with BLAS.
//
//   g_ij ~ Binomial(2, h_ij),  h_ij = sum_k Q_ik P_jk
//   log L = sum_ij g_ij log h_ij + (2 - g_ij) log(1 - h_ij)   (missing genotypes skipped)
//
// Parameters are packed as one vector x = [P (M x K, row major), Q (N x K, row major)].
// All passes work on tiles of SNPs x individuals with single-threaded OpenBLAS GEMM calls inside
// OpenMP threads. Rows of Q can be held fixed (supervised mode) and P can be held fixed (projection).
#pragma once
#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "io.hpp"
#include "linalg.hpp"

constexpr double PMIN = 1e-5, PMAX = 1 - 1e-5, QMIN = 1e-5;

struct Model {
  const Genotypes& D;
  int K;
  size_t nP, nQ;                         // M*K, N*K
  std::vector<double> Tbuf;              // per-thread EM accumulators for Q
  std::vector<char> qfix;                // qfix[i]: row i of Q is held fixed (supervised mode)
  bool pfix = false;                     // P is held fixed (projection mode)
  Model(const Genotypes& D_, int K_) : D(D_), K(K_), nP((size_t)D_.M * K_), nQ((size_t)D_.N * K_), qfix(D_.N, 0) {
    openblas_set_num_threads(1);  // parallelism comes from OpenMP over tiles
  }
  size_t size() const { return nP + nQ; }
  bool fixed(int i) const { return qfix[i] != 0; }

  // Makes x feasible: P clipped to [PMIN, PMAX], rows of Q projected onto the simplex.
  void project(double* x) const {
    double* P = x;
    double* Q = x + nP;
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < nP; t++) P[t] = std::min(std::max(P[t], PMIN), PMAX);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < D.N; i++) project_simplex(K, Q + (size_t)i * K, QMIN);
  }

  // Binomial likelihood of one genotype, without the constant C(2, g).
  static inline double geno_prob(int g, double h) {
    return g == 0 ? (1 - h) * (1 - h) : g == 2 ? h * h : h * (1 - h);
  }
  // Sums logs of genotype probabilities with one log() per 16 terms. Every term is at least
  // PMIN^2, so a product of 16 cannot underflow for bounds down to ~1e-9.
  struct LogAcc {
    double sum = 0, prod = 1;
    int n = 0;
    inline void add(double t) {
      prod *= t;
      if (++n == 16) sum += std::log(prod), prod = 1, n = 0;
    }
    inline double get() const { return sum + std::log(prod); }
  };

  // ADMIXTURE block relaxation (Alexander, Novembre & Lange 2009): one Newton/QP step for every row
  // of P given Q, then one for every row of Q given the new P.  y = F(x).  Returns log L(x).
  double block_relax(const double* x, double* y) {
    double ll;
    if (pfix) {
      ll = loglik(x);
      std::copy(x, x + nP, y);
    } else {
      ll = sqp_P(x + nP, x, y);
    }
    sqp_Q(y, x + nP, y + nP);
    return ll;
  }

  // Re-imposes the fixed parts of x (after an extrapolation step): fixed Q rows and, in projection
  // mode, P are copied from ref.
  void restore_fixed(double* x, const double* ref) const {
    if (pfix) std::copy(ref, ref + nP, x);
    for (int i = 0; i < D.N; i++)
      if (fixed(i)) std::copy(ref + nP + (size_t)i * K, ref + nP + (size_t)(i + 1) * K, x + nP + (size_t)i * K);
  }

  // ---------------------------------------------------------------------------------------
  // BLAS kernels. A tile covers SNPs [j0, j0+jn) x individuals [i0, i0+in).
  // Hessians are accumulated in packed upper-triangular form (KP = K(K+1)/2 columns):
  //   P rows:  H_j = sum_i w_ij q_i q_i' = (W Z)_j,   Z_i = packed(q_i q_i')
  //   Q rows:  H_i = sum_j w_ij f_j f_j' = (W' Y)_i,  Y_j = packed(f_j f_j')
  // with w = g/h^2 + (2-g)/(1-h)^2 and gradients D Q, D' P, d = g/h - (2-g)/(1-h).
  // ---------------------------------------------------------------------------------------
  static constexpr int TJ = 128, TI = 1024;  // tile shape of the SNP-parallel passes

  int KP() const { return K * (K + 1) / 2; }
  // H tile (jn x in) = P[j0..] Q[i0..]'
  void tile_h(const double* P, const double* Q, int j0, int jn, int i0, int in, double* H) const {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, jn, in, K, 1.0, P + (size_t)j0 * K, K,
                Q + (size_t)i0 * K, K, 0.0, H, in);
  }
  static inline double clamp_h(double h) { return std::min(std::max(h, PMIN), PMAX); }
  // On entry T holds the h tile. On exit T holds d, W holds w (both 0 for missing). Returns ll.
  double tile_wd(int j0, int jn, int i0, int in, double* T, double* W) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++) {
      const uint8_t* g = D.row(j0 + jj) + i0;
      double* t = T + (size_t)jj * in;
      double* w = W + (size_t)jj * in;
      LogAcc acc;
      for (int ii = 0; ii < in; ii++) {
        const int gi = g[ii];
        if (gi == 3) {
          t[ii] = w[ii] = 0;
          continue;
        }
        const double h = clamp_h(t[ii]);
        acc.add(geno_prob(gi, h));
        const double r1 = gi / h, r0 = (2 - gi) / (1 - h);
        t[ii] = r1 - r0;
        w[ii] = r1 / h + r0 / (1 - h);
      }
      ll += acc.get();
    }
    return ll;
  }
  void pack_outer(const double* v, double* z) const {
    for (int k = 0, p = 0; k < K; k++)
      for (int l = k; l < K; l++) z[p++] = v[k] * v[l];
  }
  void unpack_sym(const double* hp, double* H) const {
    for (int k = 0, p = 0; k < K; k++)
      for (int l = k; l < K; l++, p++) H[k * K + l] = H[l * K + k] = hp[p];
  }

  double loglik(const double* x) const {
    const double* P = x;
    const double* Q = x + nP;
    const int M = D.M, N = D.N, nsb = (M + TJ - 1) / TJ;
    double ll = 0;
#pragma omp parallel reduction(+ : ll)
    {
      std::vector<double> T((size_t)TJ * TI);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = sb * TJ, jn = std::min(TJ, M - j0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P, Q, j0, jn, i0, in, T.data());
          for (int jj = 0; jj < jn; jj++) {
            const uint8_t* g = D.row(j0 + jj) + i0;
            const double* t = T.data() + (size_t)jj * in;
            LogAcc acc;
            for (int ii = 0; ii < in; ii++)
              if (g[ii] != 3) acc.add(geno_prob(g[ii], clamp_h(t[ii])));
            ll += acc.get();
          }
        }
      }
    }
    return ll;
  }

  double em(const double* x, double* y, int ja = 0, int jb = -1) {
    const double* P0 = x;
    const double* Q0 = x + nP;
    double* P1 = y;
    double* Q1 = y + nP;
    if (jb < 0) jb = D.M;
    const int N = D.N, nt = omp_get_max_threads(), nsb = (jb - ja + TJ - 1) / TJ;
    Tbuf.assign((size_t)nt * nQ, 0.0);
    double ll = 0;
#pragma omp parallel reduction(+ : ll)
    {
      double* Tq = Tbuf.data() + (size_t)omp_get_thread_num() * nQ;
      std::vector<double> R0((size_t)TJ * TI), R1((size_t)TJ * TI), A(TJ * K), B(TJ * K), P1m(TJ * K);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = ja + sb * TJ, jn = std::min(TJ, jb - j0);
        const double* Pb = P0 + (size_t)j0 * K;
        for (int t = 0; t < jn * K; t++) P1m[t] = 1 - Pb[t];
        std::fill(A.begin(), A.end(), 0.0);
        std::fill(B.begin(), B.end(), 0.0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P0, Q0, j0, jn, i0, in, R0.data());
          for (int jj = 0; jj < jn; jj++) {  // R1 = g/h, R0 = (2-g)/(1-h)
            const uint8_t* g = D.row(j0 + jj) + i0;
            double* r0 = R0.data() + (size_t)jj * in;
            double* r1 = R1.data() + (size_t)jj * in;
            LogAcc acc;
            for (int ii = 0; ii < in; ii++) {
              const int gi = g[ii];
              if (gi == 3) {
                r0[ii] = r1[ii] = 0;
                continue;
              }
              const double h = clamp_h(r0[ii]);
              acc.add(geno_prob(gi, h));
              r1[ii] = gi / h;
              r0[ii] = (2 - gi) / (1 - h);
            }
            ll += acc.get();
          }
          const double* Qi = Q0 + (size_t)i0 * K;
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, R1.data(), in, Qi, K, 1.0,
                      A.data(), K);
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, R0.data(), in, Qi, K, 1.0,
                      B.data(), K);
          double* Ti = Tq + (size_t)i0 * K;
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, R1.data(), in, Pb, K, 1.0, Ti, K);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, R0.data(), in, P1m.data(), K, 1.0,
                      Ti, K);
        }
        for (int jj = 0; jj < jn; jj++)
          for (int k = 0; k < K; k++) {
            const double f = Pb[jj * K + k];
            const double num = f * A[jj * K + k], den = num + (1 - f) * B[jj * K + k];
            P1[(size_t)(j0 + jj) * K + k] = den > 0 ? std::min(std::max(num / den, PMIN), PMAX) : f;
          }
      }
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      double* qo = Q1 + (size_t)i * K;
      const double* q = Q0 + (size_t)i * K;
      if (D.nobs[i] == 0 || fixed(i)) {
        std::copy(q, q + K, qo);
        continue;
      }
      // sum_k q_k T_k = 2 x (observed genotypes in the pass), so normalising is the EM update
      double tot = 0;
      for (int k = 0; k < K; k++) {
        double s = 0;
        for (int t = 0; t < nt; t++) s += Tbuf[(size_t)t * nQ + (size_t)i * K + k];
        tot += (qo[k] = q[k] * s);
      }
      if (tot > 0)
        for (int k = 0; k < K; k++) qo[k] /= tot;
      else
        std::copy(q, q + K, qo);
      project_simplex(K, qo, QMIN);
    }
    if (pfix) std::copy(P0 + (size_t)ja * K, P0 + (size_t)jb * K, P1 + (size_t)ja * K);
    return ll;
  }

  // First-order (KKT) optimality of the full problem at x (maximise l over the box for P and the
  // simplex for each row of Q). gP_jk = sum_i d_ij q_ik, gQ_ik = sum_j d_ij f_jk. Violation per entry:
  //   P interior: |g|; at the lower bound: max(0, g); at the upper bound: max(0, -g).
  //   Q row: with lambda = mean of g over interior entries, |g_k - lambda| (interior) and
  //   max(0, g_k - lambda) (entries at the lower bound).
  // Returns the maxima divided by N (P gradients sum over individuals) and by M (Q gradients sum over SNPs).
  std::pair<double, double> kkt(const double* x) const {
    const double* P = x;
    const double* Q = x + nP;
    const int M = D.M, N = D.N, nt = omp_get_max_threads(), nsb = (M + TJ - 1) / TJ;
    std::vector<double> gP(nP, 0.0), gQ((size_t)nt * nQ, 0.0);
#pragma omp parallel
    {
      double* gq = gQ.data() + (size_t)omp_get_thread_num() * nQ;
      std::vector<double> T((size_t)TJ * TI), W((size_t)TJ * TI);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = sb * TJ, jn = std::min(TJ, M - j0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P, Q, j0, jn, i0, in, T.data());
          tile_wd(j0, jn, i0, in, T.data(), W.data());
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, T.data(), in, Q + (size_t)i0 * K, K,
                      1.0, gP.data() + (size_t)j0 * K, K);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, T.data(), in, P + (size_t)j0 * K, K,
                      1.0, gq + (size_t)i0 * K, K);
        }
      }
    }
    for (int t = 1; t < nt; t++)
      for (size_t u = 0; u < nQ; u++) gQ[u] += gQ[(size_t)t * nQ + u];
    const double tb = 1e-9;  // relative tolerance for "at the bound"
    double vp = 0, vq = 0;
    for (size_t u = 0; u < (pfix ? 0 : nP); u++) {
      const double f = P[u], g = gP[u];
      const double v = f <= PMIN * (1 + tb) ? std::max(0.0, g) : f >= PMAX - PMIN * tb ? std::max(0.0, -g) : std::fabs(g);
      vp = std::max(vp, v);
    }
    for (int i = 0; i < N; i++) {
      if (fixed(i)) continue;
      const double* q = Q + (size_t)i * K;
      const double* g = gQ.data() + (size_t)i * K;
      double lam = 0;
      int nf = 0;
      for (int k = 0; k < K; k++)
        if (q[k] > QMIN * (1 + tb)) lam += g[k], nf++;
      if (nf) lam /= nf;
      for (int k = 0; k < K; k++)
        vq = std::max(vq, q[k] > QMIN * (1 + tb) ? std::fabs(g[k] - lam) : std::max(0.0, g[k] - lam));
    }
    return {vp / N, vq / M};
  }

  double sqp_P(const double* Q, const double* P0, double* P1, int ja = 0, int jb = -1) const {
    if (jb < 0) jb = D.M;
    const int N = D.N, kp = KP(), nsb = (jb - ja + TJ - 1) / TJ;
    double ll = 0;
#pragma omp parallel reduction(+ : ll)
    {
      std::vector<double> T((size_t)TJ * TI), W((size_t)TJ * TI), Z((size_t)TI * kp), Hp(TJ * kp), G(TJ * K);
      std::vector<double> H(K * K), lo(K), hi(K), d(K);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = ja + sb * TJ, jn = std::min(TJ, jb - j0);
        std::fill(Hp.begin(), Hp.end(), 0.0);
        std::fill(G.begin(), G.end(), 0.0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          const double* Qi = Q + (size_t)i0 * K;
          tile_h(P0, Q, j0, jn, i0, in, T.data());
          ll += tile_wd(j0, jn, i0, in, T.data(), W.data());
          for (int ii = 0; ii < in; ii++) pack_outer(Qi + (size_t)ii * K, Z.data() + (size_t)ii * kp);
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, kp, in, 1.0, W.data(), in, Z.data(), kp, 1.0,
                      Hp.data(), kp);
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, T.data(), in, Qi, K, 1.0,
                      G.data(), K);
        }
        for (int jj = 0; jj < jn; jj++) {
          const double* f = P0 + (size_t)(j0 + jj) * K;
          unpack_sym(Hp.data() + (size_t)jj * kp, H.data());
          for (int k = 0; k < K; k++) lo[k] = PMIN - f[k], hi[k] = PMAX - f[k];
          qp_active_set(K, H.data(), G.data() + (size_t)jj * K, lo.data(), hi.data(), false, d.data());
          double* fo = P1 + (size_t)(j0 + jj) * K;
          for (int k = 0; k < K; k++) fo[k] = std::min(std::max(f[k] + d[k], PMIN), PMAX);
        }
      }
    }
    return ll;
  }

  void sqp_Q(const double* P, const double* Q0, double* Q1, int ja = 0, int jb = -1) const {
    if (jb < 0) jb = D.M;
    const int N = D.N, kp = KP(), nt = omp_get_max_threads();
    const int bi = std::max(32, std::min(512, N / (4 * nt) / 8 * 8));  // individuals per task
    const int bj = 512;                                                // SNPs per tile
    const int nib = (N + bi - 1) / bi;
#pragma omp parallel
    {
      std::vector<double> T((size_t)bj * bi), W((size_t)bj * bi), Y((size_t)bj * kp), Hp((size_t)bi * kp),
          G((size_t)bi * K);
      std::vector<double> H(K * K), lo(K), hi(K), d(K);
#pragma omp for schedule(dynamic, 1)
      for (int ib = 0; ib < nib; ib++) {
        const int i0 = ib * bi, in = std::min(bi, N - i0);
        std::fill(Hp.begin(), Hp.end(), 0.0);
        std::fill(G.begin(), G.end(), 0.0);
        for (int j0 = ja; j0 < jb; j0 += bj) {
          const int jn = std::min(bj, jb - j0);
          const double* Pj = P + (size_t)j0 * K;
          tile_h(P, Q0, j0, jn, i0, in, T.data());
          tile_wd(j0, jn, i0, in, T.data(), W.data());
          for (int jj = 0; jj < jn; jj++) pack_outer(Pj + (size_t)jj * K, Y.data() + (size_t)jj * kp);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, kp, jn, 1.0, W.data(), in, Y.data(), kp, 1.0,
                      Hp.data(), kp);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, T.data(), in, Pj, K, 1.0, G.data(),
                      K);
        }
        for (int ii = 0; ii < in; ii++) {
          const int i = i0 + ii;
          const double* q = Q0 + (size_t)i * K;
          double* qo = Q1 + (size_t)i * K;
          if (D.nobs[i] == 0 || fixed(i)) {
            std::copy(q, q + K, qo);
            continue;
          }
          unpack_sym(Hp.data() + (size_t)ii * kp, H.data());
          for (int k = 0; k < K; k++) lo[k] = QMIN - q[k], hi[k] = 1 - q[k];
          qp_active_set(K, H.data(), G.data() + (size_t)ii * K, lo.data(), hi.data(), true, d.data());
          for (int k = 0; k < K; k++) qo[k] = q[k] + d[k];
          project_simplex(K, qo, QMIN);
        }
      }
    }
  }
};
