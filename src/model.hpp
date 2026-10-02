// admixer: the admixture model and its block-relaxation Newton steps, evaluated with BLAS, for two data
// types (the template parameter Data):
//   Genotypes (io.hpp, PLINK):      g_ij ~ Binomial(2, h_ij)                       (ADMIXTURE's model)
//   GLData (beagle.hpp):            L_ij = sum_g GL_ijg P(g | h_ij)                 (NGSadmix's model)
// with h_ij = sum_k Q_ik P_jk. Everything here is shared; the data type provides the per-entry part:
// the log-likelihood, the EM ratios r1, r0 (g/h and (2-g)/(1-h) for called genotypes) and the Newton
// quantities d = dlogL/dh and w = -d2logL/dh2.
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

#include "linalg.hpp"

template <class Data>
struct Model {
  const Data& D;
  int K;
  size_t nP, nQ;                         // M*K, N*K
  double PMIN = 1e-5, PMAX = 1 - 1e-5;   // P in [PMIN, PMAX]; rows of Q on the simplex with Q >= QMIN
  double QMIN = 1e-5;                    // (ADMIXTURE: 1e-5; NGSadmix: 1e-9)
  std::vector<double> Tbuf;              // per-thread EM accumulators for Q
  std::vector<char> qfix;                // qfix[i]: row i of Q is held fixed (supervised mode)
  bool pfix = false;                     // P is held fixed (projection mode)
  // Newton Hessians from single-precision matrix products (w and Z/Y in float, accumulated in double).
  // The gradient, the per-entry pass and the QPs stay in double, so the fixed points do not change; the
  // Fitter switches it on for the quasi-Newton phase only.
  bool hess_float = false;
  Model(const Data& D_, int K_) : D(D_), K(K_), nP((size_t)D_.M * K_), nQ((size_t)D_.N * K_), qfix(D_.N, 0) {
    openblas_set_num_threads(1);  // parallelism comes from OpenMP over tiles
  }
  void set_bound(double b) { PMIN = QMIN = b, PMAX = 1 - b; }
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
  // with gradients D Q and D' P (D = (d_ij)).
  // ---------------------------------------------------------------------------------------
  static constexpr int TJ = 128, TI = 1024;  // tile shape of the SNP-parallel passes

  int KP() const { return K * (K + 1) / 2; }
  static double sum_in_order(const std::vector<double>& v) {
    double s = 0;
    for (double t : v) s += t;
    return s;
  }
  // H tile (jn x in) = P[j0..] Q[i0..]'
  void tile_h(const double* P, const double* Q, int j0, int jn, int i0, int in, double* H) const {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, jn, in, K, 1.0, P + (size_t)j0 * K, K,
                Q + (size_t)i0 * K, K, 0.0, H, in);
  }
  template <class TZ>
  void pack_outer(const double* v, TZ* z) const {
    for (int k = 0, p = 0; k < K; k++)
      for (int l = k; l < K; l++) z[p++] = v[k] * v[l];
  }
  void unpack_sym(const double* hp, double* H) const {
    for (int k = 0, p = 0; k < K; k++)
      for (int l = k; l < K; l++, p++) H[k * K + l] = H[l * K + k] = hp[p];
  }

  // log-likelihood of the entries in the model, without the data type's constant D.ll_offset().
  // all = true (genotype likelihoods): every entry, the missing ones included.
  double loglik(const double* x, bool all = false) const {
    const double* P = x;
    const double* Q = x + nP;
    const int M = D.M, N = D.N, nsb = (M + TJ - 1) / TJ;
    std::vector<double> part(nsb, 0.0);  // per-tile sums, added in order: results do not depend on scheduling
#pragma omp parallel
    {
      std::vector<double> T((size_t)TJ * TI);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = sb * TJ, jn = std::min(TJ, M - j0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P, Q, j0, jn, i0, in, T.data());
          part[sb] += D.tile_ll(j0, jn, i0, in, T.data(), PMIN, PMAX, all);
        }
      }
    }
    return sum_in_order(part);
  }

  // One EM step using the sites [ja, jb): their rows of P and all of Q (from those sites' statistics).
  //   P_jk <- P_jk A_jk / (P_jk A_jk + (1-P_jk) B_jk),  A = R1 Q, B = R0 Q
  //   Q_ik <- Q_ik (R1' P + R0' (1-P))_ik / (2 * #observed sites of i)
  double em(const double* x, double* y, int ja = 0, int jb = -1) {
    const double* P0 = x;
    const double* Q0 = x + nP;
    double* P1 = y;
    double* Q1 = y + nP;
    if (jb < 0) jb = D.M;
    const int N = D.N, nt = omp_get_max_threads(), nsb = (jb - ja + TJ - 1) / TJ;
    Tbuf.assign((size_t)nt * nQ, 0.0);
    std::vector<double> part(nsb, 0.0);
#pragma omp parallel
    {
      double* Tq = Tbuf.data() + (size_t)omp_get_thread_num() * nQ;
      std::vector<double> R0((size_t)TJ * TI), R1((size_t)TJ * TI), A(TJ * K), B(TJ * K), P1m(TJ * K);
      // static schedule: each thread's Q accumulator covers the same tiles in every run (reproducible sums)
#pragma omp for schedule(static)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = ja + sb * TJ, jn = std::min(TJ, jb - j0);
        const double* Pb = P0 + (size_t)j0 * K;
        for (int t = 0; t < jn * K; t++) P1m[t] = 1 - Pb[t];
        std::fill(A.begin(), A.end(), 0.0);
        std::fill(B.begin(), B.end(), 0.0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P0, Q0, j0, jn, i0, in, R0.data());
          part[sb] += D.tile_em(j0, jn, i0, in, R0.data(), R1.data(), PMIN, PMAX);
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
      // sum_k q_k T_k = 2 x (observed entries in the pass), so normalising is the EM update
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
    return sum_in_order(part);
  }

  // Gradient of log L at x: g[0..nP) = dlogL/dP = D Q, g[nP..) = dlogL/dQ = D' P.
  void gradient(const double* x, double* g) const {
    const double* P = x;
    const double* Q = x + nP;
    const int M = D.M, N = D.N, nt = omp_get_max_threads(), nsb = (M + TJ - 1) / TJ;
    std::vector<double> gQ((size_t)nt * nQ, 0.0);
    std::fill(g, g + nP, 0.0);
#pragma omp parallel
    {
      double* gq = gQ.data() + (size_t)omp_get_thread_num() * nQ;
      std::vector<double> T((size_t)TJ * TI), W((size_t)TJ * TI);
#pragma omp for schedule(static)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = sb * TJ, jn = std::min(TJ, M - j0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          tile_h(P, Q, j0, jn, i0, in, T.data());
          D.tile_wd(j0, jn, i0, in, T.data(), W.data(), PMIN, PMAX);
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, in, 1.0, T.data(), in, Q + (size_t)i0 * K, K,
                      1.0, g + (size_t)j0 * K, K);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, T.data(), in, P + (size_t)j0 * K, K,
                      1.0, gq + (size_t)i0 * K, K);
        }
      }
    }
    for (size_t u = 0; u < nQ; u++) {
      double s = 0;
      for (int t = 0; t < nt; t++) s += gQ[(size_t)t * nQ + u];
      g[nP + u] = s;
    }
  }

  // First-order (KKT) optimality of the full problem at x (maximise log L over the box for P and the
  // simplex for each row of Q), as the projected-gradient mapping with a unit step on the scaled gradients
  // (g/N for P, g/M for Q):  P: max |clamp(p + g) - p|,  Q rows: max |proj_simplex(q + g) - q|.
  // It is 0 exactly at a KKT point and is not inflated by entries just above a bound (where the gradient
  // can be large but the room to move is ~bound). Fixed parts are skipped. Returns (P, Q).
  std::pair<double, double> kkt(const double* x) const {
    const double* P = x;
    const double* Q = x + nP;
    const int M = D.M, N = D.N;
    std::vector<double> g(size());
    gradient(x, g.data());
    double vp = 0, vq = 0;
    for (size_t u = 0; u < (pfix ? 0 : nP); u++)
      vp = std::max(vp, std::fabs(std::min(std::max(P[u] + g[u] / N, PMIN), PMAX) - P[u]));
    std::vector<double> z(K);
    for (int i = 0; i < N; i++) {
      if (fixed(i) || D.nobs[i] == 0) continue;
      const double* q = Q + (size_t)i * K;
      for (int k = 0; k < K; k++) z[k] = q[k] + g[nP + (size_t)i * K + k] / M;
      project_simplex(K, z.data(), QMIN);
      for (int k = 0; k < K; k++) vq = std::max(vq, std::fabs(z[k] - q[k]));
    }
    return {vp, vq};
  }

  double sqp_P(const double* Q, const double* P0, double* P1, int ja = 0, int jb = -1) const {
    if (jb < 0) jb = D.M;
    const int N = D.N, kp = KP(), nsb = (jb - ja + TJ - 1) / TJ;
    const bool hf = hess_float;
    // Z_i = packed(q_i q_i') for all individuals, packed once per call (shared by the threads) unless it would
    // take more than 256 MB; otherwise per tile.
    const bool zonce = (size_t)N * kp * (hf ? sizeof(float) : sizeof(double)) <= ((size_t)256 << 20);
    std::vector<double> Zd(zonce && !hf ? (size_t)N * kp : 0);
    std::vector<float> Zf(zonce && hf ? (size_t)N * kp : 0);
    if (zonce) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < N; i++) {
        if (hf) pack_outer(Q + (size_t)i * K, Zf.data() + (size_t)i * kp);
        else pack_outer(Q + (size_t)i * K, Zd.data() + (size_t)i * kp);
      }
    }
    std::vector<double> part(nsb, 0.0);
#pragma omp parallel
    {
      std::vector<double> T((size_t)TJ * TI), W(hf ? 0 : (size_t)TJ * TI), Z(zonce || hf ? 0 : (size_t)TI * kp);
      std::vector<float> Wf(hf ? (size_t)TJ * TI : 0), Ztf(hf && !zonce ? (size_t)TI * kp : 0), Hf(hf ? TJ * kp : 0);
      std::vector<double> Hp(TJ * kp), G(TJ * K), H(K * K), lo(K), hi(K), d(K);
#pragma omp for schedule(dynamic, 1)
      for (int sb = 0; sb < nsb; sb++) {
        const int j0 = ja + sb * TJ, jn = std::min(TJ, jb - j0);
        std::fill(Hp.begin(), Hp.end(), 0.0);
        std::fill(G.begin(), G.end(), 0.0);
        for (int i0 = 0; i0 < N; i0 += TI) {
          const int in = std::min(TI, N - i0);
          const double* Qi = Q + (size_t)i0 * K;
          tile_h(P0, Q, j0, jn, i0, in, T.data());
          if (hf) {
            part[sb] += D.tile_wd(j0, jn, i0, in, T.data(), Wf.data(), PMIN, PMAX);
            if (!zonce)
              for (int ii = 0; ii < in; ii++) pack_outer(Qi + (size_t)ii * K, Ztf.data() + (size_t)ii * kp);
            const float* z = zonce ? Zf.data() + (size_t)i0 * kp : Ztf.data();
            // float product per tile of individuals (at most TI terms), added to the double accumulator
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, kp, in, 1.0f, Wf.data(), in, z, kp, 0.0f,
                        Hf.data(), kp);
            for (int t = 0; t < jn * kp; t++) Hp[t] += Hf[t];
          } else {
            part[sb] += D.tile_wd(j0, jn, i0, in, T.data(), W.data(), PMIN, PMAX);
            if (!zonce)
              for (int ii = 0; ii < in; ii++) pack_outer(Qi + (size_t)ii * K, Z.data() + (size_t)ii * kp);
            const double* z = zonce ? Zd.data() + (size_t)i0 * kp : Z.data();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, kp, in, 1.0, W.data(), in, z, kp, 1.0,
                        Hp.data(), kp);
          }
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
    return sum_in_order(part);
  }

  // Newton/QP step for every row of Q given P, from the sites [ja, jb). Tasks are blocks of individuals;
  // with few individuals the SNPs are split among the threads as well, and the partial Hessians and
  // gradients are summed before the QPs.
  void sqp_Q(const double* P, const double* Q0, double* Q1, int ja = 0, int jb = -1) const {
    if (jb < 0) jb = D.M;
    const int N = D.N, kp = KP(), nt = omp_get_max_threads();
    // Long individual blocks: the elementwise pass runs along rows of length bi, and Y is packed once per
    // block of individuals (bi = 1024, bj = 64 was fastest at K = 5-20 with 8 threads, 2-2.6x faster than
    // 56 x 512). The tasks (2 per thread) come from blocks of individuals x SNP splits. Each split holds a
    // copy of the Q Hessians (N x KP), which is allocated, zeroed and summed in every call, so there are at
    // most MAX_SPLITS of them: with many threads the blocks of individuals get shorter instead (64 threads,
    // N = 2000: 64 splits of 1024-blocks were 2-4x slower at K = 10-20 than the old code).
    constexpr int MAX_SPLITS = 8;
    const int want_nib = (2 * nt + MAX_SPLITS - 1) / MAX_SPLITS;
    const int bi = std::max(64, std::min(1024, (N / want_nib + 7) / 8 * 8));  // individuals per task
    const int bj = 64;                                                         // SNPs per tile
    const int nib = (N + bi - 1) / bi;
    const int nsp = std::max(1, std::min({(jb - ja) / bj, (2 * nt + nib - 1) / nib, MAX_SPLITS}));  // SNP splits
    std::vector<double> HpAll((size_t)nsp * N * kp, 0.0), GAll((size_t)nsp * N * K, 0.0);
#pragma omp parallel
    {
      const bool hf = hess_float;
      std::vector<double> T((size_t)bj * bi), W(hf ? 0 : (size_t)bj * bi), Y(hf ? 0 : (size_t)bj * kp);
      // float: w, Y and the Hessian products in single precision, flushed to the double accumulator every
      // FLUSH tiles (at most FLUSH * bj = 1024 terms summed in float)
      constexpr int FLUSH = 16;
      std::vector<float> Wf(hf ? (size_t)bj * bi : 0), Yf(hf ? (size_t)bj * kp : 0), Hf(hf ? (size_t)bi * kp : 0);
#pragma omp for schedule(dynamic, 1) collapse(2)
      for (int ib = 0; ib < nib; ib++)
        for (int sp = 0; sp < nsp; sp++) {
          const int i0 = ib * bi, in = std::min(bi, N - i0);
          const int sa = ja + (int)((long)(jb - ja) * sp / nsp), sz = ja + (int)((long)(jb - ja) * (sp + 1) / nsp);
          double* Hp = HpAll.data() + ((size_t)sp * N + i0) * kp;
          double* G = GAll.data() + ((size_t)sp * N + i0) * K;
          int nf = 0;  // tiles accumulated in Hf
          for (int j0 = sa; j0 < sz; j0 += bj) {
            const int jn = std::min(bj, sz - j0);
            const double* Pj = P + (size_t)j0 * K;
            tile_h(P, Q0, j0, jn, i0, in, T.data());
            if (hf) {
              D.tile_wd(j0, jn, i0, in, T.data(), Wf.data(), PMIN, PMAX);
              for (int jj = 0; jj < jn; jj++) pack_outer(Pj + (size_t)jj * K, Yf.data() + (size_t)jj * kp);
              cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, kp, jn, 1.0f, Wf.data(), in, Yf.data(), kp,
                          nf == 0 ? 0.0f : 1.0f, Hf.data(), kp);
              if (++nf == FLUSH || j0 + bj >= sz) {
                for (size_t t = 0; t < (size_t)in * kp; t++) Hp[t] += Hf[t];
                nf = 0;
              }
            } else {
              D.tile_wd(j0, jn, i0, in, T.data(), W.data(), PMIN, PMAX);
              for (int jj = 0; jj < jn; jj++) pack_outer(Pj + (size_t)jj * K, Y.data() + (size_t)jj * kp);
              cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, kp, jn, 1.0, W.data(), in, Y.data(), kp, 1.0, Hp,
                          kp);
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, K, jn, 1.0, T.data(), in, Pj, K, 1.0, G, K);
          }
        }
    }
#pragma omp parallel
    {
      std::vector<double> H(K * K), hp(kp), g(K), lo(K), hi(K), d(K);
#pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        const double* q = Q0 + (size_t)i * K;
        double* qo = Q1 + (size_t)i * K;
        if (D.nobs[i] == 0 || fixed(i)) {
          std::copy(q, q + K, qo);
          continue;
        }
        std::fill(hp.begin(), hp.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        for (int sp = 0; sp < nsp; sp++) {
          for (int t = 0; t < kp; t++) hp[t] += HpAll[((size_t)sp * N + i) * kp + t];
          for (int k = 0; k < K; k++) g[k] += GAll[((size_t)sp * N + i) * K + k];
        }
        unpack_sym(hp.data(), H.data());
        for (int k = 0; k < K; k++) lo[k] = QMIN - q[k], hi[k] = 1 - q[k];
        qp_active_set(K, H.data(), g.data(), lo.data(), hi.data(), true, d.data());
        for (int k = 0; k < K; k++) qo[k] = q[k] + d[k];
        project_simplex(K, qo, QMIN);
      }
    }
  }
};
