// Parental and paired ancestry of each individual given the ancestral frequencies P (the models of NGSremix's
// -bothanc, Nøhr et al. 2021). P is held fixed, so every individual is a separate small problem.
//
// Parental: the two alleles of individual i come from two parents with admixture proportions x and y (each on the
// simplex). With u = x.f_j and v = y.f_j (f_j = row j of P) the probability of the data at SNP j is
//   L = G0 (1-u)(1-v) + G1 (u(1-v) + v(1-u)) + G2 u v,
// where (G0, G1, G2) are the genotype likelihoods (one-hot for called genotypes; g counts the allele whose
// frequency is P). x = y = q is the ADMIXTURE model, and (x + y)/2 is the individual's own admixture.
// Paired: the ancestries of the two alleles at a SNP form an unordered pair (a, b) with probability pi_ab
// (K(K+1)/2 values on the simplex); L = sum_{a<=b} pi_ab c_ab with c_ab the genotype probability given
// frequencies f_a, f_b. This log-likelihood is concave in pi.
//
// Which parent an allele came from is not observed, so the parental likelihood is nearly flat in the direction that
// pulls the parents apart when they have the same ancestry (at x = y its expected curvature there is 0). Any
// optimiser crawls there. So the parental fit first screens every individual at x = y = q (one pass: the
// curvature in the directions that split the parents), searches along the most negative direction, and runs
// damped Newton only for the individuals where splitting raises log L by at least MIN_GAIN; the others keep
// x = y = q. The paired fit is damped Newton for every individual (concave; for small K).
// Log-likelihoods use the probabilities above (for called genotypes they include the factor 2 of a
// heterozygote that admixer's main log-likelihood omits), so the per-individual values of the three models are
// comparable with each other. The gains assume independent SNPs: without LD pruning they are inflated.
#pragma once
#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "beagle.hpp"
#include "io.hpp"
#include "linalg.hpp"

namespace parental {

struct Result {
  std::vector<double> par, pair;                // N x 2K (parent 1, parent 2), N x K(K+1)/2 (a <= b, row-major)
  std::vector<double> ll_par, ll_pair, ll_adm;  // per individual
  int passes_par = 0, passes_pair = 0, n_split = 0;
  double sec_par = 0, sec_pair = 0;
};

// log-units the parents' split must gain in the line search to be fitted. Without a difference between the parents,
// 2 x gain is roughly chi^2 with K - 1 degrees of freedom: a gain of 1 is noise, 10 is not (K <= 10).
constexpr double MIN_GAIN = 10.0;

constexpr double FLOOR = 1e-10;  // lower bound of every proportion
// An individual's ancestries with q_k < QMIN_FREE are held at their ADMIXTURE values (the parents' proportions and
// the pairs involving them are not estimated): it makes the problems smaller and avoids estimating proportions
// that the data say are ~0.
constexpr double QMIN_FREE = 0.01;

// The passes run on blocks of BL consecutive individuals held in SIMD lanes (parameters stored transposed, so
// one SNP's frequencies are applied to all lanes at once), times chunks of SNPs so that there are enough tasks
// for the threads. Blocks without an active individual are skipped; inactive lanes are computed and discarded.
constexpr int BL = 64;  // one cache line of a SNP's genotypes

// genotype weights (G0, G1, G2) of individuals ids[0 .. nl) at SNP j (0 for missing entries and lanes >= nl)
inline void lane_weights(const Genotypes& D, int j, const int* ids, int nl, double* w0, double* w1, double* w2) {
  alignas(64) uint8_t g[BL];
  const uint8_t* row = D.row(j);
  for (int l = 0; l < BL; l++) g[l] = l < nl ? row[ids[l]] : 3;
#pragma omp simd
  for (int l = 0; l < BL; l++) w0[l] = g[l] == 0, w1[l] = g[l] == 1, w2[l] = g[l] == 2;
}
inline void lane_weights(const GLData& D, int j, const int* ids, int nl, double* w0, double* w1, double* w2) {
  const float* r = D.row(j);
  const uint8_t* k = D.krow(j);
  for (int l = 0; l < BL; l++) {
    const bool in = l < nl && k[ids[l]];
    const float* x = r + (size_t)(in ? ids[l] : 0) * 3;
    w0[l] = in ? x[0] : 0, w1[l] = in ? x[1] : 0, w2[l] = in ? x[2] : 0;
  }
}

// Runs a pass: the active individuals, packed into blocks of BL lanes (a SNP's genotypes are one row, so reading
// scattered individuals costs little more than consecutive ones), times chunks of SNPs. task(ids, nl, ja, jb, A, lp)
// accumulates the sums of SNPs [ja, jb) into A (width x BL, lane-minor) and log L into lp (BL); then
// finish(i, sums, ll) per active individual.
template <class Task, class Finish>
void run_pass(int M, int N, const std::vector<char>& act, int width, Task task, Finish finish) {
  std::vector<int> idx;
  for (int i = 0; i < N; i++)
    if (act[i]) idx.push_back(i);
  const int na = idx.size(), nb = (na + BL - 1) / BL, nt = omp_get_max_threads();
  if (!nb) return;
  const int nc = std::max(1, std::min({(3 * nt + nb - 1) / nb, 64, std::max(1, M / 512)}));
  const size_t stride = (size_t)(width + 1) * BL;
  std::vector<double> buf((size_t)nb * nc * stride);
#pragma omp parallel for schedule(dynamic, 1)
  for (int tk = 0; tk < nb * nc; tk++) {
    const int b = tk / nc, c = tk % nc;
    double* A = buf.data() + (size_t)tk * stride;
    std::fill(A, A + stride, 0.0);
    task(idx.data() + b * BL, std::min(BL, na - b * BL), (int)((long)M * c / nc), (int)((long)M * (c + 1) / nc), A,
         A + (size_t)width * BL);
  }
#pragma omp parallel
  {
    std::vector<double> sum(width);
#pragma omp for schedule(static)
    for (int a = 0; a < na; a++) {
      const int b = a / BL, l = a % BL;
      std::fill(sum.begin(), sum.end(), 0.0);
      double ll = 0;
      for (int c = 0; c < nc; c++) {
        const double* A = buf.data() + ((size_t)b * nc + c) * stride;
        for (int k = 0; k < width; k++) sum[k] += A[(size_t)k * BL + l];
        ll += A[(size_t)width * BL + l];
      }
      finish(idx[a], sum.data(), ll);
    }
  }
}

// per-lane product of likelihoods with exponent tracking, folded into lp (log) at the end
struct LaneLog {
  double p[BL], e[BL];
  LaneLog() { std::fill(p, p + BL, 1.0), std::fill(e, e + BL, 0.0); }
  void renorm() {
    for (int l = 0; l < BL; l++) {
      int x;
      p[l] = std::frexp(p[l], &x);
      e[l] += x;
    }
  }
  void add_to(double* lp) const {
    for (int l = 0; l < BL; l++) lp[l] += std::log(p[l]) + e[l] * 0.6931471805599453;
  }
};

// log L of the parental model at Z (rows [x, y]) for the active individuals
template <class Data>
void pass_parental_ll(const Data& D, const double* F, int K, const std::vector<double>& Z,
                      const std::vector<char>& act, std::vector<double>& ll) {
  const int n2 = 2 * K;
  run_pass(
      D.M, D.N, act, 0,
      [&](const int* ids, int nl, int ja, int jb, double*, double* lp) {
        std::vector<double> X((size_t)n2 * BL, 0.0);
        for (int l = 0; l < nl; l++)
          for (int k = 0; k < n2; k++) X[(size_t)k * BL + l] = Z[(size_t)ids[l] * n2 + k];
        alignas(64) double w0[BL], w1[BL], w2[BL], u[BL], v[BL];
        LaneLog acc;
        for (int j = ja; j < jb; j++) {
          const double* f = F + (size_t)j * K;
          lane_weights(D, j, ids, nl, w0, w1, w2);
          std::fill(u, u + BL, 0.0), std::fill(v, v + BL, 0.0);
          for (int k = 0; k < K; k++) {
            const double* xk = X.data() + (size_t)k * BL;
            const double* yk = X.data() + (size_t)(K + k) * BL;
#pragma omp simd
            for (int l = 0; l < BL; l++) u[l] += xk[l] * f[k], v[l] += yk[l] * f[k];
          }
#pragma omp simd
          for (int l = 0; l < BL; l++) {
            const double e1 = w1[l] - w0[l], e2 = w0[l] - 2 * w1[l] + w2[l];
            const double L0 = w0[l] + e1 * (u[l] + v[l]) + e2 * u[l] * v[l];
            acc.p[l] *= w0[l] + w1[l] + w2[l] > 0 ? L0 : 1.0;
          }
          if ((j & 7) == 7) acc.renorm();
        }
        acc.add_to(lp);
      },
      [&](int i, const double*, double l) { ll[i] = l; });
}

// Derivatives of log L for the parental model at Z (rows [x, y]) for the active individuals: ll, the gradient
// G (N x 2K: d/dx, d/dy) and the negative Hessian in packed blocks Hp (N x 3 KP: xx, yy, xy; each block is
// sum_j w_j f_j f_j' with w = a^2, b^2, ab - c, where a = dl/du, b = dl/dv, c = (G0 - 2 G1 + G2) / L).
// The per-entry weights of SJ SNPs are stored and the sums formed by matrix products (as admixer's Q step).
constexpr int SJ = 256;
template <class Data>
void pass_parental_newton(const Data& D, const double* F, int K, const std::vector<double>& Z,
                          const std::vector<char>& act, std::vector<double>& ll, std::vector<double>& G,
                          std::vector<double>& Hp) {
  const int N = D.N, n2 = 2 * K, kp = K * (K + 1) / 2, width = n2 + 3 * kp;
  run_pass(
      D.M, N, act, width,
      [&](const int* ids, int nl, int ja, int jb, double* A, double* lp) {
        std::vector<double> X((size_t)n2 * BL, 0.0), FF((size_t)SJ * kp), W((size_t)5 * SJ * BL);
        for (int l = 0; l < nl; l++)
          for (int k = 0; k < n2; k++) X[(size_t)k * BL + l] = Z[(size_t)ids[l] * n2 + k];
        double *Ca = W.data(), *Cb = Ca + SJ * BL, *Waa = Cb + SJ * BL, *Wbb = Waa + SJ * BL, *Wab = Wbb + SJ * BL;
        alignas(64) double w0[BL], w1[BL], w2[BL], u[BL], v[BL];
        LaneLog acc;
        for (int s0 = ja; s0 < jb; s0 += SJ) {
          const int ns = std::min(SJ, jb - s0);
          for (int s = 0; s < ns; s++) {
            const int j = s0 + s;
            const double* f = F + (size_t)j * K;
            for (int r = 0, p = 0; r < K; r++)
              for (int c = r; c < K; c++, p++) FF[(size_t)s * kp + p] = f[r] * f[c];
            lane_weights(D, j, ids, nl, w0, w1, w2);
            std::fill(u, u + BL, 0.0), std::fill(v, v + BL, 0.0);
            for (int k = 0; k < K; k++) {
              const double* xk = X.data() + (size_t)k * BL;
              const double* yk = X.data() + (size_t)(K + k) * BL;
#pragma omp simd
              for (int l = 0; l < BL; l++) u[l] += xk[l] * f[k], v[l] += yk[l] * f[k];
            }
            double *ca = Ca + s * BL, *cb = Cb + s * BL, *waa = Waa + s * BL, *wbb = Wbb + s * BL, *wab = Wab + s * BL;
#pragma omp simd
            for (int l = 0; l < BL; l++) {
              // L = G0 + (G1 - G0)(u + v) + (G0 - 2 G1 + G2) u v
              const double e1 = w1[l] - w0[l], e2 = w0[l] - 2 * w1[l] + w2[l];
              const double L0 = w0[l] + e1 * (u[l] + v[l]) + e2 * u[l] * v[l], obs = w0[l] + w1[l] + w2[l] > 0;
              const double L = obs ? L0 : 1.0, il = obs ? 1 / L : 0.0;
              acc.p[l] *= L;
              const double a = (e1 + e2 * v[l]) * il, b = (e1 + e2 * u[l]) * il, c = e2 * il;
              ca[l] = a, cb[l] = b, waa[l] = a * a, wbb[l] = b * b, wab[l] = a * b - c;
            }
            if ((j & 7) == 7) acc.renorm();
          }
          const double* Fb = F + (size_t)s0 * K;
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, BL, ns, 1.0, Fb, K, Ca, BL, 1.0, A, BL);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, BL, ns, 1.0, Fb, K, Cb, BL, 1.0, A + (size_t)K * BL, BL);
          for (int t = 0; t < 3; t++)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kp, BL, ns, 1.0, FF.data(), kp, Waa + (size_t)t * SJ * BL,
                        BL, 1.0, A + (size_t)(n2 + t * kp) * BL, BL);
        }
        acc.add_to(lp);
      },
      [&](int i, const double* sum, double l) {
        ll[i] = l;
        std::copy(sum, sum + n2, G.begin() + (size_t)i * n2);
        std::copy(sum + n2, sum + width, Hp.begin() + (size_t)i * 3 * kp);
      });
}

// Derivatives of log L for the paired model at Z (rows pi, a <= b) for the active individuals: ll, gradient G
// (N x KP) and negative Hessian H (N x KP(KP+1)/2, packed upper triangle). With z1 = f, z0 = 1 - f and the
// per-SNP vectors Y0_p = z0a z0b, Y1_p = z1a z0b + z0a z1b, Y2_p = z1a z1b (p = (a, b)), C_p = G0 Y0 + G1 Y1 + G2 Y2,
// L = pi'C, the gradient is C / L and the negative Hessian C C' / L^2.
template <class Data>
void pass_paired_newton(const Data& D, const double* F, int K, const std::vector<double>& Z,
                        const std::vector<char>& act, std::vector<double>& ll, std::vector<double>& G,
                        std::vector<double>& H) {
  const int N = D.N, kp = K * (K + 1) / 2, kh = kp * (kp + 1) / 2;
  run_pass(
      D.M, N, act, kp + kh,
      [&](const int* ids, int nl, int ja, int jb, double* A, double* lp) {
        std::vector<double> P((size_t)kp * BL, 0.0), Y0(kp), Y1(kp), Y2(kp), R((size_t)kp * BL);
        for (int l = 0; l < nl; l++)
          for (int p = 0; p < kp; p++) P[(size_t)p * BL + l] = Z[(size_t)ids[l] * kp + p];
        alignas(64) double w0[BL], w1[BL], w2[BL], L[BL], il[BL];
        LaneLog acc;
        for (int s0 = ja; s0 < jb; s0 += SJ) {
          const int ns = std::min(SJ, jb - s0);
          for (int s = 0; s < ns; s++) {
            const int j = s0 + s;
            const double* f = F + (size_t)j * K;
            for (int a = 0, p = 0; a < K; a++)
              for (int b = a; b < K; b++, p++)
                Y0[p] = (1 - f[a]) * (1 - f[b]), Y2[p] = f[a] * f[b], Y1[p] = f[a] * (1 - f[b]) + (1 - f[a]) * f[b];
            lane_weights(D, j, ids, nl, w0, w1, w2);
            std::fill(L, L + BL, 0.0);
            for (int p = 0; p < kp; p++) {
              double* rp = R.data() + (size_t)p * BL;
              const double* pp = P.data() + (size_t)p * BL;
#pragma omp simd
              for (int l = 0; l < BL; l++) rp[l] = w0[l] * Y0[p] + w1[l] * Y1[p] + w2[l] * Y2[p], L[l] += pp[l] * rp[l];
            }
#pragma omp simd
            for (int l = 0; l < BL; l++) {
              const bool obs = w0[l] + w1[l] + w2[l] > 0;
              acc.p[l] *= obs ? L[l] : 1.0;
              il[l] = obs ? 1 / L[l] : 0.0;
            }
            for (int p = 0; p < kp; p++) {
              double* rp = R.data() + (size_t)p * BL;
              double* gp = A + (size_t)p * BL;
#pragma omp simd
              for (int l = 0; l < BL; l++) rp[l] *= il[l], gp[l] += rp[l];
            }
            for (int p = 0, h = 0; p < kp; p++) {
              const double* rp = R.data() + (size_t)p * BL;
              for (int q = p; q < kp; q++, h++) {
                const double* rq = R.data() + (size_t)q * BL;
                double* hp = A + (size_t)(kp + h) * BL;
#pragma omp simd
                for (int l = 0; l < BL; l++) hp[l] += rp[l] * rq[l];
              }
            }
            if ((j & 7) == 7) acc.renorm();
          }
        }
        acc.add_to(lp);
      },
      [&](int i, const double* sum, double l) {
        ll[i] = l;
        std::copy(sum, sum + kp, G.begin() + (size_t)i * kp);
        std::copy(sum + kp, sum + kp + kh, H.begin() + (size_t)i * kh);
      });
}

// min 0.5 d'Hd - g'd  s.t.  lo <= d <= hi,  sum(d[0..K)) = 0,  sum(d[K..2K)) = 0  (H: 2K x 2K, positive definite).
// Primal active set: d = 0 is feasible; the free-set KKT system has one multiplier per simplex.
inline void qp_two_simplex(int K, const double* H, const double* g, const double* lo, const double* hi, double* d) {
  const int n = 2 * K;
  std::vector<int> state(n, 0), fr(n);
  std::vector<double> A((size_t)(n + 2) * (n + 2)), b(n + 2), r(n);
  for (int i = 0; i < n; i++) {
    d[i] = 0;
    if (lo[i] >= 0) state[i] = -1, d[i] = lo[i];
    else if (hi[i] <= 0) state[i] = 1, d[i] = hi[i];
  }
  double nu[2] = {0, 0};
  for (int iter = 0; iter < 10 * n + 20; iter++) {
    int nf = 0, nfg[2] = {0, 0};
    for (int i = 0; i < n; i++)
      if (state[i] == 0) fr[nf++] = i, nfg[i / K]++;
    nu[0] = nu[1] = 0;
    if (nf > 0) {
      // rows: free variables, then one equality row per simplex that has a free variable
      int row_of[2] = {-1, -1}, ns = nf;
      for (int q = 0; q < 2; q++)
        if (nfg[q]) row_of[q] = ns++;
      std::fill(A.begin(), A.begin() + (size_t)ns * ns, 0.0);
      double sb[2] = {0, 0};
      for (int i = 0; i < n; i++)
        if (state[i] != 0) sb[i / K] += d[i];
      for (int a = 0; a < nf; a++) {
        const int ia = fr[a];
        double s = g[ia];
        for (int i = 0; i < n; i++)
          if (state[i] != 0) s -= H[ia * n + i] * d[i];
        b[a] = s;
        for (int c = 0; c < nf; c++) A[a * ns + c] = H[ia * n + fr[c]];
        const int rq = row_of[ia / K];
        A[a * ns + rq] = A[rq * ns + a] = 1;
      }
      for (int q = 0; q < 2; q++)
        if (row_of[q] >= 0) b[row_of[q]] = -sb[q];
      if (!solve_linear(ns, A.data(), b.data())) break;
      for (int q = 0; q < 2; q++)
        if (row_of[q] >= 0) nu[q] = b[row_of[q]];
      double t = 1;
      int block = -1, bside = 0;
      for (int a = 0; a < nf; a++) {
        const int i = fr[a];
        const double p = b[a] - d[i];
        if (p < 0 && d[i] + p < lo[i]) {
          const double ti = (lo[i] - d[i]) / p;
          if (ti < t) t = ti, block = i, bside = -1;
        } else if (p > 0 && d[i] + p > hi[i]) {
          const double ti = (hi[i] - d[i]) / p;
          if (ti < t) t = ti, block = i, bside = 1;
        }
      }
      for (int a = 0; a < nf; a++) d[fr[a]] += t * (b[a] - d[fr[a]]);
      if (block >= 0) {
        state[block] = bside;
        d[block] = bside < 0 ? lo[block] : hi[block];
        continue;
      }
    }
    for (int i = 0; i < n; i++) {
      double s = -g[i];
      for (int k = 0; k < n; k++) s += H[i * n + k] * d[k];
      r[i] = s;
    }
    for (int q = 0; q < 2; q++)
      if (!nfg[q]) {  // no free variable in this simplex: the multiplier that best balances its bound variables
        double s = 0;
        for (int k = q * K; k < (q + 1) * K; k++) s -= r[k];
        nu[q] = s / K;
      }
    int rel = -1;
    double worst = 1e-12;
    for (int i = 0; i < n; i++) {
      if (state[i] == 0) continue;
      const double mu = r[i] + nu[i / K];
      const double viol = state[i] < 0 ? -mu : mu;
      if (viol > worst) worst = viol, rel = i;
    }
    if (rel < 0) break;
    state[rel] = 0;
  }
}

// Screening pass at the ADMIXTURE solution x = y = q: log L (ll) and Mc = sum_j c_j f_j f_j' (packed, N x KP), with
// c = (G0 - 2 G1 + G2) / L. Moving the parents apart, x = q + t d, y = q - t d (d summing to 0), changes log L by
// -t^2 d'Mc d + O(t^4): a direction with d'Mc d < 0 (excess heterozygosity where the ancestries differ) is where
// the parents can split; with none, x = y = q is a local maximum.
template <class Data>
void pass_symmetric(const Data& D, const double* F, const double* Q, int K, const std::vector<char>& act,
                    std::vector<double>& ll, std::vector<double>& Mc) {
  const int N = D.N, kp = K * (K + 1) / 2;
  run_pass(
      D.M, N, act, kp,
      [&](const int* ids, int nl, int ja, int jb, double* A, double* lp) {
        std::vector<double> X((size_t)K * BL, 0.0), FF((size_t)SJ * kp), C((size_t)SJ * BL);
        for (int l = 0; l < nl; l++)
          for (int k = 0; k < K; k++) X[(size_t)k * BL + l] = Q[(size_t)ids[l] * K + k];
        alignas(64) double w0[BL], w1[BL], w2[BL], h[BL];
        LaneLog acc;
        for (int s0 = ja; s0 < jb; s0 += SJ) {
          const int ns = std::min(SJ, jb - s0);
          for (int s = 0; s < ns; s++) {
            const int j = s0 + s;
            const double* f = F + (size_t)j * K;
            for (int r = 0, p = 0; r < K; r++)
              for (int c = r; c < K; c++, p++) FF[(size_t)s * kp + p] = f[r] * f[c];
            lane_weights(D, j, ids, nl, w0, w1, w2);
            std::fill(h, h + BL, 0.0);
            for (int k = 0; k < K; k++) {
              const double* xk = X.data() + (size_t)k * BL;
#pragma omp simd
              for (int l = 0; l < BL; l++) h[l] += xk[l] * f[k];
            }
            double* cc = C.data() + (size_t)s * BL;
#pragma omp simd
            for (int l = 0; l < BL; l++) {
              const double e1 = w1[l] - w0[l], e2 = w0[l] - 2 * w1[l] + w2[l];
              const double L0 = w0[l] + 2 * e1 * h[l] + e2 * h[l] * h[l], obs = w0[l] + w1[l] + w2[l] > 0;
              const double L = obs ? L0 : 1.0;
              acc.p[l] *= L;
              cc[l] = obs ? e2 / L : 0.0;
            }
            if ((j & 7) == 7) acc.renorm();
          }
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kp, BL, ns, 1.0, FF.data(), kp, C.data(), BL, 1.0, A, BL);
        }
        acc.add_to(lp);
      },
      [&](int i, const double* sum, double l) {
        ll[i] = l;
        std::copy(sum, sum + kp, Mc.begin() + (size_t)i * kp);
      });
}

// Eigen-decomposition of a symmetric n x n matrix (cyclic Jacobi; n is small): A is destroyed, ev[k] are the
// eigenvalues and column k of V (row-major n x n) the eigenvectors.
inline void jacobi_eigen(int n, std::vector<double>& A, std::vector<double>& ev, std::vector<double>& V) {
  V.assign((size_t)n * n, 0.0);
  for (int k = 0; k < n; k++) V[k * n + k] = 1;
  for (int sweep = 0; sweep < 50; sweep++) {
    double off = 0, tot = 0;
    for (int r = 0; r < n; r++)
      for (int c = 0; c < n; c++) (r == c ? tot : off) += A[r * n + c] * A[r * n + c];
    if (off <= 1e-24 * (tot + off) + 1e-300) break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        const double apq = A[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double th = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const double t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; k++) {  // A <- J' A J
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq, A[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk, A[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          const double vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq, V[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
  ev.resize(n);
  for (int k = 0; k < n; k++) ev[k] = A[k * n + k];
}

// The direction d (zero sum, on the ancestries with q_k >= QMIN_FREE) of most negative curvature d'Mc d of row i, scaled
// so that q +- d stays feasible; false if there is none.
inline bool split_direction(int K, const double* q, const double* mc, double* d) {
  std::vector<int> S;
  for (int k = 0; k < K; k++)
    if (q[k] >= QMIN_FREE) S.push_back(k);
  const int m = S.size();
  if (m < 2) return false;
  // orthonormal basis of the zero-sum vectors on S (Helmert): column c = (1, ..., 1, -c, 0, ...) / sqrt(c (c + 1))
  std::vector<double> B((size_t)m * (m - 1), 0.0);
  for (int c = 1; c < m; c++) {
    const double nrm = 1 / std::sqrt((double)c * (c + 1));
    for (int r = 0; r < c; r++) B[r * (m - 1) + c - 1] = nrm;
    B[c * (m - 1) + c - 1] = -c * nrm;
  }
  auto Mel = [&](int a, int b) {  // Mc[S[a], S[b]] from the packed upper triangle
    int r = std::min(S[a], S[b]), c = std::max(S[a], S[b]);
    return mc[r * K - r * (r - 1) / 2 + (c - r)];
  };
  const int n = m - 1;
  std::vector<double> C((size_t)n * n, 0.0), ev, V;
  for (int a = 0; a < n; a++)
    for (int b = 0; b < n; b++) {
      double s = 0;
      for (int r = 0; r < m; r++)
        for (int c = 0; c < m; c++) s += B[r * n + a] * Mel(r, c) * B[c * n + b];
      C[a * n + b] = s;
    }
  jacobi_eigen(n, C, ev, V);
  const int kmin = std::min_element(ev.begin(), ev.end()) - ev.begin();
  if (!(ev[kmin] < 0)) return false;
  std::fill(d, d + K, 0.0);
  for (int r = 0; r < m; r++) {
    double s = 0;
    for (int a = 0; a < n; a++) s += B[r * n + a] * V[a * n + kmin];
    d[S[r]] = s;
  }
  double tmax = INFINITY;
  for (int k = 0; k < K; k++)
    if (d[k] != 0) tmax = std::min(tmax, (q[k] - FLOOR) / std::fabs(d[k]));
  for (int k = 0; k < K; k++) d[k] *= tmax;
  return true;
}

// Damped Newton (Levenberg-Marquardt) for every row at once. Rows have n parameters; pass(Z, act, ll, G, H) gives
// log L, gradient (n) and negative Hessian (hn values, in the model's packed form) at the proposed points, one data
// pass per iteration. A proposal that lowers log L is rejected, and a more damped step is computed from the stored
// derivatives of the current point (no extra pass); solve(i, z, g, h, mu, out) computes the step's end point.
// A row stops when an accepted step gains less than tol, or after max_iter passes.
template <class Pass, class Solve>
int newton_rows(int N, int n, int hn, std::vector<double>& Z, std::vector<double>& ll_out, const std::vector<char>& use,
                Pass pass, Solve solve, double tol, int max_iter) {
  std::vector<double> cur(Z), llc(N, -INFINITY), Gc((size_t)N * n), Hc((size_t)N * hn), mu(N, 0.0);
  std::vector<double> ll(N), G((size_t)N * n), H((size_t)N * hn);
  std::vector<char> act(use);
  int passes = 0;
  while (passes < max_iter && std::count(act.begin(), act.end(), 1) > 0) {
    pass(Z, act, ll, G, H);
    passes++;
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; i++) {
      if (!act[i]) continue;
      const size_t o = (size_t)i * n, oh = (size_t)i * hn;
      if (ll[i] >= llc[i]) {  // accept
        const double gain = ll[i] - llc[i];
        std::copy(Z.begin() + o, Z.begin() + o + n, cur.begin() + o);
        std::copy(G.begin() + o, G.begin() + o + n, Gc.begin() + o);
        std::copy(H.begin() + oh, H.begin() + oh + hn, Hc.begin() + oh);
        llc[i] = ll[i];
        mu[i] *= 0.25;
        if (gain < tol) {
          act[i] = 0;
          continue;
        }
      } else {  // reject: back to the current point with more damping
        mu[i] = std::max(4 * mu[i], 1e-3);
        if (mu[i] > 1e8) {
          act[i] = 0;
          continue;
        }
      }
      solve(i, cur.data() + o, Gc.data() + o, Hc.data() + oh, mu[i], Z.data() + o);
    }
  }
  for (int i = 0; i < N; i++)
    if (use[i]) {
      std::copy(cur.begin() + (size_t)i * n, cur.begin() + (size_t)(i + 1) * n, Z.begin() + (size_t)i * n);
      ll_out[i] = llc[i];
    }
  return passes;
}

// adds the smallest shift that makes the n x n matrix H positive definite, plus mu * mean diagonal
inline void make_pd(int n, std::vector<double>& H, double mu) {
  double tr = 0;
  for (int k = 0; k < n; k++) tr += H[k * n + k];
  const double base = 1e-10 * tr / n + 1e-300;
  std::vector<double> C;
  double shift = 0;
  for (int t = 0; t < 40; t++) {
    C = H;
    for (int k = 0; k < n; k++) C[k * n + k] += shift + base;
    if (cholesky_ld(n, C.data(), n)) break;
    shift = shift > 0 ? 4 * shift : 1e-6 * tr / n + 1e-300;
  }
  for (int k = 0; k < n; k++) H[k * n + k] += shift + base + mu * tr / n;
}

// Parental and paired ancestry of every individual, given P (F, M x K) and Q (N x K)
template <class Data>
Result estimate(const Data& D, const double* F, const double* Q, int K, bool parental, bool paired) {
  const int N = D.N, kp = K * (K + 1) / 2, n2 = 2 * K;
  Result r;
  std::vector<char> use(N);
  for (int i = 0; i < N; i++) use[i] = D.nobs[i] > 0;
  std::vector<char> freeK((size_t)N * K);  // ancestries that are estimated (q_k >= QMIN_FREE)
  for (size_t t = 0; t < freeK.size(); t++) freeK[t] = Q[t] >= QMIN_FREE;
  // screening pass at x = y = q (also gives the ADMIXTURE log-likelihood)
  double t0 = omp_get_wtime();
  std::vector<double> Mc((size_t)N * kp);
  r.ll_adm.assign(N, 0.0);
  pass_symmetric(D, F, Q, K, use, r.ll_adm, Mc);
  if (parental) {
    r.passes_par = 1;
    r.par.assign((size_t)N * n2, 0.0);
    for (int i = 0; i < N; i++)
      for (int k = 0; k < K; k++) r.par[(size_t)i * n2 + k] = r.par[(size_t)i * n2 + K + k] = Q[(size_t)i * K + k];
    r.ll_par = r.ll_adm;
    // line search along the split direction: x = q + t d, y = q - t d
    std::vector<double> dir((size_t)N * K), Z(r.par), start(r.par), ll(N), best(r.ll_adm);
    std::vector<char> cand(N, 0);
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; i++)
      if (use[i]) cand[i] = split_direction(K, Q + (size_t)i * K, Mc.data() + (size_t)i * kp, dir.data() + (size_t)i * K);
    for (double frac : {1.0, 0.5, 0.25, 0.1}) {
      for (int i = 0; i < N; i++)
        if (cand[i]) {
          double* z = Z.data() + (size_t)i * n2;
          for (int k = 0; k < K; k++) {
            const double q = Q[(size_t)i * K + k], dk = frac * dir[(size_t)i * K + k];
            z[k] = std::max(q + dk, FLOOR), z[K + k] = std::max(q - dk, FLOOR);
          }
          project_simplex(K, z, FLOOR), project_simplex(K, z + K, FLOOR);
        }
      pass_parental_ll(D, F, K, Z, cand, ll);
      r.passes_par++;
      for (int i = 0; i < N; i++)
        if (cand[i] && ll[i] > best[i]) {
          best[i] = ll[i];
          std::copy(Z.begin() + (size_t)i * n2, Z.begin() + (size_t)(i + 1) * n2, start.begin() + (size_t)i * n2);
        }
    }
    std::vector<char> fit(N, 0);
    for (int i = 0; i < N; i++) fit[i] = cand[i] && best[i] >= r.ll_adm[i] + MIN_GAIN;
    r.n_split = std::count(fit.begin(), fit.end(), 1);
    if (r.n_split) {
      Z = start;
      std::vector<double> llf(N);
      const int kh = 3 * kp;
      r.passes_par += newton_rows(
          N, n2, kh, Z, llf, fit,
          [&](auto& Zr, auto& act, auto& l, auto& G, auto& H) { pass_parental_newton(D, F, K, Zr, act, l, G, H); },
          [&](int i, const double* z, const double* g, const double* hp, double mu, double* out) {
            std::vector<double> H((size_t)n2 * n2), lo(n2), hi(n2), dd(n2);
            for (int a = 0, p = 0; a < K; a++)
              for (int b = a; b < K; b++, p++) {
                H[a * n2 + b] = H[b * n2 + a] = hp[p];
                H[(K + a) * n2 + K + b] = H[(K + b) * n2 + K + a] = hp[kp + p];
                H[a * n2 + K + b] = H[(K + b) * n2 + a] = H[b * n2 + K + a] = H[(K + a) * n2 + b] = hp[2 * kp + p];
              }
            make_pd(n2, H, mu);
            const char* fr = freeK.data() + (size_t)i * K;
            for (int k = 0; k < n2; k++) {
              const bool f = fr[k % K];
              lo[k] = f ? FLOOR - z[k] : 0, hi[k] = f ? 1 - z[k] : 0;
            }
            qp_two_simplex(K, H.data(), g, lo.data(), hi.data(), dd.data());
            for (int k = 0; k < n2; k++) out[k] = std::max(z[k] + dd[k], FLOOR);
          },
          1e-2, 30);
      for (int i = 0; i < N; i++)
        if (fit[i] && llf[i] > r.ll_adm[i]) {
          r.ll_par[i] = llf[i];
          std::copy(Z.begin() + (size_t)i * n2, Z.begin() + (size_t)(i + 1) * n2, r.par.begin() + (size_t)i * n2);
        }
    }
    // canonical order: parent 1 is the lexicographically larger vector
    for (int i = 0; i < N; i++) {
      double* x = r.par.data() + (size_t)i * n2;
      if (std::lexicographical_compare(x, x + K, x + K, x + n2)) std::swap_ranges(x, x + K, x + K);
    }
    r.sec_par = omp_get_wtime() - t0;
  }
  if (paired) {  // from the ADMIXTURE model, pi_aa = q_a^2, pi_ab = 2 q_a q_b; pairs with a non-free ancestry fixed
    t0 = omp_get_wtime();
    const int kh = kp * (kp + 1) / 2;
    r.pair.assign((size_t)N * kp, 0.0);
    std::vector<char> freeP((size_t)N * kp);
    for (int i = 0; i < N; i++) {
      const double* q = Q + (size_t)i * K;
      for (int a = 0, p = 0; a < K; a++)
        for (int b = a; b < K; b++, p++) {
          r.pair[(size_t)i * kp + p] = std::max(a == b ? q[a] * q[a] : 2 * q[a] * q[b], FLOOR);
          freeP[(size_t)i * kp + p] = freeK[(size_t)i * K + a] && freeK[(size_t)i * K + b];
        }
    }
    r.ll_pair = r.ll_adm;
    r.passes_pair = newton_rows(
        N, kp, kh, r.pair, r.ll_pair, use,
        [&](auto& Zr, auto& act, auto& l, auto& G, auto& H) { pass_paired_newton(D, F, K, Zr, act, l, G, H); },
        [&](int i, const double* z, const double* g, const double* hp, double mu, double* out) {
          std::vector<double> H((size_t)kp * kp), lo(kp), hi(kp), dd(kp);
          for (int p = 0, h = 0; p < kp; p++)
            for (int q = p; q < kp; q++, h++) H[p * kp + q] = H[q * kp + p] = hp[h];
          make_pd(kp, H, mu);
          const char* fr = freeP.data() + (size_t)i * kp;
          for (int p = 0; p < kp; p++) lo[p] = fr[p] ? FLOOR - z[p] : 0, hi[p] = fr[p] ? 1 - z[p] : 0;
          qp_active_set(kp, H.data(), g, lo.data(), hi.data(), true, dd.data());
          for (int p = 0; p < kp; p++) out[p] = std::max(z[p] + dd[p], FLOOR);
        },
        1e-4, 50);
    r.sec_pair = omp_get_wtime() - t0;
  }
  return r;
}

// NAME.K.parental: x (parent 1), y (parent 2), log L parental, log L ADMIXTURE; NAME.K.paired: pi_ab (a <= b),
// log L paired, log L ADMIXTURE. One line per individual (as .Q), with a header.
inline void write(const std::string& fn, const Result& r, int N, int K, bool paired) {
  FILE* f = std::fopen(fn.c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + fn);
  const int kp = K * (K + 1) / 2;
  if (paired) {
    for (int a = 1; a <= K; a++)
      for (int b = a; b <= K; b++) std::fprintf(f, "%spair_%d_%d", a == 1 && b == 1 ? "" : " ", a, b);
    std::fprintf(f, " loglik_paired loglik_admixture\n");
  } else {
    for (int p = 1; p <= 2; p++)
      for (int k = 1; k <= K; k++) std::fprintf(f, "%sparent%d_%d", p == 1 && k == 1 ? "" : " ", p, k);
    std::fprintf(f, " loglik_parental loglik_admixture\n");
  }
  for (int i = 0; i < N; i++) {
    const int n = paired ? kp : 2 * K;
    const double* v = (paired ? r.pair : r.par).data() + (size_t)i * n;
    for (int k = 0; k < n; k++) std::fprintf(f, k ? " %.6f" : "%.6f", v[k]);
    std::fprintf(f, " %.4f %.4f\n", (paired ? r.ll_pair : r.ll_par)[i], r.ll_adm[i]);
  }
  std::fclose(f);
}

}  // namespace parental
