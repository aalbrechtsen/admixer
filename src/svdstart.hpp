// admixer: SVD/ALS start (--svd). Ported from admix_prog (src/init.hpp, init_svd_als; as SCOPE / fastmixture /
// ADAMIXTURE): a randomised SVD (rank K-1) of the centred genotype matrix Z (Z_ji = g_ji/2 - p_j, 0 if missing),
// then alternating least squares on its low-rank reconstruction  Pi = p 1' + U S V'  ~  P Q', where every row
// update is an exact least-squares fit under the box (P) or simplex (Q) constraints (qp_active_set). The ALS
// starts from P uniform in [0.05, 0.95] and sparse Q rows (Exp(1) draws projected onto the simplex), as in
// admix_prog. Changes in the port: packed genotypes and genotype likelihoods (expected genotype under a flat
// prior, entry_value in search.hpp) are read through a per-SNP row of Z; the Cholesky-QR uses admixer's
// cholesky_ld with a ridge.
// Why: on the blue wildebeest data (K = 7) fastmixture reached the best solution from this start in 10 of 10
// runs and from random starts in 5 of 10 (global_search/README.md of the admix project). A single deterministic
// start rules out restarts (speed-up note 11), so admixer uses it as one start among random ones (--conv, --search).
#pragma once
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "linalg.hpp"
#include "model.hpp"
#include "search.hpp"

struct SvdSettings {
  int power_iter = 7;      // randomised-SVD power iterations
  int als_max_iter = 500;  // ALS iterations at most
  double als_tol = 1e-6;   // stop ALS when the RMS change of Q < als_tol
};

struct SvdInfo {
  int als_iter = 0;
  double seconds = 0;
};

namespace svd_detail {
// row j of Z: z_i = v_ji/2 - p_j (0 if missing)
template <class Data>
inline void z_row(const Data& D, int j, double pj, double* z) {
  double v;
  for (int i = 0; i < D.N; i++) z[i] = entry_value(D, j, i, v) ? 0.5 * v - pj : 0.0;
}
inline void z_row(const Genotypes& D, int j, double pj, double* z) {
  const uint8_t* r = D.row(j);
  for (int i = 0; i < D.N; i++) {
    const uint8_t g = GTAB.code[r[i >> 2]][i & 3];
    z[i] = g == 3 ? 0.0 : (D.haploid ? 1.0 : 0.5) * g - pj;
  }
}
// Y (M x L) = Z X, with X (N x L).
template <class Data>
void Z_times(const Data& D, const std::vector<double>& p, const std::vector<double>& X, int L, std::vector<double>& Y) {
  Y.assign((size_t)D.M * L, 0.0);
#pragma omp parallel
  {
    std::vector<double> z(D.N);
#pragma omp for schedule(dynamic, 64)
    for (int j = 0; j < D.M; j++) {
      z_row(D, j, p[j], z.data());
      double* y = Y.data() + (size_t)j * L;
      for (int i = 0; i < D.N; i++) {
        if (z[i] == 0.0) continue;
        const double* xi = X.data() + (size_t)i * L;
        for (int l = 0; l < L; l++) y[l] += z[i] * xi[l];
      }
    }
  }
}
// X (N x L) = Z' Y, with Y (M x L).
template <class Data>
void Zt_times(const Data& D, const std::vector<double>& p, const std::vector<double>& Y, int L, std::vector<double>& X) {
  const int nt = omp_get_max_threads();
  std::vector<double> part((size_t)nt * D.N * L, 0.0);
#pragma omp parallel
  {
    std::vector<double> z(D.N);
    double* xp = part.data() + (size_t)omp_get_thread_num() * D.N * L;
#pragma omp for schedule(dynamic, 64)
    for (int j = 0; j < D.M; j++) {
      z_row(D, j, p[j], z.data());
      const double* y = Y.data() + (size_t)j * L;
      for (int i = 0; i < D.N; i++) {
        if (z[i] == 0.0) continue;
        double* xi = xp + (size_t)i * L;
        for (int l = 0; l < L; l++) xi[l] += z[i] * y[l];
      }
    }
  }
  X.assign((size_t)D.N * L, 0.0);
  for (int t = 0; t < nt; t++)
    for (size_t u = 0; u < X.size(); u++) X[u] += part[(size_t)t * D.N * L + u];
}
// Gram matrix A'A of tall A (n x L).
inline std::vector<double> gram(const std::vector<double>& A, size_t n, int L) {
  std::vector<double> G((size_t)L * L, 0.0);
#pragma omp parallel
  {
    std::vector<double> loc((size_t)L * L, 0.0);
#pragma omp for schedule(static)
    for (size_t r = 0; r < n; r++) {
      const double* a = A.data() + r * L;
      for (int k = 0; k < L; k++)
        for (int l = k; l < L; l++) loc[k * L + l] += a[k] * a[l];
    }
#pragma omp critical
    for (int t = 0; t < L * L; t++) G[t] += loc[t];
  }
  for (int k = 0; k < L; k++)
    for (int l = 0; l < k; l++) G[k * L + l] = G[l * L + k];
  return G;
}
// Orthonormalise the columns of tall A (n x L) by two rounds of Cholesky-QR.
inline void orthonormalise(std::vector<double>& A, size_t n, int L) {
  for (int round = 0; round < 2; round++) {
    std::vector<double> R = gram(A, n, L);
    double tr = 0;
    for (int k = 0; k < L; k++) tr += R[k * L + k];
    for (double ridge = 1e-12 * tr / L + 1e-300;; ridge *= 100) {  // ridge until positive definite
      std::vector<double> U = R;
      for (int k = 0; k < L; k++) U[k * L + k] += ridge;
      if (cholesky_ld(L, U.data(), L)) {
        R.swap(U);
        break;
      }
    }
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < n; r++) {  // row <- row R^{-1} (forward substitution)
      double* a = A.data() + r * L;
      for (int c = 0; c < L; c++) {
        double s = a[c];
        for (int k = 0; k < c; k++) s -= a[k] * R[k * L + c];
        a[c] = s / R[c * L + c];
      }
    }
  }
}
// Eigen-decomposition of the symmetric n x n matrix A by cyclic Jacobi rotations: eigenvalues w in decreasing
// order, eigenvectors as the columns of V.
inline void sym_eigen(int n, std::vector<double> A, std::vector<double>& w, std::vector<double>& V) {
  V.assign((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++) V[(size_t)i * n + i] = 1;
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0, tot = 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) (i == j ? tot : off) += A[(size_t)i * n + j] * A[(size_t)i * n + j];
    if (off <= 1e-24 * (tot + off)) break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        const double apq = A[(size_t)p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double th = (A[(size_t)q * n + q] - A[(size_t)p * n + p]) / (2 * apq);
        const double t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; k++) {
          const double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
          A[(size_t)k * n + p] = c * akp - s * akq, A[(size_t)k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          const double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
          A[(size_t)p * n + k] = c * apk - s * aqk, A[(size_t)q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          const double vkp = V[(size_t)k * n + p], vkq = V[(size_t)k * n + q];
          V[(size_t)k * n + p] = c * vkp - s * vkq, V[(size_t)k * n + q] = s * vkp + c * vkq;
        }
      }
  }
  std::vector<int> ord(n);
  for (int i = 0; i < n; i++) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return A[(size_t)a * n + a] > A[(size_t)b * n + b]; });
  std::vector<double> Vs((size_t)n * n);
  w.resize(n);
  for (int c = 0; c < n; c++) {
    w[c] = A[(size_t)ord[c] * n + ord[c]];
    for (int k = 0; k < n; k++) Vs[(size_t)k * n + c] = V[(size_t)k * n + ord[c]];
  }
  V.swap(Vs);
}
// C (rows x K) = A (rows x L) * B (L x K)
inline std::vector<double> matmul(const std::vector<double>& A, size_t rows, int L, const std::vector<double>& B, int K) {
  std::vector<double> C(rows * K, 0.0);
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; r++)
    for (int l = 0; l < L; l++) {
      const double a = A[r * L + l];
      for (int k = 0; k < K; k++) C[r * K + k] += a * B[l * K + k];
    }
  return C;
}
// A'B for tall A (n x a), B (n x b) -> a x b
inline std::vector<double> crossprod(const std::vector<double>& A, int a, const std::vector<double>& B, int b, size_t n) {
  std::vector<double> C((size_t)a * b, 0.0);
#pragma omp parallel
  {
    std::vector<double> loc((size_t)a * b, 0.0);
#pragma omp for schedule(static)
    for (size_t r = 0; r < n; r++)
      for (int k = 0; k < a; k++) {
        const double v = A[r * a + k];
        for (int l = 0; l < b; l++) loc[k * b + l] += v * B[r * b + l];
      }
#pragma omp critical
    for (int t = 0; t < a * b; t++) C[t] += loc[t];
  }
  return C;
}
// Row-wise constrained least squares: for each row r, x_r = argmin 0.5 x'Hx - c_r'x subject to lo <= x <= hi
// (and sum x = 1 if simplex). X holds a feasible warm start on entry.
inline void rows_constrained_ls(const std::vector<double>& C, size_t rows, int K, const std::vector<double>& H,
                                std::vector<double>& X, double lo, double hi, bool simplex) {
#pragma omp parallel
  {
    std::vector<double> g(K), l(K), h(K), d(K);
#pragma omp for schedule(static)
    for (size_t r = 0; r < rows; r++) {
      double* x = X.data() + r * K;
      for (int k = 0; k < K; k++) {
        double s = C[r * K + k];
        for (int m = 0; m < K; m++) s -= H[k * K + m] * x[m];
        g[k] = s;
        l[k] = lo - x[k];
        h[k] = hi - x[k];
      }
      qp_active_set(K, H.data(), g.data(), l.data(), h.data(), simplex, d.data());
      for (int k = 0; k < K; k++) x[k] = std::min(std::max(x[k] + d[k], lo), hi);
      if (simplex) project_simplex(K, x, lo);
    }
  }
}
}  // namespace svd_detail

// x <- the SVD/ALS start (P and Q); seed: the random test matrix of the SVD and the ALS start.
template <class Data>
SvdInfo svd_start(const Data& D, const Model<Data>& m, Vec& x, unsigned long seed, const SvdSettings& is = {}) {
  using namespace svd_detail;
  const double t0 = omp_get_wtime();
  const int M = D.M, N = D.N, K = m.K;
  std::mt19937_64 rng(seed);
  std::vector<double> p(M);  // allele frequencies
#pragma omp parallel
  {
    std::vector<double> z(N);
#pragma omp for schedule(static)
    for (int j = 0; j < M; j++) {
      z_row(D, j, 0.0, z.data());  // v/2 for observed entries, 0 for missing
      double s = 0, v;
      int n = 0;
      for (int i = 0; i < N; i++)
        if (entry_value(D, j, i, v)) s += z[i], n++;
      p[j] = n ? std::min(std::max(s / n, m.PMIN), m.PMAX) : 0.5;
    }
  }
  // ALS warm start: P uniform in [0.05, 0.95], sparse Q rows (as admix_prog's init_random_sparse)
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::exponential_distribution<double> expo(1.0);
  for (size_t t = 0; t < m.nP; t++) x[t] = 0.05 + 0.9 * unif(rng);
  for (size_t t = 0; t < m.nQ; t++) x[m.nP + t] = expo(rng);
  m.project(x.data());
  SvdInfo info;
  if (K == 1) {
    for (int j = 0; j < M; j++) x[j] = p[j];
    info.seconds = omp_get_wtime() - t0;
    return info;
  }
  const int r = K - 1, L = std::min(r + 10, std::min(M, N));
  std::normal_distribution<double> norm(0.0, 1.0);
  std::vector<double> Om((size_t)N * L), Y, W;
  for (auto& v : Om) v = norm(rng);
  Z_times(D, p, Om, L, Y);
  orthonormalise(Y, M, L);
  for (int it = 0; it < is.power_iter; it++) {
    Zt_times(D, p, Y, L, W);
    orthonormalise(W, N, L);
    Z_times(D, p, W, L, Y);
    orthonormalise(Y, M, L);
  }
  Zt_times(D, p, Y, L, W);  // W = Z'Y = B'
  std::vector<double> ev, Ub;
  sym_eigen(L, gram(W, N, L), ev, Ub);  // B B' = Ub diag(ev) Ub'
  std::vector<double> Ubr((size_t)L * r);
  for (int l = 0; l < L; l++)
    for (int c = 0; c < r; c++) Ubr[(size_t)l * r + c] = Ub[(size_t)l * L + c];
  std::vector<double> US = matmul(Y, M, L, Ubr, r);  // U S = Y Ub  (M x r)
  for (int j = 0; j < M; j++)
    for (int c = 0; c < r; c++) US[(size_t)j * r + c] *= std::sqrt(std::max(ev[c], 0.0));
  std::vector<double> V = matmul(W, N, L, Ubr, r);  // V S = W Ub  (N x r)
  for (int i = 0; i < N; i++)
    for (int c = 0; c < r; c++) V[(size_t)i * r + c] /= std::sqrt(std::max(ev[c], 1e-300));
  // ALS:  Q' = (P'P)^-1 P' Pi,  P = Pi Q (Q'Q)^-1, using the low-rank form of Pi.
  std::vector<double> P(x.begin(), x.begin() + m.nP), Q(x.begin() + m.nP, x.end()), Qold;
  int als_it = 0;
  for (; als_it < is.als_max_iter; als_it++) {
    std::vector<double> VtQ = crossprod(V, r, Q, K, N);  // r x K
    std::vector<double> colQ(K, 0.0);
    for (int i = 0; i < N; i++)
      for (int k = 0; k < K; k++) colQ[k] += Q[(size_t)i * K + k];
    std::vector<double> C = matmul(US, M, r, VtQ, K);  // Pi Q
    for (int j = 0; j < M; j++)
      for (int k = 0; k < K; k++) C[(size_t)j * K + k] += p[j] * colQ[k];
    rows_constrained_ls(C, M, K, gram(Q, N, K), P, m.PMIN, m.PMAX, false);
    std::vector<double> PtUS = crossprod(P, K, US, r, M);  // K x r
    std::vector<double> Ptp(K, 0.0);
    for (int j = 0; j < M; j++)
      for (int k = 0; k < K; k++) Ptp[k] += P[(size_t)j * K + k] * p[j];
    std::vector<double> PtUSt((size_t)r * K);
    for (int k = 0; k < K; k++)
      for (int c = 0; c < r; c++) PtUSt[(size_t)c * K + k] = PtUS[(size_t)k * r + c];
    Qold = Q;
    C = matmul(V, N, r, PtUSt, K);  // Pi' P
    for (int i = 0; i < N; i++)
      for (int k = 0; k < K; k++) C[(size_t)i * K + k] += Ptp[k];
    rows_constrained_ls(C, N, K, gram(P, M, K), Q, m.QMIN, 1.0, true);
    double diff = 0;
    for (size_t t = 0; t < Q.size(); t++) diff += (Q[t] - Qold[t]) * (Q[t] - Qold[t]);
    if (std::sqrt(diff / Q.size()) < is.als_tol) {
      als_it++;
      break;
    }
  }
  std::copy(P.begin(), P.end(), x.begin());
  std::copy(Q.begin(), Q.end(), x.begin() + m.nP);
  info.als_iter = als_it;
  info.seconds = omp_get_wtime() - t0;
  return info;
}
