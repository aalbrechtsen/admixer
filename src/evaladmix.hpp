// admixer: evalAdmix correlation of residuals, corrected estimator (van Waaij et al. 2023, Genetics 225;
// evalAdmix "-method corrected", Garcia-Erill & Albrechtsen), for called genotypes and, on posterior expected
// genotypes, for genotype likelihoods (evaladmix_gl below). For genotypes it uses only the genotypes and Q:
//   1. residuals of each SNP after projecting its genotypes onto the column space of Q,
//      r_j = g_j - Pq g_j with Pq = Q (Q'Q)^{-1} Q' (missing genotypes mean-imputed inside the
//      projection only);
//   2. b = Pearson correlation of the residuals between individuals (pairwise missing-site skipping,
//      individual means);
//   3. c = correlation implied by the model, from C = (I - Pq) D (I - Pq), D = diag(mean g(2-g));
//   output b - c. Everything is evaluated with BLAS matrix products.
#pragma once
#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "beagle.hpp"
#include "io.hpp"
#include "linalg.hpp"

extern "C" void openblas_set_num_threads(int);

// Core of the corrected estimator for any per-entry values x_ij (genotypes, or expected genotypes from
// GLs). fill(j0, jn, X, O) writes the values of sites j0..j0+jn-1 into X (jn x N, row major) and the
// observation mask into O (1 = observed; values of unobserved entries are ignored). dh[i] is the model
// variance of individual i's values (D in C = (I - Pq) D (I - Pq)). Returns the N x N matrix (row major) of
// corrected residual correlations; the diagonal is NaN.
// impute = true: unobserved entries are mean-imputed inside the projection (called genotypes). impute = false:
// the values of every entry are used in the projection as they are (genotype likelihoods: an entry without
// data has E[g | GL] = 2h, the model mean), and the mask only restricts the correlations to the sites
// observed in both individuals.
template <class Fill>
inline std::vector<double> evaladmix_core(int M, int N, const double* Q, int K, const std::vector<double>& dh,
                                          int threads, Fill fill, bool impute = true) {
  openblas_set_num_threads(threads);  // the large products below use multithreaded BLAS
  const int B = 256;
  std::vector<double> X((size_t)B * N);
  std::vector<uint8_t> O((size_t)B * N);
  // per-individual mean over observed sites (for mean imputation inside the projection) and counts
  std::vector<double> mu(N, 0.0);
  std::vector<int> nobs(N, 0);
  for (int j0 = 0; j0 < M; j0 += B) {
    const int jn = std::min(B, M - j0);
    fill(j0, jn, X.data(), O.data());
    for (int jj = 0; jj < jn; jj++)
      for (int i = 0; i < N; i++)
        if (O[(size_t)jj * N + i]) mu[i] += X[(size_t)jj * N + i], nobs[i]++;
  }
  for (int i = 0; i < N; i++)
    if (nobs[i] > 0) mu[i] /= nobs[i];
  // A = Q (Q'Q)^{-1}  (N x K)
  std::vector<double> QtQ(K * K, 0.0), A((size_t)N * K);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, Q, K, Q, K, 0.0, QtQ.data(), K);
  std::vector<double> inv(K * K);
  for (int c = 0; c < K; c++) {
    std::vector<double> Mx(QtQ), e(K, 0.0);
    e[c] = 1;
    if (!solve_linear(K, Mx.data(), e.data())) throw std::runtime_error("evalAdmix: Q'Q is singular");
    for (int r = 0; r < K; r++) inv[r * K + c] = e[r];
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, Q, K, inv.data(), K, 0.0, A.data(), K);

  // residual block (SNPs j0..j0+jn): R = X - (X A) Q'  with unobserved entries mean-imputed; returns in R,
  // and the mask in O
  std::vector<double> R((size_t)B * N), Z((size_t)B * K);
  auto residuals = [&](int j0, int jn) {
    fill(j0, jn, R.data(), O.data());
    if (impute)
      for (size_t t = 0; t < (size_t)jn * N; t++)
        if (!O[t]) R[t] = mu[t % N];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, jn, K, N, 1.0, R.data(), N, A.data(), K, 0.0, Z.data(), K);
    std::vector<double> pred((size_t)jn * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, jn, N, K, 1.0, Z.data(), K, Q, K, 0.0, pred.data(), N);
    for (size_t t = 0; t < (size_t)jn * N; t++) R[t] -= pred[t];
  };
  // pass 1: mean residual per individual over its observed sites
  std::vector<double> mr(N, 0.0);
  for (int j0 = 0; j0 < M; j0 += B) {
    const int jn = std::min(B, M - j0);
    residuals(j0, jn);
    for (int jj = 0; jj < jn; jj++)
      for (int i = 0; i < N; i++)
        if (O[(size_t)jj * N + i]) mr[i] += R[(size_t)jj * N + i];
  }
  for (int i = 0; i < N; i++)
    if (nobs[i] > 0) mr[i] /= nobs[i];
  // pass 2: with a_ij = m_ij (r_ij - mean_i) and mask m_ij,
  //   num_ii' = sum_j a_ij a_i'j,  den_ii' = sum_j a_ij^2 m_i'j   (sums over sites observed in both)
  std::vector<double> num((size_t)N * N, 0.0), den((size_t)N * N, 0.0), Asq((size_t)B * N), Mk((size_t)B * N);
  for (int j0 = 0; j0 < M; j0 += B) {
    const int jn = std::min(B, M - j0);
    residuals(j0, jn);
    for (int jj = 0; jj < jn; jj++) {
      for (int i = 0; i < N; i++) {
        const size_t t = (size_t)jj * N + i;
        const bool obs = O[t] != 0;
        R[t] = obs ? R[t] - mr[i] : 0.0;
        Asq[t] = R[t] * R[t];
        Mk[t] = obs;
      }
    }
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, N, jn, 1.0, R.data(), N, 1.0, num.data(), N);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, jn, 1.0, Asq.data(), N, Mk.data(), N, 1.0, den.data(), N);
  }
  // model-implied correlation: C = D - D Pq - Pq D + Pq D Pq, with Pq = A Q' and Pq D Pq = A (Q' D A) Q'
  std::vector<double> Pq((size_t)N * N), DA((size_t)N * K), S(K * K), AS((size_t)N * K), C((size_t)N * N);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, K, 1.0, A.data(), K, Q, K, 0.0, Pq.data(), N);
  for (int i = 0; i < N; i++)
    for (int k = 0; k < K; k++) DA[(size_t)i * K + k] = dh[i] * A[(size_t)i * K + k];
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, Q, K, DA.data(), K, 0.0, S.data(), K);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, A.data(), K, S.data(), K, 0.0, AS.data(), K);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, K, 1.0, AS.data(), K, Q, K, 0.0, C.data(), N);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++)
    for (int l = 0; l < N; l++)
      C[(size_t)i * N + l] += (i == l ? dh[i] : 0.0) - dh[i] * Pq[(size_t)i * N + l] - Pq[(size_t)i * N + l] * dh[l];
  // result b - c, symmetric, NaN on the diagonal
  std::vector<double> cor((size_t)N * N);
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N; i++) {
    cor[(size_t)i * N + i] = NAN;
    for (int l = i + 1; l < N; l++) {
      const double dd = std::sqrt(den[(size_t)i * N + l] * den[(size_t)l * N + i]);
      const double b = dd > 0 ? num[(size_t)i * N + l] / dd : 0.0;
      const double cc = std::sqrt(C[(size_t)i * N + i] * C[(size_t)l * N + l]);
      const double c = cc > 0 ? C[(size_t)i * N + l] / cc : 0.0;
      cor[(size_t)i * N + l] = cor[(size_t)l * N + i] = b - c;
    }
  }
  openblas_set_num_threads(1);
  return cor;
}

// Called genotypes (PLINK): values g, D = diag(mean g(2-g)) over observed sites.
inline std::vector<double> evaladmix_corrected(const Genotypes& D, const double* Q, int K, int threads) {
  const int M = D.M, N = D.N;
  std::vector<double> dh(N, 0.0);
  for (int j = 0; j < M; j++) {
    const uint8_t* g = D.row(j);
    for (int i = 0; i < N; i++)
      if (g[i] != 3) dh[i] += g[i] * (2.0 - g[i]);
  }
  for (int i = 0; i < N; i++)
    if (D.nobs[i] > 0) dh[i] /= D.nobs[i];
  return evaladmix_core(M, N, Q, K, dh, threads, [&](int j0, int jn, double* X, uint8_t* O) {
    for (int jj = 0; jj < jn; jj++) {
      const uint8_t* g = D.row(j0 + jj);
      for (int i = 0; i < N; i++) {
        const size_t t = (size_t)jj * N + i;
        O[t] = g[i] != 3;
        X[t] = g[i];
      }
    }
  });
}

// Genotype likelihoods (beagle): the corrected estimator on the posterior expected genotypes
// e_ij = E[g | GL_ij, h_ij], h_ij = Q_i P_j (the fitted model as prior). Every entry enters the projection with
// its e; an entry without data has e = 2h, the model mean, which lies in the column space of Q and is projected
// out, so nothing is imputed. The correlations use only the sites where both individuals have data
// (max - min GL >= misTol), as evalAdmix -beagle does, and D_i = mean (e_ij - 2h_ij)^2 over those sites
// (Var(e) under the model). In simulations (1-8x and mixed depth) this matched evalAdmix -beagle -nIts 5 in
// accuracy against the genotype truth, at ~5% of its cost (see the ngsadmix report evaladmix_gl).
// P is M x K and Q is N x K in the site order of D; lo/hi are the model's bounds for h.
inline std::vector<double> evaladmix_gl(const GLData& D, const double* P, const double* Q, int K, double misTol,
                                        double lo, double hi, int threads) {
  const int M = D.M, N = D.N;
  const int B = 256;
  // e and 2h of a block of sites: H = P_block Q' by BLAS, then the posterior mean entry by entry
  auto values = [&](int j0, int jn, double* X, double* H2, uint8_t* O) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, jn, N, K, 1.0, P + (size_t)j0 * K, K, Q, K, 0.0, H2, N);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int jj = 0; jj < jn; jj++) {
      const float* l = D.row(j0 + jj);
      for (int i = 0; i < N; i++) {
        const size_t t = (size_t)jj * N + i;
        const float* g = l + 3 * (size_t)i;
        const double h = std::min(std::max(H2[t], lo), hi), a = 1 - h;
        const double w0 = g[0] * a * a, w1 = g[1] * 2 * h * a, w2 = g[2] * h * h;
        X[t] = (w1 + 2 * w2) / (w0 + w1 + w2);
        H2[t] = 2 * h;
        const float mx = std::max(g[0], std::max(g[1], g[2])), mn = std::min(g[0], std::min(g[1], g[2]));
        O[t] = mx - mn >= misTol;
      }
    }
  };
  openblas_set_num_threads(threads);
  std::vector<double> X((size_t)B * N), H2((size_t)B * N), dh(N, 0.0);
  std::vector<uint8_t> O((size_t)B * N);
  std::vector<int> n(N, 0);
  for (int j0 = 0; j0 < M; j0 += B) {
    const int jn = std::min(B, M - j0);
    values(j0, jn, X.data(), H2.data(), O.data());
    for (size_t t = 0; t < (size_t)jn * N; t++)
      if (O[t]) dh[t % N] += (X[t] - H2[t]) * (X[t] - H2[t]), n[t % N]++;
  }
  for (int i = 0; i < N; i++)
    if (n[i] > 0) dh[i] /= n[i];
  std::vector<double> Hb((size_t)B * N);
  return evaladmix_core(
      M, N, Q, K, dh, threads, [&](int j0, int jn, double* Xo, uint8_t* Oo) { values(j0, jn, Xo, Hb.data(), Oo); },
      false);
}

// evalAdmix output format: full symmetric matrix, tab-separated, "NA" on the diagonal.
inline void write_corres(const std::string& fn, const std::vector<double>& cor, int N) {
  FILE* fp = std::fopen(fn.c_str(), "w");
  if (!fp) throw std::runtime_error("cannot write " + fn);
  for (int i = 0; i < N; i++) {
    for (int l = 0; l < N; l++) {
      if (l) std::fputc('\t', fp);
      const double v = cor[(size_t)i * N + l];
      if (std::isnan(v)) std::fputs("NA", fp);
      else std::fprintf(fp, "%f", v);
    }
    std::fputc('\n', fp);
  }
  std::fclose(fp);
}
