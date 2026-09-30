// admixer: evalAdmix correlation of residuals, corrected estimator (van Waaij et al. 2023, Genetics 225;
// evalAdmix "-method corrected", Garcia-Erill & Albrechtsen). Uses only the genotypes and Q:
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

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
#include "linalg.hpp"

extern "C" void openblas_set_num_threads(int);

// Returns the N x N matrix (row major) of corrected residual correlations; the diagonal is NaN.
inline std::vector<double> evaladmix_corrected(const Genotypes& D, const double* Q, int K, int threads) {
  const int M = D.M, N = D.N;
  openblas_set_num_threads(threads);  // the large products below use multithreaded BLAS
  // per-individual mean genotype and heterozygosity over observed sites
  std::vector<double> mu(N, 0.0), dh(N, 0.0);
  for (int j = 0; j < M; j++) {
    const uint8_t* g = D.row(j);
    for (int i = 0; i < N; i++)
      if (g[i] != 3) mu[i] += g[i], dh[i] += g[i] * (2.0 - g[i]);
  }
  for (int i = 0; i < N; i++)
    if (D.nobs[i] > 0) mu[i] /= D.nobs[i], dh[i] /= D.nobs[i];
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

  // residual block (SNPs j0..j0+jn): R = G - (G A) Q'  with missing genotypes mean-imputed; returns in R
  const int B = 256;
  std::vector<double> R((size_t)B * N), Z((size_t)B * K);
  auto residuals = [&](int j0, int jn) {
    for (int jj = 0; jj < jn; jj++) {
      const uint8_t* g = D.row(j0 + jj);
      double* r = R.data() + (size_t)jj * N;
      for (int i = 0; i < N; i++) r[i] = g[i] == 3 ? mu[i] : g[i];
    }
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
    for (int jj = 0; jj < jn; jj++) {
      const uint8_t* g = D.row(j0 + jj);
      for (int i = 0; i < N; i++)
        if (g[i] != 3) mr[i] += R[(size_t)jj * N + i];
    }
  }
  for (int i = 0; i < N; i++)
    if (D.nobs[i] > 0) mr[i] /= D.nobs[i];
  // pass 2: with a_ij = m_ij (r_ij - mean_i) and mask m_ij,
  //   num_ii' = sum_j a_ij a_i'j,  den_ii' = sum_j a_ij^2 m_i'j   (sums over sites observed in both)
  std::vector<double> num((size_t)N * N, 0.0), den((size_t)N * N, 0.0), Asq((size_t)B * N), Mk((size_t)B * N);
  for (int j0 = 0; j0 < M; j0 += B) {
    const int jn = std::min(B, M - j0);
    residuals(j0, jn);
    for (int jj = 0; jj < jn; jj++) {
      const uint8_t* g = D.row(j0 + jj);
      for (int i = 0; i < N; i++) {
        const size_t t = (size_t)jj * N + i;
        const bool obs = g[i] != 3;
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
