// admixer: register-blocked kernels for small compile-time K (Model::sqp_P_small, sqp_Q_small), written with GCC
// vector extensions (VL doubles per vector: 4 for AVX2, 8 when compiled for AVX-512). The portable release build
// (ADMIXER_MULTIARCH, baseline x86-64-v2) compiles them for AVX2 + FMA and uses them only on CPUs that have both.
//   h_rows:    T (rows x n) = P (rows x K) Qt (K x n)                  [h = P Q', individuals contiguous]
//   dot_rows:  G (rows x K) += T (rows x n) Qt' (n x K)                [P-row gradients]
//   bc_rows:   C (rows x NC) += X (rows x n) Z (n x NC)                [P-row Hessians, Z padded to NC]
//   acc_cols:  AT (NC x n) += Y' (NC x jn) X (jn x n)                  [Q-row gradients / Hessians, transposed]
// n is a multiple of VL; padding columns of the operands are zero.
#pragma once
#include <algorithm>
#include <immintrin.h>

#if defined(ADMIXER_MULTIARCH) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define SMALLK_TARGET __attribute__((target("avx2,fma")))
#else
#define SMALLK_TARGET
#endif

namespace smallk {
// the kernels can run on this CPU (-DADMIXER_NO_SMALLK: never, OpenBLAS everywhere)
inline bool available() {
#if defined(ADMIXER_NO_SMALLK)
  return false;
#elif defined(ADMIXER_MULTIARCH) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return true;
#endif
}
#if defined(__AVX512F__)
constexpr int VL = 8;
#else
constexpr int VL = 4;
#endif
typedef double vd __attribute__((vector_size(VL * 8)));
typedef double vdu __attribute__((vector_size(VL * 8), aligned(8)));
[[gnu::always_inline]] SMALLK_TARGET inline vd ld(const double* p) { return *(const vdu*)p; }
[[gnu::always_inline]] SMALLK_TARGET inline void st(double* p, vd v) { *(vdu*)p = v; }
#if defined(__AVX512F__)
[[gnu::always_inline]] SMALLK_TARGET inline vd bc(double x) { return (vd)_mm512_set1_pd(x); }
#else
[[gnu::always_inline]] SMALLK_TARGET inline vd bc(double x) { return (vd)_mm256_set1_pd(x); }
#endif
[[gnu::always_inline]] SMALLK_TARGET inline double hsum(vd v) {
  double s = 0;
  for (int l = 0; l < VL; l++) s += v[l];
  return s;
}
inline int rup(int n) { return (n + VL - 1) / VL * VL; }

// ---- h = P Qt ----
template <int K, int RB, int NV>
[[gnu::always_inline]] SMALLK_TARGET inline void h_mk(const double* P, const double* Qt, int ldq, double* T, int ldt, int ii) {
  vd acc[RB][NV];
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) acc[r][v] = vd{};
  #pragma GCC unroll 64
  for (int k = 0; k < K; k++) {
    vd q[NV];
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) q[v] = ld(Qt + (size_t)k * ldq + ii + v * VL);
    #pragma GCC unroll 64
    for (int r = 0; r < RB; r++) {
      const vd p = bc(P[r * K + k]);
      #pragma GCC unroll 64
      for (int v = 0; v < NV; v++) acc[r][v] += p * q[v];
    }
  }
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) st(T + (size_t)r * ldt + ii + v * VL, acc[r][v]);
}
template <int K, int RB>
SMALLK_TARGET inline void h_rb(const double* P, const double* Qt, int ldq, int n, double* T, int ldt) {
  int ii = 0;
  for (; ii + 2 * VL <= n; ii += 2 * VL) h_mk<K, RB, 2>(P, Qt, ldq, T, ldt, ii);
  for (; ii < n; ii += VL) h_mk<K, RB, 1>(P, Qt, ldq, T, ldt, ii);
}
template <int K>
SMALLK_TARGET inline void h_rows(int rows, const double* P, const double* Qt, int ldq, int n, double* T, int ldt) {
  constexpr int RB = 4;
  int r = 0;
  for (; r + RB <= rows; r += RB) h_rb<K, RB>(P + (size_t)r * K, Qt, ldq, n, T + (size_t)r * ldt, ldt);
  for (; r < rows; r++) h_rb<K, 1>(P + (size_t)r * K, Qt, ldq, n, T + (size_t)r * ldt, ldt);
}

// ---- G += T Qt' (dot products over the individuals) ----
template <int K, int RB>
SMALLK_TARGET inline void dot_rb(const double* X, int ldx, const double* Qt, int ldq, int n, double* G) {
  vd acc[RB][K];
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int k = 0; k < K; k++) acc[r][k] = vd{};
  for (int ii = 0; ii < n; ii += VL) {
    vd x[RB];
    #pragma GCC unroll 64
    for (int r = 0; r < RB; r++) x[r] = ld(X + (size_t)r * ldx + ii);
    #pragma GCC unroll 64
    for (int k = 0; k < K; k++) {
      const vd q = ld(Qt + (size_t)k * ldq + ii);
      #pragma GCC unroll 64
      for (int r = 0; r < RB; r++) acc[r][k] += x[r] * q;
    }
  }
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int k = 0; k < K; k++) G[r * K + k] += hsum(acc[r][k]);
}
template <int K>
SMALLK_TARGET inline void dot_rows(int rows, const double* X, int ldx, const double* Qt, int ldq, int n, double* G) {
  constexpr int RB = K <= 3 ? 3 : K <= 6 ? 2 : 1;
  int r = 0;
  for (; r + RB <= rows; r += RB) dot_rb<K, RB>(X + (size_t)r * ldx, ldx, Qt, ldq, n, G + (size_t)r * K);
  for (; r < rows; r++) dot_rb<K, 1>(X + (size_t)r * ldx, ldx, Qt, ldq, n, G + (size_t)r * K);
}

// ---- C (rows x NV*VL, row stride ldc) += X (rows x n) Z (n x NV*VL, row stride ldz), broadcasting X ----
template <int NV, int RB>
SMALLK_TARGET inline void bc_rb(const double* X, int ldx, const double* Z, int ldz, int i0, int i1, double* C, int ldc) {
  vd acc[RB][NV];
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) acc[r][v] = ld(C + (size_t)r * ldc + v * VL);
  for (int ii = i0; ii < i1; ii++) {
    const double* z = Z + (size_t)ii * ldz;
    vd zv[NV];
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) zv[v] = ld(z + v * VL);
    #pragma GCC unroll 64
    for (int r = 0; r < RB; r++) {
      const vd x = bc(X[(size_t)r * ldx + ii]);
      #pragma GCC unroll 64
      for (int v = 0; v < NV; v++) acc[r][v] += x * zv[v];
    }
  }
  #pragma GCC unroll 64
  for (int r = 0; r < RB; r++)
    #pragma GCC unroll 64
    for (int v = 0; v < NV; v++) st(C + (size_t)r * ldc + v * VL, acc[r][v]);
}
// Column-tiled: RB rows x CV vectors of columns per micro-kernel, so each loaded Z vector is used RB times
// (blocking whole rows, as bc_rb alone does, leaves RB = 1 and one load per FMA once kp > 12 vectors).
template <int NV>
SMALLK_TARGET inline void bc_rows(int rows, const double* X, int ldx, const double* Z, int n, double* C) {
  constexpr int ldz = NV * VL, ldc = NV * VL;
  constexpr int CV = std::min(NV, 3), RB = VL == 8 ? 8 : 4;  // 24 (AVX-512) or 12 (AVX2) accumulators
  constexpr int CH = 128;                                     // individuals per chunk
  for (int c0 = 0; c0 < n; c0 += CH) {
    const int c1 = std::min(n, c0 + CH);
    const int r_end = rows - rows % RB;
    int v0 = 0;
    for (; v0 + CV <= NV; v0 += CV) {
      for (int r = 0; r < r_end; r += RB)
        bc_rb<CV, RB>(X + (size_t)r * ldx, ldx, Z + v0 * VL, ldz, c0, c1, C + (size_t)r * ldc + v0 * VL, ldc);
      for (int r = r_end; r < rows; r++)
        bc_rb<CV, 1>(X + (size_t)r * ldx, ldx, Z + v0 * VL, ldz, c0, c1, C + (size_t)r * ldc + v0 * VL, ldc);
    }
    for (; v0 < NV; v0++) {
      for (int r = 0; r < r_end; r += RB)
        bc_rb<1, RB>(X + (size_t)r * ldx, ldx, Z + v0 * VL, ldz, c0, c1, C + (size_t)r * ldc + v0 * VL, ldc);
      for (int r = r_end; r < rows; r++)
        bc_rb<1, 1>(X + (size_t)r * ldx, ldx, Z + v0 * VL, ldz, c0, c1, C + (size_t)r * ldc + v0 * VL, ldc);
    }
  }
}

// ---- AT (NC x n, stride lda) += Y' X: AT[c][ii] += sum_jj Y[jj*ldy + c] X[jj*ldx + ii] ----
template <int NC, int NVI>
[[gnu::always_inline]] SMALLK_TARGET inline void acc_mk(int jn, const double* X, int ldx, const double* Y, int ldy, double* AT,
                                          int lda, int ii) {
  vd acc[NC][NVI];
  #pragma GCC unroll 64
  for (int c = 0; c < NC; c++)
    #pragma GCC unroll 64
    for (int v = 0; v < NVI; v++) acc[c][v] = ld(AT + (size_t)c * lda + ii + v * VL);
  for (int jj = 0; jj < jn; jj++) {
    vd x[NVI];
    #pragma GCC unroll 64
    for (int v = 0; v < NVI; v++) x[v] = ld(X + (size_t)jj * ldx + ii + v * VL);
    #pragma GCC unroll 64
    for (int c = 0; c < NC; c++) {
      const vd y = bc(Y[(size_t)jj * ldy + c]);
      #pragma GCC unroll 64
      for (int v = 0; v < NVI; v++) acc[c][v] += y * x[v];
    }
  }
  #pragma GCC unroll 64
  for (int c = 0; c < NC; c++)
    #pragma GCC unroll 64
    for (int v = 0; v < NVI; v++) st(AT + (size_t)c * lda + ii + v * VL, acc[c][v]);
}
template <int NC>
SMALLK_TARGET inline void acc_cols(int jn, const double* X, int ldx, const double* Y, int ldy, int n, double* AT, int lda) {
  constexpr int G = 6;  // columns per group, 2 vectors of individuals: 12 accumulators
  if constexpr (NC > G) {
    acc_cols<G>(jn, X, ldx, Y, ldy, n, AT, lda);
    acc_cols<NC - G>(jn, X, ldx, Y + G, ldy, n, AT + (size_t)G * lda, lda);
  } else {
    int ii = 0;
    for (; ii + 2 * VL <= n; ii += 2 * VL) acc_mk<NC, 2>(jn, X, ldx, Y, ldy, AT, lda, ii);
    for (; ii < n; ii += VL) acc_mk<NC, 1>(jn, X, ldx, Y, ldy, AT, lda, ii);
  }
}
}  // namespace smallk
