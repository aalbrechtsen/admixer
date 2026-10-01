// admixer: small dense linear algebra: linear solves, simplex projection, box/simplex QP.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// Solves A x = b in place (A n x n row major, overwritten; b -> x). Partial pivoting.
// Returns false if A is numerically singular.
inline bool solve_linear(int n, double* A, double* b) {
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++)
      if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
    if (std::fabs(A[p * n + c]) < 1e-300) return false;
    if (p != c) {
      for (int k = 0; k < n; k++) std::swap(A[c * n + k], A[p * n + k]);
      std::swap(b[c], b[p]);
    }
    for (int r = c + 1; r < n; r++) {
      const double m = A[r * n + c] / A[c * n + c];
      if (m == 0) continue;
      for (int k = c; k < n; k++) A[r * n + k] -= m * A[c * n + k];
      b[r] -= m * b[c];
    }
  }
  for (int c = n - 1; c >= 0; c--) {
    double s = b[c];
    for (int k = c + 1; k < n; k++) s -= A[c * n + k] * b[k];
    b[c] = s / A[c * n + c];
    if (!std::isfinite(b[c])) return false;
  }
  return true;
}

// Euclidean projection of x (length n) onto {y : y_k >= lo, sum y = 1}.
inline void project_simplex(int n, double* x, double lo) {
  const double s = 1.0 - n * lo;
  std::vector<double> u(n);
  for (int k = 0; k < n; k++) u[k] = x[k] - lo;
  std::sort(u.begin(), u.end(), std::greater<double>());
  double cum = 0, theta = 0;
  for (int k = 0; k < n; k++) {
    cum += u[k];
    const double t = (cum - s) / (k + 1);
    if (u[k] - t > 0) theta = t;
  }
  for (int k = 0; k < n; k++) x[k] = std::max(x[k] - lo - theta, 0.0) + lo;
}

// Cholesky factorisation A = U'U in place of the upper triangle of the leading n x n block of A (row
// stride ld; the lower triangle is not used). Right-looking, so the inner loops are independent row
// updates that vectorise (dot-product chains are latency bound at these small sizes). Returns false if A
// is not numerically positive definite.
inline bool cholesky_ld(int n, double* A, int ld) {
  for (int j = 0; j < n; j++) {
    double* uj = A + (size_t)j * ld;
    if (!(uj[j] > 0)) return false;
    const double l = std::sqrt(uj[j]), il = 1 / l;
    uj[j] = l;
    for (int k = j + 1; k < n; k++) uj[k] *= il;
    for (int i = j + 1; i < n; i++) {
      double* ai = A + (size_t)i * ld;
      const double u = uj[i];
#pragma omp simd
      for (int k = i; k < n; k++) ai[k] -= u * uj[k];
    }
  }
  return true;
}
// Solves U'U x = b in place (U from cholesky_ld).
inline void chol_solve_ld(int n, const double* U, int ld, double* b) {
  for (int i = 0; i < n; i++) {  // U' y = b, column by column
    const double* ui = U + (size_t)i * ld;
    const double y = b[i] /= ui[i];
#pragma omp simd
    for (int k = i + 1; k < n; k++) b[k] -= ui[k] * y;
  }
  for (int i = n - 1; i >= 0; i--) {  // U x = y
    const double* ui = U + (size_t)i * ld;
    double s = b[i];
#pragma omp simd reduction(- : s)
    for (int k = i + 1; k < n; k++) s -= ui[k] * b[k];
    b[i] = s / ui[i];
  }
}

// Primal-dual active-set method (Hintermueller, Ito & Kunisch 2002) for the box QP
//   minimise 0.5 d'Hd - g'd  s.t.  lo <= d <= hi   (H symmetric PD, row major, ridge added on the diagonal).
// Each iteration fixes the variables predicted to be at a bound, solves for the others, and re-predicts
// from the primal infeasibility and the multipliers mu = (H + ridge) d - g; many bounds can change per
// iteration. It stops at an exact KKT point (unique solution) or returns false after maxit iterations or
// if a subsystem is not positive definite; the caller then uses the primal active-set method.
inline bool qp_box_pdas(int n, const double* H, const double* g, const double* lo, const double* hi, double ridge,
                        double mtol, double* d, int* state, int* fr, double* A, double* b, int maxit = 8) {
  for (int i = 0; i < n; i++) state[i] = 0;
  for (int it = 0; it < maxit; it++) {
    int nf = 0;
    for (int i = 0; i < n; i++) {
      if (state[i] == 0) fr[nf++] = i;
      else d[i] = state[i] < 0 ? lo[i] : hi[i];
    }
    for (int a = 0; a < nf; a++) {
      const int ia = fr[a];
      double s = g[ia];
      for (int i = 0; i < n; i++)
        if (state[i] != 0) s -= H[ia * n + i] * d[i];
      b[a] = s;
      for (int c = a; c < nf; c++) A[a * nf + c] = H[ia * n + fr[c]];
      A[a * nf + a] += ridge;
    }
    if (nf > 0) {
      if (!cholesky_ld(nf, A, nf)) return false;
      chol_solve_ld(nf, A, nf, b);
      for (int a = 0; a < nf; a++) d[fr[a]] = b[a];
    }
    bool changed = false, kkt = true;
    for (int i = 0; i < n; i++) {
      int ns = 0;
      if (state[i] == 0) {
        if (d[i] < lo[i]) ns = -1;
        else if (d[i] > hi[i]) ns = 1;
        if (ns != 0) kkt = false;
      } else {
        double mu = -g[i] + ridge * d[i];
        for (int k = 0; k < n; k++) mu += H[i * n + k] * d[k];
        // at the lower bound the objective must not decrease upwards (mu >= 0), at the upper bound mu <= 0
        const double viol = state[i] < 0 ? -mu : mu;
        ns = viol > mtol ? 0 : state[i];
        if (ns == 0) kkt = false;
      }
      if (ns != state[i]) state[i] = ns, changed = true;
    }
    if (kkt) return true;
    if (!changed) return false;
  }
  return false;
}

// Active-set solver for
//   minimise 0.5 d'Hd - g'd   s.t.  lo <= d <= hi  (and sum d = 0 if eq)
// H is n x n symmetric PSD (row major); d = 0 must be feasible. Result in d.
// This is the per-block quadratic subproblem of ADMIXTURE's sequential quadratic programming.
inline void qp_active_set(int n, const double* H, const double* g, const double* lo, const double* hi,
                          bool eq, double* d) {
  double tr = 0, gmax = 0;
  for (int i = 0; i < n; i++) tr += H[i * n + i], gmax = std::max(gmax, std::fabs(g[i]));
  const double ridge = 1e-10 * tr / n + 1e-300;
  const double mtol = 1e-12 * (gmax + tr / n) + 1e-300;
  // per-thread workspace: no allocation per call (this solver runs once per row of P and of Q)
  thread_local std::vector<int> ws_i;
  thread_local std::vector<double> ws_d;
  ws_i.resize(2 * n), ws_d.resize((size_t)(n + 1) * (n + 1) + 3 * n + 1);
  int* const state = ws_i.data();  // 0 free, -1 at lower bound, +1 at upper bound
  int* const fr = state + n;       // the free variables
  double* const A = ws_d.data();
  double* const b = A + (size_t)(n + 1) * (n + 1);
  double* const v = b + n + 1;
  double* const r = v + n;
  if (!eq && qp_box_pdas(n, H, g, lo, hi, ridge, mtol, d, state, fr, A, b)) return;
  for (int i = 0; i < n; i++) state[i] = 0;
  for (int i = 0; i < n; i++) {
    d[i] = 0;
    if (lo[i] >= 0) state[i] = -1, d[i] = lo[i];
    else if (hi[i] <= 0) state[i] = 1, d[i] = hi[i];
  }
  for (int iter = 0; iter < 10 * n + 20; iter++) {
    int nf = 0;
    for (int i = 0; i < n; i++)
      if (state[i] == 0) fr[nf++] = i;
    double nu = 0;
    if (nf > 0) {
      // free-set system (H_ff + ridge) d_f [+ nu 1] = g_f - H_fb d_b  [, 1'd_f = -sum d_b], by Cholesky:
      // d_f = u - nu v with u = A^-1 b, v = A^-1 1, nu = (1'u + sum d_b) / 1'v
      double sumW = 0;
      for (int i = 0; i < n; i++)
        if (state[i] != 0) sumW += d[i];
      for (int a = 0; a < nf; a++) {
        const int ia = fr[a];
        double s = g[ia];
        for (int i = 0; i < n; i++)
          if (state[i] != 0) s -= H[ia * n + i] * d[i];
        b[a] = s;
        for (int c = a; c < nf; c++) A[a * nf + c] = H[ia * n + fr[c]];
        A[a * nf + a] += ridge;
      }
      if (cholesky_ld(nf, A, nf)) {
        chol_solve_ld(nf, A, nf, b);
        if (eq) {
          for (int a = 0; a < nf; a++) v[a] = 1;
          chol_solve_ld(nf, A, nf, v);
          double su = 0, sv = 0;
          for (int a = 0; a < nf; a++) su += b[a], sv += v[a];
          nu = (su + sumW) / sv;
          for (int a = 0; a < nf; a++) b[a] -= nu * v[a];
        }
      } else {
        // not numerically positive definite: the bordered system by LU with pivoting
        const int ns = nf + (eq ? 1 : 0);
        for (int a = 0; a < nf; a++) {
          double s = g[fr[a]];
          for (int i = 0; i < n; i++)
            if (state[i] != 0) s -= H[fr[a] * n + i] * d[i];
          b[a] = s;
          for (int c = 0; c < nf; c++) A[a * ns + c] = H[fr[a] * n + fr[c]];
          A[a * ns + a] += ridge;
          if (eq) A[a * ns + nf] = A[nf * ns + a] = 1;
        }
        if (eq) A[nf * ns + nf] = 0, b[nf] = -sumW;
        if (!solve_linear(ns, A, b)) break;  // keep current feasible d
        if (eq) nu = b[nf];
      }
      // step towards the equality-constrained minimiser, stopping at the first bound
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
    // stationary on the free set: check the multipliers of the bound variables
    for (int i = 0; i < n; i++) {
      double s = -g[i] + ridge * d[i];
      for (int k = 0; k < n; k++) s += H[i * n + k] * d[k];
      r[i] = s;
    }
    if (nf == 0 && eq) {  // degenerate: pick the multiplier that best balances the bound variables
      double s = 0;
      for (int i = 0; i < n; i++) s -= r[i];
      nu = s / n;
    }
    int rel = -1;
    double worst = mtol;
    for (int i = 0; i < n; i++) {
      if (state[i] == 0) continue;
      const double mu = r[i] + nu;
      const double viol = state[i] < 0 ? -mu : mu;
      if (viol > worst) worst = viol, rel = i;
    }
    if (rel < 0) break;
    state[rel] = 0;
  }
}
