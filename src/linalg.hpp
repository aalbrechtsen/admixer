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
  std::vector<int> state(n, 0);  // 0 free, -1 at lower bound, +1 at upper bound
  for (int i = 0; i < n; i++) {
    d[i] = 0;
    if (lo[i] >= 0) state[i] = -1, d[i] = lo[i];
    else if (hi[i] <= 0) state[i] = 1, d[i] = hi[i];
  }
  std::vector<int> fr;
  std::vector<double> A, b, r(n);
  for (int iter = 0; iter < 10 * n + 20; iter++) {
    fr.clear();
    for (int i = 0; i < n; i++)
      if (state[i] == 0) fr.push_back(i);
    const int nf = fr.size(), ns = nf + (eq ? 1 : 0);
    double nu = 0;
    if (nf > 0) {
      A.assign(ns * ns, 0);
      b.assign(ns, 0);
      double sumW = 0;
      for (int i = 0; i < n; i++)
        if (state[i] != 0) sumW += d[i];
      for (int a = 0; a < nf; a++) {
        const int ia = fr[a];
        double s = g[ia];
        for (int i = 0; i < n; i++)
          if (state[i] != 0) s -= H[ia * n + i] * d[i];
        b[a] = s;
        for (int c = 0; c < nf; c++) A[a * ns + c] = H[ia * n + fr[c]];
        A[a * ns + a] += ridge;
        if (eq) A[a * ns + nf] = A[nf * ns + a] = 1;
      }
      if (eq) b[nf] = -sumW;
      if (!solve_linear(ns, A.data(), b.data())) break;  // keep current feasible d
      if (eq) nu = b[nf];
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
