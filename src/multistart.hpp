// admixer: comparing Q matrices from different starts (convergence test, --conv).
// Columns of one Q are matched to another's by the assignment that minimises the total squared
// difference (Hungarian algorithm), then the distance measures of popgenDK's testQconv.R are computed.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Minimum-cost assignment for an n x n cost matrix (row major). Returns col[r] = column assigned to row r.
inline std::vector<int> hungarian(const std::vector<double>& cost, int n) {
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0), minv(n + 1);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);
  std::vector<char> used(n + 1);
  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::fill(minv.begin(), minv.end(), INF);
    std::fill(used.begin(), used.end(), 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = 0;
      for (int j = 1; j <= n; j++) {
        if (used[j]) continue;
        const double cur = cost[(size_t)(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) minv[j] = cur, way[j] = j0;
        if (minv[j] < delta) delta = minv[j], j1 = j;
      }
      for (int j = 0; j <= n; j++)
        if (used[j]) u[p[j]] += delta, v[j] -= delta;
        else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  std::vector<int> col(n);
  for (int j = 1; j <= n; j++) col[p[j] - 1] = j - 1;
  return col;
}

struct QDistance {
  double max_abs = 0;   // largest |dQ| over all individuals and ancestries (testQconv.R "max")
  double mean_sum = 0;  // mean over individuals of sum_k |dQ| (testQconv.R "mean")
  double rmse = 0;      // sqrt(mean over individuals of sum_k dQ^2) (testQconv.R "rmse")
};

// Distance between Q matrices A and B (N x K), after matching A's columns to B's.
template <typename TA, typename TB>
inline QDistance q_distance(const TA* A, const TB* B, int N, int K) {
  std::vector<double> cost((size_t)K * K, 0.0);  // cost[a][b] = sum_i (A_ia - B_ib)^2
  for (int i = 0; i < N; i++)
    for (int a = 0; a < K; a++)
      for (int b = 0; b < K; b++) {
        const double d = (double)A[(size_t)i * K + a] - (double)B[(size_t)i * K + b];
        cost[(size_t)a * K + b] += d * d;
      }
  const std::vector<int> col = hungarian(cost, K);  // column a of A <-> column col[a] of B
  QDistance r;
  double sq = 0;
  for (int i = 0; i < N; i++) {
    double s = 0;
    for (int a = 0; a < K; a++) {
      const double d = (double)A[(size_t)i * K + a] - (double)B[(size_t)i * K + col[a]];
      r.max_abs = std::max(r.max_abs, std::fabs(d));
      s += std::fabs(d);
      sq += d * d;
    }
    r.mean_sum += s;
  }
  r.mean_sum /= N;
  r.rmse = std::sqrt(sq / N);
  return r;
}
