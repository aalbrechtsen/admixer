// admixer: --search, a split-merge search for a better optimum within a fixed budget of fits.
//
// From one fit (a random start converged loosely, or a given start), each round draws untried split-merge moves
// at random and refits them; the best refit replaces the current solution if its log-likelihood is higher.
//   move: merge components a and b (one of the NMERGE pairs with the closest P), which frees column b; split
//         component c (any other) by moving the Q of half of its individuals into column b. The halves are the
//         signs of the first principal component of the genotypes of c's main individuals (largest Q).
//   refit: P fitted to the moved Q by EM steps with Q held fixed (P_EM steps from a random P), then the usual
//         mini-batch warm-up and quasi-Newton phase to the loose tolerance TOL. One refit costs about one
//         standard run; without the warm-up it costs 4-5 (it creeps along plateaus).
// The budget counts standard runs (time of a random start fitted to -C 1e-4). Fixed charges, so that a run is
// reproducible: a loose random start 0.75 (mostly the warm-up), a refit 0.6 (measured on the reindeer data, 8
// threads: 0.5 with 107k SNPs, 0.6 with 321k). The winner is polished to -C (no warm-up).
// No cheap score predicts the winning move (neither early log-likelihoods nor the log-likelihood after the P fit
// ranked it near the top on the reindeer data), so moves are drawn at random. Developed and tested in
// global_search/README.md of the admix project (reindeer data, K = 11).
#pragma once
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "beagle.hpp"
#include "fit.hpp"
#include "io.hpp"
#include "log.hpp"
#include "model.hpp"

struct SearchSettings {
  double budget = 20;       // in standard runs
  double start_cost = 0.75; // charge of a loose random start (0.5 for a given start, --Qin)
  double refit_cost = 0.6;  // charge of a refit
  int per_round = 2;        // moves refitted per round
  int nmerge = 4;           // merge candidates: the pairs of components with the closest P
  int p_em = 20;            // EM steps (Q fixed) that fit P to a moved Q
  double tol = 10;          // log-likelihood tolerance of the start and the refits (the winner is polished)
  int pc_snps = 50000;      // at most this many SNPs (evenly spaced) for the split direction
  int pc_iter = 30;         // power iterations for the split direction
};

// Value of entry (j, i) for the split direction: genotype (copies of A2), or for GLs the expected genotype under
// a flat prior. false: missing.
inline bool entry_value(const Genotypes& D, int j, int i, double& v) {
  const uint8_t g = D.get(j, i);
  v = g;
  return g != 3;
}
inline bool entry_value(const GLData& D, int j, int i, double& v) {
  if (!D.krow(j)[i]) return false;
  const float* l = D.row(j) + 3 * (size_t)i;
  const double s = (double)l[0] + l[1] + l[2];
  if (!(s > 0)) return false;
  v = (l[1] + 2.0 * l[2]) / s;
  return true;
}

// Sign of each individual of S on the first principal component of their centred values (SNP means over S,
// missing entries 0), by power iteration on X'X without forming it. true = positive side.
template <class Data>
std::vector<char> split_direction(const Data& D, const std::vector<int>& S, const SearchSettings& ss) {
  const int n = S.size();
  const int step = std::max(1, D.M / std::max(1, ss.pc_snps));
  std::vector<int> snps;
  for (int j = 0; j < D.M; j += step) snps.push_back(j);
  const int ms = snps.size();
  std::vector<float> X((size_t)ms * n);  // centred values, SNP-major
#pragma omp parallel for schedule(static)
  for (int t = 0; t < ms; t++) {
    float* x = X.data() + (size_t)t * n;
    double s = 0, v;
    int c = 0;
    for (int u = 0; u < n; u++)
      if (entry_value(D, snps[t], S[u], v)) x[u] = (float)v, s += v, c++;
      else x[u] = NAN;
    const double mean = c ? s / c : 0;
    for (int u = 0; u < n; u++) x[u] = std::isnan(x[u]) ? 0.f : (float)(x[u] - mean);
  }
  std::vector<double> a(n), w(ms);
  for (int u = 0; u < n; u++) a[u] = 1.0 + 0.01 * u;  // deterministic start, not orthogonal to the leading vector
  for (int it = 0; it < ss.pc_iter; it++) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < ms; t++) {
      const float* x = X.data() + (size_t)t * n;
      double s = 0;
      for (int u = 0; u < n; u++) s += x[u] * a[u];
      w[t] = s;
    }
    std::vector<double> b(n, 0.0);
    const int nt = omp_get_max_threads();
    std::vector<double> part((size_t)nt * n, 0.0);
#pragma omp parallel
    {
      double* p = part.data() + (size_t)omp_get_thread_num() * n;
#pragma omp for schedule(static)
      for (int t = 0; t < ms; t++) {
        const float* x = X.data() + (size_t)t * n;
        for (int u = 0; u < n; u++) p[u] += x[u] * w[t];
      }
    }
    for (int th = 0; th < nt; th++)
      for (int u = 0; u < n; u++) b[u] += part[(size_t)th * n + u];
    double nrm = 0;
    for (double v : b) nrm += v * v;
    nrm = std::sqrt(nrm);
    if (!(nrm > 0)) break;
    for (int u = 0; u < n; u++) a[u] = b[u] / nrm;
  }
  std::vector<char> pos(n);
  for (int u = 0; u < n; u++) pos[u] = a[u] > 0;
  return pos;
}

struct Move {
  int a, b, c;            // merge b into a, split c into the freed column b
  std::vector<int> half;  // individuals whose Q of c moves to b
};

// All moves from the solution x: NMERGE closest pairs (mean squared P difference) times every other component.
template <class Data>
std::vector<Move> make_moves(const Data& D, const Model<Data>& m, const Vec& x, const SearchSettings& ss) {
  const int K = m.K, N = D.N, M = D.M;
  const double* P = x.data();
  const double* Q = x.data() + m.nP;
  std::vector<double> dist(K * K, 0.0);
  for (int j = 0; j < M; j++)
    for (int a = 0; a < K; a++)
      for (int b = a + 1; b < K; b++) {
        const double d = P[(size_t)j * K + a] - P[(size_t)j * K + b];
        dist[a * K + b] += d * d;
      }
  std::vector<std::pair<int, int>> pairs;
  for (int a = 0; a < K; a++)
    for (int b = a + 1; b < K; b++) pairs.push_back({a, b});
  std::stable_sort(pairs.begin(), pairs.end(), [&](auto u, auto v) {
    return dist[u.first * K + u.second] < dist[v.first * K + v.second];
  });
  pairs.resize(std::min<size_t>(pairs.size(), ss.nmerge));
  // the split of each component: its main individuals (largest Q), or the two with the largest Q if fewer
  std::vector<std::vector<int>> halves(K);
  for (int c = 0; c < K; c++) {
    std::vector<int> S;
    for (int i = 0; i < N; i++) {
      const double* q = Q + (size_t)i * K;
      if (std::max_element(q, q + K) - q == c) S.push_back(i);
    }
    if (S.size() < 2) {
      std::vector<int> idx(N);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + std::min(2, N), idx.end(),
                        [&](int u, int v) { return Q[(size_t)u * K + c] > Q[(size_t)v * K + c]; });
      S.assign(idx.begin(), idx.begin() + std::min(2, N));
    }
    if (S.size() < 2) continue;
    const std::vector<char> pos = split_direction(D, S, ss);
    for (size_t u = 0; u < S.size(); u++)
      if (pos[u]) halves[c].push_back(S[u]);
    if (halves[c].size() == S.size()) halves[c].clear();  // no split
  }
  std::vector<Move> moves;
  for (auto [a, b] : pairs)
    for (int c = 0; c < K; c++)
      if (c != a && c != b && !halves[c].empty()) moves.push_back({a, b, c, halves[c]});
  return moves;
}

struct SearchResult {
  double loglik = 0;   // after polishing
  int iterations = 0;  // of the polishing fit
  int rounds = 0, refits = 0, accepted = 0;
  double used = 0;     // budget used, in standard runs
};

// The search from x (a fitted solution with log-likelihood ll, at tolerance ss.tol). On return x is the polished
// best solution. fs: the settings of a standard run (warm-up, seed); the refits use its warm-up.
template <class Data>
SearchResult run_search(const Data& D, Model<Data>& m, Vec& x, double ll, const FitSettings& fs,
                        const SearchSettings& ss, double used, const std::string& table) {
  const int K = m.K;
  const size_t nP = m.nP;
  SearchResult res;
  res.used = used;
  std::mt19937_64 rng(fs.seed + 1000003);
  FILE* tf = table.empty() ? nullptr : std::fopen(table.c_str(), "w");
  if (tf) std::fprintf(tf, "round\tmerge_into\tmerged\tsplit\tmoved\tloglik\tgain\taccepted\n");
  say("Search: split-merge moves, %d per round, budget %g standard runs (start %g); refits to tolerance %g\n",
      ss.per_round, ss.budget, used, ss.tol);
  say("Search start: loglik %f\n", ll);
  std::vector<Move> moves = make_moves(D, m, x, ss);
  std::shuffle(moves.begin(), moves.end(), rng);
  size_t next = 0;
  Vec y(x.size()), z;
  while (next < moves.size() && res.used + ss.per_round * ss.refit_cost <= ss.budget + 1e-9) {
    res.rounds++;
    double best_ll = -INFINITY;
    Vec best_x;
    std::vector<std::string> rows;  // table rows of this round (the accepted flag is known at its end)
    size_t best_row = 0;
    const size_t end = std::min(moves.size(), next + ss.per_round);
    for (; next < end; next++) {
      const Move& mv = moves[next];
      z = x;
      double* Q = z.data() + nP;
      for (int i = 0; i < D.N; i++) {
        double* q = Q + (size_t)i * K;
        q[mv.a] += q[mv.b], q[mv.b] = 0;
      }
      for (int i : mv.half) {
        double* q = Q + (size_t)i * K;
        q[mv.b] = q[mv.c], q[mv.c] = 0;
      }
      // P: random start, then EM steps with every row of Q held fixed
      std::uniform_real_distribution<double> unif(0.0, 1.0);
      for (size_t t = 0; t < nP; t++) z[t] = 0.05 + 0.9 * unif(rng);
      m.project(z.data());
      std::fill(m.qfix.begin(), m.qfix.end(), 1);
      for (int e = 0; e < ss.p_em; e++) {
        m.em(z.data(), y.data());
        z.swap(y);
      }
      std::fill(m.qfix.begin(), m.qfix.end(), 0);
      FitSettings fr = fs;
      fr.prime = 0, fr.tol = ss.tol, fr.seed = fs.seed + 7919 * (res.refits + 1);
      Fitter<Model<Data>> fit(m, fr);
      fit.verbose = false;
      const FitResult r = fit.run(z);
      res.refits++, res.used += ss.refit_cost;
      say("Search round %d: merge %d into %d, split %d (%zu individuals moved): loglik %f (%+.2f), %d iterations, "
          "%.1f sec\n", res.rounds, mv.b + 1, mv.a + 1, mv.c + 1, mv.half.size(), r.loglik, r.loglik - ll,
          r.iterations, r.seconds);
      char row[160];
      std::snprintf(row, sizeof row, "%d\t%d\t%d\t%d\t%zu\t%.6f\t%.6f", res.rounds, mv.a + 1, mv.b + 1, mv.c + 1,
                    mv.half.size(), r.loglik, r.loglik - ll);
      rows.push_back(row);
      if (r.loglik > best_ll) best_ll = r.loglik, best_x = z, best_row = rows.size() - 1;
    }
    const bool accept = best_ll > ll + 0.01;
    if (tf)
      for (size_t u = 0; u < rows.size(); u++) std::fprintf(tf, "%s\t%d\n", rows[u].c_str(), accept && u == best_row);
    if (accept) {
      say("Search round %d: accepted, loglik %f -> %f\n", res.rounds, ll, best_ll);
      ll = best_ll, x = best_x;
      res.accepted++;
      moves = make_moves(D, m, x, ss);
      std::shuffle(moves.begin(), moves.end(), rng);
      next = 0;
    }
  }
  if (tf) std::fclose(tf);
  // polish: from the best solution to the tolerance of a standard run, without EM steps or warm-up
  FitSettings fp = fs;
  fp.prime = 0, fp.minibatch = 0;
  Fitter<Model<Data>> fit(m, fp);
  fit.verbose = false;
  const FitResult r = fit.run(x);
  res.loglik = r.loglik, res.iterations = r.iterations;
  say("Search: %d rounds, %d refits, %d moves accepted, %.1f of %g standard runs used; polished loglik %f "
      "(%d iterations, %.1f sec)\n", res.rounds, res.refits, res.accepted, res.used, ss.budget, r.loglik, r.iterations,
      r.seconds);
  return res;
}
