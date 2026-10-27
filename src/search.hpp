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
#include <memory>
#include <vector>

#include "beagle.hpp"
#include "fit.hpp"
#include "io.hpp"
#include "log.hpp"
#include "model.hpp"
#include "multistart.hpp"

struct SearchSettings {
  double budget = 20;       // in standard runs
  double start_cost = 0.75; // charge of a loose random start (0.5 for a given start, --Qin)
  double refit_cost = 0.6;  // charge of a refit
  int per_round = 2;        // moves refitted per round
  int nmerge = 4;           // merge candidates: the closest pairs by P, and the closest by P scaled by Q mass
  int p_em = 20;            // EM steps (Q fixed) that fit P to a moved Q
  double tol = 10;          // log-likelihood tolerance of the start and the refits (the winner is polished)
  int pc_snps = 50000;      // at most this many SNPs (evenly spaced) for the split direction
  int pc_iter = 30;         // power iterations for the split direction
  double flip_tol = 1;      // cross-flips (mixing of --conv --search): tolerance of the flip fits
  double flip_gain = 1;     // a flip is kept if it raises the log-likelihood by more (gains below are fit noise)
  // speed and move order (tested in global_search/variants): about half the time on real data, same hits
  double abort_frac = 0.001; // --search-abort-frac=F: stop a refit after its warm-up if log L < current - F |log L|,
                             // or if it is back at the current solution (max |dQ| < 0.01); 0 = off. Relative:
                             // a refit that wins in the end can still be 2,000-3,600 below after its warm-up
                             // (6.3 million on the simulated hard sets), and a fixed 1,000 stopped the winners
  double abort_cost = 0.35;  // charge of a refit stopped after its warm-up
  int patience = 6;          // --search-patience=R: stop after R rounds in a row without an accepted move; 0 = off
  bool order_split = true;   // --search-order=split: draw moves with weights = how strongly the split component
                             // falls into two groups (first / second eigenvalue); =random: uniformly
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
// missing entries 0), by power iteration on X'X without forming it. true = positive side. ratio: first / second
// eigenvalue of X'X (second by deflation): large when the individuals fall into two groups, ~1 when they do not
// (set to 1 for fewer than 4 individuals, where it means nothing).
struct SplitDir {
  std::vector<char> pos;
  double ratio = 1;
};
template <class Data>
SplitDir split_direction(const Data& D, const std::vector<int>& S, const SearchSettings& ss) {
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
  std::vector<double> w(ms);
  auto XtX = [&](const std::vector<double>& a) {  // X'X a
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
    return b;
  };
  // power iteration; lambda = |X'X a| for the normalised iterate a; deflate = the leading pair to remove
  auto power = [&](std::vector<double>& a, double& lambda, const std::vector<double>* lead, double llead) {
    lambda = 0;
    for (int it = 0; it < ss.pc_iter; it++) {
      std::vector<double> b = XtX(a);
      if (lead) {
        double d = 0;
        for (int u = 0; u < n; u++) d += (*lead)[u] * a[u];
        for (int u = 0; u < n; u++) b[u] -= llead * d * (*lead)[u];
      }
      double nrm = 0;
      for (double v : b) nrm += v * v;
      nrm = std::sqrt(nrm);
      if (!(nrm > 0)) break;
      lambda = nrm;
      for (int u = 0; u < n; u++) a[u] = b[u] / nrm;
    }
  };
  SplitDir res;
  std::vector<double> a(n);
  for (int u = 0; u < n; u++) a[u] = 1.0 + 0.01 * u;  // deterministic start, not orthogonal to the leading vector
  double l1 = 0;
  power(a, l1, nullptr, 0);
  res.pos.resize(n);
  for (int u = 0; u < n; u++) res.pos[u] = a[u] > 0;
  if (ss.order_split && n >= 4) {
    std::vector<double> a2(n);
    for (int u = 0; u < n; u++) a2[u] = (u % 2 ? 1.0 : -1.0) + 0.01 * u;
    double l2 = 0;
    power(a2, l2, &a, l1);
    res.ratio = l2 > 0 ? l1 / l2 : 1;
  }
  return res;
}

struct Move {
  int a, b, c;            // merge b into a, split c into the freed column b
  std::vector<int> half;  // individuals whose Q of c moves to b
  double weight = 1;      // --search-order=split: the split component's first / second eigenvalue
};

// All moves from the solution x: the merge pairs times every other component. Merge pairs: the NMERGE closest by
// the mean squared P difference d_ab, and the NMERGE closest by d_ab m_a m_b / (m_a + m_b), m = total Q of a
// component. The P of a component with little Q (an admixed group given a component of its own) is estimated from
// few individuals; its sampling noise, ~ 1/m_a + 1/m_b, puts it far from every other component by d_ab alone.
// On the reindeer data (K = 11) two runs stopped where the needed merge, of two such components, ranked 13th of
// 55 by d_ab and 3rd scaled.
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
  std::vector<double> mass(K, 0.0);
  for (int i = 0; i < N; i++)
    for (int k = 0; k < K; k++) mass[k] += Q[(size_t)i * K + k];
  std::vector<std::pair<int, int>> all, pairs;
  for (int a = 0; a < K; a++)
    for (int b = a + 1; b < K; b++) all.push_back({a, b});
  auto closest = [&](auto score) {  // the NMERGE pairs with the smallest score, added to pairs if new
    std::vector<std::pair<int, int>> s = all;
    std::stable_sort(s.begin(), s.end(), [&](auto u, auto v) { return score(u) < score(v); });
    for (size_t t = 0; t < std::min<size_t>(s.size(), ss.nmerge); t++)
      if (std::find(pairs.begin(), pairs.end(), s[t]) == pairs.end()) pairs.push_back(s[t]);
  };
  closest([&](std::pair<int, int> u) { return dist[u.first * K + u.second]; });
  closest([&](std::pair<int, int> u) {
    const double ma = mass[u.first], mb = mass[u.second];
    return dist[u.first * K + u.second] * (ma + mb > 0 ? ma * mb / (ma + mb) : 0.0);
  });
  // the split of each component: its main individuals (largest Q), or the two with the largest Q if fewer
  std::vector<std::vector<int>> halves(K);
  std::vector<double> ratio(K, 1.0);
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
    const SplitDir sd = split_direction(D, S, ss);
    ratio[c] = sd.ratio;
    for (size_t u = 0; u < S.size(); u++)
      if (sd.pos[u]) halves[c].push_back(S[u]);
    if (halves[c].size() == S.size()) halves[c].clear();  // no split
  }
  std::vector<Move> moves;
  for (auto [a, b] : pairs)
    for (int c = 0; c < K; c++)
      if (c != a && c != b && !halves[c].empty()) moves.push_back({a, b, c, halves[c], ratio[c]});
  return moves;
}

// The order in which moves are tried: uniform at random, or (--search-order=split) at random with weights, by
// sorting on -log(u)/weight (Efraimidis & Spirakis); moves that split a component falling into two groups first.
inline void order_moves(std::vector<Move>& moves, std::mt19937_64& rng, const SearchSettings& ss) {
  if (!ss.order_split) {
    std::shuffle(moves.begin(), moves.end(), rng);
    return;
  }
  std::uniform_real_distribution<double> unif(1e-300, 1.0);
  std::vector<std::pair<double, size_t>> key(moves.size());
  for (size_t t = 0; t < moves.size(); t++) key[t] = {-std::log(unif(rng)) / std::max(moves[t].weight, 1e-6), t};
  std::sort(key.begin(), key.end());
  std::vector<Move> out;
  out.reserve(moves.size());
  for (auto& kv : key) out.push_back(std::move(moves[kv.second]));
  moves.swap(out);
}

struct SearchResult {
  double loglik = 0;   // after polishing
  int iterations = 0;  // of the polishing fit
  int rounds = 0, refits = 0, accepted = 0;
  double used = 0;     // budget used, in standard runs
};

// The search from x (a fitted solution with log-likelihood ll, at tolerance ss.tol). On return x is the best
// solution, polished to -C. fs: the settings of a standard run (warm-up, seed); the
// refits use its warm-up.
template <class Data>
SearchResult run_search(const Data& D, Model<Data>& m, Vec& x, double ll, const FitSettings& fs,
                        const SearchSettings& ss, double used, const std::string& table) {
  const int K = m.K, N = D.N;
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
  order_moves(moves, rng, ss);
  size_t next = 0;
  int idle = 0;  // rounds in a row without an accepted move
  while (next < moves.size() && res.used + ss.per_round * ss.refit_cost <= ss.budget + 1e-9) {
    if (ss.patience > 0 && idle >= ss.patience) {
      say("Search: %d rounds without an accepted move; stopping\n", idle);
      break;
    }
    res.rounds++;
    const size_t end = std::min(moves.size(), next + ss.per_round);
    const int nm = end - next;
    // the moved starting points, P drawn at random in the order of the moves
    std::vector<Vec> zs(nm);
    std::vector<unsigned long> seeds(nm);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int u = 0; u < nm; u++) {
      const Move& mv = moves[next + u];
      Vec& z = zs[u];
      z = x;
      double* Q = z.data() + nP;
      for (int i = 0; i < N; i++) {
        double* q = Q + (size_t)i * K;
        q[mv.a] += q[mv.b], q[mv.b] = 0;
      }
      for (int i : mv.half) {
        double* q = Q + (size_t)i * K;
        q[mv.b] = q[mv.c], q[mv.c] = 0;
      }
      for (size_t t = 0; t < nP; t++) z[t] = 0.05 + 0.9 * unif(rng);
      seeds[u] = fs.seed + 7919 * (res.refits + u + 1);
    }
    std::vector<FitResult> rs(nm);
    // one refit: P by EM steps with Q held fixed, then warm-up and quasi-Newton (stopped early, abort_frac)
    auto refit = [&](Model<Data>& mm, int u) {
      Vec& z = zs[u];
      Vec y(z.size());
      mm.project(z.data());
      std::fill(mm.qfix.begin(), mm.qfix.end(), 1);
      for (int e = 0; e < ss.p_em; e++) {
        mm.em(z.data(), y.data());
        z.swap(y);
      }
      std::fill(mm.qfix.begin(), mm.qfix.end(), 0);
      FitSettings fr = fs;
      fr.prime = 0, fr.tol = ss.tol, fr.seed = seeds[u];
      Fitter<Model<Data>> fit(mm, fr);
      fit.verbose = false;
      if (ss.abort_frac > 0) {
        const double thr = ss.abort_frac * std::fabs(ll);
        fit.after_warmup = [&, thr](const Vec& v, double lv) {
          return lv < ll - thr || q_distance(v.data() + nP, x.data() + nP, N, K).max_abs < 0.01;
        };
      }
      rs[u] = fit.run(z);
    };
    for (int u = 0; u < nm; u++) refit(m, u);
    double best_ll = -INFINITY;
    int best_u = -1;
    std::vector<std::string> rows;  // table rows of this round (the accepted flag is known at its end)
    for (int u = 0; u < nm; u++) {
      const Move& mv = moves[next + u];
      const FitResult& r = rs[u];
      res.refits++, res.used += r.stopped ? ss.abort_cost : ss.refit_cost;
      say("Search round %d: merge %d into %d, split %d (%zu individuals moved): loglik %f (%+.2f), %d iterations, "
          "%.1f sec%s\n", res.rounds, mv.b + 1, mv.a + 1, mv.c + 1, mv.half.size(), r.loglik, r.loglik - ll,
          r.iterations, r.seconds, r.stopped ? "; stopped after the warm-up" : "");
      char row[160];
      std::snprintf(row, sizeof row, "%d\t%d\t%d\t%d\t%zu\t%.6f\t%.6f", res.rounds, mv.a + 1, mv.b + 1, mv.c + 1,
                    mv.half.size(), r.loglik, r.loglik - ll);
      rows.push_back(row);
      if (!r.stopped && r.loglik > best_ll) best_ll = r.loglik, best_u = u;
    }
    next = end;
    const bool accept = best_u >= 0 && best_ll > ll + 0.01;
    if (tf)
      for (size_t u = 0; u < rows.size(); u++) std::fprintf(tf, "%s\t%d\n", rows[u].c_str(), accept && (int)u == best_u);
    if (accept) {
      say("Search round %d: accepted, loglik %f -> %f\n", res.rounds, ll, best_ll);
      ll = best_ll, x = zs[best_u];
      res.accepted++, idle = 0;
      moves = make_moves(D, m, x, ss);
      order_moves(moves, rng, ss);
      next = 0;
    } else {
      idle++;
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

// col[a] = the column of B (N x K) matched to column a of A: the assignment with the least total squared
// difference (as q_distance in multistart.hpp).
template <class TB>
std::vector<int> match_columns(const double* A, const TB* B, int N, int K) {
  std::vector<double> cost((size_t)K * K, 0.0);
  for (int i = 0; i < N; i++)
    for (int a = 0; a < K; a++)
      for (int b = 0; b < K; b++) {
        const double d = A[(size_t)i * K + a] - (double)B[(size_t)i * K + b];
        cost[(size_t)a * K + b] += d * d;
      }
  return hungarian(cost, K);
}

struct FlipResult {
  double loglik;  // at flip_tol
  int flips = 0, kept = 0;
};

// Cross-flips (--search with --conv): moves single individuals of solution x into their state in another solution
// Qo (N x K, columns matched to x's), one at a time, most different first. A flip continues from x's P to
// ss.flip_tol and is kept if the log-likelihood rises by more than ss.flip_gain; after a kept flip the differences
// are recomputed. Individuals with in-between ancestry can sit stably in one of two states, and the state a run
// needs is often found only by another run (on the reindeer data, flips with states from a run's own refits
// rarely helped). Returns the log-likelihood of x (at flip_tol).
template <class Data>
FlipResult cross_flip(Model<Data>& m, Vec& x, const std::vector<float>& Qo, const FitSettings& fs,
                      const SearchSettings& ss, const char* what) {
  const int K = m.K, N = m.D.N;
  const size_t nP = m.nP;
  FitSettings ff = fs;
  ff.prime = 0, ff.minibatch = 0, ff.tol = ss.flip_tol;
  FlipResult res;
  {
    Fitter<Model<Data>> fit(m, ff);
    fit.verbose = false;
    res.loglik = fit.run(x).loglik;
  }
  std::vector<char> tried(N, 0);
  for (bool more = true; more;) {
    more = false;
    const std::vector<int> col = match_columns(x.data() + nP, Qo.data(), N, K);
    std::vector<double> d(N, 0.0);
    std::vector<int> cand;
    for (int i = 0; i < N; i++) {
      for (int a = 0; a < K; a++)
        d[i] = std::max(d[i], std::fabs(x[nP + (size_t)i * K + a] - (double)Qo[(size_t)i * K + col[a]]));
      if (d[i] > 0.01 && !tried[i]) cand.push_back(i);
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int u, int v) { return d[u] > d[v]; });
    for (int i : cand) {
      tried[i] = 1;
      Vec z = x;
      for (int a = 0; a < K; a++) z[nP + (size_t)i * K + a] = Qo[(size_t)i * K + col[a]];
      m.project(z.data());
      Fitter<Model<Data>> fit(m, ff);
      fit.verbose = false;
      const FitResult r = fit.run(z);
      res.flips++;
      const bool keep = r.loglik > res.loglik + ss.flip_gain;
      if (keep) {
        say("Cross-flip %s: individual %d (max |dQ| %.3f): loglik %f -> %f\n", what, i + 1, d[i], res.loglik, r.loglik);
        res.loglik = r.loglik, x = z;
        res.kept++;
        more = true;
        break;
      }
    }
  }
  return res;
}
