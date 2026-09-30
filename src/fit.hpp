// admixer: fitting. Near-uniform random start -> 5 EM steps -> mini-batch block-relaxation warm-up ->
// block relaxation with quasi-Newton acceleration (ADMIXTURE's algorithm) until the log-likelihood
// improves by less than tol per iteration.
#pragma once
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "log.hpp"
#include "model.hpp"

using Vec = std::vector<double>;

struct FitSettings {
  double tol = 1e-4;     // -C: stop when the log-likelihood improves by less than this per iteration
  int max_iter = 10000;  // quasi-Newton iterations at most
  int qn_secants = 3;    // quasi-Newton secant pairs (as ADMIXTURE)
  int prime = 5;         // EM steps before the main algorithm (as ADMIXTURE)
  int minibatch = 32;    // initial number of SNP mini-batches in the warm-up (0 = no warm-up)
  int mb_max_epochs = 100;
  unsigned long seed = 43;
};

struct FitResult {
  double loglik = 0;
  int iterations = 0;
  double seconds = 0;
};

class Fitter {
 public:
  Fitter(Model& m, const FitSettings& s) : m_(m), s_(s) {}
  bool verbose = true;  // print every iteration (false: silent; the caller prints a summary)

  // Random start: P uniform in [0.05, 0.95]; each row of Q near uniform, 1/K with 1% relative noise. A sparse
  // start (most Q entries at the lower bound) assigns individuals to random ancestries and traps many runs in
  // local optima. Rows of Q marked fixed and, in projection mode, P are left as given.
  void init_random(Vec& x, std::mt19937_64& rng) const {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const int K = m_.K;
    if (!m_.pfix)
      for (size_t t = 0; t < m_.nP; t++) x[t] = 0.05 + 0.9 * unif(rng);
    for (int i = 0; i < m_.D.N; i++) {
      if (m_.fixed(i)) continue;
      double* q = x.data() + m_.nP + (size_t)i * K;
      double sum = 0;
      for (int k = 0; k < K; k++) sum += (q[k] = 1 + 0.01 * unif(rng));
      for (int k = 0; k < K; k++) q[k] /= sum;
    }
    m_.project(x.data());
  }

  FitResult run(Vec& x) {
    const double t0 = omp_get_wtime();
    Vec y(x.size());
    if (verbose) say("Performing five EM steps to prime main algorithm\n");
    double prev = -INFINITY;
    for (int it = 1; it <= s_.prime; it++) {
      const double ll = m_.em(x.data(), y.data());
      x.swap(y);
      log_line(it, "EM", ll, ll - prev, t0);
      prev = ll;
    }
    double ll = m_.loglik(x.data());
    if (verbose) say("Initial loglikelihood: %f\n", ll);
    if (s_.minibatch > 1 && !m_.pfix) warmup(x, t0);
    if (verbose) say("Starting main algorithm\n");
    FitResult r = qn(x, t0);
    r.seconds = omp_get_wtime() - t0;
    return r;
  }

 private:
  Model& m_;
  FitSettings s_;

  void log_line(int it, const char* what, double ll, double delta, double t0) const {
    if (verbose) say("%d (%s) \tElapsed: %.3f\tLoglikelihood: %.6f\t(delta): %g\n", it, what, omp_get_wtime() - t0, ll,
                delta);
  }
  static double dot(const Vec& a, const Vec& b) {
    double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (size_t t = 0; t < a.size(); t++) s += a[t] * b[t];
    return s;
  }

  // Mini-batch warm-up (SNP order shuffled by the caller): each epoch visits B contiguous SNP batches
  // in random order; for each batch, one Newton/QP step for the batch's rows of P (exact: rows of P
  // are independent given Q), then one for Q using the batch's SNPs only. B is halved whenever an
  // epoch does not improve the full log-likelihood; the full-data algorithm takes over at B = 1.
  void warmup(Vec& x, double t0) {
    const int M = m_.D.M, K = m_.K;
    const size_t nP = m_.nP, nQ = m_.nQ;
    int B = std::min(s_.minibatch, std::max(1, M / 500));  // at least ~500 SNPs per batch
    if (B < 2) return;
    Vec y(x.size());
    std::mt19937_64 rng(s_.seed);
    std::vector<int> order;
    double prev = m_.loglik(x.data());
    for (int epoch = 1; B > 1 && epoch <= s_.mb_max_epochs; epoch++) {
      order.resize(B);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      for (int b : order) {
        const int ja = (int)((long)M * b / B), jb = (int)((long)M * (b + 1) / B);
        m_.sqp_P(x.data() + nP, x.data(), y.data(), ja, jb);
        std::copy(y.begin() + (size_t)ja * K, y.begin() + (size_t)jb * K, x.begin() + (size_t)ja * K);
        m_.sqp_Q(x.data(), x.data() + nP, y.data() + nP, ja, jb);
        std::copy(y.begin() + nP, y.begin() + nP + nQ, x.begin() + nP);
      }
      const double ll = m_.loglik(x.data());
      char what[48];
      std::snprintf(what, sizeof what, "mini-batch, %d batches", B);
      log_line(epoch, what, ll, ll - prev, t0);
      if (ll <= prev) B /= 2;
      prev = ll;
    }
  }

  // Quasi-Newton acceleration of the block-relaxation map F (Zhou, Alexander & Lange 2011):
  //   x_new = F(x) + V (U'U - U'V)^{-1} U'u,  u = F(x) - x,
  // with U, V the last q secant pairs. The extrapolated point is projected back onto the constraints
  // and accepted only if it beats F(x); otherwise F(F(x)) is used.
  FitResult qn(Vec& x, double t0) {
    const size_t n = x.size();
    const int q = s_.qn_secants;
    std::vector<Vec> U, V;
    Vec x1(n), x2(n), xq(n);
    double prev = -INFINITY, ll = -INFINITY;
    int it;
    for (it = 1; it <= s_.max_iter; it++) {
      m_.block_relax(x.data(), x1.data());
      const double ll1 = m_.block_relax(x1.data(), x2.data());
      Vec u(n), v(n);
      for (size_t t = 0; t < n; t++) u[t] = x1[t] - x[t], v[t] = x2[t] - x1[t];
      if ((int)U.size() == q) U.erase(U.begin()), V.erase(V.begin());
      U.push_back(u);
      V.push_back(v);
      const int h = U.size();
      std::vector<double> A(h * h), c(h);
      for (int a = 0; a < h; a++) {
        c[a] = dot(U[a], U.back());
        for (int b = 0; b < h; b++) A[a * h + b] = dot(U[a], U[b]) - dot(U[a], V[b]);
      }
      bool accepted = false;
      double llq = -INFINITY;
      if (solve_linear(h, A.data(), c.data())) {
        xq = x1;
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < n; t++)
          for (int a = 0; a < h; a++) xq[t] += V[a][t] * c[a];
        m_.project(xq.data());
        m_.restore_fixed(xq.data(), x1.data());
        llq = m_.loglik(xq.data());
        accepted = llq > ll1;
      }
      if (accepted) {
        x.swap(xq);
        ll = llq;
      } else {
        x.swap(x2);
        ll = m_.loglik(x.data());
      }
      log_line(it, "QN/Block", ll, ll - prev, t0);
      if (std::fabs(ll - prev) < s_.tol) break;
      prev = ll;
    }
    FitResult r;
    r.loglik = ll;
    r.iterations = std::min(it, s_.max_iter);
    return r;
  }
};
