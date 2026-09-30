// Unit tests of admixer's model code (run by tests/run_tests.sh; exit code = number of failures).
//  1. GL per-entry derivatives: d = dlogL/dh and the exact curvature against finite differences.
//  2. For both data types: the assembled gradient against finite differences of loglik; EM never
//     decreases the log-likelihood; block relaxation + projection keep x feasible.
//  3. Consistency: GLs that are certain (one-hot) give the genotype model's gradient, and a log-likelihood
//     that differs only by log 2 per heterozygote (the binomial coefficient).
//     The GL evalAdmix estimator on certain GLs reproduces the genotype estimator.
//  4. The beagle number parser and the Q column matching.
#include <cmath>
#include <cstdio>
#include <random>

#include "../src/beagle.hpp"
#include "../src/evaladmix.hpp"
#include "../src/io.hpp"
#include "../src/model.hpp"
#include "../src/multistart.hpp"

static int fails = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("  %-58s %s (%.2e)\n", what, ok ? "ok" : "FAILED", v);
  fails += !ok;
}

template <class Data>
static void model_checks(const Data& D, const char* name, std::mt19937_64& rng) {
  std::printf("%s model\n", name);
  std::uniform_real_distribution<double> U(0, 1);
  const int K = 3;
  Model<Data> m(D, K);
  m.set_bound(1e-6);
  std::vector<double> x(m.size()), g(m.size()), y(m.size());
  for (size_t t = 0; t < m.nP; t++) x[t] = 0.1 + 0.8 * U(rng);
  for (int i = 0; i < D.N; i++) {
    double s = 0;
    for (int k = 0; k < K; k++) s += (x[m.nP + i * K + k] = 0.2 + U(rng));
    for (int k = 0; k < K; k++) x[m.nP + i * K + k] /= s;
  }
  m.gradient(x.data(), g.data());
  double maxg = 0;
  for (size_t t = 0; t < x.size(); t++) {
    const double e = 1e-6, x0 = x[t];
    x[t] = x0 + e;
    const double lp = m.loglik(x.data());
    x[t] = x0 - e;
    const double lm = m.loglik(x.data());
    x[t] = x0;
    maxg = std::max(maxg, std::fabs(g[t] - (lp - lm) / (2 * e)) / (1 + std::fabs(g[t])));
  }
  check(maxg < 1e-5, "gradient = finite differences of loglik", maxg);
  double prev = m.loglik(x.data()), worst = 0;
  for (int it = 0; it < 30; it++) {
    m.em(x.data(), y.data());
    x.swap(y);
    const double ll = m.loglik(x.data());
    worst = std::max(worst, prev - ll);
    prev = ll;
  }
  check(worst < 1e-8, "EM never decreases the log-likelihood", worst);
  m.block_relax(x.data(), y.data());
  double infeas = 0;
  for (size_t t = 0; t < m.nP; t++) infeas = std::max(infeas, std::max(m.PMIN - y[t], y[t] - m.PMAX));
  for (int i = 0; i < D.N; i++) {
    double s = 0;
    for (int k = 0; k < K; k++) s += y[m.nP + i * K + k], infeas = std::max(infeas, m.QMIN - y[m.nP + i * K + k]);
    infeas = std::max(infeas, std::fabs(s - 1));
  }
  check(infeas < 1e-9, "block relaxation step is feasible", std::max(infeas, 0.0));
}

int main() {
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(0, 1);
  omp_set_num_threads(3);

  // 1. per-entry GL derivatives
  std::printf("GL per-entry derivatives\n");
  {
    GLData one;
    one.M = one.N = 1;
    one.gl.resize(3);
    one.keep.assign(1, 1);
    double md = 0, mw = 0;
    for (int t = 0; t < 5000; t++) {
      for (int k = 0; k < 3; k++) one.gl[k] = (float)U(rng);
      if (t % 3 == 0) one.gl[t % 2] *= 1e-3f;
      const double h = 0.01 + 0.98 * U(rng), e = 1e-5;
      auto ll = [&](double hh) { return std::log(GLData::lik(one.gl.data(), hh)); };
      double T = h, W = 0;
      one.exact_hess = true;
      one.tile_wd(0, 1, 0, 1, &T, &W, 1e-9, 1 - 1e-9);
      const double fd1 = (ll(h + e) - ll(h - e)) / (2 * e), fd2 = -(ll(h + e) - 2 * ll(h) + ll(h - e)) / (e * e);
      md = std::max(md, std::fabs(T - fd1) / (1 + std::fabs(fd1)));
      if (fd2 > 0) mw = std::max(mw, std::fabs(W - fd2) / (1 + fd2));  // W is clamped at 0
    }
    check(md < 1e-6, "d = dlogL/dh", md);
    check(mw < 1e-3, "exact curvature = -d2logL/dh2", mw);
  }

  // 2. both models on small random data
  const int M = 60, N = 25;
  Genotypes G;
  G.M = M, G.N = N;
  G.G.resize((size_t)M * N);
  for (auto& g : G.G) g = U(rng) < 0.05 ? 3 : (uint8_t)(3 * U(rng));
  G.count_observed();
  GLData L;
  L.M = M, L.N = N;
  L.gl.resize((size_t)M * N * 3);
  L.keep.resize((size_t)M * N);
  for (size_t u = 0; u < L.keep.size(); u++) {
    for (int k = 0; k < 3; k++) L.gl[3 * u + k] = (float)U(rng);
    L.keep[u] = U(rng) > 0.1;
  }
  L.nobs.assign(N, 0);
  for (size_t u = 0; u < L.keep.size(); u++) L.nobs[u % N] += L.keep[u];
  model_checks(G, "genotype", rng);
  L.exact_hess = true;
  model_checks(L, "genotype likelihood", rng);

  // 3. certain GLs = genotypes
  std::printf("certain GLs vs genotypes\n");
  {
    GLData C;
    C.M = M, C.N = N;
    C.gl.assign((size_t)M * N * 3, 0.f);
    C.keep.assign((size_t)M * N, 1);
    int nhet = 0;
    for (size_t u = 0; u < C.keep.size(); u++) {
      if (G.G[u] == 3) C.keep[u] = 0, C.gl[3 * u] = C.gl[3 * u + 1] = C.gl[3 * u + 2] = 1;
      else C.gl[3 * u + G.G[u]] = 1, nhet += G.G[u] == 1;
    }
    C.nobs = G.nobs;
    Model<Genotypes> mg(G, 3);
    Model<GLData> ml(C, 3);
    std::vector<double> x(mg.size()), gg(mg.size()), gl(mg.size());
    for (size_t t = 0; t < mg.nP; t++) x[t] = 0.1 + 0.8 * U(rng);
    for (int i = 0; i < N; i++)
      for (int k = 0; k < 3; k++) x[mg.nP + i * 3 + k] = 1.0 / 3;
    const double d = ml.loglik(x.data()) - mg.loglik(x.data()) - nhet * std::log(2.0);
    check(std::fabs(d) < 1e-6, "log-likelihoods differ by log 2 per heterozygote", std::fabs(d));
    mg.gradient(x.data(), gg.data());
    ml.gradient(x.data(), gl.data());
    double mx = 0;
    for (size_t t = 0; t < gg.size(); t++) mx = std::max(mx, std::fabs(gg[t] - gl[t]) / (1 + std::fabs(gg[t])));
    check(mx < 1e-9, "gradients are equal", mx);
    // evalAdmix: certain GLs give the genotype corrected estimator, computed with the same estimate of the model
    // variance D (the GL estimator uses mean (g - 2h)^2; the genotype estimator mean g(2 - g), which estimates the
    // same 2h(1-h) and differs only by sampling noise). Data drawn from the model.
    const int M2 = 3000, N2 = 30, K2 = 3;
    std::vector<double> x2((size_t)(M2 + N2) * K2);
    for (int j = 0; j < M2 * K2; j++) x2[j] = 0.1 + 0.8 * U(rng);
    for (int i = 0; i < N2; i++) {
      double s = 0;
      for (int k = 0; k < K2; k++) s += (x2[(size_t)M2 * K2 + i * K2 + k] = std::pow(U(rng), 2));
      for (int k = 0; k < K2; k++) x2[(size_t)M2 * K2 + i * K2 + k] /= s;
    }
    Genotypes Gf;
    Gf.M = M2, Gf.N = N2;
    Gf.G.resize((size_t)M2 * N2);
    GLData Cf;
    Cf.M = M2, Cf.N = N2;
    Cf.gl.assign((size_t)M2 * N2 * 3, 0.f);
    Cf.keep.assign((size_t)M2 * N2, 1);
    for (int j = 0; j < M2; j++)
      for (int i = 0; i < N2; i++) {
        double h = 0;
        for (int k = 0; k < K2; k++) h += x2[(size_t)M2 * K2 + i * K2 + k] * x2[(size_t)j * K2 + k];
        const int g = (U(rng) < h) + (U(rng) < h);
        Gf.G[(size_t)j * N2 + i] = g;
        Cf.gl[((size_t)j * N2 + i) * 3 + g] = 1;
      }
    Gf.count_observed();
    Cf.nobs = Gf.nobs;
    // same values and the GL estimator's D = mean (g - 2h)^2, through the shared core: must be identical
    std::vector<double> dgl(N2, 0.0);
    for (int j = 0; j < M2; j++)
      for (int i = 0; i < N2; i++) {
        double h = 0;
        for (int k = 0; k < K2; k++) h += x2[(size_t)M2 * K2 + i * K2 + k] * x2[(size_t)j * K2 + k];
        const double r = Gf.G[(size_t)j * N2 + i] - 2 * h;
        dgl[i] += r * r / M2;
      }
    const auto cg = evaladmix_core(M2, N2, x2.data() + (size_t)M2 * K2, K2, dgl, 1, [&](int j0, int jn, double* X, uint8_t* O) {
      for (size_t t = 0; t < (size_t)jn * N2; t++) X[t] = Gf.G[(size_t)j0 * N2 + t], O[t] = 1;
    });
    const auto cl = evaladmix_gl(Cf, x2.data(), x2.data() + (size_t)M2 * K2, K2, 0.05, 1e-9, 1 - 1e-9, 1);
    double me = 0;
    for (size_t t = 0; t < cg.size(); t++)
      if (!std::isnan(cg[t])) me = std::max(me, std::fabs(cg[t] - cl[t]));
    check(me < 1e-9, "evalAdmix: certain GLs give the genotype estimator (same D)", me);
  }

  // 4. parser and Q matching
  std::printf("helpers\n");
  {
    const char* s[] = {"0.333333", "1.000000", "0.000081", "1e-5", "-2.5", "7", "0.1234567890123", "3.5E+2"};
    double mx = 0;
    for (const char* v : s) {
      const char* e;
      mx = std::max(mx, std::fabs(parse_num(v, &e) - std::strtod(v, nullptr)));
    }
    check(mx < 1e-12, "parse_num = strtod", mx);
    std::vector<double> A(30), B(30);
    for (int i = 0; i < 10; i++)
      for (int k = 0; k < 3; k++) A[i * 3 + k] = U(rng);
    for (int i = 0; i < 10; i++)
      for (int k = 0; k < 3; k++) B[i * 3 + k] = A[i * 3 + (k + 1) % 3];
    const double dq = q_distance(A.data(), B.data(), 10, 3).max_abs;
    check(dq == 0, "q_distance matches permuted columns", dq);
  }
  std::printf(fails ? "%d unit test(s) FAILED\n" : "all unit tests passed\n", fails);
  return fails;
}
