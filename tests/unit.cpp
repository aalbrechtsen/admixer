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
  G.resize(M, N);
  for (int j = 0; j < M; j++)
    for (int i = 0; i < N; i++) G.set(j, i, U(rng) < 0.05 ? 3 : (uint8_t)(3 * U(rng)));
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
  // pseudohaploid: one allele per entry (0/1); the same model checks, and diploid data with homozygous calls only
  // (g = 0/2) must have exactly twice the haploid log-likelihood and gradient
  Genotypes H, G02;
  H.resize(M, N), G02.resize(M, N);
  H.haploid = true;
  for (int j = 0; j < M; j++)
    for (int i = 0; i < N; i++) {
      const uint8_t a = U(rng) < 0.05 ? 3 : (uint8_t)(U(rng) < 0.4);
      H.set(j, i, a), G02.set(j, i, a == 3 ? 3 : 2 * a);
    }
  H.count_observed(), G02.count_observed();
  model_checks(H, "pseudohaploid", rng);
  {
    Model<Genotypes> mh(H, 3), md(G02, 3);
    std::vector<double> x(mh.size()), gh(mh.size()), gd(mh.size());
    for (size_t t = 0; t < mh.nP; t++) x[t] = 0.1 + 0.8 * U(rng);
    for (int i = 0; i < N; i++) {
      double s = 0;
      for (int k = 0; k < 3; k++) s += (x[mh.nP + i * 3 + k] = 0.1 + U(rng));
      for (int k = 0; k < 3; k++) x[mh.nP + i * 3 + k] /= s;
    }
    const double lh = mh.loglik(x.data()), ld = md.loglik(x.data());
    check(std::fabs(ld - 2 * lh) < 1e-9 * std::fabs(ld), "pseudohaploid: 0/2 genotypes have 2x the log-likelihood",
          std::fabs(ld - 2 * lh));
    mh.gradient(x.data(), gh.data());
    md.gradient(x.data(), gd.data());
    double mx = 0;
    for (size_t t = 0; t < gh.size(); t++) mx = std::max(mx, std::fabs(gd[t] - 2 * gh[t]) / (1 + std::fabs(gd[t])));
    check(mx < 1e-9, "pseudohaploid: 0/2 genotypes have 2x the gradient", mx);
  }
  L.exact_hess = true;
  model_checks(L, "genotype likelihood", rng);

  // reading a .bed: diploid codes and the pseudohaploid alleles; het_allele is a fair, independent coin
  {
    const int Mb = 4, Nb = 6;
    const uint8_t gt[Mb][Nb] = {{0, 1, 2, 3, 1, 2}, {1, 1, 0, 2, 3, 1}, {2, 2, 2, 1, 0, 0}, {3, 1, 1, 1, 2, 0}};
    const uint8_t enc[4] = {0, 2, 3, 1};  // genotype (copies of A2; 3 missing) -> PLINK code
    const std::string pre = "unit_tmp";  // in the working directory, removed below
    {
      std::FILE* f = std::fopen((pre + ".bed").c_str(), "wb");
      if (!f) {
        std::printf("cannot write %s.bed\n", pre.c_str());
        std::exit(1);
      }
      const uint8_t magic[3] = {0x6c, 0x1b, 0x01};
      std::fwrite(magic, 1, 3, f);
      for (int j = 0; j < Mb; j++)
        for (int b = 0; b < (Nb + 3) / 4; b++) {
          uint8_t c = 0;
          for (int t = 0; t < 4 && 4 * b + t < Nb; t++) c |= enc[gt[j][4 * b + t]] << (2 * t);
          std::fputc(c, f);
        }
      std::fclose(f);
      std::FILE* bim = std::fopen((pre + ".bim").c_str(), "w");
      for (int j = 0; j < Mb; j++) std::fprintf(bim, "1 s%d 0 %d A C\n", j + 1, j + 1);
      std::fclose(bim);
      std::FILE* fam = std::fopen((pre + ".fam").c_str(), "w");
      for (int i = 0; i < Nb; i++) std::fprintf(fam, "f i%d 0 0 0 -9\n", i);
      std::fclose(fam);
    }
    const Genotypes dg = read_bed(pre + ".bed");
    size_t nhet = 0;
    const Genotypes hg = read_bed(pre + ".bed", true, &nhet);
    int bad_d = 0, bad_h = 0, het = 0;
    for (int j = 0; j < Mb; j++)
      for (int i = 0; i < Nb; i++) {
        const uint8_t g = gt[j][i];
        bad_d += dg.get(j, i) != g;
        const uint8_t want = g == 3 ? 3 : g == 1 ? het_allele(j + 1, i + 1) : g / 2;
        bad_h += hg.get(j, i) != want;
        het += g == 1;
      }
    check(bad_d == 0, "read_bed: diploid genotypes", bad_d);
    check(bad_h == 0 && hg.haploid, "read_bed: pseudohaploid alleles (het: het_allele)", bad_h);
    check(nhet == (size_t)het, "read_bed: heterozygous calls counted", std::fabs((double)nhet - het));
    for (const char* e : {".bed", ".bim", ".fam"}) std::remove((pre + e).c_str());
    // het_allele: about half A2 overall, and no pattern shared between neighbouring individuals or SNPs
    long ones = 0, same_i = 0, same_j = 0;
    const int S = 2000;
    for (int j = 1; j <= S; j++)
      for (int i = 1; i <= S; i++) {
        const uint8_t a = het_allele(j, i);
        ones += a, same_i += a == het_allele(j, i + 1), same_j += a == het_allele(j + 1, i);
      }
    const double n = (double)S * S;
    check(std::fabs(ones / n - 0.5) < 0.002 && std::fabs(same_i / n - 0.5) < 0.002 && std::fabs(same_j / n - 0.5) < 0.002,
          "het_allele: fair and independent between neighbours",
          std::max({std::fabs(ones / n - 0.5), std::fabs(same_i / n - 0.5), std::fabs(same_j / n - 0.5)}));
  }

  // 3. certain GLs = genotypes
  std::printf("certain GLs vs genotypes\n");
  {
    GLData C;
    C.M = M, C.N = N;
    C.gl.assign((size_t)M * N * 3, 0.f);
    C.keep.assign((size_t)M * N, 1);
    int nhet = 0;
    for (size_t u = 0; u < C.keep.size(); u++) {
      const uint8_t g = G.get(u / N, u % N);
      if (g == 3) C.keep[u] = 0, C.gl[3 * u] = C.gl[3 * u + 1] = C.gl[3 * u + 2] = 1;
      else C.gl[3 * u + g] = 1, nhet += g == 1;
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
    Gf.resize(M2, N2);
    GLData Cf;
    Cf.M = M2, Cf.N = N2;
    Cf.gl.assign((size_t)M2 * N2 * 3, 0.f);
    Cf.keep.assign((size_t)M2 * N2, 1);
    for (int j = 0; j < M2; j++)
      for (int i = 0; i < N2; i++) {
        double h = 0;
        for (int k = 0; k < K2; k++) h += x2[(size_t)M2 * K2 + i * K2 + k] * x2[(size_t)j * K2 + k];
        const int g = (U(rng) < h) + (U(rng) < h);
        Gf.set(j, i, g);
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
        const double r = Gf.get(j, i) - 2 * h;
        dgl[i] += r * r / M2;
      }
    const auto cg = evaladmix_core(M2, N2, x2.data() + (size_t)M2 * K2, K2, dgl, 1, [&](int j0, int jn, double* X, uint8_t* O) {
      for (size_t t = 0; t < (size_t)jn * N2; t++) X[t] = Gf.get(j0 + t / N2, t % N2), O[t] = 1;
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
