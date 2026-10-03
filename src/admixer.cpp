// admixer: maximum-likelihood ancestry estimation under the ADMIXTURE model from called genotypes (PLINK
// .bed, with ADMIXTURE's command line, input and output) or under the NGSadmix model from genotype
// likelihoods (beagle files). Algorithm: block relaxation with Newton/QP steps and quasi-Newton
// acceleration (as ADMIXTURE), evaluated with BLAS, from a random start; for called genotypes after 5 EM
// steps and a mini-batch warm-up.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "beagle.hpp"
#include "evaladmix.hpp"
#include "fit.hpp"
#include "io.hpp"
#include "log.hpp"
#include "model.hpp"
#include "multistart.hpp"

static const char* VERSION = "0.2.8";

static void usage(int code) {
  std::printf(
      "admixer %s: ancestry estimation under the ADMIXTURE model (called genotypes) or the NGSadmix model\n"
      "(genotype likelihoods)\n\n"
      "usage: admixer [options] inputFile K\n\n"
      "  inputFile  a PLINK .bed file (with .bim and .fam next to it), or a beagle genotype likelihood file\n"
      "             (.beagle.gz, .beagle or any other .gz file, as written by ANGSD and read by NGSadmix)\n"
      "  K          number of ancestral populations\n\n"
      "Output (working directory): inputBasename.K.Q, inputBasename.K.P.gz, inputBasename.K.corres.txt and\n"
      "inputBasename.K.log\n\n"
      "options:\n"
      "  -jX, -j X           use X threads (default 8)\n"
      "  --seed=X, -s X      random seed (default 43; 'time' uses the clock)\n"
      "  -C=X, -C X          stop when the log-likelihood improves by less than X (default 1e-4)\n"
      "  -o NAME, --out=NAME output prefix NAME instead of inputBasename\n"
      "  --no-gzip           write NAME.K.P uncompressed instead of NAME.K.P.gz\n"
      "  --no-P              do not write the P matrix (e.g. for benchmarks)\n"
      "  --supervised        supervised analysis: reads inputBasename.pop (one line per individual,\n"
      "                      a population name, or '-' if unknown); labelled individuals are held at\n"
      "                      their population; K must equal the number of population names\n"
      "  -P                  projection: reads inputBasename.K.P.in (or .P.in.gz) and estimates Q with P fixed\n"
      "  --no-evaladmix      do not write NAME.K.corres.txt, the evalAdmix correlation of residuals\n"
      "                      (corrected estimator of van Waaij et al. 2023, for GLs on the posterior expected\n"
      "                      genotypes; plot with evalAdmix's visFuns.R). It needs memory ~ 48 N^2 bytes, so it\n"
      "                      is skipped for N > 20000 individuals unless --evaladmix is given\n"
      "  --conv X            convergence test with several starts (seeds seed, seed+1, ...): stop when X runs\n"
      "                      agree with the best run (highest likelihood); writes the best run and NAME.K.conv\n"
      "  -m X, --max_runs=X  with --conv: at most X runs (default 10)\n"
      "  --conv_thres=X      with --conv: runs agree if the largest difference in any Q entry, after matching\n"
      "                      ancestries, is below X (default 0.01)\n"
      "genotype likelihood (beagle) input, defaults as NGSadmix:\n"
      "  --minMaf=X          keep sites with X < MAF < 1-X (default 0.05; 0 = keep all)\n"
      "  --misTol=X          GLs with max - min < X count as missing (default 0.05)\n"
      "  --minInd=X          keep sites with more than X individuals with data (default 0 = off)\n"
      "  --keep-missing      use the missing GL entries as data (as NGSadmix's EM) instead of leaving them out\n"
      "advanced:\n"
      "  --bound=X           P in [X, 1-X], Q >= X (default 1e-5 for genotypes as ADMIXTURE, 1e-9 for GLs as NGSadmix)\n"
      "  --prime=X           EM steps before the main algorithm (default 5 for genotypes, 0 for GLs)\n"
      "  --minibatch=X       initial number of mini-batches of the warm-up (default 32 for genotypes, 0 = off for GLs)\n"
      "  --hess=exact|em     GLs: curvature of the Newton steps (default exact; em = ADMIXTURE's EM-type weight)\n"
      "  --hess-float=X      Newton Hessians from single-precision matrix products in the quasi-Newton phase:\n"
      "                      1 on, 0 off (exact double precision), auto (default): on for called genotypes\n"
      "                      with K >= 8, off for genotype likelihoods\n"
      "  --smallk=X          1 (default): register-blocked kernels for K = 2-7 in the Newton steps; 0: OpenBLAS\n"
      "  --qn-damp=X         retry a rejected quasi-Newton extrapolation up to X times with the step scaled by\n"
      "                      --qn-damp-factor (default X = 2, factor 0.5; X = 0: ADMIXTURE's rule, F(F(x)) at once)\n"
      "  -h, --help          this help\n",
      VERSION);
  std::exit(code);
}

// Option value in "-C=X", "-CX" or "-C X" form; advances a for the last form.
static std::string opt_value(int argc, char** argv, int& a, const std::string& flag) {
  std::string o = argv[a];
  if (o.size() > flag.size()) return o[flag.size()] == '=' ? o.substr(flag.size() + 1) : o.substr(flag.size());
  if (a + 1 >= argc) usage(1);
  return argv[++a];
}

struct Options {
  FitSettings fs;
  GLFilter glf;
  int threads = 8, K = 0, conv = 0, max_runs = 10;
  double conv_thres = 0.01, bound = -1;  // bound < 0: the data type's default
  int prime = -1, minibatch = -1;        // < 0: the data type's default
  std::string input, prefix, out, hess = "exact";
  bool supervised = false, projection = false, evaladmix = true, evaladmix_forced = false, gz = true, write_P = true;
  bool smallk = true;
  std::string thread_note;  // why -j was capped, or that it exceeds the physical cores (printed in the log)  // --smallk: register-blocked kernels for small K instead of OpenBLAS in the Newton steps
};

static bool is_beagle(const std::string& f) { return !ends_with(f, ".bed"); }

// The number of threads OpenBLAS was compiled for (MAX_THREADS in its configuration string; 0 if unknown).
// The pthread builds of OpenBLAS 0.3.20 shipped with Ubuntu (MAX_THREADS=64) crash when more OpenMP threads than
// that call it at the same time, even single-threaded; the static release build uses NUM_THREADS=256.
static int openblas_max_threads() {
  const char* c = openblas_get_config();
  const char* p = c ? std::strstr(c, "MAX_THREADS=") : nullptr;
  return p ? std::atoi(p + 12) : 0;
}
// Physical cores of the machine (distinct package/core ids in /sys; 0 if unknown).
static int physical_cores() {
  std::set<std::pair<int, int>> cores;
  for (int c = 0;; c++) {  // cpu0, cpu1, ... until the first missing one
    const std::string d = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
    std::ifstream fp(d + "physical_package_id"), fc(d + "core_id");
    int pk, co;
    if (!(fp >> pk) || !(fc >> co)) break;
    cores.insert({pk, co});
  }
  return (int)cores.size();
}
// input path without its extension: data.bed -> data, data.beagle.gz -> data, data.gz -> data
static std::string input_prefix(const std::string& f) {
  std::string p = f;
  for (const char* e : {".gz", ".beagle", ".bed"})
    if (ends_with(p, e)) p = p.substr(0, p.size() - std::string(e).size());
  return p;
}

// Everything after reading the data; Data = Genotypes (PLINK) or GLData (beagle).
template <class Data>
static void analyse(Data& D, Options& o, double t0) {
  const bool gl = std::is_same_v<Data, GLData>;
  const int K = o.K;
  FitSettings& fs = o.fs;
  say("Random seed: %lu\n", fs.seed);
  if (gl)
    say("Point estimation method: Block relaxation algorithm (Newton/QP steps with the %s curvature of the GL "
        "likelihood, BLAS kernels)\n", o.hess == "em" ? "EM-type" : "exact");
  else
    say("Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)\n");
  say("Convergence acceleration algorithm: QuasiNewton, %d secant conditions", fs.qn_secants);
  if (fs.qn_damp > 0) say(", rejected extrapolations retried %dx with the step scaled by %g", fs.qn_damp, fs.qn_damp_factor);
  say("\n");
  if (fs.hess_float > 0)
    say("Newton Hessians of the main algorithm: single-precision matrix products (--hess-float=0: double)%s\n",
        gl ? "; not recommended for genotype likelihoods" : "");
  if (o.smallk && K >= 2 && K <= Model<Data>::SMALLK_MAX && smallk::available())
    say("Newton steps: register-blocked kernels for K = %d instead of OpenBLAS (--smallk=0: OpenBLAS)\n", K);
  say("Point estimation will terminate when objective function delta < %g\n", fs.tol);
  say(gl ? "Size of GL data: %dx%d\n" : "Size of G: %dx%d\n", D.N, D.M);
  say("Threads: %d\n", o.threads);
  if (!o.thread_note.empty()) say("%s", o.thread_note.c_str());

  // Supervised mode: population labels from inputBasename.pop, columns in order of first appearance
  std::vector<int> label(D.N, -1);
  if (o.supervised) {
    const std::string fn = o.prefix + ".pop";
    std::ifstream in(fn);
    if (!in) throw std::runtime_error("supervised mode needs " + fn);
    std::map<std::string, int> id;
    std::string s;
    int i = 0;
    while (in >> s) {
      if (i >= D.N) throw std::runtime_error(fn + " has more lines than individuals");
      if (s != "-") {
        auto it = id.find(s);
        if (it == id.end()) it = id.emplace(s, (int)id.size()).first;
        label[i] = it->second;
      }
      i++;
    }
    if (i != D.N) throw std::runtime_error(fn + " has fewer lines than individuals");
    if ((int)id.size() != K)
      throw std::runtime_error(fn + " names " + std::to_string(id.size()) + " populations but K = " + std::to_string(K));
    say("Supervised analysis mode: %d labelled individuals\n",
        (int)std::count_if(label.begin(), label.end(), [](int l) { return l >= 0; }));
  }

  // The mini-batch warm-up uses contiguous SNP batches: shuffle the SNP order once (undone on output).
  std::vector<int> perm(D.M);
  std::iota(perm.begin(), perm.end(), 0);
  if (!o.projection && fs.minibatch > 1) {
    std::mt19937_64 prng(fs.seed + 1);
    std::shuffle(perm.begin(), perm.end(), prng);
    D.permute(perm);
  }

  Model<Data> m(D, K);
  m.use_smallk_kernels = o.smallk;
  m.set_bound(o.bound);
  Vec x0(m.size());  // fixed parts (projection P, supervised rows) shared by all runs
  if (o.projection) {
    std::string fn = o.prefix + "." + std::to_string(K) + ".P.in";
    if (!std::ifstream(fn) && std::ifstream(fn + ".gz")) fn += ".gz";
    read_matrix(fn, x0.data(), D.M, K);
    m.pfix = true;
    say("Projection mode: P read from %s and held fixed\n", fn.c_str());
  }
  if (o.supervised) {
    for (int i = 0; i < D.N; i++)
      if (label[i] >= 0) {
        double* q = x0.data() + m.nP + (size_t)i * K;
        for (int k = 0; k < K; k++) q[k] = k == label[i];
        m.qfix[i] = 1;
      }
  }
  // One run per seed: seed, seed+1, ... (a single run without --conv).
  const int nruns = o.conv > 0 ? o.max_runs : 1;
  if (o.conv > 0)
    say("Convergence test: up to %d runs (seeds %lu-%lu); converged when %d runs agree with the best run "
        "(max |dQ| < %g)\n", nruns, fs.seed, fs.seed + nruns - 1, o.conv, o.conv_thres);
  struct Run {
    unsigned long seed;
    double loglik, seconds;
    int iters;
    std::vector<float> Q;
    QDistance d;
    bool agrees = false;
  };
  std::vector<Run> runs;
  Vec x, best_x;
  int best = -1, agreeing = 0;
  for (int r = 0; r < nruns; r++) {
    FitSettings fr = fs;
    fr.seed = fs.seed + r;
    x = x0;
    std::mt19937_64 rng(fr.seed);
    Fitter<Model<Data>> fit(m, fr);
    fit.verbose = r == 0;  // iteration details for the first run only
    fit.init_random(x, rng);
    if (o.supervised && !o.projection) D.supervised_start(label, K, x.data(), m.PMIN, m.PMAX);
    const FitResult res = fit.run(x);
    Run run{fr.seed, res.loglik, res.seconds, res.iterations,
            std::vector<float>(x.begin() + m.nP, x.end()), QDistance(), false};
    runs.push_back(std::move(run));
    if (best < 0 || res.loglik > runs[best].loglik) best = r, best_x = x;
    if (o.conv == 0) break;
    // agreement of every run with the current best run (the best can change, so recompute all)
    agreeing = 0;
    for (auto& u : runs) {
      u.d = q_distance(u.Q.data(), runs[best].Q.data(), D.N, K);
      u.agrees = u.d.max_abs < o.conv_thres;
      agreeing += u.agrees;
    }
    say("Run %d (seed %lu): loglik %f, %d iterations, %.1f sec; max |dQ| to best run %.4f; %d of %d runs agree\n",
        r + 1, fr.seed, res.loglik, res.iterations, res.seconds, runs.back().d.max_abs, agreeing, r + 1);
    if (agreeing >= o.conv) break;
  }
  x = best_x;
  const auto kkt = m.kkt(x.data());
  say("Summary: \n");
  if (o.conv > 0) {
    say("%s: %d of %d runs agree with the best run (seed %lu, max |dQ| < %g)\n",
        agreeing >= o.conv ? "Converged" : "Not converged", agreeing, (int)runs.size(), runs[best].seed, o.conv_thres);
    const std::string fn = o.out + "." + std::to_string(K) + ".conv";
    FILE* fp = std::fopen(fn.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot write " + fn);
    std::fprintf(fp, "run\tseed\tloglik\tloglik_diff\tmax_abs_dQ\tmean_sum_abs_dQ\trmse_dQ\titerations\tseconds\tagrees\n");
    std::vector<size_t> order(runs.size());  // best run first, then by decreasing log-likelihood
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return runs[a].loglik > runs[b].loglik; });
    for (size_t r : order) {
      const Run& u = runs[r];
      std::fprintf(fp, "%zu\t%lu\t%.6f\t%.6f\t%.6g\t%.6g\t%.6g\t%d\t%.2f\t%d\n", r + 1, u.seed, u.loglik,
                   u.loglik - runs[best].loglik, u.d.max_abs, u.d.mean_sum, u.d.rmse, u.iters, u.seconds, (int)u.agrees);
    }
    std::fclose(fp);
    say("Convergence table written to %s\n", fn.c_str());
  }
  say("Converged in %d iterations (%.3f sec)\n", runs[best].iters, omp_get_wtime() - t0);
  say("Loglikelihood: %f\n", runs[best].loglik);
  if (gl && o.glf.skip_missing)  // the exact log-likelihood of every entry, NGSadmix's objective
    say("Loglikelihood over all GL entries, missing ones included (as NGSadmix): %f\n", m.loglik(x.data(), true));
  say("Optimality check (max projected gradient): P %.2e, Q %.2e\n", kkt.first, kkt.second);

  say("Writing output files.\n");
  const std::string pre = o.out + "." + std::to_string(K);
  write_matrix(pre + ".Q", x.data() + m.nP, D.N, K);
  if (o.write_P) {
    std::vector<double> Pout(m.nP);
    for (int j = 0; j < D.M; j++)
      std::copy(x.begin() + (size_t)j * K, x.begin() + (size_t)(j + 1) * K, Pout.begin() + (size_t)perm[j] * K);
    write_matrix(pre + (o.gz ? ".P.gz" : ".P"), Pout.data(), D.M, K, o.gz);
  }
  if (o.evaladmix && D.N > 20000 && !o.evaladmix_forced) {
    say("evalAdmix skipped: %d individuals would need ~%.0f GB of memory (use --evaladmix to force)\n", D.N,
        48.0 * D.N * D.N / 1e9);
    o.evaladmix = false;
  }
  if (o.evaladmix) {
    const double te = omp_get_wtime();
    std::vector<double> cor;
    if constexpr (std::is_same_v<Data, Genotypes>)
      cor = evaladmix_corrected(D, x.data() + m.nP, K, o.threads);
    else
      cor = evaladmix_gl(D, x.data(), x.data() + m.nP, K, o.glf.misTol, m.PMIN, m.PMAX, o.threads);
    write_corres(pre + ".corres.txt", cor, D.N);
    say("evalAdmix correlation of residuals written to %s.corres.txt (%.2f sec)\n", pre.c_str(), omp_get_wtime() - te);
  }
}

int main(int argc, char** argv) {
  Options o;
  FitSettings& fs = o.fs;
  std::string seed_s = "43";
  std::vector<std::string> pos;
  for (int a = 1; a < argc; a++) {
    const std::string s = argv[a];
    if (s == "-h" || s == "--help") usage(0);
    else if (s.rfind("--seed", 0) == 0) seed_s = opt_value(argc, argv, a, "--seed");
    else if (s.rfind("--out", 0) == 0) o.out = opt_value(argc, argv, a, "--out");
    else if (s == "--supervised") o.supervised = true;
    else if (s == "--evaladmix") o.evaladmix = o.evaladmix_forced = true;
    else if (s == "--no-evaladmix") o.evaladmix = false;
    else if (s == "--no-gzip") o.gz = false;
    else if (s == "--no-P") o.write_P = false;
    else if (s == "--keep-missing") o.glf.skip_missing = false;
    else if (s.rfind("--conv_thres", 0) == 0) o.conv_thres = std::atof(opt_value(argc, argv, a, "--conv_thres").c_str());
    else if (s.rfind("--conv", 0) == 0) o.conv = std::atoi(opt_value(argc, argv, a, "--conv").c_str());
    else if (s.rfind("--max_runs", 0) == 0) o.max_runs = std::atoi(opt_value(argc, argv, a, "--max_runs").c_str());
    else if (s.rfind("--max-iter", 0) == 0) fs.max_iter = std::atoi(opt_value(argc, argv, a, "--max-iter").c_str());
    else if (s.rfind("--minMaf", 0) == 0) o.glf.minMaf = std::atof(opt_value(argc, argv, a, "--minMaf").c_str());
    else if (s.rfind("--misTol", 0) == 0) o.glf.misTol = std::atof(opt_value(argc, argv, a, "--misTol").c_str());
    else if (s.rfind("--minInd", 0) == 0) o.glf.minInd = std::atoi(opt_value(argc, argv, a, "--minInd").c_str());
    else if (s.rfind("--bound", 0) == 0) o.bound = std::atof(opt_value(argc, argv, a, "--bound").c_str());
    else if (s.rfind("--prime", 0) == 0) o.prime = std::atoi(opt_value(argc, argv, a, "--prime").c_str());
    else if (s.rfind("--minibatch", 0) == 0) o.minibatch = std::atoi(opt_value(argc, argv, a, "--minibatch").c_str());
    else if (s.rfind("--smallk", 0) == 0) {
      const std::string v = opt_value(argc, argv, a, "--smallk");
      if (v != "0" && v != "1") usage(1);
      o.smallk = v == "1";
    } else if (s.rfind("--hess-float", 0) == 0) {
      const std::string v = opt_value(argc, argv, a, "--hess-float");
      if (v == "auto") fs.hess_float = -1;
      else if (v == "0" || v == "1") fs.hess_float = v == "1";
      else usage(1);
    }
    else if (s.rfind("--hess", 0) == 0) o.hess = opt_value(argc, argv, a, "--hess");
    else if (s.rfind("--qn-damp-factor", 0) == 0)
      fs.qn_damp_factor = std::atof(opt_value(argc, argv, a, "--qn-damp-factor").c_str());
    else if (s.rfind("--qn-damp", 0) == 0) fs.qn_damp = std::atoi(opt_value(argc, argv, a, "--qn-damp").c_str());
    else if (s.rfind("-m", 0) == 0) o.max_runs = std::atoi(opt_value(argc, argv, a, "-m").c_str());
    else if (s == "-P") o.projection = true;
    else if (s.rfind("-j", 0) == 0) o.threads = std::atoi(opt_value(argc, argv, a, "-j").c_str());
    else if (s.rfind("-s", 0) == 0) seed_s = opt_value(argc, argv, a, "-s");
    else if (s.rfind("-C", 0) == 0) fs.tol = std::atof(opt_value(argc, argv, a, "-C").c_str());
    else if (s.rfind("-o", 0) == 0) o.out = opt_value(argc, argv, a, "-o");
    else if (!s.empty() && s[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", s.c_str());
      usage(1);
    } else pos.push_back(s);
  }
  if (pos.size() != 2) usage(1);
  o.input = pos[0];
  o.K = std::atoi(pos[1].c_str());
  if (o.K < 1 || o.threads < 1 || o.conv < 0 || o.max_runs < 1 || (o.hess != "exact" && o.hess != "em") ||
      fs.qn_damp < 0 || !(fs.qn_damp_factor > 0 && fs.qn_damp_factor < 1))
    usage(1);
  fs.seed = seed_s == "time" ? (unsigned long)std::time(nullptr) : std::strtoul(seed_s.c_str(), nullptr, 10);
  const bool gl = is_beagle(o.input);
  o.prefix = gl ? input_prefix(o.input) : strip_extension(o.input);
  if (o.out.empty()) {
    const size_t slash = o.prefix.find_last_of('/');
    o.out = slash == std::string::npos ? o.prefix : o.prefix.substr(slash + 1);
  }
  // per data type defaults (genotypes: as admixer 0.1; genotype likelihoods: as the NGSadmix benchmark showed best)
  if (o.bound < 0) o.bound = gl ? 1e-9 : 1e-5;
  fs.prime = o.prime >= 0 ? o.prime : gl ? 0 : 5;
  fs.minibatch = o.minibatch >= 0 ? o.minibatch : gl ? 0 : 32;
  // single-precision Hessians: by default for called genotypes with K >= 8 only. With genotype likelihoods
  // (bounds 1e-9, so weights up to ~1e18 next to O(1) ones) the rounded Hessians led runs to other, worse
  // optima on the NGSadmix tutorial data, and the gain at small K is small.
  if (fs.hess_float < 0) fs.hess_float = !gl && o.K >= FitSettings::HESS_FLOAT_MIN_K;
  if (const int mx = openblas_max_threads(); mx > 0 && o.threads > mx) {
    o.thread_note = "Note: -j " + std::to_string(o.threads) + " reduced to " + std::to_string(mx) +
                    ", the number of threads this OpenBLAS library was built for\n";
    o.threads = mx;
  }
  if (const int pc = physical_cores(); pc > 0 && o.threads > pc)
    o.thread_note += "Note: " + std::to_string(o.threads) + " threads on " + std::to_string(pc) +
                     " physical cores; one thread per core is usually faster\n";
  omp_set_num_threads(o.threads);
  const std::string logname = o.out + "." + std::to_string(o.K) + ".log";
  log_file() = std::fopen(logname.c_str(), "w");
  if (!log_file()) std::fprintf(stderr, "Warning: cannot write log file %s\n", logname.c_str());
  std::string cmd;
  for (int a = 0; a < argc; a++) cmd += (a ? " " : "") + std::string(argv[a]);

  try {
    say("admixer %s\n", VERSION);
    say("Command: %s\n", cmd.c_str());
    const double t0 = omp_get_wtime();
    if (gl) {
      int sites_in = 0;
      GLData D = read_beagle(o.input, o.glf, &sites_in);
      D.exact_hess = o.hess == "exact";
      size_t nmis = 0;
      for (uint8_t k : D.keep) nmis += !k;
      say("Input: genotype likelihoods (beagle) for %d individuals at %d sites; %d sites after filtering "
          "(minMaf %g, minInd %d)\n", D.N, sites_in, D.M, o.glf.minMaf, o.glf.minInd);
      if (o.glf.skip_missing)
        say("Missing GL entries (max - min GL < %g) left out: %zu of %zu (%.2f%%)\n", o.glf.misTol, nmis,
            D.keep.size(), 100.0 * nmis / D.keep.size());
      else
        say("Missing GL entries are used as data (--keep-missing)\n");
      analyse(D, o, t0);
    } else {
      Genotypes D = read_genotypes(o.input);
      analyse(D, o, t0);
    }
    say("Log written to %s\n", logname.c_str());
  } catch (const std::exception& e) {
    say_error(e.what());
    if (log_file()) std::fclose(log_file());
    return 1;
  }
  if (log_file()) std::fclose(log_file());
  return 0;
}
