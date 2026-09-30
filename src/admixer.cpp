// admixer: maximum-likelihood ancestry estimation under the ADMIXTURE model, with ADMIXTURE's
// command line, input and output. Algorithm: block relaxation with Newton/QP steps and quasi-Newton
// acceleration (as ADMIXTURE), evaluated with BLAS, started from a random start, 5 EM steps and a
// mini-batch warm-up.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "evaladmix.hpp"
#include "fit.hpp"
#include "io.hpp"
#include "log.hpp"
#include "model.hpp"
#include "multistart.hpp"

static const char* VERSION = "0.1.0";

static void usage(int code) {
  std::printf(
      "admixer %s: ancestry estimation under the ADMIXTURE model\n\n"
      "usage: admixer [options] inputFile K\n\n"
      "  inputFile  a PLINK .bed file (with .bim and .fam next to it)\n"
      "  K          number of ancestral populations\n\n"
      "Output (working directory): inputBasename.K.Q, inputBasename.K.P.gz and inputBasename.K.corres.txt\n\n"
      "options:\n"
      "  -jX, -j X           use X threads (default 8)\n"
      "  --seed=X, -s X      random seed (default 43; 'time' uses the clock)\n"
      "  -C=X, -C X          stop when the log-likelihood improves by less than X (default 1e-4)\n"
      "  -o NAME, --out=NAME output prefix NAME instead of inputBasename\n"
      "  --no-gzip           write NAME.K.P uncompressed instead of NAME.K.P.gz\n"
      "  --supervised        supervised analysis: reads inputBasename.pop (one line per individual,\n"
      "                      a population name, or '-' if unknown); labelled individuals are held at\n"
      "                      their population; K must equal the number of population names\n"
      "  -P                  projection: reads inputBasename.K.P.in (or .P.in.gz) and estimates Q with P fixed\n"
      "  --no-evaladmix      do not write NAME.K.corres.txt, the evalAdmix correlation of residuals\n"
      "                      (corrected estimator of van Waaij et al. 2023; plot with evalAdmix's visFuns.R).\n"
      "                      It needs memory ~ 48 N^2 bytes, so it is skipped for N > 20000 individuals\n"
      "                      unless --evaladmix is given\n"
      "  --conv X            convergence test with several starts (seeds seed, seed+1, ...): stop when X runs\n"
      "                      agree with the best run (highest likelihood); writes the best run and NAME.K.conv\n"
      "  -m X, --max_runs=X  with --conv: at most X runs (default 10)\n"
      "  --conv_thres=X      with --conv: runs agree if the largest difference in any Q entry, after matching\n"
      "                      ancestries, is below X (default 0.01)\n"
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

int main(int argc, char** argv) {
  FitSettings fs;
  int threads = 8;
  std::string seed_s = "43", out, input;
  bool supervised = false, projection = false, evaladmix = true, evaladmix_forced = false, gz = true;
  int K = 0, conv = 0, max_runs = 10;
  double conv_thres = 0.01;
  std::vector<std::string> pos;
  for (int a = 1; a < argc; a++) {
    const std::string o = argv[a];
    if (o == "-h" || o == "--help") usage(0);
    else if (o.rfind("--seed", 0) == 0) seed_s = opt_value(argc, argv, a, "--seed");
    else if (o.rfind("--out", 0) == 0) out = opt_value(argc, argv, a, "--out");
    else if (o == "--supervised") supervised = true;
    else if (o == "--evaladmix") evaladmix = evaladmix_forced = true;
    else if (o == "--no-evaladmix") evaladmix = false;
    else if (o == "--no-gzip") gz = false;
    else if (o.rfind("--conv_thres", 0) == 0) conv_thres = std::atof(opt_value(argc, argv, a, "--conv_thres").c_str());
    else if (o.rfind("--conv", 0) == 0) conv = std::atoi(opt_value(argc, argv, a, "--conv").c_str());
    else if (o.rfind("--max_runs", 0) == 0) max_runs = std::atoi(opt_value(argc, argv, a, "--max_runs").c_str());
    else if (o.rfind("-m", 0) == 0) max_runs = std::atoi(opt_value(argc, argv, a, "-m").c_str());
    else if (o.rfind("--max-iter", 0) == 0) fs.max_iter = std::atoi(opt_value(argc, argv, a, "--max-iter").c_str());
    else if (o == "-P") projection = true;
    else if (o.rfind("-j", 0) == 0) threads = std::atoi(opt_value(argc, argv, a, "-j").c_str());
    else if (o.rfind("-s", 0) == 0) seed_s = opt_value(argc, argv, a, "-s");
    else if (o.rfind("-C", 0) == 0) fs.tol = std::atof(opt_value(argc, argv, a, "-C").c_str());
    else if (o.rfind("-o", 0) == 0) out = opt_value(argc, argv, a, "-o");
    else if (!o.empty() && o[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", o.c_str());
      usage(1);
    } else pos.push_back(o);
  }
  if (pos.size() != 2) usage(1);
  input = pos[0];
  K = std::atoi(pos[1].c_str());
  if (K < 1 || threads < 1 || conv < 0 || max_runs < 1) usage(1);
  fs.seed = seed_s == "time" ? (unsigned long)std::time(nullptr) : std::strtoul(seed_s.c_str(), nullptr, 10);
  if (out.empty()) out = basename_noext(input);
  omp_set_num_threads(threads);
  const std::string logname = out + "." + std::to_string(K) + ".log";
  log_file() = std::fopen(logname.c_str(), "w");
  if (!log_file()) std::fprintf(stderr, "Warning: cannot write log file %s\n", logname.c_str());
  std::string cmd;
  for (int a = 0; a < argc; a++) cmd += (a ? " " : "") + std::string(argv[a]);

  try {
    say("admixer %s\n", VERSION);
    say("Command: %s\n", cmd.c_str());
    const double t0 = omp_get_wtime();
    Genotypes D = read_genotypes(input);
    say("Random seed: %lu\n", fs.seed);
    say("Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)\n");
    say("Convergence acceleration algorithm: QuasiNewton, %d secant conditions\n", fs.qn_secants);
    say("Point estimation will terminate when objective function delta < %g\n", fs.tol);
    say("Size of G: %dx%d\n", D.N, D.M);
    say("Threads: %d\n", threads);

    // Supervised mode: population labels from inputBasename.pop, columns in order of first appearance
    std::vector<int> label(D.N, -1);
    if (supervised) {
      const std::string fn = strip_extension(input) + ".pop";
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
    if (!projection && fs.minibatch > 1) {
      std::mt19937_64 prng(fs.seed + 1);
      std::shuffle(perm.begin(), perm.end(), prng);
      std::vector<uint8_t> G2((size_t)D.M * D.N);
#pragma omp parallel for schedule(static)
      for (int j = 0; j < D.M; j++) std::copy(D.row(perm[j]), D.row(perm[j]) + D.N, G2.begin() + (size_t)j * D.N);
      D.G.swap(G2);
    }

    Model m(D, K);
    Vec x0(m.size());  // fixed parts (projection P, supervised rows) shared by all runs
    if (projection) {
      std::string fn = strip_extension(input) + "." + std::to_string(K) + ".P.in";
      if (!std::ifstream(fn) && std::ifstream(fn + ".gz")) fn += ".gz";
      read_matrix(fn, x0.data(), D.M, K);
      m.pfix = true;
      say("Projection mode: P read from %s and held fixed\n", fn.c_str());
    }
    if (supervised) {
      for (int i = 0; i < D.N; i++)
        if (label[i] >= 0) {
          double* q = x0.data() + m.nP + (size_t)i * K;
          for (int k = 0; k < K; k++) q[k] = k == label[i];
          m.qfix[i] = 1;
        }
    }
    // One run per seed: seed, seed+1, ... (a single run without --conv).
    const int nruns = conv > 0 ? max_runs : 1;
    if (conv > 0)
      say("Convergence test: up to %d runs (seeds %lu-%lu); converged when %d runs agree with the best run "
          "(max |dQ| < %g)\n", nruns, fs.seed, fs.seed + nruns - 1, conv, conv_thres);
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
      Fitter fit(m, fr);
      fit.verbose = r == 0;  // iteration details for the first run only
      fit.init_random(x, rng);
      if (supervised && !projection) {  // start P at the allele frequencies of the labelled individuals
        for (int j = 0; j < D.M; j++) {
          const uint8_t* g = D.row(j);
          std::vector<double> s(K, 0.0), n(K, 0.0);
          for (int i = 0; i < D.N; i++)
            if (label[i] >= 0 && g[i] != 3) s[label[i]] += g[i], n[label[i]] += 2;
          for (int k = 0; k < K; k++)
            if (n[k] > 0) x[(size_t)j * K + k] = std::min(std::max((s[k] + 0.5) / (n[k] + 1), PMIN), PMAX);
        }
      }
      const FitResult res = fit.run(x);
      Run run{fr.seed, res.loglik, res.seconds, res.iterations,
              std::vector<float>(x.begin() + m.nP, x.end()), QDistance(), false};
      runs.push_back(std::move(run));
      if (best < 0 || res.loglik > runs[best].loglik) best = r, best_x = x;
      if (conv == 0) break;
      // agreement of every run with the current best run (the best can change, so recompute all)
      agreeing = 0;
      for (auto& u : runs) {
        u.d = q_distance(u.Q.data(), runs[best].Q.data(), D.N, K);
        u.agrees = u.d.max_abs < conv_thres;
        agreeing += u.agrees;
      }
      say("Run %d (seed %lu): loglik %f, %d iterations, %.1f sec; max |dQ| to best run %.4f; %d of %d runs agree\n",
          r + 1, fr.seed, res.loglik, res.iterations, res.seconds, runs.back().d.max_abs, agreeing, r + 1);
      if (agreeing >= conv) break;
    }
    x = best_x;
    const auto kkt = m.kkt(x.data());
    say("Summary: \n");
    if (conv > 0) {
      say("%s: %d of %d runs agree with the best run (seed %lu, max |dQ| < %g)\n",
          agreeing >= conv ? "Converged" : "Not converged", agreeing, (int)runs.size(), runs[best].seed, conv_thres);
      const std::string fn = out + "." + std::to_string(K) + ".conv";
      FILE* fp = std::fopen(fn.c_str(), "w");
      if (!fp) throw std::runtime_error("cannot write " + fn);
      std::fprintf(fp, "run\tseed\tloglik\tloglik_diff\tmax_abs_dQ\tmean_sum_abs_dQ\trmse_dQ\titerations\tseconds\tagrees\n");
      for (size_t r = 0; r < runs.size(); r++) {
        const Run& u = runs[r];
        std::fprintf(fp, "%zu\t%lu\t%.6f\t%.6f\t%.6g\t%.6g\t%.6g\t%d\t%.2f\t%d\n", r + 1, u.seed, u.loglik,
                     u.loglik - runs[best].loglik, u.d.max_abs, u.d.mean_sum, u.d.rmse, u.iters, u.seconds, (int)u.agrees);
      }
      std::fclose(fp);
      say("Convergence table written to %s\n", fn.c_str());
    }
    say("Converged in %d iterations (%.3f sec)\n", runs[best].iters, omp_get_wtime() - t0);
    say("Loglikelihood: %f\n", runs[best].loglik);
    say("Optimality check (max KKT violation): P %.2e per individual, Q %.2e per SNP\n", kkt.first, kkt.second);

    say("Writing output files.\n");
    const std::string pre = out + "." + std::to_string(K);
    write_matrix(pre + ".Q", x.data() + m.nP, D.N, K);
    std::vector<double> Pout(m.nP);
    for (int j = 0; j < D.M; j++)
      std::copy(x.begin() + (size_t)j * K, x.begin() + (size_t)(j + 1) * K, Pout.begin() + (size_t)perm[j] * K);
    write_matrix(pre + (gz ? ".P.gz" : ".P"), Pout.data(), D.M, K, gz);
    if (evaladmix && D.N > 20000 && !evaladmix_forced) {
      say("evalAdmix skipped: %d individuals would need ~%.0f GB of memory (use --evaladmix to force)\n", D.N,
                  48.0 * D.N * D.N / 1e9);
      evaladmix = false;
    }
    if (evaladmix) {
      const double te = omp_get_wtime();
      const auto cor = evaladmix_corrected(D, x.data() + m.nP, K, threads);
      write_corres(pre + ".corres.txt", cor, D.N);
      say("evalAdmix correlation of residuals written to %s.corres.txt (%.2f sec)\n", pre.c_str(),
                  omp_get_wtime() - te);
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
