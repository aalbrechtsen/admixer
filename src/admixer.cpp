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
#include "model.hpp"

static const char* VERSION = "0.1.0";

static void usage(int code) {
  std::printf(
      "admixer %s: ancestry estimation under the ADMIXTURE model\n\n"
      "usage: admixer [options] inputFile K\n\n"
      "  inputFile  a PLINK .bed file (with .bim and .fam next to it)\n"
      "  K          number of ancestral populations\n\n"
      "Output: inputBasename.K.Q and inputBasename.K.P in the working directory (as ADMIXTURE)\n\n"
      "options:\n"
      "  -jX, -j X           use X threads (default 8)\n"
      "  --seed=X, -s X      random seed (default 43; 'time' uses the clock)\n"
      "  -C=X, -C X          stop when the log-likelihood improves by less than X (default 1e-4)\n"
      "  -o NAME, --out=NAME output prefix NAME instead of inputBasename (NAME.K.Q, NAME.K.P)\n"
      "  --supervised        supervised analysis: reads inputBasename.pop (one line per individual,\n"
      "                      a population name, or '-' if unknown); labelled individuals are held at\n"
      "                      their population; K must equal the number of population names\n"
      "  -P                  projection: reads inputBasename.K.P.in and estimates Q with P held fixed\n"
      "  --evaladmix         also write NAME.K.corres.txt, the evalAdmix correlation of residuals\n"
      "                      (corrected estimator of van Waaij et al. 2023; plot with evalAdmix's visFuns.R)\n"
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
  bool supervised = false, projection = false, evaladmix = false;
  int K = 0;
  std::vector<std::string> pos;
  for (int a = 1; a < argc; a++) {
    const std::string o = argv[a];
    if (o == "-h" || o == "--help") usage(0);
    else if (o.rfind("--seed", 0) == 0) seed_s = opt_value(argc, argv, a, "--seed");
    else if (o.rfind("--out", 0) == 0) out = opt_value(argc, argv, a, "--out");
    else if (o == "--supervised") supervised = true;
    else if (o == "--evaladmix") evaladmix = true;
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
  if (K < 1 || threads < 1) usage(1);
  fs.seed = seed_s == "time" ? (unsigned long)std::time(nullptr) : std::strtoul(seed_s.c_str(), nullptr, 10);
  if (out.empty()) out = basename_noext(input);
  omp_set_num_threads(threads);

  try {
    std::printf("admixer %s\n", VERSION);
    const double t0 = omp_get_wtime();
    Genotypes D = read_genotypes(input);
    std::printf("Random seed: %lu\n", fs.seed);
    std::printf("Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)\n");
    std::printf("Convergence acceleration algorithm: QuasiNewton, %d secant conditions\n", fs.qn_secants);
    std::printf("Point estimation will terminate when objective function delta < %g\n", fs.tol);
    std::printf("Size of G: %dx%d\n", D.N, D.M);
    std::printf("Threads: %d\n", threads);

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
      std::printf("Supervised analysis mode: %d labelled individuals\n",
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
    Vec x(m.size());
    std::mt19937_64 rng(fs.seed);
    if (projection) {
      const std::string fn = strip_extension(input) + "." + std::to_string(K) + ".P.in";
      read_matrix(fn, x.data(), D.M, K);
      m.pfix = true;
      std::printf("Projection mode: P read from %s and held fixed\n", fn.c_str());
    }
    if (supervised) {
      for (int i = 0; i < D.N; i++)
        if (label[i] >= 0) {
          double* q = x.data() + m.nP + (size_t)i * K;
          for (int k = 0; k < K; k++) q[k] = k == label[i];
          m.qfix[i] = 1;
        }
    }
    Fitter fit(m, fs);
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
    const FitResult r = fit.run(x);
    const auto kkt = m.kkt(x.data());
    std::printf("Summary: \n");
    std::printf("Converged in %d iterations (%.3f sec)\n", r.iterations, omp_get_wtime() - t0);
    std::printf("Loglikelihood: %f\n", r.loglik);
    std::printf("Optimality check (max KKT violation): P %.2e per individual, Q %.2e per SNP\n", kkt.first, kkt.second);

    std::printf("Writing output files.\n");
    const std::string pre = out + "." + std::to_string(K);
    write_matrix(pre + ".Q", x.data() + m.nP, D.N, K);
    std::vector<double> Pout(m.nP);
    for (int j = 0; j < D.M; j++)
      std::copy(x.begin() + (size_t)j * K, x.begin() + (size_t)(j + 1) * K, Pout.begin() + (size_t)perm[j] * K);
    write_matrix(pre + ".P", Pout.data(), D.M, K);
    if (evaladmix) {
      const double te = omp_get_wtime();
      const auto cor = evaladmix_corrected(D, x.data() + m.nP, K, threads);
      write_corres(pre + ".corres.txt", cor, D.N);
      std::printf("evalAdmix correlation of residuals written to %s.corres.txt (%.2f sec)\n", pre.c_str(),
                  omp_get_wtime() - te);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
