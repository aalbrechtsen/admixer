// simgl: simulates sequencing reads from called genotypes (a PLINK .bed) and writes genotype likelihoods in
// beagle format (gzipped), as ANGSD would compute them with a simple error model.
//
//   depth of individual i:   d_i = --depth D, or uniform in [a, b] with --depth a,b (mixed depths),
//                            or read from a file with one depth per individual (--depth-file F)
//   reads at an entry:       n ~ Poisson(d_i)
//   allele-2 reads:          k ~ Binomial(n, q_g),  q_0 = e, q_1 = 1/2, q_2 = 1 - e   (e = --error)
//   GL_g = q_g^k (1-q_g)^(n-k), scaled to sum to 1 (missing genotypes and n = 0 give 1/3, 1/3, 1/3)
//
// usage: simgl [--depth D | --depth a,b | --depth-file F] [--error e] [--seed s] [--sites M] in.bed out.beagle.gz
// Writes out.depth (the depth of every individual). Genotypes count copies of allele 2 (.bim column 6).
#include <zlib.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/io.hpp"

int main(int argc, char** argv) {
  double d_lo = 3, d_hi = 3, err = 0.01;
  unsigned long seed = 1;
  int max_sites = 0;
  std::string depth_file;
  std::vector<std::string> pos;
  for (int a = 1; a < argc; a++) {
    const std::string o = argv[a];
    if (o == "--depth" && a + 1 < argc) {
      const std::string v = argv[++a];
      const size_t c = v.find(',');
      d_lo = std::atof(v.substr(0, c).c_str());
      d_hi = c == std::string::npos ? d_lo : std::atof(v.substr(c + 1).c_str());
    } else if (o == "--depth-file" && a + 1 < argc) depth_file = argv[++a];
    else if (o == "--error" && a + 1 < argc) err = std::atof(argv[++a]);
    else if (o == "--seed" && a + 1 < argc) seed = std::strtoul(argv[++a], nullptr, 10);
    else if (o == "--sites" && a + 1 < argc) max_sites = std::atoi(argv[++a]);
    else pos.push_back(o);
  }
  if (pos.size() != 2) {
    std::fprintf(stderr,
                 "usage: simgl [--depth D | --depth a,b | --depth-file F] [--error e] [--seed s] [--sites M] "
                 "in.bed out.beagle.gz\n");
    return 1;
  }
  try {
    Genotypes D = read_genotypes(pos[0]);
    const int M = max_sites > 0 ? std::min(max_sites, D.M) : D.M, N = D.N;
    std::mt19937_64 rng(seed);
    std::vector<double> depth(N);
    if (!depth_file.empty()) {
      std::ifstream in(depth_file);
      for (int i = 0; i < N; i++)
        if (!(in >> depth[i])) throw std::runtime_error(depth_file + ": needs one depth per individual");
    } else {
      std::uniform_real_distribution<double> u(d_lo, d_hi);
      for (double& d : depth) d = u(rng);
    }
    const double q[3] = {err, 0.5, 1 - err};
    gzFile fp = gzopen(pos[1].c_str(), "wb6");
    if (!fp) throw std::runtime_error("cannot write " + pos[1]);
    std::string line = "marker\tallele1\tallele2";
    for (int i = 0; i < N; i++)
      for (int g = 0; g < 3; g++) line += "\tInd" + std::to_string(i);
    line += '\n';
    gzwrite(fp, line.data(), line.size());
    char b[48];
    for (int j = 0; j < M; j++) {
      line = "site" + std::to_string(j + 1) + "\t0\t1";
      const uint8_t* g = D.row(j);
      for (int i = 0; i < N; i++) {
        double l[3] = {1, 1, 1};
        if (g[i] != 3) {
          const int n = std::poisson_distribution<int>(depth[i])(rng);
          const int k = n ? std::binomial_distribution<int>(n, q[g[i]])(rng) : 0;
          double mx = -INFINITY, ll[3];
          for (int t = 0; t < 3; t++) mx = std::max(mx, ll[t] = k * std::log(q[t]) + (n - k) * std::log(1 - q[t]));
          for (int t = 0; t < 3; t++) l[t] = std::exp(ll[t] - mx);
        }
        const double s = l[0] + l[1] + l[2];
        std::snprintf(b, sizeof b, "\t%.6f\t%.6f\t%.6f", l[0] / s, l[1] / s, l[2] / s);
        line += b;
      }
      line += '\n';
      gzwrite(fp, line.data(), line.size());
    }
    gzclose(fp);
    std::string dn = pos[1];
    for (const char* e : {".gz", ".beagle"})
      if (ends_with(dn, e)) dn = dn.substr(0, dn.size() - std::strlen(e));
    FILE* df = std::fopen((dn + ".depth").c_str(), "w");
    for (double d : depth) std::fprintf(df, "%.3f\n", d);
    std::fclose(df);
    std::fprintf(stderr, "simgl: %d sites x %d individuals, depth %g-%g, error %g -> %s\n", M, N, d_lo, d_hi, err,
                 pos[1].c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
