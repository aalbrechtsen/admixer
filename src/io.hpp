// admixer: input (PLINK .bed/.bim/.fam) and output (.Q/.P matrices).
#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Genotypes {
  int M = 0, N = 0;        // SNPs, individuals
  std::vector<uint8_t> G;  // M x N, SNP-major; 0/1/2 = copies of the second allele, 3 = missing
  std::vector<int> nobs;   // non-missing genotypes per individual
  const uint8_t* row(int j) const { return G.data() + (size_t)j * N; }
  void count_observed() {
    nobs.assign(N, 0);
    for (int j = 0; j < M; j++)
      for (int i = 0; i < N; i++)
        if (G[(size_t)j * N + i] != 3) nobs[i]++;
  }
};

inline bool ends_with(const std::string& s, const std::string& e) {
  return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}
// "dir/data.bed" -> "dir/data"
inline std::string strip_extension(const std::string& s) {
  const size_t slash = s.find_last_of('/'), dot = s.find_last_of('.');
  return dot != std::string::npos && (slash == std::string::npos || dot > slash) ? s.substr(0, dot) : s;
}
// "dir/data.bed" -> "data" (ADMIXTURE writes its output with this prefix in the working directory)
inline std::string basename_noext(const std::string& s) {
  const std::string p = strip_extension(s);
  const size_t slash = p.find_last_of('/');
  return slash == std::string::npos ? p : p.substr(slash + 1);
}

inline int count_lines(const std::string& fn) {
  std::ifstream in(fn);
  if (!in) throw std::runtime_error("cannot open " + fn);
  int n = 0;
  std::string line;
  while (std::getline(in, line))
    if (!line.empty()) n++;
  return n;
}

// PLINK binary genotypes. Genotypes count copies of the second allele (A2, .bim column 6), so that
// .P holds the frequency of the same allele as ADMIXTURE's .P.
inline Genotypes read_bed(const std::string& bed) {
  const std::string prefix = strip_extension(bed);
  Genotypes D;
  D.M = count_lines(prefix + ".bim");
  D.N = count_lines(prefix + ".fam");
  FILE* fp = std::fopen(bed.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open " + bed);
  unsigned char magic[3];
  if (std::fread(magic, 1, 3, fp) != 3 || magic[0] != 0x6c || magic[1] != 0x1b || magic[2] != 0x01)
    throw std::runtime_error(bed + " is not a SNP-major PLINK .bed file");
  static const uint8_t decode[4] = {0, 3, 1, 2};  // 00 hom A1, 01 missing, 10 het, 11 hom A2
  const size_t bpr = (D.N + 3) / 4;
  std::vector<unsigned char> buf(bpr);
  D.G.resize((size_t)D.M * D.N);
  for (int j = 0; j < D.M; j++) {
    if (std::fread(buf.data(), 1, bpr, fp) != bpr) throw std::runtime_error(bed + " is truncated");
    uint8_t* g = D.G.data() + (size_t)j * D.N;
    for (int i = 0; i < D.N; i++) g[i] = decode[(buf[i >> 2] >> ((i & 3) * 2)) & 3];
  }
  std::fclose(fp);
  D.count_observed();
  return D;
}

inline Genotypes read_genotypes(const std::string& file) {
  if (ends_with(file, ".bed")) return read_bed(file);
  throw std::runtime_error("input must be a PLINK .bed file (with .bim and .fam): " + file);
}

// Whitespace-separated rows x K matrix (ADMIXTURE .Q/.P format).
inline void read_matrix(const std::string& fn, double* A, size_t rows, int K) {
  FILE* fp = std::fopen(fn.c_str(), "r");
  if (!fp) throw std::runtime_error("cannot open " + fn);
  for (size_t t = 0; t < rows * K; t++)
    if (std::fscanf(fp, "%lf", &A[t]) != 1)
      throw std::runtime_error(fn + ": expected " + std::to_string(rows) + " x " + std::to_string(K) + " values");
  std::fclose(fp);
}
inline void write_matrix(const std::string& fn, const double* A, size_t rows, int K) {
  FILE* fp = std::fopen(fn.c_str(), "w");
  if (!fp) throw std::runtime_error("cannot write " + fn);
  for (size_t r = 0; r < rows; r++)
    for (int k = 0; k < K; k++) std::fprintf(fp, "%.6f%c", A[r * K + k], k + 1 == K ? '\n' : ' ');
  std::fclose(fp);
}
