// admixer: input (PLINK .bed/.bim/.fam) and output (.Q/.P matrices).
#pragma once
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

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

// Whitespace-separated rows x K matrix (ADMIXTURE .Q/.P format); plain or gzip-compressed.
inline void read_matrix(const std::string& fn, double* A, size_t rows, int K) {
  gzFile fp = gzopen(fn.c_str(), "rb");  // reads uncompressed files transparently
  if (!fp) throw std::runtime_error("cannot open " + fn);
  std::string tok;
  size_t t = 0;
  int c;
  while (t < rows * K && (c = gzgetc(fp)) != -1) {
    if (std::isspace(c)) {
      if (!tok.empty()) A[t++] = std::strtod(tok.c_str(), nullptr), tok.clear();
    } else {
      tok += (char)c;
    }
  }
  if (t < rows * K && !tok.empty()) A[t++] = std::strtod(tok.c_str(), nullptr);
  gzclose(fp);
  if (t != rows * K)
    throw std::runtime_error(fn + ": expected " + std::to_string(rows) + " x " + std::to_string(K) + " values");
}
// Writes the matrix with 6 decimals; gzip-compressed if gz.
inline void write_matrix(const std::string& fn, const double* A, size_t rows, int K, bool gz = false) {
  std::string buf;
  char v[32];
  if (gz) {
    gzFile fp = gzopen(fn.c_str(), "wb6");
    if (!fp) throw std::runtime_error("cannot write " + fn);
    for (size_t r = 0; r < rows; r++) {
      buf.clear();
      for (int k = 0; k < K; k++) {
        std::snprintf(v, sizeof v, "%.6f%c", A[r * K + k], k + 1 == K ? '\n' : ' ');
        buf += v;
      }
      gzwrite(fp, buf.data(), buf.size());
    }
    gzclose(fp);
    return;
  }
  FILE* fp = std::fopen(fn.c_str(), "w");
  if (!fp) throw std::runtime_error("cannot write " + fn);
  for (size_t r = 0; r < rows; r++)
    for (int k = 0; k < K; k++) std::fprintf(fp, "%.6f%c", A[r * K + k], k + 1 == K ? '\n' : ' ');
  std::fclose(fp);
}
