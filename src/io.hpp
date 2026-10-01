// admixer: input (PLINK .bed/.bim/.fam) and output (.Q/.P matrices).
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include "kernel.hpp"

// Called genotypes (PLINK input). Besides the data, a data type provides the per-entry part of the
// likelihood that the BLAS kernels of model.hpp need (see GLData in beagle.hpp for the other data type).
// On entry every tile holds h = sum_k Q_ik P_jk; lo/hi are the bounds h is clamped to.
struct Genotypes {
  static constexpr const char* kind = "genotypes";
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
  double ll_offset() const { return 0; }  // constant added to the log-likelihood

  // Per-entry kernels, branch-free so that they vectorise (missing entries get weight 0 and probability 1).
  // With a = 1 - h and one division inv = 1 / (h a):  1/h = a inv,  1/(1-h) = h inv.
  //   prob = h^g a^(2-g),  r1 = g/h,  r0 = (2-g)/(1-h),  d = r1 - r0,  w = r1/h + r0/(1-h).
  // The selects on g are written so that they compile to vector blends: branches on random genotypes are
  // mispredicted about every other entry (that cost ~10 ns per entry before).
  // A row is processed in chunks of BLK entries. The probabilities are multiplied into LANES independent
  // products; after every 16 factors each product is split exactly into mantissa and binary exponent, so
  // there is one log per lane per row (every probability is >= lo^2, so 16 factors cannot underflow for
  // bounds >= ~1e-19).
  // MODE 0: log-likelihood only; 1: EM ratios (A = r0 in place of h, B = r1); 2: Newton (A = d, B = w).
  static constexpr int LANES = 8, PER_LOG = 16, BLK = LANES * PER_LOG;
  template <int MODE>
  [[gnu::always_inline]]  // inlined into each CPU version of the tile kernels (kernel.hpp)
  static inline double row_pass(const uint8_t* __restrict g, double* __restrict A, double* __restrict B, int n,
                                double lo, double hi) {
    double pr[BLK], gd[BLK], prod[LANES], ex[LANES];
    for (int l = 0; l < LANES; l++) prod[l] = 1, ex[l] = 0;
    for (int c0 = 0; c0 < n; c0 += BLK) {
      const int cn = std::min(BLK, n - c0);
      const uint8_t* gc = g + c0;
      double* Ac = A + c0;
      double* Bc = MODE >= 1 ? B + c0 : nullptr;
      for (int u = 0; u < cn; u++) gd[u] = gc[u];  // widened first: the main loop then has one element type
#pragma omp simd
      for (int u = 0; u < cn; u++) {
        double h = Ac[u];
        h = h < lo ? lo : h;
        h = h > hi ? hi : h;
        const double a = 1 - h, inv = 1 / (h * a), ih = a * inv, ia = h * inv;
        const double g = gd[u];
        const bool obs = g < 2.5;
        const double gg = obs ? g : 0.0, cc = obs ? 2.0 - g : 0.0;
        const double x1 = g > 0.5 ? h : a, x2 = g > 1.5 ? h : a;
        pr[u] = obs ? x1 * x2 : 1.0;
        if (MODE >= 1) {
          const double r1 = gg * ih, r0 = cc * ia;
          if (MODE == 1) {
            Ac[u] = r0, Bc[u] = r1;
          } else {
            Ac[u] = r1 - r0, Bc[u] = r1 * ih + r0 * ia;
          }
        }
      }
      for (int u = cn; u < BLK; u++) pr[u] = 1;
      for (int r = 0; r < PER_LOG; r++)
        for (int l = 0; l < LANES; l++) prod[l] *= pr[r * LANES + l];
      // exact renormalisation: prod = m 2^e with m in [1, 2); the exponent goes to ex
      for (int l = 0; l < LANES; l++) {
        uint64_t b;
        std::memcpy(&b, &prod[l], 8);
        ex[l] += (double)((int64_t)(b >> 52) - 1023);
        b = (b & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
        std::memcpy(&prod[l], &b, 8);
      }
    }
    double ll = 0;
    for (int l = 0; l < LANES; l++) ll += std::log(prod[l]) + ex[l] * 0.69314718055994530942;
    return ll;
  }

  // log-likelihood of the tile H (jn x in) of sites j0.., individuals i0.. (all: no effect for genotypes,
  // whose missing entries carry no likelihood)
  ADMIXER_KERNEL double tile_ll(int j0, int jn, int i0, int in, const double* H, double lo, double hi, bool = false) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++) {
      ll += row_pass<0>(row(j0 + jj) + i0, const_cast<double*>(H) + (size_t)jj * in, nullptr, in, lo, hi);  // reads only
    }
    return ll;
  }
  // EM ratios: on exit R1 = g/h, R0 = (2-g)/(1-h) (0 for missing). R0 holds h on entry. Returns log L.
  ADMIXER_KERNEL double tile_em(int j0, int jn, int i0, int in, double* R0, double* R1, double lo, double hi) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += row_pass<1>(row(j0 + jj) + i0, R0 + (size_t)jj * in, R1 + (size_t)jj * in, in, lo, hi);
    return ll;
  }
  // Newton quantities: on exit T = d = dlogL/dh, W = w = -d2logL/dh2 (0 for missing). T holds h on entry.
  ADMIXER_KERNEL double tile_wd(int j0, int jn, int i0, int in, double* T, double* W, double lo, double hi) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += row_pass<2>(row(j0 + jj) + i0, T + (size_t)jj * in, W + (size_t)jj * in, in, lo, hi);
    return ll;
  }
  // Reorders the sites: row j becomes row perm[j] of the input.
  void permute(const std::vector<int>& perm) {
    std::vector<uint8_t> G2((size_t)M * N);
#pragma omp parallel for schedule(static)
    for (int j = 0; j < M; j++) std::copy(row(perm[j]), row(perm[j]) + N, G2.begin() + (size_t)j * N);
    G.swap(G2);
  }
  // Supervised start: P_jk = allele frequency among the individuals labelled k (label[i] = -1: unknown),
  // with a +0.5 / +1 pseudo-count; columns without labelled data are left as they are.
  void supervised_start(const std::vector<int>& label, int K, double* P, double lo, double hi) const {
    for (int j = 0; j < M; j++) {
      const uint8_t* g = row(j);
      std::vector<double> s(K, 0.0), n(K, 0.0);
      for (int i = 0; i < N; i++)
        if (label[i] >= 0 && g[i] != 3) s[label[i]] += g[i], n[label[i]] += 2;
      for (int k = 0; k < K; k++)
        if (n[k] > 0) P[(size_t)j * K + k] = std::min(std::max((s[k] + 0.5) / (n[k] + 1), lo), hi);
    }
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
