// admixer: input (PLINK .bed/.bim/.fam) and output (.Q/.P matrices).
#pragma once
#include <algorithm>
#include <cassert>
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
// The genotypes are kept packed, 2 bits each (as in the .bed file; 4x less memory than a byte each), with
// admixer's own codes: 0/1/2 = copies of the second allele, 3 = missing. Individual i of a row is in byte
// i / 4 at bits 2 (i % 4); bits past N in a row's last byte hold 3.
struct GenoTables {
  double val[256][4];  // the 4 genotypes of a packed byte, as doubles (the per-entry pass)
  uint8_t code[256][4];
  uint8_t from_bed[256];  // .bed byte -> packed byte (PLINK: 00 hom A1, 01 missing, 10 het, 11 hom A2)
  // pseudohaploid (one allele per entry, copies of A2): hom A1 -> 0, hom A2 -> 1; heterozygous calls are
  // left at 3 here and set by het_allele (read_bed)
  uint8_t from_bed_hap[256];
  uint8_t nhet[256];  // heterozygous calls in a .bed byte
  GenoTables() {
    static const uint8_t dec[4] = {0, 3, 1, 2}, hap[4] = {0, 3, 3, 1};
    for (int b = 0; b < 256; b++) {
      from_bed[b] = from_bed_hap[b] = nhet[b] = 0;
      for (int t = 0; t < 4; t++) {
        const uint8_t g = (b >> (2 * t)) & 3;
        code[b][t] = g, val[b][t] = g;
        from_bed[b] |= dec[g] << (2 * t);
        from_bed_hap[b] |= hap[g] << (2 * t);
        nhet[b] += g == 2;
      }
    }
  }
};
inline const GenoTables GTAB;

// Pseudohaploid analysis: the allele (0 = A1, 1 = A2) drawn for a heterozygous call of SNP number snp (line of the
// .bim, from 1) and individual number ind (line of the .fam, from 1). A fixed hash (splitmix64) of the two
// numbers: a fair coin, independent between entries, and the same in every run (no seed). Rules shared between
// individuals (the same allele for all heterozygotes of a SNP, or alternating by SNP) bias Q or create
// artificial clusters (DETAILS.md, "Pseudohaploid analysis").
inline uint8_t het_allele(uint64_t snp, uint64_t ind) {
  uint64_t z = (snp << 32 | ind) + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return (uint8_t)((z ^ (z >> 31)) & 1);
}

struct Genotypes {
  static constexpr const char* kind = "genotypes";
  int M = 0, N = 0;        // SNPs, individuals
  size_t bpr = 0;          // bytes per packed row, (N + 3) / 4
  std::vector<uint8_t> G;  // M x bpr, SNP-major, packed (see above)
  std::vector<int> nobs;   // non-missing genotypes per individual
  bool haploid = false;    // pseudohaploid: one allele per entry, g = 0/1 (read_bed with pseudohaploid = true)
  int ploidy() const { return haploid ? 1 : 2; }
  void resize(int m, int n) {
    M = m, N = n, bpr = ((size_t)n + 3) / 4;
    G.assign((size_t)M * bpr, 0xFF);  // all missing
  }
  const uint8_t* row(int j) const { return G.data() + (size_t)j * bpr; }  // packed
  uint8_t get(int j, int i) const { return (row(j)[i >> 2] >> ((i & 3) * 2)) & 3; }
  void set(int j, int i, uint8_t g) {
    uint8_t& b = G[(size_t)j * bpr + (i >> 2)];
    const int s = (i & 3) * 2;
    b = (uint8_t)((b & ~(3 << s)) | ((g & 3) << s));
  }
  // genotypes of individuals 0..N-1 of row j into out (N bytes)
  void decode(int j, uint8_t* out) const {
    const uint8_t* r = row(j);
    for (int i = 0; i < N; i++) out[i] = GTAB.code[r[i >> 2]][i & 3];
  }
  void count_observed() {
    nobs.assign(N, 0);
    constexpr int CB = 256;  // bytes (1024 individuals) per task
#pragma omp parallel for schedule(dynamic)
    for (int b0 = 0; b0 < (int)bpr; b0 += CB) {
      const int b1 = std::min((int)bpr, b0 + CB);
      for (int j = 0; j < M; j++)
        for (int b = b0; b < b1; b++) {
          const uint8_t* c = GTAB.code[G[(size_t)j * bpr + b]];
          for (int t = 0; t < 4 && 4 * b + t < N; t++) nobs[4 * b + t] += c[t] != 3;
        }
    }
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
  // TB: element type of B (float for the single-precision Hessians, see Model::hess_float).
  static constexpr int LANES = 8, PER_LOG = 16, BLK = LANES * PER_LOG;
  // HAP: pseudohaploid data (one allele per entry, g = 0/1): prob = h^g a^(1-g), r0 = (1-g)/(1-h); the other
  // formulas are the same.
  // g: packed genotypes of the row's entries (starting at a multiple of 4 individuals)
  template <int MODE, class TB = double, bool HAP = false>
  [[gnu::always_inline]]  // inlined into each CPU version of the tile kernels (kernel.hpp)
  static inline double row_pass(const uint8_t* __restrict g, double* __restrict A, TB* __restrict B, int n,
                                double lo, double hi) {
    double pr[BLK], gd[BLK], prod[LANES], ex[LANES];
    for (int l = 0; l < LANES; l++) prod[l] = 1, ex[l] = 0;
    for (int c0 = 0; c0 < n; c0 += BLK) {
      const int cn = std::min(BLK, n - c0);
      const uint8_t* gc = g + c0 / 4;
      double* Ac = A + c0;
      TB* Bc = MODE >= 1 ? B + c0 : nullptr;
      for (int b = 0; b < (cn + 3) / 4; b++)  // unpacked and widened first: the main loop has one element type
        std::memcpy(gd + 4 * b, GTAB.val[gc[b]], 4 * sizeof(double));
#pragma omp simd
      for (int u = 0; u < cn; u++) {
        double h = Ac[u];
        h = h < lo ? lo : h;
        h = h > hi ? hi : h;
        const double a = 1 - h, inv = 1 / (h * a), ih = a * inv, ia = h * inv;
        const double g = gd[u];
        const bool obs = g < 2.5;
        const double gg = obs ? g : 0.0, cc = obs ? (HAP ? 1.0 : 2.0) - g : 0.0;
        const double x1 = g > 0.5 ? h : a, x2 = HAP ? 1.0 : g > 1.5 ? h : a;
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
    assert(i0 % 4 == 0);  // a tile starts at a byte of the packed rows
    double ll = 0;
    for (int jj = 0; jj < jn; jj++) {
      ll += haploid ? row_pass<0, double, true>(row(j0 + jj) + i0 / 4, const_cast<double*>(H) + (size_t)jj * in, nullptr, in, lo, hi)
                    : row_pass<0, double>(row(j0 + jj) + i0 / 4, const_cast<double*>(H) + (size_t)jj * in, nullptr, in, lo, hi);  // reads only
    }
    return ll;
  }
  // EM ratios: on exit R1 = g/h, R0 = (2-g)/(1-h) (0 for missing). R0 holds h on entry. Returns log L.
  ADMIXER_KERNEL double tile_em(int j0, int jn, int i0, int in, double* R0, double* R1, double lo, double hi) const {
    assert(i0 % 4 == 0);  // a tile starts at a byte of the packed rows
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += haploid ? row_pass<1, double, true>(row(j0 + jj) + i0 / 4, R0 + (size_t)jj * in, R1 + (size_t)jj * in, in, lo, hi)
                    : row_pass<1>(row(j0 + jj) + i0 / 4, R0 + (size_t)jj * in, R1 + (size_t)jj * in, in, lo, hi);
    return ll;
  }
  // Newton quantities: on exit T = d = dlogL/dh, W = w = -d2logL/dh2 (0 for missing). T holds h on entry.
  ADMIXER_KERNEL double tile_wd(int j0, int jn, int i0, int in, double* T, double* W, double lo, double hi) const {
    assert(i0 % 4 == 0);  // a tile starts at a byte of the packed rows
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += haploid ? row_pass<2, double, true>(row(j0 + jj) + i0 / 4, T + (size_t)jj * in, W + (size_t)jj * in, in, lo, hi)
                    : row_pass<2>(row(j0 + jj) + i0 / 4, T + (size_t)jj * in, W + (size_t)jj * in, in, lo, hi);
    return ll;
  }  // the same with w in single precision (Hessian matrix products in float, Model::hess_float)
  ADMIXER_KERNEL double tile_wd(int j0, int jn, int i0, int in, double* T, float* W, double lo, double hi) const {
    assert(i0 % 4 == 0);  // a tile starts at a byte of the packed rows
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += haploid ? row_pass<2, float, true>(row(j0 + jj) + i0 / 4, T + (size_t)jj * in, W + (size_t)jj * in, in, lo, hi)
                    : row_pass<2, float>(row(j0 + jj) + i0 / 4, T + (size_t)jj * in, W + (size_t)jj * in, in, lo, hi);
    return ll;
  }
  // Reorders the sites: row j becomes row perm[j] of the input. In place (cycle by cycle, one row buffer), so
  // that the genotypes never exist twice.
  void permute(const std::vector<int>& perm) {
    std::vector<char> done(M, 0);
    std::vector<uint8_t> tmp(bpr);
    for (int s = 0; s < M; s++) {
      if (done[s]) continue;
      std::memcpy(tmp.data(), row(s), bpr);
      int j = s;
      for (;;) {  // row j takes row perm[j]; the cycle closes when perm[j] = s
        done[j] = 1;
        const int k = perm[j];
        if (k == s) {
          std::memcpy(G.data() + (size_t)j * bpr, tmp.data(), bpr);
          break;
        }
        std::memcpy(G.data() + (size_t)j * bpr, row(k), bpr);
        j = k;
      }
    }
  }
  // Supervised start: P_jk = allele frequency among the individuals labelled k (label[i] = -1: unknown),
  // with a +0.5 / +1 pseudo-count; columns without labelled data are left as they are.
  void supervised_start(const std::vector<int>& label, int K, double* P, double lo, double hi) const {
    std::vector<uint8_t> g(N);
    for (int j = 0; j < M; j++) {
      decode(j, g.data());
      std::vector<double> s(K, 0.0), n(K, 0.0);
      for (int i = 0; i < N; i++)
        if (label[i] >= 0 && g[i] != 3) s[label[i]] += g[i], n[label[i]] += ploidy();
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
// pseudohaploid: one allele per entry (homozygous calls their allele, heterozygous calls het_allele);
// *nhet = heterozygous calls.
inline Genotypes read_bed(const std::string& bed, bool pseudohaploid = false, size_t* nhet = nullptr) {
  const std::string prefix = strip_extension(bed);
  Genotypes D;
  D.haploid = pseudohaploid;
  size_t het = 0;
  D.resize(count_lines(prefix + ".bim"), count_lines(prefix + ".fam"));
  FILE* fp = std::fopen(bed.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open " + bed);
  unsigned char magic[3];
  if (std::fread(magic, 1, 3, fp) != 3 || magic[0] != 0x6c || magic[1] != 0x1b || magic[2] != 0x01)
    throw std::runtime_error(bed + " is not a SNP-major PLINK .bed file");
  const size_t bpr = D.bpr;
  const int tail = D.N % 4;  // padding bits of a row's last byte are set to 3 (missing)
  const uint8_t pad = tail ? (uint8_t)(0xFF << (2 * tail)) : 0;
  for (int j = 0; j < D.M; j++) {
    uint8_t* g = D.G.data() + (size_t)j * bpr;
    if (std::fread(g, 1, bpr, fp) != bpr) throw std::runtime_error(bed + " is truncated");
    if (pseudohaploid) {
      for (size_t b = 0; b < bpr; b++) {
        const uint8_t raw = g[b];
        g[b] = GTAB.from_bed_hap[raw];
        if (GTAB.nhet[raw]) {
          het += GTAB.nhet[raw];
          for (int t = 0; t < 4; t++)
            if (((raw >> (2 * t)) & 3) == 2)
              g[b] = (uint8_t)((g[b] & ~(3 << (2 * t))) | het_allele(j + 1, 4 * b + t + 1) << (2 * t));
        }
      }
    } else {
      for (size_t b = 0; b < bpr; b++) g[b] = GTAB.from_bed[g[b]];
    }
    g[bpr - 1] |= pad;
  }
  std::fclose(fp);
  D.count_observed();
  if (nhet) *nhet = het;
  return D;
}

inline Genotypes read_genotypes(const std::string& file, bool pseudohaploid = false, size_t* nhet = nullptr) {
  if (ends_with(file, ".bed")) return read_bed(file, pseudohaploid, nhet);
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
  bool extra = false;  // anything after the expected values is an error, not silently dropped
  if (t == rows * K)
    while ((c = gzgetc(fp)) != -1)
      if (!std::isspace(c)) {
        extra = true;
        break;
      }
  gzclose(fp);
  if (t != rows * K || extra)
    throw std::runtime_error(fn + ": expected " + std::to_string(rows) + " x " + std::to_string(K) +
                             " values (one row per SNP/site in the analysis: for genotype likelihoods, the sites "
                             "left after --minMaf/--minInd filtering)");
}
// Writes the matrix with 6 decimals; gzip-compressed if gz. Chunks of rows are formatted (and compressed) in
// parallel on the OpenMP threads; a compressed file is a series of gzip members, one per chunk, which gzip,
// zcat, zlib's gzread and R read as one stream. The text is the same as from a serial write.
inline std::string gzip_member(const std::string& text, int level) {
  z_stream zs{};
  if (deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("deflateInit2 failed");
  std::string out(deflateBound(&zs, text.size()), '\0');
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(text.data()));
  zs.avail_in = text.size();
  zs.next_out = reinterpret_cast<Bytef*>(&out[0]);
  zs.avail_out = out.size();
  const int rc = deflate(&zs, Z_FINISH);
  out.resize(zs.total_out);
  deflateEnd(&zs);
  if (rc != Z_STREAM_END) throw std::runtime_error("deflate failed");
  return out;
}
inline void write_matrix(const std::string& fn, const double* A, size_t rows, int K, bool gz = false) {
  FILE* fp = std::fopen(fn.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot write " + fn);
  const size_t CH = 2048, nch = (rows + CH - 1) / CH, BATCH = 256;  // rows per chunk; chunks held in memory
  std::vector<std::string> out(std::min(nch, BATCH));
  bool ok = true;
  for (size_t c0 = 0; c0 < nch; c0 += BATCH) {
    const size_t cn = std::min(BATCH, nch - c0);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < cn; c++) {
      std::string buf;
      char v[32];
      const size_t r0 = (c0 + c) * CH, r1 = std::min(rows, r0 + CH);
      for (size_t r = r0; r < r1; r++)
        for (int k = 0; k < K; k++) {
          const int len = std::snprintf(v, sizeof v, "%.6f%c", A[r * K + k], k + 1 == K ? '\n' : ' ');
          buf.append(v, len);
        }
      out[c] = gz ? gzip_member(buf, 6) : std::move(buf);
    }
    for (size_t c = 0; c < cn; c++) ok = ok && std::fwrite(out[c].data(), 1, out[c].size(), fp) == out[c].size();
  }
  ok = std::fclose(fp) == 0 && ok;
  if (!ok) throw std::runtime_error("error writing " + fn);
}
