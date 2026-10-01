// admixer: genotype likelihood input (beagle files, as ANGSD writes them and NGSadmix reads them) and the
// per-entry part of the NGSadmix likelihood (Skotte, Korneliussen & Albrechtsen 2013).
//
//   L_ij = GL0 (1-h)^2 + GL1 2h(1-h) + GL2 h^2,  h = sum_k Q_ik P_jk,  log L = sum_ij log L_ij
//
// Per entry the kernels of model.hpp need, as functions of h:
//   r1 = E[g | GL, h] / h         = 2 (GL1 (1-h) + GL2 h) / L      (EM ratios; g/h and (2-g)/(1-h) for
//   r0 = E[2-g | GL, h] / (1-h)   = 2 (GL0 (1-h) + GL1 h) / L       called genotypes)
//   d  = dlogL/dh = r1 - r0
//   w  = exact curvature -d2logL/dh2 = d^2 - 2 (GL0 - 2 GL1 + GL2) / L, clamped at 0 (log L is not concave
//        in h where GL0 GL2 > GL1^2), or the EM-surrogate curvature r1/h + r0/(1-h) (exact_hess = false).
//        The exact curvature equals the EM surrogate minus Var[g | GL, h] / (h (1-h))^2: the EM-type weight
//        overstates the curvature by the missing information, so its Newton steps are too short.
// (Near-)missing entries, max - min GL < misTol, are left out of the model (keep = 0); log(mean GL) is added
// as a constant (exact when the GLs are equal). loglik_all() of the model gives the exact log-likelihood.
#pragma once
#include <omp.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel.hpp"

struct GLData {
  static constexpr const char* kind = "genotype likelihoods";
  int M = 0, N = 0;
  std::vector<float> gl;         // (j*N + i)*3 + g, GLs as read (not rescaled: log L equals NGSadmix's)
  std::vector<uint8_t> keep;     // j*N + i: 1 = in the model, 0 = missing
  std::vector<int> nobs;         // entries in the model per individual
  std::vector<std::string> ids;  // marker names
  double ll_const = 0;           // sum of log(mean GL) over the missing entries
  bool exact_hess = true;
  const float* row(int j) const { return gl.data() + (size_t)j * N * 3; }
  const uint8_t* krow(int j) const { return keep.data() + (size_t)j * N; }
  double ll_offset() const { return ll_const; }

  static inline double lik(const float* l, double h) {
    const double a = 1 - h;
    return l[0] * a * a + 2.0 * l[1] * h * a + l[2] * h * h;
  }
  // Per-entry kernels, branch-free so that they vectorise (as Genotypes::row_pass): entries outside the model
  // get weight 0 and probability 1, and the three interleaved GLs are widened to double first.
  //   L = GL0 a^2 + 2 GL1 h a + GL2 h^2 (a = 1 - h),  r1 = 2 (GL1 a + GL2 h) / L,  r0 = 2 (GL0 a + GL1 h) / L,
  //   d = r1 - r0,  w = max(0, d^2 - 2 (GL0 - 2 GL1 + GL2) / L) (exact) or r1/h + r0/a (EM-type),
  // with one division per entry (EM-type: D = 1 / (L h a), 2/L = 2 h a D, 1/(h a) = L D).
  // Logs: LANES independent products, split exactly into mantissa and exponent after every PER_LOG factors (a
  // term can be ~bound^2 = 1e-18; 8 of them do not underflow), one log per lane per row.
  // MODE 0: log L only (all: missing entries included); 1: EM ratios (A = r0 in place of h, B = r1);
  // 2: Newton (A = d, B = w).
  static constexpr int LANES = 8, PER_LOG = 8, BLK = LANES * PER_LOG;
  template <int MODE, bool EXACT>
  [[gnu::always_inline]]  // inlined into each CPU version of the tile kernels (kernel.hpp)
  static inline double row_pass(const float* __restrict l, const uint8_t* __restrict kp, double* __restrict A,
                                double* __restrict B, int n, double lo, double hi, bool all) {
    double pr[BLK], q0[BLK], q1[BLK], q2[BLK], kd[BLK], prod[LANES], ex[LANES];
    for (int t = 0; t < LANES; t++) prod[t] = 1, ex[t] = 0;
    for (int c0 = 0; c0 < n; c0 += BLK) {
      const int cn = std::min(BLK, n - c0);
      const float* lc = l + 3 * (size_t)c0;
      double* Ac = A + c0;
      double* Bc = MODE >= 1 ? B + c0 : nullptr;
      for (int u = 0; u < cn; u++) {
        q0[u] = lc[3 * u], q1[u] = lc[3 * u + 1], q2[u] = lc[3 * u + 2];
        kd[u] = (kp[c0 + u] || (MODE == 0 && all)) ? 1.0 : 0.0;
      }
#pragma omp simd
      for (int u = 0; u < cn; u++) {
        double h = Ac[u];
        h = h < lo ? lo : h;
        h = h > hi ? hi : h;
        const double a = 1 - h, g0 = q0[u], g1 = q1[u], g2 = q2[u];
        const double L = g0 * a * a + 2.0 * g1 * h * a + g2 * h * h;
        const bool obs = kd[u] > 0.5;
        pr[u] = obs ? L : 1.0;
        if (MODE >= 1) {
          double iL, iha = 0;
          if (EXACT || MODE == 1) {
            iL = 2.0 / L;
          } else {
            const double D = 1 / (L * h * a);
            iL = 2.0 * h * a * D, iha = L * D;
          }
          const double r1 = (g1 * a + g2 * h) * iL, r0 = (g0 * a + g1 * h) * iL;
          if (MODE == 1) {
            Ac[u] = obs ? r0 : 0.0, Bc[u] = obs ? r1 : 0.0;
          } else {
            const double d = r1 - r0;
            double w;
            if (EXACT) {
              w = d * d - (g0 - 2.0 * g1 + g2) * iL;
              w = w > 0 ? w : 0.0;
            } else {
              w = (r1 * a + r0 * h) * iha;
            }
            Ac[u] = obs ? d : 0.0, Bc[u] = obs ? w : 0.0;
          }
        }
      }
      for (int u = cn; u < BLK; u++) pr[u] = 1;
      for (int r = 0; r < PER_LOG; r++)
        for (int t = 0; t < LANES; t++) prod[t] *= pr[r * LANES + t];
      for (int t = 0; t < LANES; t++) {
        uint64_t bits;
        std::memcpy(&bits, &prod[t], 8);
        ex[t] += (double)((int64_t)(bits >> 52) - 1023);
        bits = (bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
        std::memcpy(&prod[t], &bits, 8);
      }
    }
    double ll = 0;
    for (int t = 0; t < LANES; t++) ll += std::log(prod[t]) + ex[t] * 0.69314718055994530942;
    return ll;
  }

  // all = true: every entry, the missing ones included (the exact log-likelihood, as NGSadmix)
  ADMIXER_KERNEL double tile_ll(int j0, int jn, int i0, int in, const double* H, double lo, double hi, bool all = false) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)  // MODE 0 only reads the tile
      ll += row_pass<0, true>(row(j0 + jj) + 3 * (size_t)i0, krow(j0 + jj) + i0,
                              const_cast<double*>(H) + (size_t)jj * in, nullptr, in, lo, hi, all);
    return ll;
  }
  ADMIXER_KERNEL double tile_em(int j0, int jn, int i0, int in, double* R0, double* R1, double lo, double hi) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++)
      ll += row_pass<1, true>(row(j0 + jj) + 3 * (size_t)i0, krow(j0 + jj) + i0, R0 + (size_t)jj * in,
                              R1 + (size_t)jj * in, in, lo, hi, false);
    return ll;
  }
  ADMIXER_KERNEL double tile_wd(int j0, int jn, int i0, int in, double* T, double* W, double lo, double hi) const {
    double ll = 0;
    for (int jj = 0; jj < jn; jj++) {
      const float* l = row(j0 + jj) + 3 * (size_t)i0;
      const uint8_t* kp = krow(j0 + jj) + i0;
      double* t = T + (size_t)jj * in;
      double* w = W + (size_t)jj * in;
      ll += exact_hess ? row_pass<2, true>(l, kp, t, w, in, lo, hi, false)
                       : row_pass<2, false>(l, kp, t, w, in, lo, hi, false);
    }
    return ll;
  }
  void permute(const std::vector<int>& perm) {
    std::vector<float> g2(gl.size());
    std::vector<uint8_t> k2(keep.size());
#pragma omp parallel for schedule(static)
    for (int j = 0; j < M; j++) {
      std::copy(row(perm[j]), row(perm[j]) + 3 * N, g2.begin() + (size_t)j * 3 * N);
      std::copy(krow(perm[j]), krow(perm[j]) + N, k2.begin() + (size_t)j * N);
    }
    gl.swap(g2);
    keep.swap(k2);
    std::vector<std::string> id2(M);
    for (int j = 0; j < M; j++) id2[j] = ids[perm[j]];
    ids.swap(id2);
  }
  void supervised_start(const std::vector<int>& label, int K, double* P, double lo, double hi) const;
};

struct GLFilter {
  double misTol = 0.05;    // NGSadmix -misTol: entries with max - min GL < misTol count as missing
  bool skip_missing = true;  // leave the missing entries out of the model (false: use them as data)
  double minMaf = 0.05;    // keep sites with minMaf < MAF < 1 - minMaf (as NGSadmix; 0 = off)
  int minInd = 0;          // keep sites with more than minInd informative individuals (0 = off)
};

// NGSadmix's allele-frequency EM (emFrequency: start 0.3, at most 20 iterations, entries with keep != 0).
inline double em_frequency(const float* l, const uint8_t* keep, int N) {
  int n = 0;
  for (int i = 0; i < N; i++) n += keep[i];
  if (n == 0) return 0.0;
  double p = 0.3;
  for (int it = 0; it < 20; it++) {
    double s = 0;
    for (int i = 0; i < N; i++) {
      if (!keep[i]) continue;
      const double w0 = l[3 * i] * (1 - p) * (1 - p), w1 = l[3 * i + 1] * 2 * p * (1 - p), w2 = l[3 * i + 2] * p * p;
      s += (w1 + 2 * w2) / (2 * (w0 + w1 + w2));
    }
    const double pn = s / n;
    const bool done = std::fabs(pn - p) < 1e-5;
    p = pn;
    if (done) break;
  }
  return p;
}

inline double parse_num(const char* c, const char** end) {
  while (*c == ' ' || *c == '\t') c++;
  const char* s0 = c;
  bool neg = false;
  if (*c == '-' || *c == '+') neg = *c++ == '-';
  static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};
  uint64_t mant = 0;
  int nd = 0, nfrac = 0;
  while (*c >= '0' && *c <= '9') mant = 10 * mant + (*c++ - '0'), nd++;
  if (*c == '.') {
    c++;
    while (*c >= '0' && *c <= '9') mant = 10 * mant + (*c++ - '0'), nd++, nfrac++;
  }
  if (nd == 0 || nd > 18) {  // nan, inf, very long numbers  // nan, inf, ...
    char* e;
    const double r = std::strtod(s0, &e);
    *end = e;
    return r;
  }
  double v = (double)mant / p10[nfrac];
  if (*c == 'e' || *c == 'E') {
    char* e;
    const long ex = std::strtol(c + 1, &e, 10);
    v *= std::pow(10.0, (double)ex);
    c = e;
  }
  *end = c;
  return neg ? -v : v;
}

// Reads a (gzipped or plain) beagle file (header: marker allele1 allele2 then 3 columns per individual) and applies the site filters. The file is decompressed into
// memory and the lines are parsed in parallel.
inline GLData read_beagle(const std::string& fn, const GLFilter& fs, int* sites_in = nullptr) {
  gzFile fp = gzopen(fn.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open " + fn);
  gzbuffer(fp, 1 << 20);
  std::string txt;
  {
    std::vector<char> buf(1 << 22);
    int r;
    while ((r = gzread(fp, buf.data(), (unsigned)buf.size())) > 0) txt.append(buf.data(), r);
    gzclose(fp);
  }
  std::vector<size_t> starts;  // line starts (non-empty lines)
  for (size_t p = 0; p < txt.size();) {
    size_t e = txt.find('\n', p);
    if (e == std::string::npos) e = txt.size();
    if (e > p && !(e == p + 1 && txt[p] == '\r')) starts.push_back(p);
    if (e < txt.size()) txt[e] = '\0';
    p = e + 1;
  }
  if (starts.empty()) throw std::runtime_error(fn + " is empty");
  int cols = 0;
  {
    const char* c = txt.c_str() + starts[0];
    while (*c) {
      while (*c == ' ' || *c == '\t' || *c == '\r') c++;
      if (!*c) break;
      cols++;
      while (*c && *c != ' ' && *c != '\t' && *c != '\r') c++;
    }
  }
  if (cols < 6 || (cols - 3) % 3) throw std::runtime_error(fn + ": header must have 3 + 3N columns");
  GLData D;
  D.N = (cols - 3) / 3;
  const int N = D.N, nin = (int)starts.size() - 1;
  std::vector<float> gl((size_t)nin * 3 * N);
  std::vector<uint8_t> kp((size_t)nin * N), pass(nin);
  std::vector<double> lcs(nin);
  std::vector<std::string> ids(nin);
  int bad = -1;  // first bad line
  std::string why;
#pragma omp parallel
  {
    std::vector<uint8_t> kf(N);
#pragma omp for schedule(dynamic, 256)
    for (int j = 0; j < nin; j++) {
      const char* c = txt.c_str() + starts[j + 1];
      while (*c == ' ' || *c == '\t') c++;
      const char* id0 = c;
      while (*c && *c != ' ' && *c != '\t') c++;
      ids[j].assign(id0, c);
      float* l = gl.data() + (size_t)j * 3 * N;
      uint8_t* k = kp.data() + (size_t)j * N;
      const char* e;
      parse_num(c, &e), c = e;  // allele1
      parse_num(c, &e), c = e;  // allele2
      bool ok = true;
      for (int t = 0; t < 3 * N && ok; t++) {
        l[t] = (float)parse_num(c, &e);
        ok = e != c;
        c = e;
      }
      double lc = 0;
      int ninf = 0;
      for (int i = 0; i < N && ok; i++) {
        const float* g = l + 3 * i;
        const double s = (double)g[0] + g[1] + g[2];
        if (!(s > 0) || g[0] < 0 || g[1] < 0 || g[2] < 0) ok = false;
        const float mx = std::max(g[0], std::max(g[1], g[2])), mn = std::min(g[0], std::min(g[1], g[2]));
        kf[i] = mx - mn >= fs.misTol;
        k[i] = !fs.skip_missing || (kf[i] && mx - mn > 1e-6);
        ninf += kf[i];
        if (!k[i]) lc += std::log(s / 3.0);  // L ~ mean GL (exact when the three GLs are equal)
      }
      if (!ok) {
#pragma omp critical
        if (bad < 0 || j < bad) bad = j;
        continue;
      }
      bool keep = true;
      if (fs.minMaf > 0) {
        const double maf = em_frequency(l, kf.data(), N);
        keep = maf > fs.minMaf && maf < 1 - fs.minMaf;
      }
      if (fs.minInd > 0 && ninf <= fs.minInd) keep = false;
      pass[j] = keep;
      lcs[j] = lc;
    }
  }
  if (bad >= 0)
    throw std::runtime_error(fn + ": line " + std::to_string(bad + 2) + " (site " + ids[bad] +
                             ") has too few columns or likelihoods that are negative or sum to 0");
  txt.clear();
  txt.shrink_to_fit();
  for (int j = 0; j < nin; j++) {
    if (!pass[j]) continue;
    if (D.M != j) {
      std::copy(gl.begin() + (size_t)j * 3 * N, gl.begin() + (size_t)(j + 1) * 3 * N, gl.begin() + (size_t)D.M * 3 * N);
      std::copy(kp.begin() + (size_t)j * N, kp.begin() + (size_t)(j + 1) * N, kp.begin() + (size_t)D.M * N);
    }
    D.ids.push_back(std::move(ids[j]));
    D.ll_const += lcs[j];
    D.M++;
  }
  gl.resize((size_t)D.M * 3 * N);
  kp.resize((size_t)D.M * N);
  gl.shrink_to_fit();
  kp.shrink_to_fit();
  D.gl.swap(gl);
  D.keep.swap(kp);
  if (sites_in) *sites_in = nin;
  if (D.M == 0) throw std::runtime_error(fn + ": no sites left after filtering");
  D.nobs.assign(N, 0);
  for (int j = 0; j < D.M; j++)
    for (int i = 0; i < N; i++) D.nobs[i] += D.keep[(size_t)j * N + i];
  return D;
}

// Supervised start: P_jk = allele frequency (NGSadmix's EM) among the individuals labelled k.
inline void GLData::supervised_start(const std::vector<int>& label, int K, double* P, double lo, double hi) const {
#pragma omp parallel
  {
    std::vector<uint8_t> sel(N);
#pragma omp for schedule(static)
    for (int j = 0; j < M; j++)
      for (int k = 0; k < K; k++) {
        int n = 0;
        for (int i = 0; i < N; i++) n += (sel[i] = label[i] == k && krow(j)[i]);
        if (n > 0) P[(size_t)j * K + k] = std::min(std::max(em_frequency(row(j), sel.data(), N), lo), hi);
      }
  }
}
