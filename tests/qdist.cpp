// qdist: distance between two Q matrices after matching their columns (Hungarian algorithm, as --conv).
// usage: qdist A.Q B.Q K   -> prints "max_abs mean_sum rmse"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "../src/multistart.hpp"

static std::vector<double> readq(const char* fn, int K, int& N) {
  std::ifstream in(fn);
  std::vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  N = (int)v.size() / K;
  if (!in.eof() || N * K != (int)v.size() || N == 0) {
    std::fprintf(stderr, "qdist: cannot read %s as N x %d\n", fn, K);
    std::exit(1);
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc != 4) return std::fprintf(stderr, "usage: qdist A.Q B.Q K\n"), 1;
  const int K = std::atoi(argv[3]);
  int na, nb;
  const auto a = readq(argv[1], K, na), b = readq(argv[2], K, nb);
  if (na != nb) return std::fprintf(stderr, "qdist: %d vs %d rows\n", na, nb), 1;
  const QDistance d = q_distance(a.data(), b.data(), na, K);
  std::printf("%.6g %.6g %.6g\n", d.max_abs, d.mean_sum, d.rmse);
  return 0;
}
