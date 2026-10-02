# Kernel performance work (branch `perf`, 2026-10-01)

Question: why do the BLAS kernels give so little? Data: `bench/data/sim100k_K{5,20}` (simulated, M = 100k,
N = 2k, Fst 0.05, alpha 0.5, 1% missing, 2 unadmixed anchors per population). Xeon Gold 6152, load ~2-7.
Profiler: `tools/prof_kernels data.bed K threads` (`make tools/prof_kernels`).

## Diagnosis
Phase times of one sqp_P pass (K = 5, 8 threads, ns per genotype entry, summed over threads):

| phase | before | after |
|---|---|---|
| h = P Q' (GEMM) | 0.96 | 0.80 |
| elementwise d, w, log L | **12.65** | 3.02 |
| Hessian GEMM W Z | 2.88 | 2.65 |
| gradient GEMM | 0.64 | 0.63 |
| QP solves | 1.62 | 0.77 |

At K = 20 the QP took 19.5 ns/entry before and 4.1 after. The Hessian GEMM (10.8) is then the largest phase.

* The GEMMs were never the bottleneck. The **elementwise step branched on the genotype** (`g == 0 ? ... : g == 2 ? ...`,
  `if (g == 3) continue`). gcc compiled it to scalar code, and with random genotypes about every other branch is
  mispredicted (~10 ns/entry). Tile sizes made no difference, so cache misses were not the cause.
* The QP solver allocated vectors on every call, used pivoted LU on the bordered system, and added one bound per
  active-set iteration (9.4 iterations per P row at K = 20).
* sqp_Q used blocks of 56 individuals x 512 SNPs: short rows for the elementwise pass, and Y was re-packed for
  every block of individuals.

## Changes (all exact: same log-likelihood to ~1e-8, same iterations)
1. `io.hpp`: branch-free elementwise kernels (`row_pass<MODE>`). Genotypes are widened to double first, selects
   compile to blends, there is one division per entry, and logs are replaced by exact mantissa/exponent
   renormalisation (one log per lane per row).
2. `linalg.hpp`: QP with thread-local workspace, a right-looking Cholesky (vectorised row updates) with the
   equality constraint by Schur complement, and LU as a fallback. For box-only QPs (rows of P), a primal-dual
   active-set method moves many bounds per iteration and verifies KKT exactly; it falls back to the old method.
3. `model.hpp`: sqp_Q blocks of 1024 individuals x 64 SNPs (swept: 2-2.6x faster than 56 x 512).

Kernel wall times (8 threads):

| kernel | K=5 before | K=5 after | K=20 before | K=20 after |
|---|---|---|---|---|
| loglik | 0.271 | 0.077 | 0.286 | 0.081 |
| em | 0.406 | 0.176 | 0.566 | 0.324 |
| sqp_P | 0.485 | 0.210 | 1.213 | 0.590 |
| sqp_Q | 0.512 | 0.157 | 0.917 | 0.474 |

## End to end (default pipeline: 5 EM, mini-batch warm-up, brqn; seed 1)

| K | threads | before wall / CPU | after wall / CPU | speed-up | log-likelihood (both) | iterations |
|---|---|---|---|---|---|---|
| 5 | 8 | 60.5 / 473 s | 22.6 / 180 s | 2.7x | -210712543.733534 | 15 |
| 20 | 8 | 93.1 / 718 s | 57.0 / 447 s | 1.6x | -212247490.828962 | 10 |
| 5 | 44 | 14.2 / 572 s | 7.1 / 267 s | 2.0x | -210712543.733534 | 15 |

## Later steps
* **Log L from the next P step** (`fit.hpp`): the separate log L pass per QN iteration is dropped (the P step
  returns log L of its input). Same iterates. K=5: 22.6 -> 21.8 s; K=20: no change.
* **512-bit vectors** (`-mprefer-vector-width=512` in the Makefile): K=5 21.8 -> 18.8 s, K=20 56.3 -> 54.7 s.
* **Warm-up schedule:** a cap of e epochs per number of batches (`--mb-epochs`, tried with e = 1, 2, 3) saved ~12% at
  K=20 and nothing reliable at K=5. On the hard datasets it lowered S2's success rate from 3/10 to 0/10.
  **Not adopted; the option was removed.**

## Hard multi-modal datasets (admix_prog/bench/work_hard; 10 seeds each, 8 threads)

| data (K) | old: hits / median wall | new: hits / median wall |
|---|---|---|
| H1 (8) | 0/10, 23.1 s | 0/10, 8.9 s |
| L2 (10) | 4/10, 19.5 s | 4/10, 9.2 s |
| S2 (8) | 3/10, 24.9 s | 3/10, 10.3 s |

hits = runs within 1 log-unit of the best optimum known. Same seed, old vs new: |dll| < 1e-3 (same optima).

## Genotype likelihoods (beagle input, NGSadmix model)
The same rewrite of the per-entry step in `beagle.hpp` (`GLData::row_pass<MODE, EXACT>`): masks instead of the
missing-entry branch, the three GLs widened to double per chunk, one division per entry, exponent renormalisation
(every 8 factors, as the old log-per-8 rule). Data: `tools/simgl` on the first 10k sites of sim100k_K5 (N = 2000),
depth 4 and mixed depth 0.5-6.

| kernel (K=5, 8 threads) | old | new |
|---|---|---|
| elementwise step, ns/entry | 12.9 | 4.0 |
| sqp_P | 0.047 s | 0.025 s |
| sqp_Q | 0.036 s | 0.019 s |

End to end (K=5, default GL pipeline): depth 4: 9.9 -> 5.9 s (CPU 65 -> 35 s); mixed depth: 10.5-10.8 -> 5.9-6.4 s.
The log-likelihood agrees to 4e-5. Iteration counts change by -1 to +5, because the new QP solver is more exact on the
nearly singular P-row Hessians of GL data (the exact curvature is clamped at 0). Compared on every QP of the
bgl_mixed test: the new solution has a lower objective in 155 of 90,912 P rows (by up to 2x) and a higher one in 8
(by <= 4e-4 relative). On genotype data the two solvers agree to 1e-13.

`--minibatch=0` (no warm-up) is slower: K=5 28.8 s (34 iterations), K=20 126.7 s (54 iterations).
After the kernel changes the **warm-up takes about half of the run** (K=20: 31 s of 55; 23 epochs, 14 of them
at 16 batches gaining <50k log-units each), so its schedule is the next target.

Other findings: AVX-512 (`-mprefer-vector-width=512`) is ~10% faster on the elementwise step alone (not adopted).
A 20 x 20 Cholesky takes ~1.8 us on this machine, so one factorisation per row is the floor of the QP.

# Larger K (branch `hessian-speed`, 2026-10-02)

Question: can BR-QN be made faster at K ≈ 10–20, by reusing Hessians or by dropping Hessian rows/columns of
components with Q ≈ 0? Main data: `data/blue_wildebeest_noLD_r02.bed` (73 individuals, 185,861 SNPs), K = 10;
also wildebeest 37k (K = 7), `admix_prog/bench/work/anch_20000_2000_10_0.05` (sparse Q, N = 2000),
sim100k_K10/K20 and a large simulation (N = 10,000, M = 50,000, K = 20). Xeon E5-2699 v4, GCC 11.4, `-march=native`.
Comparisons are paired: the variants run at the same time with the same seed, 8 threads each, `--no-P`.
**Adopted:** only the output and the GCC workaround below. **Not adopted:** the three speed-ups, because none
of them gains at large N (the experiments are kept on the branches `hessian-speed`, `hess-reuse`, `hess-sparse`).

## Where the time goes at N = 73
One step at the converged wildebeest solution (`tools/prof_kernels`, now with the sqp_Q phase split and
`PROF_AT=prefix` to profile at a given `.Q`/`.P`), ns per genotype entry:

| phase | P pass | Q pass |
|---|---|---|
| h = P Qᵀ GEMM | 1.9 | 3.9 |
| per-entry step | 7.7 | 9.7 |
| packing Z / Y | 0.5 | 1.5 |
| Hessian GEMM | 5.5 | 9.8 |
| gradient GEMM | 1.8 | 4.7 |
| row QPs | **26.5** | 0.0 |

With few individuals the P-row QPs (one per SNP, ~2 µs each, two 10 × 10 Cholesky factorisations) dominate, not
the O(MNK²) Hessian GEMM. With N = 2000 the QPs are a small share.

## Tried, not adopted
* **K as a compile-time template parameter** for the QP: 0.76× (masked K × K system) and 0.96× (compacted free
  block) on 50k stored P-row QPs. The Cholesky is bound by its chain of square roots and divisions.
* **Warm start of the box QP** (variables at a bound with the gradient pointing out start as active, so one
  factorisation instead of two): QPs 1.40 → 0.69 µs, same solutions. End to end, wall vs the current code,
  median [range]: wildebeest r02 0.85 [0.79, 0.89] (10 seeds), 37k 0.95, anchored sim 0.98, sim100k 1.00 (± 5 %
  noise). Identical log-likelihoods and iterations on every seed, and identical global-maximum hit rates over 110
  seed pairs on 7 data sets. Gains only for small N.
* **Hessian reuse within a QN iteration** (`--hess-reuse`: F(F(x)) uses the Hessians of F(x); exact gradient, so
  the same fixed points): wildebeest 1–3 % less CPU. N = 10,000, M = 50,000, K = 20 (16 threads): −28 % per QN
  iteration, ~12 % less wall time to the same log-likelihood (within 0.01), 6 % end to end at the default `-C`
  (3 extra tail iterations); 84 MB of stored Hessians. The warm-up, half of the run, is untouched.
* **Sparse P-row Hessians** (`--hess-sparse=τ`: only pairs of ancestries both > τ in some individual, exact
  diagonal). Converged Q at τ = 1e-3: wildebeest r02 1.36 active components per individual, 16 of 55 pairs
  co-occur; wildebeest 37k 25 of 28; H1, L2 and the sim100k sets all pairs. Full fits (τ = 1e-3): wildebeest r02
  CPU 0.98× median but up to 1.48× (the changed curvature changes the QN path: 7 → 21 and 103 → 120 iterations
  on two seeds), 37k 1.00×, anchored sim and sim100k 1.02×. With many individuals every pair co-occurs, so only
  the overhead is left. The Q side (per-individual block over the active set) would save at most ~8 % of a
  wildebeest step and nothing on dense data; not tried.
* Considered from the literature and not tried: lazy Hessians refreshed every m iterations (Doikov, Chayti &
  Jaggi 2023), sub-sampled Hessians (Roosta-Khorasani & Mahoney 2016), per-row quasi-Newton (PQN-R, Hansen,
  Plantenga & Kolda 2015), Anderson/DAAREM instead of the ZAL quasi-Newton (Henderson & Varadhan 2019).
  Hessian-free CG and float Hessians were judged not worth it at K = 10.

## Adopted
* **Output:** `.P.gz` was written by `snprintf` plus single-threaded gzip (2.0–2.7 s for 186k × 10). Chunks of 2048
  rows are now formatted and compressed in parallel, one gzip member each (read as one stream by gzip, zcat,
  zlib and R): 2.6 → 0.45 s, same text, file 1 % larger. `--no-P` skips the P matrix. A wildebeest K = 10 run:
  10.2 → 8.7 s wall (8.5 s with `--no-P`).
* **GCC 11 miscompile of the stopping rule:** GCC 11.4 at `-O3 -march=native` (AVX2 + FMA) computed `ll − prev`
  in `qn()` as `−ll`, so `-C` never fired (22 instead of 12 iterations). Correct with `-O2`,
  `-fno-tree-slp-vectorize`, `-mno-fma` or x86-64-v2; sanitizers clean. Cause: basic-block vectorisation around
  the inlined OpenMP extrapolation loop, now in a non-inlined `extrapolate()` (bit-identical results). A test
  checks that the run ends on a change below `-C`. The release binary (GCC 14) was not affected.
