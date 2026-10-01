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
