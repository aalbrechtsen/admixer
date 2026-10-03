# admixer: details

Background for [README.md](README.md): the algorithms, the convergence test, advanced options, performance,
testing and the differences from ADMIXTURE and NGSadmix.

## Algorithms

**Called genotypes (ADMIXTURE model).** g_ij ~ Binomial(2, h_ij) with h_ij = Σ_k Q_ik P_jk.
1. The fit starts from a random P and a near-uniform Q.
2. It takes 5 EM steps.
3. A mini-batch warm-up follows: block-relaxation steps on SNP batches. The number of batches is halved
   whenever an epoch does not improve the likelihood.
4. Then it runs ADMIXTURE's block relaxation until the log-likelihood improves by less than `-C`. Each
   iteration takes one Newton/QP step for every row of P and then for every row of Q, accelerated by
   quasi-Newton extrapolation with 3 secant pairs.
5. **Damped extrapolation** (`--qn-damp`, default 1). ADMIXTURE discards an extrapolated point that does not beat
   F(x) and continues from F(F(x)). On hard, multimodal data this happens in 36–90 % of the iterations, and the
   run crawls. admixer first retries the extrapolation with half the step (`--qn-damp-factor`, default 0.5) and
   discards it only if that also fails. When no extrapolation is rejected (most data), the run is unchanged.
   * Hard simulated data (14 sets, M = 10,000–100,000, N = 600–2,000, K = 6–12, 5–10 seeds each): 1.2–4.2× faster
     where extrapolations were rejected (the more rejections, the larger the gain), identical where none were.
     Every seed reached the same optimum as without damping, or a higher one.
   * Genotype likelihoods (NGSadmix tutorial data, K = 3–6; simulated GLs, M = 10,000, N = 2,000): the same optimum
     for every seed, 11 % less time in total (per data set and K from 2 % slower to 1.8× faster).
   * `--qn-damp=0` restores ADMIXTURE's rule.
6. **Single-precision Hessians** (`--hess-float`, default auto: on for called genotypes with K ≥ 8). In the main
   algorithm, the per-row Newton Hessians (the matrix products W Z and Wᵀ Y, the largest part of a pass at larger
   K) are computed in single precision and accumulated in double. The gradient, the log-likelihood and the QP
   steps stay in double, so the solutions the algorithm converges to are unchanged; the iterates differ at about
   1e-6. The EM steps and the warm-up stay in double: there, the rounding changed which local optimum some runs
   on multimodal data reached.
   * Called genotypes (12 hard sets with 10 seeds and M = 100,000 data, K = 5–20, 8 threads): every seed reached
     the same optimum as in double precision. At K ≥ 8 the runs were 6–35 % faster (more on hard data, where the
     Hessians also needed fewer iterations); at K = 5–7 only 0–6 %, so auto leaves it off there.
   * Genotype likelihoods: not used by default. With the GL bounds (1e-9) single precision loses the curvature
     of most entries next to the near-bound ones, and on the NGSadmix tutorial data (K = 4–6) runs reached
     worse optima and took 1.5–1.9× longer.
   * `--hess-float=0` gives exact double precision, `--hess-float=1` forces single precision.

The parameter bounds and the stopping rule are those of ADMIXTURE.

**Genotype likelihoods (NGSadmix model).** The genotype is summed out:
L_ij = GL0 (1 − h)² + GL1 · 2h(1 − h) + GL2 · h² (Skotte, Korneliussen & Albrechtsen 2013).
* **Algorithm:** the same block relaxation with quasi-Newton acceleration, from a random start. It uses no
  EM steps and no warm-up, because neither helped on the NGSadmix tutorial data, and the warm-up made the
  best solution rarer.
* **Curvature:** the Newton steps use the exact second derivative of log L_ij in h, clamped at 0. For GLs,
  ADMIXTURE's EM-type weight overstates the curvature by the missing information Var[g | GL] / (h(1 − h))²,
  so its steps are too short. It was 2–4× slower.
* **Filters and bounds:** as NGSadmix. Sites with an estimated MAF ≤ 0.05 are removed, and the bounds are 1e-9.
* **Missing data:** GLs with max − min < 0.05 (`--misTol`, NGSadmix's definition of missing) are left out of
  the model, and their nearly constant likelihood is added as a constant. The log also gives the exact
  log-likelihood over all entries, which is NGSadmix's objective.
* **evalAdmix correlation of residuals:** computed by default, as for genotypes. It applies the corrected
  estimator to the posterior expected genotypes E[g | GL, h], with the fitted model as prior.
  * Every entry enters the projection; an entry without data has E[g] = 2h, which the projection removes, so
    nothing is imputed.
  * The correlations use only the sites where both individuals have data, as `evalAdmix -beagle` does.

  In simulations at 1–8× and mixed 0.5–6× depth, its accuracy against the correlations from the true
  genotypes matched `evalAdmix -beagle` (within 7 %, unbiased, no depth artefacts), at about 5 % of its cost
  (0.2–0.3 s vs 10–20 s for 200 individuals). On the NGSadmix tutorial data the two agree with r = 0.999.
  At very low depth it is slightly worse than `evalAdmix -beagle`: when half of the individuals are at 0.1×,
  the correlations between the higher-depth individuals are shifted down by about 0.004–0.009. Pairs of
  two 0.1× individuals share too few sites to be informative with either tool.

**Common to both.**
* Every pass over the data works on tiles of SNPs × individuals: a matrix product gives the tile of
  H = P Qᵀ, an element-wise step computes the likelihood terms, and further matrix products give the EM
  statistics, the gradients and the Newton Hessians. The tiles run in parallel on OpenMP threads.
* The same seed and number of threads give identical results.
* The log ends with a first-order optimality check: the largest step of the projected gradient
  (gradient / N for P, gradient / M for Q), which is 0 at a local optimum.

## Convergence test (`--conv`)

Hard data sets can have several local optima, so one run may not find the best solution.
`--conv X` runs admixer with consecutive seeds (`--seed`, `--seed`+1, ...), reading the data only once.
After each run it matches the ancestries of every run to those of the best run so far (highest
log-likelihood), and a run agrees with the best run if the largest difference in any Q entry is below
`--conv_thres`. It stops when X runs (including the best) agree, or after `--max_runs` runs. This is the
Q-matrix criterion of popgenDK's `testQconv.R`, with an exact optimal matching of the ancestries.

The output files are those of the best run. `data.K.conv` lists every run, sorted by log-likelihood (best
run first), with these columns:
* seed;
* the log-likelihood and its difference to the best run;
* the distances to the best run's Q: the largest absolute difference, the mean per-individual sum of
  absolute differences, and the RMSE;
* iterations and seconds;
* whether the run agrees.

The log says whether the runs converged.

Example: `admixer --seed=30 --conv 3 data.bed 8` uses seeds 30, 31, 32, ... until 3 runs agree
(at most 10 runs).

## Advanced options

Mainly for benchmarking: `--bound=X` (P in [X, 1 − X] and Q ≥ X; default 1e-5 for genotypes as ADMIXTURE,
1e-9 for GLs as NGSadmix), `--prime=X` (EM steps before the main algorithm), `--minibatch=X` (initial number
of mini-batches of the warm-up), `--hess=exact|em` (curvature of the Newton steps for GLs), `--qn-damp=X` and
`--qn-damp-factor=F` (retries of a rejected quasi-Newton extrapolation with the step scaled by F; default 1 and
0.5, 0 = ADMIXTURE's rule), `--smallk=0|1` (register-blocked kernels for K = 2–7 instead of OpenBLAS; default 1),
`--hess-float=auto|0|1` (single-precision Newton Hessians in the main algorithm;
default auto: called genotypes with K ≥ 8) and `--max-iter=X`.

## Performance

Most of the time per iteration is spent in passes over all genotype entries. Each pass computes a tile of
H = P Qᵀ (BLAS), the per-entry likelihood terms, and the Hessians and gradients (BLAS).
* The matrix products are a small part of the time; the per-entry step dominates. It is written without
  branches on the genotype (they were mispredicted about every other entry), so the compiler vectorises it.
  It uses one division per entry and tracks the exponent of the likelihood product exactly instead of taking
  logs. GCC vectorises it only with `-fno-trapping-math` (always added by the Makefile): otherwise it turns
  selects such as `obs ? 2 - g : 0` into branches around the arithmetic, and without AVX-512's masked
  instructions the loop then stays scalar. Up to version 0.2.5 that happened on AVX2 CPUs: the flag made runs
  15–30 % faster there for K = 5–20 (M = 100,000, N = 2,000, 8 threads) with bit-identical results, and the
  per-entry step 1.6–2.1× faster. AVX-512 code was vectorised already (with masked instructions).
* The small Newton/QP problems per row of P and Q use a Cholesky-based active-set solver; rows of P (box
  constraints only) first try a primal-dual active-set method, which moves many bounds per iteration.
* **Small K (2–7):** OpenBLAS's matrix products are inefficient when the inner dimension is only K (or
  K(K+1)/2 for the Hessians). For these K the Newton passes use register-blocked kernels (`src/smallk.hpp`,
  compiled for each K) for H = P Qᵀ, the Hessians and the gradients, fused with the per-entry step in blocks of
  8–16 SNPs so that the intermediate tiles stay in the CPU caches. Same results to rounding; 9–22 % faster runs
  with 8 threads and 22–27 % with 44 (M = 100,000, N = 2,000, K = 3–7), also at N = 10,000 and N = 73, and
  10–37 % for genotype likelihoods. `--smallk=0` uses OpenBLAS instead. The portable binary uses them on CPUs
  with AVX2 and FMA.
* With 20,000 SNPs × 2,000 individuals and 8 threads, a run is 90–116× faster than ADMIXTURE 1.3.0 (K = 5–20).
  The October 2026 kernels made a run 1.7–3.2× faster than admixer 0.2.1 (M = 100k, N = 2k, K = 5–20,
  8 threads), with the same iterates.
* **Threads:** the passes scale up to the number of physical cores (3.0–3.7× from 8 to 44 threads at M = 100,000,
  N = 2,000); more threads than cores (hyper-threads) make runs slower, and the log says so. OpenBLAS libraries
  built for fewer threads than requested (the Ubuntu 22.04 package: 64) crash when more OpenMP threads call them,
  so `-j` is reduced to that limit, with a note in the log; the static release binary allows 256.
* Faster solvers for larger K were tried in October 2026 (warm-started QPs, reused Hessians, sparse Hessians
  for components with Q ≈ 0). They gave up to 15 % with few individuals and nothing at large N, so they were not
  adopted.

Measurements, profiles and what was tried: [`bench/perf_results.md`](bench/perf_results.md). Kernel profiler:
`make tools/prof_kernels; tools/prof_kernels data.bed K threads` (also for `.beagle.gz`).

## Prebuilt binary

`./build-static.sh` (needs podman or docker) builds `dist/admixer-linux-x86_64.tar.gz` in the
manylinux_2_28 container (glibc 2.28, GCC 14). OpenBLAS (with `DYNAMIC_ARCH`, so its kernels are chosen for
the CPU at run time) and zlib are built from source once into `.build/` and linked statically, as are
libstdc++, libgcc and libgomp; the script fails if the binary needs any shared library besides glibc.
The code is compiled for x86-64-v2, and the per-entry tile kernels (`tile_ll`, `tile_em`, `tile_wd` in
`io.hpp` and `beagle.hpp`) also for AVX2 and AVX-512 (`ADMIXER_KERNEL` in `kernel.hpp`, GCC `target_clones`); the
version is chosen when the program starts. Without the AVX-512 kernels the binary was about 2× slower on
an AVX-512 CPU; with them it is as fast as a `-march=native` build (20,000 SNPs, 2,000 individuals, 16
threads, Xeon Gold 6152: K = 5 2.7 s vs 2.7 s, K = 10 196 s vs 203 s, same log-likelihoods). On an AVX2 CPU
(Xeon E5-2699 v4, M = 100,000, N = 2,000, 8 threads) the AVX2 kernels and `-fno-trapping-math` (0.2.6) made it
1.7–2.1× faster than the 0.2.5 build, as fast as a `-march=native` build (K = 5 20.0 s vs 20.8 s, K = 10
31.9 s vs 32.3 s). To release:
run the script, run `tests/run_tests.sh --bin admixer`, and attach the tarball to a GitHub release.

## Testing

```
make test                                    # or: tests/run_tests.sh [--bin PATH] [-j N] [--update]
```

`tests/run_tests.sh` checks a build in about 15 seconds. Its exit code is the number of failed checks.
It uses the data in `tests/data`:
* `sim.bed`: 200 individuals, 4,000 SNPs, K = 3, with the true Q;
* two beagle files made from `sim.bed` with `tools/simgl`, at 3× depth and at mixed 0.5–6× depth per
  individual.

It checks:
* the unit tests (`tests/unit.cpp`): derivatives against finite differences for both data types, EM
  monotonicity, feasibility of the steps, certain GLs = genotypes (likelihood, gradient and evalAdmix), the parser
  and the Q matching;
* the command line and error handling;
* for both input types: the log-likelihood against the references in `tests/expected.tsv` (±0.01; higher is
  reported as IMPROVED), the optimality check, accuracy against the true Q, and the output formats;
* the evalAdmix correlations for GLs: ≈ 0 under the correct model, and the same pattern as from the true
  genotypes for a misfit (K = 2 for 3 populations, `sim2000.bed` with the GL fit's P);
* that the same seed gives the same Q, and that 1 thread reaches the same optimum;
* `--conv`, `--supervised`, `-P` (Q from a fixed P equals the joint fit), `--keep-missing` and `--hess=em`.

Iterations and times are compared with the references, and large increases give a warning, not a failure.
After an intended change of results, `tests/run_tests.sh --update` rewrites `tests/expected.tsv`.

`tools/simgl` (`make tools/simgl`) simulates reads from PLINK genotypes and writes beagle GLs. Each
individual gets a fixed depth, a depth drawn from a range (`--depth 0.5,6`), or a depth from a file. The
reads are Poisson, with a sequencing error rate.

## Differences from ADMIXTURE and NGSadmix

**ADMIXTURE:**
* Same model, likelihood, parameter bounds and stopping rule. Same algorithm, except that a rejected
  quasi-Newton extrapolation is retried once with half the step (`--qn-damp=0`: ADMIXTURE's rule), and for
  K ≥ 8 the Newton Hessians of the main loop are computed in single precision (`--hess-float=0`: double). Different
  start: random P, near-uniform Q, 5 EM steps, then a mini-batch warm-up.
* Not implemented: cross-validation (`--cv`), bootstrap standard errors (`-B`), penalised estimation
  (`-l`), haploid data, the EM method and the Fst printout. `-m` is admixer's maximum number of runs, not
  ADMIXTURE's method option.
* Default 8 threads instead of 1. The P matrix is gzip-compressed, the evalAdmix correlation of
  residuals is written by default, and the screen output is also saved to `data.K.log`.

**NGSadmix:**
* Same model, likelihood, filters (`--minMaf`, `--misTol`, `--minInd`) and bounds.
* A different algorithm (see Algorithms) and stopping rule.
* Missing GL entries are left out of the model by default (`--keep-missing` for NGSadmix's behaviour).
* The output uses ADMIXTURE's names (`.Q`, `.P.gz`) instead of `.qopt` and `.fopt.gz`, with the same
  content.
* Not implemented: `-minLrt`, `-dymBound`, `-printInfo`, and starting values from files.

## Plotting the example figures

The README figures were made with evalAdmix's [`visFuns.R`](https://github.com/GenisGE/evalAdmix).

Blue wildebeest:

```r
source("https://raw.githubusercontent.com/GenisGE/evalAdmix/master/visFuns.R")
pop <- read.table("blue_wildebeest_noLD.fam")[, 1]
q <- read.table("blue_wildebeest_noLD_admixerMult.7.Q")
ord <- orderInds(pop = pop, q = q)
plotAdmix(q, pop = pop, ord = ord, rotatelab = 15, padj = 0.15, cex.lab = 1.4, col = 2:8)
r <- as.matrix(read.table("blue_wildebeest_noLD_admixerMult.7.corres.txt"))
plotCorRes(r, pop = pop, ord = ord, max_z = 0.25, rotatelabpop = 20, adjlab = 0.05, title = "")
```

NGSadmix tutorial data (`pop.info` from the NGSadmix tutorial: population and individual per line, in the
order of the beagle file):

```r
source("https://raw.githubusercontent.com/GenisGE/evalAdmix/master/visFuns.R")
pop <- read.table("pop.info")[, 1]
q <- read.table("input.3.Q")
ord <- orderInds(pop = pop, q = q)
plotAdmix(q, pop = pop, ord = ord, cex.lab = 1.2, col = c("#1b9e77", "#d95f02", "#7570b3"))
r <- as.matrix(read.table("input.3.corres.txt"))
plotCorRes(r, pop = pop, ord = ord, max_z = 0.1, title = "")
```

Time per run against ADMIXTURE: `bench/plot_wildebeest_time.R`.
