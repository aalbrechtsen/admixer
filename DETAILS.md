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
5. **Damped extrapolation** (`--qn-damp`, default 2). ADMIXTURE discards an extrapolated point that does not beat
   F(x) and continues from F(F(x)). On hard, multimodal data this happens in 36–90 % of the iterations, and the
   run crawls. admixer first retries the extrapolation with half the step (`--qn-damp-factor`, default 0.5), then
   with a quarter, and discards it only if both fail (0.2.2–0.2.7: one retry). When no extrapolation is rejected
   (most data), the run is unchanged.
   * Hard simulated data (14 sets, M = 10,000–100,000, N = 600–2,000, K = 6–12, 5–10 seeds each): 1.2–4.2× faster
     where extrapolations were rejected (the more rejections, the larger the gain), identical where none were.
     Every seed reached the same optimum as without damping, or a higher one.
   * Genotype likelihoods (NGSadmix tutorial data, K = 3–6; simulated GLs, M = 10,000, N = 2,000): the same optimum
     for every seed, 11 % less time in total (per data set and K from 2 % slower to 1.8× faster).
   * The second retry (0.2.8): hard sets (12 sets at M = 10,000–20,000 and one at M = 400,000, N = 2,000, K = 8;
     5–6 seeds each) 20 % less time in total, best-optimum hits 32 → 32, the same optimum in 65 of 66 seeds; easy
     data and GLs (sim100k K = 5–20, wildebeest, NGSadmix tutorial K = 3–6, simulated GLs) the same optimum in 30
     of 30 runs and the same time. A third retry gained another 2 %.
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

## Split-merge search (`--search`)

On hard data every random start can end at its own local optimum, and more starts help little: on the reindeer
data below (N = 96, M = 321,005, K = 11), 48 random starts ended at 48 different optima, and the best of them
was 6,500 log-likelihood units below the best solution found. Two kinds of optima occur:
* **partitions:** which populations share a component. These optima are thousands of units apart;
* **single individuals:** an individual with in-between ancestry sits stably in one of two states. These are
  0.2–200 units apart.

`--search[=X]` starts from one random start, fitted loosely (`-C 10`), and then repeats rounds:
1. **Moves.** A move merges component b into component a, which frees column b, and splits another
   component c: the Q of c of half of c's main individuals (largest Q) moves to column b.
   * The pair (a, b) is one of the 4 pairs with the most similar P (mean squared difference d_ab), or one of
     the 4 with the smallest d_ab · m_a m_b / (m_a + m_b), where m is a component's total Q. The P of a
     component with little Q (an admixed group given a component of its own) comes from few individuals. Its
     noise, about 1/m_a + 1/m_b, puts it far from every other component by d_ab alone. On the data below,
     the merge two runs needed ranked 13th of 55 by d_ab and 3rd scaled.
   * The halves are the signs of the first principal component of those individuals' genotypes (GLs: the
     expected genotypes). It is computed by power iteration on at most 50,000 evenly spaced SNPs.
   * There are up to 8 × (K − 2) moves (4–8 pairs). Each round refits 2 untried ones, drawn with weights
     λ1/λ2 of the split component (the first two eigenvalues of its main individuals' centred genotypes, the
     second by deflation): a component whose individuals clearly fall into two groups is split first.
     `--search-order=random` draws them uniformly.
2. **Refit.** P is fitted to the moved Q by 20 EM steps with Q held fixed, from a random P. Then the fit runs
   as usual (mini-batch warm-up, then quasi-Newton) to `-C 10`. Starting from the moved Q without the warm-up
   took 4–5 times as long: the fit creeps along plateaus. After the warm-up, a refit is stopped if its
   log-likelihood is more than 0.1 % of |log L| below the current solution (`--search-abort-frac`), or if it is
   back at the current solution (max |ΔQ| < 0.01).
3. **Accept.** The better refit replaces the current solution if its log-likelihood is higher by more than
   0.01. Then the moves are rebuilt from the new solution.

The search stops when the budget is used, no move is left, or 6 rounds in a row accepted no move
(`--search-patience`). The best solution is then polished to `-C` (default
1e-4).

**Budget.** X counts standard runs (a random start fitted to `-C 1e-4`). Each step has a fixed charge, so a
run is reproducible for a given seed:
* the loose start: 0.75 (0.5 with `--Qin`);
* each refit: 0.6, or 0.35 if it is stopped after its warm-up.

The default, 20, allows 16 rounds (32 refits); with the patience stop most searches end earlier.

**Why moves are drawn at random (weighted).** No cheap score predicted the winning move. On the reindeer
data, the move that won after a full refit ranked 12th–30th of 36:
* by the log-likelihood after the first iterations;
* by the log-likelihood after fitting P alone.

The ranking turns late, after a plateau. The λ1/λ2 weights only make likely splits come earlier.

**Output.** `NAME.K.search` has one row per refit: round, the components merged (merge_into, merged) and split,
the number of individuals moved, the log-likelihood (at `-C 10`), the gain over the current solution, and
whether the move was accepted. The log has the same per refit and a summary.

`--Qin=FILE` (and `--Pin=FILE`) gives the start instead: P is fitted to Q by the same 20 EM steps unless
`--Pin` is given. Without `--search` the fit continues from it with no EM priming and no warm-up (unless
`--prime`/`--minibatch` are given). `--search` and `--Qin` cannot be combined with `-P`, `--supervised`,
`--conv` or `--from`.

**Reindeer data, K = 11**, plain search (without the early stop, patience stop and weighted move order that are now the default; steve, 8 threads per run, 10 seeds each). Each run is compared with the best
solution found (log L −15,042,991.78) by the largest difference in any Q entry, max |ΔQ|, after matching the
components. Runs agree when max |ΔQ| < 0.01, the rule of `--conv`. A standard run takes 23.1 s, data reading
included.

| | time (× standard run) | same solution (max \|ΔQ\| < 0.01) | max \|ΔQ\| ≤ 0.06 (2 individuals differ) | max \|ΔQ\| ≤ 0.31 (≤ 10 individuals differ) | another partition (max \|ΔQ\| = 1) |
|---|---|---|---|---|---|
| 10 standard runs | 1 each | 0 | 0 | 0 | 10 |
| `--search=10`¹ | 11.9 | 1 | 1 | 7 | 3 |
| `--search` (20), merge pairs by d_ab only² | 20.6 | 2 | 4 | 8 | 2 |
| `--search` (20) | 19.7 | 2 | 3 | 9 | 1 |

In log-likelihood terms:
* max |ΔQ| 0.058 is 26 units below the best;
* 0.15 is 33 or 219 units below;
* 0.29–0.31 is 280–590 units below;
* another partition is 23,000–29,000 below.

A small log-likelihood gap does not mean the same Q.

So most searches find the right partition. But only 2 of 10 also settle every individual: the remaining
differences are a few individuals with in-between ancestry in one of two states.

¹ Run with a refit charge of 0.5 (9 rounds), before the charge was set to 0.6 from these timings; with 0.6,
`--search=10` makes 7 rounds.

² The first version. Seeds 3 and 9 stopped where the needed merge was not proposed (see above); with the scaled
pairs they end 279 and 26 below the best.

On a 107,002-SNP subset of the same data, `--search=10` (seeds 3 and 7) reached the subset's best solution
(−5,011,936.11), in 62 s against 5.9 s for a standard run.

**Defaults** (`global_search/variants`, 27 October 2026). The early stop, the patience
stop and the weighted move order were tested on ten hard data sets, 10 seeds each, against the plain search
(none of the three). A run counts if it reaches the best solution found on that set (max |ΔQ| < 0.01). Time in
standard runs (median); reindeer on steve, the other sets on dylan (Xeon Gold 6152), 8 threads per run.

| data (N, M, K) | standard run | plain search | default | 4 lineages¹ |
|---|---|---|---|---|
| reindeer (96, 321,005, 8) | 5/10 | 10/10, 19.6× | 7/10, 7.6× | 10/10, 22.5× |
| reindeer (96, 321,005, 9) | 3/10 | 8/10, 19.2× | 8/10, 9.9× | 10/10, 21.5× |
| reindeer (96, 321,005, 10) | 0/10 | 9/10, 17.7× | 9/10, 10.0× | 10/10, 20.4× |
| reindeer (96, 321,005, 11) | 0/10 | 2/10, 19.4× | 0/10 (10/10 right partition), 14.1× | 0/10 (10/10 right partition), 21.6× |
| blue wildebeest r02 (73, 185,861, 7) | 2/10 | 10/10, 17.2× | 10/10, 7.5× | 10/10, 19.0× |
| blue wildebeest kenya2024 (73, 806,614, 7) | 8/10 | 10/10, 20.9× | 10/10, 7.7× | 10/10, 23.8× |
| simulated L2 (600, 10,000, 10) | 4/10 | 9/10, 7.4× | 9/10, 6.0× | 9/10, 8.8× |
| simulated S2, C2, S5, O8 (600–1000, 10,000–20,000, 8–12) | 0–1/10 | 0–1/10 | 0–1/10 | 0–1/10 |

So the default mostly reaches the best as often as the plain search, in about half the time on the real data.
The exception is reindeer K = 8: three searches started close to the best (67 units below, 11 individuals
different) and stopped after 6 rounds without an accepted move; the winning move came in rounds 8–11 with
`--search-patience=0` (10/10, 15.5×). If the time allows, `--search-patience=0` is safer.
Run alone, a search on 1000 Genomes chr20 (N = 2,590, M = 157,318, K = 10) took 25.5 standard runs: a refit
costs about 0.6 there too, but the P fits and split directions add some.

¹ Tested, not released: 4 searches from random starts sharing one budget by successive halving (all 4 share
half, the better 2 a quarter, the best one the rest). When the two best, which shared nothing, agreed
(max |ΔQ| < 0.01), they were at the best solution in 31 of 32 runs. `--conv X --search` (below) gives the same
kind of independent check with full searches.

On the simulated sets S2, C2, S5 and O8 (sister populations at Fst 0.01) the optima differ by tens to hundreds of
log-likelihood units within the same partition, and no setting helps. A fixed early-stop threshold (1,000 units)
stopped every refit there, also the winning ones (L2: 4/10), hence the threshold relative to |log L|. Refitting
the two moves of a round at once (threads split between them) was slower and was dropped.

**Convergence check: `--conv X --search`.** Each `--conv` run is a full search from its own random start (seeds
`--seed`, `--seed`+1, ...), and the searches share nothing. After each search the usual agreement test follows;
the run stops when X searches agree with the best (max |ΔQ| < `--conv_thres`), and the log says "Converged".
This is the same test as `--conv` with standard runs, only with searches.

Then the searches are mixed, whether they converged or not (since 0.2.13; in 0.2.12 only when they did not):
* a copy of the best search takes single individuals' states from each search it disagrees with: the
  individual's Q row from that search (columns matched), refitted to `-C 1`, and kept if the log-likelihood
  rises by more than 1 (cross-flips, the individuals with the largest differences first);
* this repeats over all searches while any flip is kept (at most 3 passes), and the mix is polished to `-C`;
* the searches themselves are not changed. The mix is one more result, the row `mix` in `NAME.K.conv`. It is
  written if it is another solution (max |ΔQ| ≥ `--conv_thres` to the best search) with a higher
  log-likelihood; otherwise the best search is written.

The status line ("Converged" / "Not converged") comes from the independent searches only. If they converged
but the mix is a better solution, the log says so ("But the mix of the N searches is a better solution ...")
and that the agreeing searches share a local optimum. The mix is the best found, not a converged solution; to
check it, run `--search` with other seeds and compare their Q with it. Mixing always runs because agreement can
be false: on reindeer K = 11, 3 independent searches agreed at −15,043,018.02, 26 below the best, which the
mixing reached in 208 s (table below). An earlier version let the runs take each other's states before every
agreement test: on reindeer K = 11 (`--search=10 --conv 2/3`), 20 of 20 replicates then reported "converged",
but only 2 were at the best solution, because the runs were made to agree.

Tests (reindeer, steve, 8 threads; standard run 22.5 s at K = 10 and 23.9 s at K = 11):

| | result | time |
|---|---|---|
| K = 10, `--conv 3 --search`, seeds 1 and 101 | converged after 3 and 4 searches, at the best solution (max \|ΔQ\| < 1e-4) | 490 s (22×) |
| K = 11, `--conv 3 --search`, seeds 1 and 101 | converged after 7 and 9 searches, at −15,043,018.02: 26 below the best, 2 individuals differ (max \|ΔQ\| 0.058) | 1,894–1,955 s (80×) |
| K = 11, `--conv 6 --search -m 6`, seed 1 (cannot converge) | not converged; the mix (1 flip kept, 208 s) is the best solution, −15,042,991.78 (max \|ΔQ\| 6e-6) | 1,928 s (81×) |

So agreement of independent searches is good evidence, not proof: at K = 11, three searches agreed on an
optimum 26 units below the best, which the mixing of the same kind of searches reaches; so since 0.2.13 the
searches are always mixed.

**Limits.**
* The search is local, and some optima cannot be left by any split-merge move. In the run that ended 23,402
  below the best (seed 1), two South Greenland individuals shared the component of the two Olrik_Fjord
  individuals. Moving those two individuals back to the South Greenland component leaves this optimum
  (+22,918 after the refit). But a split-merge move always merges two components, and splits only a third
  one, so it cannot do that. Use `--conv X --search` to see whether independent searches agree; if not, the
  mix of the searches keeps the best individual states found.
* For GLs the refits have no warm-up by default (`--minibatch=0`), so each costs more than the charge. The
  search runs and is tested on the test data, but its budget is not calibrated for GLs.

## SVD/ALS start (`--svd`)

`--svd` starts a run from a randomised SVD of the genotypes instead of a random start. It is ported from
admix_prog (`src/init.hpp`) and works as SCOPE and fastmixture do:
1. A rank K − 1 SVD of the centred genotypes, with 7 power iterations.
2. Alternating least squares for P and Q, with box (P) and simplex (Q) constraints solved row by row. The ALS
   stops when Q changes by less than 1e-6 (RMS) or after 500 iterations.
3. The fit then goes straight to the main algorithm, without EM priming or the mini-batch warm-up (unless
   `--prime`/`--minibatch` are given).

The random test matrix and the ALS start depend on the seed. GLs use the expected genotypes.

**Default.** Off; with `--conv` it is used for the first run only (the other runs start at random). A single
deterministic start rules out restarts (speed-up note 11), so it is one start among random ones.

Tests: 10 seeds each, 8 threads; compared with the best solution found by max |ΔQ| < 0.01.

| data | single random start | single `--svd` run | time of `--svd` (× standard run) |
|---|---|---|---|
| blue wildebeest popgen24, N = 73, M = 146,632, K = 7 | 8/10 | 10/10 | 1.5–2.3 (SVD/ALS 4–9 s) |
| reindeer, N = 96, M = 321,005, K = 11 | 0/10 | 0/10 (9 of 10 72,000–75,000 below) | 2.9–4.7 (SVD/ALS 14–53 s) |

* On the reindeer data the SVD start was no better than a random start. As the first run of
  `--svd --conv 3 -m 10` it ranked 2nd–7th of 10 runs, and no replicate converged.
* As the start of `--search` it reached the best in 1 of 10 runs (random start: 2 of 10), so the search
  keeps random starts unless `--svd` is given.
* For comparison: fastmixture 1.3.0, whose default start is the same kind of SVD/ALS, reached the
  wildebeest optimum in 10/10 runs at 8–16× an admixer standard run.

## Parental and paired ancestry

`src/parental.hpp`; run with `--parental`/`--paired` after the fit (and after evalAdmix), or with `--from` on the P and Q
files of an earlier run (rounded to 6 decimals, then projected onto the bounds), with P and Q fixed, so every individual is a separate
problem. The models are those of NGSremix (`-bothanc 1`). With u = x·fⱼ and v = y·fⱼ for the parents'
proportions x, y, the probability of the data at SNP j is L = G0(1−u)(1−v) + G1(u(1−v) + v(1−u)) + G2·uv, with
(G0, G1, G2) the genotype likelihoods (one-hot for called genotypes). NGSremix halves the heterozygote term of
the parental model; for called genotypes that is a constant, for genotype likelihoods it changes the model, and
admixer uses the term above. The per-individual log-likelihoods in the output files include the factor 2 of a
heterozygote (admixer's main log-likelihood for called genotypes omits it), so the three models' values are
comparable.

* **Parental fit.** x = y = q is a stationary point, and with equal parents the expected curvature in the
  direction that splits them is 0 (G0 − 2G1 + G2 summed over the genotypes is 0), so EM and Newton both crawl
  there (EM with SQUAREM needed up to 1,500 passes on NGSremix's PLINK example). admixer:
  1. one pass at x = y = q gives each individual's log-likelihood and Mc = Σⱼ cⱼ fⱼfⱼᵀ with
     c = (G0 − 2G1 + G2)/L; moving the parents apart along d (zero sum) changes log L by −t² dᵀMc d + O(t⁴);
  2. the most negative eigenvector of Mc on the zero-sum subspace of the ancestries with q ≥ 0.01 is the split
     direction; log L is evaluated at x = q ± t·d for 4 step lengths (4 passes);
  3. individuals whose split gains at least 10 log-units get damped Newton steps (Levenberg–Marquardt) on (x, y)
     jointly: exact gradient and Hessian (blocks Σ a²ffᵀ, Σ b²ffᵀ, Σ (ab − c)ffᵀ with a = ∂ℓ/∂u, b = ∂ℓ/∂v,
     formed by BLAS products over 256-SNP chunks), the smallest shift that makes the Hessian positive definite,
     and an active-set QP with one equality per parent; a step that lowers log L is retried with more damping
     from the stored derivatives (no extra pass); at most 30 passes, stop at a gain < 0.01;
  4. everyone else keeps x = y = q (gain 0), and nobody is reported below the ADMIXTURE log-likelihood.

  The passes put 64 individuals in SIMD lanes (parameters transposed), skip individuals that are done, and split
  the SNPs into chunks for the threads. Validation: on NGSremix's PLINK example (126 individuals, 104,290 SNPs,
  K = 2) all 9 individuals with a gain above 10 in a 3-start EM fit to convergence are found, with proportions
  within 0.0002, and they agree with NGSremix's within 0.0003. For the individuals with smaller gains admixer
  reports x = y = q, where NGSremix reports a weakly determined split (differences up to 0.19).
* **Paired fit** (`--paired`): log L is concave in π, so plain damped Newton from the ADMIXTURE point
  (π_aa = q_a², π_ab = 2q_a q_b) converges in 5–18 passes; pairs with an ancestry below 0.01 are held fixed.
  On NGSremix's GL example (6 individuals, K = 3) the estimates agree with NGSremix's within 0.002–0.02 and the
  log-likelihoods within 0.01, except where a pair with an ancestry below 0.01 is held fixed (0.3 log-units).
* **Cost** (8 threads, Xeon Gold 6152): see the README. The paired Hessian costs (K(K+1)/2)² per genotype.

## Pseudohaploid analysis (`--pseudohaploid`)

Each entry is one allele x ∈ {0, 1} (copies of the second allele) with P(x = 1) = h = Σₖ qᵢₖ pⱼₖ, so the
likelihood per entry is h^x (1 − h)^(1−x). The per-entry kernel is the diploid one with 1 in place of 2 in
(2 − g), and the same Newton/QP, EM and quasi-Newton steps follow; for 0/2-coded data the haploid
log-likelihood and gradient are exactly half of the diploid ones (unit test), so both give the same estimates.

Heterozygous calls of diploid input are turned into one allele by `het_allele` (`src/io.hpp`): bit 0 of the
splitmix64 hash of (SNP number, individual number), a fair coin that is independent between entries and the
same in every run, without a seed. Then P(x = 1) = h² + ½·2h(1 − h) = h, as the model assumes. A rule shared
between the individuals does not have this property. With "allele 1 at even, allele 2 at odd SNP numbers" every
heterozygote of a SNP gets the same allele, so P(x = 1) = h² or 1 − (1 − h)²: for an unadmixed individual that
is only another frequency, which P absorbs, but for an admixed one it differs from any mixture of the
ancestries' values by ±qₐq_b(pₐ − p_b)², so Q is biased. Alternating by SNP and individual (or per individual
from a random start) gives two groups of individuals whose heterozygotes always receive opposite alleles, and
the fit uses ancestries for those groups. On simulated data (M = 100,000, N = 2,000, K = 5, Fst 0.05, Dirichlet
0.5 admixture, 1,990 of 2,000 admixed; RMSE of Q against the truth):

| analysis | RMSE of Q |
|---|---|
| diploid genotypes | 0.020 |
| pseudohaploid, heterozygous calls by the hash (`--pseudohaploid`) | 0.029 |
| truly pseudohaploid data (one random allele per call, coded 0/2), `--pseudohaploid` or diploid model | 0.029 |
| heterozygous calls: allele 1 at even, allele 2 at odd SNP numbers | 0.042 |
| heterozygous calls: alternating by SNP + individual number | 0.774 (odd and even individuals split) |

evalAdmix needs the model variance of each individual's values: for genotypes it is estimated by the mean of
g(2 − g), which is 0 for a single allele, so pseudohaploid runs use the mean of (x − h)² with the fitted h, as
for genotype likelihoods. Parental and paired ancestry need both alleles and are not available.

## Advanced options

Mainly for benchmarking: `--bound=X` (P in [X, 1 − X] and Q ≥ X; default 1e-5 for genotypes as ADMIXTURE,
1e-9 for GLs as NGSadmix), `--prime=X` (EM steps before the main algorithm), `--minibatch=X` (initial number
of mini-batches of the warm-up), `--hess=exact|em` (curvature of the Newton steps for GLs), `--qn-damp=X` and
`--qn-damp-factor=F` (retries of a rejected quasi-Newton extrapolation with the step scaled by F; default 2 and
0.5, 0 = ADMIXTURE's rule), `--smallk=0|1` (register-blocked kernels for K = 2–7 instead of OpenBLAS; default 1),
`--hess-float=auto|0|1` (single-precision Newton Hessians in the main algorithm;
default auto: called genotypes with K ≥ 8) and `--max-iter=X`.

For the search (`--search`; tests in [Split-merge search](#split-merge-search---search)):
* `--search-patience=R`: stop after R rounds in a row without an accepted move (default 6; 0 = use the whole
  budget). About halves the time on real data; on reindeer K = 8 it cost 3 of 10 hits (7/10 against 10/10).
* `--search-abort-frac=F`: stop a refit after its warm-up if it is more than F · |log L| below the current
  solution, or back at it (max |ΔQ| < 0.01); such a refit is charged 0.35 standard runs (default 0.001; 0 = off).
* `--search-order=split|random`: draw the moves with weights λ1/λ2 of the split component (default `split`), or
  uniformly.

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
* **Warm-up batches:** the mini-batch warm-up calls the P and Q steps once per SNP batch (up to 32 per epoch).
  Their work buffers are kept between calls, and P tiles get smaller (down to 32 SNPs) when a batch has too
  few for the threads (0.2.7). Same results; at N = 50,000 or M = 500,000 with 44 threads the warm-up is up
  to 9 % and a run up to 6 % shorter.
* **Memory:** the genotypes are kept packed, 2 bits per entry as in the `.bed` file (0.2.11; before, one byte),
  and the SNP shuffle before the warm-up is done in place. At N = 50,000 and M = 100,000 (or N = 10,000 and
  M = 500,000), K = 10, the peak memory fell from 9.8 GB to 1.9–2.1 GB; at N = M = 500,000 it would be ~65 GB
  instead of ~500 GB. The per-entry step unpacks 4 genotypes per byte from a table, so the passes are not
  slower, and reading a `.bed` file is faster (no decoding, no second copy): those runs took 10–14 % less time
  with 44 threads (load 26 → 5 s, fit 3–5 % faster), and runs with M = 100,000, N = 2,000 the same time
  (8 threads). Results are bit-identical.
* Faster solvers for larger K were tried in October 2026 (warm-started QPs, reused Hessians, sparse Hessians
  for components with Q ≈ 0). They gave up to 15 % with few individuals and nothing at large N, so they were not
  adopted.

Measurements, profiles and what was tried: [`bench/perf_results.md`](bench/perf_results.md). Kernel profiler:
`make tools/prof_kernels; tools/prof_kernels data.bed K threads` (also for `.beagle.gz`).

## Prebuilt binary

The tarball is committed as `bin/admixer-linux-x86_64.tar.gz` (the README's download link); for a new version, copy
the new build there. `./build-static.sh` (needs podman or docker) builds `dist/admixer-linux-x86_64.tar.gz` in the
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
  quasi-Newton extrapolation is retried with half and then a quarter of the step (`--qn-damp=0`: ADMIXTURE's
  rule), and for K ≥ 8 the Newton Hessians of the main loop are computed in single precision (`--hess-float=0`:
  double). Different
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
