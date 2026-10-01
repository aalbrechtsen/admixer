# admixer

Fast maximum-likelihood estimation of ancestry proportions (Q) and ancestral allele frequencies (P) from

* **called genotypes** (PLINK `.bed`), under the ADMIXTURE model, with ADMIXTURE's command line, input and
  output; or
* **genotype likelihoods** (beagle files, as written by ANGSD), under the NGSadmix model, for
  low-depth sequencing data.

Both use ADMIXTURE's optimisation algorithm: block relaxation with Newton steps and quasi-Newton
acceleration, evaluated with BLAS matrix products. For genotype likelihoods, the Newton steps use the exact
curvature of the likelihood.
* **Genotypes:** it reaches the same likelihood as ADMIXTURE 1.3.0 and was 40–90× faster in our benchmarks
  (20,000 SNPs, 2,000 individuals, K = 5–20, 8 threads).
* **Genotype likelihoods:** on the NGSadmix tutorial data, it reached the best solution 8–84× faster than
  NGSadmix 32 (K = 3–6), and from more random starts.

## Install

Requires a C++17 compiler with OpenMP (e.g. g++), OpenBLAS and zlib. On Ubuntu/Debian:

```
sudo apt-get install build-essential libopenblas-dev zlib1g-dev
git clone git@github.com:aalbrechtsen/admixer.git
cd admixer
make
make test        # optional: the test suite, about 10 seconds
```

This builds the `admixer` binary in the current directory. `sudo make install` copies it to
`/usr/local/bin` (or `make install PREFIX=$HOME/.local`). To link a different BLAS, use for example
`make BLAS=-lblis`.

## Usage

```
admixer [options] data.bed K            # called genotypes: data.bed, data.bim, data.fam
admixer [options] data.beagle.gz K      # genotype likelihoods (any input not ending in .bed)
```

The output is written to the working directory, in ADMIXTURE's format, with the input name without its
extension (`data`) as prefix:

* `data.K.Q`: admixture proportions, one row per individual;
* `data.K.P.gz`: ancestral allele frequencies, one row per SNP (gzip-compressed; `--no-gzip` for `data.K.P`).
  For beagle input, the rows are the sites that pass the filters, in input order;
* `data.K.corres.txt`: the evalAdmix correlation of residuals between individuals (corrected estimator; for
  genotype likelihoods on the posterior expected genotypes), for checking the model fit; plot it with evalAdmix's `visFuns.R`. It takes a few seconds for
  thousands of individuals but needs about 48 N² bytes of memory, so it is skipped above 20,000 individuals
  unless `--evaladmix` is given;
* `data.K.log`: the log printed to the screen (command line, progress, final log-likelihood, optimality check).

| option | meaning |
|---|---|
| `-jX`, `-j X` | number of threads (default 8) |
| `--seed=X`, `-s X` | random seed (default 43; `time` uses the clock) |
| `-C=X`, `-C X` | stop when the log-likelihood improves by less than X (default 1e-4) |
| `-o NAME` | output prefix `NAME` instead of `data` |
| `--no-gzip` | write the P matrix uncompressed (`NAME.K.P`) |
| `--supervised` | supervised analysis; reads `data.pop` (one line per individual: a population name, or `-` if unknown). Labelled individuals are held at their population; K must equal the number of population names, and columns follow the order of first appearance in `data.pop` |
| `-P` | projection; reads `data.K.P.in` (or `data.K.P.in.gz`) and estimates Q with P held fixed |
| `--conv X` | convergence test with several starts (see below): stop when X runs agree with the best run |
| `-m X`, `--max_runs=X` | with `--conv`: at most X runs (default 10) |
| `--conv_thres=X` | with `--conv`: runs agree if the largest difference in any Q entry is below X (default 0.01) |
| `--no-evaladmix` | do not compute the evalAdmix correlation of residuals |
| `--evaladmix` | compute it even for more than 20,000 individuals |
| `--minMaf=X` | beagle input: keep sites with X < MAF < 1 − X (default 0.05, as NGSadmix; 0 keeps all) |
| `--misTol=X` | beagle input: GLs with max − min < X count as missing (default 0.05, as NGSadmix) |
| `--minInd=X` | beagle input: keep sites with more than X individuals with data (default 0 = off) |
| `--keep-missing` | beagle input: use the missing GL entries as data (as NGSadmix's EM) instead of leaving them out |

Advanced options, mainly for benchmarking: `--bound=X` (P in [X, 1 − X] and Q ≥ X; default 1e-5 for
genotypes as ADMIXTURE, 1e-9 for GLs as NGSadmix), `--prime=X` (EM steps before the main algorithm),
`--minibatch=X` (initial number of mini-batches of the warm-up), `--hess=exact|em` (curvature of the Newton
steps for GLs) and `--max-iter=X`.

Examples: `admixer -j16 data.bed 5`, `admixer -j16 input.beagle.gz 3`

### Convergence test with several starts

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

## Algorithms

**Called genotypes (ADMIXTURE model).** g_ij ~ Binomial(2, h_ij) with h_ij = Σ_k Q_ik P_jk.
1. The fit starts from a random P and a near-uniform Q.
2. It takes 5 EM steps.
3. A mini-batch warm-up follows: block-relaxation steps on SNP batches. The number of batches is halved
   whenever an epoch does not improve the likelihood.
4. Then it runs ADMIXTURE's block relaxation until the log-likelihood improves by less than `-C`. Each
   iteration takes one Newton/QP step for every row of P and then for every row of Q, accelerated by
   quasi-Newton extrapolation with 3 secant pairs.

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

## Example: blue wildebeest (PLINK), K = 7

This example follows the popgenDK exercise
[Admixture proportions from called genotypes: blue wildebeest](https://github.com/popgenDK/courses/blob/main/current_exercises/admixture/admixture_called_genotypes_animal.ipynb).

**The data.** 73 blue wildebeest from seven sampling localities; the locality is the first column of the
`.fam` file.

**How the input was made.** The input `blue_wildebeest_noLD` is made in the exercise from 990,980 called SNPs
(`blue_wildebeest_thin`). The SNPs are LD-pruned with PCAone, which corrects for population structure:

```
PCAone -b blue_wildebeest_thin -k 6 -D -o pcaone
PCAone -B pcaone.residuals -F pcaone.mbim --ld-r2 0.1 --ld-bp 1000000 -o pcaone
plink --bfile blue_wildebeest_thin --extract pcaone.ld.prune.in --make-bed --out blue_wildebeest_noLD --chr-set 29
```

This leaves 37,567 SNPs. The data are not included in this repository.

### One run

```
admixer --seed 1 -j10 -o blue_wildebeest_noLD_admixer blue_wildebeest_noLD.bed 7
```

<details>
<summary>Full screen output = log file <code>blue_wildebeest_noLD_admixer.7.log</code> (click to expand)</summary>

```
admixer 0.2.0
Command: admixer --seed 1 -j10 -o blue_wildebeest_noLD_admixer blue_wildebeest_noLD.bed 7
Random seed: 1
Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)
Convergence acceleration algorithm: QuasiNewton, 3 secant conditions
Point estimation will terminate when objective function delta < 0.0001
Size of G: 73x37567
Threads: 10
Performing 5 EM steps to prime main algorithm
1 (EM) 	Elapsed: 0.020	Loglikelihood: -3905731.572719	(delta): inf
2 (EM) 	Elapsed: 0.033	Loglikelihood: -3589909.043398	(delta): 315823
3 (EM) 	Elapsed: 0.046	Loglikelihood: -3575200.604387	(delta): 14708.4
4 (EM) 	Elapsed: 0.059	Loglikelihood: -3574247.667174	(delta): 952.937
5 (EM) 	Elapsed: 0.066	Loglikelihood: -3574138.765123	(delta): 108.902
Initial loglikelihood: -3574096.334450
1 (mini-batch, 32 batches) 	Elapsed: 0.106	Loglikelihood: -3305128.521990	(delta): 268968
2 (mini-batch, 32 batches) 	Elapsed: 0.140	Loglikelihood: -3093758.676583	(delta): 211370
3 (mini-batch, 32 batches) 	Elapsed: 0.172	Loglikelihood: -3020493.145233	(delta): 73265.5
4 (mini-batch, 32 batches) 	Elapsed: 0.204	Loglikelihood: -3005025.468846	(delta): 15467.7
5 (mini-batch, 32 batches) 	Elapsed: 0.235	Loglikelihood: -2983492.604547	(delta): 21532.9
6 (mini-batch, 32 batches) 	Elapsed: 0.266	Loglikelihood: -2970637.518189	(delta): 12855.1
7 (mini-batch, 32 batches) 	Elapsed: 0.297	Loglikelihood: -2963043.082693	(delta): 7594.44
8 (mini-batch, 32 batches) 	Elapsed: 0.328	Loglikelihood: -2955627.401733	(delta): 7415.68
9 (mini-batch, 32 batches) 	Elapsed: 0.358	Loglikelihood: -2949074.319921	(delta): 6553.08
10 (mini-batch, 32 batches) 	Elapsed: 0.389	Loglikelihood: -2945153.291887	(delta): 3921.03
11 (mini-batch, 32 batches) 	Elapsed: 0.420	Loglikelihood: -2941630.583157	(delta): 3522.71
12 (mini-batch, 32 batches) 	Elapsed: 0.450	Loglikelihood: -2940097.110490	(delta): 1533.47
13 (mini-batch, 32 batches) 	Elapsed: 0.481	Loglikelihood: -2937886.236820	(delta): 2210.87
14 (mini-batch, 32 batches) 	Elapsed: 0.511	Loglikelihood: -2936857.473163	(delta): 1028.76
15 (mini-batch, 32 batches) 	Elapsed: 0.542	Loglikelihood: -2936705.528617	(delta): 151.945
16 (mini-batch, 32 batches) 	Elapsed: 0.573	Loglikelihood: -2936839.315741	(delta): -133.787
17 (mini-batch, 16 batches) 	Elapsed: 0.595	Loglikelihood: -2934505.434620	(delta): 2333.88
18 (mini-batch, 16 batches) 	Elapsed: 0.617	Loglikelihood: -2934593.456450	(delta): -88.0218
19 (mini-batch, 8 batches) 	Elapsed: 0.637	Loglikelihood: -2933738.627437	(delta): 854.829
20 (mini-batch, 8 batches) 	Elapsed: 0.658	Loglikelihood: -2933552.673171	(delta): 185.954
21 (mini-batch, 8 batches) 	Elapsed: 0.679	Loglikelihood: -2933570.207690	(delta): -17.5345
22 (mini-batch, 4 batches) 	Elapsed: 0.698	Loglikelihood: -2933229.690093	(delta): 340.518
23 (mini-batch, 4 batches) 	Elapsed: 0.717	Loglikelihood: -2933135.647354	(delta): 94.0427
24 (mini-batch, 4 batches) 	Elapsed: 0.736	Loglikelihood: -2933145.977744	(delta): -10.3304
25 (mini-batch, 2 batches) 	Elapsed: 0.755	Loglikelihood: -2933015.506707	(delta): 130.471
26 (mini-batch, 2 batches) 	Elapsed: 0.773	Loglikelihood: -2933009.051346	(delta): 6.45536
27 (mini-batch, 2 batches) 	Elapsed: 0.791	Loglikelihood: -2933005.262884	(delta): 3.78846
28 (mini-batch, 2 batches) 	Elapsed: 0.809	Loglikelihood: -2933001.307338	(delta): 3.95555
29 (mini-batch, 2 batches) 	Elapsed: 0.827	Loglikelihood: -2933000.262047	(delta): 1.04529
30 (mini-batch, 2 batches) 	Elapsed: 0.845	Loglikelihood: -2932999.566725	(delta): 0.695322
31 (mini-batch, 2 batches) 	Elapsed: 0.863	Loglikelihood: -2932999.375594	(delta): 0.191131
32 (mini-batch, 2 batches) 	Elapsed: 0.882	Loglikelihood: -2932998.737526	(delta): 0.638068
33 (mini-batch, 2 batches) 	Elapsed: 0.900	Loglikelihood: -2932998.786367	(delta): -0.0488406
Starting main algorithm
1 (QN/Block) 	Elapsed: 0.940	Loglikelihood: -2932965.459753	(delta): inf
2 (QN/Block) 	Elapsed: 0.977	Loglikelihood: -2932963.623793	(delta): 1.83596
3 (QN/Block) 	Elapsed: 1.022	Loglikelihood: -2932963.275774	(delta): 0.348019
4 (QN/Block) 	Elapsed: 1.062	Loglikelihood: -2932963.129045	(delta): 0.146729
5 (QN/Block) 	Elapsed: 1.098	Loglikelihood: -2932963.106727	(delta): 0.0223178
6 (QN/Block) 	Elapsed: 1.134	Loglikelihood: -2932963.079817	(delta): 0.0269099
7 (QN/Block) 	Elapsed: 1.171	Loglikelihood: -2932963.073994	(delta): 0.005823
8 (QN/Block) 	Elapsed: 1.207	Loglikelihood: -2932963.071934	(delta): 0.00205984
9 (QN/Block) 	Elapsed: 1.243	Loglikelihood: -2932963.070470	(delta): 0.00146464
10 (QN/Block) 	Elapsed: 1.279	Loglikelihood: -2932963.069341	(delta): 0.00112899
11 (QN/Block) 	Elapsed: 1.315	Loglikelihood: -2932963.069302	(delta): 3.84306e-05
Summary: 
Converged in 11 iterations (1.381 sec)
Loglikelihood: -2932963.069302
Optimality check (max projected gradient): P 7.78e-05, Q 2.34e-08
Writing output files.
evalAdmix correlation of residuals written to blue_wildebeest_noLD_admixer.7.corres.txt (0.08 sec)
Log written to blue_wildebeest_noLD_admixer.7.log
```

</details>

This run takes 1.4 seconds and ends at log-likelihood −2932963.07. It writes these files:

| file | size | content |
|---|---|---|
| `blue_wildebeest_noLD_admixer.7.Q` | 4.5 kB | admixture proportions, 73 rows × 7 columns (as ADMIXTURE) |
| `blue_wildebeest_noLD_admixer.7.P.gz` | 0.9 MB | ancestral allele frequencies, 37,567 rows × 7 columns, gzipped (`--no-gzip` for plain text) |
| `blue_wildebeest_noLD_admixer.7.corres.txt` | 50 kB | **evalAdmix correlation of residuals**, 73 × 73, `NA` on the diagonal |
| `blue_wildebeest_noLD_admixer.7.log` | 5 kB | the screen output: command, settings, every iteration, final log-likelihood, optimality check |

The evalAdmix correlations are computed by default, in 0.1 s here. You do not need to run evalAdmix
separately; the file is read directly by evalAdmix's `plotCorRes` (see below).

![Admixture proportions, seed 1](docs/blue_wildebeest_noLD_admixer.admix.png)

The fit looks wrong:
- the W-Serengeti samples are split between two clusters;
- the three C-Luangwa samples all appear admixed in the same proportions.

The evalAdmix correlation of residuals (`blue_wildebeest_noLD_admixer.7.corres.txt`, written by default)
confirms it. There are strong positive correlations within C-Luangwa and N-Selous, and negative correlations
between them and B-Ethosha.

![evalAdmix, seed 1](docs/blue_wildebeest_noLD_admixer.evaladmix.png)

### Several runs with a convergence test

```
admixer --seed 0 -j10 -o blue_wildebeest_noLD_admixerMult blue_wildebeest_noLD.bed 7 --conv 3
```

This tries seeds 0, 1, 2, ... until three runs agree with the best run, in 11 seconds.
`blue_wildebeest_noLD_admixerMult.7.conv` lists the runs, best first:

```
run  seed  loglik           loglik_diff    max_abs_dQ   ...  agrees
9    8     -2906767.495807   0.000000      0                 1
1    0     -2906767.495808  -0.000001      9.53674e-07       1
6    5     -2906767.495820  -0.000013      2.5034e-05        1
8    7     -2932625.706040  -25858.210232  0.99993           0
2    1     -2932817.816830  -26050.321022  0.99993           0
...
```

Three of the nine runs reach the same solution, which is 26,000 log-likelihood units better than the seed-1
run. The output files are those of the best run (seed 8):
- `blue_wildebeest_noLD_admixerMult.7.Q`, `.7.P.gz` and `.7.corres.txt` (evalAdmix computed for the best run);
- `.7.log`, with one summary line per run;
- `blue_wildebeest_noLD_admixerMult.7.conv`, the table above.

![Admixture proportions, best of several runs](docs/blue_wildebeest_noLD_admixerMult.admix.png)

Each locality now has its own ancestry, and the residual correlations are close to zero.

![evalAdmix, best of several runs](docs/blue_wildebeest_noLD_admixerMult.evaladmix.png)

The plots were made with evalAdmix's [`visFuns.R`](https://github.com/GenisGE/evalAdmix):

```r
source("https://raw.githubusercontent.com/GenisGE/evalAdmix/master/visFuns.R")
pop <- read.table("blue_wildebeest_noLD.fam")[, 1]
q <- read.table("blue_wildebeest_noLD_admixerMult.7.Q")
ord <- orderInds(pop = pop, q = q)
plotAdmix(q, pop = pop, ord = ord, rotatelab = 15, padj = 0.15, cex.lab = 1.4, col = 2:8)
r <- as.matrix(read.table("blue_wildebeest_noLD_admixerMult.7.corres.txt"))
plotCorRes(r, pop = pop, ord = ord, max_z = 0.25, rotatelabpop = 20, adjlab = 0.05, title = "")
```

### Comparison with ADMIXTURE

Ten runs of each program on the same data (seeds 1–10, K = 7, 8 threads, one run at a time; admixer 0.1.0,
which gives the same results as 0.2.0):

```
admixture -j8 -s $seed blue_wildebeest_noLD.bed 7
admixer -j8 -s $seed blue_wildebeest_noLD.bed 7
```

![Log-likelihood of 10 runs, ADMIXTURE vs admixer](docs/wildebeest_loglik_admixture_vs_admixer.png)

Both programs reach the best likelihood in 5 of the 10 runs. The other runs end in local optima 10,000–27,000
units lower. Over 30 seeds the counts were 18/30 for admixer and 13/30 for ADMIXTURE.

The best runs agree:
- admixer: −2906767.4958 to −2906767.4958;
- ADMIXTURE: −2906767.503 to −2906767.656.

ADMIXTURE's runs are a little lower because its stopping rule ends them slightly before the optimum.

![Time per run, ADMIXTURE vs admixer](docs/wildebeest_time_admixture_vs_admixer.png)

The median time per run was 3.2 s for admixer 0.1.0 and 53 s for ADMIXTURE, 17× faster (admixer 0.2.0 is
another ~30% faster on these data). So `--conv 3` with admixer (11 s above) takes much less time than a single
ADMIXTURE run.

## Example: genotype likelihoods, NGSadmix tutorial data, K = 3

The NGSadmix tutorial file `input.gz` holds beagle GLs for 100 HapMap individuals (ASW, CEU, CHB, MXL, YRI)
at 50,000 sites.

```
admixer --seed 1 -j10 input.gz 3
```

The MAF filter keeps 49,475 sites, and 3.3 % of the GL entries are missing. The end of the log:

```
26 (QN/Block) 	Elapsed: 1.221	Loglikelihood: -3865964.313412	(delta): 1.29216e-05
Summary: 
Converged in 26 iterations (2.112 sec)
Loglikelihood: -3865964.313412
Loglikelihood over all GL entries, missing ones included (as NGSadmix): -3865964.313411
Optimality check (max projected gradient): P 2.93e-05, Q 3.12e-11
Writing output files.
evalAdmix correlation of residuals written to input.3.corres.txt (0.25 sec)
Log written to input.3.log
```

With 8 threads, NGSadmix 32 (`NGSadmix -likes input.gz -K 3 -P 8 -seed 1`) needs 205 iterations and 15 s on
the same data. It stops at −3865964.43, 0.12 below the optimum.

In a benchmark with 25 random starts per K (8 threads, run with the same algorithm in the standalone ngsadmix
development code), the expected time to reach the best solution
was:

| K | admixer | NGSadmix 32 | starts reaching the best solution (admixer / NGSadmix) |
|---|---|---|---|
| 3 | 2.5 s | 20 s | 100 % / 100 % |
| 4 | 3.1 s | 85 s | 100 % / 48 % |
| 5 | 6.3 s | 522 s | 64 % / 16 % |
| 6 | 10.5 s | 883 s | 64 % / 12 % |

## Testing

```
make test                                    # or: tests/run_tests.sh [--bin PATH] [-j N] [--update]
```

`tests/run_tests.sh` checks a build in about 10 seconds. Its exit code is the number of failed checks.
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
* Same model, likelihood, parameter bounds, algorithm and stopping rule. Different start: random P,
  near-uniform Q, 5 EM steps, then a mini-batch warm-up.
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

## Citation

For the ADMIXTURE model and algorithm: Alexander, Novembre & Lange (2009) *Genome Research* 19:1655–1664.
For the genotype likelihood model: Skotte, Korneliussen & Albrechtsen (2013) *Genetics* 195:693–702.
For the evalAdmix output: Garcia-Erill & Albrechtsen (2020) *Molecular Ecology Resources* 20:936–949, and
van Waaij et al. (2023) *Genetics* 225:iyad157.
