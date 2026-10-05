# admixer

Fast maximum-likelihood estimation of ancestry proportions (Q) and ancestral allele frequencies (P) from

* **called genotypes** (PLINK `.bed`), under the ADMIXTURE model, with ADMIXTURE's command line, input and
  output; or
* **genotype likelihoods** (beagle files, as written by ANGSD), under the NGSadmix model, for
  low-depth sequencing data.

Both use ADMIXTURE's optimisation algorithm: block relaxation with Newton steps and quasi-Newton
acceleration, evaluated with BLAS matrix products. For genotype likelihoods, the Newton steps use the exact
curvature of the likelihood.
* **Genotypes:** it reaches the same likelihood as ADMIXTURE 1.3.0 and was 90–116× faster in our benchmarks
  (20,000 SNPs, 2,000 individuals, K = 5–20, 8 threads; 127–235× with 64 threads; Xeon Gold 6152 server).
* **Genotype likelihoods:** on the NGSadmix tutorial data, it reached the best solution 12–118× faster than
  NGSadmix 32 (K = 3–6), and from more random starts.

Details beyond everyday use (algorithms, performance, testing, differences from ADMIXTURE and NGSadmix) are in
[DETAILS.md](DETAILS.md).

## Install

### Prebuilt binary (Linux x86_64)

Nothing else needs to be installed: OpenBLAS, zlib and the C++/OpenMP runtimes are built in. It runs on any
x86_64 Linux with glibc 2.28 or newer (RHEL/Rocky/Alma 8+, Ubuntu 20.04+, Debian 10+), and uses AVX2 or
AVX-512 when the CPU has it. The binary in `bin/` is the current version (0.2.10).

```
wget https://github.com/aalbrechtsen/admixer/raw/main/bin/admixer-linux-x86_64.tar.gz
tar xzf admixer-linux-x86_64.tar.gz
./admixer
```

### From source

Requires a C++17 compiler with OpenMP (e.g. g++), OpenBLAS and zlib. On Ubuntu/Debian:

```
sudo apt-get install build-essential libopenblas-dev zlib1g-dev
git clone https://github.com/aalbrechtsen/admixer.git
cd admixer
make
make test        # optional: the test suite, about 15 seconds (see DETAILS.md)
```

<details>
<summary>On a cluster with environment modules (<code>module load</code>)</summary>

Load a compiler and OpenBLAS before running `make` (`module avail gcc openblas` lists what is installed):

```
git clone https://github.com/aalbrechtsen/admixer.git
cd admixer
module load gcc openblas
make
```

If `make` stops with `cblas.h: No such file or directory`, OpenBLAS is not loaded; with
`zlib.h: No such file or directory`, also run `module load zlib`. The binary links OpenBLAS from the module,
so run `module load openblas` before using it (otherwise it fails with `error while loading shared
libraries: libopenblas.so.0`), or use the prebuilt binary above, which needs no modules.

</details>

This builds the `admixer` binary in the current directory. `sudo make install` copies it to
`/usr/local/bin` (or `make install PREFIX=$HOME/.local`). To link a different BLAS, use for example
`make BLAS=-lblis`. The prebuilt binary is made with `./build-static.sh` (see DETAILS.md).

## Usage

A worked example with figures, from the fit to the parental and paired ancestry of a finished run (1000 Genomes
chr20): [docs/USAGE.md](docs/USAGE.md).

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
* with `--parental`: `data.K.parental`, the admixture proportions of each individual's two parents (see
  [Parental and paired ancestry](#parental-and-paired-ancestry));
* `data.K.log`: the log printed to the screen (command line, progress, final log-likelihood, optimality check).

| option | meaning |
|---|---|
| `-jX`, `-j X` | number of threads (default 8) |
| `--seed=X`, `-s X` | random seed (default 43; `time` uses the clock) |
| `-C=X`, `-C X` | stop when the log-likelihood improves by less than X (default 1e-4) |
| `-o NAME` | output prefix `NAME` instead of `data` |
| `--no-gzip` | write the P matrix uncompressed (`NAME.K.P`) |
| `--no-P` | do not write the P matrix (e.g. for benchmarks) |
| `--supervised` | supervised analysis; reads `data.pop` (one line per individual: a population name, or `-` if unknown). Labelled individuals are held at their population; K must equal the number of population names, and columns follow the order of first appearance in `data.pop` |
| `-P` | projection; reads `data.K.P.in` (or `data.K.P.in.gz`) and estimates Q with P held fixed |
| `--conv X` | convergence test with several starts (see below): stop when X runs agree with the best run |
| `-m X`, `--max_runs=X` | with `--conv`: at most X runs (default 10) |
| `--conv_thres=X` | with `--conv`: runs agree if the largest difference in any Q entry is below X (default 0.01) |
| `--no-evaladmix` | do not compute the evalAdmix correlation of residuals |
| `--evaladmix` | compute it even for more than 20,000 individuals |
| `--parental` | also estimate the parents' admixture proportions (`data.K.parental`) |
| `--paired` | also estimate the paired ancestries (`data.K.paired`; cost grows as K⁴) |
| `--from=PREFIX` | no fit: read `PREFIX.K.P.gz` (or `.P`) and `PREFIX.K.Q` of an earlier run and only write `NAME.K.parental` (and `.paired` with `--paired`); the log goes to `NAME.K.from.log` |
| `--minMaf=X` | beagle input: keep sites with X < MAF < 1 − X (default 0.05, as NGSadmix; 0 keeps all) |
| `--misTol=X` | beagle input: GLs with max − min < X count as missing (default 0.05, as NGSadmix) |
| `--minInd=X` | beagle input: keep sites with more than X individuals with data (default 0 = off) |
| `--keep-missing` | beagle input: use the missing GL entries as data (as NGSadmix's EM) instead of leaving them out |

Advanced options for benchmarking (`--bound`, `--prime`, `--minibatch`, `--hess`, `--max-iter`): see
[DETAILS.md](DETAILS.md#advanced-options).

Examples: `admixer -j16 data.bed 5`, `admixer -j16 input.beagle.gz 3`

### Convergence test with several starts

Hard data sets can have several local optima, so one run may not find the best solution. `--conv X` runs
admixer with consecutive seeds (`--seed`, `--seed`+1, ...), reading the data only once, until X runs agree with
the best run (largest difference in any Q entry below `--conv_thres`, after matching the ancestries), or at
most `--max_runs` runs. The output files are those of the best run, and `data.K.conv` lists every run. Details:
[DETAILS.md](DETAILS.md#convergence-test---conv).

Example: `admixer --seed=30 --conv 3 data.bed 8` uses seeds 30, 31, 32, ... until 3 runs agree
(at most 10 runs).

### Parental and paired ancestry

A walk-through with figures is in [docs/USAGE.md](docs/USAGE.md#4-parental-admixture-from-the-finished-run).

admixer can estimate, for each individual and with P held fixed, the admixture proportions of its two
parents (the parental model of NGSremix, Nøhr et al. 2021). The two alleles at a SNP come one from each parent,
whose ancestries are drawn from the parents' proportions x and y; x = y = Q is the ADMIXTURE model, and an
individual whose parents differ in ancestry (e.g. a first-generation admixed individual) has x ≠ y.
`data.K.parental` has a header and one line per individual: x (`parent1_1` … `parent1_K`), y (`parent2_1` …),
then the log-likelihood of the parental model and of the ADMIXTURE model for that individual. The gain (the
difference of the two) measures the evidence that the parents differ: without a difference, twice the gain is
roughly χ² with K − 1 degrees of freedom. Gains below 10 are reported as 0, with x = y = Q. The order of the two
parents is arbitrary (parent 1 is the lexicographically larger vector). Ancestries with Q < 0.01 are held at their
value. The SNPs are assumed independent: without LD pruning the gains are inflated.

Which parent an allele came from is never observed, so the likelihood is nearly flat in the direction that pulls
two parents with the same ancestry apart. admixer therefore checks every individual at x = y = Q first (one pass
over the data gives the curvature in every direction that splits the parents), searches along the most promising
direction, and fits (damped Newton steps) only the individuals where the split pays. This would add 5–26 % to the
run time, so it is not done by default: add `--parental` to a run, or compute it later from the P and Q of a
finished run with `--from` (same input file and filters as that run; no fit is done):

```
admixer data.bed 5                          # the fit: data.5.Q, data.5.P.gz
admixer data.bed 5 --from data              # reads them, writes data.5.parental (log: data.5.from.log)
admixer data.bed 5 --from data --paired     # paired ancestry only: data.5.paired
admixer data.bed 5 --from data --parental --paired   # both
```

Costs (8 threads: 1000 Genomes chr20, 2,590 individuals × 157,318 SNPs, K = 5: 5.9 s after a 34 s fit, or 7.9 s in all with `--from`; K = 10:
18 s after 70 s; simulated data without parental differences, 2,000 × 100,000, K = 5 and 10: 1.2–1.3 s). On the
1000 Genomes data the recently admixed populations stand out (K = 5, individuals with a gain above 10, all
SNPs: CLM 41/97, PUR 47/104, MXL 33/65, PEL 30/87, ASW 22/61, ACB 25/96; YRI 1/120, ESN 0/106, MSL 0/87). On
LD-pruned SNPs (28,426) only these populations remain (ASW 9, CLM 8, PEL 7, MXL 6, PUR 6, ACB 4; one CHB); without
pruning, every population has some individuals above 10, so prune first.

With `--paired`, `data.K.paired` gives for each individual the probabilities of the K(K+1)/2 unordered pairs of
ancestries of the two alleles at a SNP (`pair_a_b`, a ≤ b; NGSremix's paired ancestry, concave, fitted by Newton
steps), with the log-likelihoods. It costs K(K+1)/2 squared per genotype, so it is only run with `--paired` (same
1000 Genomes data: 13 s at K = 5, 205 s at K = 10). Ancestries with Q < 0.01 are left out of the pairs.

## How it works

* **Called genotypes:** ADMIXTURE's model and algorithm. From a random P and near-uniform Q, it takes 5 EM steps,
  then a mini-batch warm-up on SNP batches. Then it runs block relaxation with one Newton/QP step per row of P
  and of Q, accelerated by quasi-Newton extrapolation; an extrapolation that does not improve the likelihood is
  retried with half and then a quarter of the step before it is discarded. For K ≥ 8 the Newton Hessians of the
  main loop are computed in single precision, which changes the iterates only at ~1e-6 and not the optimum.
  ADMIXTURE's bounds and stopping rule apply.
* **Genotype likelihoods:** NGSadmix's model, with the same block relaxation, using the exact curvature in the
  Newton steps. NGSadmix's filters and bounds apply, and missing GLs are left out.
* **Implementation:** every pass works on tiles of SNPs × individuals with BLAS matrix products and vectorised
  per-entry kernels, in parallel on OpenMP threads. The same seed and number of threads give identical results.
* The log ends with an optimality check (the largest projected-gradient step; 0 at a local optimum).

Details: [DETAILS.md](DETAILS.md) (algorithms, performance, testing, and the differences from ADMIXTURE and NGSadmix).

## Example: blue wildebeest (PLINK), K = 7

The times in both examples were measured with admixer 0.2.6 (0.2.8 gives the same times) on one server
(2 × Xeon E5-2699 v4, 44 cores, AVX2); the GL benchmark table with 0.2.4, which gives the same times as 0.2.6
and 0.2.8 there. The ADMIXTURE and NGSadmix runs used for the comparisons, from this server and an earlier
one, are listed in [`bench/reference_runs/`](bench/reference_runs/).

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
admixer 0.2.8
Command: admixer --seed 1 -j10 -o blue_wildebeest_noLD_admixer blue_wildebeest_noLD.bed 7
Random seed: 1
Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)
Convergence acceleration algorithm: QuasiNewton, 3 secant conditions, rejected extrapolations retried 2x with the step scaled by 0.5
Newton steps: register-blocked kernels for K = 7 instead of OpenBLAS (--smallk=0: OpenBLAS)
Point estimation will terminate when objective function delta < 0.0001
Size of G: 73x37567
Threads: 10
Performing 5 EM steps to prime main algorithm
1 (EM) 	Elapsed: 0.018	Loglikelihood: -3905731.572719	(delta): inf
2 (EM) 	Elapsed: 0.029	Loglikelihood: -3589909.043398	(delta): 315823
3 (EM) 	Elapsed: 0.040	Loglikelihood: -3575200.604387	(delta): 14708.4
4 (EM) 	Elapsed: 0.051	Loglikelihood: -3574247.667174	(delta): 952.937
5 (EM) 	Elapsed: 0.059	Loglikelihood: -3574138.765123	(delta): 108.902
Initial loglikelihood: -3574096.334450
1 (mini-batch, 32 batches) 	Elapsed: 0.080	Loglikelihood: -3305128.521999	(delta): 268968
2 (mini-batch, 32 batches) 	Elapsed: 0.097	Loglikelihood: -3093758.676591	(delta): 211370
3 (mini-batch, 32 batches) 	Elapsed: 0.113	Loglikelihood: -3020493.145226	(delta): 73265.5
4 (mini-batch, 32 batches) 	Elapsed: 0.129	Loglikelihood: -3005025.468828	(delta): 15467.7
5 (mini-batch, 32 batches) 	Elapsed: 0.145	Loglikelihood: -2983492.604518	(delta): 21532.9
6 (mini-batch, 32 batches) 	Elapsed: 0.162	Loglikelihood: -2970637.518166	(delta): 12855.1
7 (mini-batch, 32 batches) 	Elapsed: 0.178	Loglikelihood: -2963043.082651	(delta): 7594.44
8 (mini-batch, 32 batches) 	Elapsed: 0.194	Loglikelihood: -2955627.401695	(delta): 7415.68
9 (mini-batch, 32 batches) 	Elapsed: 0.210	Loglikelihood: -2949074.319894	(delta): 6553.08
10 (mini-batch, 32 batches) 	Elapsed: 0.226	Loglikelihood: -2945153.291869	(delta): 3921.03
11 (mini-batch, 32 batches) 	Elapsed: 0.242	Loglikelihood: -2941630.583144	(delta): 3522.71
12 (mini-batch, 32 batches) 	Elapsed: 0.258	Loglikelihood: -2940097.110484	(delta): 1533.47
13 (mini-batch, 32 batches) 	Elapsed: 0.274	Loglikelihood: -2937886.236819	(delta): 2210.87
14 (mini-batch, 32 batches) 	Elapsed: 0.290	Loglikelihood: -2936857.473163	(delta): 1028.76
15 (mini-batch, 32 batches) 	Elapsed: 0.306	Loglikelihood: -2936705.528618	(delta): 151.945
16 (mini-batch, 32 batches) 	Elapsed: 0.322	Loglikelihood: -2936839.315742	(delta): -133.787
17 (mini-batch, 16 batches) 	Elapsed: 0.336	Loglikelihood: -2934505.434620	(delta): 2333.88
18 (mini-batch, 16 batches) 	Elapsed: 0.351	Loglikelihood: -2934593.456450	(delta): -88.0218
19 (mini-batch, 8 batches) 	Elapsed: 0.365	Loglikelihood: -2933738.627437	(delta): 854.829
20 (mini-batch, 8 batches) 	Elapsed: 0.379	Loglikelihood: -2933552.673171	(delta): 185.954
21 (mini-batch, 8 batches) 	Elapsed: 0.393	Loglikelihood: -2933570.207690	(delta): -17.5345
22 (mini-batch, 4 batches) 	Elapsed: 0.407	Loglikelihood: -2933229.690093	(delta): 340.518
23 (mini-batch, 4 batches) 	Elapsed: 0.420	Loglikelihood: -2933135.647354	(delta): 94.0427
24 (mini-batch, 4 batches) 	Elapsed: 0.434	Loglikelihood: -2933145.977744	(delta): -10.3304
25 (mini-batch, 2 batches) 	Elapsed: 0.447	Loglikelihood: -2933015.506707	(delta): 130.471
26 (mini-batch, 2 batches) 	Elapsed: 0.459	Loglikelihood: -2933009.051346	(delta): 6.45536
27 (mini-batch, 2 batches) 	Elapsed: 0.472	Loglikelihood: -2933005.262884	(delta): 3.78846
28 (mini-batch, 2 batches) 	Elapsed: 0.485	Loglikelihood: -2933001.307338	(delta): 3.95555
29 (mini-batch, 2 batches) 	Elapsed: 0.497	Loglikelihood: -2933000.262047	(delta): 1.04529
30 (mini-batch, 2 batches) 	Elapsed: 0.510	Loglikelihood: -2932999.566725	(delta): 0.695322
31 (mini-batch, 2 batches) 	Elapsed: 0.523	Loglikelihood: -2932999.375594	(delta): 0.191131
32 (mini-batch, 2 batches) 	Elapsed: 0.536	Loglikelihood: -2932998.737526	(delta): 0.638068
33 (mini-batch, 2 batches) 	Elapsed: 0.548	Loglikelihood: -2932998.786367	(delta): -0.0488406
Starting main algorithm
1 (QN/Block) 	Elapsed: 0.586	Loglikelihood: -2932965.459753	(delta): inf
2 (QN/Block) 	Elapsed: 0.612	Loglikelihood: -2932963.623793	(delta): 1.83596
3 (QN/Block) 	Elapsed: 0.639	Loglikelihood: -2932963.275774	(delta): 0.348019
4 (QN/Block) 	Elapsed: 0.682	Loglikelihood: -2932963.129045	(delta): 0.146729
5 (QN/Block) 	Elapsed: 0.706	Loglikelihood: -2932963.106727	(delta): 0.0223178
6 (QN/Block) 	Elapsed: 0.730	Loglikelihood: -2932963.079817	(delta): 0.0269099
7 (QN/Block) 	Elapsed: 0.754	Loglikelihood: -2932963.073994	(delta): 0.005823
8 (QN/Block) 	Elapsed: 0.778	Loglikelihood: -2932963.071934	(delta): 0.00205984
9 (QN/Block) 	Elapsed: 0.802	Loglikelihood: -2932963.070470	(delta): 0.00146464
10 (QN/Block) 	Elapsed: 0.826	Loglikelihood: -2932963.069341	(delta): 0.00112899
11 (QN/Block) 	Elapsed: 0.850	Loglikelihood: -2932963.069302	(delta): 3.8431e-05
Quasi-Newton: 1 of 11 extrapolations rejected, 0 accepted after damping
Summary: 
Converged in 11 iterations (0.915 sec)
Loglikelihood: -2932963.069302
Optimality check (max projected gradient): P 7.78e-05, Q 2.34e-08
Writing output files.
evalAdmix correlation of residuals written to blue_wildebeest_noLD_admixer.7.corres.txt (0.10 sec)
Log written to blue_wildebeest_noLD_admixer.7.log
```

</details>

This run takes 1.0 second and ends at log-likelihood −2932963.07. It writes these files:

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

This tries seeds 0, 1, 2, ... until three runs agree with the best run, in 7 seconds.
`blue_wildebeest_noLD_admixerMult.7.conv` lists the runs, best first:

```
run  seed  loglik           loglik_diff    max_abs_dQ   ...  agrees
9    8     -2906767.495807   0.000000      0                 1
1    0     -2906767.495808  -0.000001      9.53674e-07       1
6    5     -2906767.495820  -0.000013      2.5034e-05        1
8    7     -2932625.706058  -25858.210251  0.99993           0
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

The plots were made with evalAdmix's [`visFuns.R`](https://github.com/GenisGE/evalAdmix); the code is in
[DETAILS.md](DETAILS.md#plotting-the-example-figures).

### Comparison with ADMIXTURE

Ten runs of each program on the same data (seeds 1–10, K = 7, 8 threads, one run at a time). The
log-likelihoods in the left plot are from admixer 0.1.0; the times are from admixer 0.2.6 and ADMIXTURE 1.3.0,
measured on the same server:

```
admixture -j8 -s $seed blue_wildebeest_noLD.bed 7
admixer -j8 -s $seed blue_wildebeest_noLD.bed 7
```

<p float="left">
  <img src="docs/wildebeest_loglik_admixture_vs_admixer.png" alt="Log-likelihood of 10 runs, ADMIXTURE vs admixer" width="56%" />
  <img src="docs/wildebeest_time_admixture_vs_admixer.png" alt="Time per run, ADMIXTURE vs admixer" width="40%" />
</p>

Both programs reach the best likelihood in 5 of the 10 runs. The other runs end in local optima 10,000–27,000
units lower. Over 30 seeds the counts were 18/30 for admixer and 13/30 for ADMIXTURE.

The best runs agree:
- admixer: −2906767.4958 to −2906767.4958;
- ADMIXTURE: −2906767.503 to −2906767.656.

ADMIXTURE's runs are a little lower because its stopping rule ends them slightly before the optimum.

The median time per run was 1.1 s for admixer 0.2.6 and 53 s for ADMIXTURE, 47× faster (admixer 0.1.0 took
3.2 s, 0.2.1 2.2 s, 0.2.2 1.7 s, 0.2.4 1.3 s). admixer 0.2.6 also reaches the best likelihood in 5 of the 10
runs. So `--conv 3` with admixer (7 s above) takes much less time than a single ADMIXTURE run.

## Example: genotype likelihoods, NGSadmix tutorial data, K = 3

The NGSadmix tutorial file `input.gz` holds beagle GLs for 100 HapMap individuals (ASW, CEU, CHB, MXL, YRI)
at 50,000 sites.

```
admixer --seed 1 -j10 input.gz 3
```

The MAF filter keeps 49,475 sites, and 3.3 % of the GL entries are missing. The end of the log:

```
27 (QN/Block) 	Elapsed: 0.801	Loglikelihood: -3865964.313411	(delta): 1.45286e-07
Quasi-Newton: 1 of 27 extrapolations rejected, 1 accepted after damping
Summary: 
Converged in 27 iterations (1.685 sec)
Loglikelihood: -3865964.313411
Loglikelihood over all GL entries, missing ones included (as NGSadmix): -3865964.313411
Optimality check (max projected gradient): P 2.94e-05, Q 1.52e-10
Writing output files.
evalAdmix correlation of residuals written to input.3.corres.txt (0.25 sec)
Log written to input.3.log
```

With 8 threads, NGSadmix 32 (`NGSadmix -likes input.gz -K 3 -P 8 -seed 1`) needs 205 iterations and 16 s on
the same data. It stops at −3865964.43, 0.12 below the optimum.

In a benchmark with 25 random starts per K (8 threads per run, 10 runs at a time; `bench/run_gl_tutorial.sh`
for admixer 0.2.4, `NGSadmix -likes input.gz -K $K -P 8 -seed $seed` for NGSadmix 32), the expected time to reach the best
solution (mean fit time per run, without reading the input, / fraction of runs that reach it) was:

| K | admixer | NGSadmix 32 | starts reaching the best solution (admixer / NGSadmix) |
|---|---|---|---|
| 3 | 1.7 s | 21 s | 100 % / 100 % |
| 4 | 2.1 s | 81 s | 96 % / 48 % |
| 5 | 4.8 s | 492 s | 56 % / 16 % |
| 6 | 7.0 s | 823 s | 60 % / 12 % |

The seed-1 run above (K = 3): admixture proportions and the evalAdmix correlation of residuals, which are close
to 0, except for weak structure within MXL:

<p float="left">
  <img src="docs/ngsadmix_tutorial_admixer.admix.png" alt="Admixture proportions, NGSadmix tutorial data, K = 3" width="52%" />
  <img src="docs/ngsadmix_tutorial_admixer.evaladmix.png" alt="evalAdmix correlation of residuals, NGSadmix tutorial data, K = 3" width="40%" />
</p>

## Citation

For the ADMIXTURE model and algorithm: Alexander, Novembre & Lange (2009) *Genome Research* 19:1655–1664.
For the genotype likelihood model: Skotte, Korneliussen & Albrechtsen (2013) *Genetics* 195:693–702.
For the evalAdmix output: Garcia-Erill & Albrechtsen (2020) *Molecular Ecology Resources* 20:936–949, and
van Waaij et al. (2023) *Genetics* 225:iyad157.
