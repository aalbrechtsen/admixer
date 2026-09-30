# admixer

Fast maximum-likelihood estimation of ancestry proportions under the ADMIXTURE model, with
ADMIXTURE's command line, input and output. It uses ADMIXTURE's own optimisation algorithm (block
relaxation with Newton steps and quasi-Newton acceleration), evaluated with BLAS matrix products and
started with a mini-batch warm-up. It reaches the same likelihood as ADMIXTURE 1.3.0; in our benchmarks
(20,000 SNPs, 2,000 individuals, K = 5–20, 8 threads) it was 40–90× faster.

## Install

Requires a C++17 compiler with OpenMP (e.g. g++), OpenBLAS and zlib. On Ubuntu/Debian:

```
sudo apt-get install build-essential libopenblas-dev zlib1g-dev
git clone git@github.com:aalbrechtsen/admixer.git
cd admixer
make
```

This builds the `admixer` binary in the current directory. `sudo make install` copies it to
`/usr/local/bin` (or `make install PREFIX=$HOME/.local`). To link a different BLAS, use for example
`make BLAS=-lblis`.

## Usage

```
admixer [options] data.bed K
```

The input is a PLINK binary file set (`data.bed`, `data.bim`, `data.fam`). The output is written to the
working directory, in the same format as ADMIXTURE:

* `data.K.Q`: admixture proportions, one row per individual;
* `data.K.P.gz`: ancestral allele frequencies, one row per SNP (gzip-compressed; `--no-gzip` for `data.K.P`);
* `data.K.corres.txt`: the evalAdmix correlation of residuals between individuals (corrected estimator),
  for checking the model fit; plot it with evalAdmix's `visFuns.R`. It takes a few seconds for thousands of
  individuals but needs about 48 N² bytes of memory, so it is skipped above 20,000 individuals unless
  `--evaladmix` is given.
* `data.K.log`: the log printed to the screen (command line, progress, final log-likelihood).

| option | meaning |
|---|---|
| `-jX`, `-j X` | number of threads (default 8) |
| `--seed=X`, `-s X` | random seed (default 43; `time` uses the clock) |
| `-C=X`, `-C X` | stop when the log-likelihood improves by less than X (default 1e-4) |
| `-o NAME` | output prefix `NAME` instead of `data` |
| `--no-gzip` | write the P matrix uncompressed (`NAME.K.P`) |
| `--supervised` | supervised analysis; reads `data.pop` (one line per individual: a population name, or `-` if unknown). Labelled individuals are held at their population; K must equal the number of population names, and columns follow the order of first appearance in `data.pop` |
| `-P` | projection; reads `data.K.P.in` (or `data.K.P.in.gz`) and estimates Q with P held fixed |
| `--no-evaladmix` | do not compute the evalAdmix correlation of residuals |
| `--evaladmix` | compute it even for more than 20,000 individuals |
| `--conv X` | convergence test with several starts (see below): stop when X runs agree with the best run |
| `-m X`, `--max_runs=X` | with `--conv`: at most X runs (default 10) |
| `--conv_thres=X` | with `--conv`: runs agree if the largest difference in any Q entry is below X (default 0.01) |

Example: `admixer -j16 data.bed 5`

### Convergence test with several starts

Hard data sets can have several local optima, so one run may not find the best solution.
`--conv X` runs admixer with consecutive seeds (`--seed`, `--seed`+1, ...), reading the data only once.
After each run it matches the ancestries of every run to those of the best run so far (highest
log-likelihood), and a run agrees with the best run if the largest difference in any Q entry is below
`--conv_thres`. It stops when X runs (including the best) agree, or after `--max_runs` runs. This is the
Q-matrix criterion of popgenDK's `testQconv.R`, with an exact optimal matching of the ancestries.

The output files are those of the best run. `data.K.conv` lists every run, sorted by log-likelihood (best run first): seed, log-likelihood and its
difference to the best run, the distances to the best run's Q (largest absolute difference, mean
per-individual sum of absolute differences, RMSE), iterations, seconds and whether it agrees.
The log says whether the runs converged.

Example: `admixer --seed=30 --conv 3 data.bed 8` uses seeds 30, 31, 32, ... until 3 runs agree
(at most 10 runs).

## Example: blue wildebeest, K = 7

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

This run takes 4 seconds and ends at log-likelihood −2932963.07. It writes these files:

| file | size | content |
|---|---|---|
| `blue_wildebeest_noLD_admixer.7.Q` | 4.5 kB | admixture proportions, 73 rows × 7 columns (as ADMIXTURE) |
| `blue_wildebeest_noLD_admixer.7.P.gz` | 0.9 MB | ancestral allele frequencies, 37,567 rows × 7 columns, gzipped (`--no-gzip` for plain text) |
| `blue_wildebeest_noLD_admixer.7.corres.txt` | 50 kB | **evalAdmix correlation of residuals**, 73 × 73, `NA` on the diagonal |
| `blue_wildebeest_noLD_admixer.7.log` | 5 kB | the screen output: command, settings, every iteration, final log-likelihood, optimality check |

The evalAdmix correlations are computed by default, in 0.1 s here. You do not need to run evalAdmix
separately; the file is read directly by evalAdmix's `plotCorRes` (see below). The end of the log:

```
Converged in 11 iterations (3.809 sec)
Loglikelihood: -2932963.069302
Optimality check (max KKT violation): P 7.78e-05 per individual, Q 2.34e-08 per SNP
Writing output files.
evalAdmix correlation of residuals written to blue_wildebeest_noLD_admixer.7.corres.txt (0.11 sec)
Log written to blue_wildebeest_noLD_admixer.7.log
```

<details>
<summary>Full screen output of the run (click to expand)</summary>

```
admixer 0.1.0
Command: admixer --seed 1 -j10 -o blue_wildebeest_noLD_admixer blue_wildebeest_noLD.bed 7
Random seed: 1
Point estimation method: Block relaxation algorithm (Newton/QP steps, BLAS kernels)
Convergence acceleration algorithm: QuasiNewton, 3 secant conditions
Point estimation will terminate when objective function delta < 0.0001
Size of G: 73x37567
Threads: 10
Performing five EM steps to prime main algorithm
1 (EM) 	Elapsed: 0.027	Loglikelihood: -3905731.572719	(delta): inf
2 (EM) 	Elapsed: 0.038	Loglikelihood: -3589909.043398	(delta): 315823
3 (EM) 	Elapsed: 0.046	Loglikelihood: -3575200.604387	(delta): 14708.4
4 (EM) 	Elapsed: 0.054	Loglikelihood: -3574247.667174	(delta): 952.937
5 (EM) 	Elapsed: 0.063	Loglikelihood: -3574138.765123	(delta): 108.902
Initial loglikelihood: -3574096.334450
1 (mini-batch, 32 batches) 	Elapsed: 0.186	Loglikelihood: -3305128.521989	(delta): 268968
2 (mini-batch, 32 batches) 	Elapsed: 0.277	Loglikelihood: -3093758.676473	(delta): 211370
3 (mini-batch, 32 batches) 	Elapsed: 0.360	Loglikelihood: -3020493.145204	(delta): 73265.5
4 (mini-batch, 32 batches) 	Elapsed: 0.445	Loglikelihood: -3005025.468841	(delta): 15467.7
5 (mini-batch, 32 batches) 	Elapsed: 0.529	Loglikelihood: -2983492.604508	(delta): 21532.9
6 (mini-batch, 32 batches) 	Elapsed: 0.611	Loglikelihood: -2970637.518139	(delta): 12855.1
7 (mini-batch, 32 batches) 	Elapsed: 0.699	Loglikelihood: -2963043.082644	(delta): 7594.44
8 (mini-batch, 32 batches) 	Elapsed: 0.782	Loglikelihood: -2955627.401759	(delta): 7415.68
9 (mini-batch, 32 batches) 	Elapsed: 0.862	Loglikelihood: -2949074.319889	(delta): 6553.08
10 (mini-batch, 32 batches) 	Elapsed: 0.954	Loglikelihood: -2945153.291885	(delta): 3921.03
11 (mini-batch, 32 batches) 	Elapsed: 1.033	Loglikelihood: -2941630.583178	(delta): 3522.71
12 (mini-batch, 32 batches) 	Elapsed: 1.112	Loglikelihood: -2940097.110520	(delta): 1533.47
13 (mini-batch, 32 batches) 	Elapsed: 1.189	Loglikelihood: -2937886.236841	(delta): 2210.87
14 (mini-batch, 32 batches) 	Elapsed: 1.267	Loglikelihood: -2936857.473182	(delta): 1028.76
15 (mini-batch, 32 batches) 	Elapsed: 1.345	Loglikelihood: -2936705.528626	(delta): 151.945
16 (mini-batch, 32 batches) 	Elapsed: 1.424	Loglikelihood: -2936839.315748	(delta): -133.787
17 (mini-batch, 16 batches) 	Elapsed: 1.492	Loglikelihood: -2934505.434622	(delta): 2333.88
18 (mini-batch, 16 batches) 	Elapsed: 1.560	Loglikelihood: -2934593.456459	(delta): -88.0218
19 (mini-batch, 8 batches) 	Elapsed: 1.622	Loglikelihood: -2933738.627439	(delta): 854.829
20 (mini-batch, 8 batches) 	Elapsed: 1.691	Loglikelihood: -2933552.673173	(delta): 185.954
21 (mini-batch, 8 batches) 	Elapsed: 1.754	Loglikelihood: -2933570.207691	(delta): -17.5345
22 (mini-batch, 4 batches) 	Elapsed: 1.814	Loglikelihood: -2933229.690094	(delta): 340.518
23 (mini-batch, 4 batches) 	Elapsed: 1.872	Loglikelihood: -2933135.647354	(delta): 94.0427
24 (mini-batch, 4 batches) 	Elapsed: 1.932	Loglikelihood: -2933145.977744	(delta): -10.3304
25 (mini-batch, 2 batches) 	Elapsed: 1.990	Loglikelihood: -2933015.506707	(delta): 130.471
26 (mini-batch, 2 batches) 	Elapsed: 2.047	Loglikelihood: -2933009.051346	(delta): 6.45536
27 (mini-batch, 2 batches) 	Elapsed: 2.105	Loglikelihood: -2933005.262884	(delta): 3.78846
28 (mini-batch, 2 batches) 	Elapsed: 2.161	Loglikelihood: -2933001.307338	(delta): 3.95555
29 (mini-batch, 2 batches) 	Elapsed: 2.218	Loglikelihood: -2933000.262047	(delta): 1.04529
30 (mini-batch, 2 batches) 	Elapsed: 2.276	Loglikelihood: -2932999.566725	(delta): 0.695322
31 (mini-batch, 2 batches) 	Elapsed: 2.334	Loglikelihood: -2932999.375594	(delta): 0.191131
32 (mini-batch, 2 batches) 	Elapsed: 2.391	Loglikelihood: -2932998.737526	(delta): 0.638068
33 (mini-batch, 2 batches) 	Elapsed: 2.449	Loglikelihood: -2932998.786367	(delta): -0.0488406
Starting main algorithm
1 (QN/Block) 	Elapsed: 2.571	Loglikelihood: -2932965.459753	(delta): inf
2 (QN/Block) 	Elapsed: 2.685	Loglikelihood: -2932963.623793	(delta): 1.83596
3 (QN/Block) 	Elapsed: 2.801	Loglikelihood: -2932963.275774	(delta): 0.348019
4 (QN/Block) 	Elapsed: 2.915	Loglikelihood: -2932963.129045	(delta): 0.146729
5 (QN/Block) 	Elapsed: 3.026	Loglikelihood: -2932963.106727	(delta): 0.0223178
6 (QN/Block) 	Elapsed: 3.140	Loglikelihood: -2932963.079817	(delta): 0.0269099
7 (QN/Block) 	Elapsed: 3.252	Loglikelihood: -2932963.073994	(delta): 0.005823
8 (QN/Block) 	Elapsed: 3.364	Loglikelihood: -2932963.071934	(delta): 0.00205984
9 (QN/Block) 	Elapsed: 3.476	Loglikelihood: -2932963.070470	(delta): 0.00146464
10 (QN/Block) 	Elapsed: 3.590	Loglikelihood: -2932963.069341	(delta): 0.001129
11 (QN/Block) 	Elapsed: 3.702	Loglikelihood: -2932963.069302	(delta): 3.84306e-05
Summary: 
Converged in 11 iterations (3.809 sec)
Loglikelihood: -2932963.069302
Optimality check (max KKT violation): P 7.78e-05 per individual, Q 2.34e-08 per SNP
Writing output files.
evalAdmix correlation of residuals written to blue_wildebeest_noLD_admixer.7.corres.txt (0.11 sec)
Log written to blue_wildebeest_noLD_admixer.7.log
```

</details>

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

This tries seeds 0, 1, 2, ... until three runs agree with the best run, in 33 seconds.
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

Ten runs of each program on the same data (seeds 1–10, K = 7, 8 threads, one run at a time):

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

The median time per run is 3.2 s for admixer and 53 s for ADMIXTURE, 17× faster. So `--conv 3` with admixer
(33 s above) takes less time than a single ADMIXTURE run.

## Differences from ADMIXTURE

* Same model, likelihood, parameter bounds, algorithm and stopping rule. Different start: random P,
  near-uniform Q, 5 EM steps, then a mini-batch warm-up.
* Not implemented: cross-validation (`--cv`), bootstrap standard errors (`-B`), penalised estimation
  (`-l`), haploid data, the EM method and the Fst printout. Only PLINK `.bed` input. Note that `-m` is
  admixer's maximum number of runs, not ADMIXTURE's method option.
* Default 8 threads instead of 1; the P matrix is gzip-compressed, the evalAdmix correlation of
  residuals is written by default, and the screen output is also saved to `data.K.log`.
* The log ends with a first-order optimality (KKT) check of the solution.

## Citation

For the model and algorithm: Alexander, Novembre & Lange (2009) *Genome Research* 19:1655–1664.
For the evalAdmix output: Garcia-Erill & Albrechtsen (2020) *Molecular Ecology Resources* 20:936–949, and
van Waaij et al. (2023) *Genetics* 225:iyad157.
