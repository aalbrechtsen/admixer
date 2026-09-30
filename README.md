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

Example: `admixer -j16 data.bed 5`

## Differences from ADMIXTURE

* Same model, likelihood, parameter bounds, algorithm and stopping rule. Different start: random,
  5 EM steps, then a mini-batch warm-up.
* Not implemented: cross-validation (`--cv`), bootstrap standard errors (`-B`), penalised estimation
  (`-l`), haploid data, the EM method (`-m em`) and the Fst printout. Only PLINK `.bed` input.
* Default 8 threads instead of 1; the P matrix is gzip-compressed, the evalAdmix correlation of
  residuals is written by default, and the screen output is also saved to `data.K.log`.
* The log ends with a first-order optimality (KKT) check of the solution.

## Citation

For the model and algorithm: Alexander, Novembre & Lange (2009) *Genome Research* 19:1655–1664.
For the evalAdmix output: Garcia-Erill & Albrechtsen (2020) *Molecular Ecology Resources* 20:936–949, and
van Waaij et al. (2023) *Genetics* 225:iyad157.
