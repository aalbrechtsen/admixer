# admixer

Fast maximum-likelihood estimation of ancestry proportions under the ADMIXTURE model, with
ADMIXTURE's command line, input and output. It uses ADMIXTURE's own optimisation algorithm (block
relaxation with Newton steps and quasi-Newton acceleration), evaluated with BLAS matrix products and
started with a mini-batch warm-up. It reaches the same likelihood as ADMIXTURE 1.3.0; in our benchmarks
(20,000 SNPs, 2,000 individuals, K = 5–20, 8 threads) it was 40–90× faster.

## Install

Requires a C++17 compiler with OpenMP (e.g. g++) and OpenBLAS. On Ubuntu/Debian:

```
sudo apt-get install build-essential libopenblas-dev
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
working directory as `data.K.Q` (admixture proportions, one row per individual) and `data.K.P` (ancestral
allele frequencies, one row per SNP), in the same format as ADMIXTURE.

| option | meaning |
|---|---|
| `-jX`, `-j X` | number of threads (default 8) |
| `--seed=X`, `-s X` | random seed (default 43; `time` uses the clock) |
| `-C=X`, `-C X` | stop when the log-likelihood improves by less than X (default 1e-4) |
| `-o NAME` | output prefix `NAME` instead of `data` (`NAME.K.Q`, `NAME.K.P`) |
| `--supervised` | supervised analysis; reads `data.pop` (one line per individual: a population name, or `-` if unknown). Labelled individuals are held at their population; K must equal the number of population names, and columns follow the order of first appearance in `data.pop` |
| `-P` | projection; reads `data.K.P.in` and estimates Q with P held fixed |
| `--evaladmix` | also writes `NAME.K.corres.txt`, the evalAdmix correlation of residuals (corrected estimator); plot it with evalAdmix's `visFuns.R` |

Example: `admixer -j16 --evaladmix data.bed 5`

## Differences from ADMIXTURE

* Same model, likelihood, parameter bounds, algorithm and stopping rule. Different start: random,
  5 EM steps, then a mini-batch warm-up.
* Not implemented: cross-validation (`--cv`), bootstrap standard errors (`-B`), penalised estimation
  (`-l`), haploid data, the EM method (`-m em`) and the Fst printout. Only PLINK `.bed` input.
* Default 8 threads instead of 1.
* The log ends with a first-order optimality (KKT) check of the solution.

## Citation

For the model and algorithm: Alexander, Novembre & Lange (2009) *Genome Research* 19:1655–1664.
For `--evaladmix`: Garcia-Erill & Albrechtsen (2020) *Molecular Ecology Resources* 20:936–949, and
van Waaij et al. (2023) *Genetics* 225:iyad157.
