# Reference runs of ADMIXTURE 1.3.0 and NGSadmix 32

Timings of the comparison programs, kept so that they do not have to be rerun. Every row is one run.

| file | content |
|---|---|
| `admixture_1.3.0.tsv` | ADMIXTURE 1.3.0 (`admixture -jT -s SEED data.bed K`), all runs measured so far |
| `admixture_per_iteration.tsv` | time of one EM step and one quasi-Newton iteration by number of threads, ADMIXTURE vs admixer |
| `ngsadmix32.tsv` | NGSadmix 32 (`NGSadmix -likes input.gz -K K -P 8 -seed SEED`) on the NGSadmix tutorial GLs, 25 seeds per K = 3-6 |
| `admixer_readme_runs.tsv` | admixer 0.2.1 (`30f49d6`) to 0.2.6 on the README examples, same server and conditions |
| `logs_e5-2699v4.tar.gz` | the raw logs of the e5-2699v4 runs |

Columns: `server`, `data`, `M` (SNPs), `N` (individuals), `K`, `threads`, `seed`, `wall_s`, `cpu_s` (empty when not
recorded), `loglik` (as printed by the program; NGSadmix's re-scored with the same likelihood as admixer),
`iterations` (quasi-Newton iterations for ADMIXTURE, EM/QN iterations for NGSadmix), `protocol`, `source`.

**Servers.**
* `gold6152`: 2 x Xeon Gold 6152 (44 cores, 88 threads, AVX-512); the runs of `admix_prog/report2` and of the
  README up to admixer 0.2.1 (September - 1 October 2026).
* `e5-2699v4`: 2 x Xeon E5-2699 v4 (44 cores, 88 threads, AVX2, no AVX-512); 2 October 2026, idle apart from the
  stated protocol.

ADMIXTURE 1.3.0 does not use AVX-512: it gives the same log-likelihood for every seed on both servers and about
the same time (blue wildebeest, seeds 1-10: median 53.0 s on gold6152, 53.3 s on e5-2699v4). admixer is faster on
gold6152 (AVX-512 kernels), so compare admixer with these runs only on the same server.

**Data.**
* `blue_wildebeest_noLD`: 73 blue wildebeest, 37,567 SNPs (the README example); `blue_wildebeest_noLD_r02`: the
  same individuals, 185,861 SNPs (LD pruning at r2 > 0.2). K = 7.
* `sim100k`: M = 100,000, N = 2,000, Fst 0.05, Dirichlet(0.5), 2 unadmixed individuals per population, 1 % missing
  (`bench/data/sim100k_K*.bed`).
* `anch_20000_2000`: M = 20,000, N = 2,000, Fst 0.05 (`admix_prog/bench/work`).
* `H1`, `L2`, `S2`: hard simulated data sets, M = 10,000, N = 600 (`admix_prog/bench/work_hard`).
* `input.gz`: NGSadmix tutorial genotype likelihoods, 100 HapMap individuals, 50,000 sites.
