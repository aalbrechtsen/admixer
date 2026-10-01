#!/bin/bash
# README table "genotype likelihoods, NGSadmix tutorial data": admixer with 25 random starts per K = 3-6 on the
# NGSadmix training data, 8 threads per run, 10 runs in parallel (as the original benchmark,
# ngsadmix/bench/run_training.sh). Fit time = the last "Elapsed" of the log (the fit without reading the input).
# Success = within 5 log-units of the best log-likelihood over all methods in the original benchmark
# (local optima are >= 16 units worse). Expected time = mean fit time / success rate.
# usage: bench/run_gl_tutorial.sh [BIN]   -> bench/work/gl_tutorial/summary.tsv
cd "$(dirname "$0")"
BIN=$(readlink -f "${1:-../admixer}")
NG=/kellyData/home/albrecht/codex/admix/ngsadmix
LIKES=$NG/data/input.gz
W=work/gl_tutorial; mkdir -p $W; W=$(readlink -f $W)
for K in 3 4 5 6; do for s in $(seq 1 25); do
  echo "cd $W && /usr/bin/time -f '%e %U %S' -o K${K}_s$s.time $BIN $LIKES $K -j 8 -s $s -o K${K}_s$s --no-evaladmix > K${K}_s$s.out 2>&1"
done; done > $W/jobs.txt
xargs -P 10 -I{} -d '\n' bash -c "{}" < $W/jobs.txt
# best log-likelihood per K in the original benchmark (all methods, NGSadmix rescored with the same likelihood)
python3 - "$NG/bench/work/training/runs.csv" "$W" <<'EOF'
import csv, glob, os, re, sys
best = {}
for r in csv.DictReader(open(sys.argv[1])):
    if r['loglik']:
        k = int(r['K']); best[k] = max(best.get(k, -1e300), float(r['loglik']))
W = sys.argv[2]
rows = []
for K in (3, 4, 5, 6):
    fits, hits, walls = [], 0, []
    for s in range(1, 26):
        out = open(os.path.join(W, f'K{K}_s{s}.out')).read()
        # the main log-likelihood (entries in the model); the "all entries" line is NGSadmix's objective
        ll = float(re.search(r'Loglikelihood over all GL entries, missing ones included \(as NGSadmix\): (\S+)', out).group(1))
        fit = float(re.findall(r'Elapsed: ([0-9.]+)', out)[-1])
        wall = float(open(os.path.join(W, f'K{K}_s{s}.time')).read().split()[0])
        fits.append(fit); walls.append(wall); hits += ll > best[K] - 5
    mean = sum(fits) / len(fits)
    rows.append((K, len(fits), hits, mean, sorted(walls)[12], mean / (hits / len(fits)) if hits else float('inf'), best[K]))
with open(os.path.join(W, 'summary.tsv'), 'w') as f:
    f.write('K\truns\tsuccess\tfit_mean_s\twall_median_s\texpected_s\tbest_loglik\n')
    for r in rows:
        f.write('%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.3f\n' % r)
print(open(os.path.join(W, 'summary.tsv')).read())
EOF
