#!/bin/bash
# End-to-end timing of admixer binaries: bench/run_e2e.sh TAG "BIN1 BIN2 ..." "K ..." THREADS [SEEDS]
# Data: bench/data/sim100k_K$K.bed (M=100k, N=2k, anchored). Output: bench/work/e2e_TAG.tsv
cd "$(dirname "$0")"
TAG=$1; BINS=$2; KS=$3; TH=$4; SEEDS=${5:-1}
OUT=work/e2e_$TAG.tsv
[ -f $OUT ] || echo -e "bin\tK\tseed\tthreads\tloglik\titers\twall_s\tcpu_s" > $OUT
for K in $KS; do for s in $SEEDS; do for b in $BINS; do
  d=work/$TAG/$(basename $b)_K${K}_s$s; mkdir -p $d
  /usr/bin/time -f "%e %U %S" -o $d/time ./$b data/sim100k_K$K.bed $K -j $TH --seed=$s -o $d/out --no-gzip --no-evaladmix > $d/stdout 2>&1
  ll=$(grep -o "Loglikelihood: [-0-9.]*" $d/stdout | tail -1 | cut -d' ' -f2)
  it=$(grep -c "QN/Block" $d/stdout)
  read w u sy < $d/time
  echo -e "$(basename $b)\t$K\t$s\t$TH\t$ll\t$it\t$w\t$(echo "$u+$sy" | bc)" >> $OUT
done; done; done
