#!/bin/bash
# bench/run_var.sh TAG BIN K THREADS SEED "extra args"  -> appends to work/var.tsv
cd "$(dirname "$0")"
TAG=$1; b=$2; K=$3; TH=$4; s=$5; EXTRA=$6
OUT=work/var.tsv
[ -f $OUT ] || echo -e "tag\tbin\tK\tseed\tthreads\textra\tloglik\titers\twarmup_end_s\twall_s\tcpu_s" > $OUT
d=work/var/${TAG}_K${K}_t${TH}_s$s; mkdir -p $d
/usr/bin/time -f "%e %U %S" -o $d/time ./$b data/sim100k_K$K.bed $K -j $TH --seed=$s -o $d/out --no-gzip --no-evaladmix $EXTRA > $d/stdout 2>&1
ll=$(grep -o "Loglikelihood: [-0-9.]*" $d/stdout | tail -1 | cut -d' ' -f2)
it=$(grep -c "QN/Block" $d/stdout)
wu=$(grep -E "\((EM|mini-batch)" $d/stdout | tail -1 | grep -o "Elapsed: [0-9.]*" | cut -d' ' -f2)
read w u sy < $d/time
echo -e "$TAG\t$b\t$K\t$s\t$TH\t$EXTRA\t$ll\t$it\t$wu\t$w\t$(echo "$u+$sy" | bc)" >> $OUT
