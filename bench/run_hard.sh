#!/bin/bash
# bench/run_hard.sh TAG BIN "extra args" : 10 seeds on the hard datasets H1 (K 8), L2 (K 10), S2 (K 8)
# from admix_prog/bench/work_hard -> work/hard.tsv
cd "$(dirname "$0")"
TAG=$1; b=$2; EXTRA=$3; H=/kellyData/home/albrecht/codex/admix/admix_prog/bench/work_hard
OUT=work/hard.tsv
[ -f $OUT ] || echo -e "tag\tdata\tK\tseed\tloglik\titers\twall_s\tcpu_s" > $OUT
for dk in H1:8 L2:10 S2:8; do D=${dk%:*}; K=${dk#*:}; for s in 1 2 3 4 5 6 7 8 9 10; do
  d=work/hard/${TAG}_${D}_s$s; mkdir -p $d
  /usr/bin/time -f "%e %U %S" -o $d/time ./$b $H/$D.bed $K -j 8 -s $s -o $d/out --no-gzip --no-evaladmix $EXTRA > $d/stdout 2>&1
  ll=$(grep -o "Loglikelihood: [-0-9.]*" $d/stdout | tail -1 | cut -d' ' -f2); it=$(grep -c "QN/Block" $d/stdout)
  read w u sy < $d/time
  echo -e "$TAG\t$D\t$K\t$s\t$ll\t$it\t$w\t$(echo "$u+$sy" | bc)" >> $OUT
done; done
