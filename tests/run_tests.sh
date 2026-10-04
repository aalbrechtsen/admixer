#!/bin/bash
# admixer test suite: correctness, all modes, both input types, output formats and regressions against
# the reference results in tests/expected.tsv. Runs in well under 2 minutes.
#
# usage: tests/run_tests.sh [--bin PATH] [-j THREADS] [--update] [--keep]
#   --bin PATH  admixer binary to test (default ./admixer, built with make)
#   -j N        threads per run (default 8)
#   --update    write tests/expected.tsv from this run (after an intended change of results)
#   --keep      keep the work directory (tests/work)
# Exit code: number of failed checks (0 = all passed).
#
# Checks that compare with tests/expected.tsv: the log-likelihood must be within LLTOL of the reference
# (higher by more is reported as IMPROVED and passes); iterations and times are reported, and more than
# 1.5x the reference iterations or 2x the reference time gives a warning, not a failure.
set -u
export LC_ALL=C
cd "$(dirname "$0")/.."
ROOT=$PWD
BIN=$ROOT/admixer
TH=8
UPDATE=0
KEEP=0
LLTOL=0.01
while [ $# -gt 0 ]; do
  case $1 in
    --bin) BIN=$(readlink -f "$2"); shift ;;
    -j) TH=$2; shift ;;
    --update) UPDATE=1 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown option $1"; exit 99 ;;
  esac
  shift
done
T0=$(date +%s.%N)
make -s tests/unit tests/qdist || { echo "cannot build the test helpers"; exit 99; }
[ -x "$BIN" ] || { echo "no admixer binary at $BIN (run make)"; exit 99; }
W=$ROOT/tests/work
rm -rf "$W"; mkdir -p "$W"; cd "$W"
for f in "$ROOT"/tests/data/*; do ln -s "$f" .; done
EXP=$ROOT/tests/expected.tsv
NEW=$W/expected.new.tsv
printf "test\tloglik\titerations\tseconds\n" > "$NEW"
FAILS=0 WARNS=0
pass() { printf "  %-62s ok\n" "$1"; }
fail() { printf "  %-62s FAILED %s\n" "$1" "${2:-}"; FAILS=$((FAILS + 1)); }
warn() { printf "  %-62s warning: %s\n" "$1" "$2"; WARNS=$((WARNS + 1)); }
# numeric comparison: ok "description" "awk condition using a, b" a b
# (a value that is not a finite number, e.g. nan or empty, always fails)
ok() {
  if [[ $3 =~ ^[-+]?[0-9.]+([eE][-+]?[0-9]+)?$ ]] && awk -v a="$3" -v b="${4:-0}" "BEGIN{exit !($2)}"; then pass "$1"
  else fail "$1" "(values: $3 ${4:-})"; fi
}
field() { grep -m1 "^$2" "$1" | sed "s/^$2 *//"; }   # value after a log prefix
ll_of() { field "$1" "Loglikelihood:" | awk '{print $1}'; }
iters_of() { grep -m1 "^Converged in" "$1" | awk '{print $3}'; }
secs_of() { grep -m1 "^Converged in" "$1" | sed 's/.*(\([0-9.]*\) sec).*/\1/'; }
kkt_of() { grep -m1 "^Optimality check" "$1" | sed 's/.*P \([^,]*\), Q \(.*\)/\1 \2/'; }
nrows() { if [[ $1 == *.gz ]]; then zcat "$1" | wc -l; else wc -l < "$1"; fi; }
# run NAME args... : runs admixer with output prefix NAME; stdout+stderr in NAME.out
run() { local n=$1; shift; "$BIN" -j"$TH" -o "$n" "$@" > "$n.out" 2>&1; echo $? > "$n.rc"; }
# reference check of a finished run
refcheck() {
  local n=$1 log=$2 ll it s ref rit rs
  ll=$(ll_of "$log"); it=$(iters_of "$log"); s=$(secs_of "$log")
  printf "%s\t%s\t%s\t%s\n" "$n" "$ll" "$it" "$s" >> "$NEW"
  [ $UPDATE = 1 ] && { pass "$n: reference updated (loglik $ll, $it iterations, $s s)"; return; }
  read -r ref rit rs < <(awk -F'\t' -v n="$n" '$1 == n {print $2, $3, $4}' "$EXP" 2>/dev/null)
  if [ -z "${ref:-}" ]; then warn "$n" "no reference in tests/expected.tsv (run with --update)"; return; fi
  if awk -v a="$ll" -v b="$ref" -v t=$LLTOL 'BEGIN{exit !(a > b + t)}'; then
    printf "  %-62s IMPROVED (%s vs reference %s)\n" "$n: log-likelihood" "$ll" "$ref"
  else
    ok "$n: log-likelihood $ll = reference $ref (±$LLTOL)" "a >= b - $LLTOL" "$ll" "$ref"
  fi
  awk -v a="$it" -v b="$rit" 'BEGIN{exit !(a > 1.5 * b)}' && warn "$n: iterations" "$it vs reference $rit"
  awk -v a="$s" -v b="$rs" 'BEGIN{exit !(a > 2 * b && a > 1)}' && warn "$n: time" "${s}s vs reference ${rs}s"
  printf "  %-62s %s iterations (ref %s), %s s (ref %s)\n" "$n: cost" "$it" "$rit" "$s" "$rs"
}
# format checks of Q (N x K rows summing to 1) and P (M x K)
formats() {
  local n=$1 K=$2 N=$3 M=$4 pfile=$1.$2.P.gz
  ok "$n: Q has $N rows" "a == b" "$(nrows "$n.$K.Q")" "$N"
  ok "$n: Q has $K columns and rows sum to 1" "a == 0" \
    "$(awk -v K="$K" '{s=0; for(k=1;k<=NF;k++) s+=$k; if (NF != K || s < 0.9999 || s > 1.0001) bad++} END{print bad+0}' "$n.$K.Q")"
  ok "$n: P has $M rows in [0,1]" "a == b" \
    "$(zcat "$pfile" | awk -v K="$K" 'NF == K {g=1; for(k=1;k<=NF;k++) if ($k < 0 || $k > 1) g=0; n+=g} END{print n+0}')" "$M"
  ok "$n: log file written" "a == 1" "$(grep -c '^Log written' "$n.$K.log")"
}

# parental (and with paired = 1, paired) ancestry file of run n: header + N rows, simplices summing to 1, log L of the
# model >= log L of the ADMIXTURE model
parfiles() {
  local n=$1 K=$2 N=$3 f=$1.$2.parental
  ok "$n: parental file has a header and $N rows" "a == b + 1" "$(nrows "$f")" "$N"
  ok "$n: parental: 2K + 2 columns, both parents sum to 1" "a == 0" \
    "$(awk -v K="$K" 'NR > 1 {s=0; t=0; for(k=1;k<=K;k++) {s+=$k; t+=$(K+k)} if (NF != 2*K+2 || s < 0.9999 || s > 1.0001 || t < 0.9999 || t > 1.0001) bad++} END{print bad+0}' "$f")"
  ok "$n: parental log L >= ADMIXTURE log L" "a == 0" "$(awk -v K="$K" 'NR > 1 && $(2*K+1) < $(2*K+2) - 1e-6 {bad++} END{print bad+0}' "$f")"
  if [ "${4:-0}" = 1 ]; then
    local kp=$((K * (K + 1) / 2)) g=$1.$2.paired
    ok "$n: paired file has a header and $N rows" "a == b + 1" "$(nrows "$g")" "$N"
    ok "$n: paired: K(K+1)/2 + 2 columns summing to 1" "a == 0" \
      "$(awk -v P="$kp" 'NR > 1 {s=0; for(k=1;k<=P;k++) s+=$k; if (NF != P+2 || s < 0.9999 || s > 1.0001) bad++} END{print bad+0}' "$g")"
    ok "$n: paired log L >= ADMIXTURE log L" "a == 0" "$(awk -v P="$kp" 'NR > 1 && $(P+1) < $(P+2) - 1e-6 {bad++} END{print bad+0}' "$g")"
  fi
}

echo "== unit tests"
if "$ROOT/tests/unit" > unit.out 2>&1; then pass "unit tests (tests/unit.cpp)"; else cat unit.out; fail "unit tests"; fi

echo "== command line"
"$BIN" --help > help.out 2>&1; ok "--help exits with 0" "a == 0" $?
"$BIN" nonexistent.bed 3 > e1.out 2>&1; rc=$?
ok "missing input file: non-zero exit" "a != 0" $rc
ok "missing input file: error message" "a >= 1" "$(grep -c '^Error' e1.out)"
"$BIN" sim.bed 0 > e2.out 2>&1; ok "K = 0 is rejected" "a != 0" $?
run e3 --supervised sim.bed 4; ok "supervised with K != number of populations is rejected" "a != 0" "$(cat e3.rc)"

echo "== PLINK input (sim: 200 individuals, 4000 SNPs, K = 3)"
run plink sim.bed 3 -s 1
ok "plink: exit code 0" "a == 0" "$(cat plink.rc)"
refcheck plink plink.3.log
read -r kp kq < <(kkt_of plink.3.log)
ok "plink: KKT P < 1e-3" "a < 1e-3" "$kp"; ok "plink: KKT Q < 1e-5" "a < 1e-5" "$kq"
# the stopping rule: the run ends on a change below -C (1e-4), and no change is of the size of log L itself
# (GCC 11 -O3 -march=native once miscompiled ll - prev into -ll, so the rule never fired)
ok "plink: last log-likelihood change < -C (1e-4)" "a < 1e-4 && a > -1e-4" \
  "$(grep 'QN/Block' plink.3.log | tail -1 | sed 's/.*(delta): //')"
ok "plink: no log-likelihood change of the size of log L" "a == 0" \
  "$(grep 'QN/Block' plink.3.log | awk -F'\t' 'NR > 1 {split($3, l, " "); split($4, d, " "); if (d[2] > 0.01 * -l[2]) n++} END{print n+0}')"
formats plink 3 200 4000
parfiles plink 3 200
ok "plink: evalAdmix correlations 200 x 200" "a == 40000" "$(awk '{n+=NF} END{print n}' plink.3.corres.txt)"
read -r qmax qms qrmse < <("$ROOT/tests/qdist" plink.3.Q sim.true.Q 3)
ok "plink: Q close to the truth (RMSE < 0.05)" "a < 0.05" "$qrmse"
run plink_rep sim.bed 3 -s 1
ok "plink: same seed gives the same Q" "a == 0" "$(cmp -s plink.3.Q plink_rep.3.Q; echo $?)"
run plink_noP sim.bed 3 -s 1 --no-P --no-evaladmix --no-parental
ok "plink --no-parental: no parental file" "a == 0" "$( [ ! -e plink_noP.3.parental ]; echo $?)"
run plink_pair sim.bed 3 -s 1 --no-evaladmix --paired
parfiles plink_pair 3 200 1
ok "plink --no-P: no P file, same Q" "a == 0" "$( [ ! -e plink_noP.3.P.gz ] && [ ! -e plink_noP.3.P ] && cmp -s plink.3.Q plink_noP.3.Q; echo $?)"
run plink_plain sim.bed 3 -s 1 --no-gzip --no-evaladmix
ok "plink: P.gz (one gzip member per chunk of rows) = --no-gzip text" "a == 0" "$(zcat plink.3.P.gz | cmp -s - plink_plain.3.P; echo $?)"
run plink_j1 sim.bed 3 -s 1 -j1
ok "plink: 1 thread gives the same optimum (|dll| < 1e-3)" "a < 1e-3 && a > -1e-3" \
  "$(awk -v a="$(ll_of plink_j1.3.log)" -v b="$(ll_of plink.3.log)" 'BEGIN{print a-b}')"
run plink_conv sim.bed 3 -s 1 --conv 2 -m 4
ok "plink --conv: converged" "a == 1" "$(grep -c '^Converged: ' plink_conv.3.log)"
ok "plink --conv: table has one row per run" "a >= 3" "$(nrows plink_conv.3.conv)"
ok "plink --conv: best run >= single run" "a >= b - $LLTOL" "$(ll_of plink_conv.3.log)" "$(ll_of plink.3.log)"
run plink_sup --supervised sim.bed 3 -s 1
ok "plink --supervised: exit code 0" "a == 0" "$(cat plink_sup.rc)"
refcheck plink_sup plink_sup.3.log
ok "plink --supervised: labelled individuals keep their population" "a == 30" \
  "$(paste sim.pop plink_sup.3.Q | awk '$1 != "-" {c = index("ABC", $1) + 1; if ($c > 0.999) n++} END{print n+0}')"
zcat plink.3.P.gz > sim.3.P.in
run plink_proj -P sim.bed 3 -s 2
ok "plink -P: exit code 0" "a == 0" "$(cat plink_proj.rc)"
read -r qmax qms qrmse < <("$ROOT/tests/qdist" plink_proj.3.Q plink.3.Q 3)
ok "plink -P: Q from the fixed P equals the joint fit (max |dQ| < 1e-3)" "a < 1e-3" "$qmax"
rm -f sim.3.P.in

echo "== beagle input (GLs of the first 2000 SNPs: 3x depth; mixed 0.5-6x depth)"
for d in d3 mixed; do
  n=bgl_$d
  run $n sim_$d.beagle.gz 3 -s 1
  ok "$n: exit code 0" "a == 0" "$(cat $n.rc)"
  refcheck $n $n.3.log
  read -r kp kq < <(kkt_of $n.3.log)
  ok "$n: KKT P < 1e-3" "a < 1e-3" "$kp"; ok "$n: KKT Q < 1e-5" "a < 1e-5" "$kq"
  M=$(grep -m1 '^Input: genotype likelihoods' $n.3.log | sed 's/.*; \([0-9]*\) sites after filtering.*/\1/')
  formats $n 3 200 "$M"
  parfiles $n 3 200
  ok "$n: full log-likelihood reported" "a == 1" "$(grep -c '^Loglikelihood over all GL entries' $n.3.log)"
  read -r qmax qms qrmse < <("$ROOT/tests/qdist" $n.3.Q sim.true.Q 3)
  ok "$n: Q close to the truth (RMSE < 0.12)" "a < 0.12" "$qrmse"
  ok "$n: evalAdmix correlations 200 x 200" "a == 40000" "$(awk '{n+=NF} END{print n}' $n.3.corres.txt)"
  ok "$n: evalAdmix correlations ~0 under the correct model (|mean| < 0.01)" "a < 0.01 && a > -0.01" \
    "$(awk '{for(k=1;k<=NF;k++) if ($k != "NA") {s+=$k; n++}} END{print s/n}' $n.3.corres.txt)"
done
# evalAdmix for GLs against the genotypes: a misfit (K = 2 for 3 populations) of the first 2000 SNPs. The GL
# fit's P is given to the true genotypes (sim2000.bed, projection -P), so both evaluate the same model; the GL
# correlations must show the same pattern as the genotype ones
run gl2 sim_d3.beagle.gz 2 -s 1 --minMaf=0
zcat gl2.2.P.gz > sim2000.2.P.in
run gt2 -P sim2000.bed 2 -s 1
rm -f sim2000.2.P.in
ok "evalAdmix GL vs genotypes, misfit K = 2: correlation of the matrices > 0.75" "a > 0.75" \
  "$(paste gt2.2.corres.txt gl2.2.corres.txt | awk '{n=NF/2; for(k=1;k<=n;k++) if ($k != "NA") {x=$k; y=$(k+n); sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; m++}} END{print (sxy-sx*sy/m)/sqrt((sxx-sx*sx/m)*(syy-sy*sy/m))}')"
run bgl_rep sim_d3.beagle.gz 3 -s 1
ok "beagle: same seed gives the same Q" "a == 0" "$(cmp -s bgl_d3.3.Q bgl_rep.3.Q; echo $?)"
run bgl_keep sim_mixed.beagle.gz 3 -s 1 --keep-missing
full=$(field bgl_mixed.3.log "Loglikelihood over all GL entries, missing ones included (as NGSadmix):")
ok "beagle --keep-missing: full log-likelihood >= that of the default fit" "a >= b - 1e-3" "$(ll_of bgl_keep.3.log)" "$full"
run bgl_emh sim_d3.beagle.gz 3 -s 1 --hess=em
ok "beagle --hess=em: same optimum as the exact curvature (|dll| < 0.05)" "a < 0.05 && a > -0.05" \
  "$(awk -v a="$(ll_of bgl_emh.3.log)" -v b="$(ll_of bgl_d3.3.log)" 'BEGIN{print a-b}')"
run bgl_sup --supervised sim_d3.beagle.gz 3 -s 1
ok "beagle --supervised: exit code 0" "a == 0" "$(cat bgl_sup.rc)"
refcheck bgl_sup bgl_sup.3.log
ok "beagle --supervised: labelled individuals keep their population" "a == 30" \
  "$(paste sim.pop bgl_sup.3.Q | awk '$1 != "-" {c = index("ABC", $1) + 1; if ($c > 0.999) n++} END{print n+0}')"
zcat bgl_d3.3.P.gz > sim_d3.3.P.in
run bgl_proj -P sim_d3.beagle.gz 3 -s 2
read -r qmax qms qrmse < <("$ROOT/tests/qdist" bgl_proj.3.Q bgl_d3.3.Q 3)
ok "beagle -P: Q from the fixed P equals the joint fit (max |dQ| < 1e-3)" "a < 1e-3" "$qmax"
rm -f sim_d3.3.P.in
run bgl_conv sim_d3.beagle.gz 3 -s 1 --conv 2 -m 4
ok "beagle --conv: converged" "a == 1" "$(grep -c '^Converged: ' bgl_conv.3.log)"

if [ $UPDATE = 1 ]; then cp "$NEW" "$EXP"; echo "reference results written to tests/expected.tsv"; fi
T=$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.0f", b-a}')
echo
if [ $FAILS = 0 ]; then echo "ALL TESTS PASSED ($WARNS warnings, ${T}s; binary $BIN)"; else echo "$FAILS CHECK(S) FAILED ($WARNS warnings, ${T}s)"; fi
cd "$ROOT"; [ $KEEP = 1 ] || rm -rf "$W"
exit $FAILS
