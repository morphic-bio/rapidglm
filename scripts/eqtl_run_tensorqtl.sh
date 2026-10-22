#!/usr/bin/env bash
# tensorQTL 1.0.10 (CPU torch) on chr22 with the same inputs as FastQTL: cis permutation mode
# (fixed 10,000 permutations, beta approximation) and cis_nominal mode.
set -euo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
mode=$1   # cis | cis_nominal
OUT=$W/runs/tensorqtl/$mode
mkdir -p $OUT
cd $OUT
cmd="$W/venv-tensorqtl/bin/python -I -m tensorqtl $W/data/genotypes/chr22 $W/data/expression/chr22.expr_int.bed.gz chr22 --covariates $W/data/covariates/covariates.fastqtl.txt --mode $mode --window 1000000 --seed 20261016 -o $OUT"
echo "$cmd" > command.txt
start=$(date +%s.%N)
/usr/bin/time -v -o time.txt taskset -c ${CPUS:-0-7,12-15} nice -n 10 $cmd > stdout.txt 2> stderr.txt
end=$(date +%s.%N)
echo "wall_seconds $(echo "$end - $start" | bc)" > wall.txt
echo done
