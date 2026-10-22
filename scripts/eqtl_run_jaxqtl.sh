#!/usr/bin/env bash
# jaxQTL 0.3.1 (PyPI, CPU JAX) on chr22. One process; JAX/XLA CPU threads are limited by the
# CPU affinity mask (taskset -c 0-7,12-15 = 12 logical CPUs).
# usage: run_jaxqtl.sh NAME [extra jaxqtl args...]
set -euo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
CHR=${CHR:-22}
name=$1; shift
sub=${SUB:-cis}
OUT=$W/runs/jaxqtl/$name
mkdir -p $OUT
cd $OUT
args=(--bfile $W/data/genotypes/chr$CHR --covar $W/data/covariates/covariates.jaxqtl.tsv
      --window 1000000 --tss-centered --normalize-covar --seed 20261016 --verbose --out chr$CHR "$@")
echo "CPUS=${CPUS:-0-7,12-15} EQX_ON_ERROR=${EQX_ON_ERROR:-raise} ${JAXQTL_BIN:-jaxqtl} $sub ${args[*]}" > command.txt
start=$(date +%s.%N)
/usr/bin/time -v -o time.txt taskset -c ${CPUS:-0-7,12-15} nice -n 10 ${JAXQTL_BIN:-$W/venv/bin/jaxqtl} $sub "${args[@]}" > stdout.txt 2> stderr.txt
end=$(date +%s.%N)
echo "wall_seconds $(echo "$end - $start" | bc)" > wall.txt
echo done
