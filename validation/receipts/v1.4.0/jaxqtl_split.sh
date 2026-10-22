#!/usr/bin/env bash
# jaxQTL 0.3.1 cis mapping of chr22 counts (NB) as four processes of three efficiency cores each, the
# genes split as in the reference runs (data/genelists/chr22.partK.txt). usage: jaxqtl_split.sh OUT ARGS...
set -u
EQTL=/mnt/pikachu/rapidglm/validation/work/eqtl
out=$1; shift
sets=(16-18 19-21 22-24 25-27)
pids=()
for k in 0 1 2 3; do
  mkdir -p $out/part$k
  ( cd $out/part$k && EQX_ON_ERROR=nan taskset -c ${sets[$k]} $EQTL/venv/bin/jaxqtl cis \
      --bfile $EQTL/data/genotypes/chr22 --covar $EQTL/data/covariates/covariates.jaxqtl.tsv \
      --pheno $EQTL/data/expression/chr22.counts.bed.gz --offset $EQTL/data/expression/offset_log_libsize.tsv \
      --model nb --window 1000000 --tss-centered --normalize-covar --seed 20261016 \
      --gene-list $EQTL/data/genelists/chr22.part$k.txt --out chr22 "$@" > stdout.txt 2> stderr.txt ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
exit $rc
