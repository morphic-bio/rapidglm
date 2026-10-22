#!/usr/bin/env bash
# FastQTL (GTEx fork, commit 59bcdc0, v2.184_gtex) on chr22: nominal and adaptive permutation passes.
# Parallelised with FastQTL's own --chunk K N option, one process per chunk on CPUs 0-7,12-15.
set -euo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
FQ=$W/tools/fastQTL
CHR=${CHR:-22}
NCHUNK=${NCHUNK:-12}
NPROC=${NPROC:-12}
SEED=20261016
VCF=$W/data/genotypes/chr${CHR}.vcf.gz
BED=$W/data/expression/chr${CHR}.expr_int.bed.gz
COV=$W/data/covariates/covariates.fastqtl.txt.gz
mode=$1   # nominal | permute
OUT=$W/runs/fastqtl/$mode
mkdir -p $OUT/chunks
cd $OUT
case $mode in
  nominal) extra="" ;;
  permute) extra="--permute 1000 10000" ;;
  *) echo "mode?"; exit 1 ;;
esac
cmd="$FQ --vcf $VCF --bed $BED --cov $COV --window 1e6 --seed $SEED $extra"
echo "$cmd --chunk K $NCHUNK --out chunks/chunk_K.txt.gz --log chunks/chunk_K.log" > command.txt
start=$(date +%s.%N)
/usr/bin/time -v -o time.txt sh -c "seq 1 $NCHUNK | taskset -c 0-7,12-15 nice -n 10 xargs -P $NPROC -I K $cmd --chunk K $NCHUNK --out chunks/chunk_K.txt.gz --log chunks/chunk_K.log > /dev/null 2> stderr.txt"
end=$(date +%s.%N)
echo "wall_seconds $(echo "$end - $start" | bc)" > wall.txt
# merge (header from first chunk that has one, if any)
first=1
for k in $(seq 1 $NCHUNK); do
  f=chunks/chunk_$k.txt.gz
  [ -s $f ] || continue
  if [ $first = 1 ]; then zcat $f; first=0; else zcat $f | { if zcat $f | head -1 | grep -q '^gene_id\|^pid\|^phenotype_id'; then tail -n +2; else cat; fi; }; fi
done | gzip > chr${CHR}.$mode.txt.gz
echo done
