#!/usr/bin/env bash
# Long jaxQTL runs on chr22, one configuration at a time, each as 4 processes x 3 CPUs.
set -uo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
CNT="--pheno $W/data/expression/chr22.counts.bed.gz --offset $W/data/expression/offset_log_libsize.tsv --model nb"
INT="--pheno $W/data/expression/chr22.expr_int.bed.gz --model gaussian"
export EQX_ON_ERROR=nan
echo "start $(date -Is)"
SUB=nominal bash $(dirname "$0")/eqtl_run_jaxqtl_split.sh nominal_nb_score_spa $CNT --test score --spa; echo "nominal_nb $(date -Is)"
SUB=nominal bash $(dirname "$0")/eqtl_run_jaxqtl_split.sh nominal_gaussian_score $INT --test score; echo "nominal_gauss $(date -Is)"
bash $(dirname "$0")/eqtl_run_jaxqtl_split.sh nb_score_perm1000 $CNT --test score --nperm 1000; echo "nb_perm $(date -Is)"
JAXQTL_BIN="$W/venv/bin/python -I $(dirname "$0")/eqtl_jaxqtl_gauss_ncx2.py" bash $(dirname "$0")/eqtl_run_jaxqtl_split.sh gaussian_score_perm1000_ncx2 $INT --test score --nperm 1000; echo "gauss_perm_ncx2 $(date -Is)"
bash $(dirname "$0")/eqtl_run_jaxqtl_split.sh gaussian_score_perm1000_tdist $INT --test score --nperm 1000; echo "gauss_perm_tdist $(date -Is)"
echo "all_done $(date -Is)"
