#!/usr/bin/env bash
# Full 1.5.0 receipts on biocloudgpu1, from the checked dev.3 source tree.
# Run through supervise.py under an exclusive Slurm allocation. Timed cores
# 6-21 have siblings 30-45 idle; drivers use 0-5. CPU policy is read-only.
set -euo pipefail
MAIN=/data/lhhung/rapidglm
cd $MAIN
R=validation/receipts/v1.5.0
RUNS=validation/work/benchmark/runs-v1.5.0-gpu
G=$RUNS/glm-compare
P=$MAIN/validation/work/scaling-preview/inputs
GAS=$MAIN/validation/work/benchmark/gasperini
PILOT=$MAIN/validation/work/replogle/k562-essential/pilot-v1.5.0-gpu
FULL=$MAIN/validation/work/replogle/k562-essential/full-v1.5.0-gpu
EQTL=$MAIN/validation/work/eqtl
EGW=$EQTL/runs-genomewide-v1.5.0-gpu
LOCK=$MAIN/validation/work/gpu-timing.lock
ALL=6-21
ONE=6
EIGHT=6-13
TWELVE=6-17
PCORES=0-5
timed() { python3 scripts/quiet_run.py --cpus $ALL -- taskset -c 0-5 "$@"; }
step() { echo "$(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) $*" >> $R/status.txt; }
# A command on CPUS under the lock, after those CPUs are idle; records wall, CPU, peak memory, the
# total CPU busy time, including the command, and the concurrent host load in DIR/{time.log,host.json}.
guarded() {
  local cpus=$1 dir=$2; shift 2
  python3 "$R/guarded.py" --cpus "$cpus" --directory "$dir" -- "$@"
}

# OMP_NUM_THREADS is not exported: tensorQTL (torch) takes its threads from it. It is set per command
# only where the 1.4.0 runner set it (the CRT commands).
export OPENBLAS_NUM_THREADS=1
export RAPIDGLM_QUIET_LOAD=8
export RAPIDGLM_TIMING_LOCK=$LOCK
export RAPIDGLM_QUIET_SIBLINGS=1
export RAPIDGLM_MEASUREMENT_GUARD=measurement_guard
export RAPIDGLM_TIMED_FIRST_CPU=6
export RAPIDGLM_TIMED_EIGHT_CPUS=$EIGHT
export RAPIDGLM_DIVISION=slow
unset OMP_NUM_THREADS
: "${SCEPTRE_LIBRARY:?Set SCEPTRE_LIBRARY to the installed comparator library}"
export SCEPTRE_LIBRARY
export RAPIDGLM_TIMING_AUDIT=$MAIN/$R/timing-audit.jsonl
export PYTHONPATH=$PWD/scripts:$MAIN/$R${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$RUNS" "$G"
if [ "$#" -eq 0 ]; then
  set -- glmacc glmspeed batch solvers solvers_control bulkval edgerval bulk cooks rust staging pert50k pertfull \
    seeds untestable crt replogle qtl special vmath_e2e replogle_full eqtl_gw compress
fi
active_stage=
trap 'code=$?; if [ -n "$active_stage" ]; then step "$active_stage exit $code"; fi' EXIT

for stage in "$@"; do
  if [ "$stage" != smoke ]; then python3 "$R/verify_source.py"; fi
  active_stage=$stage
  export RAPIDGLM_STAGE=$stage
  unset RAPIDGLM_MIN_AVAILABLE_GIB
  case $stage in pertfull|seeds|crt|replogle_full|eqtl_gw) export RAPIDGLM_MIN_AVAILABLE_GIB=100;; esac
  step "$stage queued (timed commands wait for quiet host and CPUs)"
  case $stage in
    smoke)
      guarded $ONE "$MAIN/validation/work/release-1.5.0-gpu/runner-smoke" python3 -c 'import time; start=time.monotonic(); x=0;
while time.monotonic()-start<3: x+=1
print(x)'
      python3 scripts/quiet_run.py --cpus $ONE -- python3 $R/docker_measure.py --cpus $ONE --image rapidglm-arm64-review-toolchain:local -- \
        python3 -c 'import time; start=time.monotonic(); x=0;
while time.monotonic()-start<3: x+=1
print(x)' ;;
    glmacc)  # accuracy is untimed on CPUs 0-5, eight workers
      taskset -c $PCORES python3 benchmarks/glm/problems.py $G/problems-v1.5.0 > $R/glm_problems.log 2>&1
      taskset -c $PCORES python3 benchmarks/glm/run_comparison.py $G/problems-v1.5.0 --reference --accuracy --reference-binary $RUNS/bin/glm_reference --jobs 5 \
        > $R/glm_accuracy.log 2>&1;;
    glmspeed)
      python3 scripts/quiet_run.py -- taskset -c $ONE python3 benchmarks/glm/run_comparison.py $G/problems-v1.5.0 --speed --budget 15 \
        > $R/glm_speed.log 2>&1;;
    batch)
      timed python3 benchmarks/glm/batch.py --cpus 1:6 4:6-9 16:6-21 --work $G/batch-v1.5.0 \
        --output $R/engine_batch.json > $R/engine_batch.log 2>&1;;
    solvers)
      for n in airway sim5000 thyroid; do
        d=$MAIN/validation/work/benchmark/$n; [ $n = airway ] && d=$MAIN/validation/work/bulk/airway
        for mode in fixed joint; do
          python3 scripts/quiet_run.py --cpus $ONE -- python3 $R/docker_measure.py --cpus $ONE --data $d --image rapidglm-solvers:1.5.0 -- \
            nice -n 10 build-solvers/rapidglm_solver_benchmark /data $mode 1 0 both > $R/solv_${n}_${mode}.json 2> $R/solv_${n}_${mode}.log
          step "solvers $n $mode exit $?"
        done
      done;;
    solvers_control)  # repeat three solver runs on one Zen 3 core
      mkdir -p $R/ecore-control
      for spec in "airway fixed" "airway joint" "thyroid fixed"; do
        set -- $spec
        d=$MAIN/validation/work/benchmark/$1; [ $1 = airway ] && d=$MAIN/validation/work/bulk/airway
        python3 scripts/quiet_run.py --cpus $ONE -- python3 $R/docker_measure.py --cpus $ONE --data $d --image rapidglm-solvers:1.5.0 -- \
          nice -n 10 build-solvers/rapidglm_solver_benchmark /data $2 1 0 both > $R/ecore-control/solv_$1_$2.json 2> /dev/null
        step "solvers_control $1 $2 exit $?"
      done;;
    bulkval)
      python3 scripts/quiet_run.py --cpus $EIGHT -- taskset -c $PCORES python3 scripts/validate_bulk.py --work $RUNS/bulkval --output $R/BULK_DE.json > $R/bulk_de.log 2>&1;;
    edgerval)  # agreement with edgeR 4.6.2's reference tables (one thread each, not the paper's timings)
      taskset -c $PCORES python3 scripts/validate_edger.py --cpus $PCORES --work $RUNS/edger \
        --output $R/EDGER_QL.json > $R/edger_ql.log 2>&1;;
    bulk)  # DESeq2, PyDESeq2, glmGamPoi, edgeR and the RapidGLM interfaces
      timed python3 scripts/benchmark_applications.py --datasets airway thyroid32 thyroid128 --sceptre-datasets \
        --repeats 3 --cpus 1:6 8:6-13 --work $RUNS --output $R/applications_bulk_small.json \
        > $R/applications_bulk_small.log 2>&1
      step "bulk small exit $?"
      timed python3 scripts/benchmark_applications.py --datasets thyroid --sceptre-datasets \
        --repeats 2 --cpus 1:6 8:6-13 --work $RUNS --output $R/applications_bulk_large.json \
        > $R/applications_bulk_large.log 2>&1;;
    cooks)
      taskset -c 0-5 python3 scripts/validate_cooks_flags.py --work $RUNS/cooks_flags --threads 8 --output $R/COOKS_FLAGS.json > $R/cooks_flags.log 2>&1
      taskset -c 0-5 python3 scripts/validate_cooks.py --work $RUNS/cooks --runs $RUNS --thyroid-output $R/COOKS_THYROID.json \
        --legacy-output $R/COOKS_LEGACY.json > $R/cooks.log 2>&1;;
    rust)  # the policy hook is appended because this recipe replaces PYTHONPATH with bindings/python
      for n in airway thyroid32 thyroid128 thyroid; do
        d=validation/work/benchmark/$n; [ $n = airway ] && d=validation/work/bulk/airway
        extra=; [ $n = thyroid32 ] && extra="--deseq2 $RUNS/thyroid32/deseq2-t1/genes.tsv"
        RAPIDGLM_C_LIBRARY=$PWD/build/librapidglm_c.so PYTHONPATH=$PWD/bindings/python:$MAIN/$R:$MAIN/$R/policy_hook \
          python3 scripts/quiet_run.py --cpus $EIGHT -- taskset -c $PCORES validation/local-data/pydeseq2-venv/bin/python \
          bindings/rust/tests/compare_pydeseq2.py $d --json $R/rust_$n.json $extra > $R/rust_$n.log 2>&1
        step "rust $n exit $?"
      done;;
    staging)
      taskset -c $PCORES python3 scripts/validate_sceptre_staging.py --threads 5 --panels S1_public S2_public \
        --work ${RUNS}/staging --output $R/SCEPTRE_STAGING.public.json \
        --sceptre-library $SCEPTRE_LIBRARY > $R/sceptre_staging.log 2>&1;;
    pert50k)
      timed python3 scripts/benchmark_applications.py --datasets --sceptre-datasets gasperini50k --repeats 1 \
        --cpus 1:6 8:6-13 --work $RUNS --output $R/perturbation_50k.json > $R/perturbation_50k.log 2>&1;;
    pertfull)
      export RAPIDGLM_MIN_AVAILABLE_GIB=100
      python3 -c 'from quiet_run import wait_memory; wait_memory(100)'
      timed python3 scripts/benchmark_applications.py --datasets --sceptre-datasets gasperini --threads 8 --repeats 1 \
        --cpus 8:6-13 --work $RUNS --output $R/perturbation_full.json > $R/perturbation_full.log 2>&1
      unset RAPIDGLM_MIN_AVAILABLE_GIB;;
    seeds)  # after pertfull: the ks run of the full screen under six seeds against SCEPTRE
      taskset -c $PCORES python3 scripts/validate_perturbation_seeds.py --runs $RUNS \
        --work $RUNS/seeds --threads 8 --output $R/PERTURBATION_SEEDS.json > $R/perturbation_seeds.log 2>&1;;
    untestable)
      python3 scripts/quiet_run.py --cpus $TWELVE -- python3 scripts/benchmark_untestable_pairs.py --work $RUNS/untestable --binary build/rapidglm \
        --label 1.5.0 --threads 12 --cpus $TWELVE --options '{"sequential_screen": true, "early_stop_p": 0.002}' --output $R/UNTESTABLE_PAIRS.json > $R/untestable.log 2>&1;;
    crt)
      for panel in t50_g1000 t200_g1000 t100_g7418; do
        responses=$P/responses_g1000.mtx; [ $panel = t100_g7418 ] && responses=$GAS/responses.mtx
        for variant in ks default; do
          d=$RUNS/crt/${panel}_$variant; mkdir -p $d
          extra=''; [ $variant = ks ] && extra=', "fit_check": "ks"'
          cat > $d/config.json <<CFG
{"responses": "$responses", "assignments": "$GAS/assignments.mtx", "design": "$GAS/design.tsv", "guides": "$GAS/guides.tsv",
 "pairs": "$P/pairs_$panel.tsv", "moi": "high", "cell_qc": false, "threads": 16, "seed": 20260930, "output": "$d/rapid"$extra}
CFG
          OMP_NUM_THREADS=1 guarded $ALL $d build/rapidglm sceptre $d/config.json
          step "crt $panel $variant exit $? wall $(grep Elapsed $d/time.log | awk '{print $NF}') $(tr -d '\n' < $d/host.json)"
        done
      done;;
    replogle)  # the K562 essential low-MOI pilot: single cell (16 threads) and pseudobulk (one thread)
      for fc in strict ks; do
        python3 scripts/quiet_run.py --cpus $ALL -- python3 scripts/replogle_pilot.py sc --pilot $PILOT --cpus $ALL --threads 16 --fit-check $fc \
          > $R/replogle_sc_$fc.log 2>&1
        step "replogle sc $fc exit $?"
      done
      python3 scripts/quiet_run.py --cpus $ALL -- python3 scripts/replogle_pilot.py sceptre --pilot $PILOT --cpus $ALL --threads 16 \
        > $R/replogle_sceptre.log 2>&1
      step "replogle sceptre exit $?"
      python3 scripts/quiet_run.py --cpus $ONE -- python3 scripts/replogle_pilot.py pseudobulk --pilot $PILOT --cpus $ONE --threads 1 \
        > $R/replogle_pseudobulk.log 2>&1
      step "replogle pseudobulk exit $?"
      for d in $PILOT/pseudobulk/datasets/*; do
        taskset -c 0-5 Rscript scripts/reference_edger_fits.R "$d" perturbed "$d/edger" \
          "$MAIN/validation/local-data/R_glmgampoi:$MAIN/validation/local-data/R_library" > "$d/edger/fit_check.log" 2>&1
      done
      for fc in strict ks; do
        python3 scripts/replogle_pilot.py compare --pilot $PILOT --threads 16 --fit-check $fc \
          --summary $R/replogle_sc_$fc.json > /dev/null 2>&1
      done
      python3 scripts/replogle_pilot.py compare --pilot $PILOT --threads 1 --summary $R/replogle_pseudobulk.json \
        > /dev/null 2>&1;;
    qtl)  # chr22 of Geuvadis (the 1.4.0 eqtl stage, with single and double precision at 10,000 permutations):
          # RapidGLM qtl, FastQTL, tensorQTL and jaxQTL on twelve Zen 3 cores
      E=$RUNS/eqtl; mkdir -p $E
      for fam in gaussian nb; do
        if [ $fam = nb ]; then ph=$EQTL/runs/rapidglm/inputs/chr22.counts.bed
          lib="\"library_sizes\": \"$EQTL/runs/rapidglm/inputs/library_sizes.tsv\","
        else ph=$EQTL/runs/rapidglm/inputs/chr22.expr_int.bed; lib=""; fi
        mkdir -p $E/rapidglm_$fam
        cat > $E/rapidglm_$fam/config.json <<CFG
{"phenotypes": "$ph", "genotypes": "$EQTL/data/genotypes/chr22", "covariates": "$EQTL/data/covariates/covariates.fastqtl.txt", $lib
 "family": "$fam", "window": 1000000, "permutations": 1000, "seed": 20261016, "threads": 12, "output": "$MAIN/$E/rapidglm_$fam/chr22"}
CFG
        guarded $TWELVE $E/rapidglm_$fam build/rapidglm qtl $E/rapidglm_$fam/config.json
        step "qtl rapidglm $fam exit $?"
      done
      # 10,000 permutations: the default (single-precision scores) and double precision (1.4.1 arithmetic).
      for precision in single double; do
        name=rapidglm_gaussian10000; [ $precision = double ] && name=rapidglm_gaussian10000_double
        mkdir -p $E/$name
        python3 - "$E" "$name" "$precision" <<'PYQ'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);name,precision=sys.argv[2:4];c=json.loads((p/'rapidglm_gaussian/config.json').read_text())
c['permutations']=10000;c['output']=str((p/name/'chr22').resolve())
if precision=='double':c['permutation_precision']='double'
(p/name/'config.json').write_text(json.dumps(c))
PYQ
        guarded $TWELVE $E/$name build/rapidglm qtl $E/$name/config.json
        step "qtl $name exit $?"
      done
      # FastQTL (GTEx fork 59bcdc0): adaptive permutations, its own 12 chunks as 12 processes.
      mkdir -p $E/fastqtl/chunks
      guarded $TWELVE $E/fastqtl sh -c "seq 1 12 | xargs -P 12 -I K $EQTL/tools/fastQTL --vcf $EQTL/data/genotypes/chr22.vcf.gz \
        --bed $EQTL/data/expression/chr22.expr_int.bed.gz --cov $EQTL/data/covariates/covariates.fastqtl.txt.gz --window 1e6 \
        --seed 20261016 --permute 1000 10000 --chunk K 12 --out $MAIN/$E/fastqtl/chunks/chunk_K.txt.gz --log $MAIN/$E/fastqtl/chunks/chunk_K.log"
      step "qtl fastqtl exit $?"
      # tensorQTL 1.0.10 (CPU torch): cis mode, 10,000 permutations.
      guarded $TWELVE $E/tensorqtl $EQTL/venv-tensorqtl/bin/python -I -m tensorqtl $EQTL/data/genotypes/chr22 \
        $EQTL/data/expression/chr22.expr_int.bed.gz chr22 --covariates $EQTL/data/covariates/covariates.fastqtl.txt \
        --mode cis --window 1000000 --seed 20261016 -o $MAIN/$E/tensorqtl
      step "qtl tensorqtl exit $?"
      # jaxQTL 0.3.1: the NB score test with SPA and ACAT, and with 1000 permutations; four processes of
      # three cores each (one process on twelve cores is slower), genes split as in the reference runs.
      for spec in "nb_score_spa_acat --test score --spa --acat" "nb_score_perm1000 --test score --nperm 1000"; do
        set -- $spec; name=$1; shift
        guarded $TWELVE $E/jaxqtl_$name bash $R/jaxqtl_split.sh $MAIN/$E/jaxqtl_$name "$@"
        step "qtl jaxqtl $name exit $?"
      done
      taskset -c 0-5 $EQTL/venv/bin/python scripts/summarize_review_eqtl.py --runs $E --output $E/reference_gene_level.tsv.gz
      taskset -c 0-5 python3 scripts/compare_eqtl.py --reference $E/reference_gene_level.tsv.gz --run gaussian $E/rapidglm_gaussian/chr22 \
        --run gaussian10000 $E/rapidglm_gaussian10000/chr22 --run gaussian10000_double $E/rapidglm_gaussian10000_double/chr22 \
        --run nb $E/rapidglm_nb/chr22 --output $R/EQTL_COMPARE.json > $R/compare_eqtl.log 2>&1;;
    special)  # the low-level special-functions benchmark: C++ (accuracy untimed on 0-5), Python and R on CPU 6
      for division in slow fast; do
        RAPIDGLM_DIVISION=$division taskset -c $PCORES python3 scripts/benchmark_special_functions.py --build build --cpu $ONE --untimed-cpus $PCORES \
          --r-library "$MAIN/validation/local-data/R_rapidglm:$MAIN/validation/local-data/R_library" \
          --output $R/SPECIAL_FUNCTIONS.$division.json > $R/special_functions.$division.log 2>&1
      done;;
    vmath_e2e)  # the end-to-end workloads of validation/VMATH.json, one thread on CPU 6, 1.5.0 against 1.4.1
      taskset -c $PCORES python3 $R/vmath_e2e.py --output $R/VMATH_E2E.json \
        > $R/vmath_e2e.log 2>&1;;
    replogle_full)  # every perturbation of the screen; four tools, each on 16 single-thread workers (CPUs 6-21)
      taskset -c $PCORES python3 scripts/replogle_pilot.py full-prepare --pilot $FULL --jobs 5 > $R/replogle_full_prepare.log 2>&1
      step "replogle_full prepare exit $?"
      for tool in rapidglm deseq2 rapidglm_edger_ql edger; do
        timed python3 scripts/replogle_pilot.py full --pilot $FULL --tool $tool --binary $PWD/build/rapidglm \
          --cpus $ALL --threads 1 > $R/replogle_full_$tool.log 2>&1
        step "replogle_full $tool exit $?"
      done
      # The 20 pilot perturbations' outputs against the replogle stage's (pilot driver, each job alone).
      taskset -c $PCORES python3 scripts/replogle_pilot.py full-compare --pilot $FULL --jobs 5 --worst 10 \
        --identical-to $PILOT --summary $FULL/full_compare.json > $R/replogle_full_compare.log 2>&1;;
    eqtl_gw)  # genome-wide Gaussian cis-eQTL on the 1.4.1 inputs: RapidGLM (default precision) and tensorQTL, 10,000 permutations
      GENO=$EQTL/data/genotypes/autosomes; BED=$EQTL/data/expression-genomewide/autosomes.expr_int.bed
      COV=$EQTL/data/covariates/covariates.fastqtl.txt
      mkdir -p $EGW/rapidglm_gaussian10000
      cat > $EGW/rapidglm_gaussian10000/config.json <<CFG
{"phenotypes": "$BED", "genotypes": "$GENO", "covariates": "$COV", "family": "gaussian", "window": 1000000,
 "permutations": 10000, "seed": 20261016, "threads": 12, "output": "$EGW/rapidglm_gaussian10000/autosomes"}
CFG
      guarded $TWELVE $EGW/rapidglm_gaussian10000 build/rapidglm qtl $EGW/rapidglm_gaussian10000/config.json
      step "eqtl_gw rapidglm exit $?"
      mkdir -p $EGW/tensorqtl
      guarded $TWELVE $EGW/tensorqtl $EQTL/venv-tensorqtl/bin/python -I -m tensorqtl $GENO $BED.gz autosomes \
        --covariates $COV --mode cis --permutations 10000 --window 1000000 --seed 20261016 -o $EGW/tensorqtl
      step "eqtl_gw tensorqtl exit $?"
      taskset -c $PCORES $EQTL/venv/bin/python scripts/summarize_review_eqtl.py --runs $EGW \
        --tensorqtl $EGW/tensorqtl/autosomes.cis_qtl.txt.gz --tensorqtl-only --output $EGW/reference_gene_level.tsv.gz \
        > $R/summarize_eqtl_gw.log 2>&1
      taskset -c $PCORES python3 scripts/compare_eqtl.py --reference $EGW/reference_gene_level.tsv.gz \
        --run gaussian10000 $EGW/rapidglm_gaussian10000/autosomes --output $EGW/compare_eqtl.json > $R/compare_eqtl_gw.log 2>&1;;
    compress)  # untimed, after every timed command: gzip the full-screen dataset tables and matrices (as for 1.4.1)
      before=$(du -sb $FULL/pseudobulk/datasets | cut -f1)
      find $FULL/pseudobulk/datasets -type f \( -name '*.tsv' -o -name '*.mtx' \) -size +64k -print0 \
        | taskset -c $PCORES xargs -0 -r -P 5 -n 16 gzip -6
      after=$(du -sb $FULL/pseudobulk/datasets | cut -f1)
      echo "{\"bytes_before\": $before, \"bytes_after\": $after, \"rule\": \"gzip -6 of every .tsv and .mtx over 64 KiB in the dataset folders\", \"finished_utc\": \"$(date -u +%FT%TZ)\"}" > $FULL/compression.json;;
    *) echo "unknown stage $stage" >&2; exit 2;;
  esac
  if [ "$stage" != smoke ]; then python3 "$R/verify_source.py"; fi
  step "$stage exit $?"
  active_stage=
  if [ "$stage" != smoke ]; then taskset -c 0-5 python3 scripts/stamp_review_receipts.py $R; fi
done
