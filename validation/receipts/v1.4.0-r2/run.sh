#!/usr/bin/env bash
# Review rerun for RapidGLM 1.4.0-dev.3. Source commit is recorded in source.json.
# The v1.4.0 receipt directory is read-only provenance for this run. Each timed stage holds
# the shared timing lock. Every timed run is confined to efficiency cores, one thread per core: one thread
# to CPU 16, four to 16-19, eight to 16-23, twelve to 16-27 (the eQTL tools), sixteen to 16-31. Runs that
# share CPUs with other sessions' untimed jobs (20-31) wait for those CPUs to be idle where the harness
# supports it, and the total CPU busy time on those CPUs is recorded (including the timed command).
#   bash run.sh STAGE...   stages as in the case below, in the order given
set -euo pipefail
cd /mnt/pikachu/rapidglm
R=validation/receipts/v1.4.0-r2
RUNS=validation/work/benchmark/runs-v1.4.0-r2
G=validation/work/glm-compare
P=/mnt/pikachu/rapidglm_scaling_preview_20261001/inputs
GAS=$PWD/validation/work/benchmark/gasperini
PILOT=$PWD/validation/work/replogle/k562-essential/pilot-v1.4.0-r2
EQTL=$PWD/validation/work/eqtl
LOCK=/mnt/pikachu/e2e_bench_20260926/pikachu_timed.lock
ALL=16-31
ONE=16
EIGHT=16-23
TWELVE=16-27
PCORES=0-7,12-15
timed() { python3 scripts/quiet_run.py -- taskset -c $ALL "$@"; }
step() { echo "$(date +%FT%T) load $(cut -d' ' -f1-3 /proc/loadavg) $*" >> $R/status.txt; }
# A command on CPUS under the lock, after those CPUs are idle; records wall, CPU, peak memory and the
# total CPU busy time, including the command, in DIR/{time.log,host.json}.
guarded() {
  local cpus=$1 dir=$2; shift 2
  mkdir -p $dir
  (
    flock 9
    waited=$(python3 $R/cpu_guard.py wait $cpus)
    python3 scripts/quiet_run.py --wait-only --cpus "$cpus"
    before=$(python3 $R/cpu_guard.py busy $cpus)
    set +e
    /usr/bin/time -v -o $dir/time.log taskset -c $cpus "$@" > $dir/stdout.log 2> $dir/stderr.log
    code=$?
    set -e
    after=$(python3 $R/cpu_guard.py busy $cpus)
    echo "{\"waited_for_idle_seconds\": $waited, \"cpus_busy_seconds\": $(echo "$after - $before" | bc), \"exit\": $code}" > $dir/host.json
    exit $code
  ) 9> $LOCK
}
export OPENBLAS_NUM_THREADS=1
export RAPIDGLM_QUIET_LOAD=4
export SCEPTRE_LIBRARY=<comparator R library>

for stage in "$@"; do
  step "$stage queued (timed commands wait for quiet host and CPUs)"
  case $stage in
    glmacc)  # accuracy is not timed: performance cores, eight at a time, with the lock held
      python3 scripts/quiet_run.py -- taskset -c $PCORES python3 benchmarks/glm/problems.py $G/problems-v1.4.0 > $R/glm_problems.log 2>&1 &&
      python3 scripts/quiet_run.py -- taskset -c $PCORES python3 benchmarks/glm/run_comparison.py $G/problems-v1.4.0 --reference --accuracy \
        > $R/glm_accuracy.log 2>&1;;
    glmspeed)
      python3 scripts/quiet_run.py -- taskset -c $ONE python3 benchmarks/glm/run_comparison.py $G/problems-v1.4.0 --speed --budget 15 \
        > $R/glm_speed.log 2>&1;;
    batch)
      timed python3 benchmarks/glm/batch.py --cpus 1:16 4:16-19 16:16-31 --work $G/batch-v1.4.0 \
        --output $R/engine_batch.json > $R/engine_batch.log 2>&1;;
    solvers)
      for n in airway sim5000 thyroid; do
        d=$PWD/validation/work/benchmark/$n; [ $n = airway ] && d=$PWD/validation/work/bulk/airway
        for mode in fixed joint; do
          python3 scripts/quiet_run.py -- docker run --rm --cpuset-cpus $ONE -e OPENBLAS_NUM_THREADS=1 -v $d:/data:ro rapidglm-solvers:1.4.0 \
            build-solvers/rapidglm_solver_benchmark /data $mode 1 0 both > $R/solv_${n}_${mode}.json 2> $R/solv_${n}_${mode}.log
          step "solvers $n $mode exit $?"
        done
      done;;
    solvers_control)  # three of the runs on one performance core
      mkdir -p $R/pcore-control
      for spec in "airway fixed" "airway joint" "thyroid fixed"; do
        set -- $spec
        d=$PWD/validation/work/benchmark/$1; [ $1 = airway ] && d=$PWD/validation/work/bulk/airway
        python3 scripts/quiet_run.py -- docker run --rm --cpuset-cpus 4 -e OPENBLAS_NUM_THREADS=1 -v $d:/data:ro rapidglm-solvers:1.4.0 \
          build-solvers/rapidglm_solver_benchmark /data $2 1 0 both > $R/pcore-control/solv_$1_$2.json 2> /dev/null
        step "solvers_control $1 $2 exit $?"
      done;;
    bulkval)
      python3 scripts/quiet_run.py -- taskset -c $EIGHT python3 scripts/validate_bulk.py --output $R/BULK_DE.json > $R/bulk_de.log 2>&1;;
    edgerval)  # agreement with edgeR 4.6.2's reference tables (one thread each, not the paper's timings)
      python3 scripts/quiet_run.py -- taskset -c $ONE python3 scripts/validate_edger.py --cpus $ONE --work $RUNS/edger \
        --output $R/EDGER_QL.json > $R/edger_ql.log 2>&1;;
    bulk)  # DESeq2, PyDESeq2, glmGamPoi, edgeR and the RapidGLM interfaces
      timed python3 scripts/benchmark_applications.py --datasets airway thyroid32 thyroid128 --sceptre-datasets \
        --repeats 3 --cpus 1:16 8:16-23 --work $RUNS --output $R/applications_bulk_small.json \
        > $R/applications_bulk_small.log 2>&1
      step "bulk small exit $?"
      timed python3 scripts/benchmark_applications.py --datasets thyroid --sceptre-datasets \
        --repeats 2 --cpus 1:16 8:16-23 --work $RUNS --output $R/applications_bulk_large.json \
        > $R/applications_bulk_large.log 2>&1;;
    cooks)
      taskset -c 0-7 python3 scripts/validate_cooks_flags.py --threads 8 --output $R/COOKS_FLAGS.json > $R/cooks_flags.log 2>&1
      cp -n validation/COOKS_THYROID.json validation/COOKS_LEGACY.json $R/
      taskset -c 0-7 python3 scripts/validate_cooks.py --runs $RUNS --thyroid-output $R/COOKS_THYROID.json \
        --legacy-output $R/COOKS_LEGACY.json > $R/cooks.log 2>&1;;
    rust)
      for n in airway thyroid32 thyroid128 thyroid; do
        d=validation/work/benchmark/$n; [ $n = airway ] && d=validation/work/bulk/airway
        extra=; [ $n = thyroid32 ] && extra="--deseq2 $RUNS/thyroid32/deseq2-t1/genes.tsv"
        RAPIDGLM_C_LIBRARY=$PWD/build/librapidglm_c.so PYTHONPATH=$PWD/bindings/python \
          python3 scripts/quiet_run.py -- taskset -c $EIGHT validation/local-data/pydeseq2-venv/bin/python \
          bindings/rust/tests/compare_pydeseq2.py $d --json $R/rust_$n.json $extra > $R/rust_$n.log 2>&1
        step "rust $n exit $?"
      done;;
    staging)
      python3 scripts/quiet_run.py -- taskset -c $EIGHT python3 scripts/validate_sceptre_staging.py --threads 8 \
        --work validation/work/staging-v1.4.0 --output $R/SCEPTRE_STAGING.all.json \
        --sceptre-library $SCEPTRE_LIBRARY > $R/sceptre_staging.log 2>&1;;
    pert50k)
      timed python3 scripts/benchmark_applications.py --datasets --sceptre-datasets gasperini50k --repeats 1 \
        --cpus 1:16 8:16-23 --work $RUNS --output $R/perturbation_50k.json > $R/perturbation_50k.log 2>&1;;
    pertfull)
      timed python3 scripts/benchmark_applications.py --datasets --sceptre-datasets gasperini --threads 8 --repeats 1 \
        --cpus 8:16-23 --work $RUNS --output $R/perturbation_full.json > $R/perturbation_full.log 2>&1;;
    seeds)  # after pertfull: the ks run of the full screen under six seeds against SCEPTRE
      python3 scripts/quiet_run.py -- taskset -c $EIGHT python3 scripts/validate_perturbation_seeds.py --runs $RUNS \
        --work $RUNS/seeds --threads 8 --output $R/PERTURBATION_SEEDS.json > $R/perturbation_seeds.log 2>&1;;
    untestable)
      python3 scripts/quiet_run.py -- python3 scripts/benchmark_untestable_pairs.py --work $RUNS/untestable --binary build/rapidglm \
        --label 1.4.0 --threads 12 --cpus 20-31 --output $R/UNTESTABLE_PAIRS.json > $R/untestable.log 2>&1;;
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
          step "crt $panel $variant exit $? wall $(grep Elapsed $d/time.log | awk '{print $NF}') $(cat $d/host.json)"
        done
      done;;
    replogle)  # the K562 essential low-MOI pilot: single cell (16 threads) and pseudobulk (one thread)
      for fc in strict ks; do
        python3 scripts/quiet_run.py -- python3 scripts/replogle_pilot.py sc --pilot $PILOT --cpus $ALL --threads 16 --fit-check $fc \
          > $R/replogle_sc_$fc.log 2>&1
        step "replogle sc $fc exit $?"
      done
      python3 scripts/quiet_run.py -- python3 scripts/replogle_pilot.py sceptre --pilot $PILOT --cpus $ALL --threads 16 \
        > $R/replogle_sceptre.log 2>&1
      step "replogle sceptre exit $?"
      python3 scripts/quiet_run.py -- python3 scripts/replogle_pilot.py pseudobulk --pilot $PILOT --cpus $ONE --threads 1 \
        > $R/replogle_pseudobulk.log 2>&1
      step "replogle pseudobulk exit $?"
      for d in $PILOT/pseudobulk/datasets/*; do
        taskset -c 0-7 Rscript scripts/reference_edger_fits.R "$d" perturbed "$d/edger" \
          "$PWD/validation/local-data/R_glmgampoi:$PWD/validation/local-data/R_library" > "$d/edger/fit_check.log" 2>&1
      done
      for fc in strict ks; do
        python3 scripts/replogle_pilot.py compare --pilot $PILOT --threads 16 --fit-check $fc \
          --summary $R/replogle_sc_$fc.json > /dev/null 2>&1
      done
      python3 scripts/replogle_pilot.py compare --pilot $PILOT --threads 1 --summary $R/replogle_pseudobulk.json \
        > /dev/null 2>&1;;
    eqtl)  # chr22 of Geuvadis: RapidGLM qtl, FastQTL, tensorQTL and jaxQTL on twelve efficiency cores
      E=$RUNS/eqtl; mkdir -p $E
      for fam in gaussian nb; do
        if [ $fam = nb ]; then ph=$EQTL/runs/rapidglm/inputs/chr22.counts.bed
          lib="\"library_sizes\": \"$EQTL/runs/rapidglm/inputs/library_sizes.tsv\","
        else ph=$EQTL/runs/rapidglm/inputs/chr22.expr_int.bed; lib=""; fi
        mkdir -p $E/rapidglm_$fam
        cat > $E/rapidglm_$fam/config.json <<CFG
{"phenotypes": "$ph", "genotypes": "$EQTL/data/genotypes/chr22", "covariates": "$EQTL/data/covariates/covariates.fastqtl.txt", $lib
 "family": "$fam", "window": 1000000, "permutations": 1000, "seed": 20261016, "threads": 12, "output": "$PWD/$E/rapidglm_$fam/chr22"}
CFG
        guarded $TWELVE $E/rapidglm_$fam build/rapidglm qtl $E/rapidglm_$fam/config.json
        step "eqtl rapidglm $fam exit $?"
      done
      mkdir -p $E/rapidglm_gaussian10000
      python3 - "$E" <<'PYQ'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);c=json.loads((p/'rapidglm_gaussian/config.json').read_text())
c['permutations']=10000;c['output']=str((p/'rapidglm_gaussian10000/chr22').resolve())
(p/'rapidglm_gaussian10000/config.json').write_text(json.dumps(c))
PYQ
      guarded $TWELVE $E/rapidglm_gaussian10000 build/rapidglm qtl $E/rapidglm_gaussian10000/config.json
      # FastQTL (GTEx fork 59bcdc0): adaptive permutations, its own 12 chunks as 12 processes.
      mkdir -p $E/fastqtl/chunks
      guarded $TWELVE $E/fastqtl sh -c "seq 1 12 | xargs -P 12 -I K $EQTL/tools/fastQTL --vcf $EQTL/data/genotypes/chr22.vcf.gz \
        --bed $EQTL/data/expression/chr22.expr_int.bed.gz --cov $EQTL/data/covariates/covariates.fastqtl.txt.gz --window 1e6 \
        --seed 20261016 --permute 1000 10000 --chunk K 12 --out $E/fastqtl/chunks/chunk_K.txt.gz --log $E/fastqtl/chunks/chunk_K.log"
      step "eqtl fastqtl exit $?"
      # tensorQTL 1.0.10 (CPU torch): cis mode, 10,000 permutations.
      guarded $TWELVE $E/tensorqtl $EQTL/venv-tensorqtl/bin/python -I -m tensorqtl $EQTL/data/genotypes/chr22 \
        $EQTL/data/expression/chr22.expr_int.bed.gz chr22 --covariates $EQTL/data/covariates/covariates.fastqtl.txt \
        --mode cis --window 1000000 --seed 20261016 -o $PWD/$E/tensorqtl
      step "eqtl tensorqtl exit $?"
      # jaxQTL 0.3.1: the NB score test with SPA and ACAT, and with 1000 permutations; four processes of
      # three cores each (one process on twelve cores is slower), genes split as in the reference runs.
      for spec in "nb_score_spa_acat --test score --spa --acat" "nb_score_perm1000 --test score --nperm 1000"; do
        set -- $spec; name=$1; shift
        guarded $TWELVE $E/jaxqtl_$name bash $R/jaxqtl_split.sh $PWD/$E/jaxqtl_$name "$@"
        step "eqtl jaxqtl $name exit $?"
      done
      taskset -c 0-7 $EQTL/venv/bin/python scripts/summarize_review_eqtl.py --runs $E --output $E/reference_gene_level.tsv.gz
      taskset -c 0-7 python3 scripts/compare_eqtl.py --reference $E/reference_gene_level.tsv.gz --run gaussian $E/rapidglm_gaussian/chr22 \
        --run gaussian10000 $E/rapidglm_gaussian10000/chr22 --run nb $E/rapidglm_nb/chr22 \
        --output $R/EQTL_COMPARE.json > $R/compare_eqtl.log 2>&1;;
    *) echo "unknown stage $stage" >&2; exit 2;;
  esac
  step "$stage exit $?"
  taskset -c 0-7 python3 scripts/stamp_review_receipts.py $R
done
