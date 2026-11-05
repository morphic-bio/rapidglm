#!/usr/bin/env bash
#SBATCH --job-name=rapidglm-160-smoke
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/data/lhhung/rapidglm-1.6.0/validation/work/release-1.6.0-gpu/logs/smoke-%j.log
set -euo pipefail
source /data/lhhung/env/env.sh
cd /data/lhhung/rapidglm-1.6.0
unset OMP_NUM_THREADS
exec taskset -c 0-5 nice -n 10 python3 validation/receipts/v1.6.0/supervise_accepted.py "runner-smoke-${SLURM_JOB_ID}" smoke
