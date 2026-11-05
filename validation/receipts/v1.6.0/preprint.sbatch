#!/usr/bin/env bash
#SBATCH --job-name=rapidglm-preprint-160
#SBATCH --exclusive
#SBATCH --time=UNLIMITED
#SBATCH --output=/data/lhhung/rapidglm-1.6.0/validation/work/release-1.6.0-gpu/logs/preprint-%j.log
set -euo pipefail
source /data/lhhung/env/env.sh
cd /data/lhhung/rapidglm-1.6.0
unset OMP_NUM_THREADS
exec taskset -c 0-5 nice -n 10 python3 validation/receipts/v1.6.0/preprint_chain.py
