#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build="${BUILD_DIR:-build}"
mapfile -t deps < <(scripts/local_deps.sh)
# Tests call BLAS concurrently from OpenMP workers; see README.
export OPENBLAS_NUM_THREADS=1
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release "${deps[@]}"
cmake --build "$build" -j "${BUILD_JOBS:-2}"
ctest --test-dir "$build" --output-on-failure
python3 tests/test_cli.py "$build/rapidglm"
