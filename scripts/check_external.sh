#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build="${BUILD_DIR:-build}"
[[ "$build" = /* ]] || build="$PWD/$build"
prefix="$build/install"
mapfile -t deps < <(scripts/local_deps.sh)
export OPENBLAS_NUM_THREADS=1
cmake --install "$build" --prefix "$prefix"
cmake -S examples/external -B "$build/external" -DCMAKE_PREFIX_PATH="$prefix" "${deps[@]}" "$@"
cmake --build "$build/external" -j 2
"$build/external/consumer"
"$build/external/c_consumer"
