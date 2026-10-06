#!/usr/bin/env bash
# Builds and tests the R and Python drop-in bindings against stats::glm,
# MASS::glm.nb and statsmodels. R_LIBS may name libraries holding cpp11/MASS;
# PYBIND11_DIR may point at pybind11's CMake directory.
set -euo pipefail
cd "$(dirname "$0")/.."
build="${BUILD_DIR:-build}"
export OPENBLAS_NUM_THREADS=1
mapfile -t deps < <(scripts/local_deps.sh)
pybind="${PYBIND11_DIR:-$(python3 -m pybind11 --cmakedir)}"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DRAPIDGLM_PYTHON=ON -Dpybind11_DIR="$pybind" \
  -DPython_EXECUTABLE="$(command -v python3)" "${deps[@]}"
cmake --build "$build" --target _rapidglm -j "${BUILD_JOBS:-2}"
PYTHONPATH="$PWD/bindings/python${PYTHONPATH:+:$PYTHONPATH}" python3 bindings/python/tests/test_statsmodels.py
cmake --build "$build" --target rapidglm_c rapidglm_cli -j "${BUILD_JOBS:-2}"
RAPIDGLM_C_LIBRARY="$PWD/$build/librapidglm_c.so" PYTHONPATH="$PWD/bindings/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 bindings/python/tests/test_capi.py "$build/rapidglm"
RAPIDGLM_C_LIBRARY="$PWD/$build/librapidglm_c.so" PYTHONPATH="$PWD/bindings/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 bindings/python/tests/test_deseq.py "$build/rapidglm"
library="$(mktemp -d)"
trap 'rm -rf "$library"' EXIT
R CMD INSTALL --no-test-load --library="$library" bindings/r
R_LIBS="$library${R_LIBS:+:$R_LIBS}" Rscript bindings/r/tests/parity.R
cmake --build "$build" --target rapidglm_cli -j "${BUILD_JOBS:-2}"
R_LIBS="$library${R_LIBS:+:$R_LIBS}" RAPIDGLM_CLI="$PWD/$build/rapidglm" Rscript bindings/r/tests/applications.R
