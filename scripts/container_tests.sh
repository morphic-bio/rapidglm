#!/usr/bin/env bash
# Every test suite, as run in the container (Dockerfile): C/C++ unit tests, CLI
# tests, the installed CMake package with C and C++ consumers, the Python layer,
# the PyDESeq2 back end (when PyDESeq2 is installed) and the R package.
set -euo pipefail
cd "$(dirname "$0")/.."
build="${BUILD_DIR:-build}"
export OPENBLAS_NUM_THREADS=1
step() { printf '\n== %s\n' "$*"; }
step "C/C++ unit tests"
ctest --test-dir "$build" --output-on-failure
step "CLI tests"
python3 tests/test_cli.py "$build/rapidglm"
step "Installed CMake package (C++ and C consumers)"
scripts/check_external.sh
step "Python layer over the C interface"
python3 bindings/python/tests/test_capi.py "$build/rapidglm"
python3 bindings/python/tests/test_deseq.py "$build/rapidglm"
if [ -x /opt/pydeseq2/bin/python ]; then
  step "RapidGLM as a PyDESeq2 inference back end"
  /opt/pydeseq2/bin/python bindings/python/tests/test_pydeseq2.py
fi
step "R package"
Rscript bindings/r/tests/parity.R
RAPIDGLM_CLI="$PWD/$build/rapidglm" Rscript bindings/r/tests/applications.R
step "All test suites passed"
