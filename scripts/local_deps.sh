#!/usr/bin/env bash
# Print CMake cache arguments for a LAPACKE/pthread-OpenBLAS tree unpacked from
# Debian packages (see DEPENDENCIES.md). Prints nothing when the tree is absent,
# so system installations are found by the ordinary CMake search.
root="${RAPIDGLM_DEPS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/validation/local-data/numerical-deps/root/usr}"
lib="$root/lib/x86_64-linux-gnu"
blas="$lib/openblas-pthread/libopenblas.so"
if [[ -f "$root/include/lapacke.h" && -f "$lib/liblapacke.a" && -f "$blas" ]]; then
  printf '%s\n' "-DLAPACKE_INCLUDE_DIR=$root/include" "-DLAPACKE_LIBRARY=$lib/liblapacke.a" \
    "-DBLAS_openblas_LIBRARY=$blas" "-DLAPACK_openblas_LIBRARY=$blas"
fi
