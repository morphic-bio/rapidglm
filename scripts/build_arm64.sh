#!/usr/bin/env bash
# Cross-build the library, C interface, CLI and tests in a fresh amd64 container:
#   mkdir -p build-arm64
#   docker run --rm --platform linux/amd64 -v "$PWD":/src:ro \
#     -v "$PWD/build-arm64":/out ubuntu:24.04 bash /src/scripts/build_arm64.sh
# Add --cpuset-cpus to docker run when sharing the host with other work.
# The optional argument is a Boost.Test filter; "all" runs the full suite (slow).
# EdgeR agreement also runs when the five local reference datasets are present.
set -euo pipefail
[ -f /.dockerenv ] || { echo "run this in a container (see the header)" >&2; exit 2; }
[ "$(dpkg --print-architecture)" = amd64 ] || { echo "use an amd64 Ubuntu 24.04 container" >&2; exit 2; }
[ -d /out ] && [ -w /out ] || { echo "mount a writable output directory at /out" >&2; exit 2; }
filter="${1:-ars_*,philox_*,exponential_ziggurat*,thinned_*,hazard_*,batched_*,crt_draw_*,kernels_agree_with_direct_sums,market_reader_is_independent_of_entry_order,sha1_matches_published_digests,leverages_and_fixed_dispersion_fits,rank_windows_equal_average_rank_trims,edger_ql_finds_changed_genes_and_is_reproducible,count_loader_equals_the_csc_reader,deviance_moment_table_interpolates,vmath_*}"
export DEBIAN_FRONTEND=noninteractive
dpkg --add-architecture arm64
if ! rg_arch=$(sed -n '/^Architectures:/p' /etc/apt/sources.list.d/ubuntu.sources) || [ -z "$rg_arch" ]; then
    sed -i 's/^Types: deb$/Types: deb\nArchitectures: amd64/' /etc/apt/sources.list.d/ubuntu.sources
fi
cat > /etc/apt/sources.list.d/arm64.sources <<SOURCES
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble noble-updates
Components: main universe
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
SOURCES
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake make file git g++-aarch64-linux-gnu \
    qemu-user libboost-dev python3-numpy python3-pandas \
    liblapacke-dev:arm64 libopenblas-pthread-dev:arm64 libstdc++6:arm64 libgomp1:arm64 > /dev/null

aarch64-linux-gnu-g++ --version > /out/compiler.txt
qemu-aarch64 --version > /out/qemu.txt
printf '%s\n' 'qemu-aarch64 -cpu max' > /out/emulated-cpu.txt
cd "$(dirname "$0")/.."
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
printf 'phase\tseconds\n' > /out/timings.tsv
start=$SECONDS
cmake -S . -B "$build" -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchain-aarch64-linux-gnu.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DRAPIDGLM_TESTS=ON -DRAPIDGLM_BENCHMARKS=OFF \
    -DBoost_INCLUDE_DIR=/usr/include -DLAPACKE_INCLUDE_DIR=/usr/include \
    -DLAPACKE_LIBRARY=/usr/lib/aarch64-linux-gnu/liblapacke.so \
    -DBLAS_openblas_LIBRARY=/usr/lib/aarch64-linux-gnu/libopenblas.so \
    -DLAPACK_openblas_LIBRARY=/usr/lib/aarch64-linux-gnu/libopenblas.so
cmake --build "$build" --parallel "${BUILD_JOBS:-4}" --target \
    rapidglm_c rapidglm_c_static rapidglm_cli rapidglm_tests rapidglm_c_test
printf 'configure_build\t%s\n' "$((SECONDS - start))" >> /out/timings.tsv

# Keep the versioned shared library and its symlinks, and inspect an archive
# member as well: `file librapidglm_c.a` alone cannot identify its architecture.
cp -a "$build"/librapidglm_c.so* "$build/librapidglm_c.a" \
    "$build/rapidglm" "$build/rapidglm_tests" "$build/rapidglm_c_test" /out/
for artifact in "$build"/librapidglm_c.so* "$build/librapidglm_c.a" \
    "$build/rapidglm" "$build/rapidglm_tests" "$build/rapidglm_c_test"; do
    original=$(sha256sum "$artifact" | cut -d' ' -f1)
    copied=$(sha256sum "/out/$(basename "$artifact")" | cut -d' ' -f1)
    [ "$original" = "$copied" ] || exit 1
done
aarch64-linux-gnu-ar p /out/librapidglm_c.a c_api.cpp.o > "$build/static-member.o"
: > /out/artifacts.txt
for artifact in /out/rapidglm /out/rapidglm_tests /out/rapidglm_c_test \
    /out/librapidglm_c.so.*.*.* "$build/static-member.o"; do
    description="$(file -L "$artifact")"
    echo "$description" | tee -a /out/artifacts.txt
    [[ "$description" == *"ELF 64-bit LSB"* && "$description" == *"ARM aarch64"* ]] || exit 1
done

cat > /out/rapidglm-qemu <<'WRAPPER'
#!/bin/sh
exec qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu -E OPENBLAS_NUM_THREADS=1 "$(dirname "$0")/rapidglm" "$@"
WRAPPER
chmod +x /out/rapidglm-qemu
start=$SECONDS
[ "$filter" = all ] && set -- || set -- "--run_test=$filter"
for mode in slow fast; do
    export RAPIDGLM_DIVISION=$mode
    qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu /out/rapidglm_tests \
        --log_level=test_suite --report_level=short "$@" 2>&1 | tee /out/unit-tests-$mode.log
    ctest --test-dir "$build" -R '^(c_api|division_override)$' --output-on-failure 2>&1 | tee /out/c-api-test-$mode.log
    python3 tests/test_cli.py /out/rapidglm-qemu 2>&1 | tee /out/cli-tests-$mode.log
done
unset RAPIDGLM_DIVISION
printf 'unit_and_cli_tests\t%s\n' "$((SECONDS - start))" >> /out/timings.tsv

have_edger=true
for dataset in airway kang sim3 sim3b sim8; do
    for input in "validation/work/bulk/$dataset/counts.mtx" "validation/work/bulk/$dataset/x.tsv" \
        "validation/work/edger/$dataset/reference.json" "validation/work/edger/$dataset/filter.tsv" \
        "validation/work/edger/$dataset/genes_ql.tsv" "validation/work/edger/$dataset/samples.tsv"; do
        [ -f "$input" ] || have_edger=false
    done
done
if "$have_edger"; then
    start=$SECONDS
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD" \
        python3 scripts/validate_edger.py --binary /out/rapidglm-qemu \
        --datasets airway kang sim3 sim3b sim8 --work /out/edger \
        --output /out/EDGER_QL.arm64.json | tee /out/edger.log
    python3 - <<'PY'
import json
with open('/out/EDGER_QL.arm64.json') as f:
    results = json.load(f)
for name, result in results.items():
    if name.startswith('_'): continue
    assert result['filter_mismatches'] == 0, (name, result)
    assert result['discoveries'] == result['discoveries_edger'] == result['discoveries_shared'], (name, result)
PY
    printf 'edger_agreement\t%s\n' "$((SECONDS - start))" >> /out/timings.tsv
else
    echo 'edgeR agreement skipped: local datasets/reference tables are absent' | tee /out/edger.log
fi
