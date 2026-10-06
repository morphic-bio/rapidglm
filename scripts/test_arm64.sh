#!/usr/bin/env bash
# Cross-compiles the unit tests for arm64 and runs them under QEMU user
# emulation, which implements the Arm AES instructions. Run it in a fresh amd64
# Ubuntu 24.04 container from the repository root:
#
#   docker run --rm -v "$PWD":/src:ro ubuntu:24.04 bash /src/scripts/test_arm64.sh
#
# The optional argument is a Boost.Test filter; the default runs the tests of
# the random generators and the samplers, and "all" runs every test (slow under
# emulation). The script installs packages and edits the APT sources, so it
# refuses to run outside a container.
set -euo pipefail
filter="${1:-ars_*,philox_*,exponential_ziggurat*,thinned_*,hazard_*,batched_*}"
[ -f /.dockerenv ] || { echo "run this in a container (see the header)" >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive
dpkg --add-architecture arm64
sed -i 's/^Types: deb$/Types: deb\nArchitectures: amd64/' /etc/apt/sources.list.d/ubuntu.sources
cat > /etc/apt/sources.list.d/arm64.sources <<SOURCES
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble noble-updates
Components: main universe
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
SOURCES
apt-get update -qq
apt-get install -y -qq --no-install-recommends g++-aarch64-linux-gnu qemu-user libboost-dev \
    liblapacke-dev:arm64 libopenblas-pthread-dev:arm64 libstdc++6:arm64 libgomp1:arm64 > /dev/null
cd "$(dirname "$0")/.."
build="$(mktemp -d)"
sources="$(sed -n 's/^ *add_executable(rapidglm_tests \(.*\))$/\1/p' CMakeLists.txt) src/c_api.cpp"
[ -n "${sources% src/c_api.cpp}" ] || { echo "no test sources found in CMakeLists.txt" >&2; exit 2; }
echo "$sources" | tr ' ' '\n' | xargs -P "$(nproc)" -I{} sh -c \
    'aarch64-linux-gnu-g++ -std=c++17 -O2 -DNDEBUG -fopenmp -Wno-psabi -Iinclude -c "$1" -o "$2/$(basename "$1" .cpp).o"' \
    sh {} "$build"
aarch64-linux-gnu-g++ -fopenmp "$build"/*.o -o "$build/rapidglm_tests" -llapacke -lopenblas
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
[ "$filter" = all ] && set -- || set -- "--run_test=$filter"
qemu-aarch64 -cpu max "$build/rapidglm_tests" --log_level=message "$@"
