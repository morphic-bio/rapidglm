# RapidGLM

A fast and portable library for count regression. RapidGLM is an MIT-licensed
C++17/OpenMP engine for Gaussian, Poisson, binomial and negative binomial
generalized linear models. It fits many responses that share a design, with
Wald, likelihood ratio, score and quasi-likelihood tests and deterministic
permutation or conditional-randomization calibration. Results are identical at
any thread count.

Applications are built from published methods and documentation; GPL reference
implementations are run only through their public interfaces:
- **bulk differential expression** following the DESeq2 procedure (`bulk_de`);
- **perturbation-screen analysis** following the SCEPTRE method (`sceptre`), for
  a list of pairs or for every pair of named targets and responses
  (`sceptre_grid`). A pair's result depends only on the seed, its response and
  its target, not on what else is tested or in what order;
- **quasi-likelihood differential expression** following edgeR 4 (`edger_ql`);
- **cis-QTL mapping** with Gaussian, Poisson and NB score tests (`cis_qtl`).

One library serves several interfaces:
- a C++ header library and a CLI;
- a stable C ABI;
- Python and R packages;
- an inference back end for PyDESeq2.

## Benchmarks

The current release is **1.4.1**, which fixes undefined behavior for a
zero-column weighted cross-product. The benchmark tables and receipts below
remain **1.4.0 measurements** on `f0ecfbd` (`v1.4.0-dev.4`); they have not been
remeasured for 1.4.1. [Patch validation](validation/receipts/v1.4.1/README.md).

Version 1.4.0 uses the library code of `v1.4.0-dev.4`. All figures below come
from the complete final release run, completed 2026-10-22; the
[release receipts](validation/receipts/v1.4.0/README.md) retain source hashes,
tests, all 18 stages and quiet-start evidence.

One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.
[Host details](docs/BENCHMARK.md#host). The table gives wall seconds at one /
eight threads. RapidGLM uses its R interface here. Jaccard measures discovery
overlap at FDR 0.1. The SCEPTRE rows use matched skew-normal acceptance
(`fit_check: "ks"`); the stricter RapidGLM default adds confirmation draws.
The R rows time API calls after input loading; SCEPTRE rows sum calibration,
power and discovery calls, excluding setup.

| Analysis | Reference (1 / 8) | RapidGLM (1 / 8) | Speedup | Jaccard |
|---|---|---|---|---|
| DESeq2 1.48.2, airway (8 samples) | 6.07 / 2.63 | 0.966 / 0.238 | 6.3× (1 thread) | 0.986 |
| DESeq2, thyroid (128 samples) | 118 / 31.3 | 9.87 / 1.75 | 12.0× (1 thread) | 0.997 |
| DESeq2, thyroid (647 samples) | 483 / 83.7 | 35.1 / 6.51 | 13.8× (1 thread) | 1.000 |
| SCEPTRE 0.99.0, high MOI (50,000 cells) | 1,524 / 251 | 333 / 59.6 | 4.6× (1 thread) | 0.79 |
| SCEPTRE, high MOI (207,324 cells) | – / 1,472 | – / 323 | 4.6× (8 threads) | 0.89 |

The speedup depends on the workload and acceptance rule. On the low-MOI
Replogle pilot at 16 threads, whole-command speedups over SCEPTRE are
4.38× with `ks` and
1.63× with the strict default.
[The benchmark](docs/BENCHMARK.md) reports both rules and interfaces separately.
Across six high-MOI seeds, Jaccard with SCEPTRE ranges from
0.89 to 0.96.

- **GLM engine:** with R-compatible aliasing, all 39
  identifiable accuracy problems fit;
  maximum coefficient error 6.6e-07 in units of the reference standard error.
  Median single-fit speedups are 5.4× over R `glm.fit`,
  8.3× over statsmodels and 3.2× over fastglm
  (one thread, n ≥ 100,000; [GLM comparison](docs/GLM_COMPARISON.md)).
- **NB solvers:** 2.2–2.8× over tight Levenberg–Marquardt with fixed
  dispersion and 1.7–5.8× over tight BFGS with estimated dispersion
  on identical problems ([solver comparison](docs/SOLVERS.md)).
- **Bulk agreement:** Jaccard 0.983–1.000
  with DESeq2 on five validation datasets ([bulk DE](docs/BULK_DE.md)).
- **Gaussian cis-QTL:** 1.29× over tensorQTL at
  10,000 permutations on Geuvadis chromosome 22, with the agreement and scope in
  [the QTL report](docs/QTL_GAUSSIAN_SPEED.md).

Datasets, tool versions, timing scopes and all current tables are in
[the benchmark](docs/BENCHMARK.md). Every current figure comes from a final
receipt under [validation/](validation).

## Quick start

The container builds everything and runs every test suite:

```bash
docker build -t rapidglm .
docker run --rm rapidglm                                  # all tests
docker run --rm -v "$PWD":/data rapidglm rapidglm bulk-de /data/config.json
```

It holds the CLI and libraries in `/usr/local`, Python with `rapidglm` on its
path, and R with the `rapidglm` package. For Singularity or Apptainer:
`singularity build rapidglm.sif docker-daemon://rapidglm:latest`.

Python ([bindings/python](bindings/python)):

```python
import rapidglm
res = rapidglm.bulk_de(design, counts, lfc_shrink="normal", threads=8)   # counts: samples x genes
hits = rapidglm.sceptre(design, responses, assignments, guides, pairs, moi="high", threads=8)

from rapidglm.deseq import DeseqDataSet, DeseqStats                      # PyDESeq2-style interface
from rapidglm.pydeseq2 import RapidGLMInference                          # back end for PyDESeq2 itself
```

R ([bindings/r](bindings/r)):

```r
res <- rapidglm::bulk_de(counts, ~ cell + dex, col_data, contrast = c("dex", "trt", "untrt"))
fit <- glm(y ~ x, family = poisson, data = d, method = rapidglm::glm_fit)
```

C ([docs/C_API.md](docs/C_API.md)) and the CLI ([docs/CLI.md](docs/CLI.md))
expose the same functions.

## Interfaces and workflows

- *Score scans* (`scan`, `score_scan`) are the engine's core pattern. Each
  response's null model is fitted once, many predictors are tested against it,
  and the test is calibrated by asymptotic score, permutation or CRT
  ([docs/CLI.md](docs/CLI.md#score-scans)). The perturbation, pseudobulk,
  accessibility and dose workflows are thin layers over it.
- *Pseudobulk DE* defaults to a quasi-likelihood F test with empirical Bayes
  dispersion shrinkage ([docs/QL_TEST.md](docs/QL_TEST.md)). Tests take a
  coefficient, a contrast or several columns jointly
  ([docs/HYPOTHESES.md](docs/HYPOTHESES.md)).
- *Bulk DE* (`bulk-de`, `bulk_de`) covers size factors, shrunken dispersions,
  Wald or LR tests, Cook's outliers, independent filtering and optional
  normal-prior fold-change shrinkage ([docs/BULK_DE.md](docs/BULK_DE.md)).
- *Quasi-likelihood DE* (`edger-ql`, `edger_ql`) follows edgeR 4's default
  workflow: filtering by expression, TMM, quasi-dispersions from adjusted
  unit deviances and moderated F tests
  ([docs/CLI.md](docs/CLI.md#quasi-likelihood-differential-expression-edgers-procedure)).
- *Drop-in GLM interfaces:* `glm(..., method = rapidglm::glm_fit)` and
  `rapidglm::glm_nb()` in R, and `rapidglm.statsmodels.GLM` in Python.
  `scripts/check_bindings.sh` builds and tests both.
- *C interface* (`librapidglm_c`): a stable C ABI over GLM fitting, `bulk_de`,
  `sceptre` and `score_scan`. Python, R and Rust build on it; other languages
  can bind to it the same way ([docs/C_API.md](docs/C_API.md#language-bindings)).
- *Rust* ([bindings/rust](bindings/rust)): bindings over the C interface and a
  port of PyDESeq2's workflow whose per-gene fits run in RapidGLM.
- *Special functions* (`include/rapidglm/vmath.hpp`, a standalone header):
  RapidGLM's vectorized `log`, `log1p`, `exp`, `digamma` and `trigamma`, the
  same bits on every processor within each division policy. `lgamma` uses
  the C library in slow mode and RapidGLM's positive-argument kernel in fast
  mode (negative arguments keep libm); also
  `rapidglm_<f>_array` in the C interface, `rapidglm.special` in Python and
  `special_<f>()` in R ([METHODS.md](METHODS.md#numerical-backend-special-functions)).

The numerical policy defaults to `slow`. Set `RAPIDGLM_DIVISION=fast` before
first use or pass an explicit `division="fast"` option for chips with fast
division. [Backend choices](METHODS.md#division-policy) records the alternatives
and accuracy checks; [CLI](docs/CLI.md) and [C API](docs/C_API.md) document the
per-call options.

## Continuous integration and releases

GitHub Actions ([.github/workflows](.github/workflows)) runs on every push and
pull request:
- builds with unit and CLI tests in Release and ASan/UBSan configurations, each
  with and without OpenMP, on Ubuntu 24.04 with system LAPACKE and pthread
  OpenBLAS;
- checks that the installed CMake package builds external C++ and C consumers;
- runs the Python layer, the PyDESeq2 back end, the Rust binding and the
  container's test suites;
- checks formatting with clang-format 23.1.1.

Pushing a tag `vX.Y.Z` that matches the CMake project version builds and tests
it, then publishes a release with the headers, CMake package, libraries, a
Linux CLI build and the source archive. Release notes are in
[CHANGELOG.md](CHANGELOG.md). When the repository variable `PUBLISH_CONTAINER` is
`true`, the release also pushes a multi-arch (amd64 and arm64) container to
`ghcr.io`.

## Building

Requires CMake, a C++17 compiler, Boost, LAPACKE and a BLAS/LAPACK implementation;
OpenMP is optional. On Debian/Ubuntu the numerical development packages are
`liblapacke-dev` and `libopenblas-pthread-dev`. Without them, `scripts/ci.sh` uses
packages unpacked under `validation/local-data/numerical-deps` (or
`RAPIDGLM_DEPS_ROOT`); `BUILD_DIR` selects the build directory. R and Python are
only needed for the bindings and the independent validation harnesses. The public
headers link these numerical libraries through the exported `rapidglm::rapidglm`
CMake target. Cholesky and triangular solves are done in-house by default;
`-DRAPIDGLM_LAPACK_SOLVES=ON` uses LAPACK for large systems instead, with the
caveats in [docs/CLI.md](docs/CLI.md).

### Cross-compiling for aarch64 Linux

From an x86 Linux host with Docker:

```bash
mkdir -p build-arm64
docker run --rm --platform linux/amd64 -v "$PWD":/src:ro \
  -v "$PWD/build-arm64":/out ubuntu:24.04 bash /src/scripts/build_arm64.sh
```

The script installs the cross compiler and arm64 LAPACKE/OpenBLAS, then uses
[`cmake/toolchain-aarch64-linux-gnu.cmake`](cmake/toolchain-aarch64-linux-gnu.cmake)
to build the header-only engine's consumers, shared and static C libraries,
command-line program and tests. It checks the artifacts' architecture and runs
selected unit tests, a C99 caller of the shared library and the small CLI suite
under QEMU. When the five local edgeR reference datasets are available, it also
runs their agreement checks. Artifacts, logs and elapsed times go in
`build-arm64/`; the source mount stays read-only. Pass `all` after the script
name for the full C++ suite, which is slow under emulation.

The final selected ARM64 suite is recorded in
[ARM64.json](validation/ARM64.json): 29 C++ cases with 5,281,712 assertions,
28 CLI cases and the C99 caller pass under QEMU. The clean clone does not
mount external edgeR comparison tables. The earlier full cross-architecture
application study remains explicitly historical in
[ARM64_DEV3.json](validation/ARM64_DEV3.json); its numerical differences are
not relabeled as final-run measurements. These checks establish functionality
under emulation, without an ARM hardware performance measurement.

ARM uses the portable kernels in place of the x86 AVX2 and SHA paths. Compare
application results with numerical tolerances: BLAS and floating-point rounding
can differ. Integer resampling streams are identical; floating-point statistics
can round differently. Linux aarch64 uses software binary128
`long double`, so some paths are slower, especially under QEMU. The optional
binary128 benchmark reference fitter is excluded; ARM performance and the R
and Python packages are outside these checks.

### Threading and numerical benchmarks

Parallelism is across response features, controlled by `threads`. The perturbation
workflow also schedules pairs and resampling draws as tasks within the same OpenMP
worker team; results do not depend on the thread count.
Use a **thread-safe** BLAS configured with one internal worker, e.g. pthread
OpenBLAS with `OPENBLAS_NUM_THREADS=1`, when using the outer OpenMP loops.
A serial OpenBLAS build needs `USE_LOCKING=1` to support concurrent callers;
setting the environment variable cannot add missing locking. The numerical
tests exercise concurrent factorization, solve and inverse calls.

For isolated fitter, factorization, RNG and resampling timings:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAPIDGLM_BENCHMARKS=ON
cmake --build build -j 2
python3 scripts/benchmark_kernels.py
```

The benchmark repeats each case in five processes and records timing ranges,
checksums and host load. `--baseline /path/to/old/rapidglm_benchmarks` alternates
the two executables. These measurements exclude setup and are not end-to-end
workflow timings.

- [CLI, file formats and C++ usage](docs/CLI.md)
- [Statistical contract and clean-room sources](METHODS.md)
- [Dependency licences](DEPENDENCIES.md)
- [Comparison with R glm, MASS glm.nb, statsmodels and others](docs/GLM_COMPARISON.md)
- [Paired R/native boundary and thread-safety diagnosis](docs/BOUNDARY_DIAGNOSTIC.md)

## Validation and status

RapidGLM is validated by continuous integration, by a receipt for each feature
under [validation/](validation), and by comparison with the reference tools run
as black boxes. Agreement is reported as measured, per feature, not as a
qualification. An early acceptance plan was retired on 2026-09-30 with two of
its calibration checks not met: a Kolmogorov-Smirnov bound for the Poisson CRT
with known propensities (D = 0.057; [CALIBRATION.json](validation/CALIBRATION.json))
and one of eight bounds on simulated dense nulls (107 rejections in 10,000 at
level 0.01, an upper limit of 0.0136 against 0.013;
[S6_CALIBRATION.json](validation/S6_CALIBRATION.json)).

Skew-normal p-values of the perturbation workflow are extrapolations beyond
the resampled draws. From 1.0.0 they are accepted by a stricter check, after
direct counts over millions of draws contradicted some that the earlier check
accepted, and from 1.1.0 the smallest are confirmed on 199,999 draws
([SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)). The checks
reduce such cases; they cannot rule them out
([METHODS.md](METHODS.md#resampling)).

Force the portable kernels, integer generators and SHA-1 path on x86 with
`cmake -S . -B build-portable -DRAPIDGLM_FORCE_PORTABLE=ON` and run both
`ctest --test-dir build-portable` and `python3 tests/test_cli.py build-portable/rapidglm`.
CI includes this configuration; the whole ARM64 suite runs under QEMU on the
weekly schedule, manual runs and tags, with the short sampler suite on pushes.

Cis-QTL mapping is available as `rapidglm qtl config.json` for Gaussian,
Poisson and NB response models, using phenotype BED and PLINK 1 genotypes.
It reports nominal scores and permutation-calibrated gene p-values. See
[configuration and output columns](docs/CLI.md#cis-qtl-mapping-qtl) and
[the method](METHODS.md#cis-qtl-mapping). PLINK A1 is the counted allele;
output starts use 1-based coordinates.

On ARM64 the skew-normal KS discrepancy uses the double-precision identity
Phi(z) − 2 T(z, shape), checked against the promoted Boost CDF. Inferential
tails retain the distribution CDF. This avoids per-draw software binary128
work in the KS check; integer resampling streams retain the same definitions.
