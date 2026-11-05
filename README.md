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

The current released version is **1.6.0**. The tables below report the completed 1.6.0 measurements.

RapidGLM **1.6.0** measurements on development tag `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`). All 24 stages passed the timing audit; the release library is unchanged from the measured dev.2 source.

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5. Exclusive Slurm allocation; schedutil governor, boost enabled and SMT enabled, unchanged. Each timed command starts after two ten-second quiet samples. More than 5% other CPU activity on selected cores or any sibling invalidates the command. Host load, other-user processes and vLLM CPU use are retained per command. From the resumed bulk stage on 2026-11-03, a user-authorized runtime service cpuset confines vLLM to CPUs 0-5,22-29,46-47; the reservation receipt records its application and restoration. Earlier accepted measurements retain their original guarded host conditions.

The table gives wall seconds at one / eight threads through the R interface. Jaccard is discovery overlap at FDR 0.1. SCEPTRE comparisons use matched skew-normal acceptance (`fit_check: "ks"`); the stricter default is reported separately. API timings exclude input loading; CLI measurements are in the benchmark report.

| Analysis | Reference (1 / 8) | RapidGLM R (1 / 8) | Speedup | Jaccard |
| --- | --- | --- | --- | --- |
| DESeq2, airway (8 samples) | 5.12 / 2.48 | 0.847 / 0.244 | 6.05× (1 thread) | 0.9862 |
| DESeq2, thyroid (128 samples) | 95.9 / 26.3 | 8.27 / 1.51 | 11.6× (1 thread) | 0.9975 |
| DESeq2, thyroid (647 samples) | 401 / 70.2 | 30 / 5.54 | 13.4× (1 thread) | 0.9999 |
| SCEPTRE, high MOI (50,000 cells) | 1.38e+03 / 207 | 246 / 35.9 | 5.6× (1 thread) | 0.8462 |
| SCEPTRE, high MOI (207,324 cells) | — / 1.11e+03 | — / 182 | 6.1× (8 threads) | 0.8952 |

With R-compatible aliasing, the engine fits 39 of 39 identifiable binary128-reference problems; maximum coefficient error is 1.729e-06 reference standard errors. See the [GLM comparison](docs/GLM_COMPARISON.md) and [NB solvers](docs/SOLVERS.md).

Gaussian cis-QTL, both tools at 10,000 permutations on twelve cores:

| Scope | RapidGLM seconds | tensorQTL seconds | tensorQTL / RapidGLM |
| --- | --- | --- | --- |
| Chromosome 22 | 35.86 | 107 | 2.983× |
| Genome-wide | 1212 | 3069 | 2.531× |


The full Replogle pseudobulk screen, sixteen single-thread workers per tool:

| Workflow | Reference / RapidGLM screen time | Median discovery Jaccard |
| --- | --- | --- |
| DESeq2-style bulk DE | 5.123× | 1 |
| edgeR-style quasi-likelihood | 7.544× | 1 |


[Application benchmarks](docs/BENCHMARK.md), [bulk agreement](docs/BULK_DE.md) and [QTL details](docs/QTL_GAUSSIAN_SPEED.md) give timing scopes, agreements and limitations. The [receipt collection](validation/receipts/v1.6.0/README.md) retains tests, hashes, exits, quiet starts, CPU-policy checks and reruns.

The previous 1.4.0/1.4.1 receipts used a different workstation. Compare reference/tool ratios measured on the same host; changes in absolute seconds do not isolate code changes.

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
