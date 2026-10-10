# RapidGLM

A fast and portable library for count regression. RapidGLM is an MIT-licensed
C++17/OpenMP engine for Gaussian, Poisson, binomial and negative binomial
generalized linear models. It fits many responses that share a design, with
Wald, likelihood ratio, score and quasi-likelihood tests and deterministic
permutation or conditional-randomization calibration. Results are identical at
any thread count.

Two widely used methods are built on it, written from their publications and
documentation (the reference implementations were run only as black boxes):
- **bulk differential expression** following the DESeq2 procedure (`bulk_de`);
- **perturbation-screen analysis** following the SCEPTRE method (`sceptre`), for
  a list of pairs or for every pair of named targets and responses
  (`sceptre_grid`). A pair's result depends only on the seed, its response and
  its target, not on what else is tested or in what order.

One library serves several interfaces:
- a C++ header library and a CLI;
- a stable C ABI;
- Python and R packages;
- an inference back end for PyDESeq2.

## Benchmarks

Measured on 2026-10-10 with version 1.3.0 on one workstation (wall-clock
seconds; one / eight threads, each thread on its own efficiency core of an
Intel Core i9-13900KF; [docs/BENCHMARK.md](docs/BENCHMARK.md#host)). Jaccard is
the overlap of discoveries with the reference tool at a false-discovery rate of
0.1. RapidGLM times are for its R package; the CLI and Python package take
similar times. The SCEPTRE rows accept the skew-normal fit as SCEPTRE does
(`fit_check: "ks"`); RapidGLM's default is a stricter check with confirmation
draws, which takes 27–40% longer on these panels
([docs/CLI.md](docs/CLI.md#perturbation-workflow)).

| Analysis | Reference (1 / 8) | RapidGLM (1 / 8) | Speed-up (1 thread) | Jaccard |
|---|---|---|---:|---:|
| DESeq2 1.48.2, airway (8 samples) | 6.09 / 2.75 | 1.02 / 0.28 | 6.0× | 0.932 |
| DESeq2, GTEx thyroid (128 samples) | 119 / 31.3 | 9.56 / 1.74 | 12.4× | 0.997 |
| DESeq2, GTEx thyroid (647 samples) | 481 / 83.8 | 35.1 / 6.55 | 13.7× | 1.000 |
| SCEPTRE 0.99.0, Gasperini screen (50,000 cells) | 1,517 / 251 | 336 / 59.4 | 4.5× | 0.79 |
| SCEPTRE, Gasperini screen (207,324 cells) | – / 1,473 | – / 335 | 4.4× (8 threads) | 0.89 |

Both perturbation tools are Monte Carlo tests, so their overlap turns on the
draws: over six seeds the Jaccard index on the full screen is 0.89 to
0.96, and 0.90 to 0.96 between two RapidGLM seeds
([validation/PERTURBATION_SEEDS.json](validation/PERTURBATION_SEEDS.json)).
Bulk DE computes Cook's distances with DESeq2's robust dispersion as
implemented in PyDESeq2 (MIT; Muzellec et al. 2023), the default from 1.2.0.
RapidGLM's own estimator (`legacy_cooks: false`) flags more genes in designs
with many small groups and gives Jaccard 0.58 and 0.90 on the two thyroid sets
([docs/BULK_DE.md](docs/BULK_DE.md#legacy-cooks-distances-2026-10-01)). The
timings are provisional: the workstation's performance cores were not all
available, and the runs will be repeated when it is repaired.

**Engine:**
- *Accuracy.* All 39 identifiable problems from a binary128-referenced suite
  are fitted, with coefficient errors of at most 6.6e-7 standard errors.
- *Single fits.* Median speed-up over R `glm.fit` 5.6×, statsmodels 8.9× and
  fastglm 3.5× (one thread, n ≥ 1e5;
  [docs/GLM_COMPARISON.md](docs/GLM_COMPARISON.md)).
- *Solvers.* Against Ceres Solver on identical problems, the safeguarded Newton
  solvers are 2.1–2.7× faster than Levenberg–Marquardt (fixed dispersion) and
  1.7–5.7× faster than BFGS (estimated dispersion), at equal or better accuracy
  ([docs/SOLVERS.md](docs/SOLVERS.md)).
- *Validation datasets.* Bulk DE agrees with DESeq2 at Jaccard 0.93–1.00 on
  five datasets ([docs/BULK_DE.md](docs/BULK_DE.md)). Perturbation analysis
  agrees with SCEPTRE at the Monte Carlo floor
  ([docs/SCEPTRE_STAGING.md](docs/SCEPTRE_STAGING.md)).

Datasets, tools and the harness are described in
[docs/BENCHMARK.md](docs/BENCHMARK.md). Every number has a receipt under
[validation/](validation).

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
- *Drop-in GLM interfaces:* `glm(..., method = rapidglm::glm_fit)` and
  `rapidglm::glm_nb()` in R, and `rapidglm.statsmodels.GLM` in Python.
  `scripts/check_bindings.sh` builds and tests both.
- *C interface* (`librapidglm_c`): a stable C ABI over GLM fitting, `bulk_de`,
  `sceptre` and `score_scan`. Python, R and Rust build on it; other languages
  can bind to it the same way ([docs/C_API.md](docs/C_API.md#language-bindings)).
- *Rust* ([bindings/rust](bindings/rust)): bindings over the C interface and a
  port of PyDESeq2's workflow whose per-gene fits run in RapidGLM.

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
