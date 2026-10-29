# RapidGLM against the standard GLM fitters

This page records what the standard R and Python GLM fitters are built on, how
RapidGLM compares on accuracy, stability and single-thread speed, and the
drop-in interfaces that let existing R and statsmodels code use the engine.
Measurements, commands and per-problem results are in
[GLM_COMPARISON.json](../validation/GLM_COMPARISON.json). Comparators ran as
black boxes; their documentation, not their source, describes them below.
The accuracy, single-thread speed and many-response tables are from version
1.4.0, completed 2026-10-22, on the library code of `v1.4.0-dev.4`.
Timings are final, not provisional, on the workstation's 16 identical
efficiency cores, one thread per core. One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional. Each command has a retained quiet start.

## What the standard fitters are built on

| Fitter | Algorithm (documented) | Convergence default | Rank deficiency | Inference |
|---|---|---|---|---|
| R `stats::glm` / `glm.fit` | IWLS; each weighted least-squares step by QR, LINPACK `dqrdc2` with limited column pivoting ([glm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html), [qr](https://stat.ethz.ch/R-manual/R-devel/library/base/html/qr.html)) | \|dev − dev_old\|/(\|dev\| + 0.1) < 1e-8, 25 iterations ([glm.control](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.control.html)) | Near-dependent columns aliased, coefficient `NA` (measured threshold: relative residual 1e-11 at the default epsilon) | `summary.glm` from the final IWLS weights |
| R `MASS::glm.nb` | Alternates a `glm` fit at fixed θ with θ score/information iterations; Poisson start and moment θ ([glm.nb](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/glm.nb.html)) | Same `glm.control` | As `glm` | As `glm`; `SE.theta` from observed information |
| Python `statsmodels` GLM | IRLS; WLS step by `numpy.linalg.lstsq` (SVD) by default ([GLM.fit](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit.html)) | Deviance, tol 1e-8, 100 iterations | Small singular values truncated: minimum-norm estimates | Full results classes |
| Python `statsmodels` NegativeBinomial | Discrete-model maximum likelihood with θ; BFGS, 35 iterations by default | Optimizer defaults | Hessian inversion warnings | Full results classes |
| Python scikit-learn GLMs | L-BFGS or Newton–Cholesky; `PoissonRegressor` has an L2 penalty (alpha 1) by default ([PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)) | tol 1e-4 | Solver warnings | None |
| Python glum | IRLS with line search (`irls-ls`) for unpenalized fits ([glum](https://glum.readthedocs.io/en/latest/glm.html)) | gradient_tol 1e-4 | Singular-matrix errors | Covariance method |
| R fastglm | IWLS with Eigen decompositions, column-pivoted QR by default, step halving (Marschner 2011) ([CRAN manual](https://cran.r-project.org/web/packages/fastglm/fastglm.pdf)) | tol 1e-8, 100 iterations | Pivoted solution | Standard errors |
| RapidGLM | Fisher scoring with likelihood step halving; Cholesky of X'WX while its reciprocal condition is ≥ 1e-8, Householder QR of √W·X otherwise ([METHODS](../METHODS.md)) | Deviance change < 1e-9 and coefficient change < max(1e-9, 100·eps·cond) | Error by default; R-style aliasing with `singular_ok` | Information at the final estimates |

RapidGLM also estimates the NB θ jointly by profile maximum likelihood and
flags coefficients without a finite maximum (for example, a group whose counts
are all zero) instead of letting them drift.

## Accuracy and stability

Forty-four problems (n = 2,000; Gaussian, Poisson, binomial and NB with
estimated θ) cover baseline designs, columns scaled by 1e-6 to 1e6,
uncentered covariates (+1e4), near-collinearity (relative residual 1e-6,
1e-9, 1e-12), exact aliasing, prior weights including zeros, large offsets,
sparse and large counts, rare events, extreme dispersion (θ = 0.05 and
near-Poisson), a group of all-zero counts and complete separation. Each
identifiable problem has a reference maximum-likelihood fit computed by Newton's
method in IEEE binary128. Errors are measured in reference standard errors.

| Tool | Problems | Failures | Median error (SE) | Max error (SE) | Median SE error | Max SE error |
|---|---:|---:|---:|---:|---:|---:|
| RapidGLM engine (R-compatible aliasing) | 39 | 0 | 3.1e-15 | 6.6e-07 | 8.4e-16 | 2.5e-08 |
| R glm(method = rapidglm::glm_fit) | 26 | 0 | 3.6e-14 | 1.5e-06 | 2.8e-15 | 1.1e-07 |
| R rapidglm::glm_nb | 13 | 0 | 1.9e-14 | 3.0e-07 | 4.2e-15 | 7.9e-08 |
| Python rapidglm.statsmodels.GLM | 26 | 3 | 3.3e-15 | 6.6e-07 | 1.0e-15 | 1.1e-01 |
| R stats::glm.fit | 26 | 0 | 1.8e-10 | 3.6e-06 | 2.4e-07 | 1.9e-05 |
| R MASS::glm.nb | 13 | 0 | 3.1e-09 | 1.5e-04 | 6.5e-10 | 1.4e-04 |
| R fastglm | 26 | 3 | 2.2e-10 | 1.1e-06 | 2.9e-07 | 1.1e-01 |
| R speedglm | 26 | 12 | 5.3e-12 | 3.4e-08 | 2.0e-07 | 1.9e-05 |
| Python statsmodels GLM | 26 | 4 | 9.3e-12 | 7.4e-04 | 1.1e-08 | 1.1e-01 |
| Python statsmodels NegativeBinomial | 13 | 8 | 1.7e-04 | 3.3e-04 | 1.6e-02 | 1.9e-02 |
| Python glum (defaults) | 26 | 8 | 1.6e-06 | 3.4e-04 | 4.1e-02 | 6.2e-01 |
| Python glum (tight tolerances) | 26 | 6 | 2.8e-12 | 2.9e-04 | 4.9e-02 | 6.2e-01 |
| Python scikit-learn (defaults) | 26 | 13 | 4.6e-07 | 3.5e-04 | – | – |
| Python scikit-learn (tol 1e-10) | 26 | 11 | 3.8e-12 | 2.6e-07 | – | – |

Problems at relative collinearity 1e-12 are excluded above because R, and
RapidGLM with R-compatible options, alias that column by design; both then
report the same fit. Findings:

- Before this work RapidGLM refused exactly aliased designs and declared
  designs with condition numbers above about 1e6 rank deficient. With the QR
  fallback and R-style aliasing it fits every identifiable problem, and makes
  the same aliasing decisions as R on the aliased and 1e-12 problems.
- RapidGLM's standard errors have median relative error 8.4e-16 (maximum 2.5e-08); R's `glm.fit` maximum is 1.9e-05 and `glm.nb`'s is 1.4e-04. Near condition number 1e9, double precision limits every fitter.
- The Python drop-in's three "failures" are the aliased designs, which it hands
  to statsmodels' own fitting and so returns statsmodels' minimum-norm
  estimates by design.
- statsmodels returns minimum-norm estimates for aliased designs (the same fit,
  a different parametrization); its discrete NegativeBinomial model fails 8 of
  13 problems (convergence, scaling, weights). scikit-learn and glum stop at
  about 1e-4 gradient tolerance by default and fail scaled, uncentered or
  collinear designs even with tight tolerances. speedglm fails scaled and
  uncentered designs with "computationally singular" errors.
- On complete separation RapidGLM stops with status `separation` (largest
  coefficient 31); R warns that the algorithm did not converge and returns
  coefficients near 7,400.

## Single-thread speed

Each fit ran on in-memory data with one BLAS/OpenMP thread; the minimum of up
to seven repeats is reported, in seconds. Tools use their defaults; "tight"
variants use tolerances comparable to RapidGLM's. All fits ran one at a time on
one efficiency core of an Intel Core i9-13900KF
([BENCHMARK.md](BENCHMARK.md#host)); no performance-core times enter these tables.

Medians over the six problems with n = 1e5 or 1e6 (p = 5, 20, 50), as speed
ratios relative to the RapidGLM engine:

| Comparator | Poisson | Binomial | Gaussian |
|---|---:|---:|---:|
| R `glm.fit` | 5.5× slower | 4.8× slower | 5.4× slower |
| R `glm()` | 6.4× slower | 5.9× slower | 7.1× slower |
| statsmodels GLM | 5.9× slower | 6.0× slower | 11.1× slower |
| fastglm | 3.2× slower | 3.1× slower | 3.2× slower |
| speedglm | 3.2× slower | 2.9× slower | 2.2× slower |
| glum, defaults | 1.5× slower | 1.4× slower | 2.2× slower |
| glum, tight | 4.4× slower | 1.7× slower | 2.2× slower |
| scikit-learn, defaults (tol 1e-4) | 1.0× slower | 1.4× faster | 1.6× slower |
| scikit-learn, tol 1e-10 | 1.3× slower | 1.0× slower | 1.6× slower |

Absolute times in seconds at n = 1,000,000:

| Problem | RapidGLM | R glm.fit | R glm | R glm + rapidglm | fastglm | speedglm | statsmodels | rapidglm.statsmodels | glum (tight) | sklearn (tight) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Poisson, p = 5 | 0.28 | 1.25 | 1.72 | 1.23 | 0.73 | 0.82 | 1.36 | 0.50 | 40.51 | 0.42 |
| Poisson, p = 20 | 0.78 | 4.38 | 5.01 | 2.96 | 3.44 | 2.70 | 5.98 | 1.61 | 6.57 | 1.00 |
| Poisson, p = 50 | 2.32 | 16.41 | 17.72 | 9.29 | 15.47 | 10.02 | 26.96 | 5.53 | 16.10 | 2.96 |
| Binomial, p = 5 | 0.31 | 0.82 | 1.19 | 1.21 | 0.53 | 0.54 | 1.20 | 0.56 | 0.49 | 0.35 |
| Binomial, p = 20 | 0.67 | 3.54 | 4.05 | 2.69 | 2.86 | 2.11 | 6.03 | 1.53 | 1.74 | 0.91 |
| Binomial, p = 50 | 1.81 | 13.02 | 14.21 | 8.35 | 12.52 | 8.17 | 23.50 | 5.13 | 3.23 | 1.90 |
| Gaussian, p = 5 | 0.09 | 0.40 | 0.47 | 0.47 | 0.23 | 0.21 | 0.86 | 0.30 | 0.19 | 0.10 |
| Gaussian, p = 20 | 0.29 | 1.64 | 1.97 | 1.96 | 1.39 | 0.65 | 4.34 | 1.09 | 0.65 | 0.67 |
| Gaussian, p = 50 | 0.92 | 6.36 | 7.25 | 7.23 | 6.28 | 2.21 | 19.90 | 4.24 | 2.05 | 3.30 |

R here links the reference BLAS, which slows R's QR steps; an optimized BLAS
would narrow the R ratios. glum with tight tolerances needs many more
iterations on the Poisson problems (40.51 s at p = 5).

Negative binomial with estimated θ:

| Problem | RapidGLM | rapidglm::glm_nb | MASS::glm.nb | statsmodels NB |
|---|---:|---:|---:|---:|
| n = 10,000, p = 5 | 0.006 | 0.019 | 0.121 | 0.025 |
| n = 10,000, p = 20 | 0.011 | 0.037 | 0.218 | 0.053 |
| n = 10,000, p = 50 | 0.040 | 0.125 | 0.303 | 0.130 |
| n = 100,000, p = 5 | 0.057 | 0.226 | 1.215 | 0.256 |
| n = 100,000, p = 20 | 0.118 | 0.409 | 1.568 | 0.520 |
| n = 100,000, p = 50 | 0.419 | 1.352 | 3.681 | 1.779 |

statsmodels' NB model is fast at its defaults but failed 8 of the 13 accuracy
problems.

### Many responses with one design

The genomics case: 1,000 count responses on 5,000 observations with the same
five-column design (library-size covariate plus three covariates), totals in
seconds. RapidGLM's `fit_batch` parallelizes across responses; the other rows
loop over single fits in one thread. statsmodels' NB time is extrapolated from
200 responses.

| Tool | Poisson | NB (θ estimated) |
|---|---:|---:|
| RapidGLM `fit_batch`, 1 thread | 1.09 | 3.01 |
| RapidGLM `fit_batch`, 4 threads | 0.29 | 0.79 |
| RapidGLM `fit_batch`, 16 threads | 0.08 | 0.21 |
| Python `rapidglm.fit` loop | 1.16 | 3.17 |
| R `rapidglm::glm_fit` / `glm_nb` loop | 2.87 | 10.08 |
| R `glm.fit` / `MASS::glm.nb` loop | 5.55 | 142.57 |
| glmGamPoi `glm_gp`, whole batch | – | 5.83 |
| Python statsmodels GLM / NegativeBinomial loop | 4.94 | 23.30 |

The 1,000 fitted coefficient sets differ from R by at most
1.84e-06 relative for
Poisson and 9.17e-04 for NB;
statsmodels NB differs from R by up to
9.01e-03. Threads occupy
separate efficiency cores ([ENGINE_BATCH.json](../validation/ENGINE_BATCH.json)).

### Joint Newton and grouped designs

NB maximum-likelihood fits take joint Newton steps in coefficients and log θ
using observed information. Cox-Reid and ridge fits still alternate. Designs
with as many distinct rows as columns also use per-group sums; grouped
Poisson fits start at the closed-form MLE. The development studies remain in
[JOINT_NEWTON.json](../validation/JOINT_NEWTON.json) and
[GROUPED_SOLVER.json](../validation/GROUPED_SOLVER.json). Their historical
timings are not included in the final 1.4.0 tables above.

## Drop-in interfaces

**R** (`bindings/r`, package `rapidglm`): `glm(..., method = rapidglm::glm_fit)`
uses the engine inside any existing `glm()` call; `rapidglm::glm()` is the same
with the method as default, and `rapidglm::glm_nb()` returns a `negbin` object
like `MASS::glm.nb()`. The package uses R's own LAPACK. Gaussian identity fits
stay with `glm.fit`, whose single QR is also the `qr` component `glm()`
stores. End to end, `glm()` with the RapidGLM method has speed ratios
of 0.98–1.91× relative to R `glm()` on Poisson and binomial
problems with n ≥ 100,000 (reference seconds / RapidGLM seconds). Model-frame
construction and the QR stored in the result remain R-side costs.
`glm_nb` is 2.42–6.37× faster than `MASS::glm.nb` on the NB
speed problems, and
14.1×
in the many-response loop. Results are ordinary
`glm` objects, and `summary`, `anova`, `predict(se.fit = TRUE)`, `vcov`,
profile `confint`, `drop1`, all residual types, `hatvalues`, `cooks.distance`,
`rstandard` and `anova.negbin` agree with `stats::glm`/`MASS::glm.nb` to
rounding (coefficients ~1e-13, influence ~1e-9;
[tests](../bindings/r/tests/parity.R)). Other families or links fall back to
`stats::glm.fit`.

**Python** (`bindings/python`): `rapidglm.statsmodels.GLM` subclasses
`statsmodels.api.GLM` and returns statsmodels `GLMResults`; offsets, exposure,
frequency and variance weights, binomial trial counts, fixed-α NB and robust
covariances agree with statsmodels to rounding
([tests](../bindings/python/tests/test_statsmodels.py)). Unsupported
families, rank-deficient designs and non-IRLS options fall back to
statsmodels. `rapidglm.fit()` exposes the engine directly, including NB with
estimated θ. End to end, including model setup, median speedups of the
drop-in over `statsmodels.api.GLM` are 3.40× (Poisson),
3.09× (binomial) and 3.82× (Gaussian), at
n ≥ 100,000.

## Limitations

- Native families are Gaussian (identity), Poisson (log), binomial (logit) and
  NB (log). Other families and links use the host package's fitter.
- Packaging is not finished. `R CMD build` from the repository gives a
  self-contained source package that passes `R CMD check` (status OK, tests
  included), but it relies on system Boost headers; CRAN would need `BH`. The
  Python module links LAPACKE and BLAS and has no wheel build.
- In R the end-to-end gain is limited by `glm()`'s own bookkeeping; a
  many-response R interface to `fit_batch` would expose the engine's full
  speed.
- The comparisons are single-thread, on an efficiency core of a workstation
  with background services ([BENCHMARK.md](BENCHMARK.md) describes the host).
