# RapidGLM against the standard GLM fitters

This page records what the standard R and Python GLM fitters are built on, how
RapidGLM compares on accuracy, stability and single-thread speed, and the
drop-in interfaces that let existing R and statsmodels code use the engine.
Measurements, commands and per-problem results are in
[GLM_COMPARISON.json](../validation/GLM_COMPARISON.json). Comparators ran as
black boxes; their documentation, not their source, describes them below.
The accuracy, single-thread speed and many-response tables are from version
1.2.0 (2026-10-06); the page was first written on 2026-09-29.

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

Forty-five problems (n = 2,000; Gaussian, Poisson, binomial and NB with
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
- RapidGLM's standard errors have median relative error 8e-16 (at most 2e-8,
  at condition number 1e9); R's use weights from the start of the last iteration
  and differ by up to 2e-5, `glm.nb`'s by up to 1.4e-4. At condition numbers
  near 1e9 every fitter is limited by double precision (errors near 1e-6 SE).
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
([BENCHMARK.md](BENCHMARK.md#host)); a performance core is about 1.9× faster
for RapidGLM and R alike.

Medians over the six problems with n = 1e5 or 1e6 (p = 5, 20, 50), as speed
ratios relative to the RapidGLM engine:

| Comparator | Poisson | Binomial | Gaussian |
|---|---:|---:|---:|
| R `glm.fit` | 5.5× slower | 4.9× slower | 5.6× slower |
| R `glm()` | 6.5× slower | 5.8× slower | 7.1× slower |
| statsmodels GLM | 6.4× slower | 6.5× slower | 12.1× slower |
| fastglm | 3.4× slower | 3.3× slower | 3.7× slower |
| speedglm | 3.3× slower | 2.9× slower | 2.3× slower |
| glum, defaults | 1.6× slower | 1.4× slower | 2.3× slower |
| glum, tight | 4.6× slower | 1.7× slower | 2.3× slower |
| scikit-learn, defaults (tol 1e-4) | 1.1× slower | 1.4× faster | 1.7× slower |
| scikit-learn, tol 1e-10 | 1.3× slower | 1.1× slower | 1.7× slower |

Absolute times in seconds at n = 1,000,000:

| Problem | RapidGLM | R glm.fit | R glm | R glm + rapidglm | fastglm | speedglm | statsmodels | rapidglm.statsmodels | glum (tight) | sklearn (tight) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Poisson, p = 5 | 0.29 | 1.25 | 1.75 | 1.25 | 0.73 | 0.84 | 1.39 | 0.51 | 37.56 | 0.42 |
| Poisson, p = 20 | 0.79 | 4.51 | 5.17 | 3.12 | 3.56 | 2.81 | 6.15 | 1.62 | 6.55 | 1.00 |
| Poisson, p = 50 | 2.23 | 16.62 | 17.80 | 9.37 | 15.79 | 10.17 | 28.74 | 5.67 | 16.03 | 2.97 |
| Binomial, p = 5 | 0.31 | 0.85 | 1.21 | 1.26 | 0.54 | 0.55 | 1.24 | 0.57 | 0.50 | 0.35 |
| Binomial, p = 20 | 0.68 | 3.51 | 4.10 | 2.87 | 2.94 | 2.17 | 6.21 | 1.55 | 1.75 | 0.92 |
| Binomial, p = 50 | 1.74 | 13.17 | 14.42 | 8.55 | 12.76 | 8.31 | 25.52 | 5.19 | 3.22 | 1.91 |
| Gaussian, p = 5 | 0.09 | 0.41 | 0.48 | 0.48 | 0.24 | 0.21 | 0.90 | 0.31 | 0.20 | 0.10 |
| Gaussian, p = 20 | 0.29 | 1.71 | 2.03 | 2.02 | 1.42 | 0.67 | 4.41 | 1.11 | 0.66 | 0.67 |
| Gaussian, p = 50 | 0.87 | 6.83 | 7.47 | 7.32 | 6.59 | 2.27 | 20.68 | 4.34 | 2.05 | 3.29 |

R here links the reference BLAS, which slows R's QR steps; an optimized BLAS
would narrow the R ratios. glum with tight tolerances needs many more
iterations on the Poisson problems (37.6 s at p = 5).

Negative binomial with estimated θ:

| Problem | RapidGLM | rapidglm::glm_nb | MASS::glm.nb | statsmodels NB |
|---|---:|---:|---:|---:|
| n = 10,000, p = 5 | 0.006 | 0.020 | 0.122 | 0.025 |
| n = 10,000, p = 20 | 0.011 | 0.039 | 0.221 | 0.053 |
| n = 10,000, p = 50 | 0.037 | 0.127 | 0.308 | 0.135 |
| n = 100,000, p = 5 | 0.058 | 0.237 | 1.221 | 0.262 |
| n = 100,000, p = 20 | 0.118 | 0.433 | 1.574 | 0.531 |
| n = 100,000, p = 50 | 0.404 | 1.434 | 3.778 | 1.803 |

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
| RapidGLM `fit_batch`, 1 thread | 1.07 | 3.02 |
| RapidGLM `fit_batch`, 4 threads | 0.29 | 0.79 |
| RapidGLM `fit_batch`, 16 threads | 0.08 | 0.21 |
| Python `rapidglm.fit` loop | 1.15 | 3.15 |
| R `rapidglm::glm_fit` / `glm_nb` loop | 3.09 | 10.51 |
| R `glm.fit` / `MASS::glm.nb` loop | 5.61 | 143.37 |
| glmGamPoi `glm_gp`, whole batch | – | 5.86 |
| Python statsmodels GLM / NegativeBinomial loop | 5.00 | 23.16 |

The 1,000 fitted coefficient sets agree with R's to 1.8e-6 relative for Poisson
(R's tolerance) and 9.2e-4 for NB, where `glm.nb` stops earlier; statsmodels' NB
differs from R by up to 9e-3. Threads are on separate efficiency cores
([ENGINE_BATCH.json](../validation/ENGINE_BATCH.json)).

### Joint Newton NB fits, grouped designs and glmGamPoi (2026-09-30)

This section records the change as measured on 2026-09-30, on performance
cores of a shared host; its times are not comparable with the tables above.

NB maximum-likelihood fits now take joint Newton steps in the coefficients and
log θ, using the observed information (see [METHODS.md](../METHODS.md)).
Previously they alternated coefficient fits at fixed θ with θ updates. Cox-Reid
and ridge fits still alternate.
- **Iterations:** the six NB speed problems above need 3–4 iterations instead
  of 8–13, and are 1.6–2.1× faster at one thread.
- **Accuracy:** the log likelihood matches the alternating fit's to printed
  precision. On the binary128-referenced accuracy suite the engine still fits
  all 39 identifiable problems. Its median coefficient error drops from 1e-11
  to 3e-15 reference standard errors.

Designs whose rows take exactly as many distinct values as there are columns
(intercept-only, one-way, full factorial) also compute each step from
per-group sums. Their Poisson fits start at the closed-form MLE, which cut
Poisson batches from 0.67 s to 0.42 s (four groups) and 0.54 s to 0.35 s
(intercept only) ([receipt](../validation/GROUPED_SOLVER.json)).

The table below uses 1,000 NB responses on 5,000 observations at one thread.
The grouped designs have a depth offset. Times are in seconds, taken as the
minimum of five alternating runs on a shared host
([receipt](../validation/JOINT_NEWTON.json)). glmGamPoi 1.25.1 was run as a
black box with ML dispersions and no shrinkage, taking the minimum of three
runs.

| Design | Alternating (0991369) | Joint Newton | 16 threads | glmGamPoi |
|---|---:|---:|---:|---:|
| Five covariates | 3.75 | 1.83 | 0.21 | 3.11 |
| Four groups | 2.90 | 1.39 | 0.16 | 1.67 |
| Intercept only | 2.39 | 1.23 | 0.14 | 1.65 |

For every interior fit the joint and alternating log likelihoods agree within
4e-10, and the same fits reach the θ bound in both. Panel tables are
byte-identical at 1 and 16 threads. glmGamPoi's times vary by about 20% between
sessions (1.7–2.1 s on four groups). It does one pass:
coefficients at a moment dispersion, the dispersion at those means, then the
coefficients again. That pass falls short of the maximum log likelihood by a
median of 1e-5, and by at most 7.5e-3, on the five-covariate batch. RapidGLM
iterates to the joint MLE and is still faster at one thread; glmGamPoi has no
threading.

## Drop-in interfaces

**R** (`bindings/r`, package `rapidglm`): `glm(..., method = rapidglm::glm_fit)`
uses the engine inside any existing `glm()` call; `rapidglm::glm()` is the same
with the method as default, and `rapidglm::glm_nb()` returns a `negbin` object
like `MASS::glm.nb()`. The package uses R's own LAPACK. Gaussian identity fits
stay with `glm.fit`, whose single QR is also the `qr` component `glm()`
stores. End to end, `glm()` with the RapidGLM method is 1.2–1.9× faster for
Poisson and binomial at n ≥ 1e5 (most at p = 50; equal for binomial at p = 5
with n = 1e6): model-frame construction and the QR stored in the result are
R-side costs a drop-in must still pay. `glm_nb` is 2.4–6.1× faster than
`MASS::glm.nb`, and 14× in the many-response loop. Results are ordinary
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
estimated θ. End to end, including statsmodels' model setup, the drop-in is
3.4× (Poisson), 3.1× (binomial) and 3.8× (Gaussian) faster than
`statsmodels.api.GLM` at n ≥ 1e5.

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
  with background services; its performance cores were not all available.
  They are to be repeated when that processor is repaired.
