# RapidGLM against the standard GLM fitters — 2026-09-29

This page records what the standard R and Python GLM fitters are built on, how
RapidGLM compares on accuracy, stability and single-thread speed, and the
drop-in interfaces that let existing R and statsmodels code use the engine.
Measurements, commands and per-problem results are in
[GLM_COMPARISON.json](../validation/GLM_COMPARISON.json). Comparators ran as
black boxes; their documentation, not their source, describes them below.

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
| RapidGLM engine (R-compatible aliasing) | 39 | 0 | 9.6e-12 | 1.6e-05 | 7.6e-14 | 2.1e-07 |
| R glm(method = rapidglm::glm_fit) | 26 | 0 | 3.6e-14 | 2.2e-06 | 2.8e-15 | 1.7e-07 |
| R rapidglm::glm_nb | 13 | 0 | 2.2e-11 | 2.8e-06 | 3.5e-13 | 3.5e-08 |
| Python rapidglm.statsmodels.GLM | 26 | 3 | 3.3e-15 | 1.6e-05 | 1.0e-15 | 1.1e-01 |
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
- RapidGLM's standard errors have median relative error 8e-14 (at most 2e-7,
  at condition number 1e9); R's use weights from the start of the last iteration
  and differ by up to 2e-5, `glm.nb`'s by up to 1.4e-4. At condition numbers
  near 1e9 every fitter is limited by double precision (errors near 1e-5 SE).
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
variants use tolerances comparable to RapidGLM's. The host is shared, so
these are diagnostic timings.

Medians over the six problems with n = 1e5 or 1e6 (p = 5, 20, 50), as speed
ratios relative to the RapidGLM engine:

| Comparator | Poisson | Binomial | Gaussian |
|---|---:|---:|---:|
| R `glm.fit` | 4.8× slower | 5.5× slower | 5.2× slower |
| R `glm()` | 5.8× slower | 6.6× slower | 6.2× slower |
| statsmodels GLM | 5.3× slower | 11.8× slower | 19.2× slower |
| fastglm | 2.7× slower | 2.6× slower | 2.8× slower |
| speedglm | 2.8× slower | 2.8× slower | 2.1× slower |
| glum, defaults | 1.3× slower | 1.2× slower | 1.8× slower |
| glum, tight | 4.1× slower | 1.4× slower | 1.8× slower |
| scikit-learn, defaults (tol 1e-4) | 1.3× faster | 1.7× faster | 1.8× slower |
| scikit-learn, tol 1e-10 | equal (1.03×) | 1.16× faster | 1.7× slower |

Absolute times in seconds at n = 1,000,000:

| Problem | RapidGLM | R glm.fit | R glm | R glm + rapidglm | fastglm | speedglm | statsmodels | rapidglm.statsmodels | glum (tight) | sklearn (tight) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Poisson, p = 5 | 0.19 | 0.70 | 1.06 | 0.75 | 0.36 | 0.46 | 0.72 | 0.32 | 20.30 | 0.22 |
| Poisson, p = 20 | 0.45 | 2.47 | 2.64 | 1.78 | 1.50 | 1.35 | 5.51 | 0.89 | 3.18 | 0.49 |
| Poisson, p = 50 | 1.46 | 11.60 | 11.55 | 6.67 | 7.09 | 5.05 | 13.16 | 3.05 | 9.88 | 1.43 |
| Binomial, p = 5 | 0.19 | 0.59 | 0.68 | 0.99 | 0.35 | 0.33 | 1.79 | 0.34 | 0.29 | 0.17 |
| Binomial, p = 20 | 0.43 | 2.90 | 2.37 | 2.26 | 2.05 | 1.79 | 6.96 | 0.99 | 1.22 | 0.73 |
| Binomial, p = 50 | 1.64 | 14.05 | 14.90 | 7.58 | 10.29 | 6.79 | 27.95 | 4.55 | 3.12 | 0.96 |
| Gaussian, p = 5 | 0.06 | 0.22 | 0.25 | 0.25 | 0.14 | 0.14 | 0.76 | 0.20 | 0.10 | 0.06 |
| Gaussian, p = 20 | 0.16 | 0.94 | 1.16 | 1.13 | 0.71 | 0.54 | 6.17 | 0.65 | 0.45 | 0.43 |
| Gaussian, p = 50 | 0.73 | 3.77 | 4.23 | 4.16 | 5.29 | 1.08 | 22.11 | 2.10 | 1.10 | 1.41 |

R here links the reference BLAS, which slows R's QR steps; an optimized BLAS
would narrow the R ratios. Individual runs varied with host load (for example
the glum-tight Poisson p = 5 time); every per-run load is in the logs.

Negative binomial with estimated θ:

| Problem | RapidGLM | rapidglm::glm_nb | MASS::glm.nb | statsmodels NB |
|---|---:|---:|---:|---:|
| n = 10,000, p = 5 | 0.009 | 0.017 | 0.063 | 0.014 |
| n = 10,000, p = 20 | 0.015 | 0.032 | 0.116 | 0.028 |
| n = 10,000, p = 50 | 0.043 | 0.106 | 0.176 | 0.106 |
| n = 100,000, p = 5 | 0.076 | 0.183 | 0.643 | 0.152 |
| n = 100,000, p = 20 | 0.132 | 0.327 | 0.853 | 0.338 |
| n = 100,000, p = 50 | 0.423 | 1.079 | 2.175 | 1.197 |

statsmodels' NB model is fast at its defaults but failed 8 of the 13 accuracy
problems.

These RapidGLM NB times predate the joint Newton fit of 2026-09-30, which takes
0.004–0.259 s on the same problems (1.6–2.1× faster; see below).

### Many responses with one design

The genomics case: 1,000 count responses on 5,000 observations with the same
five-column design (library-size covariate plus three covariates), totals in
seconds. RapidGLM's `fit_batch` parallelizes across responses; the other rows
loop over single fits in one thread. statsmodels' NB time is extrapolated from
200 responses.

| Tool | Poisson | NB (θ estimated) |
|---|---:|---:|
| RapidGLM `fit_batch`, 1 thread | 0.74 | 3.81 |
| RapidGLM `fit_batch`, 4 threads | 0.19 | 0.99 |
| RapidGLM `fit_batch`, 16 threads | 0.10 | 0.39 |
| Python `rapidglm.fit` loop | 0.77 | 3.78 |
| R `rapidglm::glm_fit` / `glm_nb` loop | 1.77 | 8.06 |
| R `glm.fit` / `MASS::glm.nb` loop | 2.97 | 71.63 |
| Python statsmodels GLM / NegativeBinomial loop | 2.41 | 11.76 |

The 1,000 fitted coefficient sets agree with R's to 1.8e-6 relative for Poisson
(R's tolerance) and 9.2e-4 for NB, where `glm.nb` stops earlier; statsmodels' NB
differs from R by up to 9e-3.

### Joint Newton NB fits, grouped designs and glmGamPoi (2026-09-30)

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
stores. End to end, `glm()` with the RapidGLM method is 1.4–1.5× faster for
Poisson and binomial at n ≥ 1e5 (up to 2× at p = 50, equal at p = 5 with
n = 1e6): model-frame construction and the QR stored in the result are R-side
costs a drop-in must still pay. `glm_nb` is 2.6× faster than `MASS::glm.nb`,
and 9× in the many-response loop. Results are ordinary
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
3.1× (Poisson), 5.7× (binomial) and 6.0× (Gaussian) faster than
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
- The comparisons are single-thread and on a shared host; a clean-host run is
  still needed for publication.
