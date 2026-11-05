# RapidGLM against the standard GLM fitters

Completed RapidGLM 1.6.0 measurements on `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`); the release library is unchanged from the measured dev.2 source.

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5.

[GLM_COMPARISON.json](../validation/GLM_COMPARISON.json) retains every problem and tool result. Reference tools were used through their public interfaces.

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
| --- | --- | --- | --- | --- | --- | --- |
| RapidGLM engine (R-compatible aliasing) | 39 | 0 | 3.3e-15 | 1.7e-06 | 8.4e-16 | 2.1e-08 |
| R glm(method = rapidglm::glm_fit) | 26 | 0 | 3.6e-14 | 2.9e-06 | 2.8e-15 | 5.7e-08 |
| R rapidglm::glm_nb | 13 | 0 | 1.9e-14 | 1.3e-06 | 4.2e-15 | 1.6e-06 |
| Python rapidglm.statsmodels.GLM | 26 | 3 | 3.6e-15 | 1.7e-06 | 1e-15 | 0.11 |
| R glm.fit | 26 | 0 | 1.8e-10 | 3.6e-06 | 2.4e-07 | 1.9e-05 |
| R MASS::glm.nb | 13 | 0 | 3.1e-09 | 0.00015 | 6.5e-10 | 0.00014 |
| R fastglm | 26 | 3 | 2.2e-10 | 1.1e-06 | 2.9e-07 | 0.11 |
| R speedglm | 26 | 12 | 5.3e-12 | 3.4e-08 | 2e-07 | 1.9e-05 |
| Python statsmodels GLM | 26 | 4 | 9.3e-12 | 0.00074 | 1.1e-08 | 0.11 |
| Python statsmodels NegativeBinomial | 13 | 8 | 0.00017 | 0.00033 | 0.016 | 0.019 |
| Python glum | 26 | 8 | 1.6e-06 | 0.00034 | 0.041 | 0.62 |
| Python glum (tight) | 26 | 6 | 2.8e-12 | 0.00029 | 0.049 | 0.62 |
| Python scikit-learn | 26 | 13 | 4.6e-07 | 0.00035 | — | — |
| Python scikit-learn (tight) | 26 | 11 | 3.8e-12 | 2.6e-07 | — | — |

The historical and current accuracy runs have identical convergence/status and established pass/fail outcomes. Regenerated Gaussian responses differ by at most 4.44e-16 because of BLAS dispatch during input generation; controlled dispatch reproduces both input sets exactly. Each current tool reads the same current bytes and is scored against its corresponding quad-precision reference. The input trace records the comparison.

Errors use reference standard-error units. Relative-collinearity 1e-12 problems are excluded from the identifiable set because R-compatible aliasing drops that column by design. Python delegates rank-deficient designs to statsmodels, retaining its minimum-norm parameterization.

## Single-thread speed

Minimum seconds over up to seven repeats, in-memory inputs, one BLAS/OpenMP thread on Zen 3 core 6. Tight variants use tolerances comparable to RapidGLM's. The per-family medians below cover n ≥ 100,000 and divide reference time by engine time.

| Reference | Poisson | Binomial | Gaussian |
| --- | --- | --- | --- |
| glm.fit | 6.61× | 5.59× | 6.87× |
| glm | 7.68× | 6.64× | 8.6× |
| statsmodels | 5.59× | 5.6× | 11.1× |
| fastglm | 2.72× | 2.54× | 2.82× |
| speedglm | 3.68× | 3.18× | 2.63× |
| glum | 1.44× | 1.17× | 1.94× |
| glum-tight | 3.85× | 1.4× | 1.94× |
| sklearn | 0.793× | 0.569× | 1.41× |
| sklearn-tight | 1.1× | 0.863× | 1.43× |

Absolute seconds at n = 1,000,000:

| Problem | rapidglm | glm.fit | glm | r-rapidglm | fastglm | speedglm | statsmodels | py-rapidglm | glum-tight | sklearn-tight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binomial, n=1,000,000, p=5 | 0.2255 | 0.625 | 0.91 | 0.968 | 0.323 | 0.405 | 0.7665 | 0.4377 | 0.3039 | 0.2143 |
| binomial, n=1,000,000, p=20 | 0.4269 | 2.596 | 2.901 | 2.078 | 1.322 | 1.499 | 18.45 | 1.031 | 1.036 | 0.467 |
| binomial, n=1,000,000, p=50 | 1.193 | 9.128 | 9.88 | 6.062 | 5.297 | 5.324 | 31.84 | 3.074 | 1.747 | 0.9292 |
| gaussian, n=1,000,000, p=5 | 0.05758 | 0.29 | 0.321 | 0.322 | 0.111 | 0.181 | 0.5929 | 0.2408 | 0.1079 | 0.0602 |
| gaussian, n=1,000,000, p=20 | 0.1669 | 1.182 | 1.501 | 1.505 | 0.648 | 0.504 | 2.505 | 0.7593 | 0.3846 | 0.3544 |
| gaussian, n=1,000,000, p=50 | 0.6039 | 4.471 | 5.188 | 5.221 | 2.673 | 1.534 | 9.636 | 2.455 | 1.118 | 1.424 |
| poisson, n=1,000,000, p=5 | 0.1827 | 0.902 | 1.293 | 0.926 | 0.395 | 0.591 | 0.904 | 0.3895 | 24.62 | 0.2356 |
| poisson, n=1,000,000, p=20 | 0.4622 | 3.197 | 3.624 | 2.372 | 1.586 | 1.858 | 3.539 | 1.068 | 3.853 | 0.4787 |
| poisson, n=1,000,000, p=50 | 1.514 | 11.35 | 12.35 | 6.67 | 6.391 | 6.581 | 12.99 | 3.339 | 9.066 | 1.399 |

NB with estimated dispersion:

| Problem | rapidglm | r-rapidglm.nb | glm.nb | statsmodels-nb |
| --- | --- | --- | --- | --- |
| nb, n=10,000, p=5 | 0.004305 | 0.015 | 0.091 | 0.01978 |
| nb, n=10,000, p=20 | 0.008422 | 0.031 | 0.17 | 0.04518 |
| nb, n=10,000, p=50 | 0.03036 | 0.095 | 0.254 | 0.1044 |
| nb, n=100,000, p=5 | 0.04324 | 0.189 | 0.915 | 0.2195 |
| nb, n=100,000, p=20 | 0.08484 | 0.329 | 1.207 | 0.3483 |
| nb, n=100,000, p=50 | 0.2853 | 1.064 | 3.044 | 0.9794 |

### Many responses with one design

One thousand responses on 5,000 observations with a shared five-column design. RapidGLM parallelizes across responses; loop interfaces use one thread. statsmodels NB is extrapolated from 200 responses. [ENGINE_BATCH.json](../validation/ENGINE_BATCH.json) records the complete measurements.

| Tool | Family | Threads | Seconds | Max relative coefficient difference vs R |
| --- | --- | --- | --- | --- |
| rapidglm fit_batch | poisson | 1 | 0.7215 | 1.84e-06 |
| rapidglm fit_batch | poisson | 4 | 0.1833 | 1.84e-06 |
| rapidglm fit_batch | poisson | 16 | 0.05844 | 1.84e-06 |
| R glm.fit | poisson | 1 | 4.168 | 0 |
| R rapidglm::glm_fit | poisson | 1 | 2.074 | 1.84e-06 |
| Python statsmodels | poisson | 1 | 3.884 | 1.84e-06 |
| Python rapidglm.fit | poisson | 1 | 0.7735 | 1.84e-06 |
| rapidglm fit_batch | nb | 1 | 2.221 | 0.000917 |
| rapidglm fit_batch | nb | 4 | 0.5575 | 0.000917 |
| rapidglm fit_batch | nb | 16 | 0.1509 | 0.000917 |
| R glm.nb | nb | 1 | 103.3 | 0 |
| R rapidglm::glm_nb | nb | 1 | 8.156 | 0.000917 |
| R glmGamPoi::glm_gp | nb | 1 | 4.441 | 0.0297 |
| Python statsmodels | nb | 1 | 18.89 | 0.00901 |
| Python rapidglm.fit | nb | 1 | 2.338 | 0.000917 |

### Joint Newton and grouped designs

NB maximum-likelihood fits take joint Newton steps in coefficients and log θ
using observed information. Cox-Reid and ridge fits still alternate. Designs
with as many distinct rows as columns also use per-group sums; grouped
Poisson fits start at the closed-form MLE. The development studies remain in
[JOINT_NEWTON.json](../validation/JOINT_NEWTON.json) and
[GROUPED_SOLVER.json](../validation/GROUPED_SOLVER.json). Their historical
timings are not included in the 1.6.0 tables above.

## Drop-in interfaces

R's `glm(..., method = rapidglm::glm_fit)` uses the engine while retaining ordinary glm objects and R-side model-frame/QR work; Gaussian identity fits stay with glm.fit. `rapidglm::glm_nb()` returns a negbin object. Python's `rapidglm.statsmodels.GLM` returns statsmodels GLMResults, with unsupported families, rank-deficient designs and non-IRLS options delegated to statsmodels. `rapidglm.fit()` exposes the engine directly. The accuracy and end-to-end timing tables above include these interfaces; their setup costs differ from engine-only times. The R and Python parity tests retain their declared inference scope.

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
- The comparisons are single-thread, on one Zen 3 core of a workstation
  with background services ([BENCHMARK.md](BENCHMARK.md) describes the host).
