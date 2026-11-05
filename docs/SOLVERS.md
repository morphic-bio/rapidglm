# Negative binomial solvers against Ceres Solver

RapidGLM fits negative binomial GLMs with its own safeguarded Newton methods.
This page compares them with Ceres Solver 2.2 (Apache-2.0), a widely used optimized
C++ library whose default nonlinear least-squares method is
Levenberg–Marquardt. Both solve the same problems. Receipt:
[validation/SOLVERS.json](../validation/SOLVERS.json). Harness:
[benchmarks/solvers](../benchmarks/solvers), built and run in a container
(`benchmarks/solvers/Dockerfile`).

## Methods

**RapidGLM.**
- *Fixed dispersion:* Newton steps on the coefficients with the observed
  information. For the log link it is positive definite, so every step is an
  ascent direction. Step halving accepts only steps that do not lower the
  likelihood.
- *Maximum-likelihood dispersion:* joint Newton steps in the coefficients and
  log θ, with the same step halving. Where the joint likelihood is not concave
  in log θ, θ is held and the coefficients take their Newton step until they
  settle.
- *Diverging coefficients:* coefficients that diverge (an all-zero group) are
  detected. Once the information they leave behind turns singular, they are
  held and the fit finishes on the rest.

**Ceres.**
- *Fixed dispersion:*
  - Levenberg–Marquardt and dogleg trust regions on the deviance residuals
    `sign(y − μ)·sqrt(d_i)`, analytic Jacobian. The sum of their squares is the
    deviance, so the least-squares minimum is the maximum-likelihood estimate.
  - L-BFGS on the negative log likelihood, as PyDESeq2 uses.
- *Maximum-likelihood dispersion:* L-BFGS and BFGS on the negative log
  likelihood in (β, log θ), with analytic gradients.
- *Settings:* each Ceres method runs at its default tolerances and at tight
  ones (function 1e-12, gradient 1e-12, parameter 1e-10).

**Common setup.**
- Every gene gets the same counts, log size-factor offsets and dispersions
  (from `bulk_de`, not timed).
- Ceres starts from the intercept-only fit; RapidGLM uses its own start.
- RapidGLM is timed two ways. In the "RapidGLM" columns its fits share what
  they derive from the design alone (X'X, its rank check and the rows' nonzero
  entries), computed once and not timed, as `fit_batch` and `bulk_de` do
  (`FitOptions::design`). In the "plain `fit`" columns each gene is fitted on its
  own. The estimates are the same. The benchmark's last argument chooses
  (`shared`, `plain` or `both`).
- One thread per solver.
- Every answer is scored by the same log likelihood. It is evaluated stably at
  large θ: `lgamma` and `digamma` differences are replaced by power sums above
  θ = 1e7, because plain differences lose their precision there.

The "shortfall" is a gene's log-likelihood deficit from the best answer any
method found for it. "Worse" counts the genes with a shortfall above 1e-6
relative to the log likelihood.

## Results

Completed RapidGLM 1.6.0 measurements on `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`); the release library is unchanged from the measured dev.2 source.

One thread on Zen 3 core 6 under the quiet-start and contention protocol in [BENCHMARK.md](BENCHMARK.md#host). Wall times cover all genes. Shortfalls and failures are reported alongside time; Ceres default and tight tolerances remain separate.

**Fixed dispersion:**

| Dataset | Method | Seconds | Genes with relative shortfall > 1e-6 | Largest log-likelihood shortfall | Failed |
| --- | --- | --- | --- | --- | --- |
| airway | rapidglm_newton | 0.1772 | 0 | 1.4e-09 | 0 |
| airway | rapidglm_newton_plain | 0.193 | 0 | 1.4e-09 | 0 |
| airway | ceres_levenberg_marquardt/ceres_default | 0.2461 | 0 | 3.41e-06 | 0 |
| airway | ceres_dogleg/ceres_default | 0.2566 | 0 | 3.41e-06 | 0 |
| airway | ceres_lbfgs/ceres_default | 0.3686 | 9323 | 56.7 | 0 |
| airway | ceres_levenberg_marquardt/tight | 0.3531 | 0 | 3.13e-09 | 0 |
| airway | ceres_dogleg/tight | 0.5475 | 0 | 5.32e-08 | 0 |
| airway | ceres_lbfgs/tight | 0.5044 | 0 | 9.48e-08 | 0 |
| sim5000 | rapidglm_newton | 1.147 | 0 | 7.64e-11 | 0 |
| sim5000 | rapidglm_newton_plain | 1.263 | 0 | 7.64e-11 | 0 |
| sim5000 | ceres_levenberg_marquardt/ceres_default | 1.985 | 0 | 0.00273 | 0 |
| sim5000 | ceres_dogleg/ceres_default | 2.048 | 0 | 0.00275 | 0 |
| sim5000 | ceres_lbfgs/ceres_default | 3.118 | 362 | 13.5 | 0 |
| sim5000 | ceres_levenberg_marquardt/tight | 3.568 | 0 | 2.62e-09 | 0 |
| sim5000 | ceres_dogleg/tight | 3.701 | 0 | 2.65e-09 | 0 |
| sim5000 | ceres_lbfgs/tight | 4.018 | 0 | 6.42e-10 | 0 |
| thyroid | rapidglm_newton | 5.389 | 0 | 7.89e-08 | 0 |
| thyroid | rapidglm_newton_plain | 5.914 | 0 | 7.89e-08 | 0 |
| thyroid | ceres_levenberg_marquardt/ceres_default | 8.112 | 0 | 0.000718 | 0 |
| thyroid | ceres_dogleg/ceres_default | 8.317 | 0 | 0.00107 | 0 |
| thyroid | ceres_lbfgs/ceres_default | 12.23 | 23163 | 2.45e+03 | 0 |
| thyroid | ceres_levenberg_marquardt/tight | 12.13 | 0 | 8.86e-09 | 0 |
| thyroid | ceres_dogleg/tight | 12.43 | 0 | 1.32e-08 | 0 |
| thyroid | ceres_lbfgs/tight | 24.87 | 3 | 0.04 | 0 |

**Estimated dispersion:**

| Dataset | Method | Seconds | Genes with relative shortfall > 1e-6 | Largest log-likelihood shortfall | Failed |
| --- | --- | --- | --- | --- | --- |
| airway | rapidglm_newton | 0.9804 | 198 | 0.000602 | 0 |
| airway | rapidglm_newton_plain | 1.068 | 198 | 0.000602 | 0 |
| airway | ceres_bfgs/ceres_default | 1.128 | 17804 | 28.4 | 0 |
| airway | ceres_lbfgs/ceres_default | 1.112 | 17884 | 28.4 | 0 |
| airway | ceres_bfgs/tight | 1.751 | 6 | 0.327 | 0 |
| airway | ceres_lbfgs/tight | 1.962 | 8 | 0.327 | 0 |
| sim5000 | rapidglm_newton | 2.455 | 0 | 1.27e-05 | 0 |
| sim5000 | rapidglm_newton_plain | 2.576 | 0 | 1.27e-05 | 0 |
| sim5000 | ceres_bfgs/ceres_default | 10.3 | 120 | 42.3 | 0 |
| sim5000 | ceres_lbfgs/ceres_default | 10.29 | 120 | 42.3 | 0 |
| sim5000 | ceres_bfgs/tight | 12.94 | 0 | 8.53e-06 | 0 |
| sim5000 | ceres_lbfgs/tight | 13.18 | 0 | 5.58e-05 | 0 |
| thyroid | rapidglm_newton | 12.54 | 0 | 1.39e-05 | 0 |
| thyroid | rapidglm_newton_plain | 13.54 | 0 | 1.39e-05 | 0 |
| thyroid | ceres_bfgs/ceres_default | 37.79 | 21364 | 8.38e+03 | 0 |
| thyroid | ceres_lbfgs/ceres_default | 37.75 | 21475 | 8.38e+03 | 0 |
| thyroid | ceres_bfgs/tight | 76.31 | 2 | 21.7 | 0 |
| thyroid | ceres_lbfgs/tight | 80.45 | 3 | 0.032 | 0 |

RapidGLM bounds theta at 1e8; near-Poisson shortfalls must be interpreted with that constraint. Plain fit repeats design work for each gene. Three control measurements repeat airway fixed, airway joint and thyroid fixed on the same core; their per-method times remain in the receipt. The historical defects and development measurements below describe how the solver was established and are not substituted for these 1.6.0 timings.

## Defects the comparison found

The first runs showed RapidGLM's maximum-likelihood fits stopping short on
about 2,800 airway genes. All of them were fixed with regression tests
(`tests/test_engine.cpp`, `nb_ml_*`):
1. *Moment start at the bound.* The moment estimate from the Poisson pilot can
   show no overdispersion in nearly saturated designs, which put the start at
   the θ bound. The likelihood is flat there, so Newton stopped at once. Such
   fits now start from the configured θ.
2. *Overshoot on bimodal profiles.* Some genes have a profile likelihood in θ
   with an interior maximum, a dip and a lower Poisson asymptote. Where the
   joint likelihood was not concave, a unit step in θ could carry the fit past
   the maximum toward the bound. θ is now held until the coefficients settle.
3. *Singular information.* Diverging coefficients drove the information to
   zero while θ was still moving. They are now held and excluded from the
   final covariance; their standard errors were already reported as missing.

4. *Local maximum below the Poisson limit.* The opposite of case 2: an
   interior local maximum lower than the Poisson limit. When the moment start
   shows no overdispersion, `fit()` now runs from both o.theta and the bound
   and keeps the higher likelihood.

After the fixes, all 22,369 airway fits converge. Against the previous commit
([validation/JOINT_NEWTON_FIX.json](../validation/JOINT_NEWTON_FIX.json)):
- **Unchanged:**
  - the three 1,000-response batches are identical, to a coefficient
    difference of 3.5e-16;
  - the E1 calibration hash, the binary128 accuracy suite, and the
    perturbation and dose panels (up to 1e-14).
- **Changed:**
  - In the Kang pseudobulk ML fits, 4 genes reach higher log likelihoods (by up
    to 60.5) and none lower.
  - The quasi-likelihood and LR pseudobulk tables follow from those fits. QL
    discoveries are unchanged; LR discoveries agree at Jaccard 0.998.
- **Cost:** the search from both starts costs 8% on the general 1,000-response
  batch (2.05 s against 1.90 s, against 3.3 s for glmGamPoi).

Fixed-dispersion fits, and so `bulk_de`, were never affected.
