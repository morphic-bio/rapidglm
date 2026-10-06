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
- One thread per solver.
- Every answer is scored by the same log likelihood. It is evaluated stably at
  large θ: `lgamma` and `digamma` differences are replaced by power sums above
  θ = 1e7, because plain differences lose their precision there.

The "shortfall" is a gene's log-likelihood deficit from the best answer any
method found for it. "Worse" counts the genes with a shortfall above 1e-6
relative to the log likelihood.

## Results

These runs are from a shared host (1-minute load 3–13). Times are wall seconds
for all genes. Each cell gives the time and either the largest shortfall or the
number of genes worse than the best answer.

**Fixed dispersion** (DESeq2's final GLM):

| Data (samples × columns, genes) | RapidGLM | LM default | LM tight | Dogleg tight | L-BFGS default | L-BFGS tight |
|---|---|---|---|---|---|---|
| airway (8 × 5, 22,369) | 0.38 s; max 1e-9 | 0.33 s; max 3e-6 | 0.46 s; max 5e-8 | 0.69 s | 0.42 s; 9,260 worse | 0.59 s |
| simulation (5,000 × 5, 1,000) | 1.50 s; max 9e-11 | 1.24 s; max 3e-3 | 2.34 s; max 3e-9 | 2.44 s | 2.31 s; 386 worse | 2.89 s |
| GTEx thyroid (647 × 12, 28,769) | 7.94 s; max 8e-8 | 5.85 s; max 7e-4 | 9.50 s; max 5e-8 | 9.94 s | 8.07 s; 23,130 worse | 17.8 s; 3 worse |

**Maximum-likelihood dispersion:**

| Data | RapidGLM joint Newton | BFGS default | BFGS tight | L-BFGS tight |
|---|---|---|---|---|
| airway | 0.98 s; 198 worse, max 6e-4 (θ at its 1e8 bound) | 0.73 s; 17,804 worse, max 28 | 1.27 s; 6 worse, max 0.33 | 1.44 s; 8 worse |
| simulation | 2.09 s; max 1e-5 | 6.30 s; 120 worse, max 42 | 8.12 s; max 9e-6 | 7.99 s |
| GTEx thyroid | 20.6 s; max 1e-5 | 23.4 s; 21,364 worse, max 8.4e3 | 47.4 s; 2 worse, max 22 | 49.9 s; 3 worse |

Summary:
- **Fixed dispersion:** at equal accuracy, RapidGLM is 1.2–1.6× faster than
  Levenberg–Marquardt and dogleg. Levenberg–Marquardt at Ceres's default
  tolerances is faster still, but stops up to 3e-3 short of the optimum.
- **Maximum-likelihood dispersion:** RapidGLM is 1.3–3.8× faster than BFGS or
  L-BFGS at tight tolerances. On airway and thyroid it is also more accurate:
  tight BFGS still misses on a few genes, by up to 22 in log likelihood.
- **Default tolerances:** the quasi-Newton methods stop far from the optimum on
  most genes, as much as 8,400 log-likelihood units short on thyroid.
- **The θ bound:** RapidGLM keeps θ ≤ 1e8. For near-Poisson genes that costs
  at most 6e-4 in log likelihood against an unbounded θ.
- **Simplicity:** RapidGLM's methods reuse the factorizations the fitter
  already needs for standard errors, and need no external optimizer.

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
