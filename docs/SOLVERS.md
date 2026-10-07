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

These runs are from version 1.2.0 (2026-10-06), each on one efficiency core of
an Intel Core i9-13900KF (see [BENCHMARK.md](BENCHMARK.md#host)). Times are
wall seconds for all genes. Each cell gives the time and either the largest
shortfall or the number of genes worse than the best answer.

**Fixed dispersion** (DESeq2's final GLM):

| Data (samples × columns, genes) | RapidGLM | RapidGLM, plain `fit` | LM default | LM tight | Dogleg tight | L-BFGS default | L-BFGS tight |
|---|---|---|---|---|---|---|---|
| airway (8 × 5, 22,369) | 0.21 s; max 2e-9 | 0.23 s | 0.33 s; max 3e-6 | 0.47 s; max 1e-9 | 0.70 s | 0.45 s; 9,254 worse | 0.59 s |
| simulation (5,000 × 5, 1,000) | 1.60 s; max 9e-11 | 1.66 s | 2.37 s; max 3e-3 | 4.28 s; max 3e-9 | 4.41 s | 4.08 s; 386 worse | 5.28 s |
| GTEx thyroid (647 × 12, 28,769) | 7.43 s; max 8e-8 | 8.25 s | 10.2 s; max 7e-4 | 15.3 s; max 3e-8 | 15.8 s | 15.9 s; 23,162 worse | 32.0 s; 3 worse |

**Maximum-likelihood dispersion:**

| Data | RapidGLM joint Newton | RapidGLM, plain `fit` | BFGS default | BFGS tight | L-BFGS tight |
|---|---|---|---|---|---|
| airway | 1.12 s; 198 worse, max 6e-4 (θ at its 1e8 bound) | 1.22 s | 1.24 s; 17,804 worse, max 28 | 1.92 s; 6 worse, max 0.33 | 2.17 s; 8 worse |
| simulation | 3.20 s; max 1e-5 | 3.32 s | 11.2 s; 120 worse, max 42 | 14.0 s; max 9e-6 | 14.3 s |
| GTEx thyroid | 15.4 s; max 1e-5 | 17.1 s | 43.3 s; 21,364 worse, max 8.4e3 | 87.6 s; 2 worse, max 22 | 92.3 s; 3 worse |

Summary:
- **Fixed dispersion:** at equal accuracy, RapidGLM is 2.1–2.7× faster than
  Levenberg–Marquardt and 2.1–3.4× faster than dogleg. It is also 1.4–1.6×
  faster than Levenberg–Marquardt at Ceres's default tolerances, which stops
  up to 3e-3 short of the optimum. Both take a median of 3 to 5 iterations;
  RapidGLM's are cheaper.
- **Maximum-likelihood dispersion:** RapidGLM is 4.4–6.0× faster than BFGS or
  L-BFGS at tight tolerances on the simulation and thyroid, and 1.7–1.9× faster
  on airway. On airway and thyroid it is also more accurate: tight BFGS still
  misses on a few genes, by up to 22 in log likelihood.
- **Default tolerances:** the quasi-Newton methods stop far from the optimum on
  most genes, as much as 8,400 log-likelihood units short on thyroid.
- **The θ bound:** RapidGLM keeps θ ≤ 1e8. For near-Poisson genes that costs
  at most 6e-4 in log likelihood against an unbounded θ.
- **Plain `fit`:** fitting each gene on its own takes 3–11% longer than sharing
  the design-only work. Against plain `fit`, tight Levenberg–Marquardt takes
  1.9–2.6× as long and tight BFGS 1.6–5.1×.
- **Simplicity:** RapidGLM's methods reuse the factorizations the fitter
  already needs for standard errors, and need no external optimizer.
- **Core type:** three of the runs were repeated on a performance core
  (`performance_core_control` in the receipt). RapidGLM took 0.11 s, 0.63 s
  and 3.67 s (airway fixed, airway ML, thyroid fixed), 1.8–2.0× less than
  above; the tight Ceres methods gained 1.7–2.0×. There RapidGLM is 2.1× and
  2.4× faster than tight Levenberg–Marquardt (airway, thyroid) and 1.8×
  faster than tight BFGS on airway.
- **Before 1.2.0** a fixed-dispersion fit spent about two thirds of its time
  outside the Newton iterations (response-only likelihood terms, a
  confirming iteration and per-fit design work; [CHANGELOG.md](../CHANGELOG.md)).
  RapidGLM then took as long as tight Levenberg–Marquardt on thyroid.

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
