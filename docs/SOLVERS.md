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

These final runs use version 1.4.0 (`v1.4.0-dev.4`), completed 2026-10-22, each on one efficiency core of
an Intel Core i9-13900KF (see [BENCHMARK.md](BENCHMARK.md#host)). Times are
wall seconds for all genes. Each cell gives the time and either the largest
shortfall or the number of genes worse than the best answer.

**Fixed dispersion** (DESeq2's final GLM):

| Data (samples × columns, genes) | RapidGLM | RapidGLM, plain `fit` | LM default | LM tight | Dogleg tight | L-BFGS default | L-BFGS tight |
|---|---|---|---|---|---|---|---|
| airway (8 × 5, 22,369) | 0.20 s; max 1e-9 | 0.21 s | 0.35 s; max 3e-6 | 0.47 s; max 2e-9 | 0.70 s | 0.44 s; 9,323 worse | 0.59 s |
| simulation (5,000 × 5, 1,000) | 1.66 s; max 8e-11 | 1.72 s | 2.53 s; max 3e-3 | 4.60 s; max 3e-9 | 4.76 s | 4.01 s; 362 worse | 5.17 s |
| GTEx thyroid (647 × 12, 28,769) | 7.07 s; max 8e-8 | 7.82 s | 10.3 s; max 7e-4 | 15.2 s; max 1e-8 | 15.8 s | 15.7 s; 23,163 worse | 32.0 s; 3 worse |

**Maximum-likelihood dispersion:**

| Data | RapidGLM joint Newton | RapidGLM, plain `fit` | BFGS default | BFGS tight | L-BFGS tight |
|---|---|---|---|---|---|
| airway | 1.12 s; 198 worse, max 6e-4 (θ at its 1e8 bound) | 1.22 s | 1.24 s; 17,804 worse, max 28 | 1.90 s; 6 worse, max 0.33 | 2.16 s; 8 worse |
| simulation | 3.29 s; max 1e-5 | 3.46 s | 11.1 s; 120 worse, max 42 | 14.0 s; max 9e-6 | 14.3 s |
| GTEx thyroid | 15.0 s; max 1e-5 | 16.5 s | 43.3 s; 21,364 worse, max 8.4e3 | 87.6 s; 2 worse, max 22 | 92.4 s; 3 worse |

The final fixed-dispersion speedups are
2.15–2.77× over tight
Levenberg–Marquardt and 2.23–3.52× over tight
dogleg. With estimated dispersion, speedups over tight BFGS are
1.71–5.85×. Accuracy and boundary shortfalls are
reported in the same cells; a fast early stop is not treated as an equally
accurate optimum.

Plain `fit` repeats the design work for each gene. Its wall time divided by
the shared-design path ranges from
1.04–1.11×.
Against plain `fit`, tight Levenberg–Marquardt takes
1.95–2.67×
as long, and tight BFGS takes
1.56–5.30× as long.

RapidGLM bounds θ at 1e8; its near-Poisson boundary shortfalls should be
interpreted with that constraint. Quasi-Newton default tolerances can stop
well short of the optimum, as the tables show. RapidGLM reuses factorizations
already needed for standard errors and needs no external optimizer.

Three control runs repeat on efficiency CPU 16, under the same quiet-start
protocol: airway fixed, airway joint and thyroid fixed. Across all methods,
first measurement / repeat ranges from 0.988 to
1.035. Their per-method times are in
`efficiency_core_control` in the receipt. All timings use the workstation's
16 identical efficiency cores, one thread per core; they are final, not
provisional. One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.

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
