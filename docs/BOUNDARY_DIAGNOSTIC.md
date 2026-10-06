# Paired R/native convergence diagnosis

The two original pseudobulk failures are now handled as `converged_boundary`.
Their count likelihoods stop on deviance stability, boundary coefficients are
flagged, and their donor SEs and Wald tests are suppressed. Their finite
treatment effects remain testable. The separate BLAS concurrency defect was
corrected earlier. The generator is unchanged, and acceptance thresholds remain
provisional. The original diagnosis and current fix are distinguished below.

## Why R stops

NME7 and MRPS25 each have zero counts in both samples of one donor. With an
unpenalized donor intercept, the likelihood approaches its supremum as that
donor's coefficient tends to negative infinity. There is no finite maximum for
that nuisance coefficient. Its fitted means approach zero while its coefficient
continues moving by approximately minus one per IRLS iteration.

R's documented `glm.fit` stopping criterion is
`abs(deviance_new - deviance_old) / (0.1 + abs(deviance_new)) < epsilon`.
Before the fix, RapidGLM required
`max_j(abs(beta_new[j] - beta_old[j]) / (1 + abs(beta_old[j]))) < tolerance`.
The harness sets both tolerances to `1e-9` and supplies identical starting
coefficients. See [R's glm.control documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.control.html)
and `include/rapidglm/model.hpp`.

At iteration 22 of the matched-start Poisson fits:

| Quantity | NME7 | MRPS25 |
|---|---:|---:|
| R relative deviance change | 7.73096e-10 | 7.18313e-10 |
| Native relative coefficient change | 0.0450022 | 0.0459825 |
| Maximum native/R coefficient difference | 2.75e-12 | 1.29e-11 |
| Maximum native/R fitted-mean difference | 5.51e-14 | 6.75e-14 |
| Sum of fitted means for the zero-count donor | 2.07523e-9 | 1.95667e-9 |
| Gap to independent Poisson likelihood supremum | 2.07523e-9 | 1.9567e-9 |
| R status | converged | converged |
| Native status before the fix | iteration_limit | iteration_limit |
| Native status after the fix | converged_boundary | converged_boundary |

The identifiable treatment estimates already match an independent likelihood
oracle to about 1e-15. The oracle profiles each donor intercept analytically,
assigns zero means to zero-total donors, and solves the remaining scalar
treatment score by Brent's method. It does not fit an infinite intercept.

Adding one count to the zero donor removes the boundary: both engines then
converge for Poisson and fixed-theta NB within the ordinary iteration budget.
A seeded synthetic boundary and its positive-count control reproduce this
contrast. Raising the iteration limit does not create a finite boundary MLE.

The implemented count-family stopping rule uses relative deviance change and
retains the old coefficient precision guard for finite coefficients. Flagged
boundary coefficients are exempt from that guard. A LAPACK SVD projects the
IRLS direction into the null space of positive-count rows; a numerical
certificate checks zero change on positive counts, nonpositive change on zeros,
and a strict decrease on at least one zero row. The remaining-design null space
identifies coefficients with no unique finite value on this limiting face.
This handles changes of reference donor and predictor units. It is not an
exhaustive LP separation detector; uncertified nonstationary fits still fail.
No data, weights or columns are removed from the fit.

Boundary coefficients retain their finite iteration values for diagnostics,
with explicit flags and NaN SEs and covariance rows/columns. Their ordinary and
robust Wald tests are invalid. Finite treatment Wald tests remain available.
LR and score inference reject boundary fits conservatively; the perturbation
workflow retains affected nulls with an explicit status and missing tests.
Gaussian and binomial stopping rules are unchanged.

The new regression suite covers Poisson, fixed/moment/ML NB, alternate reference
donors, signed predictor scaling, analytic treatment uncertainty, ridge, zero
weights, positive-count controls, compact results and workflow inference guards.
The deviance is separately checked against independent saturated likelihoods
and near-saturation curvature. All 32 C++ cases, six CLI tests, ASan/UBSan and
the installed external consumer pass.

The matched-start Poisson fits now stop at iteration 22 for both real genes.
Their treatment estimates differ from the independent oracle by less than
1e-15. All six diagnostic cases have the expected statuses and coefficient
flags, suppress donor inference only in boundary cases, and retain treatment
tests. NB proceeds past its Poisson pilot and flags its upper theta bound.
The full 1,000-gene pseudobulk panel now has 1,000 valid treatment tests:
998 ordinary fits and two explicit boundary fits. Its tables match byte-for-byte
across 1/2/8/16 threads. All 96 retained ordinary fits and LR comparisons are
numerically unchanged; all 8,000 E1 calibration rows are byte-identical.

## NB comparison limitations

Before this fix, RapidGLM's ML-NB fit stopped in its Poisson pilot. R's `glm.nb`
proceeds, but its `converged` flag is not evidence that all parameters have a
regular finite optimum. With default starts, both real genes produce
`NaNs produced` warnings and theta in the billions, approaching the Poisson
limit. With the shared start, NME7 errors with
`missing value where TRUE/FALSE needed`; MRPS25 emits warnings too.
All warnings, errors, estimates and public trace output are retained.

R couples coefficient, theta and alternation budgets through `maxit`; the
native outer limit remains 60. Exact iteration comparisons therefore isolate
Poisson and fixed-theta NB. See the public
[glm.nb documentation](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/glm.nb.html).
No R/MASS implementation source was inspected or copied.

## Separate BLAS concurrency defect

The original local LAPACK migration linked Ubuntu's serial OpenBLAS package.
That build is unsafe for the engine's concurrent callers. A deterministic
regression solves and inverts 1,024 well-conditioned matrices, with analytic
solutions, at several thread counts. It contains no RNG, GLM iterations or
boundary data. The serial build passed at one thread but failed at two and
eight threads in all three preserved replays, with errors of order one.

The local release, sanitizer and external-consumer builds now link pthread
OpenBLAS 0.3.20 with `OPENBLAS_NUM_THREADS=1`. The regression passes at
1/2/8/16 outer threads. All twelve extended original-CLI replays (three per
requested thread count) were byte-identical and retained
`iteration_limit` before the stopping-rule fix. This corrected the earlier audit's attribution of enormous
coefficients and false convergence solely to boundary handling. The boundary
stopping mismatch remains independently reproducible with one thread.

A thread-safe BLAS with one internal worker is required. A serial OpenBLAS
build needs `USE_LOCKING=1`; setting `OPENBLAS_NUM_THREADS=1` does not add locks
to a library built without them. See the
[upstream build options](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/build_system.md).
The prior serial-package recommendation has been corrected. No custom solver
or per-fit global thread-setting call was introduced.

## RNG and input equivalence

- The real gene counts, design and offsets are fixed data. Neither fitter uses
  RNG. Every R fit checks that `.Random.seed` is unchanged, including error paths.
- Synthetic counts are generated once, serially, using R Mersenne-Twister,
  Inversion/Rejection and seed `20260926`. Both engines consume those same
  values, rather than assuming equal seeds imply equal streams in different
  implementations.
- R writes little-endian IEEE binary64 inputs for both engines. The native
  executable echoes every consumed design, response, offset, weight and start;
  the Python harness verifies the bytes exactly.
- Production resampling uses Philox4x32-10, with private state keyed by seed,
  test ID and draw ID. Raw streams and retained null statistics are checked
  for exact equality at 1/2/8/16 threads.
- The E1 validation generator owns a separate `std::mt19937_64` and distribution
  state per dataset, seeded by dataset ID. It does not share a mutable generator
  between workers. Complete dataset outputs are now directly compared across
  1/2/8/16 threads. This establishes scheduling reproducibility in this build,
  not cross-platform equivalence of C++ standard-library distributions.

## Reproduce and inspect

Use a thread-safe numerical dependency as described in the README, then:

```bash
cmake --build build -j 2
OPENBLAS_NUM_THREADS=1 build/rapidglm_tests \
  --run_test=cholesky_concurrent_independent_systems --log_level=message
python3 scripts/diagnose_boundary.py \
  --work validation/work/boundary-diagnostic-safe \
  --receipt validation/BOUNDARY_FIX.json
```

The pseudobulk inputs are exported by `scripts/reference_applications.R`. R requires MASS, Matrix and jsonlite; Python
requires NumPy, SciPy, pandas and matplotlib. The diagnostic executable is
built with `RAPIDGLM_TESTS=ON`.

`scripts/reference_boundary.R` invokes the public R APIs with default and
matched starts. `tests/diagnose_boundary.cpp` calls the production native API.
Each iteration prefix is a fresh fit from the same start with a truncated
budget; the harness does not insert a replacement solver. Prefix plots exclude
repeated final results after an engine has already stopped.

The work directory contains per-case `r_prefix.tsv`, `native_prefix.tsv`,
derived metrics, full-run summaries, R trace JSON files and original CLI replay
outputs. `convergence.png` and `convergence.svg` show the donor coefficient,
relative deviance change and relative coefficient change. Before-switch
evidence remains under `validation/work/boundary-diagnostic` and
`validation/work/blas-concurrency-before`.

Tracked receipts:

- [BOUNDARY_DIAGNOSTIC.json](../validation/BOUNDARY_DIAGNOSTIC.json): paired
  results before the stopping-rule fix, oracle, exact-input checks, seeds,
  versions, hashes and commands.
- [BLAS_THREAD_SAFETY.json](../validation/BLAS_THREAD_SAFETY.json): failing and
  corrected backend configurations, regression logs and revalidation.
- [BOUNDARY_FIX.json](../validation/BOUNDARY_FIX.json): current paired results,
  finite-treatment oracle errors and explicit boundary-inference checks.
- [BOUNDARY_FIX_CHECKS.json](../validation/BOUNDARY_FIX_CHECKS.json): unit, CLI,
  sanitizer, external-consumer, pseudobulk, ordinary-fit and calibration replays.

Historical G3/G5 calibration failures are retained. This diagnosis does not
establish their cause or qualify the provisional release gates.
