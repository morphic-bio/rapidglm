# Numerical implementation and reuse review

The initial GLM fitter in `include/rapidglm/model.hpp` was written in this
repository in commit `88661b5`. It was not imported from an established GLM
library. Its coefficient iterations implement IRLS/Fisher scoring; its NB
iterations alternate coefficient fitting and bounded optimization of log(theta).
The initial scaled Cholesky factorization and triangular solves were also written here.
Boost provides special functions. R/MASS execute through public APIs in separate
reference processes. The statistical sources are recorded in `METHODS.md`.

Existing numerical implementations should be used for the linear algebra.
The clean-room constraint does not require a custom matrix solver.

## Existing project code checked

- [FastBMA](https://github.com/lhhunghimself/fastBMA), inspected at
  `a03ce6922b0bc8cf007fbc1a46e0a883a74d0251`, is MIT licensed and links
  OpenBLAS. It uses the library in three ways:
  - one large `cblas_dgemm` computes X'X (and a `dgemv` X'b) once, with
    OpenBLAS's own threads enabled for that call only;
  - its model search mostly avoids factorization: adding or removing a
    regressor updates the existing Cholesky factor with `cblas_dtrsv` and
    Givens rotations (`dlartg`), in O(p^2);
  - [`dpotrf`](https://github.com/lhhunghimself/fastBMA/blob/a03ce6922b0bc8cf007fbc1a46e0a883a74d0251/src/fastBMA.hpp#L2300)
    builds a factor from scratch only for a search's first models, or when an
    update gives an unusable R^2.

  Genes are spread over OpenMP threads with `schedule(dynamic)`, which
  RapidGLM's `parallel_for` adopted on 2026-09-30 (below). GLM IRLS changes all
  observation weights each iteration, so the update machinery is not a direct
  substitute for refactorizing weighted information. The wrappers do not
  propagate LAPACK's `info`; RapidGLM keeps its explicit failure reporting.
- The local STAR-suite SLAM-seq `slam/source/SlamSolver.cpp` and
  `slam/source/libem/slam_vb_overdisp.cpp` fit scalar mixture proportions by EM.
  They offer likelihood caching and fit-reuse patterns, but not a GLM matrix
  solver. Their existing `tests/slam/test_slam_fit_reuse.cpp` checks serial/parallel
  equality and cache invalidation.

## Backend direction

For p predictors, a fresh dense Cholesky factorization costs O(p^3); a single
column insertion/deletion update costs O(p^2) once the relevant cross-products
are available. The savings are in the predictor count, not automatically the
observation count. Building weighted cross-products from n observations still
costs O(n p^2).

The applicable reuse points in the current engine are:

| Work | Reuse opportunity |
|---|---|
| Gaussian batch with fixed design and prior weights | One factor for all responses; triangular solves per response. Currently `fit_batch` refits independently. |
| Gaussian nested models at fixed weights | Predictor insertion/deletion updates or block solves can replace full refactorization. |
| Repeated scores from one fitted null | `ScoreWorkspace` already uses a Schur complement to avoid fitting the augmented model for each draw. The perturbation workflow still rebuilds the workspace for each pair; it can also be cached per shared null. |
| Count/binomial IRLS | Working weights usually change for every observation. Reuse is valid only where the actual weighted information stays unchanged or its change has useful low rank. |

The optimization priority is factor reuse and applicable updates on top of a
library factorization, not a new implementation of dense Cholesky itself.

A diagnostic profile on the public S2 fitter panel (12 NB responses, 45,919
observations, four predictors) counted 334 Cholesky factorizations, including
Poisson pilots. Its 334 weighted cross-products visited 15,336,946 rows and
performed 153,369,460 lower-triangle accumulator updates. The completed E1
calibration used 3,992,000 resampled scores; the full-public S2 workflow used
3,391,209,988 CRT cell-level random draws. Factorization is outside the resampling
draw loop. These counts establish substantial repeated observation-level work;
they do not establish Cholesky as a major bottleneck or a backend speedup.
The profile is diagnostic, with selected routines prevented from inlining to
expose call counts, not a performance comparison.

Use [LAPACK](https://github.com/Reference-LAPACK/lapack) for factorizations and
solves and [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) or another compatible
BLAS for matrix operations, following FastBMA's approach. Both upstream projects
publish permissive BSD licenses. The standard C interfaces (LAPACKE and CBLAS)
avoid maintaining Fortran ABI declarations. Reusing these libraries directly
does not require importing FastBMA's application and model-search code.

The production backend now calls LAPACKE `dpotrf_work`, `dpotrs_work` and
`dpotri_work`, following this direction. Diagonal scaling, the configurable
scaled-pivot rank threshold and explicit error checks remain. Row-major lower
storage is interpreted as column-major upper storage, avoiding conversion copies.
The dependency is exported through CMake for external consumers.

Since 2026-09-30, systems of at most 16 columns are factored, solved and
inverted in-house, with an exact reciprocal condition number from the explicit
inverse (LAPACK's `dpocon` estimate is used above that). For such small
systems the per-call LAPACK overhead and OpenBLAS's shared buffer lock dominate,
and concurrent callers serialize on the lock.

In OpenBLAS 0.3.20, `dpotrf`, `dpotrs` (via `dtrsm`), `dpotri` and `dpocon` (via
`dtrsv`) each take a scratch buffer from a process-wide pool
(`blas_memory_alloc`) guarded by one mutex; `dlartg` takes none. In gdb
samples of eight threads calling `dpotrs` on 5 × 5 systems, 21 of 32 stacks were
waiting on that mutex. A standalone driver calling the Fortran routines gave
these ratios of total `dpotrf` calls per second, 8 threads against 1 (shared host):

| p | pthread OpenBLAS 0.3.20 | reference LAPACK 3.10 |
|---|---|---|
| 5 | 0.20 | 4.6 |
| 20 | 1.06 | 4.6 |
| 50 | 5.6 | 4.5 |
| 200 | 4.6 | 4.5 |

`cblas_dtrsv` behaves the same way (n = 5: 25M calls/s on one thread, 3.0M on
eight). OpenBLAS is 2.7–3.6× faster per call than reference LAPACK from p = 20,
so it stays in use above 16 columns. The serial OpenBLAS package gave wrong
results under concurrent calls; a serial build needs `USE_LOCKING=1` for
concurrent callers (see the [boundary diagnostic](BOUNDARY_DIAGNOSTIC.md)).

With LAPACK, the bulk DE workflow
on airway (22,369 genes, 8 samples) takes 1.93 s on one thread and 3.10 s on
eight; in-house, it takes 1.40 s and 0.44 s. On the frozen panels the change
moves statistics at rounding level only, the discovery sets are identical and
E1 is byte-identical ([SMALL_CHOLESKY.json](../validation/SMALL_CHOLESKY.json)).
Larger systems keep the LAPACK path.

Also since 2026-09-30, `parallel_for` hands out items one at a time
(`schedule(dynamic)`) instead of in fixed blocks, following FastBMA. Per-feature
costs vary widely; bulk genes near a coefficient boundary can take hundreds of
iterations. Each item is independent, so outputs do not change: all 12 frozen
panel cases and the airway bulk table are byte-identical. On airway bulk-de,
8 threads went from 0.434 s to 0.380 s and 16 threads from 0.421 s to 0.383 s;
the perturbation panels, pseudobulk, dose and scan timings stayed within
run-to-run noise. Coarser chunks (8 or 64 items) were slower on a panel with
few response batches, which then went to a few threads.

Count-boundary diagnostics also use LAPACK `dgesvd_work` for null spaces after
a fast Cholesky check of the positive-count design. SVD is used only when that
design may be rank deficient, while checking convergence. A projected IRLS
direction must satisfy the count-support recession inequalities before any
coefficient is flagged. This adds no custom factorization and changes neither
the draw loop nor the thread-safe backend configuration.

`benchmarks/kernels.cpp` isolates factorization, solve, inverse, four-family fits,
uniform generation, CRT sampling, score accumulation and combined sampling/score
costs. `scripts/benchmark_kernels.py` repeats and optionally alternates baseline
and current executables. Inputs, fit setup and score-bank generation are outside
the timed regions. NB timing here uses fixed theta; it does not benchmark ML
dispersion estimation. Timing ranges and host load are retained.

The CRT sampler is unchanged: one uniform RNG value, one precomputed propensity
lookup and a comparison per cell. There is no inverse-CDF or GLM evaluation in
that loop. OpenMP already parallelizes response features and response batches;
the per-test draw loop is sequential. Keep BLAS single-threaded within those
outer loops. Polynomial sampling approximations and additional draw-level
parallelism were not introduced, following the author's direction.

See `validation/LAPACK_BACKEND.json` for historical migration checks and measurements.
Passing numerical tests does not resolve the two calibration checks that were
not met (README, Validation and status).

**Concurrency correction:** the initially selected serial OpenBLAS package
failed a new deterministic concurrent solve/inverse regression. Its earlier
small-panel thread checks were insufficient to establish backend thread safety.
The local builds now use pthread OpenBLAS with `OPENBLAS_NUM_THREADS=1`.
Serial OpenBLAS requires `USE_LOCKING=1` for concurrent callers. The regression
and corrected-backend revalidation are retained in
`validation/BLAS_THREAD_SAFETY.json`; the paired R/native investigation is in
[BOUNDARY_DIAGNOSTIC.md](BOUNDARY_DIAGNOSTIC.md). Historical timings below remain
measurements of the original configuration, not a qualification of that build.

The migration passed 25 C++ cases in release and sanitizer builds, five CLI
cases and an installed external consumer using a public S1 response. Across
96 retained fits and their LR comparisons, the maximum difference measured as
`abs(new-old)/(1+abs(old))` was 1.35e-12, with matching status/nonfinite fields.
A 96-dataset calibration sample was byte-identical before/after the backend
change and at 1/2/8/16 threads.

Two diagnostic benchmark sessions, each with five repetitions per executable,
measured approximately 1.5x faster factorization at 64 predictors and 1.9x at
128 predictors. Inversion improved across the tested sizes. Small factorizations
and solves can regress; four-predictor whole-fit timings were essentially
unchanged. Both sessions and their variability are retained. These shared-host
microbenchmarks do not establish an end-to-end workflow speedup.

## Deferred: shared cross-products for Gaussian batches

Not implemented; recorded on 2026-09-30 for work on Gaussian data. FastBMA
computes X'X and X'b once with one threaded `dgemm` and evaluates every model
from sub-blocks. It appends b to the Cholesky factor of X'X, so the residual sum
of squares is the square of the last diagonal element. The same idea would make
Gaussian batches with a shared, unweighted design far cheaper than per-response
fits:
- `fit_batch`: factor X'X once; form X'Y for all responses with one `dgemm`;
  solve for all coefficients with one multi-right-hand-side solve; take each
  residual sum of squares as y'y − b'X'y; and scale one inverse by each
  response's variance for the standard errors;
- score scans of Gaussian responses: residualize responses and predictors on
  the design once, then one matrix product gives every pair's statistic.
  Permutations can be batched the same way, as extra predictor columns;
- effect fits in Gaussian scans: FastBMA's column-append Cholesky update
  (a triangular solve and one square root) replaces a refit.

It applies only when X'X is shared. Count and logistic GLMs solve with X'WX,
and W depends on each response's fitted means and changes every iteration; no
current workflow or validation panel is Gaussian. The count-model analogues are
already in place: grouped designs use per-group sums, and the scan fits each
response's null once and reuses its inverse information. Use cases that would
need it: eQTL mapping on normalized expression, limma-style linear models on
log-expression, and permutation-calibrated Gaussian scans. The published
method for all-pairs eQTL products is Matrix eQTL
([Shabalin 2012](https://doi.org/10.1093/bioinformatics/bts163)); the
`MatrixEQTL` package is LGPL-3, so only the paper may be followed. Keep BLAS
single-threaded inside RapidGLM's outer OpenMP loops. Do not copy FastBMA's
global `openblas_set_num_threads` switching, which is unsafe there; one large
`dgemm` per response block, called from each worker, amortizes the buffer lock.
