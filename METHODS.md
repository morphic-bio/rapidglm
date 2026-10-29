# RapidGLM method contract

Version 0.1, 2026-09-26. This is an independent implementation from the sources
below. No SCEPTRE, edgeR, DESeq2, PyDESeq2, Signac or perturb-resample implementation is
read, translated, linked or redistributed. Reference packages execute only in
separate comparator processes. Numerical and biological qualification are
separate; the acceptance plan this contract was first written against was
retired on 2026-09-30 (README, Validation and status).

## Models and fitting

Rows of X are observations; columns are explicit predictors (including an
intercept if wanted). eta = X beta + offset. Prior weights multiply each
observation's log likelihood. Responses are finite, nonnegative integer counts
for Poisson/NB; proportions in [0,1] for binomial (weights may encode trials);
finite real values for Gaussian. Zero prior weight excludes an observation.
No implicit intercept, normalization, missing-value deletion or factor coding.

Families: Gaussian identity, Poisson log, binomial logit, NB2 log with
Var(Y)=mu+mu^2/theta. Fisher scoring/IRLS uses
w_i=a_i (dmu/deta)^2/V(mu), score contributions
s_i=a_i (y_i-mu_i)(dmu/deta)/V(mu), and solves
(X'WX + lambda I) delta = X's - lambda beta. For NB at a fixed theta the step
uses the observed information instead, w_i = a_i theta mu_i (theta + y_i) /
(theta + mu_i)^2, which is positive for the log link. That is Newton's method,
which converges quadratically where Fisher scoring converges only linearly: on
airway's 22,369 final fits the mean number of iterations fell from 10.4 to 4.5.
Covariances still use the expected information. Step halving requires a
nondecreasing penalized likelihood. Optional ridge penalizes all coefficients;
its covariance is the inverse penalized information, not an unpenalized MLE
covariance. Warm starts are explicit. Gaussian SEs use residual dispersion
sum a_i(y_i-mu_i)^2/(n_positive_weight-p); Gaussian likelihood profiles the
ML dispersion using n_positive_weight. Binomial/Poisson dispersion is one.
Sources: [Nelder & Wedderburn 1972](https://doi.org/10.2307/2344614),
[R GLM documentation](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/glm.html).

Count-family coefficient iterations require relative deviance change
`abs(D_new-D_old)/(0.1+abs(D_new)) < tolerance`, with `ridge * sum(beta^2)`
added to the deviance when penalized. Finite coefficients must also satisfy the
existing relative coefficient-change tolerance, retaining ordinary-fit precision.
Gaussian and binomial stopping rules are unchanged. The deviance criterion follows
[R's documented convention](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/glm.control.html).

From 1.2.0 a step that could only confirm convergence is not taken. After each
step solved by Cholesky factorization, the score u at the new estimate and that
step's factor give the chord step d = H^-1 u and the deviance change u'd it
promises. When both meet the criteria above, and no working weight has moved by
more than a quarter since the factor was formed, the fit has converged without
another cross-product, factorization and likelihood. The chord step is then
applied as a last refinement, unless it would lower the likelihood or a
coefficient is at a boundary, so the estimate agrees with that of a further
full step to second order in the step. Steps solved by QR, and grouped designs,
converge as before. The deviance serves only as the scale of the criterion and
is evaluated in double precision.

Likelihood terms that depend on the response alone are computed once per
response: counts up to 64 through their histogram, larger ones by Stirling's
series with one logarithm each, in the form whose terms are of the size of the
result (the difference lgamma(y + theta) - lgamma(y + 1) for theta <= y + 1,
and lgamma(y + theta) - lgamma(theta) less y log theta otherwise). Their sums
agree with binary128 references to 1e-16 relative. Designs in which at most
half of the entries are nonzero, such as designs of factors, form X'WX, X'v and
X b from each row's nonzero entries.

For unpenalized count fits, a candidate boundary direction is projected into the
null space of the positive-count design using LAPACK SVD. A numerical certificate
requires zero change on positive counts, nonpositive change on zero counts and
a strict decrease for at least one zero-count observation. Zero prior weights
are excluded. The null space of the remaining design identifies coefficients
without a unique finite value at that limiting face. Those coefficients are
exempt from the coefficient-change guard, not from likelihood convergence.
Their finite iteration values are retained only as diagnostics; flags identify
them and their standard errors and covariance rows/columns are NaN.

This is a certificate for the projected IRLS direction, not an exhaustive linear
programming separation detector. Uncertified changing coefficients remain subject
to the coefficient guard and iteration limit. Data, weights and design columns
are never dropped or altered. The geometric condition and distinction between
finite and nonfinite coefficients follow
[Correia, Guimaraes and Zylkin](https://arxiv.org/abs/1903.01633);
the projection/certificate workflow is this implementation's choice.

NB likelihood per observation is
lgamma(y+theta)-lgamma(theta)-lgamma(y+1)
+theta log(theta/(theta+mu))+y log(mu/(theta+mu)). Default ML (no ridge, no
Cox-Reid adjustment) takes joint Newton steps in (beta, u = log theta) from the
Poisson pilot and the moment theta, using the observed information:
- The coefficient block A = X'WX has W = a theta mu (theta+y)/(theta+mu)^2 > 0,
  so it is positive definite for the log link.
- The mixed derivatives are c = X'[a theta mu (y-mu)/(theta+mu)^2]. The
  curvature in u is h_u = theta^2 d2l/dtheta2 + theta dl/dtheta.
- With slope g_u + c'A^-1 s and Schur complement h_u + c'A^-1 c, the step is
  du = -slope/Schur when the Schur complement is negative, and otherwise one
  unit up the slope. du is clipped to +-2 and to the theta bounds, and then
  dbeta = A^-1 (s + c du).
- Both cases ascend, and step halving on the joint likelihood keeps every
  iterate an improvement.
- Theta keeps its exact value when du = 0. For counts above 64 near the upper
  bound, the likelihood's lgamma differences carry about 1e-10 of rounding,
  which a one-ulp change of theta would expose to the acceptance test.
- Convergence requires the relative objective change below the tolerance, on
  the deviance scale at the start. The relative coefficient change must be
  below max(tolerance, resolution) and the change in log theta below the
  tolerance.

Ridge-penalized and Cox-Reid fits alternate coefficient fits at fixed theta
with a bounded one-dimensional Newton optimization in log(theta). All fits
recompute the expected information at the final parameters for standard
errors. `iterations` counts joint steps for default NB ML fits and coefficient
steps otherwise. Bounds [1e-6,1e8] and
boundary flags are our numerical choices; upper-bound fits approximate Poisson.
The moment estimate uses sum a_i mu_i^2 / sum a_i((y_i-mu_i)^2-mu_i),
clipped to these bounds, after a Poisson pilot fit. Shared dispersion maximizes
the sum of feature profile likelihoods; a supplied positive per-feature trend
can instead fix theta. Trend estimation is an explicit log-dispersion versus
log-mean regression of independent ML fits, not empirical Bayes shrinkage.
Sources: [MASS glm.nb documentation](https://stat.ethz.ch/R-manual/R-patched/library/MASS/html/glm.nb.html),
[Lawless 1987](https://doi.org/10.2307/3314912). Shared/trend pooling and bounds
are RapidGLM choices, not claims of edgeR/DESeq2 equivalence.

With `cox_reid`, theta maximizes the Cox-Reid adjusted profile likelihood
log L(beta, theta) - log det(X'WX)/2, W = diag(a mu theta/(theta+mu)), at the
current means (Cox and Reid 1987; McCarthy, Chen and Smyth 2012), alternating
with coefficient fits at fixed theta. Columns flagged at a boundary are left
out of X. The adjustment's first and second theta derivatives are analytic
(tr(A^-1 A') and tr(A^-1 A'') - tr((A^-1 A')^2)). Where the adjusted
likelihood is flat, near the Poisson limit, the alternation converges slowly
and linearly, so three successive log theta iterates are extrapolated by
Aitken's method whenever their ratio of differences lies in (-1, 1). Plain ML
fits do not use this extrapolation.

Score scans (`scan`, `score_scan`) fit each response's null model once, with
the observations retained, and build its score workspace. Each paired
predictor g then gets the efficient score u = g'r - c'A^-1 X'r and information
I = g'Wg - c'A^-1 c, where r are the score residuals, W the working weights,
A = X'WX and c = X'Wg; z = u / sqrt(I scale).
- **Binary predictors** use the sparse form, O(nnz(g) p) per statistic.
- **One-step estimate:** u/I, with standard error sqrt(scale/I).
- **Asymptotic calibration** refers z to the normal distribution.
- **Permutation** redraws a binary predictor's 1s as a uniform subset of the
  same size within strata (Floyd's algorithm), or permutes a continuous
  predictor's values within strata (Fisher-Yates on a fresh copy per draw).
- **CRT** draws a binary predictor from its logistic regression on the design.
  The regression is fitted once per predictor.
- **Staged budgets:** after each budget a skew-normal fit that is accepted
  (`fit_check`, Resampling below) supplies the tail probability; otherwise the
  empirical tail with +1 correction is used.
- **Adjustment:** BH or Bonferroni within each pair family.
- **Effect fits:** optional full fits add the predictor as a column and report
  its Wald estimate. They are warm-started from the null fit, as in the
  perturbation workflow.

The perturbation workflow is a configuration of this scan. Validation (exact
equivalence, null calibration for four families and all calibrations, thread
invariance, throughput) is in `validation/SCAN.json`.

The quasi-likelihood test (`test=ql`) follows Lund, Nettleton, McCarthy and
Smyth (2012) with the recipe of glmGamPoi (Ahlmann-Eltze and Huber 2020):
- Cox-Reid adjusted ML dispersions phi_g are fitted per feature.
- The trend phi_t is a weighted running median of phi_g against the mean
  fitted count m_g. The window covers max(10% of features, 100) ranks, with
  Gaussian weights over +-3 standard deviations of the rank offset, and the
  lower weighted median is used. Ends take the nearest full window.
- Full and reduced models are refitted at theta = 1/phi_t. Their deviance
  difference D is the LR statistic at that common dispersion.
- The QL dispersion is s2 = (1 + m phi_g)/(1 + m phi_t), the ratio of the
  feature's NB variance to the trended one at its mean, on n - rank residual
  df d.
- Smyth's (2004) prior s2/s0^2 ~ F(d, d0) is fitted by maximum likelihood.
  d0 is profiled on a log grid over [1e-3, 1e7] and refined by golden
  section; for fixed d0 the log likelihood is concave in log s0^2 and is
  maximized by damped Newton. The upper limit stands for d0 = infinity.
- With at least 100 features, log s0^2 is a natural cubic spline in log m.
  The spline has boundary knots at the range and interior knots at its
  thirds, in the truncated power basis of Hastie, Tibshirani and Friedman
  (2009). Features with m <= 1e-8 keep the intercept-only scale.
- The posterior is s2_post = (d0 s0^2 + d s2)/(d0 + d), and
  F = D/(q s2_post) is referred to F(q, d0 + d) with q tested columns.
- Reported standard errors and covariances are those of the trended fit,
  multiplied by s2_post.

Validation is in `validation/QL_TEST.json` and [CLI.md](docs/CLI.md).
Sources: [Lund et al. 2012](https://doi.org/10.1515/1544-6115.1826),
[Smyth 2004](https://doi.org/10.2202/1544-6115.1027),
[McCarthy et al. 2012](https://doi.org/10.1093/nar/gks042),
[Ahlmann-Eltze and Huber 2020](https://doi.org/10.1093/bioinformatics/btaa1009).

Poisson and NB fits without a ridge penalty check whether the design's rows
take exactly p distinct values that form an invertible p x p matrix U
(well-conditioned: reciprocal condition above 1e-10). This covers
intercept-only, one-way and full factorial layouts. Then eta is constant within
each group, X'WX = U' diag(W_g) U and X'W z = U' S_g, so each Fisher step is
U^-1 (S_g / W_g), computed from per-group sums without the cross-product.
Poisson fits (including the NB pilot) start at the closed-form MLE
b_g = log(sum_g a y / sum_g a e^offset) when every group total is positive.
That start is also the NB MLE at any theta when offsets are constant within
groups. Step halving, boundary detection, theta estimation and the final
covariance are unchanged. The iterates equal the Cholesky path's up to
rounding. Other designs, `solver=qr` and ridge fits use the general path.

Least-squares steps use a Jacobi-scaled Cholesky factorization of X'WX while
its reciprocal condition number is at least 1e-8. The factorization, solves,
inverse and an exact reciprocal condition number are computed in-house: for
small systems LAPACK's call overhead and OpenBLAS's shared buffer lock dominate
and serialize concurrent callers, and at every size the results then do not
depend on the BLAS kernels chosen for the processor. With the build option
`RAPIDGLM_LAPACK_SOLVES`, systems above 16 columns use LAPACK instead (`dpotrf`,
`dpotrs`, `dpotri`, and its `dpocon` estimate). Otherwise, and for
the final covariance of such fits, Householder QR of sqrt(W)X (LAPACK dgeqrf)
is used, whose error grows with cond(sqrt(W)X) rather than its square; its
triangular solve and (R'R)^-1 are in-house (LAPACK `dtrtrs` and `dpotri` with
the option). The
coefficient-change criterion is floored at 100 eps cond(sqrt(W)X), the
resolution of the step itself. Designs with a poorly conditioned or singular
prior-weighted X'X are checked for aliasing by that QR: a column whose weighted
norm after projection on the retained preceding columns falls below 1e-11 of
its own norm is aliased. This is the limited pivoting documented for R's
[qr](https://stat.ethz.ch/R-manual/R-patched/library/base/html/qr.html), and
1e-11 is the threshold R's glm shows at its default epsilon (measured, not
documented). By default an aliased design is a `rank_deficient` failure; with
`singular_ok` the retained columns are fitted and aliased coefficients are NaN
(R's NA), with LR degrees of freedom counting retained columns. Boundary
responses, separation candidates, invalid input and iteration limits are
explicit statuses. Complete or
quasi separation is conservatively flagged when unpenalized binomial iterations
reach boundary predictions and diverging coefficients; this is not an exhaustive
linear-programming separation detector. Invalid fits do not produce valid tests.
`converged_boundary` reports a stabilized count likelihood with flagged
coefficients. `FitResult::ok()` accepts this status so NB initialization can
proceed; it does not establish validity of every parameter's inference.
Sparse input is CSC with observations in rows and features in columns (transpose
of the biological feature-by-cell convention). Only one response per worker is
dense. Compact batch outputs omit observation-sized fit state.

## Numerical backend: special functions

`include/rapidglm/vmath.hpp` (1.5.0) gives log, log1p, exp, lgamma, digamma
and trigamma of double, over arrays and one value at a time. It needs only
the C++17 standard library and can be copied into other projects on its own.
The default slow policy uses RapidGLM's log, log1p, exp, digamma and
trigamma kernels; lgamma uses the C library's `lgamma_r`. The fast policy
selects different log1p, positive-argument lgamma and array digamma kernels,
as described under Division policy. Negative-argument lgamma always uses
`lgamma_r` (`std::lgamma` writes the global `signgam`, which threaded
callers must not do).

The following algorithms describe the slow policy. Its own functions use no
division except in digamma and trigamma:

- **log.** x = 2^k m with m in [sqrt(2)/2, sqrt(2)), taken from the bits, and
  g = m - 1 (exact). Then log x = k log 2 + g - g^2/2 + g^3 R(g), where:
  - log 2 is split in two parts, so that k log2_hi is exact;
  - k log2_hi + g, and that sum plus -g^2/2, are split exactly (Fast2Sum);
  - the rounding error of g^2 is kept;
  - R has degree 19 and is evaluated by Estrin's scheme;
  - everything but the leading part is added before the last rounding.

  Subnormal x are scaled by 2^54 first.
- **log1p.** u = 1 + x is rounded, and c = x - (u - 1), its rounding error, is
  computed exactly (with the larger of x and 1 first). Then
  log1p(x) = log u + c/u, evaluated as for log with:
  - g + r = (m - 1) + c 2^-k split exactly (g = x when k = 0);
  - r/u formed as r 2^-k (1 - g + g^2), with no division.

  For |x| < 2^-60 the result is x, which is correctly rounded there.
- **exp.** exp(x) = 2^k exp(r), with k = round(x / log 2) and r = x - k log 2
  (log 2 in two parts). exp(r) is the Taylor polynomial of degree 13, and 2^k
  is added to the exponent bits. For |x| > 708 the scaling takes two steps,
  for overflow and gradual underflow. This is the exp of `kernels.hpp`, moved
  here: `detail::exp_array` now calls `vmath::exp_array`, with the same bits
  on [-700, 700].
- **digamma.**
  - On z in [0.5, 2.5): psi(z) = (z - x0) P(v)/D(v), with v = z - 0.5, x0 the
    positive root in two parts, P of degree 16 and D(v) = z (z + 1)(z + 2).
  - Below 0.5: psi(x) = psi(x + 1) - 1/x. From 2.5 to 5: psi(x - n) plus the
    sum of 1/(x - j). Both are free of cancellation.
  - From 5: log x - 1/(2x) + G(1/x^2)/x^2, with G of degree 9 and RapidGLM's
    log.
- **trigamma.** psi'(x) = sum over j < m of 1/(x + j)^2 + psi'(x + m), with
  m = ceil(6 - x) and at most 6. Then psi'(y) = 1/y + 1/(2y^2) + H(1/y^2)/y^3,
  with H of degree 8.
- **Polynomials.** They are Chebyshev interpolants computed with mpmath at 60
  digits, converted to monomials and rounded to double
  (`scripts/fit_vmath_polynomials.py`, which reproduces them).
- **Subnormal intermediates.** For 1/x above 2^500, digamma and trigamma clamp
  or drop their 1/x^2 terms, far below half an ulp, because subnormal
  arithmetic costs over 100 ns an element.
- **Domains.** digamma and trigamma are defined for x >= 0; negative x gives
  NaN.
- **NaN.** RapidGLM's functions return a NaN argument unchanged, keeping R's
  NA.

**Bit identity.** RapidGLM's functions give the same bits on every processor
and path:
- AVX2 with FMA on x86-64 (dispatched at run time);
- Advanced SIMD on AArch64;
- the portable path (scalar, `std::fma`);
- the one-value forms (the scalar code compiled for FMA on AVX2 hosts).

They use only IEEE operations, fused multiply-adds and integer operations on
the bits, in one order. A rounded product never feeds an addition, so
floating-point contraction cannot change the result. Vector lanes outside a
function's main domain take the scalar code. `tests/test_vmath.cpp` checks:
- the portable and dispatched paths and the one-value form against each
  other, on 256,000 arguments;
- digests of all outputs recorded on x86-64 AVX2, which the portable build
  and AArch64 under QEMU reproduce;
- 1,350 mpmath reference values;
- special values.

Slow-mode lgamma and negative-argument fast-mode lgamma follow the platform's
libm. Positive-argument fast-mode lgamma has its own cross-path digest. Fast
log1p, lgamma and digamma use AVX2 where available and the portable scalar
kernels on AArch64; unchanged functions retain their Advanced SIMD paths.
On an x86-64 processor without FMA, the
portable path calls the C library's `fma`, which is correctly rounded but
slow (`RAPIDGLM_FORCE_PORTABLE` builds take this path only).

**Accuracy of the slow policy.** Largest error in ulps of the correctly rounded value, for 10^6
random arguments per range, against long double references (glibc's `logl`,
`log1pl`, `expl`, `lgammal`; Boost's long double digamma and trigamma):

| Function | (1e-300, 1e-5) | (1e-5, 3) | (3, 100) | (100, 1e300) | NB | Other |
|---|---|---|---|---|---|---|
| log | 0.06 | 0.66 | 0.53 | 0.12 | 0.61 | |
| log1p | 0.50 | 0.68 | 0.52 | 0.50 | | (-1, 0): 0.65 |
| exp | | | | | | (-1, 1) and (-700, 700): 0.85 |
| lgamma (glibc) | 0.55 | 1.78 | 2.22 | 1.07 | 2.25 | |
| digamma | 0.99 | 3.74 | 2.28 | 0.61 | 3.15 | |
| trigamma | 1.90 | 2.48 | 2.24 | 0.99 | 2.40 | |

For comparison:
- glibc: log 0.52, log1p 0.81, exp 0.51.
- Boost's double-precision digamma and trigamma: up to 3.25 and 4.21.
- Boost with promotion to long double: up to 0.70, at 2 to 20 times the cost.

`tests/test_vmath.cpp` also checks 1,350 points against mpmath, including the
neighbourhoods of lgamma's zeros and digamma's root. Largest errors there:
log 0.51, log1p 0.49, exp 0.62, digamma 1.9, trigamma 1.3 ulp.

The NB arguments are y + 1/alpha, with y geometric (mean 20) and alpha
log-uniform from 1e-8 to 1e3.

### Division policy

`RAPIDGLM_DIVISION=slow|fast` selects the special-function policy, with `slow`
as the default. The environment is read once on first use. CMake can set a build
fallback with `-DRAPIDGLM_DIVISION_DEFAULT=fast`. Explicit call options override
the environment and restore the previous policy on return or exception. Worker
threads inherit the calling thread's resolved policy. There is no CPU-model
autodetection; callers select the policy suited to their processor.

The fast policy is measured on one Zen 3 core of the Threadripper PRO 5965WX,
under an exclusive Slurm allocation with the core's SMT sibling idle. Log and
exp retain the division-free array kernels. Log1p uses the earlier atanh-series
kernel with division; positive lgamma uses the earlier polynomial kernel; array
digamma uses the Boost-derived rational algorithm with one division per recurrence
step. The combined recurrence was measured and rejected for its error. Array
trigamma retains the existing kernel: the faster Boost-derived array candidate
exceeded its accuracy bound. Negative lgamma arguments use the platform's libm.
Within each policy, the own kernels use the same operation order on AVX2,
portable and AArch64 paths; separate known-answer fingerprints test both modes.
The old slow-policy fingerprints remain unchanged.

For individual engine calls, the fast policy keeps glibc log, log1p, exp and
lgamma_r, and Boost's double-policy digamma and trigamma. The Boost-derived FMA
trigamma candidate has lower dependent-call latency, but made edgeR QL 1.5%
slower in the paired end-to-end check and gave no bulk-DE improvement. The
engine therefore retains Boost double. The rejected candidate and its
accuracy and timing measurements are retained in DIVISION_CHOICES.json.

### Choice of backends

The slow-division choices below were made on a Gracemont efficiency core
(CPU 16) on pikachu. These are historical development
measurements; the 1.5.0 receipts run re-measures this stage on Zen 3. Gracemont
cores issue about one 256-bit
division per 32 cycles. Times are ns per element on 4,096 arguments in
cache, for the NB arguments unless stated (`validation/VMATH.json`, "choices";
`validation/SPECIAL_FUNCTIONS.json`).

| Function | Engine, one value at a time | Engine, arrays | vmath arrays (C, Python, R) | Measured and not chosen |
|---|---|---|---|---|
| log | glibc, 4.7 | RapidGLM in edgeR-QL's unit deviances; elsewhere not batched | RapidGLM, 3.3; 0.67 ulp | SLEEF 6.8 (divides); RapidGLM one at a time 9.6; arrays in the NB likelihood, deviance and theta score (slower end to end) |
| log1p | glibc, 10.1-11.6 (and 1.4.1's log plus a division) | RapidGLM in edgeR-QL's unit deviances | RapidGLM, 4.6; 0.68 ulp | SLEEF 7.9; RapidGLM one at a time 12.7 |
| exp | glibc, 4.9-7.9 | RapidGLM: log-link means, floored NB fits, edgeR fixed-dispersion fits | RapidGLM, 2.6; 0.86 ulp | SLEEF 4.4; glibc one at a time 7.9 |
| lgamma | glibc `lgamma_r`, 11.5 | none | glibc `lgamma_r`, 11.5; 2.25 ulp | SLEEF 41.4; Boost double 32.9; Boost promoted 230 |
| digamma | Boost double policy, 5.6-10.0 per call | none | RapidGLM, 10.0; 3.7 ulp | Boost double as an array 11.2; Boost's algorithm in AVX2 13.3; first RapidGLM version 12.3; RapidGLM one at a time 11.9-13.8 per call |
| trigamma | Boost double policy, 5.2-7.2 per call | none | RapidGLM, 6.6; 2.7 ulp | Boost double as an array 7.5; Boost's algorithm in AVX2 7.6; first version 8.4; RapidGLM one at a time 6.6-11.7 per call |

Times are ns per element (per call where stated). Notes:
- On arguments below 3, Boost's scalar digamma and trigamma are faster: 7.3
  and 6.8 ns against RapidGLM's 10.7 and 14.0.
- On a performance core (2.2 GHz, untimed), RapidGLM's arrays take 2.6 (log),
  3.2 (log1p) and 1.5 ns (exp) against glibc's 5.2, 11.4 and 9.5. There,
  Boost's trigamma algorithm in AVX2 is faster than RapidGLM's (3.1 against
  3.6 ns).
- RapidGLM's portable path, without FMA instructions, takes 32-75 ns.

SLEEF 3.9.0 (u10, AVX2), the first backend tried, made the count paths slower
on these cores because its double-double log divides. Ratios against 1.4.1,
one thread:

| workload | ratio |
|---|---|
| bulk DE, GTEx thyroid | 0.883 |
| NB batch fit | 0.882 |
| edgeR-QL, thyroid | 0.978 |
| bulk DE, Replogle AAAS | 1.000 |

On a performance core (capped at 2.2 GHz, untimed) the same build was faster,
by 2-7% on the NB batch, bulk DE and edgeR-QL. It was removed.

**Call sites.**
- One value at a time, the engine keeps glibc's log, log1p and exp, as 1.4.1
  did. RapidGLM's are faster only as arrays.
- The log-link means, the floored NB fits of bulk DE and edgeR's
  fixed-dispersion fits take their exponentials as one array.
- edgeR's unit deviances take their logarithms as arrays (two gathers).
- The other count paths take their logarithms one at a time again, because
  the arrays did not pay there:
  - NB likelihood, count deviance and theta score: 2.991 against 2.893 s on
    the NB batch fit;
  - bulk DE's floored deviance: 34.89 against 34.66 s on thyroid, 35.69
    against 35.85 s on Replogle AAAS. This is within the noise below.
- digamma and trigamma at the engine's call sites (theta Newton steps,
  edgeR-QL's squeezed variances, bulk DE's prior variance) are boost::math's
  double-precision algorithms, without promotion to long double. Per call
  they are faster: 5.2-9.9 against 12.7-14.2 ns for digamma with one argument
  repeated. In the workloads the two choices differed by -1.7% to +0.4%.
- Rebuilding with a change in code layout only moved bulk DE on thyroid by
  1.5% on these cores, so differences of that size are not attributed to the
  functions.

## Tests and effects

For an added covariate g, let c=X'Wg and H=X'WX. The efficient score is
U=g's-c'H^-1 X's, with information I=g'Wg-c'H^-1 c. Report U/sqrt(I phi)
and its normal tail. A reusable workspace stores the null weights, score and
inverse information. Wald uses beta_j/SE_j; Gaussian uses Student t with
residual degrees of freedom, others normal. LR compares twice the maximized
log likelihood difference with chi-square, for explicitly nested models on
identical observations/weights. NB theta is reestimated in both nested models.
HC0 sandwich covariance is H^-1 [sum x_i x_i' s_i^2] H^-1.
Wald/HC0 tests reject flagged boundary coefficients and retain inference for
unflagged finite coefficients. Ordinary LR and score tests conservatively reject
any fit with coefficient-boundary flags; their usual reference distributions
are not qualified here for boundary models. The perturbation workflow retains
such nulls as `null_converged_boundary` with missing tests and adjusted p=1.
Sources: [Rao 1948](https://doi.org/10.1017/S0305004100023987),
[Wilks 1938](https://doi.org/10.1214/aoms/1177732360),
[White 1980](https://doi.org/10.2307/1912934).

Linear hypotheses Cβ = 0 (k rows, full row rank, k < p) generalize the single
coefficient:
- **Reduced model.** When the rows name coefficients, the reduced design drops
  those columns. Otherwise it is X N, with N an orthonormal basis of the null
  space of C from its SVD (LAPACK `dgesvd`). Any basis gives the same model.
- **LR and QL tests** compare the full and reduced fits on k degrees of freedom:
  χ²_k, or F(k, d0 + residual df).
- **Wald tests** take one row c. The full model is fitted in a reparametrized
  design whose column k (the largest |c_k|) is x_k / c_k and whose other
  columns are x_j − (c_j / c_k) x_k. Its coefficient k is c'β, with the usual
  Wald test and standard error. The design spans the same model, and a contrast
  of diverging coefficients that is itself finite (for example two cell lines
  when the reference line has only zeros) keeps a finite test. For a single
  coefficient the design is X itself.

For log links, adjusted log2 fold change = beta_g/log(2),
SE = SE(beta_g)/log(2). A binomial coefficient is a log odds ratio, not a fold
change. Gaussian coefficients are mean differences. The perturbation workflow
also reports observed treatment mean / fitted-null treatment mean as
`sceptre_mean_ratio` (undefined if denominator is zero), and its log2 transform
separately. Its compatibility with 0.99.0 must be measured; it is not named a
log fold change. BH sorts p and takes reverse cumulative minima of m*p_(i)/i;
Bonferroni is min(1,m*p).
Source: [Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

**The adjustment family** of the perturbation workflow (`adjustment_family`;
`SceptreOptions::family` in C++), within each kind of pair (discovery, negative
control, positive control):
- `tested`, the default from 1.3.0: the pairs that pass pair QC. A pair that
  fails pair QC has no adjusted p-value. A pair that passes and whose test then
  fails (a null model without a regular fit, say) stays in the family with
  p = 1.
- `listed`, the default before 1.3.0: every listed pair, those that fail pair QC
  counting with p = 1. The adjusted p-values then depend on how many untestable
  pairs the list happens to hold.
- `family_size` gives the number of hypotheses of a kind when it is larger than
  the pairs at hand, for a caller that lists only some of them; the others
  count with p = 1, exactly as if they were listed.

SCEPTRE's documentation says that pairs below its nonzero-cell thresholds "are
excluded" and have missing p-values and significance calls, and that the
resulting p-values "are subjected to a multiplicity adjustment"; it does not
name the family in so many words. Its outputs do: on the Gasperini benchmark
panel (10,000 discovery pairs listed, 9,718 passing pairwise QC) SCEPTRE
0.99.0's 101 calls are BH at 0.1 over the 9,718, not over the 10,000 (97
calls). `tested` is that family. Sources:
[pairwise QC](https://timothy-barry.github.io/sceptre-book/run-qc.html),
[multiple testing](https://timothy-barry.github.io/sceptre-book/set-analysis-parameters.html).

By default only pairs rejected at `alpha` in their testing family (after
guide integration) receive this full effect fit; the others report the
mean-ratio effect only. The fit, when run, is unchanged by this selection.
For regular effect fits, the workflow uses the cached null coefficients plus
a zero perturbation coefficient as the starting point. Under NB ML it also
uses the null theta and skips the otherwise repeated Poisson/moment pilot.
It then performs the same full optimization and covariance calculation; theta
is not held fixed. Moment-dispersion fits retain their established pilot.
Failed or boundary warm starts revert to the fresh-fit path. Independent
pairs are OpenMP tasks within the existing team, joined before integrating
guides or adjusting p-values. Neither change modifies the resampling test.

## Resampling

Permutation uniformly shuffles labels within each supplied stratum, preserving
its treatment count. CRT draws independent Bernoulli(p_i) labels using a fitted
logistic propensity or caller-supplied probabilities. Three samplers draw
this law, each from its own random stream: `hazard` (the default, below),
`thinned` (the default of releases before 1.5.0) and `direct`. The `thinned`
draw groups observations by q=2^-b with q/2<p_i<=q, visits Bernoulli(q)
candidates through geometric gaps and accepts each with probability p_i/q;
levels with q>=1/4 use one uniform per observation. The law is unchanged and
the expected work is O(sum p_i) rather than O(n). When the mean p_i is at least
1/16, and for `crt_draw=direct`, one uniform is compared with each p_i in
observation order, reproducing earlier streams. Thinning by acceptance is the
standard construction of [Lewis & Shedler 1979](https://doi.org/10.1002/nav.3800260304). A sampler callback permits
other documented conditional laws; no lineage sampler is copied or qualified.

`crt_draw=hazard`, the default, draws the same law without rejection.
Observation i is given an interval of length -log(1 - p_i) on a line, the
intervals laid end to end in observation order. The points of a unit-rate
Poisson process on the line are generated by adding standard exponential gaps,
and an observation is selected
when at least one point falls in its interval. An interval of length L holds
no point with probability exp(-L) = 1 - p_i, and disjoint intervals are
independent, so the labels are independent Bernoulli(p_i). The expected number
of points is the line's length, sum -log(1 - p_i), which for small p_i is the
expected number of treated cells: one gap per treated cell instead of about
1.4 candidates with two uniforms, a logarithm and a division each, and the
selection comes out in order, with no levels to merge. Gaps come from the
256-layer exponential ziggurat of [Marsaglia & Tsang 2000](https://doi.org/10.18637/jss.v005.i08): one 64-bit
word gives the layer (8 bits) and the position (53 other bits), and about one
gap in a hundred needs more. The words come from ARS, the AES-based
counter generator of [Salmon et al. 2011](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),
with 10 rounds: block b of a draw is ARS of the counter (b, 0) under a 128-bit
key, the first two words of the draw's Philox stream, so a draw is still a
function of seed, test and draw alone and no state is shared between threads.
ARS is implemented from the published algorithm and reproduces the
known-answer vectors distributed with Random123. With the processor's AES
instructions (AES-NI on x86, the Arm AES instructions) a word costs about a
third of a Philox word; AES is exact integer arithmetic, so both give the
words of the table-lookup implementation the tests compare them with
(`scripts/test_arm64.sh` runs those tests for arm64 under QEMU). A processor
without the instructions would spend more on the lookups than the method
saves, so `hazard` then draws with the thinned sampler, from thinned's stream.
CRT p-values therefore depend on the kind of processor; the run record and the
C, Python and R results state the sampler used (`crt_draw_used`). Builds with
`RAPIDGLM_FORCE_PORTABLE` have no AES path and take the fallback too, so the
tests cover both paths on x86. A point's observation is found from a table with
one entry per observation over equal stretches of the line, then by stepping
along the intervals' ends (8 + 4 bytes per observation, as for `thinned`).
Lengths are added in double precision, so each probability is exact to within
the rounding of the line's length (2e-13 for a line of 1,000); p_i = 1 is given
length 40, missed with probability 4e-18. A line longer than a quarter of the
number of observations, and a mean p_i of at least 1/16, use the direct
stream. Batches cross the line stretch by stretch with every draw keeping its
own generator, so batched draws equal one-at-a-time draws.
Thinned CRT draws are generated and scored in batches of up to 256 draws.
Each level's members and their probabilities are walked once per batch, in
blocks of 4,096 members, with every draw advancing through a block before the
next block is read; each draw keeps its own generator and consumes it in the
same order as alone. The batch is then scored by walking the per-cell records
(residual, weight and weighted design row, stored together) in blocks of
2,048 cells, each draw adding its records in ascending cell order. Every draw's
labels and statistic are therefore bit-for-bit those of one-at-a-time drawing,
which `tests/test_batching.cpp` checks against the earlier implementation; the
batching removes most scattered memory reads, which limited many-thread runs.
With several threads the perturbation workflow keeps a ring of active pairs
(eight at 32 threads), response by response, and every thread takes chunks of
32 draws from them, so the response's score records and the pairs' samplers
are shared in cache instead of each thread streaming its own. The thread that
completes a stage applies the staging rule; idle threads fit upcoming
responses' null models and start the next pairs. With complement controls a
target's CRT sampler depends only on the target and is built once for all its
pairs, within `sampler_cache_mb`. Philox blocks are computed eight at a time,
with AVX2 when available. Draw k of a pair always uses its own key and is
written to slot k, so none of this changes any result.

**Batch scoring** (`batch_scoring`, the default from 1.6.0 for CRT with
complement controls). Instead of the ring of active pairs, each thread takes
one response and scores all its pairs together, stage by stage. In each round
every active pair contributes its next draws: a target's stored stage with
`reuse_draws`, otherwise up to 8,192 draws from the pair's own stream,
generated in batches of 256 as above. All of them are scored in one pass over
the response's score records, a sparse × dense product S = D R with D the
draws' treated cells and R the records (residual, weight, weighted design
row). The pass visits the cells in blocks of 32,768, 2 MB of records with six
design columns, so that a block's records stay in cache while every draw adds
those of its cells in the block, two draws at a time with AVX2 vectors where
available. A draw's sums are thus added in the order of pair-by-pair scoring,
and statistics, p-values and draws are bit for bit those of the pair pipeline
on every processor path and at any thread count. D is too sparse (about 0.4%
of the cells per draw on the Gasperini screen) for dense matrix products.
Before the responses, every listed target's propensity model and sampler are
prepared in parallel, and with `reuse_draws` its first-stage draws are
stored; a stage that another thread is storing is waited for, and stored or
generated draws are checked once (ascending, unique, in range) instead of at
every cell. On 100 targets × 7,418 genes of the Gasperini screen (8 threads,
Xeon E5-2640 v4) scoring took 2.3 to 3.0 times less thread time; a run with
`reuse_draws` and `confirm_draws: 0` took 1.92 times less wall time and a
default run, where drawing the labels takes most of the time, 1.22 times less
([BATCH_SCORING.json](validation/BATCH_SCORING.json)). The alternatives
measured there (block sizes, one, two or four draws at a time, prefetching, a
cell-major layout of stored draws, and response nulls and propensity models
fitted in lockstep as matrix products) and the reasons for the choice are
recorded with it. Shared screens, sequential screens and early stopping keep
the pair pipeline.

Optionally (`reuse_draws`, high MOI only), the CRT draws of a target are shared
by all its tests: the random streams are keyed by the target instead of the
pair, and each stage's draws are stored once per target (within
`draw_cache_mb`) and scored against every gene. Each test is still a CRT with
the same law and staging; only the resamples are common to the target's genes,
so their errors are correlated. It is meant for routine discovery; reported
statistics use the default's independent draws.

Optionally (`shared_screen`, high MOI only), only the first-stage screen is
shared. Each test is first screened with its target's first-stage draws, from
a stream keyed by the target and apart from the pairs' own; a test whose
empirical p-value there exceeds `shared_screen_p` ends with it. Every other test is then staged as in the default from its own stream,
starting again at draw 0, so its result is the default's. The shared screen can
end a test that the default would have continued; with `shared_screen_p` above
the default screen this is rare, and a test whose p-value is small passes with
near certainty. The engine exposes this as `ResampleOptions::prescreen` and
`prescreen_test_id`, for any tests that share a sampler.

Optionally (`sequential_screen`), a screen ends as soon as its outcome is
certain. With B draws in the first budget and threshold s, let h be the smallest
count of draws at least as extreme as the statistic (on the deciding side: the
smaller of the two sides for two-sided tests) at which the screen's empirical
p-value exceeds s. Every 32 draws, and at B, the screen ends once that count has
reached h, because further draws cannot lower it. A test that continues has
drawn all B, so it continues exactly when the full screen does and its result is
unchanged. A test that ends reports the Besag-Clifford sequential p-value
([Besag & Clifford 1991](https://doi.org/10.1093/biomet/78.2.301)) h/l, where l
is the draw at which the count reached h (two-sided: twice that of the side that
reached h last, at most 1), raised to at least the smallest p-value of a test the
full screen ends; both steps keep it valid. The p-value depends only on the
draws, not on when the check ran, so results do not depend on scheduling.

Optionally (`early_stop_p`), the budgets after the screen end early in the same
way. For a budget of B draws and threshold e, h is the smallest count at which
the empirical p-value over B draws exceeds e, and not less than the count of
the stage before. Counts run over all the test's draws from its stream, those
of the earlier budgets included. Every 32 draws, and at B, the stage ends once
the count on the deciding side has reached h, and the test reports the
Besag-Clifford p-value h/l as above, raised to at least the empirical p-value
of count h over B draws, which is above e. A test whose count stays below h,
as it does whenever its empirical p-value over the B draws is at most e, draws
all B and is then treated exactly as without the option (the skew-normal fit,
or the next budget), so its result is unchanged. The stopped tests' p-values are
valid: with counts h1 <= h2 <= ... for the budgets B1 < B2 < ..., a reported
p-value of at most a needs fewer than hk draws as extreme as the statistic
among the first ceil(hk/a) - 1, for the first stage k that can report so small
a value, and under the null that has probability at most a (two-sided: a/2 for
each side). They are coarse, with a relative standard error near 1/sqrt(h): h
is 5 for e = 0.002 with 4,999 draws.

A stage that may end early is drawn one chunk at a time, each chunk ending at
the next check, so that no draw is generated beyond the check that ends it;
with such stages the workflow keeps about one active pair per thread instead
of the small ring, and threads without such a chunk of their own draw for the
pairs in the other stages. `tests/test_batching.cpp` checks that the draws
generated equal the draws the results use, at several team sizes, and works
out every stopped test's p-value and draws independently from the default's
draws.

A response's state is allocated by the thread that prepares it and freed by the
thread that ends its last pair. With glibc such blocks stay in the allocating
thread's arena, so the workflow returns free pages to the system every 256
responses; more often costs time, because each return interrupts every thread.
Philox4x32-10 is implemented from the published algorithm; seed is the 64-bit
key, test ID and draw ID partition the 128-bit counter with a per-draw block
index. Limits are checked. No thread ID contributes to streams. Floating point
reductions within a test have a fixed order.

**Stream keys of the perturbation workflow** (`stream_keys`). A stream is a
64-bit seed and a 32-bit test ID; draw k of a test uses the stream's draw k.
- `identity`, the default from 1.3.0: a pair's stream depends only on the
  run's seed, the response and the target. Its record is therefore the same
  wherever the pair is listed, whatever else is listed, and however the run is
  divided among threads, tiles, shards or calls.
  - *Keys.* The response's key is the 64-bit FNV-1a hash of its ID
    (`response_ids`: one string per response, such as a gene or peak name), or
    its column index when no IDs are given; the index identifies a response
    only while the matrix keeps its columns. The target's key is the FNV-1a
    hash of its name (of the guide's ID for a per-guide test). Names in one run
    must have distinct keys, which is checked.
  - *Stream.* Philox keyed by the run's seed turns a domain number into a
    64-bit key (counter words: the domain, 0, 2^32 - 1, 2^32 - 1); Philox with
    that key turns the two keys (counter words: the response key, then the
    target key, low words first) into the stream: its seed is the first two
    output words and its test ID the third. Domains: 1 a pair, 2 a target whose
    draws are reused, 3 a target's shared screen; 4 to 6 the same for per-guide
    tests. Two of N pairs share a stream with probability about N²/2⁹⁷.
  - *Duplicates.* A pair listed twice within a kind would be one test counted
    twice, and is an error. A pair listed under two kinds (a positive control
    that is also a discovery pair) is one test and has one p-value.
- `position`, the default before 1.3.0: the test ID is the pair's place in the
  list (the target's order of first appearance for reused draws and the shared
  screen), under the run's seed. It reproduces earlier releases; a pair's
  p-value then changes when the list is reordered or other pairs are added or
  removed. Each p-value is valid under either setting.

Sources:
[Salmon et al. 2011](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),
[Barry et al. 2021](https://doi.org/10.1186/s13059-021-02545-2).

The perturbation workflow schedules chunks of 16 draws as OpenMP tasks within
its existing worker team. Each draw has private sampler scratch space and
writes to its own indexed score slot. All tasks finish before the staged tail
calculation, which consumes scores in draw order. Response-batch count does
not limit draw parallelism. RNG streams, budgets, score accumulation and tail
calculations are unchanged by scheduling. Without OpenMP 4.5, or outside an
active team, draws run serially. Low-level callers must explicitly opt in and
supply a sampler safe for concurrent calls.

Empirical tails include ties and use (1+extreme)/(B+1). Two-sided is twice the
smaller one-sided tail, capped at one. Fixed B is the calibrated empirical
mode. Staged budgets are cumulative, default 499,4999,19999. In approximate
mode a moment-fitted skew normal is accepted only with finite admissible
moments and an empirical CDF discrepancy <=1.36/sqrt(B). Otherwise draw the
next stage. The discrepancy is a diagnostic, not a valid fitted-parameter KS
p-value. At the final stage fallback is empirical.

That discrepancy is taken over the whole distribution and says little about
the far tail, where the p-value is read. With `fit_check=strict` (the default
since 1.0.0; `ks` restores the acceptance above alone) a fit must also be sound
in the tail on the statistic's side, the smaller fitted tail for a two-sided
test. Each comparison is made at the 1% level:
- the draws at least as extreme as the statistic are not more than a Poisson
  count with the fit's expectation allows;
- the fit expects at least 1/100 of a draw beyond the most extreme draw;
- the draws beyond the fitted quantiles that have 1, 4, 16, ... expected draws
  beyond them, down to a tail of 1/64, are within the Poisson limits (5, 10,
  27, 84, ...);
- the sample's excess kurtosis exceeds the fit's by no more than 3 sqrt(24/B).

A p-value that the budget cannot check, one for which the fit expects fewer
than half a draw beyond the statistic, is in addition accepted only on the
last budget, where the fit is made and checked again on all the draws. Since
1.1.0 it is not accepted there either: the test draws on to `confirm_draws` in
all, by default ten times the last budget (199,999), and the fit made on those
draws is reported if it is sound, the empirical p-value otherwise. Tests whose
fit is rejected on the last budget do not draw the confirmation;
`confirm_draws=0` turns it off. The
checks look for a tail heavier than the fit: a lighter one makes the p-value
conservative.

On 100 targets x 7,418 genes of the Gasperini screen the tail was counted over
1 to 4 million draws for each of the 25 pairs that had been a discovery and
for one more ([SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)).
Seven counts contradict the p-value of the KS-accepted fit, which is 8 to
several thousand times too small; all seven pairs have positive statistics,
where the null statistics have a heavy right tail, and SCEPTRE 0.99.0 reports
five of them as discoveries. The strict check accepts five of the seven in at
most 2% of repeats and the two mildest (8 and 40 times too small) in 42-44%;
of the other 19 pairs it accepts 17 in at least 96% of repeats, one in 78% and
one never. Draws rise by 5%. A rejected pair reports the empirical p-value of
the last budget, at least 1e-4 with 19,999 draws, so it cannot be a discovery
at a genome-wide cut-off: recovering true discoveries with heavy-tailed nulls
needs more draws for those pairs, which is not implemented.

An optional empirical screen (`screen`, default 0.02 for the perturbation
workflow and 0 for scans) reproduces the staging that SCEPTRE 0.99.0's outputs
show:
- The first budget always ends with its empirical p-value, which is final
  unless it is at most the screen.
- Only then are the later budgets drawn, with the approximation as above.

The observed staging (*calibrated* from black-box outputs):
- first-stage p-values are multiples of 1/500 (B1 = 499);
- the smallest seen at stage 1 is 0.024;
- 31 of 1,500 null pairs (2.1%) went on to the second stage.

Both stages estimate the same tail probability, and the first-stage p-value is
exact, so the screen does not act as a significance-based stopping rule on the
reported p-value. Screened resampling meets the scan's null-calibration bounds
([SCEPTRE_STAGING.json](validation/SCEPTRE_STAGING.json)). No other
significance-based early stopping is implemented: naive optional stopping
changes the null law. Empirical mode always uses the final budget. The skew-normal moment fit
and CDF follow [Azzalini's distribution formulas](https://www.azzalini.it/SN/Intro/intro.html)
and Boost.Math. Approximate tails and staged fit acceptance require calibration;
no saddlepoint default. [Phipson & Smyth 2010](https://doi.org/10.2202/1544-6115.1585)
motivates nonzero Monte Carlo p-values.

## Perturbation workflow

Inputs: sparse responses, fixed binary guide calls, guide-to-target mapping,
explicit design/offset, cell QC measurements, and separate discovery/positive/
negative pairs. Guide calling is upstream. Low MOI removes zero/multiple-guide
cells. QC defaults clip UMI/detected-feature percentiles at 1/99 and mitochondrial
fraction at 0.2; at least seven response-nonzero cells per treatment/control.
Union pools guides; singleton tests each; Bonferroni tests each then returns
min(1,k*min(p)), retaining failed guides in k. NT controls exclude treatment
cells. High MOI requires complement controls. Negative-control targets must be
explicitly identified, never inferred from no-call or WT labels. BH is applied
separately to the discovery, power and calibration families (Adjustment family
above); QQ output includes pointwise Beta order-statistic 95% bands (not
simultaneous bands).

Pair QC is decided first, from the target's treated cells and the response's
nonzero cells alone. A pair that fails it costs nothing more: no cells are
gathered, and with complement controls a response's null model is fitted only
when one of its pairs passes.

A request can name targets and responses instead of listing pairs
(`sceptre_grid`; `targets` in the CLI): every discovery target and every
negative-control target against every named response. One pass over each
response's nonzero cells counts them for all targets, the pairs that pass pair
QC are tested, and the others are only tallied, by target and kind, as having
too few treated cells with a nonzero count or too few control cells with one.
The records are those of listing the same pairs (identity stream keys), and
the `listed` family counts the untested pairs without holding them.
Sources: [SCEPTRE QC documentation](https://katsevich-lab.github.io/sceptre/reference/run_qc.html),
[analysis parameters](https://timothy-barry.github.io/sceptre-book/set-analysis-parameters.html).

Response nulls are fitted on all QC-passing cells for complement tests, reused
per response. With NT controls the null is fitted on treatment plus NT cells,
separately for every pair because the subset depends on the tested guides.
This subset policy is a RapidGLM choice pending black-box qualification
against pinned 0.99.0. The current book is not version-pinned and does not
completely specify its dispersion algorithm or staged acceptance rules. The 2024 paper motivates
NB score permutations and moment-fitted skew-normal tails:
[Barry et al. 2024](https://doi.org/10.1186/s13059-024-03254-2).

The default `null_model=sceptre` follows SCEPTRE's documented response model:
it fits Poisson coefficients and holds their fitted means fixed while
maximizing conditional NB likelihood over theta. `null_model=direct` instead
uses the configured GLM (joint NB ML by default) and is required for other
families, fixed or moment dispersion and ridge fits. It uses the existing conditional-theta optimizer, not joint NB
coefficient updates. SCEPTRE's public methods chapter describes the two-stage
recipe as an acceleration; its exact historical iteration rules are not part
of this mode's contract. The native recipe is checked against public R GLM/
theta APIs and the pinned SCEPTRE 0.99.0 black-box scores.

With residuals `s_i=(y_i-mu_i)/(1+mu_i/theta)` and weights
`w_i=mu_i/(1+mu_i/theta)`, the compatibility score numerator is `g' s` and its
information is `g' W g - (X' W g)' (X' W X)^-1 (X' W g)`. Supplied prior weights
multiply both residuals and working weights. The direct mode retains its
additional nuisance-score projection in the numerator. That term need not
vanish for Poisson coefficients under an NB working model, so the compatibility
flag selects the numerator convention as well as the fit.
Source: [SCEPTRE methods, equation 10.12 and footnote 1](https://timothy-barry.github.io/sceptre-book/methods.html#computing-the-test-statistics).

Compatibility fits are tagged separately from direct fits, supply no NB
coefficient covariance/SEs, and are rejected by ordinary Wald/LR/normal-tail
score inference. Resampling supplies their p-values. Effect-model estimates
still use direct GLM fitting. Tail approximation, staging, RNG and draw budgets
are unchanged by this model flag. Until 2026-09-29 the default was
`null_model=direct`; its earlier receipts remain valid for that setting.

### Mergeable null start

With NT controls, each pair's null is fitted on its own cells: the NT cells and
the target's treated cells (above). From 1.6.0 that fit starts by default from
the response's **reference** (`null_start=refine`), not from scratch
(`null_start=cold`, the behaviour of earlier releases).

1. **Reference.** The response's null is fitted once on all retained cells:
   Poisson coefficients beta_ref, then theta_ref by conditional ML, exactly as
   for complement controls. Every per-cell term below is evaluated at
   (beta_ref, theta_ref) and the fitted means mu_i of that fit.
2. **Totals.** For a set of cells the totals are the Poisson score
   sum a_i (y_i - mu_i) x_i, the Poisson information sum a_i mu_i x_i x_i', the
   conditional theta score and curvature terms of `estimate_theta` at theta_ref,
   their derivative in beta, sum a_i t mu_i (y_i - mu_i)/(t + mu_i)^2 x_i with
   t = theta_ref, and sum a_i y_i log y_i. The NT cells' totals are summed once
   per response. A pair adds the totals of its cells outside the NT cells. The
   order is fixed: the NT totals, then the target's cells in ascending order.
   A pair's totals therefore depend only on its set of cells and not on which
   of them carry the target's label. Permuting the labels within the pair, as
   the resampling does, leaves the null unchanged, as it does for the cold fit.
3. **One-step start.** One Fisher-scoring step, beta_1 = beta_ref + I^-1 s, and
   one Newton step in u = log theta, linearized in beta with the cross term,
   from the pair's totals. This is the one-step estimator
   ([Le Cam 1956](https://projecteuclid.org/euclid.bsmsp/1200501652);
   [Bickel 1975](https://doi.org/10.1080/01621459.1975.10479884)). The start
   differs from the pair's fit by the share of the other targets' cells in the
   reference, about a fifth of the cells on the Replogle pilot. The step
   corrects that to first order only, so it is used as a warm start
   ([Friedman, Hastie and Tibshirani 2010](https://doi.org/10.18637/jss.v033.i01)),
   not as the null.
4. **Refinement.** Newton passes over the pair's cells compute, in one pass
   each, the exact gradient, the Poisson deviance and, while the relative
   coefficient step exceeds 1e-6, the information. A step that raises the
   deviance by more than the cold fit's step acceptance allows (1e-12 of the
   log likelihood) is halved, up to 30 times. The fit has converged when the
   predicted gain of the next step relative to (0.1 + deviance), and its
   largest relative coefficient change, are below the tolerance (the floor
   100 eps/sqrt(rcond) applies as in `fit()`). These are the chord criteria of
   the cold IRLS. That last step is then taken, as the cold fit takes its
   chord refinement.
5. **Theta.** Newton in log theta at the converged means, from the start's
   theta, with the score and curvature of `estimate_theta`. It stops when the
   log-scale step is below a tenth of the tolerance or the score is below
   1e-10. A non-concave or large (|step| > 1) iteration hands over to today's
   procedure: the moment start followed by `estimate_theta`.
6. **Fallback.** The cold fit is used for every pair of a response whose
   reference did not converge or has a boundary. It is also used for a pair
   whose refinement does not converge within 30 passes or meets a singular
   information. Boundaries and rank deficiency thus keep the cold fit's statuses.
7. **Score.** The score records and the covariate-adjusted information are
   computed in one pass from the pair's means and theta. The information is
   summed over the rows' nonzero entries rather than as a dense product.

The refined null agrees with the cold fit to the convergence tolerance.
Empirical p-values are the same unless a draw lies within rounding of the
observed statistic. Skew-normal tail p-values are continuous in all the draws'
statistics and can differ in the last digits, and by more far in a tail.

On the Replogle K562 pilot (20 targets, 8,563 genes, 174,761 tested pairs,
8 threads), refine and cold were compared on the same binary:
- Agreement: coefficients within 1.2e-11 and theta within 6e-10 at the 99th
  percentile (6.4e-4 at most, on flat high-theta likelihoods). Statistics are
  within a relative 4.8e-8. All 158,699 empirical p-values are identical, as
  are the statuses and the discoveries of every family. 77 of 16,062
  tail-approximated p-values differ by more than 1e-4, all of them below 3e-17.
- Work per pair: 9 exact passes at the median (99th percentile 16), against
  about 55 for the cold fit (4.6 IRLS iterations, two theta rounds and a dense
  51 x 51 information). 7.4% of pairs hand theta over, and no pair fell back
  to the cold fit.
- Time: null work fell from 20.9 to 5.1 thread-ms per pair. The `ks` run's
  wall time fell from 1,262 s to 592 s. The `strict` run fell from 3,666 s to
  3,028 s: its draws take most of its time and are the same work under both
  starts.

[MERGEABLE_NULL.json](validation/MERGEABLE_NULL.json) records the measured
alternatives, their numbers and the reasons for this choice:
- the one-step null without refinement, from all-cells or NT-only references;
- an NT-only base with a quadratic model;
- a lockstep batched cold fit;
- an index-mapped score kernel.
SCEPTRE's per-pair refit is described from its documentation and black-box
outputs only.

## Bulk differential expression

`bulk_de` (CLI `bulk-de`) follows the procedure published for DESeq2
([Love, Huber & Anders 2014](https://doi.org/10.1186/s13059-014-0550-8)) and
its [documentation](https://bioconductor.org/packages/release/bioc/html/DESeq2.html).
No DESeq2 source was read. DESeq2 1.48.2 was run only as a black box, and
details the documentation leaves open were calibrated from its outputs; they
are marked *calibrated* below. From 1.2.0 the robust dispersion of Cook's
distances, and from 1.4.0 the starting dispersion and the floored fits, follow
PyDESeq2's implementation (MIT; Muzellec et al. 2023; see DEPENDENCIES.md),
which PyDESeq2 states follows DESeq2's code. Counts are samples × genes
(m samples, p design columns, m > p), tested coefficient j.

1. **Size factors** ([Anders & Huber 2010](https://doi.org/10.1186/gb-2010-11-10-r106)).
   `ratio`: s_i is the median over genes without zeros of y_ig / G_g, with G_g
   the geometric mean of gene g. `poscounts`: G_g is the m-th root of the
   product of the positive counts, medians use positive counts only, and the
   factors are scaled to geometric mean 1. Factors can also be supplied.
2. **Base mean** μ̄_g = mean_i y_ig / s_i. All-zero genes are reported and skipped.
3. **Initial means.** For designs with as many distinct rows as columns,
   μ̂_i = s_i × the mean normalized count of i's design cell, floored at 0.5.
   Otherwise an NB fit at the starting dispersion below, falling back to
   Poisson and then to least squares; where a fitted count falls below 0.5 the
   fit is redone with fitted counts floored at 0.5 (the floored fits, step 8)
   and its unfloored means are used, as in DESeq2 and PyDESeq2. The starting
   dispersion (PyDESeq2) is the smaller of a rough value, from a least-squares
   regression of the normalized counts on the design (fitted f floored at 1,
   residual r): max(0, Σ_i (r_i² − f_i) / f_i² / (m − p)), and a moments value,
   (v − x̄ μ̄) / μ̄² with v the variance of the normalized counts and x̄ the mean
   of 1 / s_i; clamped to [1e-8, max(10, m)]. When the size factors differ
   widely (pseudobulk samples of very different numbers of cells) the moments
   value is often negative, so such genes start, and keep, 1e-8 (step 4).
4. **Gene-wise dispersion** α_g maximizes the Cox-Reid adjusted profile
   likelihood at μ̂ (Newton in log θ from the rough value). *Calibrated:* genes
   whose rough value is at the 1e-8 floor keep 1e-8, as in DESeq2's output
   (`keep_minimum_start`; false gives the maximizer for every gene).
5. **Trend** over genes with α_g ≥ 1e-7 (*calibrated*). `parametric`:
   α_tr = a0 + a1 / μ̄, a gamma GLM with identity link, refitted without genes
   whose ratio α_g / α_tr lies outside [1e-4, 15] until the summed squared log
   coefficient changes fall below 1e-6; if it fails, `local`. `local` is a
   RapidGLM substitute for DESeq2's locfit: the running median of log α_g over
   log μ̄, window max(10% of genes, 100). `mean` is the mean of α_g.
6. **Prior** log α ~ N(log α_tr, σ²). With s_lr the MAD of log α_g − log α_tr
   over the trend genes, σ² = max(s_lr² − ψ1((m − p)/2), 0.25). For m − p ≤ 3,
   σ² is the grid value in [0, 1] (1,000 points) whose simulated
   log(χ²_{m−p} / (m − p)) + N(0, σ²) density is closest in Kullback-Leibler
   divergence to the observed residuals. The simulation uses 20,000 common
   Philox draws, and densities are histograms on 100 bins over the observed
   range (a RapidGLM choice).
7. **MAP** α maximizes the Cox-Reid adjusted likelihood plus the log prior,
   from the trend. Dispersion outliers, log α_g > log α_tr + 2 s_lr, keep α_g.
8. **Test.** An NB GLM at the final α with offsets log s_i. Wald z = β_j / SE_j,
   or the LR test against the model without column j (χ²₁). log2 fold change
   = β_j / log 2. DESeq2 fits with fitted counts floored at `min_mean` (0.5,
   its documented `minmu`): where a fitted count of the maximum likelihood fit
   falls below it, or a coefficient has no finite estimate (for example a group
   of zeros), the gene is refitted with the floor, by PyDESeq2's iterations
   (iteratively reweighted least squares on μ_f = max(s exp(Xβ), 0.5), ridge
   1e-6, coefficients within [−30, 30], until the deviance changes by at most
   1e-8 of itself plus 0.1). Its Wald SE is the sandwich
   (H + 10⁻⁶ I)⁻¹ H (H + 10⁻⁶ I)⁻¹ with H = X'WX at the floored means
   (*calibrated*: DESeq2's SEs; PyDESeq2 uses the unfloored means here), its LR
   statistic the difference of the two likelihood deviances at the returned,
   unfloored means (calibrated by public `nbinomLRT` runs), and its Cook's
   distances use the unfloored means and the leverages of the floored weights.
   A group of zeros then has a finite fold change. The C++ test example gives
   about −9.390, SE 1.055; the separate R-simulated calibration gave −9.35,
   SE 1.10 in both tools. Failed floored fits report `floored_fit_failed`.
   Bulk bounds and ridge scale with the design columns' maximum magnitudes;
   the sandwich expression above uses normalized columns and is transformed
   back to input units. Crossing a bound or reaching the IRLS limit triggers bounded optimization.
   The Python inference interface retains PyDESeq2's coefficient units and
   hands every native optimizer fallback, failed fit and bound-active fit back
   to its default inference. Boundary warm starts outside the bounds restart
   from the least-squares initialization instead of clipping a coefficient
   vector whose linear predictors are valid only before clipping.
9. **Cook's distance** ([Cook 1977](https://doi.org/10.2307/1268249)):
   D_i = R_i² / p × h_i / (1 − h_i)², with R_i = (y_i − μ_i) / √(μ_i + α_rob μ_i²)
   and h_i the hat values of the final fit. Before computing a distance,
   leverage within 4 × double-precision machine epsilon of 1 is treated as
   exactly 1. Its distance is undefined (NaN), so it is excluded from finite
   maxima and from replacement. This applies to both robust-dispersion modes
   and to refits after replacement. If any observation is saturated, the
   largest distance across all samples is undefined, so the legacy count-rank
   exemption below is not applied: an above-cutoff finite distance in an
   eligible replicated cell flags the gene. Saturation alone does not flag
   a gene. This avoids last-bit weight changes granting or removing the
   exemption.
   The cutoff is the 0.99 quantile
   of F(p, m − p). DESeq2 uses a robust dispersion α_rob here that its paper
   and documentation do not specify in reproducible detail.
   - By default (`legacy_cooks`, from 1.2.0), α_rob is DESeq2's robust
     dispersion as implemented in PyDESeq2 (MIT; Muzellec et al. 2023,
     Bioinformatics 39:btad547; see DEPENDENCIES.md):
     - **Variance.** In each cell of at least three samples, take the trimmed
       mean of the normalized counts, then the scaled trimmed mean of their
       squared deviations from it. Trim and scale are 1/3 and 2.04 below 3.5
       samples, 1/4 and 1.86 below 23.5, and 1/8 and 1.51 otherwise. v is the
       maximum over cells; without such cells it is 1.51 times the 1/8-trimmed
       variance of all samples.
     - **Dispersion.** α_rob = (v − μ̄)/μ̄², floored at 0.04.
     - **Flagging.** A gene is not flagged when three or more samples have
       larger counts than the sample with the largest distance.

     These rules follow the public PyDESeq2 implementation. The historical
     claim of equality to inverted DESeq2 dispersions is not established by
     the current receipts. The public-assay inversion check supports per-gene
     consistency within 1e-12 on a declared subset with all fitted means at
     least 0.5; its maximum relative range is 4.684e-15 there and 3.027e-9 over
     all eligible genes, including low means. This is a check of the reference
     inversion, rather than of equality to the native estimator. See
     [COOKS_INVERSION.json](validation/receipts/v1.4.0-r2/COOKS_INVERSION.json)
     and [BULK_DE.md](docs/BULK_DE.md#legacy-cooks-distances-2026-10-01).
   - With `legacy_cooks: false`, α_rob is RapidGLM's own estimator, the default
     before 1.2.0. It pools the squared scaled MADs of the normalized counts
     within design cells of at least three samples (the ordinary variance if
     there are none), converts the pooled variance v to (v − μ̄) / μ̄² and floors
     that at 0.25 α_tr. With many small cells it gives smaller dispersions, so
     larger distances and more flagged genes than DESeq2.
   - In cells with at least 7 replicates, counts above the cutoff become
     floor(s_i × the 20% trimmed mean of normalized counts). Such genes are
     re-estimated from step 3 with the trend and prior kept.
   - Otherwise a gene whose largest D over samples in cells of 3–6 replicates
     exceeds the cutoff gets no p-value (`cooks_outlier`).
10. **Independent filtering**
    ([Bourgon, Gentleman & Huber 2010](https://doi.org/10.1073/pnas.0914005107))
    on μ̄. Take 50 quantile levels θ from the fraction of zero means to 0.95.
    At each θ, genes with μ̄ at or above the type-7 quantile pass (the `>=`
    rule is *calibrated*), and the BH rejections at α are counted. A LOWESS
    ([Cleveland 1979](https://doi.org/10.1080/01621459.1979.10481038); span
    0.2, *calibrated*, with three robustness passes) is fitted to rejections
    over θ. If the maximum rejection count is at most ten, choose the first θ.
    Otherwise choose the lowest θ whose rejections exceed the curve's maximum
    minus the residual RMS, calculated over quantiles with positive rejections.
    These are calibrated rules following PyDESeq2, checked against black-box
    DESeq2 results. BH adjustment then runs
    over the passing genes with p-values; the others get `NA`.
11. **Fold-change shrinkage** (optional, `lfc_shrink=normal`): the paper's
    empirical normal prior, as DESeq2's `lfcShrink(type = "normal")` computes it
    for a user-supplied model matrix. Tests and p-values are unchanged.
    - **Prior.** Each design column that varies across samples gets a
      zero-centred normal prior on its log2 coefficient. Constant columns (the
      intercept) get variance 1e6. The variance is (q / z_0.975)², matching the
      0.05 upper quantile q of the absolute maximum likelihood coefficients to a
      normal distribution. The quantile is weighted by 1 / (1 / μ̄ + α_tr), as
      documented. The weights are scaled to sum to the number of genes, and the
      type-7 rule is applied to the sample with each value repeated by its
      weight (*calibrated*: this reproduces DESeq2's prior variances from its own
      estimates to rounding). Coefficients with no finite estimate count as
      infinite.
    - **Posterior mode.** It maximizes the NB log likelihood at the final
      dispersion plus the log prior (penalty 1 / (σ² ln² 2) per coefficient on
      the natural-log scale), by Fisher scoring with step halving.
    - **Posterior SD.** The sandwich (H + P)⁻¹ H (H + P)⁻¹, with H = X'WX from
      fitted counts floored at 0.5, DESeq2's documented `minmu` (*calibrated*:
      this form reproduces DESeq2's `lfcSE`).
    - **Counts.** Where outliers were replaced, the replaced counts are used,
      consistent with the test (a RapidGLM choice). DESeq2's `lfcShrink` uses the
      original counts (black-box observation).

**Final 1.4.0 agreement with DESeq2 1.48.2.** On the five validation
datasets, discovery Jaccard ranges from 0.9834
to 1.0000; p-value Spearman is at least
0.999813. The one- and eight-thread
tables are byte-identical on every dataset. Size factors differ by at most
4.108e-15 relative.
These are final-code measurements from [BULK_DE.json](validation/BULK_DE.json);
[the bulk report](docs/BULK_DE.md) gives the simulations, null checks,
shrinkage, Cook's-distance and pseudobulk comparisons.

## Quasi-likelihood differential expression

`edger_ql` (CLI `edger-ql`) follows the default quasi-likelihood workflow of
edgeR 4 (`filterByExpr`, `normLibSizes`, `glmQLFit`, `glmQLFTest`; Chen, Chen,
Lun, Baldoni and Smyth 2025, NAR 53:gkaf018; Lund et al. 2012). It was written
from the papers, the documentation of edgeR and limma and black-box runs of
edgeR 4.6.2; no edgeR or limma source was read. Rules the documentation leaves
open were settled by the black-box runs, recorded with the evidence for each in
the specification the implementation follows (`docs/EDGER_QL_SPEC.md`).

1. **Filtering.** A gene is kept when its count per million of library, at
   least `min_count` reads per median library (10), holds in at least m
   samples and its total is at least `min_total_count` (15). m is the inverse of
   the largest leverage of the design, grown beyond `large_n` (10) by
   `min_prop` (0.7) per sample. Library sizes are then recomputed over the kept
   genes.
2. **TMM** (Robinson and Oshlack 2010). The reference sample has the upper
   quartile of count shares closest to their mean. For each sample, genes with
   reads in it and the reference are trimmed by 30% of log ratios and 5% of
   average log shares at each end (average ranks for ties), and the factor is
   the inverse-variance weighted mean log ratio. Factors are scaled to a product
   of 1.
3. **Average log CPM.** A one-group NB fit at dispersion 0.05 of the counts
   plus 2 × library / mean library, with offsets log(library + 4 × library /
   mean library).
4. **The NB dispersion φ.** The Cox-Reid adjusted profile likelihood summed
   over the ceil(t G) genes of highest average log CPM, t = 0.02 + 0.98 (20 /
   (G √(n − p)))^(1/3) (1 when G √(n − p) ≤ 20), maximized in φ^(1/4) on
   [0, 4^(1/4)] by Brent's method (tolerance 1e-5).
5. **Quasi-dispersions.** Each gene is fitted at the working dispersion φ/a.
   Each observation's unit deviance d is divided by s and given df (1 − h) k,
   with k = 2E²/V and s = V/(2E) the mean and variance of the NB unit deviance
   at mean μ/a and dispersion φ and h the leverage; the gene's s² is the sum of
   the first over the sum of the second. The scale a is max(1, the 0.9 quantile
   of the fourth power of the LOWESS fit (span 0.5, three robustness passes) of
   s²^(1/4) on average log CPM), computed at a = 1 and again at that value, with
   the genes refitted each time. The moments E and V are exact (*RapidGLM
   choice*): sums over the NB distribution, counts above 1000 by quadrature,
   tabulated per dispersion on 64 points per decade of the mean. edgeR uses an
   approximation of them that differs from the exact values by up to about 1e-3
   for dispersions up to 0.3 and by more at larger dispersions and means; it is
   not documented.
6. **Moderation** (Smyth 2004, with limma's weighting for unequal df). s² is
   floored at 1e-12 of its median; the log-moment terms log s² − ψ(d/2) +
   log(d/2) are weighted by 1/ψ'(d/2) and smoothed against average log CPM by a
   weighted LOWESS (span 0.3 + 0.7 (500/G)^(1/3), 1 for G ≤ 500; anchors at
   1/200 of the range); the prior df d0 maximizes the likelihood of the scaled F
   model with the prior scale tied to the trend by ψ(d0/2) − log(d0/2). Its
   search ends at 8134.84478, the value edgeR 4.6.2 reports when the
   likelihood keeps increasing. s²_post = (d0 s²_prior + d s²)/(d0 + d).
7. **Tests.** F = (D_reduced − D_full)/(q s²_post) on q and d0 + d df, with
   deviances at φ/a; Benjamini-Hochberg over the kept genes.
   The total denominator df is uncapped in RapidGLM. On the 12-gene/8-sample
   CLI fixture, public edgeR reports df.total=72 versus about 8140.82 here;
   gene 0 has p=.24615077 versus .24226256. The fixture's absolute .005
   p-value tolerance does not establish identical tails for small panels.
   See [EDGER_SMALL_PANEL_LIMITATION.json](validation/receipts/v1.4.0-r2/EDGER_SMALL_PANEL_LIMITATION.json).
8. **Fold changes.** From fits at φ/a of the counts plus 0.125 × library /
   mean library with offsets log(library + 0.25 × library / mean library).

Fits at a fixed dispersion are by Newton's method (observed information),
warm-started across passes: whole steps while the Newton decrement (about twice
the deviance still to gain) shrinks, then steps halved until the deviance does
not increase, until the decrement is below 1e-14 × (100 + the gene's total
count). After a whole step, convergence is confirmed without a new
factorization when the decrement computed with the previous information, times
exp of the largest change in a linear predictor (which bounds the change of
every weight), is below that threshold; otherwise the next step computes the
decrement. The bound assumes well-conditioned information. Near a group of
zeros and fitted means around 1e-12, rounding can make the shortcut stop before
the full test; observed differences in statistics are negligible. A fit that
does not converge in 60 steps is refitted by `fit`.
Leverages come from the Cholesky factor of X'WX, or from a Householder QR of
W^(1/2)X when fitted means near zero make that factor ill-conditioned.

**Final 1.4.0 agreement with edgeR 4.6.2** (eight datasets).
The kept-gene sets are identical. Maximum discrepancies are
5.107e-15 relative for TMM
factors, 6.509e-12 absolute
for average log CPM and
3.398e-05
relative for the common dispersion. At FDR 0.1, discoveries are identical on
airway, kang, sim3, sim3b, sim8. On the thyroid subsets, black-box checks of edgeR's
public fitted values identify inconsistent reported deviances or failed fits
in 22,
7 and
9 genes (32, 128 and 647 samples).
Without reading edgeR's implementation, the reason for those reported
deviances cannot be established. Over the other genes, with BH recalculated
on that subset, discovery counts are:

| Samples | RapidGLM | edgeR | Shared |
|---|---:|---:|---:|
| 32 | 1,370 | 1,363 | 1,363 |
| 128 | 822 | 821 | 821 |
| 647 | 7,480 | 7,477 | 7,477 |

Source: [EDGER_QL.json](validation/EDGER_QL.json), from the complete final release run.

## Other applications

Pseudobulk sums cells within caller-specified sample/group units and fits sample
NB GLMs with explicit offsets (typically log library size). No automatic TMM,
size-factor shrinkage or biological replicate invention. DA fits binary group
membership ~ peak count + supplied covariates (including log depth), comparing
to the model without peak count by LR. This matches the documented regression
direction of [Signac](https://stuartlab.org/signac/reference/findmarkers).
Dose trend tests a supplied numeric dose column using the selected GLM family.
Count-model assumptions for RNA, peaks and antibodies are identical NB2 working
assumptions, not assertions that each modality follows an NB sampling law.

`edger-ql` accepts nonnegative integer counts only; fractional counts are
rejected before normalization or likelihood evaluation. This is a RapidGLM
input restriction; edgeR also supports workflows using fractional counts.

Additional edgeR workflow choices: dispersion genes must have row sums at
least 5. Every trial dispersion uses this same set; if any gene fails, that
objective value is invalid. The likelihood-ratio statistic is clamped at zero,
and BH adjustment includes only finite p-values. RapidGLM currently uses all
selected dispersion genes, including when there are more than 10,000; edgeR's
systematic 10,000-row subset is not implemented. The moderation LOWESS anchor
spacing is range/200, including at most 200 distinct abundances; this is a
RapidGLM choice, differing from the black-box specification's zero spacing
for small sets. The prior-df search is bounded by 8134.84478 but returns its
numerical interior optimum (about 8134.8433 at the upper boundary), rather than
that exact endpoint. Counts occupy the sparse input and a dense kept-gene
matrix; TMM temporarily transposes the dense matrix. These memory costs remain.

## Cis-QTL mapping

`cis_qtl` and `rapidglm qtl` test each dosage by its efficient score against a
covariate-only Gaussian, Poisson or NB null. With working weights W and score
residual r, U = g'r − g'WX(X'WX)⁻¹X'r and I = g'Wg − g'WX(X'WX)⁻¹X'Wg.
The reported z is U / sqrt(I × scale); the one-step estimate is U/I and its
SE sqrt(scale/I). Gaussian scale is the residual variance, count-model scale
is one. Missing dosages are imputed to each variant's observed mean.

Each phenotype's inclusive cis window is [start−window, end+window]. CLI
BED intervals are zero-based half-open and convert to the 1-based inclusive
[start+1,end]; point records with start=end anchor at that supplied coordinate.
PLINK allele A1 (ALT in plink2-generated files) is counted. Genotypes and
covariates are matched to BED sample IDs. Tied positions retain input order;
exact score ties select the first variant in that order.

Each permutation shuffles the response and log-library offset together and
refits the null. For unweighted Gaussian fits without offsets, ridge or robust
covariance, the permutation scores instead use a projection computed once per
gene. The observed information basis supplies S = Q'G and I = diag(G'G) −
diag(S'S), where Q = X D⁻¹L⁻ᵀ and LL' = D⁻¹(X'X)D⁻¹. Thus Q has orthonormal
columns, G_res = G − QS, and G_res'(I−QQ')y_perm = G_res'y_perm.

Batches of 512 responses retain the existing CounterRng streams and shuffle
order. Each shuffle's generator values are computed in bulk, eight Philox
blocks at a time with the vectorized Philox, before its Fisher–Yates swaps
run over them; each range r takes its rejection threshold 2⁶⁴ mod r from a
table and its remainder without a division, as the integer part of
((⌈2¹²⁸/r⌉x) mod 2¹²⁸)·r/2¹²⁸, which equals x mod r for every 64-bit x
(Lemire, Kaser and Kurz 2019). A shuffle that rejects a value (each value with
probability below n/2⁶⁴) continues with the following blocks, so every shuffle
is that of the generator drawn value by value. On an efficiency core the
shuffles of 10,000 permutations of 462 samples take 32 ms instead of 59 ms,
two 64-bit divisions per swap having taken most of the time. Packed
matrix products give Y_perm'G_res and Y_perm'Q; each scale is
(|y|² − |Q'y_perm|²)/(n−p). If the residual squared norm is at most 1e-6 of
|y|², explicit least-squares residuals supply both the scale and the score
input to avoid cancellation. The observed statistics and information are
unchanged. Double-precision products use 128-sample by 128-column cache panels
and six-by-eight register tiles, with runtime-dispatched AVX2/FMA and a portable
path. Every sum visits samples in order across panels; packing buffers belong
to the calling gene worker. No external BLAS matrix product is used, preserving
independence from its build and avoiding shared packing-buffer contention.

**Single-precision permutation scores.** The score product Y_perm'G_res takes
about 97% of the Gaussian permutation time. By default
(`permutation_precision: single`) it is computed in single precision, in three
steps:

1. Before its conversion to float, each permuted response is centred or
   residualized in double precision. G_res is orthogonal to the covariates, so
   in exact arithmetic y_b minus any vector in their span has the scores of
   the residual r_b = y_b − Q t_b; for the float product what matters is that
   no large component that the covariates explain remains. A permutation
   leaves the covariates little to explain, so by default its centred response
   y_b − ȳ1 stands for r_b (ȳ is the phenotype mean when the design has a
   constant column, and 0 otherwise). Its sum of squares SS_c is the same for
   every permutation. A permutation with SS_c > 2·rss_b, which the covariates
   happen to fit well, keeps r_b, using the t_b = Q'y_b already computed for
   its scale, and Q T' is formed for those permutations alone, with the sums
   used before. Columns that took the explicit least-squares fallback already
   hold residuals. With SS_c ≤ 2·rss_b the bounds on the float rounding of the
   response and of every float sum (γ|g_res||response| by Cauchy–Schwarz) are
   at most √2 times those for r_b, and the part of y_b − ȳ1 in the covariate
   span, of norm √(SS_c − rss_b) ≤ √rss_b, adds at most the float rounding of
   G_res times that norm. Either way no large shared component, such as the
   phenotype mean, cancels in single precision. At the Geuvadis dimensions
   (462 samples, 43 covariates) a permutation's R² against the covariates is
   about 0.09 (SS_c/rss_b ≈ 1.1), so every permutation is centred, which
   saves the product Q T' and its subtraction. Each response is divided by its
   residual standard deviation before the conversion, so the float entries are
   O(1) whatever the phenotype's units: like the double path, the single path
   does not depend on the scale of the phenotype, which can lie far outside the
   float range.
2. G_res is converted to float once per gene, by the packed double product
   Q S, and stored in groups of sixteen variants, each sample's sixteen in one
   64-byte cache line.
3. Six-by-sixteen float register tiles (AVX2/FMA, or a portable tile on other
   processors, including ARM64) sum each 128-sample panel from zero. Each
   panel's sums are then added to double partials, in sample order, so 462
   samples need four float partial sums per element. The scores are never
   stored: after the last panel, each tile multiplies its partials by the
   variants' inverse information, takes |z| and folds it into its six
   permutations' running maxima, skipping NaN and infinity as before. The
   twelve float sums occupy twelve of the sixteen YMM registers (the compiled
   sample loop keeps them there without spills), so the double partials
   (768 bytes per tile) live in an L1 scratch, read and written once per
   panel; tiles small enough to keep double partials in registers (2 × 16,
   4 × 8) ran four times slower. Variants are taken 128 at a time: each row
   tile's partials for the block (6 kB) stay in L1 while it passes over the
   block's packed genotypes (236 kB at 462 samples), which stay in L2 for
   every row tile. Before this, each 512-permutation batch wrote a 512 × m
   double score matrix (18 MB at 4,500 variants) and read it back for the
   maxima; with twelve threads that traffic left the caches, and each core ran
   at 83% of its single-thread rate, against 94% now (48 synthetic genes of
   4,500 variants on twelve efficiency cores, `benchmarks/qtl_precision.cpp`).

Each panel sum is within γ₁₂₉ = 129u/(1−129u) ≈ 7.7e-6 (u = 2⁻²⁴) of the
panel's sum of |terms|. Scales and their fallback, information, the maxima,
observed statistics, nominal output, refitted permutations and the Beta
calibration stay in double precision.

Single and double results were compared on test shapes covering 37–462
samples, 1–4,500 variants, 1–43 covariates, a phenotype mean of 10⁶,
phenotypes scaled by 10³⁸ and 10⁻⁴², nearly collinear covariates and
one-carrier variants. Permutation maxima agree to within 1.03e-6 of
max(1, |z|): relative for maxima of at least 1 and absolute below, because a
small score can lose more of its relative accuracy to cancellation. The
largest difference is 9.7e-7 with centred responses (step 1), and was 1.03e-6
with every permutation residualized; on the Geuvadis-sized shape it is 3.9e-7
(2.9e-7 residualized). Simulated gene-level beta_p agrees to within 3.3e-7
relative. On Geuvadis chromosome 22 (580 genes), statuses, leads and observed
statistics are identical for every gene. With 1,000 permutations, empirical
p-values are identical except in 12 genes, all with |z| < 4: in each, one
permutation maximum lies within the single-precision tie tolerance below the
observed statistic (see below) and counts as extreme, which raises p by
1/1001. beta_p agrees to within 3.8e-5 relative, at beta_p ≈ 2 × 10⁻⁴⁹, and
to within 5.0e-6 wherever beta_p > 10⁻¹⁰; beta_k and beta_n agree to within
2.9e-6 relative. With 10,000 permutations, 76 empirical p-values (|z| ≤ 4.2)
rise by 1 to 3 counts out of 10,001 through the tie tolerance, beta_p agrees
to within 2.3e-5 relative (4.1e-6 wherever beta_p > 10⁻¹⁰), and both
precisions find the same 267 eGenes at Storey q < 0.05. Residualizing every
permutation gave 12 and 77 such genes, and beta_p within 2.5e-5 relative at
both budgets.

Maxima that are equal in exact arithmetic are spread by float rounding: their
relative variance is about 3e-14 at 462 samples, above the double threshold
of 64ε. For single-precision maxima, the Beta calibration's constant-maxima
test therefore uses a relative variance of 1e-8 (a relative SD of 1e-4), six
orders of magnitude above that spread. Maxima with a relative SD of 1e-3 are
still fitted, as in double precision.

Results do not depend on the thread count, the batch boundaries or the order
in which genes reach a worker. The AVX2 and portable float tiles
round differently (fused or separate multiply-add), and other architectures
can differ too, so single-precision results can differ in the last bits
between them. `permutation_precision: double` keeps the 1.4.1 products. It
reproduces 1.4.1 output except where a permutation maximum lies within 1e-9
relative below the observed statistic (see below). Geuvadis chromosome 22
(1,000 permutations), a 60-gene subset with 10,000 permutations and nominal
output, and an NB subset are byte-identical to 1.4.1.

Successful maxima give the empirical p=(1+extreme)/(1+successful). A maximum
counts as extreme when it is at least |z|(1 − τ), with τ = 1e-5 for
single-precision scores and 1e-9 otherwise. The observed statistic and the
permutation maxima are rounded along different paths, and some designs make
them equal in exact arithmetic. With an intercept only and a private variant
in every sample, for example, every permutation has the observed largest |z|
among those variants, so the exact permutation p-value is 1. Without τ,
rounding decides on which side each such tie falls; in tests of this design
permutation_p then ranged from about 0.2 to 0.9 in either precision. In single
precision, τ is ten times the largest measured spread of such ties (1e-6
relative). Projected double-precision scores spread them mainly through the
rounding of their scales (|y|² − |Q'y_perm|²). That spread grows with the
squared ratio of the phenotype mean to its residual SD, from about 1e-15 at
mean zero to 1e-10 at 300 SDs, so τ covers means up to several hundred
residual SDs. Counting
near-ties as extreme is conservative: it raises the empirical p-value by the
share of maxima within τ below |z|.

Fewer than ten usable permutations gives `permutations_failed`. Singular null
information becomes a per-gene status; singular permutation information is
counted as a failed draw.
The Beta calibration follows jaxQTL's noncentral chi-square reference and
maximum-likelihood Beta shapes (credited in DEPENDENCIES.md). Nearly constant
maxima fail calibration using a relative variance test; failed shapes and
noncentrality are NaN, with `ok_beta_failed` retaining nominal results.

**Genotype storage and memory.** `Genotypes` holds either dosages, eight bytes
per call with NaN for a missing call, or hard calls: the two-bit codes of a
variant-major PLINK 1 .bed (00 two copies of A1, 01 missing, 10 one copy, 11
none), a quarter of a byte per call. `rapidglm qtl` loads only chromosomes
present in the phenotype BED and keeps their codes as read; it repacks them
only when the BED's samples are not the .fam's samples in .fam order. Each
gene worker decodes its cis variants into its double-precision genotype block
as it tests the gene, and computes each variant's mean dosage from the codes,
four calls at a time through a 256-entry table of byte sums. Sums of whole
dosages are exact, so the decoded values, the means and every result are bit
for bit those of the same calls stored as dosages. For the
genome-wide Geuvadis panel (6,317,351 variants, 462 samples) the calls take
0.68 GiB instead of 21.7 GiB. For each variant the CLI also keeps the chromosome
name and position, and packs the ID and alleles into one string (31 bytes
rather than 128 for three strings). MAF filtering compacts the codes and these
vectors in place and then releases their spare capacity.

Each gene worker keeps its buffers from gene to gene in a workspace that grows
to the largest gene the worker has met and is never shrunk, so later genes
allocate, fault in and clear no memory. In single precision it holds the
genotype block in double precision (3.7 kB per cis variant at 462 samples),
the packed float projection (1.8 kB), the projection's 64-sample block
(0.5 kB) and the basis (0.3 kB): about 6.5 kB per cis variant of its largest
gene, with no score matrix, plus about 5 MB of batch buffers. In double
precision the scores of a 512-permutation batch (4.1 kB per cis variant) take
the place of the float projection and its block: about 8.2 kB. glibc keeps a
freed block below its mmap threshold (which rises to 32 MB as large blocks
are freed) resident in the arena of the thread that freed it; as workers met
ever larger genes, the blocks they outgrew raised the genome-wide peak by
0.8 GiB, so once a worker's buffers have grown by a megabyte or more it returns
the free pages (`malloc_trim`). Windows are largest in the HLA region: 143
chromosome 6 genes have more than 15,000 cis variants (up to 26,032), so with
12 threads every worker then holds buffers of that size, and they set the
genome-wide peak. Measured untimed with 12 threads and 10 permutations, which
allocate the same per-gene buffers as 10,000, the genome-wide scan peaks at
3.35 GiB of resident memory (3.52 GiB for the previous build measured the same
way; 25.8 GiB when calls were stored as dosages, close to the 25.9 GiB of the
1.4.1 run with 10,000 permutations). Chromosome 22 peaks at 0.65 GiB in single
precision, with 1,000 or 10,000 permutations (0.90 and 0.94 GiB before), and
at 0.88 GiB in double precision with 10,000 (0.84 GiB before).

Nominal output keeps every cis pair in memory (48 bytes each). Flattening
allocates the full table while every gene's rows are still held, so it
briefly needs twice that. For very large nominal scans, partition phenotypes
by chromosome.

**Final 1.4.0 QTL agreement.** On Geuvadis chromosome 22 (580 genes,
462 samples), Gaussian 10,000-permutation calibration finds
267 eGenes at Storey q < 0.05. Discovery Jaccard is
1.0000 with
FastQTL's adaptive 1,000–10,000 run and
0.9963
with tensorQTL at 10,000 permutations. NB at 1,000 permutations finds
168 eGenes, with Jaccard
0.9882 against
jaxQTL NB at the same budget. Full metrics and limitations are in
[EQTL_COMPARE.json](validation/EQTL_COMPARE.json) and
[the QTL report](docs/QTL_GAUSSIAN_SPEED.md). This panel does not establish
genome-wide performance or agreement.

On ARM64, the KS discrepancy of the skew-normal fit uses the double-precision
CDF identity Phi(z) − 2 T(z, shape), with promotion disabled for Owen's T.
Its absolute error is checked against the promoted Boost distribution CDF on
a grid of shapes and statistics. The inferential tail and strict tail-soundness
checks retain their distribution CDF and quantiles. Integer generator streams
are unchanged; floating-point statistics and fitted-tail p-values can round
differently across architectures. Exact empirical-p comparisons are recorded
in the historical [dev.3 ARM64 study](validation/ARM64_DEV3.json), without
relaxing its comparison criteria. The final selected test suite has its own
[ARM64 receipt](validation/ARM64.json).
