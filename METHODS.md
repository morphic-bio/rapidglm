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
logistic propensity or caller-supplied probabilities. The default `thinned`
draw groups observations by q=2^-b with q/2<p_i<=q, visits Bernoulli(q)
candidates through geometric gaps and accepts each with probability p_i/q;
levels with q>=1/4 use one uniform per observation. The law is unchanged and
the expected work is O(sum p_i) rather than O(n). When the mean p_i is at least
1/16, and for `crt_draw=direct`, one uniform is compared with each p_i in
observation order, reproducing earlier streams. Thinning by acceptance is the
standard construction of [Lewis & Shedler 1979](https://doi.org/10.1002/nav.3800260304). A sampler callback permits
other documented conditional laws; no lineage sampler is copied or qualified.

`crt_draw=hazard` draws the same law without rejection. Observation i is given
an interval of length -log(1 - p_i) on a line, the intervals laid end to end in
observation order. The points of a unit-rate Poisson process on the line are
generated by adding standard exponential gaps, and an observation is selected
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
saves, so `hazard` then draws with the thinned sampler, from thinned's stream;
the run record's `crt_draw_used` tells which. A point's observation is found from a table with
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
   statistic the difference of the two floored deviances, and its Cook's
   distances use the unfloored means and the leverages of the floored weights.
   A group of zeros then has a finite fold change, as in DESeq2 (−9.35, SE 1.10
   in tests/test_bulk.cpp's example, equal to DESeq2's). Before 1.4.0 such genes
   were reported by the LR test as `boundary_lrt`; that status remains only if
   the floored fit fails.
9. **Cook's distance** ([Cook 1977](https://doi.org/10.2307/1268249)):
   D_i = R_i² / p × h_i / (1 − h_i)², with R_i = (y_i − μ_i) / √(μ_i + α_rob μ_i²)
   and h_i the hat values of the final fit. The cutoff is the 0.99 quantile
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

     These values equal those recovered from DESeq2's outputs to 1e-12, and
     they reproduce DESeq2's discoveries
     ([BULK_DE.md](docs/BULK_DE.md#legacy-cooks-distances-2026-10-01)).
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
    over θ. The chosen θ is the lowest whose rejections exceed the curve's
    maximum minus the residual SD, √(Σ residual² / 49). BH adjustment then runs
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

## Quasi-likelihood differential expression

`edger_ql` (CLI `edger-ql`) follows the default quasi-likelihood workflow of
edgeR 4 (`filterByExpr`, `normLibSizes`, `glmQLFit`, `glmQLFTest`; Chen, Chen,
Lun, Baldoni and Smyth 2025, NAR 53:gkaf018; Lund et al. 2012). It was written
from the papers, the documentation of edgeR and limma and black-box runs of
edgeR 4.6.2; no edgeR or limma source was read. Rules the documentation leaves
open were settled by the black-box runs, recorded with the evidence for each in
the specification the implementation follows (`validation/work/edger/EDGER_QL_SPEC.md`).

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
8. **Fold changes.** From fits at φ/a of the counts plus 0.125 × library /
   mean library with offsets log(library + 0.25 × library / mean library).

Fits at a fixed dispersion are by Fisher scoring, warm-started across passes:
whole steps while the Newton decrement (about twice the deviance still to gain)
shrinks, then steps halved until the deviance does not increase, until the
decrement is below 1e-12; a fit that does not converge in 60 steps is refitted
by `fit`. Leverages come from the Cholesky factor of X'WX, or from a
Householder QR of W^(1/2)X when fitted means near zero make that factor
ill-conditioned.

**Agreement with edgeR 4.6.2** (`scripts/validate_edger.py`; eight datasets).
The kept genes, TMM factors (to 5e-15), average log CPM (to 1e-11) and φ (to
1e-8) agree. On airway, kang and three simulated sets the discoveries at FDR
0.1 are identical and p-values agree to within 0.007 in log10. On the GTEx
thyroid sets, for 22, 7 and 9 genes edgeR's `glmFit` reports a deviance that is
not the deviance of its own fitted values, or flags the fit as failed, for the
full or the reduced design (black-box check, `scripts/reference_edger_fits.R`);
its F statistics for those genes use the reported deviance. Over the other
genes p-values agree to within 0.1 in log10 (0.005 for 32 samples), and with
Benjamini-Hochberg over those genes the discoveries differ by at most 7 (1,370
against 1,363 for 32 samples, 822 against 821 for 128, 7,480 against 7,477 for
647). Without edgeR's implementation the reason for those deviances cannot be
checked.

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
