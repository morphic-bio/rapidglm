# CLI and public API

Build with `scripts/ci.sh`. Requires a C++17 compiler, CMake, Boost >=1.65,
LAPACKE and BLAS/LAPACK. OpenMP is enabled by default
(`-DRAPIDGLM_OPENMP=OFF` for scalar builds).

`-DRAPIDGLM_LAPACK_SOLVES=ON` (default off) hands Cholesky factorizations of
more than 16 columns, and the triangular solves of the QR path, to LAPACK. By
default RapidGLM does this work itself at every size, so those results do not
depend on the linked BLAS. LAPACK is faster for designs with hundreds of
columns, but its level-3 kernels are chosen per processor at run time and are
then part of your results: with the option on, checking them is your
responsibility. Run `ctest` on each kind of machine that will run the fits. As
an example, the pthread OpenBLAS 0.3.20 of Ubuntu 22.04 selected its Cooperlake
kernel on an Intel Xeon 6975P-C and returned wrong triangular solves (`dtrsm`)
from order 32 up, while its SkylakeX and Haswell kernels were correct
(`OPENBLAS_CORETYPE` selects the kernel); see `validation/LAPACK_SOLVES.json`.
CMake prints a warning when the option is on, and every run record states
`lapack_solves`.
No R runtime is needed by the engine or CLI. `rapidglm --help` lists commands.

## Common inputs

Run `build/rapidglm COMMAND config.json`. Paths are relative to the working
directory, not the config file. Configuration keys:

| Key | Meaning |
|---|---|
| `responses` | MatrixMarket coordinate file, **features × observations**, real/integer/pattern, general symmetry; duplicates are summed |
| `design` | Numeric TSV, no header, **observations × predictors**; include an explicit intercept |
| `covariates`, `formula` | Alternative to `design`: header TSV and formula such as `~ log(depth) + factor(batch)` |
| `output` | Output prefix; writes `.tsv` and `.run.json` |
| `family` | `nb` (default), `poisson`, `binomial`, `gaussian` |
| `dispersion` | `ml` (default), `moment`, `fixed`; `shared` for `fit`/`pseudobulk-de`/`dose-trend`; `trend` for `fit` only. `sceptre` and `da-lr` reject `shared`/`trend` |
| `theta` | Positive NB size for fixed dispersion, default initial value 10 |
| `offset`, `weights` | Optional numeric vector files, one value per observation |
| `threads` | Positive integer, default 1 |
| `ridge`, `robust` | Penalty on all coefficients (default 0), HC0 covariance (default false) |
| `cox_reid` | NB with `dispersion=ml` only: estimate theta from the Cox-Reid adjusted profile likelihood (default false). Not accepted with `test=ql`, which always uses it |
| `tolerance`, `max_iterations` | Default 1e-9 and 100 |

Formulas support an intercept, `0`/`-1`, additive numeric terms, `log(name)`,
and `factor(name)`. Factors use sorted levels with the first as reference when
an intercept is present. Interactions, splines and arbitrary R expressions
are not parsed; supply a prepared design matrix for them. Invalid terms fail.
No missing-value deletion occurs. All input observations must have matching
orders; MatrixMarket and numeric designs have no identifiers to align for you.

Run records contain the parameters, seed, versions, elapsed wall time including
input ingestion, Linux peak RSS, host load, invalid-row counts and SHA-1 input
fingerprints. Validation receipts additionally use SHA-256. `NA` denotes
undefined values in TSV. Failed tests remain in multiple-testing denominators
with adjustment input p=1. Exit 2 means malformed input or execution failure;
scientifically invalid rows are retained with diagnostics and exit 0.

Count fits can return `converged_boundary`: the likelihood has stabilized,
but some coefficients have no unique finite estimate. `fit` adds `boundary0`,
`boundary1`, etc.; regression output adds `coefficient_boundary` for the tested
column. A flagged `beta`/`coefficient` is the finite iteration value only, not a
finite MLE. Its SE and Wald p-value are `NA`. Unflagged finite treatment effects
retain their tests. Ordinary LR and score inference reject boundary fits;
the perturbation workflow retains boundary nulls as `null_converged_boundary`.
In C++, inspect `FitResult::boundary_coefficients` or `has_boundary()`;
`ok()` accepts both `converged` and `converged_boundary` and does not guarantee
that every possible inference is valid.

## Commands

`fit` fits each response and returns coefficients, standard errors, theta,
log likelihood and diagnostics. Shared theta profiles the summed feature
likelihood. `trend` estimates a log-dispersion/log-mean trend, then refits;
currently this CLI option is limited to `fit`. C++ callers can supply their own
per-feature theta vector to `fit_batch`. All batch fit results are compact and
omit observation-sized state. A single `fit` API call retains this state by default.

`pseudobulk-de` accepts already aggregated sample counts, or cell counts and a
`units` file of zero-based sample/group indices. In the latter case `design`
has one row per aggregation unit. Units must all have cells. Provide offsets
explicitly. The tested hypothesis is one of (see
[Hypotheses](#hypotheses-contrasts-and-joint-tests)): `coefficient`, the
zero-based tested column (default last); `coefficients`, several columns tested
jointly; or `contrast`, one weight per design column.
`test` is `ql`, `lr` or `wald`. The default is `ql` for `pseudobulk-de` with
`family=nb`, `lr` for other `pseudobulk-de` families and `wald` for
`dose-trend`; the run record stores the choice. Biological replicates and
normalization are caller choices; no cells are treated as biological replicates
automatically.

`test=ql` is a quasi-likelihood F test with empirical Bayes shrinkage (Lund et
al. 2012), following glmGamPoi's recipe (Ahlmann-Eltze and Huber 2020):
1. Each feature is fitted with a Cox-Reid adjusted ML dispersion.
2. Dispersions are smoothed against the mean fitted count by a weighted running
   median over max(10% of features, 100).
3. The full and reduced models are refitted at the trended dispersion.
4. Each feature's QL dispersion (1 + m·phi) / (1 + m·phi_trend) is shrunk
   toward a scaled F prior (Smyth 2004). With at least 100 features, the
   prior's scale follows a natural spline in log mean.
5. F = deviance difference / (tested df × shrunken QL dispersion) is referred
   to F(tested df, prior df + residual df).

It needs `family=nb`, `dispersion=ml`, no weights, no ridge penalty, no robust
covariance, and at least two features with valid dispersion estimates. The
table then reports the trended theta in `theta`, standard errors scaled by the
shrunken QL dispersion, and extra columns `theta_feature`, `mean_fitted`,
`ql_dispersion`, `ql_prior`, `ql_posterior` and `df_residual`. The run record
adds `ql_df_prior`, `ql_abundance_trend`, `ql_trend_window` and
`ql_dispersion_features`. Boundary fits, which have no ordinary LR, get `NA`
p-values as with `test=lr`.

In a mock-comparison null (all 70 four-versus-four splits of eight Kang donors,
1,000 genes), `ql` rejected 4.0%, 0.73% and 0.10% of null genes at p <= 0.05,
0.01 and 0.001. edgeR's QL test rejected 4.7%, 0.88% and 0.11%, and
glmGamPoi's 3.9%, 0.72% and 0.11%. Per-gene ML LR rejected 7.4%, 1.6% and
0.19%, and Wald 9.6%, 3.3% and 0.90%. On the real stimulation contrast its BH
(0.05) discoveries match glmGamPoi and edgeR QL with Jaccard 0.97
(`validation/QL_TEST.json`). With few replicates, treat `lr` and `wald` p-values
as anticonservative.

`dose-trend` uses the same regression testing interface; put continuous dose in
the tested design column. For `family=gaussian`, coefficients are mean differences.

`da-lr` uses peak counts as predictors and `group` (a numeric 0/1 vector file)
as response. `design` contains the intercept and nuisance predictors, including
log depth. It compares each augmented model to the common reduced model by LR.
Supply the same count transformation as the comparator. Coefficients are log
odds ratios; the fold-change fields are missing by design.

### Hypotheses: contrasts and joint tests

`pseudobulk-de`, `dose-trend` and `bulk-de` accept at most one of these keys:
- `coefficient`: one zero-based design column (the default is the last);
- `coefficients`: a JSON array of columns, tested jointly, e.g. `[1, 2, 3]`;
- `contrast`: a JSON array with one weight per design column, e.g.
  `[0, -1, 1, 0, 0]` for the second minus the first non-intercept column.

LR, QL and bulk LRT tests take any of them; the Wald test takes one coefficient
or one contrast. The table's `coefficient` and `standard_error` (and bulk-de's
`log2FoldChange` and `lfcSE`) then hold the contrast's estimate and standard
error. For joint tests they are `NA`, and the statistic has as many degrees of
freedom as there are columns. `coefficient_boundary` is 1 when a tested estimate
has no finite value. The run record stores the tested rows as `hypothesis`, for
example `[[0, -1, 1, 0, 0]]`. Fold-change shrinkage takes a single coefficient
only.

In C++, `Hypothesis::coefficient`, `Hypothesis::coefficients` and
`Hypothesis::contrast` build hypotheses (`hypothesis.hpp`) for
`regression_tests`, `quasi_likelihood_tests` and `BulkOptions::hypothesis`.
`reduced_design` gives the model under the hypothesis. `tested_design` gives
the reparametrized design whose column `column` is the contrast; for a one-row
hypothesis, `FeatureTest::fit` holds the fit of that design.

`regression_tests` and `likelihood_ratio` require genuinely nested designs.
The API verifies common responses, weights, offsets, family, parameter counts
and lack of ridge, but nesting of the supplied columns is the caller's contract.
Ordinary chi-square LR inference is not offered for penalized fits.

## Score scans

`scan` fits each response's null model once and tests many predictors against
it with the efficient score statistic. This is the engine's general pattern;
the perturbation workflow below is one configuration of it.

| Key | Meaning |
|---|---|
| `predictors` | MatrixMarket, **predictors × observations**, like `responses`. A predictor whose stored values are all 0/1 is binary; any other is continuous |
| `pairs` | Optional TSV with header `response`, `predictor` and optionally `family` (zero-based indices). Without it every response is tested against every predictor |
| `calibration` | `permutation` (default), `asymptotic` or `crt` |
| `null_model` | `direct` (default): the configured GLM. `sceptre`: Poisson coefficients with a conditional NB theta (NB ML only; resampling only) |
| `strata` | Optional vector file of nonnegative integer strata; permutation shuffles within them |
| `budgets`, `approximation`, `fit_check`, `side`, `seed` | As for `sceptre`: staged draw budgets (default 499, 4999, 19999), the skew-normal tail (`skew_normal` or `no_approximation`), `both`/`left`/`right`, and the RNG seed |
| `screen` | Empirical screen of the first budget (default 0, off; see `sceptre`) |
| `effect_fits` | `significant` (default), `all` or `none`: full fits with the predictor added |
| `alpha`, `adjustment` | Rejection level for significant effect fits and discovery counts (default 0.1); `BH` (default) or `bonferroni` within each `family` |
| `min_nonzero` | Binary predictors: nonzero responses required among the 1s and among the 0s (default 0) |
| `crt_draw`, `parallel_draws`, `parallel_pairs`, `warm_start_effects` | As for `sceptre` |

Calibration:
- **`asymptotic`**: a normal tail of the score z.
- **`permutation`**: permutes a binary predictor's 1s (a uniform subset of the
  same size, within strata) or a continuous predictor's values (within strata).
- **`crt`**: redraws a binary predictor from its logistic regression on the
  design. It is not available for continuous predictors.

Resampled p-values use the SCEPTRE-style staged budgets and skew-normal tail.
Draw k of pair i uses a counter RNG keyed by (seed, i, k), so the output does
not depend on `threads`.

The output has one row per pair, in pair order, with these columns:
- `response`, `predictor`, `family`, `status`;
- `statistic` (score z), `p_value`, `adjusted_p`;
- `estimate` and `standard_error`: the one-step score estimate of the
  predictor's coefficient, u/I with SE sqrt(scale/I), for the direct null model;
- `effect` and `effect_se`: the Wald estimate from a full fit with the
  predictor added;
- `draws` and `approximation`.

Statuses are `ok`, `null_<fit status>`, `failed_pair_qc`, `invalid_score`,
`propensity_<fit status>`, `ok_effect_boundary` and `ok_effect_fit_failed`.
The run record adds `null_model`, `calibration` and `effect_fits`.

Configured like `sceptre` (union-of-guides predictors, the same pair order,
`null_model=sceptre`, `calibration=crt`), a scan reproduces the complement-control
perturbation results exactly (`validation/SCAN.json`).

## Bulk differential expression

`bulk-de` runs the DESeq2 procedure on sample counts: size factors, Cox-Reid
gene-wise dispersions, a dispersion trend, MAP dispersions under a log-normal
prior, a Wald or LR test, Cook's distance outlier handling and independent
filtering ([METHODS.md](../METHODS.md#bulk-differential-expression),
[BULK_DE.md](BULK_DE.md)). `responses` holds integer counts (genes × samples);
`design` includes the intercept, and `coefficient` is the tested column
(default last; `coefficients` and `contrast` as above). It always fits NB
models with its own dispersions and size
factors, so `dispersion`, `cox_reid`, `ridge`, `robust`, `offset` and `weights`
are rejected.

| Key | Meaning |
|---|---|
| `test` | `wald` (default) or `lrt` (against the design under the hypothesis) |
| `size_factors` | `ratio` (default; genes without zeros) or `poscounts` |
| `size_factor_values` | Optional vector file of size factors to use as given |
| `trend` | `parametric` (default; falls back to `local` if it fails), `local` or `mean` |
| `min_mean` | DESeq2's `minmu`: the floor on fitted counts in its fits (default 0.5); genes with a fitted count below it are fitted with the floor ([METHODS.md](../METHODS.md#bulk-differential-expression)) |
| `keep_minimum_start` | Keep the 1e-8 floor for genes whose starting dispersion is at it, as DESeq2's output does (default true) |
| `outlier_sd` | Dispersion outliers, above the trend by this many residual SDs, keep their gene-wise value (default 2) |
| `cooks_filter`, `cooks_cutoff` | Flag Cook's outliers (default true); cutoff (default 0: the 0.99 quantile of F(p, m − p)) |
| `legacy_cooks` | Compute Cook's distances with DESeq2's robust dispersion as implemented in PyDESeq2 (MIT; Muzellec et al. 2023), which reproduces DESeq2's outlier flags (default true from 1.2.0; false: RapidGLM's own estimator, which flags more genes in designs with many small groups). Recorded as `cooks_dispersion` in the run record |
| `min_replicates_for_replace` | Replace outlier counts in design cells with at least this many samples, then refit (default 7; 0 disables) |
| `independent_filtering`, `alpha` | Filter on the mean normalized count (default true); target FDR (default 0.1) |
| `lfc_shrink` | `none` (default) or `normal`: also report the tested fold change under DESeq2's normal prior, as `lfcShrink(type = "normal")` does; tests are unchanged |
| `seed` | Seed of the prior-variance simulation used with three or fewer residual df |

The table has DESeq2's column names: `gene`, `status`, `baseMean`,
`log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `dispGeneEst`, `dispFit`,
`dispMAP`, `dispersion`, `dispOutlier`, `maxCooks` and `replaced`. Statuses:
- `ok`, or `ok_boundary` under `lrt` when another coefficient has no finite estimate;
- `all_zero`;
- `boundary_lrt`: the tested coefficient has no finite estimate and the fit
  with fitted counts floored at `min_mean` failed, so the LR test is reported
  and the fold change is `NA` (from 1.4.0 a group of zeros has a finite fold
  change and a Wald test, as in DESeq2);
- `wald_floored`, `floored_fit_failed`: the floored fit gave no Wald SE, or
  under `lrt` failed;
- `cooks_outlier`: the p-value is removed;
- `fit_<status>`, `reduced_<status>` and `wald_<status>` for failed fits.

`padj` is `NA` for genes removed by the filter. With `lfc_shrink=normal` the
table adds `log2FoldChangeShrunk` (posterior mode) and `lfcSEShrunk`
(posterior SD). The run record adds `test`, `size_factors`, `trend`,
`asymptotic_dispersion` and `extra_poisson` (the parametric trend a0 + a1 /
mean), `log_residual_variance`, `prior_variance`, `cooks_cutoff`,
`filter_threshold`, `filter_theta`, `replaced_genes` and `lfc_prior_variance`
(log2 scale, one per design column; empty without shrinkage).
`.filter.tsv` lists the filter's quantiles, rejections and LOWESS fit. These
keys are rejected by other commands.

## Quasi-likelihood differential expression (edgeR's procedure)

`edger-ql` runs the default quasi-likelihood pipeline of edgeR 4 on sample
counts: filtering by expression, TMM normalization, one NB dispersion, quasi-
dispersions from adjusted unit deviances, their empirical Bayes moderation
against average log CPM, and quasi-likelihood F tests
([METHODS.md](../METHODS.md#quasi-likelihood-differential-expression)).
`responses` holds counts (genes × samples) and `design` includes the intercept;
`coefficient`, `coefficients` and `contrast` choose the hypothesis as above
(default: the last column). Offsets come from the normalized library sizes, so
`offset`, `weights`, `family`, `covariates`, `formula` and `units` are
rejected.

| Key | Meaning |
|---|---|
| `filter` | Filter by expression as edgeR's `filterByExpr` (default true) |
| `min_count`, `min_total_count`, `large_n`, `min_prop` | The filter's settings (defaults 10, 15, 10, 0.7) |
| `normalization` | `tmm` (default) or `none` (factors of 1) |
| `dispersion` | The NB dispersion to use; default: estimated, as the common Cox-Reid dispersion of the most abundant genes |
| `prior_count` | Prior count of the reported log fold changes (default 0.125) |

The table has one row per gene with edgeR's column names where there is one:
`gene`, `status` (`ok`, `filtered`, `fit_failed`), `total`, `AveLogCPM`,
`logFC` (log2, from the counts plus prior counts), `coefficient` (the same on
the natural-log scale), `unshrunk_coefficient` (maximum likelihood), `F`,
`PValue`, `FDR` (Benjamini-Hochberg over the kept genes), `deviance`,
`deviance_adj`, `df_residual_adj`, `s2`, `s2_prior`, `s2_post` and `df_total`.
Filtered genes have missing values. `.samples.tsv` lists the library sizes
before and after filtering, the normalization factors and the effective library
sizes. The run record adds `kept`, `minimum_group` (the filter's minimum number
of samples), `dispersion`, `top_proportion` and `top_genes` (the genes the
dispersion was estimated from), `average_ql_dispersion` (the scale the
dispersion is divided by for the fits), `df_prior` and `discoveries_fdr_0_1`.

## Perturbation workflow

`rapidglm sceptre config.json` adds these inputs:

- `responses`: for this command the counts are read straight into a compact
  count matrix (`rapidglm::CountMatrix`): each gene's nonzero counts, ascending
  by cell, as eight-byte entries (4-byte cell, 4-byte count) in one buffer, with
  the offset where each gene starts. The values must be non-negative integers
  below 2^32, one entry per line. The text is parsed once, in parallel on
  `threads` threads; a file ordered by cell, as writers produce it, needs no
  sorting (other orders are sorted gene by gene). The input fingerprint for
  the run record is computed on its own thread meanwhile.
- `assignments`: MatrixMarket **guides × cells**, containing fixed binary calls.
- `guides`: TSV header `id<TAB>target<TAB>non_targeting`, one row per guide in
  assignment order; `non_targeting` is 0 or 1. WT/no-call cells are never relabeled NT.
- `pairs`: TSV header `response<TAB>target<TAB>kind`, with zero-based response
  index and kind `discovery`, `negative`, or `positive`. Negative-control
  targets must contain explicitly NT guides. Supply individual NT targets
  or predeclared NT guide groups if testing them against other NT cells. A pair
  may be listed once within a kind (`stream_keys` below).
- `targets` (from 1.3.0), instead of or beside `pairs`: TSV header
  `target<TAB>kind`, kind `discovery` or `negative`. Every target is then tested
  against every response, or against the responses of `grid_responses` (a text
  file with one zero-based response index per line), without a list of the
  pairs. Only the pairs that pass pair QC get a row in the output table; the
  others are tallied in `OUTPUT.tally.tsv`, one row per target and kind:
  `requested`, `passing`, `few_treatment` (too few treated cells with a nonzero
  count) and `few_control` (enough of those, too few control cells with one).
  The rows are those of listing the same pairs. `pairs` may then hold further
  pairs, such as positive controls, but none of the grid's in its kind.
- `response_ids` (from 1.3.0, optional): a text file with one ID per response
  (gene or peak names, say), which identifies each response in the stream keys.

**Stream keys** (`stream_keys`, from 1.3.0). With `identity`, the default, a
pair's draws depend only on `seed`, its response and its target: its row is the
same wherever it is listed, whatever else is listed, and however the run is
split into tiles or separate runs. The response is identified by its ID in
`response_ids`, or by its index when that file is not given; the index
identifies it only while the matrix keeps its rows. `position` keys the draws by
the pair's place in the list, as releases before 1.3.0 did, and reproduces
their tables; a pair's p-value then changes with the order and content of the
list (METHODS.md, Resampling).

**Adjustment family** (`adjustment_family`, from 1.3.0). Adjusted p-values are computed
within each kind of pair. With `tested`, the default, the family is the pairs
that pass pair QC, as in SCEPTRE; a pair that fails pair QC has no adjusted
p-value. With `listed`, as before 1.3.0, it is every listed pair, the failed
ones counting with p = 1. `family_size_discovery`, `family_size_negative` and
`family_size_positive` give a family's size when it is larger than the pairs at
hand, for a run that lists only some of them. Effect fits of the significant
pairs follow the family. The run record stores the keys, the family, and the
pairs listed, passing pair QC and in each family (METHODS.md, Tests and
effects).

Parameters: `moi` (`low`/`high`), `control_group` (`nt_cells`/`complement`),
`resampling_mechanism` (`permutations`/`crt`), `integration`
(`union`/`singleton`/`bonferroni`), `side` (`left`/`right`/`both`),
`approximation` (`skew_normal`/`no_approximation`), `seed`, increasing cumulative
`budgets` (default [499,4999,19999]), `screen` (default 0.02: the first budget's
empirical p-value is final unless at most 0.02, as in SCEPTRE; 0 fits the skew
normal from the first budget), `adjustment` (`BH`/`bonferroni`), `alpha`
(default 0.1). Empirical mode always draws the final budget. Approximate mode
tries a moment-fitted skew normal at each budget after the screen, then falls
back to empirical. `fit_check` (`strict`/`ks`, default `strict`) says when the
fit is accepted. `ks` accepts on the fit's overall discrepancy alone, as
releases before 1.0.0 did and as SCEPTRE's outputs are reproduced with. That
test says little about the far tail, where the p-value is read: on the
Gasperini panel it accepted fits whose p-values were 8 to several thousand
times below the tail counted over millions of draws, all for pairs with
positive statistics. `strict` also requires that the draws in the tail where
the statistic lies are no heavier than the fit says, and it accepts a p-value
that the current budget cannot check (fewer than half a draw expected beyond
the statistic) only on the last budget (METHODS.md, Resampling;
[SKEW_NORMAL_CHECK.json](../validation/SKEW_NORMAL_CHECK.json)). A pair whose
fit fails there reports the empirical p-value of the last budget, which with
19,999 draws is at least 1e-4.

`confirm_draws` (with `fit_check: "strict"`; default ten times the last
budget, 199,999 for the default budgets) confirms the smallest p-values. The
last budget cannot check a p-value below about 5e-5 (19,999 draws), and a
mildly heavy tail can pass its check there. A pair whose fit is sound on the
last budget but whose p-value is beyond that budget's reach therefore draws on
to `confirm_draws` in all; the fit made on those draws is reported if it is
sound, and their empirical p-value (at least 1e-5 with 199,999) otherwise. A
fit that is rejected on the last budget ends there, so only a few dozen pairs
per hundred targets draw the confirmation. `confirm_draws: 0` turns it off and
gives the results of 1.0.0; the run record states the number in effect. Skew-normal tail probabilities are clamped to [0, 1]; far in
a tail the CDF can round to a tiny negative value. SCEPTRE does not document
its staging. The screen reproduces what its outputs show; the random draws
themselves differ, so borderline pairs can differ within Monte Carlo noise
([SCEPTRE_STAGING.md](SCEPTRE_STAGING.md)).

`crt_draw` defaults to `thinned`, which draws the same independent Bernoulli
labels as `direct` from a different random stream in O(expected treated
cells) instead of O(cells) per draw. `direct` reproduces CRT p-values from
releases before this option. `hazard` (opt-in) draws the same independent
Bernoulli labels again, from a third stream, with one random gap per treated
cell and no rejection, the gaps coming from an AES-based generator: generating
a draw takes less than half the time of `thinned` (METHODS.md, Resampling).
P-values differ from `thinned`'s within Monte Carlo error, as between two
seeds. `hazard` needs the processor's AES instructions (x86 since about 2010,
Apple and server Arm processors); on a processor without them it draws with
the `thinned` sampler instead and the program says so. The run record states
the sampler asked for (`crt_draw`) and the one used (`crt_draw_used`), so
results of `hazard` depend on which kind of processor ran them. Permutation
tests are unaffected.

`parallel_draws` defaults true. With OpenMP 4.5 or later and `threads > 1`,
draws are scheduled in batches of up to 256 across the existing worker team,
even for a single response. Thinned CRT draws within a batch are generated and
scored together (METHODS.md, Resampling), with results identical to drawing
them one at a time. There is no nested thread pool. Set false to compare with
serial draws inside each response batch. Both settings use the same RNG
streams and produce identical scientific output; scalar builds run serially.

`parallel_pairs` defaults true and schedules independent pairs across the same
worker team after their shared response null is fitted. With both options and
`threads > 1`, the workflow runs a ring of active pairs that all threads draw
for (METHODS.md, Resampling). `sampler_cache_mb` bounds the
memory for CRT samplers shared by all pairs of a target with complement
controls, 12 bytes per cell per target. By default every target keeps its
sampler, up to half of the machine's memory. A target beyond the budget
rebuilds its sampler for every pair (about 4 ms at 200,000 cells), which is
slow: the program then prints a warning, and the run record states how many
targets were kept and rebuilt (`samplers`). A run with `checkpoint_dir` and no
tile sizes avoids this by splitting the targets into blocks that fit (see
below). Results do not depend on either setting.

`reuse_draws` (default false; CRT with complement controls, i.e. `moi: high`)
draws each target's CRT resamples once and scores every gene against them: the
random stream of a test is keyed by its target instead of by the pair. Draw
generation then happens once per target rather than once per pair, which on
the Gasperini panel makes the workflow about five times faster again. P-values
differ from the default's (they come from different resamples), and the
p-values of tests that share a target are computed from the same resamples, so
their errors are correlated across genes; each test's calibration is
unchanged. Use it for routine discovery; statistics to be reported need the
default's independent draws.

`shared_screen` (default false; same requirement) shares only the first-stage
screen. Each pair is first screened with its target's shared first-stage draws
(one stream per target). A pair whose shared empirical p-value is above
`shared_screen_p` (default 0.05) ends there with that p-value. Every other pair
is then tested from its own draws exactly as in the default, so its p-value is
the default's: every p-value below `shared_screen_p`, and so in practice every
discovery, is unchanged (adjusted p-values are computed over all pairs). A pair that the default would have continued past its first stage
can end at the shared screen; a threshold above the default's screen (0.02)
keeps that rare, and pairs with small p-values pass it with near certainty.
Only p-values above `shared_screen_p` from the first stage come from shared
draws. Use one of `reuse_draws` and `shared_screen`.

`sequential_screen` (default false; any MOI) ends each screen (the empirical
`screen`, and the shared screen) as soon as its outcome is certain: once so many
draws are at least as extreme as the observed statistic, on each side for
two-sided tests, that the screen's p-value must exceed its threshold whatever
the remaining draws. It is checked every 32 draws. The decision to continue is
the one the full screen makes, so every pair that continues has the default's
record, including every p-value at or below the screen. A pair that ends early
reports the Besag-Clifford sequential p-value (two-sided: twice the smaller
side's), which is valid, raised to at least the smallest p-value the full screen
ends with; `draws` is then the number used. Most null pairs end after a few
dozen draws instead of the full first budget.

`early_stop_p` (default 0, off; any MOI) also ends the budgets after the
screen early. A pair that has passed the screen stops drawing as soon as it is
certain that its empirical p-value over the whole budget would be above
`early_stop_p`: once so many of its draws so far, the screen's included, are at
least as extreme as the observed statistic (on each side for two-sided tests).
It is checked every 32 draws. A pair whose empirical p-value over the budget is
at most `early_stop_p` is never stopped and has the same record as without the
option. A pair that stops reports the Besag-Clifford sequential p-value, as
with `sequential_screen`, in place of the skew-normal one; it is valid, above
`early_stop_p`, and coarse (with the default budgets and 0.002 it rests on 5
extreme draws on the deciding side, a relative standard error near 45%, much as
a p-value near the screen from the first budget), and `draws` is the number
used. Set it well above every p-value that can be a discovery. Adjusted
p-values are computed from the p-values the pairs report, so in a family whose
Benjamini-Hochberg cut-off is above `early_stop_p` (a small family of positive
controls, say) a stopped pair can be a discovery, with its sequential p-value.
And the rule counts draws, so a pair can stop although its skew-normal p-value
without the option would be below `early_stop_p`: with 0.002 on 100 targets x
7,418 genes of the Gasperini panel that held for 243 of the 13,492 stopped
pairs, the smallest at 1.2e-4, against a discovery cut-off of 3e-6
(`validation/CRT_BATCHING.json`). It can be used with or without
`sequential_screen` and `shared_screen`.

`draw_cache_mb` (default 8192) bounds the memory for stored draws, about 4 bytes
per treated cell per draw per target and stage (the first stage only with
`shared_screen`); stages over the budget are drawn again from the same stream,
so results do not depend on it. The run record states `reuse_draws`,
`shared_screen`, `sequential_screen` and `early_stop_p`.

### Tiles and checkpoints (`sceptre`)

`checkpoint_dir` splits the run into tiles of target contexts × responses and
writes each finished tile there. Rerunning the same command resumes from the
tiles already written, so a stopped or crashed run loses at most the tile in
progress. The merged tables are byte-identical to a run without tiles, at any
tile size and thread count.

- `tile_targets`, `tile_responses`: tile size, in target contexts and
  responses. Defaults: all targets by 256 responses, or 64 targets by all
  responses with `reuse_draws`, whose stored draws then cover one tile's targets
  only. When neither is given and the CRT samplers of all targets do not fit
  `sampler_cache_mb`, the targets are split into blocks that do fit, each
  across all responses; later phases and tile tasks take the tile size from
  `plan.txt`, whatever their own budget.
- `phase`: `all` (default) runs every missing tile and merges. For separate
  workers (for example a job array), run `prepare` once, then `tile` with
  `tile` set to 0 ... count−1 in any order and on any machine that sees the
  directory, then `merge`. `prepare` prints the tile count; `plan.txt` records it.
- With complement controls, `prepare` fits each response's null model once
  (`nulls.bin`); tiles rebuild each response's score state from its fitted
  coefficients instead of refitting it. The rebuild is checked to reproduce the
  fit exactly, and a response for which it does not keeps its full fit.
- `plan.txt` fingerprints the inputs and the settings that change results
  (not `threads`, scheduling or cache sizes); a directory holding another
  run is refused. Tiles are written through a temporary file and renamed, so a
  tile file is always complete. The run record lists the tile count and the
  tiles computed by that process.
- `inputs.tsv` records each input's SHA-1 with its size and modification time.
  A later phase or tile task takes the fingerprint from there when the file's
  size and time are unchanged, instead of reading the whole file again to hash
  it; a changed file is hashed again. Files modified in the last two seconds
  are not recorded. Propensity models,
draws and effect models remain specific to each pair. Both task options use
one pool with at most `threads` workers.

`effect_fits` selects which tested pairs also get a full GLM effect fit (adjusted
log2FC and SE): `significant` (default) fits pairs whose adjusted p is at most
`alpha` in their testing family, after guide integration; `all` fits every
tested pair; `none` fits none. Skipped rows' `log2fc`/`log2fc_se` are
`NA` with status `ok`. The choice never changes p-values, statistics or the
mean-ratio fields, and a fitted row has exactly the estimate `all` would give.
Use `all` when effect sizes of non-significant pairs are needed, for example
for concordance plots. The run record stores the setting.

`warm_start_effects` defaults true. For regular fits, the full effect model
starts from the null coefficients with a zero perturbation coefficient; NB ML
also starts from the null dispersion, avoiding another Poisson pilot. It still
optimizes the full model and computes its covariance. Failed or boundary warm
starts fall back to the fresh-fit path; NB moment-dispersion fits use that path
directly. Set false for a fresh-fit numerical comparison. Warm starts may
change the final low-order digits of log2FC and its SE, while resampling results
are unchanged.

Cell QC: `cell_qc` defaults true, `qc_lower`/`qc_upper` default 0.01/0.99,
`max_mito` defaults 0.2; optional vector paths `qc_total`, `qc_detected`,
`qc_mito` provide full-assay measurements. Otherwise totals/detection are
computed from supplied responses, and mitochondrial filtering is unavailable.
For a response subset, supply full-assay QC measurements. `min_nonzero` defaults
7 in each arm. `assignment_qc` defaults true and removes zero/multiple-guide
cells in low MOI. Set false only for an explicitly frozen, already curated
cell universe; this is recorded in parameters.
High MOI requires complement controls. It defaults to CRT; low MOI defaults to
NT controls and permutations. The resampling mechanism can be explicitly changed.

`null_model` selects the response-null fitting and score recipe:

- `sceptre` (default): Poisson coefficients and fitted means followed by
  conditional NB ML dispersion, holding those means fixed, as the SCEPTRE book
  documents. Scores use SCEPTRE's raw numerator and covariate-adjusted
  information denominator. This requires `family=nb`, `dispersion=ml`, and
  `ridge=0`.
- `direct`: the configured GLM fit; NB with `dispersion=ml` jointly
  estimates its coefficients and dispersion. Scores use nuisance projection.
  Set it explicitly for Poisson/Gaussian responses, fixed or moment dispersion,
  or ridge fits; the default rejects those combinations rather than switching
  silently. It was the default before 2026-09-29.

The setting is available only on the `sceptre` command. It applies to both shared
complement nulls and treatment-plus-NT nulls. It changes the null mean-ratio
fields and potentially the resampling p-values. Full effect-model log2FC and
SE still come from direct GLM fits, optionally warm-started from this null.
The RNG and draw budgets remain RapidGLM's, and the staging follows `screen`;
this flag does not claim p-value equivalence with SCEPTRE, whose draws differ.

Every perturbation `.run.json` explicitly records `null_model`,
`null_fit_method`, `score_numerator`, `effect_fit_method`, `resampling_policy`
(`empirical_screen` or `rapidglm`) and `screen`, including when the default mode
was omitted from the input.
The scientific TSV columns retain their existing definitions and schema.
See [SCEPTRE model validation](SCEPTRE_MODEL.md) for measured agreement and scope.

Outputs include test status, cell counts, raw/adjusted p, adjusted GLM log2FC
and SE, and the separately defined mean-ratio fields in `METHODS.md`.
Bonferroni integration reports the selected guide's effect; it is not a pooled
estimate. The `.qq.tsv` contains negative-control order statistics and pointwise
95% Beta bands. Discovery, calibration and power are separate testing families.

## C++ integration (provisional API)

```cpp
#include <rapidglm/rapidglm.hpp>
rapidglm::Matrix X(4, 1, 1.0);
rapidglm::FitOptions options;
options.family = rapidglm::Family::poisson;
auto null = rapidglm::fit(X, {1, 0, 3, 2}, options);
if (null.ok()) {
    rapidglm::ScoreWorkspace score(X, null);
    rapidglm::Vector g{0, 0, 1, 1};
    rapidglm::ResampleOptions draws;
    draws.budgets = {999};
    draws.approximate = false;
    auto result = rapidglm::resample(score, g,
        rapidglm::permutation_sampler(g), draws);
}
```

`SparseCSC` uses **observations × features**, with sorted unique row indices.
`score_scan` (`scan.hpp`; `ScanOptions`, `ScanPair`, `ScanResult`), `resample_permutation`
(any covariate, within strata), `fit_batch`, `regression_tests`, `quasi_likelihood_tests`,
`differential_accessibility`, `sceptre`, `aggregate_pseudobulk`,
`estimate_shared_theta` and `estimate_dispersion_trend` are header-only
engine/workflow entry points; `running_median` and `fit_variance_prior`
(`shrinkage.hpp`) expose the QL test's trend and empirical Bayes prior.
`bulk_de` and `estimate_size_factors` (`bulk.hpp`; `BulkOptions`, `BulkGene`,
`BulkResult`) take counts as a samples × genes `SparseCSC`.
`BulkOptions::lfc_prior_variance` supplies the fold-change prior variances
instead of estimating them.
`FitOptions::cox_reid` requests the Cox-Reid adjusted dispersion, and
`FitResult::cox_reid` reports whether the final theta used it. `Sampler` is a callback receiving
`CounterRng&` and filling sorted unique treated indices; built-in permutation
supports strata and CRT accepts known/fitted propensities. Degenerate generated
binary covariates have score zero and are retained rather than redrawn.
The low-level `ResampleOptions::parallel_draws` defaults false because custom
samplers may carry mutable state. Opt in only for a sampler safe to call
concurrently, and call `resample` within an existing OpenMP team (for example,
inside `parallel_for`). `SceptreOptions::parallel_draws` defaults true because
the workflow constructs the built-in concurrent-safe samplers.
`FitOptions::nb_warm_start` defaults false. It requires NB ML and a nonempty
`start`, and uses `theta` as an initial dispersion without a Poisson pilot.
`fit_effect_model` applies the workflow's guarded warm-start/fallback policy.
`SceptreOptions::null_model` defaults to `FitMethod::sceptre_null`; select
`FitMethod::direct` for joint GLM fits and non-NB families. `fit_sceptre_null`
returns a tagged `FitResult` whose coefficients are Poisson estimates and
whose theta is conditional NB ML. It omits coefficient covariance/SEs;
ordinary Wald and LR inference reject it. `ScoreWorkspace::statistic` and
`resample` support its documented statistic, while `ScoreWorkspace::test`
declines an ordinary Gaussian-tail p-value for this mode.

`FitOptions::solver` defaults to `Solver::automatic` (Cholesky of X'WX while
well conditioned, Householder QR otherwise); `Solver::qr` forces QR. With
`FitOptions::singular_ok`, aliased columns (relative residual below
`alias_tolerance`, default 1e-11 as in R's glm) are dropped and reported as NaN
coefficients; `FitResult::aliased`, `rank` and `used_qr` describe the fit.
Without it an aliased design returns `rank_deficient`. Cholesky factors, their
solves and inverses and the QR back substitution are computed in-house unless
`RAPIDGLM_LAPACK_SOLVES` is defined (see the build notes above); Householder QR,
condition estimates and SVD always use LAPACK. All LAPACK calls go
through `rapidglm/lapack.hpp`; defining `RAPIDGLM_R_LAPACK` routes them to the
LAPACK that R links (used by the R package in `bindings/r`). R and Python
drop-in interfaces are described in [GLM_COMPARISON.md](GLM_COMPARISON.md).

`ResamplingWorkspace(design, null_fit)` prepares reusable score state for
multiple pairs sharing a null. Its `resample` overload shares that state
across callers. `Sampler` accepts the existing two-argument callbacks;
built-in samplers also accept a third `SamplerScratch&` argument to reuse
buffers. Binary score evaluation has a corresponding `BinaryScoreScratch&`
overload. Scratch must be owned by each executing task. The workflow manages
these lifetimes automatically.

Install with `cmake --install build --prefix PREFIX`. An external CMake project
uses `find_package(rapidglm CONFIG REQUIRED)` and links `rapidglm::rapidglm`.
Run `scripts/check_external.sh` for a complete example. The C++ header API is
provisional and may change between minor versions; the CLI configuration and
the C interface are the stable interfaces.
