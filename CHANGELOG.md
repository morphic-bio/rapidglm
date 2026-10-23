# Changelog

## 1.4.1 (2026-10-22)

- Return an empty weighted cross-product matrix immediately when the design
  has zero columns. This avoids undefined behavior from forming row pointers
  into empty storage, including in a Cook's replacement refit when every
  coefficient is at a boundary. Add a regression covering six row counts.
- The focused solver and bulk suites pass ASan/UBSan with OpenMP (29 cases,
  4,553 assertions). All 28 checked nonempty-design cross-products are
  bit-identical to 1.4.0; this patch changes no fitting or resampling rule.
- Benchmark tables and raw receipts remain the completed **1.4.0** run on
  `f0ecfbd` (`v1.4.0-dev.4`). They have not been remeasured for 1.4.1.
  See the [patch validation](validation/receipts/v1.4.1/README.md) and
  [benchmark provenance](validation/receipts/v1.4.0/README.md).

## 1.4.0 (2026-10-22)

The library is the code of `v1.4.0-dev.4`; this release adds the receipts and
documentation measured with it. Every stage in the complete 18-stage release
run was measured afresh. One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.

- **Bulk and pseudobulk DE:** fitted-count flooring (default `min_mean: 0.5`),
  rough/moments starting dispersions, bounded fallback with covariate-unit
  invariance, and independent filtering following the documented and
  black-box-validated rules. Replaced-count base means and failed-refit
  p-values are corrected. Low-count and boundary genes can change results
  relative to 1.3.0; shared trends and priors can also move other genes.
- **Floored-fit interfaces:** C `rapidglm_nb_floored_fits` and Python
  `capi.nb_floored_fits`; the PyDESeq2 inference backend uses native fits and
  delegates unsupported/fallback cases to DefaultInference. MIT PyDESeq2
  ports carry attribution in DEPENDENCIES.md.
- **Quasi-likelihood DE:** `edger-ql` / C++ `edger_ql` implements filtering,
  TMM, common Cox-Reid dispersion, adjusted deviances, moderation and QL
  testing following edgeR 4's publications, documentation and public API
  outputs. Near-Poisson deviances and fit failure handling are stable.
  Inputs must be nonnegative integer counts. Known reference-fit differences
  and small-panel moderation limits are documented in METHODS.md.
- **Cis-QTL mapping:** Gaussian, Poisson and NB score tests, empirical
  permutation calibration and Beta approximation, with PLINK/BED input,
  checked outputs, stable ties, per-gene singular statuses and failed-draw
  accounting. Only phenotype chromosomes are decoded; MAF filtering is
  in-place. Validation covers chromosome 22, not a genome-wide run.
- **Faster Gaussian permutations:** project genotypes once per gene and
  score batches of 512 responses with packed AVX2/FMA or portable kernels.
  On the final matched 10,000-permutation panel, RapidGLM takes
  117.19 s versus tensorQTL's 151.53 s
  (1.29×), with unchanged empirical p-values
  and lead variants relative to dev.3. See
  [QTL_GAUSSIAN_SPEED.md](docs/QTL_GAUSSIAN_SPEED.md).
- **Numerical and input handling:** tiled weighted cross products and
  leverages, vectorized predictor/mean updates, quicker MatrixMarket input
  and SHA-accelerated fingerprints where available; binding, matrix input
  and QTL file validation fixes. Perturbation streams and acceptance-rule
  definitions are unchanged from 1.3.0.
- **ARM64:** cross-build support, selected QEMU checks, and double-precision
  KS CDF checks with unchanged integer streams. Floating-point results can
  differ by architecture; numerical limits are retained explicitly.
- **Validation:** native and portable C++ (172 cases each), C99, CLI (28
  cases on each build), Python/R, Rust, selected ARM64 and container suites
  pass. Final agreements and timing scopes are in the
  [release collection](validation/receipts/v1.4.0/README.md).

## 1.3.0 (2026-10-10)

The perturbation workflow (`sceptre`). Bulk DE, single fits, batch fits and
score scans are unchanged. The library is the code of the development release
`v1.3.0-dev.1`; this release adds the receipts and documents measured with it.

**For a user of 1.2.0 who changes no option:** every resampled p-value of
`sceptre` changes, because a pair's draws are now keyed by the pair itself and
no longer by its place in the list; the new values are as valid as the old and
differ from them as two seeds differ, so discoveries can change at the margin.
Adjusted p-values are now computed over the pairs that pass pair QC, so in a
list that includes pairs failing pair QC they are smaller, more pairs can be
discoveries, and the failing pairs report a missing adjusted p-value instead of
1. A list that holds the same response and target twice within a kind is now
an error. `stream_keys: "position"` with `adjustment_family: "listed"`
reproduces the tables of 1.2.0 byte for byte (checked on six runs).

- **Streams keyed by identity** (`stream_keys`, default `identity`; C++
  `StreamKeys`). A pair's random stream depends only on the seed, its response
  and its target (its guide for per-guide tests), so its record is the same
  wherever it is listed, whatever else is listed, and however the run is
  divided among threads, tiles, shards or separate runs. `response_ids` names
  the responses (gene or peak names); without it a response is identified by
  its column. Reused draws and the shared screen are keyed by the target. The
  keys are FNV-1a hashes of the names, turned into a stream by Philox
  (`name_key`, `keyed_stream`; [METHODS.md](METHODS.md), Resampling).
  `position` keeps the keys of earlier releases.
- **The adjustment family** (`adjustment_family`, default `tested`; C++
  `AdjustmentFamily`): the pairs that pass pair QC, as in SCEPTRE, whose 101
  calls on the Gasperini benchmark are BH over the 9,718 pairs passing its
  pairwise QC and not over the 10,000 listed. `listed` is the family of earlier
  releases. `family_size_*` gives a family's size to a run that lists only some
  of its pairs. The run record stores the pairs listed, passing pair QC and in
  each family.
- **No work for a pair that cannot be tested.** Pair QC is decided first, from
  the target's cells and the response's nonzero cells. A response's null model
  is fitted only when one of its pairs passes (complement controls), a pair's
  cells are gathered only when it passes (non-targeting controls), and the
  start saved for a later effect fit is kept only for tested pairs (352 bytes
  less for every other listed pair). Records are unchanged: with position keys
  the tables are those of 1.2.0.
- **Targets and responses without a list** (`sceptre_grid`, `sceptre_pair_qc`;
  `targets` and `grid_responses` in the CLI; `rapidglm_sceptre_grid` in C;
  `sceptre_grid` in Python and R). Every named target is tested against every
  named response. One pass over each response's nonzero cells decides pair QC
  for all targets; the pairs that pass are tested and get records, and the
  others are tallied by target and kind (too few treated cells with a nonzero
  count, or too few control cells with one). The records are those of listing
  the same pairs.
- Measured on a synthetic screen in which 5% of the requested pairs pass pair
  QC (20,000 cells; 12.4 million peak pairs; 12 threads on efficiency cores,
  with `sequential_screen` and `early_stop_p`;
  [validation/UNTESTABLE_PAIRS.json](validation/UNTESTABLE_PAIRS.json)):

  | Run | 1.2.0, every pair listed | 1.3.0, every pair listed | 1.3.0, targets and responses named |
  |---|---|---|---|
  | High MOI | 202 s, 11.9 GB | 192 s, 7.9 GB | 167 s, 1.1 GB |
  | Low MOI | 87 s, 11.8 GB | 59 s, 7.5 GB | 37 s, 0.7 GB |

  Listing only the pairs that pass takes the time of the request. A listed pair
  that is not tested now costs about 0.57 kB of memory in the CLI (0.92 kB in
  1.2.0), most of it the list itself and the pair's row; in a request it costs
  none. What remains of a list's time is reading it and writing a row for
  every pair.
- Engine: `ResampleOptions::prescreen_seed` (a shared prescreen under its own
  seed), `adjust_pvalues(p, bonferroni, family_size)`, and
  `SceptreResult::null_fits`.
- Every benchmark was measured again with this version
  ([docs/BENCHMARK.md](docs/BENCHMARK.md)). The perturbation workflow is 4.5
  times faster than SCEPTRE on 50,000 cells (one thread) and 4.4 times on
  207,324 cells (eight threads), about a tenth faster than 1.2.0. With the new
  draws the full screen gives 96 discoveries against SCEPTRE's 101 at the
  benchmark's seed (Jaccard 0.89; 101 and 0.92 with the draws of 1.2.0). Over
  six seeds the Jaccard index is 0.89 to 0.96, and 0.90 to 0.96 between two
  RapidGLM seeds
  ([validation/PERTURBATION_SEEDS.json](validation/PERTURBATION_SEEDS.json),
  `scripts/validate_perturbation_seeds.py`).
- The SCEPTRE-agreement receipt takes the mean discovery count of its gate G2
  over 200 seeds (`--gate-seeds`) instead of ten. Over ten seeds the mean's
  standard error is a third of the gate's margin on the public high-MOI panel,
  and the outcome turned on the seeds: 12.4 (pass) with the draws of 1.2.0 and
  13.2 (fail) with those of 1.3.0, for the same ten seeds. Over 200 seeds both
  give 12.3 against SCEPTRE's 11, within the margin of 1.65
  ([docs/SCEPTRE_STAGING.md](docs/SCEPTRE_STAGING.md)).

## 1.2.0 (2026-10-07)

The library is the code of the development release `v1.2.0-dev.1`; this
release adds the receipts and documents measured with it.

- `legacy_cooks` is now the default in bulk differential expression (CLI, C,
  Python, R and the PyDESeq2-style API): Cook's distances use DESeq2's robust
  dispersion as implemented in PyDESeq2 (MIT; Muzellec et al. 2023), with
  PyDESeq2's flagging rule. Default discoveries then agree with DESeq2's on GTEx
  thyroid at Jaccard 0.997 (128 samples) and 1.000 (647 samples), against 0.58
  and 0.90 with RapidGLM's own estimator, which sets aside more genes in
  designs with many small groups
  ([docs/BULK_DE.md](docs/BULK_DE.md#legacy-cooks-distances-2026-10-01)).
  `legacy_cooks: false` selects that estimator and gives the tables of 1.1.0.
  Genes flagged or replaced as Cook's outliers, and so some p-values and
  adjusted p-values, change for designs with replicated cells.
- In the Rust binding, `BulkOptions::legacy_cooks` is an `Option<bool>`; `None`
  keeps the default.
- Fits are faster, with estimates that agree with 1.1.0's to rounding
  ([METHODS.md](METHODS.md)). A fixed-dispersion NB fit of 647 observations and
  12 columns takes 132 to 151 microseconds on one core, against 245; `bulk-de`
  on GTEx thyroid (647 samples, one thread) takes 21 s against 49 s, and 0.63 s
  against 0.98 s on airway. The time went into work that the estimates did not
  need:
  - A further Newton step is no longer taken only to confirm convergence: the
    last step's factor predicts it (the chord step), which is then applied as a
    refinement. Fits report one iteration fewer.
  - Likelihood terms that depend on the response alone used three long double
    `lgamma` calls per count above 64; they now take one logarithm each, by
    Stirling's series in a form without cancellation, and are more accurate
    near the Poisson limit. The polygamma sums of dispersion estimation
    likewise.
  - The deviance that scales the convergence criterion, and the signature that
    identifies a fit's response, are computed in a fraction of the time.
  - Designs with at most half their entries nonzero (designs of factors) form
    their cross-products from the nonzero entries of each row.
  - Fits of many responses to one design share its cross-product, rank check
    and row groups (`FitOptions::design`, from `prepare_design`); `fit_batch`,
    `bulk_de`, score scans and the perturbation workflow do so.
  - The rank check of the positive-count design is skipped when no count is
    zero.
- Results change in the last digits. Where a dispersion estimate lies on a flat
  likelihood (genes close to Poisson), it can settle at another value. On
  airway four gene-wise estimates move. The prior variance, a grid choice,
  moves with them from 0.697 to 0.700, and the independent-filtering threshold
  from 3.86 to 4.50 (DESeq2's value): 5,164 genes are discoveries against
  5,142 (31 gained, 9 lost). On GTEx thyroid and the other four validation
  datasets every column agrees to 3e-10 (relative), with either setting of
  `legacy_cooks`.
- Every benchmark was measured again with this version, each thread on its own
  efficiency core of one workstation
  ([docs/BENCHMARK.md](docs/BENCHMARK.md#host)). Bulk DE is 6.1 to 13.9 times
  faster than DESeq2 on one thread (airway to 647 thyroid samples), and the
  perturbation workflow 3.9 to 4.5 times faster than SCEPTRE. Against Ceres
  Solver, fixed-dispersion fits are 2.1 to 2.7 times faster than
  Levenberg–Marquardt at equal accuracy
  ([docs/SOLVERS.md](docs/SOLVERS.md)); the solver benchmark also times a plain
  `fit` per gene (last argument `shared`, `plain` or `both`).
- New receipt [validation/CRT_SCALING.json](validation/CRT_SCALING.json): the
  conditional randomization test on panels of the Gasperini screen, with
  `fit_check: "ks"` and with the defaults
  (`scripts/prepare_crt_panels.py`). `validation/COOKS_LEGACY.json` and
  `validation/COOKS_THYROID.json` are written by `scripts/validate_cooks.py`,
  and the SCEPTRE agreement receipt repeats its seeds under the defaults.
- Benchmark harnesses can confine runs to listed CPUs and record the share of
  those CPUs that other processes used (`--cpus`).
- The release workflow publishes a pre-release for a tag with a suffix, such as
  `v1.2.0-dev.1`.

## 1.1.0 (2026-10-06)

- Small skew-normal p-values are confirmed on more draws by default
  (`confirm_draws`; `ResampleOptions::confirm`; C, Python and R interfaces).
  With the strict fit check, a fit that is sound on the last budget but whose
  p-value that budget cannot check (fewer than half a draw expected beyond the
  statistic: a p-value below about 5e-5 with 19,999 draws) is no longer
  reported from there. The pair draws on to ten times the last budget in all,
  199,999 for the default budgets, and the fit made on those draws is reported
  if it is sound, the empirical p-value otherwise. A fit rejected on the last
  budget ends there as before, so only candidates for very small p-values draw
  more. `confirm_draws` sets another number; 0 gives the results of 1.0.0. On
  100 targets x 7,418 genes of the Gasperini screen, 55 of 721,266 pairs draw
  the confirmation (14% more draws); 50 fits are confirmed and 5 rejected,
  among them the discovery that direct counts contradict, and the 13
  discoveries that remain are the ones a four-budget run to 199,999 finds
  ([validation/SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)).
  P-values of pairs that draw the confirmation change; a tiled plan includes
  the setting, so a checkpoint written by 1.0.0 is not resumed.
- A stage that can end early is drawn by several threads once it has gone
  1,024 draws without ending (one more chunk in flight per further 1,024
  draws; at most about 3% of draws generated beyond a stage's end). Long
  stages no longer run on one thread: that panel takes 289 s with the
  confirmation draws, against 292 s without them in 1.0.0. Results do not
  depend on it.

## 1.0.0 (2026-10-06)

- Tiled, checkpointed `sceptre` runs (`checkpoint_dir`, `tile_targets`,
  `tile_responses`, `phase`, `tile`): a run resumes from its finished tiles,
  tiles can run as separate jobs, and the merged output is byte-identical to a
  single run. Response null fits are made once and rebuilt from their
  coefficients in each tile (checked to be exact). Library: `SceptreShard`,
  `sceptre_layout()`, `sceptre_nulls()`.
- Faster CRT resampling with identical results
  ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)):
  - thinned CRT draws are generated and scored in batches, over per-cell score
    records stored together, so each batch reads the sampler and the records
    once instead of once per draw;
  - with several threads, all threads work on a ring of active pairs, taking
    chunks of draws, and a target's CRT sampler is shared by its pairs
    (`sampler_cache_mb`);
  - Philox blocks are computed eight at a time, with AVX2 when available.
  On the 207,324-cell Gasperini panel (high MOI, 32 threads) output is
  byte-identical to 0.2.0, and the 100-target x 7,418-gene point takes 1,080 s
  instead of 4,755 s.
- `reuse_draws` (opt-in, high MOI): each target's CRT draws are generated once
  and scored against every gene, with random streams keyed by target. P-values
  differ from the default's and share resamples within a target, so it is
  meant for routine discovery; reported statistics need the default's
  independent draws. On the 100-target x 7,418-gene point it takes about 5
  minutes instead of 18.
- `shared_screen` (opt-in, high MOI) with `shared_screen_p` (default 0.05):
  pairs are screened with their target's shared first-stage draws; pairs above
  the threshold end there, and all others are tested from their own draws
  exactly as by default, so their p-values are the default's (every p-value
  below the threshold and, in practice, every discovery). A pair can end at the shared screen that
  the default would have continued. Engine: `ResampleOptions::prescreen`.
  Batch callables of `staged_null` now take the stream (`test`) as their first
  argument, and stored draws are used only for their own stream.
- Cholesky factorizations, their solves and inverses, and the QR back
  substitution and (R'R)^-1 are now computed in-house at every size (before:
  LAPACK above 16 columns and for the QR triangular solves). Results no longer
  depend on the BLAS's `dtrsm`: the pthread OpenBLAS 0.3.20 of Ubuntu 22.04
  returned wrong triangular solves from order 32 up with its Cooperlake kernel
  (Intel Xeon 6975P-C). Fits that take the QR path change in the last digits
  (relative 1e-13 on airway and GTEx thyroid; calls unchanged), as can fits
  with more than 16 columns. The new CMake option `RAPIDGLM_LAPACK_SOLVES`
  restores the LAPACK path, with a configure-time warning; run records state
  `lapack_solves` (`validation/LAPACK_SOLVES.json`).
- `sequential_screen` (opt-in): screens end as soon as their outcome is
  certain, so the decision to continue, and every continuing pair's record, are
  the default's; pairs that end early report a Besag-Clifford sequential
  p-value from the draws used. Engine: `ResampleOptions::sequential_screen`.
  Such screens are drawn one chunk at a time with about one active pair per
  thread, so no draws are generated beyond a screen's end (before, threads
  drew ahead: on 50 targets x 1,000 genes at 16 threads `sequential_screen`
  took 94 s, now 53 s, against 113 s without it; output unchanged).
- The skew-normal fit to the null statistics is accepted by a stricter check
  by default (`fit_check`, `strict` or `ks`; `ResampleOptions::fit_check`; in
  the C, Python and R interfaces too). The earlier check, the fit's overall KS
  discrepancy, says little about the far tail where the p-value is read: on
  100 targets x 7,418 genes of the Gasperini screen, counting the tail over 1
  to 4 million draws contradicted the accepted fit for seven pairs, all with
  positive statistics, whose p-values were 8 to several thousand times too
  small; six had been discoveries. `strict` also requires that the draws in the
  tail on the statistic's side are no heavier than the fit says (draws beyond
  the statistic, the most extreme draw, counts beyond fitted quantiles, the
  fourth moment), and accepts a p-value beyond the reach of the current budget
  only on the last one. It accepts five of the seven in at most 2% of repeats
  and the two mildest in 42-44%, and 17 of the 19 other pairs in at least 96%;
  draws rise by 5% and that panel's discoveries go from 20 to 15
  ([validation/SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)).
  `fit_check: "ks"` gives the earlier results exactly. P-values of pairs whose
  fit is now rejected, or confirmed on the last budget, change.
- The C interface accepts `RAPIDGLM_CRT_HAZARD`; Python and R accept
  `crt_draw="hazard"`.
- `early_stop_p` (opt-in, default 0): the budgets after the screen also end
  early. A pair that has passed the screen stops drawing once its empirical
  p-value over the whole budget must exceed `early_stop_p`, and reports a
  Besag-Clifford sequential p-value from the draws used, in place of the
  skew-normal one; a pair whose empirical p-value over the budget is at most
  `early_stop_p` keeps its record. Engine: `ResampleOptions::early_stop`;
  `NullStages` now treats screens and later budgets as sequential stages, with
  chunks that end at the next check. With `sequential_screen` and
  `early_stop_p` 0.002 at 16 threads, 100 targets x 7,418 genes of the
  Gasperini panel take 362 s instead of 542 s (65.9M draws instead of 123.1M):
  13,492 of the 15,057 pairs past the screen stop, after 1,314 draws on
  average, and the 22 discoveries and their records are unchanged
  ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)).
- `crt_draw: "hazard"` (opt-in; `CrtDraw::hazard`): CRT labels from the points
  of a unit-rate Poisson process on a line where observation i owns an interval
  of length -log(1 - p_i). The law is the same independent Bernoulli(p_i) as
  `thinned` and `direct`, from a third random stream: one exponential gap
  (256-layer ziggurat) per treated cell, no rejection and no levels to merge,
  in the same 12 bytes per cell. The gaps' random words come from ARS with 10
  rounds, an AES-based counter generator (`AesCounterRng`) keyed per draw from
  the draw's Philox stream and computed with the processor's AES instructions
  (x86 AES-NI, Arm), at about a third of the cost of a Philox word. With
  `sequential_screen` and `early_stop_p` 0.002 at 16 threads, 100 targets x
  7,418 genes of the Gasperini panel take 279 s instead of 362 s. On a
  processor without AES instructions `hazard` draws with the thinned sampler;
  run records state `crt_draw_used`. Tail probabilities of seven Gasperini
  pairs agree with `thinned`'s within Monte Carlo error over 2 to 8 million
  draws each ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)).
- CI job and script for arm64 (`scripts/test_arm64.sh`): the unit tests are
  cross-compiled and the generator and sampler tests run under QEMU user
  emulation, which covers the Arm AES path.
- A sequential screen whose threshold the first budget cannot resolve (every
  test ends at it) now ends once a draw is as extreme as the statistic on the
  deciding side, with that draw's sequential p-value. Before, it ended at the
  first check, and reported 1 for two-sided tests and NaN for one-sided ones.
- `sceptre` reads its responses into a compact count matrix
  (`CountMatrix`: per gene, 4-byte cell and 4-byte count for each nonzero, in
  one buffer with per-gene offsets) instead of the double CSC, with a parallel
  single-pass MatrixMarket loader (`read_market_counts`): each thread parses a
  piece of the text into per-gene lists, which are then copied in file order
  into the buffer, so a cell-ordered file needs no sort. On the Gasperini
  matrix (6 GB, 445M nonzeros, 16 threads) loading and set-up fall from about
  80 s to 8 s and the load's peak memory from 23 GB to 12 GB; outputs are
  unchanged. `sceptre`, `sceptre_layout` and `sceptre_nulls` accept either
  container. The responses' SHA-1 is computed on its own thread.
- A tiled run records its inputs' SHA-1 fingerprints in the checkpoint
  directory (`inputs.tsv`) with each file's size and modification time; later
  phases and tile tasks take a fingerprint from there when the file's size and
  time are unchanged instead of hashing the file again (about 16 s per task
  for the 6 GB Gasperini matrix: a tile task's start falls from 25 s to 9 s).
  A file modified in the last two seconds is not recorded.
- CRT samplers: 12 bytes per cell (4-byte cell index) instead of 16, with the
  same draws. Every target now keeps its sampler by default
  (`SceptreOptions::sampler_cache_bytes` unlimited; the CLI's
  `sampler_cache_mb` defaults to half of the machine's memory, was 4096): with
  the old default only a third of the full Gasperini panel's 3,776 targets
  were kept and the rest rebuilt their sampler for every pair. With a third
  kept, 50 targets x 1,000 genes takes 132 s instead of 111 s. Targets that
  do not fit are counted (`SceptreResult::samplers_rebuilt`, run record
  `samplers`) and the CLI warns; a tiled run without tile sizes then splits the
  targets into blocks that fit, recorded in the plan.
- The perturbation workflow returns freed response state to the system every
  256 responses (glibc `malloc_trim`). Blocks freed by another thread than the
  one that allocated them stayed in per-thread arenas, and resident memory grew
  by a few megabytes per response over a run.
- Rust binding ([bindings/rust](bindings/rust)) over the C interface, with a
  port of PyDESeq2 0.5.2's workflow (MIT, with attribution) whose per-gene fits
  run in RapidGLM. It makes the same calls as PyDESeq2 with the RapidGLM back
  end on airway and GTEx thyroid at 128 and 647 samples (Jaccard 0.984 at 32
  samples, from Cook's distances at leverage-1 samples;
  `validation/RUST_DESEQ.json`). CI runs `cargo test`, clippy and rustfmt.

## 0.2.0 (2026-10-01)

- `legacy_cooks` (CLI, C, Python, R and the PyDESeq2-style API) computes Cook's
  distances with DESeq2's robust dispersion as implemented in PyDESeq2 (MIT;
  Muzellec et al. 2023), with PyDESeq2's flagging rule. Discoveries then match
  DESeq2's on GTEx thyroid: Jaccard 0.997 and 1.000 at 128 and 647 samples,
  against 0.58 and 0.90 by default ([docs/BULK_DE.md](docs/BULK_DE.md)).
  Attribution and licence: DEPENDENCIES.md.

## 0.1.0 (2026-10-01)

First release.

- **Engine.**
  - Gaussian, Poisson, binomial and negative binomial GLMs for many responses
    that share a design, over sparse column matrices, with OpenMP across
    responses and identical results at any thread count.
  - Safeguarded Newton solvers: observed information for the NB log link,
    likelihood step halving, Cholesky with a QR fallback, and R-compatible
    aliasing. Diverging coefficients are detected. NB dispersions are
    estimated jointly or by Cox–Reid adjusted profile likelihood.
  - Wald, likelihood ratio, score and quasi-likelihood F tests of linear
    hypotheses Cβ = 0.
  - Score scans calibrated by asymptotic score, permutation or conditional
    randomization, with staged budgets and a skew-normal tail.
- **Applications.**
  - `bulk_de`: the DESeq2 procedure, with normal-prior fold-change shrinkage.
  - `sceptre`: the SCEPTRE method for single-cell CRISPR screens.
  - Pseudobulk quasi-likelihood DE, differential abundance and dose trends.
- **Interfaces.**
  - CLI with JSON configurations.
  - Stable C ABI (`librapidglm_c`).
  - Python package: `bulk_de`, `sceptre`, `scan`, `fit_batch`; a
    PyDESeq2-style `DeseqDataSet`/`DeseqStats`; `RapidGLMInference` for
    PyDESeq2; a statsmodels drop-in.
  - R package: `bulk_de()`, `sceptre()`, and `glm()`/`glm.nb()` drop-ins.
  - A Dockerfile that builds everything and runs all test suites.
- **Validation.** Receipts under `validation/` for agreement with DESeq2,
  SCEPTRE, R `glm`/`glm.nb`, statsmodels, glmGamPoi and Ceres Solver, and for
  the application benchmark ([docs/BENCHMARK.md](docs/BENCHMARK.md)).
- **Known differences.**
  - DESeq2's robust dispersion for Cook's distances is unspecified; RapidGLM's
    substitute flags more genes in designs with many small groups.
  - The local dispersion trend uses a running median instead of locfit.
  - Fold-change shrinkage uses the normal prior only.
