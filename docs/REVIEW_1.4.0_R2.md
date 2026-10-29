# RapidGLM 1.4.0-dev.3 review response

Branch: `fix/review-1.4.0-r2`. Engine changes through `a979b96`; primary documentation reconciled in `b27e684`. Publication recipes and scope corrections are in separate commits, merged after timing finished. The comparison recipe used for the final live check is `239cfe9`. No push, tag or release was made. The request to name the candidate dev.3 is reflected in the engine and Python versions; the explicit instruction to wait for review before tagging takes precedence over the later tag instruction.

All reference behaviour was established from published material and public black-box calls. No R/Bioconductor implementation source was read. PyDESeq2, jaxQTL and tensorQTL are the permitted sources; implementation provenance is recorded in DEPENDENCIES.md. Only public validation data was used. Untimed checks ran on CPUs 0–7 without the timing lock. Timings use efficiency cores and the shared lock, with two quiet host/core samples immediately before each measured command.

## Findings B1–B11

| Finding | Disposition and commit | Regression/evidence |
|---|---|---|
| B1 | Fixed `e07a837`: stable `log1p` deviance. The moments routine already took the exact Poisson branch below 1e-10; that branch was verified instead of replaced. | `near_poisson_deviance_and_null_calibration`: phi 1e-12,1e-16,1e-20 vs zero, production interpolation factors, nominal null rejection rate; CLI pins `nb_panel` F/P against a public edgeR fixture. |
| B2 | Fixed `63876eb`, `43d5f67`: projected bounded BFGS fallback; covariate/ridge scaling; restart invalid boundary warm starts; Python delegates all native optimizer fallback cases to DefaultInference. PyDESeq2 credited. | `floored_fits_optimize_bounds_and_covariate_units`; `successful_cooks_replacement_refits_a_floored_gene`; Python optimizer metadata and bound/interior fallback tests. |
| B3 | Fixed `7228a66`; docs reconciled `b27e684`: at most ten rejections selects first theta; residual RMS uses positive-rejection quantiles. | `independent_filter_few_rejections_and_positive_residual_rms`; `FILTER_RULES.json`: five saved reference selections and HNRNPR theta=0, threshold agreement. |
| B4 | Fixed `fa3f842`: discard first-pass statistics before replacement refit, recompute base mean, p=1 on failed/all-zero refit; BH uses the resulting filtering family. | `cooks_replacement_all_zero_discards_first_test`; successful replacement test pins new baseMean=129/14. |
| B5 | Fixed `98e23df`, `43d5f67`: try floored full/reduced fits after failed ML, expose failed floor status. | `floored_lrt_and_replacement_have_explicit_status`; successful replacement test. |
| B6 | Fixed `03bd257`: singular observed score information becomes a gene status, singular permutation information counts as failure. | `qtl_singular_information_is_a_gene_status`; `qtl_failure_status_paths`. |
| B7 | Fixed `fa29b0a`: clean empty-fit error; trial dispersion invalid if any member of the fixed gene set fails. | `edger_all_fits_failed_is_a_clean_error`. |
| B8 | Fixed `f09113f`: lengths checked at R and C layers; coefficient ranges validated in R/Python. | R applications invalid-vector tests; Python `test_bad_hypothesis_dimensions`; C99 caller. |
| B9 | Fixed `3571fd8`: strict unsigned MatrixMarket size parsing and overflow checks in both loaders; safe handling of signed-minimum integer tokens. | CLI `test_malformed_input` negative/overflow cases. |
| B10 | Fixed `af6ebbe`: QTL streams checked on open and after writing. | CLI `test_qtl_output_stream_errors`, including missing directory and failing output sink. |
| B11 | Fixed by explicit restriction `23e81d7`: edgeR workflow rejects fractional counts before likelihood/normalization. | `edger_rejects_fractional_counts`; METHODS and CLI docs state integer-only input, without implying that edgeR has the same restriction. |

## Review section 3

| Item | Disposition | Evidence |
|---|---|---|
| Degenerate QTL Beta maxima | Fixed `ad3e805`: relative variance guard; failed k,n,ncp all NaN. | `qtl_degenerate_beta_reports_missing_parameters`. |
| Gaussian library sizes | Rejected `b2675fd`. | CLI input validation test. |
| PLINK fam fields/bed length/sample order/n mod 4 | Fixed `b2675fd`; malformed rows and mismatched binary dimensions fail clearly. | CLI 81-sample fixture, independently reordered fam/genotype columns, malformed/truncated/extra bed tests. |
| Point BED | Supported `b2675fd`; coordinate convention and point exception documented. | CLI point-record fixture; docs/CLI.md and METHODS. |
| Stable lead tie | Fixed `73c5e4c` with stable input order. | Window/nominal fixture includes actual cis variants and equal positions. |
| Failed-permutation output | Fixed `73c5e4c`: explicit count, consistent observed lead/statistics. | `qtl_failure_status_paths`. |
| QTL memory | Improved `8a8e689`: decode BED chromosomes only; compact MAF in place; release nominal vectors when flattened. | CLI chromosome-selection regression. Allocated matrix capacity may remain until the scan finishes; not a genome-wide memory benchmark. |
| edgeR dispersion objective | Fixed under B7. | Fixed gene set or invalid objective. |
| LOWESS anchors | Documented choice `a81cd6d`: range/200 even for <=200 unique values. | METHODS, published edgeR specification. |
| 10,000-row dispersion subset | Documented omission `a81cd6d`: all selected genes used. | METHODS explains potential difference for very large feature sets. |
| Row sum>=5, LR clamp, finite-only BH | Documented `a81cd6d`. | METHODS. |
| Prior-df endpoint | Documented `a81cd6d`: interior numerical solution ~8134.8433 versus bound 8134.84478. | METHODS. |
| edgeR misspelled keys/prior_count | Fixed `a81cd6d`. | CLI unknown-key/prior_count tests and `edger_prior_count_is_validated`. |
| Dense input/transposed memory | Documented remaining cost `a81cd6d`. | METHODS. |
| API version before symbol lookup | Fixed `79687f6`; feature API is 2. | Python mock old-library test. |
| C99 new entry points | Added `ba53a0b`. | C99 application tests. |
| Replogle ok_* tested statuses | Fixed `5e39145`. | `tests/test_replogle.py`. |
| Common negative pairs | Fixed `5e39145`. | All tools score the same negative-pair intersection; regression fixture. |
| strict agreement versus ks timing | Fixed `5e39145` and receipt harness: distinct strict/ks agreement reports paired with their own timings. | New Replogle receipts. |

## Review sections 4–5 and runbook 2–4

| Item | Disposition/evidence |
|---|---|
| CHANGELOG default PyDESeq2, unchanged other genes, pilot range, exact shortcut, linear loader claims | Corrected `bb901a1`; historic receipt numbers explicitly labelled, shared trend/prior effects and sorting described. |
| Floored example/reference equality | Corrected `bb901a1`; `31597ef` pins public DESeq2 Wald LFC/SE; `a979b96` pins the LR result. |
| LR likelihood at returned means | Extra fix `a979b96`: optimize with floor, evaluate reported LR at returned unfloored means, as the public DESeq2 call does. Test LR=92.02103007895477 and p=8.576025085782601e-22. Previous floor-evaluated LR was 86.10276345412666. |
| Moments start | `98458cf`: public PyDESeq2 helper reproduction, rough .66873563 versus moments -.37142857; native minimum starts at 1e-8. |
| Core-type factor | `e0ce2cf`, `8405453`: rounded 1.7–2.1 throughout current docs and public host notes. Original v1.4.0 controls span 1.656–2.087 across 22 method runs (`CORE_FACTOR.json`); the historical v1.3.0 SOLVERS receipt spans 1.663–2.081. Numeric timing measurements are unchanged. |
| Exact streams versus floating statistics | README corrected `bb901a1`: integer streams exact, statistics may round differently. |
| Rust note | Corrected `7c37bd6`: port remains unfloored; singleton design-cell count renamed without calling it leverage one. |
| QTL documentation | `bb901a1`: method, config, output columns, start convention and counted A1 allele. |
| edgeR specification | Published as docs/EDGER_QL_SPEC.md, references inspected for documentation/papers/black-box provenance. |
| Stale QTL/edgeR outputs | Historic figures labelled; new comparison uses new outputs and freshly rerun references. |
| edgeR near-zero phi, all-fits-failed, fractional counts, factors | Added B1/B7/B11 tests. |
| edgeR shortcut/full, halving, fallback, scale>1, joint contrast | `ba53a0b`: `edger_newton_shortcut_halving_fallback_scale_and_joint_tests`. |
| edgeR validation assertions/reference inputs | `31597ef`: checked-in public black-box fixture; CI compares it. Validation asserts identical filtering, norm factors and bounded log10-p discrepancy. |
| Gaussian batched/unbatched and meaningful windows | `73c5e4c`: Gaussian paths agree; window fixture now includes tested variants. |
| QTL null_*, permutations_failed, ok_beta_failed | `4d9803f`: explicit failure-path regression. |
| Forced portable x86 | `8c8631f`: RAPIDGLM_FORCE_PORTABLE disables accelerated kernel/integer/SHA paths, full suite in CI. |
| Whole ARM64 / CI | `8c8631f`, `934aab5`, `9e9a2ca`, `1f4d324`: all 169 cases, CLI28/C99 and ten application comparisons complete under QEMU. Functional classifications match; the unchanged numerical relative criterion is not met universally. Scheduled/manual/tag full CI and short push filter; forced portable x86 full CI. |
| ARM64 KS runtime | `9e9a2ca`: ARM-only double-precision Phi-2T identity for KS discrepancies; inferential tails retain promoted routines. Grid regression 7 shapes x201 points within 2e-14. |
| Paper scripts/receipts | `65cae3d`: committed pending Cook/glmGamPoi scripts and receipts and seed options; glmGamPoi release receipt preserved as requested. |
| Published QTL recipe | `fc8323e`: use the checked-in q-value wrapper, honor `EQTL_WORK`, and name published scripts in the guide; isolated-work regression passes and runs in CI. Live new-output equivalence check is recorded with the final QTL receipt. |
| Geuvadis preparation | `65cae3d`, `e0ce2cf`: eqtl_ scripts published; docs/GEUVADIS.md records data/terms/checksums/tool versions. |
| 50k seeds | `be7f0b0`: benchmark seed table byte-identical to v1.3.0, SHA256 34ceeef29a2df3ddb5e2f478a6bb06ae1099a2c4787ab22d14f71789c691c78b; two historical six-seed receipts validated for dev.3. |
| Cook flags | New untimed and formal COOKS_FLAGS receipts, per-gene reasons retained. |
| 130 positive controls | PANEL_COUNTS.json records input checksum and 130 positive pairs. |
| Historical precision claims | Fresh `THYROID_PRECISION.json`: size-factor, dispersion, LFC and p-value median relative differences are 1.688e-15, 1.478e-4, 6.117e-5 and 1.157e-4; the all-four 1e-4 bound fails. `COOKS_INVERSION.json` checks live public R assays: maximum relative within-gene range 4.684e-15 on 28,212 genes with all means >=0.5, supporting scoped 1e-12 consistency; 3.027e-9 across all 28,486 eligible genes, failing a universal bound. This does not establish equality to the native robust estimator. Published calculation wrappers and docs carry the scope. |
| Timed receipt provenance/quiet host/CPU accounting | `e6adedb`, `d910d69`, `f57eadb`, `de057f9`: isolated output directory, version/producer commit, two per-core/global quiet intervals, measured CPU busy time explicitly includes the command. |
| Equivalent Replogle steps | `5e39145`: reference timings omit lfcShrink and classic edgeR; outputs match the minimal native workflow. All target reference fits inspected through public functions. |
| Matching QTL permutation counts | New Gaussian 1,000 and 10,000, NB 1,000, tensor 10,000, FastQTL adaptive 1,000–10,000, jaxNB 1,000 and SPA/ACAT reports. |


## Verification

All final engine tests use the code of `a979b96`, with measurement source
`b27e684` (documentation changes only between those commits). Native x86 passes
169 C++ cases / 5,387,714 assertions; forced portable x86 passes 169 /
5,381,705. Both pass all 28 CLI tests and C99. ARM64 under QEMU passes all
169 cases / 5,387,711 assertions, all 28 CLI tests and C99 with CPU `max`;
zero C++ cases are skipped. Python CAPI (10 cases), DDS (4), PyDESeq2 backend (3), statsmodels,
R applications/parity, Rust tests/clippy/fmt, formatting, quiet-run/Replogle
helpers and installed CMake C/C++ consumers pass. `TESTS.json` preserves the
logs; `validation/ARM64_DEV3.json` records toolchains, QEMU and applications.
Remote CI has not run because this branch has not been pushed.

All ten architecture comparisons have identical discoveries and statuses.
Both QTL empirical permutation-p columns match exactly, as do all SC empirical
p-values, draw counts and approximation flags. Fitted-tail p-values have exact
mismatches: low-MOI strict/ks 34/36; high-MOI 23/23. Maximum absolute p
changes are 4.6994e-15 (low) and 7.9797e-16 (high). The original 1e-8 relative
criterion remains unchanged and fails in some numeric columns. Examples:

- A tiny airway gene dispersion is 1.0656577e-8 versus 1.1245183e-8, relative
  difference .0552 despite absolute difference 5.886e-10.
- A boundary edgeR coefficient is 32.1377533 versus 32.1320150, relative
  difference 1.7855e-4. The gene has zero counts in every sample of one group;
  both fits have deviance near 1e-12, along a flat likelihood boundary.
- A fitted SC tail can be zero versus about 7.497e-136. Its empirical p and
  resampling counts are unchanged.

All four x86 forced-portable SC tables match dispatched x86 exactly. An
identical-input tail probe has 24/567 architecture/toolchain mismatches,
maximum absolute 8.2005e-143. The same Boost 1.83 on both architectures retains
these differences; x86 Boost 1.74 and 1.83 match. Fitted-tail evaluation is one
identified source; compiler, libm and long-double effects are not isolated
from each other. The numeric acceptance criterion is unresolved. QEMU elapsed
time is verification runtime, without an ARM hardware performance claim.

## Changes on the validation data

The following are effects of the combined fixes; no per-fix ablation is
claimed. Individual regressions above establish the failure paths and rules.
All five standard bulk and eight edgeR validation datasets retain their old
hit counts and agreement; bulk gates G1-G4 pass, including 70,000 null tests.

| Native bulk CLI dataset | Old hits | New hits | Reference hits | New Jaccard |
|---|---:|---:|---:|---:|
| airway | 4,833 | 4,833 | 4,834 | .986234 |
| thyroid32 | 1,636 | 1,623 | 1,615 | .985285 |
| thyroid128 | 792 | 790 | 792 | .997475 |
| thyroid647 | 7,177 | 7,178 | 7,177 | .999861 |

Thyroid32 now reports 30 floored-fit failures (previously one numerical
failure), leaving 28,648 status-ok genes versus 28,677. Fourteen old discoveries
are lost and one gained: nine now have failed floor fits, one becomes a Cook
outlier and four cross the BH threshold. The gained gene 23378 was formerly
flagged. The filter still selects theta = 0. The explicit failures replace
previous quiet fallbacks; they are retained for review.

Thyroid128 has five explicit floor failures, Cook flags 228 -> 219 and
replacements 69 -> 67. The finite-p family grows 28,541 -> 28,545. Genes 13956 and 23179
retain essentially identical raw p-values, while adjusted p moves
.0999879295 -> .1000019427, crossing .1. Formal Cook comparison is 219 native,
224 reference, 217 shared. The original v1.4.0 values were 228, 224, 222 shared;
the separate v1.3.0 paper values were 216, 224, 210 shared. Nine old native flags
vanish (seven now ok, two failed floor fits), with none added. The seven
reference-only genes comprise five replacement-stage differences and two
without finite native floor-fit distances; two native-only genes reflect
replacement-stage differences.

Thyroid647 has two formerly flagged replacement genes now tested. Gene 6028
adds one discovery; gene 22952 is nonsignificant. Their fitted coefficients
and SEs are unchanged, while base means and Cook distances update. With Cook
filtering disabled, native default and reference discoveries still match
exactly on thyroid128/647 (884/7,180); own-estimator results 866/7,050 are
unchanged. The formal flag cache matches a fresh public DESeq2 run exactly
in baseMean, pvalue, maxCooks and the complete 28,769 × 128 Cook assay.

| Bulk fitted-call seconds, one core | Native R old | Native R new | DESeq2 new | New observed speed ratio |
|---|---:|---:|---:|---:|
| airway | .957 | .956 | 6.108 | 6.39 |
| thyroid32 | 4.374 | 4.895 | 31.469 | 6.43 |
| thyroid128 | 9.519 | 9.854 | 119.666 | 12.14 |
| thyroid647 | 35.434 | 35.241 | 483.237 | 13.71 |

The thyroid128 PyDESeq2 backend costs 19.97 s versus 14.14 s originally on one
core, and 7.53 s versus 6.74 s on eight. Its hit counts remain 787 versus 792 from
PyDESeq2; airway remains 4,663 versus 4,690. This backend does not guarantee
identical default results. Native edgeR CLI one-core seconds are
.2881/1.5676/3.3619/13.1155, versus reference .784/6.263/20.714/91.279.
The observed ratios are 2.72/4.00/6.16/6.96. Native eight-core seconds are
.1666/.5146/1.1196/4.1502; reference edgeR is timed only on one core.

Bulk uses the minimum of three small/two large repeats, with all repeats
preserved. All 198 commands start below one-minute load 3.967. Maximum
selected-core other-work fraction is .128; two selected eight-core minima
have .0526 and .0506. All starts satisfy the quiet predicate, without implying
complete isolation. API timing covers fitted calls; CLI timing covers whole
commands. These are observed timings, without a causal runtime ablation.

## Replogle pilot

Both native SC tables and all three fresh SCEPTRE output tables are
byte-identical to v1.4.0. Strict 10,979 versus 10,972 discoveries, Jaccard .9495;
ks 10,982 versus 10,972, Jaccard .9492. New whole-command times: strict 1,925.90 s,
ks 722.72 s, SCEPTRE 3,273.57 s, versus old 2,018.16/769.97/3,684.40 s.
New ratios are 1.70 and 4.53; strict/ks 2.66.
The original panel's corrected negative comparison has 132 common pairs.

An independent untimed check uses the exact 10,000 public reference-selected
control pairs. Every pair is tested by all three methods. Strict and ks
native tables are byte-identical: 492/10,000 at p<=.05 and 95 at p<=.01,
versus reference 489 and 89. `REPLOGLE_MATCHED_NEGATIVES.json` embeds the recipe,
hashes and scope. This check does not replace the original-panel timings.

All 20 pseudobulk targets are complete. Bulk Jaccard spans .9721–1.0000
and QL .9998–1.0000. Both native workflows report status `ok` for all 8,563
genes of every target. The only changed native discovery set is HNRNPR bulk:
four -> two, losing genes 2167 and 2364 and matching the reference's two.
Its filter changes theta .7755102 -> 0 and threshold 30.53236 -> .8438335;
all other target filter selections and native discovery sets are unchanged.

| Median whole-command seconds per target, one core | Native old | Native new | Reference new | Ratio of new medians |
|---|---:|---:|---:|---:|
| DESeq2 workflow | 35.06 | 35.195 | 198.46 | 5.64 |
| edgeR QL workflow | 4.64 | 4.74 | 30.11 | 6.35 |

The new reference fitted-call/QL-step medians are 193.5995/29.087 s. Old
reference whole-command medians were 236.34/159.345 s and included extra
shrinkage/classic work; the new times use matching minimal workflows and
outputs. Each target/tool has one timed command, without repeated minima.
All 80 pseudobulk commands start at load <=3.99; maximum end load is 5.79.
Raw CPU, memory and start/end load records are retained in `COMPARISON.json`;
no per-core other-work fraction was collected for this pilot.

Public full/reduced fit diagnostics cover all 171,260 target-gene rows.
Maximum reported-versus-at-fitted deviance difference is 1.0232e-12 for both
models. Available full-fit failure flags are all false. Reduced iteration
and failure metadata is absent, so no convergence status is inferred.
`REPLOGLE_FIT_DIAGNOSTICS.json` and recipe commit `824c740` preserve this
aggregate check. Large p-value differences remain visible despite matching
hit sets, including GATA1's maximum absolute log10-p difference 14.7515.

The review's GATA1 gene 2779 discrepancy persists: native F = 797.5486821
versus public reference 872.3274820. The public reduced deviance is
973.4014164, equal to the deviance evaluated at its fitted means; no reduced
iteration/failure metadata is returned. An independent score-equation solve
on the 48-lane reduced design gives 897.5359127185186 at the native working
dispersion, matching the native-implied 897.5359127185184 to 2.27e-13.
At the reference working dispersion, the same solve gives 897.5434334,
75.8579830 below the public returned fit. The discrepancy is a reduced-fit
difference in this example, rather than a reported/at-fitted discrepancy.
`GATA1_FIT_DIAGNOSTIC.json` and `f6337d0` preserve inputs, scope and the
independent calculation; no reference failure flag is inferred.

## QTL

Both 1,000-permutation native tables retain all 580 status-ok genes. Every
common numeric field and every lead variant matches the original v1.4.0
exactly. The added `permutations_failed` column is zero throughout.
`QTL_RESULT_CHANGES.json` records comparisons and table hashes. Gaussian
10,000 also has 580 valid genes. All four fresh reference tables have 580
valid genes, so each agreement comparison below uses 580 common genes.

| Tool / model | Permutations | Old whole-command seconds | New seconds |
|---|---|---:|---:|
| RapidGLM Gaussian | 1,000 | 27.39 | 27.99 |
| RapidGLM Gaussian | 10,000 | — | 256.70 |
| RapidGLM NB score | 1,000 | 1,117.92 | 1,098.42 |
| tensorQTL Gaussian | 10,000 | 160.14 | 144.99 |
| FastQTL Gaussian | adaptive 1,000–10,000 | 1,147.74 | 1,022.68 |
| jaxQTL NB score | 1,000 | 3,257.26 | 3,193.10 |
| jaxQTL NB SPA+ACAT | none | 58.04 | 58.11 |

At the matched Gaussian 10,000 budget, tensorQTL is 1.77 times faster than
native in this measurement. At NB 1,000, native is 2.91 times faster than
jaxQTL. FastQTL's adaptive budget and SPA+ACAT are reported separately; the
1,000-native/10,000-reference comparison is not used as a speed ratio.

Gene calls use beta-approximated permutation p-values and public qvalue defaults
at q<.05; SPA+ACAT uses its own gene-level p-values. These are agreements of
statistical procedures, without requiring identical Monte Carlo draws.

| Native run | Reference | Native / reference eGenes | Shared | Jaccard | P Spearman | Same lead |
|---|---|---:|---:|---:|---:|---:|
| Gaussian 1,000 | FastQTL adaptive | 270 / 267 | 267 | .9889 | .999672 | 1.000 |
| Gaussian 1,000 | tensorQTL 10,000 | 270 / 266 | 266 | .9852 | .999683 | 1.000 |
| Gaussian 10,000 | FastQTL adaptive | 267 / 267 | 267 | 1.0000 | .999913 | 1.000 |
| Gaussian 10,000 | tensorQTL 10,000 | 267 / 266 | 266 | .9963 | .999929 | 1.000 |
| NB 1,000 | jaxQTL NB 1,000 | 168 / 170 | 168 | .9882 | .999557 | .950 |
| NB 1,000 | jaxQTL NB SPA+ACAT | 168 / 194 | 157 | .7659 | .812265 | .902 |

The original v1.4.0 stage had no agreement receipt. `EQTL_COMPARE.json` now
uses the new native outputs and freshly rerun references. The published
wrapper/`EQTL_WORK` recipe was executed on those same files from clean
publication commit `239cfe9`; all reported metrics match the stage comparison
exactly. The wrapper hashes also match. The receipt and `TESTS.json` retain
this live equivalence check separately from the timed engine producer.

Each tool/budget has one whole-command timing on twelve efficiency cores.
All seven readiness summaries have one-minute load <=3.6612; total global/core
quiet waiting is 1,371.31 s, excluded from command seconds. Individual
ten-second samples are not retained. Readiness summaries preserve the final
load, timestamp and wait duration; `host.json` covers the separate initial
CPU-guard wait. Recorded CPU busy seconds include the measured job. The
progress monitor observed host load above 20 during the final long reference;
a quiet start does not establish continuing isolation. The stage timeline is
retained as `status.txt`. GNU time peak RSS is a per-process maximum rather
than aggregate memory for multi-process tools. These are observed timings,
without a claim that all shared-host contention was eliminated.

## Remaining documented limits

In addition to the architecture criterion, the 12-gene/8-sample public edgeR
fixture reports denominator df = 72 versus native about 8140.82. Gene 0 has
F = 1.36757876 versus 1.36719559 and p = .24226256 versus .24615077. The fixture's
absolute .005 p tolerance does not establish identical small-panel tails.
`EDGER_SMALL_PANEL_LIMITATION.json` records all 12 rows. The specification's
previous no-cap observation is limited to its larger calibration panels; no
general cap rule was inferred from one new probe.

Other retained method choices are listed above and in METHODS: dense edgeR
input/transposition, all dispersion genes rather than a 10k subset, LOWESS
anchors, finite-only BH, nonnegative LR, and the prior-df endpoint. QTL memory
is reduced by chromosome selection and in-place MAF compaction, but allocated
matrix capacity can remain until completion; no genome-wide memory claim is
made. Noninteger edgeR counts are explicitly rejected.

## Receipts and review decisions

The complete collection and reproduction notes are in
[validation/receipts/v1.4.0-r2/README.md](../validation/receipts/v1.4.0-r2/README.md).
`COMPARISON.json` includes all 72 bulk timing comparisons, old/new pilot
metrics and raw timing records, native pilot call/status changes, all seven
QTL timings and readiness summaries. `source.json` preserves the clean timed
producer `b27e684` and binary hashes; publication commits do not replace it.
All 26 original receipt checksums still match `BASELINE_RECEIPTS.json`.

Publication commits: `fc8323e` fixes the public qvalue path/environment with
regression and CI coverage; `97ede8c` publishes precision/control/comparison
helpers and scopes six documents; `4aa728c` retains raw timing records;
`f6337d0` publishes the independent GATA1 solve; `8405453` corrects public
core-factor host notes without changing numeric measurements; `824c740`
records all pilot fits and call changes; `9048b0d`/`239cfe9` document timing
collection limits and retain QTL readiness summaries. Final result files and
this report are committed separately. No engine, binding or CMake code changed
after the measurement freeze.

- Perturbation timing choice remains pending: rerun pert50k/pertfull/seeds/crt
  on a quiet host or retain the historical v1.3.0 receipts. None was started.
  The 50k benchmark-seed table is byte-identical to v1.3.0, validating the two
  historical six-seed receipts for the candidate.
- The original full-screen reference/seed stages failed. There is no completed
  new full-screen six-seed receipt to validate; historical evidence is retained
  pending the timing choice. Five remaining full-screen reuse-draws seeds
  require an explicit request.
- Review the result changes, failures, remaining numeric criterion and claims
  before any push, tag or release. No Gasperini all-by-all, full Replogle
  essential screen or genome-wide QTL run was started.
