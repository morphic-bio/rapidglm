# Review receipts for 1.4.0-dev.3

These receipts belong to branch `fix/review-1.4.0-r2`. Timed engine/source
producer: `b27e684a13e5dedbcf5ed4b9b416061a2077a59d`; last inferential change:
`a979b964b20c1bdd1f23ba3378b1c1ecf4bf86d5`. The measured worktree was clean.
`source.json` records executable/shared-library hashes. Publication-only recipe
and documentation commits are listed in the final review report; the final
comparison collection and live wrapper check used recipe commit `239cfe9`; they do not
replace the producer commit of measurements already made.

The original `validation/work/benchmark/receipts/v1.4.0` is unchanged.
The six affected stages all completed successfully; `status.txt` retains the
stage timeline. The published QTL wrapper reproduces every stage comparison
metric exactly on the new outputs.
`BASELINE_RECEIPTS.json` records its 26 JSON checksums. `COMPARISON.json` retains
old/new metrics; `RESULT_CHANGES.json` records changed native CLI calls and
Cook flags, statuses, p-values and table hashes. The combined result changes
are not a per-fix ablation.

| Receipt | Contents |
|---|---|
| `source.json` | Producer, binaries, CPU/quiet-start contract and scope |
| `TESTS.json` | Final x86 native/portable, CLI, C99, bindings, Rust, helper and installed-consumer checks, with logs |
| `BULK_DE.json` | Five datasets, 70,000 null tests and all G1–G4 gates |
| `EDGER_QL.json` | Eight public datasets: filters, factors, dispersions, moderated tests and discoveries |
| `applications_bulk_small.json` | Airway, thyroid32 and thyroid128; three repeats, all interfaces and references |
| `applications_bulk_large.json` | Thyroid647; two repeats, all interfaces and references |
| `COOKS_FLAGS.json` | Thyroid128: 219/224 flags, 217 shared and per-gene reasons |
| `COOKS_LEGACY.json`, `COOKS_THYROID.json` | Default/own estimator and no-Cook-filter comparisons; historical alpha-rob blocks identified separately |
| `COOKS_REFERENCE_CHECK.json` | Cached flag reference versus fresh public DESeq2 outputs: common gene columns and full Cook assay exact |
| `COOKS_INVERSION.json` | Direct public R assays: scoped 1e-12 consistency and failed universal bound; calculation source included |
| `THYROID_PRECISION.json` | Fresh own-estimator median differences; historical all-four 1e-4 bound not met |
| `FILTER_RULES.json` | Saved public selections and HNRNPR theta=0 |
| `REPLOGLE_FIT_DIAGNOSTICS.json` | All twenty public full/reduced fit checks, reported-versus-at-fitted deviances and available failure/iteration metadata |
| `GATA1_FIT_DIAGNOSTIC.json` | Public full/reduced deviances and independent one-factor score solve for the large GATA1 F difference |
| `EDGER_SMALL_PANEL_LIMITATION.json` | Public 12-gene fixture: denominator df and tail differences |
| `replogle_sc_strict.json`, `replogle_sc_ks.json` | Each mode's agreement, timing and shared negative-control intersection |
| `replogle_pseudobulk.json` | Twenty targets; matching minimal native/reference workflows and whole-command times |
| `REPLOGLE_MATCHED_NEGATIVES.json` | Untimed check on the exact 10,000 reference-selected negative pairs; all valid, rejection rates and recipe source |
| `REPLOGLE_IDENTITY.json` | Strict/ks native tables byte-identical to original v1.4.0 |
| `QTL_RESULT_CHANGES.json` | Both 1,000-permutation native models: common numeric fields and leads unchanged; added failed-permutation counts zero |
| `EQTL_COMPARE.json` | New Gaussian/NB outputs against freshly rerun FastQTL, tensorQTL and jaxQTL references |
| `COMPARISON.json` | Before/after validation metrics, native pilot call/status changes and hashes, complete timing records, quality audit and matched-count QTL times |
| `RESULT_CHANGES.json` | Feature-level changes for the four native CLI bulk benchmark datasets |
| `ARM64_DIAGNOSTICS.json` | Worst application rows and numerical scales |
| `ARM64_ROUNDING_DIAGNOSTICS.json` | Four SC x86 forced-portable/dispatched comparisons: exact |
| `ARM64_TAIL_PROBE.json` | Identical-input tail evaluation, same-Boost controls and embedded probe source |
| `PANEL_COUNTS.json` | 130 positive controls and public input checksum |
| `PERTURBATION_IDENTITY.json` | Untimed 50k benchmark-seed table identical to v1.3.0 |
| `CORE_FACTOR.json` | Untimed analysis of the 22 original v1.4.0 solver controls: range 1.656–2.087, rounded 1.7–2.1 |
| `BASELINE_RECEIPTS.json` | Checksums of the untouched original receipts |

The full architecture receipt is [`validation/ARM64.json`](../../ARM64.json).
All 169 C++ cases, 28 CLI tests and C99 pass under QEMU; all ten applications
have identical discoveries and statuses. The unchanged 1e-8 relative numeric
criterion is not met universally. Empirical resampling p-values, draws and
approximation flags match exactly; fitted-tail p-values can differ. This is
verification under emulation, without native ARM performance measurements.

Run the affected stages from the repository root:

```sh
taskset -c 0-7 bash validation/receipts/v1.4.0-r2/run.sh bulkval edgerval bulk cooks replogle eqtl > /tmp/rapidglm-r2-final-stages.log 2>&1
```


After the stages finish, collect the timing/results comparison (the stderr
capture retains the QTL readiness summaries):

```sh
taskset -c 0-7 python3 scripts/collect_review_comparison.py --root "$PWD" --quiet-log /tmp/rapidglm-r2-final-stages.log
taskset -c 0-7 python3 scripts/compare_review_qtl.py --root "$PWD"
taskset -c 0-7 python3 scripts/summarize_replogle_fits.py --pilot validation/work/replogle/k562-essential/pilot-v1.4.0-r2 --output validation/receipts/v1.4.0-r2/REPLOGLE_FIT_DIAGNOSTICS.json
```

Capture the stage command's stderr at the named path when repeating this
collection. The extra matched-negative, precision and GATA1 checks are untimed;
their receipts embed calculation source or identify the published helper and
input hashes. They are additional checks, separate from the timing stages.

The timed wrappers hold `/mnt/pikachu/e2e_bench_20260926/pikachu_timed.lock` and
pin commands to efficiency CPUs 16–31 (one:16; eight:16–23; QTL twelve:16–27).
Each command starts after two ten-second intervals with one-minute load <=4
before/after and every selected CPU's busy fraction <.2. Waits are excluded
from measured seconds. Untimed checks use CPUs 0–7 without the lock.

Bulk seconds are the minimum over the recorded repeats, following the original
harness. API measurements time the fitted call; CLI measurements include the
command. The R-versus-R bulk comparisons use fitted-call times. edgeR is timed
on one thread only. Replogle compares matching whole commands and their minimal
outputs. Old Replogle reference wall times included additional work and are
therefore shown as historical measurements, not as directly comparable costs.
SC and QTL use one timed command per tool/mode. Recorded peak RSS is the
per-process maximum and does not represent aggregate process-tree memory for
multi-process commands. QTL readiness summaries retain the final load, timestamp and quiet-wait
duration; individual ten-second core/load samples are not retained. Its
`host.json` idle-wait field covers the first CPU-guard wait only; the subsequent
global/core quiet wait is recorded separately in `COMPARISON.json`.
QTL times are whole commands on twelve cores: Gaussian fixed 10,000 versus
fixed 10,000, FastQTL adaptive 1,000–10,000 separately, NB fixed 1,000 versus
fixed 1,000, and SPA+ACAT without permutations separately.

`cpus_busy_seconds` includes the measured command. Bulk `others_cpu` estimates
the fraction on selected cores attributable to other work; `contended` uses
threshold .2. All 198 bulk starts have load <3.967. Maximum recorded
`others_cpu` is .128; two selected eight-core minima have .0526/.0506. Retaining
these values avoids treating a quiet start as proof of complete isolation.

Cached public bulk/edgeR/Cook reference tables are reused where the reference
procedure and inputs are unchanged. The Cook flag cache is independently
checked against a fresh public R run. Replogle and QTL reference workflows are
rerun. Reference implementations were accessed through public APIs, without
reading R/Bioconductor source in this review. Permitted PyDESeq2 provenance is
credited in DEPENDENCIES.md.

No pert50k/pertfull/seeds/crt timing rerun was started: the user's choice between
quiet-host reruns and historical v1.3.0 timings remains pending. The validated
historical 50k six-seed receipts live in `validation/PERTURBATION_SEEDS_50K*.json`.
The original full-screen SCEPTRE/seed stages failed; no completed new full-screen
six-seed receipt is claimed. No Gasperini all-by-all, full Replogle essential
screen or genome-wide QTL was started. No push, tag or release was made.

Intermediate `*_UNTIMED.json` probes remain local audit files and are excluded
from this published collection; their producers predate the final timing freeze.
