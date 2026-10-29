# RapidGLM 1.4.0 release receipts

The final 1.4.0 release uses library commit `f0ecfbd7c8dba698bbb50db5fd61785a2a463800`, annotated as
`v1.4.0-dev.4`. The `v1.4.0` release commit adds the measurements and documents.
The complete final run finished 2026-10-22 13:16:57 PDT.

All 18 stages ran in the original order on the measured commit. Source,
eight executable/library hashes, container image IDs and the R install check
are in [source.json](source.json). The runner verified the frozen source and
artifacts before and after every stage. Earlier collections remain unchanged.
The CPU-policy instrumentation is retained with these receipts and separately
hashed in source.json. Its temporary startup link for the Rust environment was
removed after timing; the measured library source stayed unchanged.
The checksum-recorded supplemental `policy_hook/usercustomize.py` applies
the same memory admission check before every untimed full-screen seed command.
It was added and checked before the seed stage, without changing the library
or the three existing CPU-controller files.
The separately hashed `supervise_rerun.py` monitors the eQTL-stage rerun with
the same synchronous checks and one-second policy loop; the original three
controller files remain unchanged.

## Tests

Native and portable x86 builds each passed 172 C++ cases,
the C99 caller and 28 CLI cases. Python binding and
statsmodels parity, 10 Python C-API cases,
4 Python DESeq cases, and R parity/application tests
passed. Rust passed 5 unit,
3 integration and 1
documentation cases. The timing guard passed 3
tests; Replogle/QTL recipe checks and C/C++ formatting passed.

All container suites passed from a clean clone of the measured tag; the
[container test log](container-tests.log) is retained with this collection. The selected ARM64 suite
passed 29 C++ cases with
5,281,712 assertions, C99 and 28 CLI
cases under QEMU. It does not include a native ARM performance measurement or
the older full cross-architecture application study. See [TESTS.json](TESTS.json),
[ARM64.json](ARM64.json), and the explicitly historical
[ARM64_DEV3.json](../../ARM64_DEV3.json).

These functional tests predate the final-policy timing restart. Their
source-identity checks and pass/fail results remain valid; their elapsed
diagnostics are not benchmark measurements and are outside the CPU-policy audit.

## Timing protocol and stage results

Timings are **final, not provisional**, on the **16 identical efficiency cores,
one thread per core**, of the Intel Core i9-13900KF workstation (125 GiB RAM).
One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.
The [CPU policy audit](CPU_POLICY_AUDIT.jsonl) logs every one-second check and
a synchronous policy check at every quiet start; the supervisor stops the run
if turbo, the frequency policy, or its monitoring continuity changes.
One, four, eight, twelve and sixteen threads use CPUs 16, 16–19, 16–23,
16–27 and 16–31. Untimed preparation uses CPUs 0–7 at nice 10.

Each timed command holds the shared timing lock and starts after two consecutive
ten-second samples with host load ≤4 and every selected CPU below 20% busy.
Waits are outside the measurements. All 843 retained quiet
starts passed. All 71776 one-second policy checks
also passed. The start-load ranges below are from the successful quiet predicates;
queued-stage loads in [status.txt](status.txt) are not measurement starts.
Full-screen commands additionally require at least 100 GiB available memory
and no other process with RSS ≥8 GiB.
All 6 untimed full-screen seed
commands also have individual memory-admission evidence in the timing audit.

| Stage | Final exit | Quiet starts | Start-load range | CPU-policy start checks | Attempts |
|---|---:|---:|---:|---:|---:|
| glmacc | 0 | 0 | untimed | 0 | 1 |
| glmspeed | 0 | 409 | 0.221–2.627 | 409 | 1 |
| batch | 0 | 16 | 0.819–1.423 | 16 | 1 |
| solvers | 0 | 6 | 0.812–1.561 | 6 | 1 |
| solvers_control | 0 | 3 | 1.332–1.495 | 3 | 1 |
| bulkval | 0 | 61 | 0.495–3.314 | 61 | 1 |
| edgerval | 0 | 0 | untimed | 0 | 1 |
| bulk | 0 | 200 | 0.491–3.881 | 200 | 1 |
| cooks | 0 | 0 | untimed | 0 | 1 |
| rust | 0 | 12 | 1.764–3.513 | 12 | 1 |
| staging | 0 | 0 | untimed | 0 | 1 |
| pert50k | 0 | 15 | 1.490–3.593 | 15 | 1 |
| pertfull | 0 | 8 | 2.639–3.705 | 8 | 1 |
| seeds | 0 | 0 | untimed | 0 | 1 |
| untestable | 0 | 13 | 1.672–3.314 | 13 | 1 |
| crt | 0 | 6 | 2.728–3.196 | 6 | 1 |
| replogle | 0 | 87 | 1.030–3.741 | 87 | 1 |
| eqtl | 0 | 7 | 2.379–3.777 | 7 | 2 |

The eQTL stage was repeated in full after its final jaxQTL permutation command
recorded 6.51% estimated other selected-core activity, exceeding the retained
5% threshold. All seven measurements from that first eQTL attempt were
discarded and archived; the preceding seventeen stages passed and were retained.
The table reports the accepted attempt's starts and loads. Across both attempts,
all 850 quiet starts had
synchronous policy checks; 7 belong to the
discarded eQTL attempt. Both monitored segments passed every policy check,
with no timed work during the intervening setup. The accepted measurements
have no contention flags. See [STAGE_RERUNS.json](STAGE_RERUNS.json) and the
invalidated-command records in TIMING_AUDIT.json for the discarded evidence.

Detailed readiness samples and contention checks are retained in
[TIMING_AUDIT.json](TIMING_AUDIT.json) and [timing-audit.jsonl](timing-audit.jsonl).
The untimed accuracy and validation stages can record diagnostic elapsed times;
those are not used as benchmark performance measurements.

One unpublished preliminary attempt was stopped after an input-environment
mismatch was detected. The corrected final run uses the original NumPy 1.26.4
input generator. No measurement from the preliminary attempt enters these
tables. Its reason and unpublished source identity remain in source.json.

The first timing attempt used turbo on with the old BIOS and was interrupted
by the host shutdown during bulk. The next attempt used fixed 2.2 GHz with
turbo off and ended with exit 125 when the host setting changed to the user's
final choice: turbo on, CPUs 8-11 capped. Both attempts are void. All 18 stages
were restarted under the final setting; no earlier timings enter these tables. The reason, archive locations and retained functional-test
scope are in [INVALIDATED_RUNS.json](INVALIDATED_RUNS.json).

## Comparison with development receipts

Scientific and timing review: accepted final-policy collection

The measured library is f0ecfbd7c8dba698bbb50db5fd61785a2a463800,
tagged v1.4.0-dev.4. Both prior clock/BIOS attempts are void. Their measurements
are excluded from these findings and every final performance table.

The initial final-policy eQTL stage exited zero, but its last jaxQTL command
recorded 6.51% estimated other selected-core activity (threshold 5%). The
entire seven-command eQTL stage was archived and was repeated successfully on the
same measured commit. The first seventeen stages passed and are retained.
No timing from the discarded eQTL attempt enters the primary receipts.

The final workstation has BIOS 1836, microcode 0x133, 253 W sustained and
short-term power limits, and four 32 GiB modules at 4,200 MT/s (125 GiB usable).
Turbo is on. Faulty P CPUs 8-11 are capped at 2.2 GHz and never timed.
Measurements use subsets of identical E CPUs 16-31, one hardware thread per
core, under the shared lock after the retained quiet-start predicate. The
policy is checked synchronously at each start and once per second. Absolute
seconds from different host settings do not isolate implementation changes;
matched tool speed-up ratios are reported separately.

GLM accuracy and batch

The 176 accuracy input arrays are byte-identical to the development baseline.
All 534 raw accuracy/reference results agree exactly after excluding elapsed
diagnostics and normalizing problem paths. The accuracy summary is identical.
The 408 speed problem/tool identities and error states match. Batch scientific
fields are unchanged. Of fifteen stored batch elapsed timings, thirteen are
within 5% of the carried dev.2 baseline; two are 5.2-7.2% faster. The median
paired ratio is 1.0003. Host and measurement conditions differ, so those
individual differences are not claimed as isolated library improvements. All 132 additional speed arrays have now been hashed and are byte-identical,
for 308 matching input arrays in total. Hashing was deferred until timing
finished to avoid file-cache interference.

Across 33 native GLM cases, the median paired final/development elapsed-time
ratio is 0.9911095427; the range is 0.8882-1.0698, with ten changes exceeding 5%.
This is a median of paired ratios, not a ratio of median elapsed times. The
historical GLM raw baseline is dev.2, carried by the dev.3 review. Larger
differences combine host changes, different start conditions and runtime
variation. Old GLM logs record ending loads; those values do not establish
selected-core contention. For tiny R fits, a 1 ms difference can be a large
percentage (4 to 5 ms is +25%; 2 to 1 ms is -50%). Fresh matched comparisons
are authoritative; native data and scientific results remain unchanged.

Solvers

Five of the six scientific receipts match the carried development baseline
exactly. In sim5000 fixed mode, 20 stored likelihood-shortfall diagnostics
differ by at most 1.00000008274e-9; other stored scientific fields are equal.
The solver benchmark source is unchanged, but it obtains offsets and
dispersions through bulk_de, whose header changed during review. This identifies
a changed input-preparation dependency, without proving the cause of each
rounding difference.

All scientific fields agree exactly between the three initial solver runs
and their efficiency-core controls. Across 22 method timings, the
initial/repeat ratios range from 0.9877769713 to 1.0352543898, median
1.0019176512. These controls support the current measurement consistency.

Against the carried dev.2 timings, airway fixed-mode Newton takes 0.198174 s
versus 0.216837 s (8.61% less), and the independent-fit variant takes
0.214310 s versus 0.234533 s (8.62% less). Airway joint finite-fit CPU totals
rise about 5.0-5.1%; a thyroid L-BFGS boundary subtotal changes by 0.22 ms
(5.04%). Other stored solver timing changes are within 5%. These are
cross-host comparisons, with the changed input-preparation dependency noted
above; they do not isolate a solver implementation gain. The fresh matched
Ceres comparisons and efficiency-core repeats are the performance evidence.

Contention-accounting review

The retained application harness marks estimates above 20% as contended;
CRT/QTL whole-command guards use 5%. The audit also lists three application
repeat estimates above 5% for review: airway Python DESeq-style 5.31%,
airway R 5.18%, and thyroid32 PyDESeq2 7.11%. They are upper-bound estimates
that can count unreaped tool workers, not direct measurements of external
work. All three are third repeats. The reported R and PyDESeq2 minima come
from other repeats (0.238 s and 20.527 s). Python DESeq-style selects
0.335241 s; its other repeats are 0.339009/0.340830 s, within 1.7%.
These observations do not establish a performance anomaly requiring a
protocol rejection. All raw repeats and the distinct thresholds remain
visible in TIMING_AUDIT.json; the threshold was not changed after measurement.

Bulk and edgeR correctness

Both receipts agree scientifically with the dev.3 review; all four bulk
gates pass. Differences from dev.2 were already present in the reviewed code:
the bulk mock-null BH discoveries/split changed from 0.557142857 to
0.471428571 after filtering changes; 56 edgeR numeric diagnostics differ by
at most 6.197150637e-7. The untimed edgeR correctness-stage elapsed fields
will be removed from the published primary correctness receipt. Performance
tables use the guarded application measurements instead.

Small-dataset bulk application measurements

All scientific summary fields match dev.3 exactly across airway, thyroid32
and thyroid128. The 162 repeat measurements have no contention flags. Across
54 timing summaries, final/dev.3 elapsed-time ratios range from 0.96799117
to 1.02171124, median 0.99591870; none differs by more than 5%. Matched
R-interface speed-up ratios remain within 4.1% of the reviewed values.
At one/eight threads respectively, the fresh R/DESeq2 speed-ups are
6.2826/11.0546 for airway, 6.4482/9.3457 for thyroid32, and
11.9963/17.8816 for thyroid128. Earlier dev.2 scientific differences are
already present in dev.3; this run adds no new scientific changes.

Complete bulk application measurements

The full thyroid dataset also matches dev.3 scientifically. Across all four
datasets, 198 repeat measurements yield 72 timing summaries with final/dev.3
ratios of 0.96799117-1.02171124, median 0.99680908; none changes by more
than 5%. The fresh full-thyroid R/DESeq2 speed-ups are 13.7545 at one thread
and 12.8672 at eight threads, versus 13.7124 and 12.7969 in the review.
All eight matched R-interface speed-up ratios remain within 4.1% of dev.3.
No bulk timing rerun is indicated by the retained guards or these comparisons.

Cook's diagnostics

All recomputed scientific fields in COOKS_FLAGS, COOKS_THYROID and
COOKS_LEGACY match dev.3 exactly. Two carried historical blocks are absent:
alpha_rob_black_box and alpha_rob_check described an analysis of 2026-10-01
that the current script does not recompute. Their absence must not be
described as a changed estimate or as fresh validation of native/reference
robust-dispersion equality. The old collections preserve those diagnostics.

Rust comparison

The available raw baseline is dev.2; the preserved dev.3 collection has no
raw Rust application receipt. All four stored native Rust count summaries
and discovery-call summaries match that baseline. Rust implementation changes
from dev.2 are documentation-only. The Python backend's optimizer handoff
changed: optimized and bound-active fits now return to DefaultInference,
alongside failed fits. Its native bulk dependency also changed during review.
The relative Python/Rust column comparisons therefore change, while the fresh
Python measurements independently agree with the reviewed bulk workflow.

Rust remains an unfloored workflow with its own trend and Cook handling;
Python inference uses floored fits. These timings do not compare identical
fit semantics. The 128/647-sample Python comparisons are 9.26%/13.41% slower
than dev.2, consistent with the changed comparison dependency and closely
matching their fresh bulk timing repeats. Native Rust timings beyond airway
remain within 2% of dev.2. Airway changes from 0.26 to 0.22 s: 40 ms in a
short whole-command measurement rounded to 0.01 s. Startup/runtime variation
and the changed host prevent assigning that difference solely to the BIOS.

Staging

All recomputed staging scientific fields match the available dev.2 raw
receipt carried by the development review. No fresh dev.3 raw staging
receipt is present. Both public panels pass the agreement-floor and discovery
count gates; all six null-calibration configurations pass. The count gate
uses 200 seeds, while the reported Monte Carlo overlap summaries use ten.
The reference stage probe has 1,469 first-stage and 31 promoted pairs among
1,500 tested pairs. The complete comparison is retained privately; only the
explicitly public panels appear in the public comparison receipt.

The unchanged full-screen seed driver lacked individual memory admission
checks. Before that stage began, a supplemental runtime startup hook was
added with its checksum recorded in source.json. It calls the existing
100 GiB/no-other-8-GiB-job guard immediately before each native full-screen
seed command. Smoke checks established gate-before-exec ordering, prevention
of execution when the gate fails, preservation of the CPU hook, and no
profiling. No measured library source or existing CPU-controller file changed.
The final audit requires one retained memory gate per seed as well as the
full-screen performance-command gates.

50,000-cell high-MOI application comparison

All scientific summary fields match the available dev.2 raw receipt carried
by the development review. Timings are compared with the published 1.3.0
perturbation measurements carried by dev.3: all fourteen elapsed-time ratios
remain within 2.7%, and matched speed-up ratios within 2.8%. Fresh R-interface
speed-ups at one/eight threads are 4.5777/4.2118 with ks acceptance, and
3.5577/3.3337 with the strict default. The Python ks result at eight threads
is 4.9081x; that does not describe every interface or acceptance mode.

These are application-workflow timings. R/Python measure the public API call
after input loading; the CLI measures the complete command. SCEPTRE reports
the sum of its calibration, power and discovery calls, excluding setup.
The README, primary receipt and ratio-review drafts state this scope. It must
not be conflated with the actual whole-command low-MOI measurements.

Full-cell high-MOI application comparison

All seven scientific summaries match the carried published 1.3.0 baseline
exactly; no fresh dev.3 full-panel measurement was available. All seven
commands passed their memory and quiet-start guards, without an OOM or
contention flag. Initial and per-command memory admissions found at least
112.4 GiB available and no other process of 8 GiB RSS or more.

At eight threads, fresh ks workflow speed-ups are 5.0965x for CLI, 5.2310x
for Python and 4.5558x for R. Strict-default speed-ups are respectively
3.6672x, 3.7470x and 3.3390x. The acceptance rule and interface must accompany
each headline.

The largest elapsed-time change is Python ks: 305.664 to 281.495 seconds,
7.91% faster; the matched speed-up ratio rises 8.56%. The other six elapsed
changes are within 4.41%. The Python sceptre API function is AST-identical
to v1.3.0. Selected-core background CPU was low in both runs (0.51% then,
0.22% now), so these records do not establish a CPU-contention explanation
for the full difference. Host BIOS, power, memory and build settings differ,
and the older run lacked the new global quiet-start predicate. With one
repeat per mode, individual causal effects cannot be isolated. The fresh
guarded measurements are reported with these comparison limits.

Seed robustness

All six recomputed seed results match the carried 1.3.0 scientific results
exactly. The raw receipt adds dataset="gasperini" and options={} metadata;
these are schema additions, not changed estimates. Discoveries range from
96 to 102 versus SCEPTRE's 101. Jaccard with SCEPTRE spans
0.89423-0.96040 (mean 0.93231); between-seed Jaccard spans 0.90385-0.96000
(mean 0.93085 over 15 pairs). All six seed commands have individual memory
admissions, with at least 117.8 GiB available and no other large job. Their
elapsed times remain correctness diagnostics.

The pseudobulk pilot's edgeR performance values come from its separately
guarded QL-reference calls. The extra untimed edgeR fit checks supply agreement
tables, while the stored timing summary retains the guarded values. The
low-MOI SCEPTRE wrapper preserves the GNU-time whole-command wall field;
its internal phase counters use separate keys.

Untestable-pair scaling

All four RNA/ATAC by high/low-MOI scientific summaries match both the dev.2
raw receipt and the published 1.3.0 run. Listed, prefiltered and grid modes
test the same passing pair set and produce identical tested records within
each case. The current options explicitly set sequential_screen=true and
early_stop_p=0.002, as requested.

The dev.2 run used CLI defaults: sequential screening was off and early
stopping was 0. Its much longer timings are an unmatched configuration
comparison, labeled descriptive only in COMPARISON.json. The published
1.3.0 run records the same explicit options as the final run. Against that
matched baseline, all twelve workflow-wall-time ratios lie within 2.3%
(range 0.97767-1.01460). This protocol distinction explains the apparent
roughly fourfold difference from dev.2; it is not a measured code speed-up.

CRT scaling

All six CRT commands passed their guards. Their whole-command times differ
from the carried published panel study by only 0.3-1.4%: 121.07/126.70 s
for 50 targets x 1,000 genes, 396.99/418.86 s for 200 x 1,000, and
1,564.99/1,644.56 s for 100 x 7,418 (ks/strict respectively). These matched
timing rows are included in COMPARISON.json. All twelve matched raw CRT tables are byte-identical to the available
dev.2 baseline; hashing was performed after timing ended.

Low-MOI pilot timings

Fresh whole-command times on sixteen efficiency cores are 1,939.75 s for
strict RapidGLM, 721.67 s for ks, and 3,164.09 s for SCEPTRE: speed-ups
1.6312x and 4.3844x respectively. Native times differ from dev.3 by +0.72%
and -0.15%; SCEPTRE is 3.34% faster. Matched speed-up ratios change by
-4.03% and -3.20%, within the expected few-percent range.

The native modes store the same 174,761 null fits. Strict mode adds
1,218.08 s (20.3 minutes), making its elapsed time 2.688 times ks. The
SCEPTRE whole-command wall is only 21.729 s above its internal public-call
sum. These measurements attribute the large strict/ks dichotomy to the
confirmation policy; startup does not account for that twenty-minute gap.
The completed pilot's scientific summaries match dev.3 exactly in both
single-cell modes and across all twenty pseudobulk targets. All 146 stored
pilot timing comparisons are within 3.5% of dev.3; none changes by more
than 5%. Pseudobulk timings alone differ by at most 1.8%. Across twenty
targets the summed wall times are 703.74/3,984.44 s for RapidGLM/DESeq2,
and 93.52/589.83 s for RapidGLM QL/edgeR (one efficiency core). These give
summed-time ratios of 5.66x and 6.31x, separate from per-target summaries.
Differences from dev.2 are already present in the reviewed dev.3 results.
All 147 matched single-cell and pseudobulk raw tables are byte-identical
to dev.3 (seven single-cell and 140 pseudobulk tables).


Matched GLM ratio review

Across 255 matched reference/native GLM speed-up ratios, the median
final/historical ratio is 0.96120, the range is 0.49803-1.24507, and 120
change by more than 5%. Thus individual GLM ratios do not all remain within
a few percent across settings. Timer quantization explains some extreme
small-fit percentages, but not every larger change. For Gaussian n=1,000,000,
p=20, statsmodels moves from 7.86371 to 4.34490 s while native moves from
0.319478 to 0.287051 s: the speed-up changes from 24.61x to 15.14x.
For 130 comparisons where all four times exceed 50 ms, the median ratio
is 0.96488 and the range remains 0.61494-1.12759. Among the large-problem
summaries, the median paired ratio changes for fastglm and statsmodels are
-6.03% and -5.51%. The other tool medians are within 4.4%.

The 308 input-array hashes and all scientific accuracy results agree; the
per-fit tool wrappers and timing routines are unchanged. The drivers add
per-command quiet starts and an explicit reference-binary path selector;
these change start conditions rather than fitting algorithms. The historical
END-load records cannot
establish selected-core interference. BIOS, power, memory, build/runtime
and measurement conditions differ, so these comparisons do not identify
a separate causal contribution for each change. The fresh guarded values
replace the earlier performance claims. Neither stable scientific output
nor a stable median establishes that every historical timing ratio holds.
The complete per-case evidence is in GLM_REVIEW.json and HEADLINE_COMPARISON.json.
For the long statsmodels example, both runs report NumPy 2.2.6, SciPy 1.15.3
and statsmodels 0.14.4, with the same three fit iterations and identical
estimates. The old two repeats are 7.864/9.087 s; all four fresh repeats are
4.345-4.411 s. The fixed 15-second budget permits different repeat counts.
Thus this is not explained by millisecond quantization or one lucky minimum.
The available records do not isolate the cause of this cross-host timing
change, and no specific contention or implementation cause is asserted.

QTL: accepted full-stage rerun

All seven accepted measurements use twelve efficiency cores, CPUs 16-27,
with the same seed, data and declared permutation budgets. Gaussian at
10,000 permutations takes 117.19 s versus tensorQTL's 151.53 s: 1.29303x.
NB score at 1,000 permutations takes 1,092.60 s versus jaxQTL's 3,112.27 s:
2.84850x. Gaussian at 1,000 takes 16.38 s; FastQTL's adaptive 1,000-10,000
run takes 1,004.62 s; jaxQTL SPA+ACAT takes 52.74 s and uses a different
calibration. Only matched-budget ratios support the corresponding headlines.

Relative to dev.3, Gaussian times are 41.48% and 54.35% lower at 1,000 and
10,000 permutations, following the intentional Gaussian implementation
change after that review. TensorQTL is 4.51% slower, FastQTL 1.77% faster,
native NB 0.53% faster and jaxQTL NB permutations 2.53% faster. SPA+ACAT
is 9.24% faster (a 5.37 s difference). Its whole-command time includes
startup and JIT work. With one accepted measurement per tool/budget and
changed host conditions, those reference timing differences do not isolate
an implementation effect. The matched NB speed-up differs by only -2.01%;
the Gaussian ratio improves by 2.2893 times relative to the old ratio,
which also includes the historical host difference.

All three native QTL tables are byte-identical to the prior optimized-code
checks. Against dev.3, every non-Beta column remains exact; Beta p-values
and both shape parameters stay within the recorded 1e-8 relative bound.
The tensorQTL table is identical after decompression. All 21 additional
reference tables agree exactly: twelve FastQTL chunks, eight complete jaxQTL
parquet tables and the combined reference summary. Parquet values are
compared exactly, independently of container-byte metadata.

At Storey q < 0.05, 10,000 Gaussian permutations yield 267 eGenes, all shared
with FastQTL (Jaccard 1.0000), and 266 shared with tensorQTL (Jaccard 0.9963).
Nominal-p Spearman correlations are 0.9999128 and 0.9999286, with identical
lead variants for both Gaussian references. NB gives 168 eGenes versus
jaxQTL's 170, all 168 shared (Jaccard 0.9882, Spearman 0.9995569; 95% lead
agreement). Gaussian and NB are different models; their cross-model rows
are descriptive comparisons. These are chromosome-22 panel findings,
not evidence for genome-wide runtime or agreement.

Final audit and raw-table coverage

All eighteen stages passed; eQTL has two attempts, with the complete first
attempt excluded after its last command exceeded the existing 5% threshold.
The accepted collection has 843 quiet starts and no contention flags. Both
monitored segments passed all 71,776 periodic policy checks and all 850
synchronous quiet-start checks, including seven starts from the discarded
attempt. The maximum observed poll gap is 3.609 s, within the retained 5 s
limit. Each full-screen command and all six seed commands have the required
memory admissions. The temporary Rust startup link has been removed.

All 235 available dev.3 raw-table matches are byte-identical: 88 bulk tables
and 147 Replogle tables. All 33 matched Gasperini tables from the carried
published 1.3.0 baseline are also byte-identical. Against dev.2, 150 of 263
raw tables are byte-identical; changes occur in the bulk/Python and pseudobulk
workflows reviewed above, while CRT, batch and perturbation outputs match.
The newer dev.3 tables agree exactly wherever available. Reference QTL
checks are retained separately in QTL_FINAL.json. The eight artifact hashes,
two container image IDs, original controller hashes and all 118 protected
receipt hashes were rechecked successfully before document assembly.


Machine-readable comparisons are in [COMPARISON.json](COMPARISON.json),
[TABLE_IDENTITY.json](TABLE_IDENTITY.json), [QTL_FINAL.json](QTL_FINAL.json),
and [HEADLINE_COMPARISON.json](HEADLINE_COMPARISON.json).
Historical comparisons distinguish changed options or timing scopes from
matched-code repeats. All current documentation tables use the final collection.

## Paper receipt synchronization

After updating the public clone, run `make receipts` in the paper repository.
The handoff should retain the final host description above, including BIOS
1836, microcode 0x133, the 253 W limits, turbo on, and the exclusion of faulty
performance CPUs 8–11 from timing. The primary receipts carry this setting in
`_benchmark_host_configuration`; `source.json` retains its verified values.
The primary files remain under `validation/`. In particular:

- `APPLICATIONS_BULK.json` contains final bulk/QL performance timings;
  `EDGER_QL.json` contains correctness comparisons only.
- `APPLICATIONS_PERTURBATION.json` separates high-MOI interfaces, thread counts
  and acceptance modes. `REPLOGLE_SC_KS.json` and `REPLOGLE_SC_STRICT.json`
  give whole-command low-MOI comparisons. Their speedups have different scopes.
- `QTL_FINAL.json` gives final whole-command QTL times; `EQTL_COMPARE.json`
  gives reference agreement. Compare Gaussian tools at 10,000 permutations.
- `SOLVERS.json` uses `efficiency_core_control` for the three repeated controls.
  No performance-core times enter the final tables.
- `ARM64.json` describes the final selected suite. The earlier full study is
  retained as `ARM64_DEV3.json` and is explicitly historical.
- `RUST_DESEQ.json` compares workflows with different low-count fit semantics.
  Its Cook's summary is `cooks_outliers_thyroid32`; singleton design cells do
  not establish leverage one.

The historical robust-dispersion inversion diagnostics remain in the dev.3
collection for provenance. They were not recomputed and are absent from both
the fresh raw Cook's outputs and the primary final receipts. They are not
used in any current table.

The measured commit in the receipts is the library commit above, rather than
the subsequent release commit that packages the receipts and documents.
