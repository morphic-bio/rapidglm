# Gaussian cis-QTL permutation performance

The final RapidGLM 1.4.0 run takes **117.19 s** for 10,000
Gaussian permutations, compared with **151.53 s** for
tensorQTL 1.0.10: **1.29×** speedup on this panel.
Source: the complete final release run on `v1.4.0-dev.4`
(`f0ecfbd7c8dba698bbb50db5fd61785a2a463800`), completed 2026-10-22.

## Workload and timing scope

Geuvadis chromosome 22: 580 genes, 462 samples, 42 covariates plus intercept,
1 Mb cis window and seed 20261016. The run uses twelve of the workstation's
**16 identical efficiency cores, one thread per core** (CPUs 16–27).
One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.
The retained CPU-policy audit covers every quiet start and one-second checks
throughout both monitored segments. The initial eQTL-stage attempt was discarded
after its final jaxQTL command exceeded the 5% contention threshold (6.51%
estimated other selected-core activity). All seven eQTL commands were repeated;
only that accepted stage supplies the table below. See
[STAGE_RERUNS.json](../validation/receipts/v1.4.0/STAGE_RERUNS.json).
Timings are final, not provisional. Each command holds the shared timing lock
and starts after two consecutive ten-second quiet samples. All commands,
loads, successful readiness samples, CPU accounting and binary hashes are
retained. Waiting is excluded. There is one final measurement per tool/budget.

All entries below are whole-command timings from the final `eqtl` stage:

| Tool | Model | Permutations | Wall seconds | Start load | Peak RSS (GiB) |
|---|---|---|---|---|---|
| rapidglm_gaussian | Gaussian | 1000 | 16.38 | 2.379 | 1.141 |
| rapidglm_nb | NB score | 1000 | 1092.60 | 3.777 | 0.876 |
| rapidglm_gaussian10000 | Gaussian | 10000 | 117.19 | 3.321 | 1.145 |
| fastqtl | Gaussian | adaptive 1000-10000 | 1004.62 | 3.405 | 0.080 |
| tensorqtl | Gaussian | 10000 | 151.53 | 3.391 | 1.402 |
| jaxqtl_nb_score_spa_acat | NB SPA+ACAT | 0 | 52.74 | 3.316 | 0.851 |
| jaxqtl_nb_score_perm1000 | NB score | 1000 | 3112.27 | 3.311 | 0.972 |

FastQTL uses adaptive 1,000–10,000 permutations, so its budget differs from
the fixed 10,000 comparison. jaxQTL SPA+ACAT uses a different calibration from
permutation maxima. GNU time peak RSS is a per-process maximum, not aggregate
process-tree memory for multiprocess tools. The claim is limited to this
chromosome-22 panel and CPU configuration; no genome-wide run is included.

## Implementation

Gaussian permutations project covariates out of genotypes once per gene and
score 512 shuffled responses at a time. The in-house packed matrix product
uses 128-sample/128-column cache panels and six-by-eight AVX2/FMA register
tiles, with a portable fallback. Explicit least-squares residuals handle
cancellation when residual squared norm is at most 1e-6 of response squared
norm. Buffers belong to each gene worker; no external BLAS matrix product is
required. The observed statistic, permutation generator, Beta calibration
and other model paths retain their definitions. Equations are in
[METHODS.md](../METHODS.md#cis-qtl-mapping).

## Numerical comparison

The final native tables are byte-identical to the prior optimized-code
checks for Gaussian at both budgets and NB at 1,000. Against the dev.3
Gaussian tables, all statuses, lead variants, empirical permutation p-values,
observed p-values, estimates, SEs and successful/failed draw counts remain
identical. Only rounding-level Beta-fit values move:

| Permutations | Largest relative beta_p difference | Largest absolute difference | Largest-relative-difference gene |
|---|---|---|---|
| 1000 | 1.072e-10 | 4.015e-13 | ENSG00000244752.2 |
| 10000 | 1.945e-10 | 3.182e-13 | ENSG00000260065.1 |

At Storey q < 0.05, 10,000 Gaussian permutations find 267 eGenes;
Jaccard is 1.0000 with FastQTL
and 0.9963 with tensorQTL.
The tensorQTL table is byte-identical to dev.3 after decompression. NB finds
168 eGenes, with Jaccard
0.9882 against jaxQTL NB
at the matched 1,000-permutation budget. Full agreement metrics are in
[EQTL_COMPARE.json](../validation/EQTL_COMPARE.json).

## Development comparison and memory

The preserved dev.3 whole-command Gaussian 10,000 run took
256.70 s; the final run takes
45.7% of that time. The final run uses turbo on under BIOS 1836, microcode 0x133, 253 W sustained and short-term limits, and 4 x 32 GiB at 4,200 MT/s. It pins every timed command to identical efficiency cores and audits the policy at every start and once per second. The dev.3 review used turbo on with the old BIOS. Power limits and memory speed can change absolute seconds, so historical absolute-time ratios do not isolate implementation changes; matched tool-to-tool speed-up ratios are the comparison of interest.
This historical absolute-time ratio does not isolate the implementation change. The headline tensorQTL comparison uses only the
final receipt run, with the same CPU policy for both tools.
Larger bounded batches exchange extra memory for throughput. Final native
peak RSS is 1.145 GiB versus
1.402 GiB for tensorQTL. The batch allocation does
not grow with total permutation count.

## Tests and reproduction

Native and forced-portable suites each pass all 172 C++ cases, C99 and all
28 CLI cases. The selected ARM64 build passes 29 cases and 5,281,712
assertions, C99 and 28 CLI cases under QEMU. Bindings, Rust and the tagged
container suites pass. ARM checks do not establish native ARM performance.

[QTL_FINAL.json](../validation/QTL_FINAL.json) retains all final command
timings and numerical comparisons. The complete runner is
[run.sh](../validation/receipts/v1.4.0/run.sh); source and binary hashes are in
[source.json](../validation/receipts/v1.4.0/source.json). Earlier profiling and
kernel-tuning diagnostics remain historical development artifacts and are
not used as final release timings.
