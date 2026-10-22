# Application benchmarks

Final RapidGLM 1.4.0 measurements, completed 2026-10-22. The library is the code of
`v1.4.0-dev.4`; all 18 stages were rerun for this release. The
[collection](../validation/receipts/v1.4.0/README.md) retains commands, source
and binary hashes, stage exits, readiness samples, and comparisons with the
preserved development collections.

## Host

Intel Core i9-13900KF, 125 GiB RAM. All timed commands use the **16 identical
efficiency cores, one thread per core**: CPU 16 for one thread, 16–19 for four,
16–23 for eight, 16–27 for twelve and 16–31 for sixteen. These timings are
**final, not provisional**. No performance-core timing is mixed into the tables.
Three solver controls repeat on CPU 16 under the same protocol.
One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.
A synchronous check before each measurement and a one-second CPU-policy monitor
stop the run if these settings change;
[CPU_POLICY_AUDIT.jsonl](../validation/receipts/v1.4.0/CPU_POLICY_AUDIT.jsonl)
retains every check.

Each timing holds the shared lock and starts after two consecutive ten-second
samples with one-minute host load ≤4 and every selected CPU below 20% busy.
Both samples are retained in the timing audit. Untimed preparation uses CPUs
0–7 at nice 10. The full-screen commands additionally require 100 GiB available
memory and no other process with RSS ≥8 GiB. Waits are excluded from timings.
Application receipts record estimated other CPU use during each command;
[TIMING_AUDIT.json](../validation/receipts/v1.4.0/TIMING_AUDIT.json) records the checks.

## Datasets

| Name | Source and terms | Size | Test |
|---|---|---|---|
| airway | Bioconductor `airway` | 8 samples, 22,369 genes | dex, `~ cell + dex` |
| thyroid32, thyroid128 | Samples drawn once with a fixed seed from thyroid | 32 and 128 samples | as thyroid |
| thyroid | GTEx v8 thyroid from recount3, which is public domain | 647 samples, 28,769 genes | sex, `~ centre + Hardy scale + age + sex` |
| gasperini50k | 50,000 cells drawn once with a fixed seed from gasperini | 50,000 cells | as gasperini |
| gasperini | Gasperini et al. 2019 at-scale screen, GEO GSE120861 (NCBI places no restrictions; submitters may claim rights) | 207,324 cells, 7,418 response genes, 7,667 guides | high MOI, complement controls, CRT |

Thyroid preparation:
- recount3 stores base-level coverage sums. Read counts are those sums divided by
  the sample's average mapped read length, rounded, as recount3's documentation
  describes.
- Samples are GTEx's RNA-seq release set with a known Hardy scale, from the two
  large centres.
- Genes have at least 10 reads in at least 5% of samples.

Gasperini preparation:
- Guides are the 20-nt spacers listed for each cell, grouped by the published
  gRNA groups.
- Pairs are drawn once with a fixed seed from the published pair table,
  excluding the published zero-inflated outlier genes:
  - every `positive_ctrl` pair that remains (130);
  - 10,000 candidate-enhancer pairs;
  - 5,000 non-targeting pairs.
- Covariates are computed over every gene: log total UMIs, log detected genes,
  log(1 + guides called), percent mitochondrial and preparation batch.
- phenoData lists no guides for 1,527 cells; those cells keep a guide count of
  zero.

## Tools

Bulk:
- `rapidglm-cli`: `bulk-de`, the whole process including input parsing.
- `rapidglm-python`: `rapidglm.bulk_de`, the call only.
- `rapidglm-deseq`: `rapidglm.deseq`, the PyDESeq2-style interface, calls only.
- `rapidglm-r`: `rapidglm::bulk_de` in R, the call only. It runs the same C
  interface with R's BLAS. The package is installed in
  `validation/local-data/R_rapidglm`.
- `pydeseq2`: PyDESeq2 0.5.2 with its default inference. Timed from
  `DeseqDataSet(...)` to `summary()`, including PyDESeq2's input validation.
- `pydeseq2-rapidglm`: the same with `rapidglm.pydeseq2.RapidGLMInference`.
- `deseq2`: DESeq2 `DESeq()` and `results()`, using `MulticoreParam` at 8
  threads.
- `deseq2-glmgampoi`: the same with `fitType = "glmGamPoi"`. DESeq2 does not
  parallelize it and recommends it with the LRT; the Wald test is used here for
  comparability.

- `rapidglm-edger-cli`: the complete `edger-ql` command.
- `edger`: edgeR 4.6.2 QL fitting and testing through its public R interface.

Perturbation:
- `rapidglm-cli`: `sceptre`, the whole process.
- `rapidglm-python`: `rapidglm.sceptre`, the call only.
- `rapidglm-r`: `rapidglm::sceptre` in R, the call only.
- `rapidglm-cli-ks`, `rapidglm-python-ks`, `rapidglm-r-ks`: the same three
  with `fit_check = "ks"`, which accepts the skew-normal fit as SCEPTRE does.
  These match the skew-normal acceptance rule. The plain names run RapidGLM's defaults:
  the stricter check, with confirmation draws
  ([CLI.md](CLI.md#perturbation-workflow)).
- `sceptre`: SCEPTRE 0.99.0, the sum of `run_calibration_check`,
  `run_power_check` and `run_discovery_analysis`, using `parallel = TRUE` at 8
  threads.
  - SCEPTRE draws its own negative-control pairs, as many as the panel has.
  - Guide calls, covariates and discovery and positive-control pairs are
    identical to RapidGLM's.

Each tool runs at 1 and 8 threads where supported, `--repeats` times in
rotation. The minimum wall time is reported. Agreement is measured against
DESeq2 at 1 thread, or SCEPTRE at its lowest measured thread count (one on
  the 50,000-cell panel, eight on the full benchmark panel), over the same genes or discovery
pairs: the Jaccard index of BH discoveries at 0.1 and the Spearman correlation
of p-values.

## Results: bulk DE

Seconds at one / eight threads; minima of three rotated repeats for airway,
thyroid32 and thyroid128, and two for thyroid647. Source:
[APPLICATIONS_BULK.json](../validation/APPLICATIONS_BULK.json).

| Data | RapidGLM (R) | DESeq2 | DESeq2 + glmGamPoi | PyDESeq2 | PyDESeq2 + RapidGLM | Jaccard with DESeq2 |
|---|---|---|---|---|---|---|
| airway, 8 samples | 0.966 / 0.238 | 6.07 / 2.63 | 15.2 | 53.1 / 12.3 | 2.98 / 2.53 | 0.986 |
| thyroid, 32 samples | 4.88 / 0.784 | 31.5 / 7.33 | 37.9 | 105 / 20.5 | 12.8 / 5.73 | 0.985 |
| thyroid, 128 samples | 9.87 / 1.75 | 118 / 31.3 | 85.9 | 115 / 23 | 19.9 / 7.55 | 0.997 |
| thyroid, 647 samples | 35.1 / 6.51 | 483 / 83.7 | 272 | 174 / 35.8 | 61.6 / 17.4 | 1.000 |

Quasi-likelihood DE, from the same fresh benchmark; RapidGLM includes CLI input/output.

| Data | RapidGLM CLI (1 / 8) | edgeR (1 thread) | Speedup | Jaccard |
|---|---|---|---|---|
| airway | 0.282 / 0.162 | 0.777 | 2.76× | 1.0000 |
| thyroid32 | 1.56 / 0.503 | 6.21 | 3.99× | 0.9834 |
| thyroid128 | 3.33 / 1.11 | 20.6 | 6.17× | 0.9904 |
| thyroid647 | 13.1 / 4.08 | 91 | 6.97× | 0.9995 |

## Results: high-MOI perturbation

Gasperini panels, one repeat per tool/thread/mode. `ks` uses SCEPTRE-style
skew-normal acceptance; `strict` uses RapidGLM's default with confirmation
draws. RapidGLM R/Python times cover API calls; CLI times include parsing and
writing. SCEPTRE times sum calibration, power and discovery calls. Agreement
uses discovery pairs tested by both tools. Source:
[APPLICATIONS_PERTURBATION.json](../validation/APPLICATIONS_PERTURBATION.json).

| Cells | Threads | Acceptance | RapidGLM R / Python / CLI | SCEPTRE | R speedup | Discoveries R / SCEPTRE | Jaccard | p Spearman |
|---|---|---|---|---|---|---|---|---|
| 50,000 | 1 | ks | 333 / 331 / 338 | 1,524 | 4.58× | 28 / 24 | 0.793 | 0.9853 |
| 50,000 | 1 | strict | 428 / 424 / 430 | 1,524 | 3.56× | 28 / 24 | 0.793 | 0.9853 |
| 50,000 | 8 | ks | 59.6 / 51.1 / 54 | 251 | 4.21× | 28 / 24 | 0.793 | 0.9853 |
| 50,000 | 8 | strict | 75.3 / 66.6 / 68.6 | 251 | 3.33× | 28 / 24 | 0.793 | 0.9853 |
| 207,324 | 8 | ks | 323 / 281 / 289 | 1,472 | 4.56× | 96 / 101 | 0.894 | 0.9864 |
| 207,324 | 8 | strict | 441 / 393 / 402 | 1,472 | 3.34× | 96 / 101 | 0.894 | 0.9864 |

Across six seeds on the full benchmark panel, RapidGLM makes
96–102 discoveries;
Jaccard with SCEPTRE is 0.894–0.960
and between RapidGLM seeds 0.904–0.960.
See [PERTURBATION_SEEDS.json](../validation/PERTURBATION_SEEDS.json).

## Results: low-MOI perturbation

Replogle K562 essential pilot: 13,618 cells, 20 perturbation targets and 8,563
genes. Both tools use the same covariates and 16 efficiency cores. These are
whole-command wall times, including startup and output. One repeat per mode;
the same SCEPTRE reference run is used for both comparisons.

| Acceptance | RapidGLM CLI | SCEPTRE | Speedup | Discoveries RapidGLM / SCEPTRE | Jaccard | p Spearman |
|---|---|---|---|---|---|---|
| ks | 722 | 3,164 | 4.38× | 10982 / 10972 | 0.9492 | 0.9888 |
| strict | 1,940 | 3,164 | 1.63× | 10979 / 10972 | 0.9495 | 0.9888 |

The high-MOI and low-MOI panels use different resampling/control designs and
different numbers of tested pairs. Their speedups are workload-specific.
Confirmation draws also change the amount of work. Sources:
[REPLOGLE_SC_KS.json](../validation/REPLOGLE_SC_KS.json) and
[REPLOGLE_SC_STRICT.json](../validation/REPLOGLE_SC_STRICT.json).

## CRT panel scaling

Nested target/gene panels from the 207,324-cell Gasperini screen, with 16 threads.
Sources and seed: [CRT_SCALING.json](../validation/CRT_SCALING.json).

| Targets × genes | Pairs tested | Seconds (ks / strict) | Draws, millions (ks / strict) | Discoveries (ks / strict) |
|---|---|---|---|---|
| 50 × 1,000 | 48,596 | 121 / 127 | 29.4 / 31.0 | 6 / 6 |
| 200 × 1,000 | 193,824 | 397 / 419 | 116.8 / 122.4 | 5 / 5 |
| 100 × 7,418 | 721,266 | 1,565 / 1,645 | 437.0 / 456.7 | 25 / 14 |

## Requests with mostly untestable pairs

Synthetic 20,000-cell screen, 300 targets, ten negative-control groups, 6,000
genes and 40,000 peaks. All final runs use `sequential_screen: true` and
`early_stop_p: 0.002` on twelve efficiency cores. Each cell below gives wall
seconds / peak RSS in decimal GB. The three request forms are every pair
listed, passing pairs listed, and named targets/responses without a pair list.
Source: [UNTESTABLE_PAIRS.json](../validation/UNTESTABLE_PAIRS.json).

| Panel | Pairs requested | Passing QC | All listed | Passing listed | Named targets |
|---|---|---|---|---|---|
| Genes, high MOI | 1,860,000 | 101,564 | 42.3 / 1.37 | 38.5 / 0.39 | 38.4 / 0.38 |
| Peaks, high MOI | 12,400,000 | 645,408 | 189 / 7.89 | 166 / 1.14 | 165 / 1.02 |
| Genes, low MOI | 1,860,000 | 60,024 | 8.59 / 1.16 | 5.57 / 0.12 | 5.56 / 0.12 |
| Peaks, low MOI | 12,400,000 | 373,181 | 58.5 / 7.52 | 36.9 / 0.67 | 36.9 / 0.67 |

Tested records are identical across all three request forms. Naming targets
avoids materializing the failing pairs. The original dev.2 run omitted the
screening options and therefore measured a different amount of resampling;
its timings are not the final-release comparison.

## Cis-QTL and pseudobulk

[QTL_GAUSSIAN_SPEED.md](QTL_GAUSSIAN_SPEED.md) reports the matched 10,000-permutation
Gaussian comparison and the final NB/reference timings.
[BULK_DE.md](BULK_DE.md) reports the Replogle pseudobulk agreement from the same
release run. No genome-wide eQTL or full Replogle screen is included.
