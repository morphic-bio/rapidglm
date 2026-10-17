# Application benchmark (app note)

RapidGLM's two applications against their reference implementations, run as
black boxes through their public interfaces:
- bulk differential expression against DESeq2 and PyDESeq2;
- the perturbation workflow against SCEPTRE.

Driver: `scripts/benchmark_applications.py`. Data preparation:
`scripts/prepare_benchmark_data.py`. The receipt, with timings, host load and
agreement, is written wherever `--output` points. Final numbers need a quiet
host: runs whose 1-minute load exceeds a quarter of the cores are marked
`contended`. With `--cpus THREADS:LIST` the runs at a thread count are confined
to the listed CPUs. Each run then waits until other processes use less than a
fifth of those CPUs, records the share others used while it ran (`others_cpu`),
and is marked `contended` when that share exceeds a fifth.

## Host

The results below, and those in [GLM_COMPARISON.md](GLM_COMPARISON.md) and
[SOLVERS.md](SOLVERS.md), were measured on one workstation (Intel Core
i9-13900KF, 125 GiB) with version 1.3.0, on 2026-10-09 and 2026-10-10.

- **Cores.** The processor has eight performance cores and sixteen efficiency
  cores. Four logical CPUs, on two performance cores, were offline after
  processor faults. Every timed run was therefore confined to efficiency cores,
  which are identical, with one thread per core: one thread on CPU 16, four on
  16–19, eight on 16–23 and sixteen on 16–31.
- **Core type.** Times on an efficiency core are about 1.5 to 1.9 times those
  on a performance core, and the factor differs between tools. The solver
  receipt repeats three runs on a performance core
  ([SOLVERS.md](SOLVERS.md#results)).
- **Other load.** Timed runs held a lock, one at a time. Desktop services and
  a RAID consistency check kept every CPU about 7% busy. Another session ran
  untimed jobs on CPUs 20–31 at times; the receipts record `others_cpu` for
  every run on those CPUs.
- **Provisional.** The timings will be repeated when the processor is
  repaired. Agreement with the reference tools does not depend on the host.

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
- `pydeseq2`: PyDESeq2 0.5 with its default inference. Timed from
  `DeseqDataSet(...)` to `summary()`, including PyDESeq2's input validation.
- `pydeseq2-rapidglm`: the same with `rapidglm.pydeseq2.RapidGLMInference`.
- `deseq2`: DESeq2 `DESeq()` and `results()`, using `MulticoreParam` at 8
  threads.
- `deseq2-glmgampoi`: the same with `fitType = "glmGamPoi"`. DESeq2 does not
  parallelize it and recommends it with the LRT; the Wald test is used here for
  comparability.

Perturbation:
- `rapidglm-cli`: `sceptre`, the whole process.
- `rapidglm-python`: `rapidglm.sceptre`, the call only.
- `rapidglm-r`: `rapidglm::sceptre` in R, the call only.
- `rapidglm-cli-ks`, `rapidglm-python-ks`, `rapidglm-r-ks`: the same three
  with `fit_check = "ks"`, which accepts the skew-normal fit as SCEPTRE does.
  These are the like-for-like runs. The plain names run RapidGLM's defaults:
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
DESeq2 at 1 thread, or SCEPTRE at 1 thread, over the same genes or discovery
pairs: the Jaccard index of BH discoveries at 0.1 and the Spearman correlation
of p-values.

## Results (version 1.3.0, 2026-10-10)

Receipts:
- [validation/APPLICATIONS_BULK.json](../validation/APPLICATIONS_BULK.json):
  airway and the 32- and 128-sample sets with three repeats, the 647-sample set
  with two;
- [validation/APPLICATIONS_PERTURBATION.json](../validation/APPLICATIONS_PERTURBATION.json):
  one repeat;
- [validation/PERTURBATION_SEEDS.json](../validation/PERTURBATION_SEEDS.json):
  the full screen's agreement with SCEPTRE under several seeds;
- [validation/ENGINE_BATCH.json](../validation/ENGINE_BATCH.json): 1,000 NB and
  Poisson responses on 5,000 observations
  ([GLM_COMPARISON.md](GLM_COMPARISON.md#many-responses-with-one-design));
- [validation/CRT_SCALING.json](../validation/CRT_SCALING.json): the
  conditional randomization test on larger panels;
- [validation/UNTESTABLE_PAIRS.json](../validation/UNTESTABLE_PAIRS.json):
  requests in which most pairs fail pair QC.

No run was marked `contended`: other processes used at most 9% of a run's
CPUs in the bulk runs and 2% in the perturbation runs.

Bulk DE, one thread / eight threads, in seconds:

| Data | RapidGLM (R) | DESeq2 | DESeq2 + glmGamPoi | PyDESeq2 | PyDESeq2 + RapidGLM | Jaccard with DESeq2 (RapidGLM) |
|---|---|---|---|---|---|---:|
| airway, 8 samples | 1.02 / 0.28 | 6.09 / 2.75 | 15.4 | 53.5 / 12.5 | 3.21 / 3.36 | 0.932 |
| thyroid, 32 samples | 4.35 / 0.78 | 31.6 / 7.56 | 38.2 | 105 / 20.7 | 7.90 / 5.54 | 0.984 |
| thyroid, 128 samples | 9.56 / 1.74 | 119 / 31.3 | 86.3 | 116 / 23.0 | 13.7 / 5.54 | 0.997 |
| thyroid, 647 samples | 35.1 / 6.55 | 481 / 83.8 | 272 | 173 / 37.4 | 43.3 / 13.3 | 1.000 |

- **Speed-up over DESeq2:** 6.0×, 7.3×, 12.4× and 13.7× on one thread, and
  9.7×, 9.7×, 18.0× and 12.8× on eight.
- **Interfaces:** the CLI, Python and the PyDESeq2-style API give the same
  table as R, in similar times (on the 647-sample set 38.6, 34.7 and 35.7 s on
  one thread, against 35.1 s from R).
- **PyDESeq2:** with RapidGLM as its inference back end PyDESeq2 takes 4 to 17
  times less on one thread. Most of what remains is PyDESeq2's own data
  handling, which is why eight threads gain little.
- **Thyroid agreement** depends on the robust dispersion behind Cook's
  distances. With RapidGLM's own estimator (`legacy_cooks: false`, the default
  before 1.2.0) the Jaccard indices at 128 and 647 samples are 0.579 and 0.904
  ([BULK_DE.md](BULK_DE.md#larger-multi-factor-designs)).
- Bulk DE did not change in 1.3.0; these runs repeat those of 1.2.0 within 2%.

Perturbation, in seconds, against SCEPTRE 0.99.0. Agreement is over the
discovery pairs that both tools tested. Acceptance `ks` accepts the skew-normal
fit as SCEPTRE does; `default` is RapidGLM's stricter check with confirmation
draws:

| Data | Threads | Acceptance | RapidGLM (R / Python / CLI) | SCEPTRE | Discoveries (RapidGLM / SCEPTRE) | Jaccard | p-value Spearman |
|---|---:|---|---|---:|---|---:|---:|
| Gasperini, 50,000 cells | 1 | `ks` | 336 / 337 / 344 | 1,517 | 28 / 24 | 0.79 | 0.985 |
| | 1 | default | 431 / 428 / 436 | 1,517 | 28 / 24 | 0.79 | 0.985 |
| | 8 | `ks` | 59.4 / 52.1 / 52.8 | 251 | | | |
| | 8 | default | 75.3 / 68.4 / 68.5 | 251 | | | |
| Gasperini, 207,324 cells | 8 | `ks` | 335 / 306 / 291 | 1,473 | 96 / 101 | 0.89 | 0.986 |
| | 8 | default | 446 / 411 / 406 | 1,473 | 96 / 101 | 0.89 | 0.986 |

- **Speed-up over SCEPTRE** with `ks`, from R: 4.5× (50,000 cells, one thread),
  4.2× (eight threads) and 4.4× (207,324 cells, eight threads). With the
  defaults: 3.5×, 3.3× and 3.3×. The runs are about a tenth faster than with
  1.2.0, which fitted a null model for every response and gathered the cells of
  every pair before pair QC.
- **Agreement turns on the draws.** From 1.3.0 a pair's draws are keyed by the
  pair itself, so these runs use other draws than those of 1.2.0, which made
  29 and 101 discoveries (Jaccard 0.83 and 0.92) with the same seed. On the
  full screen six seeds give 96 to 102 discoveries and Jaccard indices
  with SCEPTRE of 0.89 to 0.96; two RapidGLM seeds overlap at
  0.90 to 0.96
  ([PERTURBATION_SEEDS.json](../validation/PERTURBATION_SEEDS.json)). With
  `stream_keys: "position"` the 1.2.0 tables are reproduced byte for byte.
- The CLI's times include parsing the MatrixMarket input: 1.4 GB for the 50,000
  cells and 6 GB for all cells.

The conditional randomization test on panels of the Gasperini screen (207,324
cells): every pair of T targets and G response genes, both drawn at random with
a fixed seed (`scripts/prepare_crt_panels.py`), with 16 threads on the 16
efficiency cores:

| Targets × genes | Pairs tested | Seconds (`ks` / default) | Draws, millions (`ks` / default) | Discoveries (`ks` / default) |
|---|---:|---|---|---|
| 50 × 1,000 | 48,596 | 123 / 128 | 29.4 / 31.0 | 6 / 6 |
| 200 × 1,000 | 193,824 | 398 / 421 | 116.8 / 122.4 | 5 / 5 |
| 100 × 7,418 | 721,266 | 1,577 / 1,662 | 437.0 / 456.7 | 25 / 14 |

On the largest panel the defaults leave 14 of the 25 discoveries made with
`ks`. Of the eleven that are dropped, direct counts of the tail contradict
five and one was not counted; they contradict none of the 14
([validation/SKEW_NORMAL_CHECK.json](../validation/SKEW_NORMAL_CHECK.json)).

Requests in which most pairs fail pair QC. A synthetic screen (20,000 cells,
300 targets and 10 non-targeting groups, 6,000 genes and 40,000 peaks, most
nonzero in 0.2–3% of cells; `scripts/benchmark_untestable_pairs.py`) in which
about 5% of the target–feature pairs pass pair QC, run with
`sequential_screen` and `early_stop_p: 0.002` on 12 threads. Wall seconds /
peak memory in GB, for every pair listed, for only the passing pairs listed,
and for the targets and responses named without a list (`targets`):

| Run (pairs requested; passing) | 1.2.0, listed | 1.2.0, passing only | 1.3.0, listed | 1.3.0, passing only | 1.3.0, named |
|---|---|---|---|---|---|
| Genes, high MOI (1.86 million; 101,564) | 45.7 / 1.97 | 41.0 / 0.40 | 42.9 / 1.36 | 39.4 / 0.37 | 38.7 / 0.34 |
| Peaks, high MOI (12.4 million; 645,408) | 202 / 11.9 | 168 / 1.08 | 192 / 7.87 | 167 / 1.16 | 167 / 1.05 |
| Genes, low MOI (1.86 million; 60,024) | 12.8 / 1.80 | 5.5 / 0.12 | 8.8 / 1.16 | 5.5 / 0.12 | 5.5 / 0.12 |
| Peaks, low MOI (12.4 million; 373,181) | 87.0 / 11.8 | 38.0 / 0.67 | 59.3 / 7.52 | 36.7 / 0.67 | 36.7 / 0.67 |

- With 1.3.0 the tested pairs have identical rows in all three; with 1.2.0
  listing only the passing pairs changed their p-values, because draws were
  keyed by a pair's place in the list.
- Naming the targets and responses costs what listing only the passing pairs
  costs, without the caller having to find them.
- A listed pair that is not tested costs about 0.57 kB of memory (0.92 kB in
  1.2.0). What a full list still costs in time is reading it and writing a row
  for every pair.
