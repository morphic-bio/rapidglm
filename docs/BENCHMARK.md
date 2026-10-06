# Application benchmark (app note)

RapidGLM's two applications against their reference implementations, run as
black boxes through their public interfaces:
- bulk differential expression against DESeq2 and PyDESeq2;
- the perturbation workflow against SCEPTRE.

Driver: `scripts/benchmark_applications.py`. Data preparation:
`scripts/prepare_benchmark_data.py`. The receipt, with timings, host load and
agreement, is written wherever `--output` points. Final numbers need a quiet
host: runs whose 1-minute load exceeds a quarter of the cores are marked
`contended`.

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

## Results (2026-10-01, quiet host)

Receipts:
- [validation/APPLICATIONS_BULK.json](../validation/APPLICATIONS_BULK.json):
  airway and the 32-sample set with three repeats, the 128- and 647-sample sets
  with one;
- [validation/APPLICATIONS_PERTURBATION.json](../validation/APPLICATIONS_PERTURBATION.json):
  one repeat;
- [validation/ENGINE_BATCH.json](../validation/ENGINE_BATCH.json): 1,000 NB and
  Poisson responses on 5,000 observations; re-timed 2026-10-01 at load about 2-4
  with glmGamPoi added (`glm_gp` without Cox-Reid or shrinkage, 3.15 s against
  1.96 s for `fit_batch` at 1 thread and 0.17 s at 16).

The 1-minute load was 1–4 during the bulk runs and 2–9 during the perturbation
runs.

Bulk DE, one thread / eight threads, in seconds:

| Data | RapidGLM (R) | DESeq2 | DESeq2 + glmGamPoi | PyDESeq2 | PyDESeq2 + RapidGLM | Jaccard with DESeq2 (RapidGLM) |
|---|---|---|---|---|---|---:|
| airway, 8 samples | 0.88 / 0.24 | 3.49 / 1.60 | 8.19 | 21.5 / 5.83 | 1.97 / 1.84 | 0.935 |
| thyroid, 32 samples | 4.27 / 0.73 | 17.5 / 4.35 | 21.6 | 41.8 / 9.63 | 5.91 / 3.20 | 0.963 |
| thyroid, 128 samples | 11.9 / 1.98 | 61.9 / 17.2 | 45.7 | 48.3 / 11.2 | 13.3 / 4.31 | 0.579 |
| thyroid, 647 samples | 52.6 / 8.52 | 254 / 45.2 | 143 | 78.6 / 17.1 | 50.7 / 11.3 | 0.904 |

- **Interfaces:** the CLI, Python and the PyDESeq2-style API give the same
  table as R, in similar times.
- **Thyroid agreement:** the lower Jaccard indices at 128 and 647 samples come
  from Cook's-distance flagging. Without it, they are 0.971 and 0.964
  ([BULK_DE.md](BULK_DE.md#larger-multi-factor-designs-2026-10-01)).

Perturbation, in seconds, against SCEPTRE 0.99.0. Agreement is over the
discovery pairs that both tools tested:

| Data | Threads | RapidGLM (R / Python / CLI) | SCEPTRE | Discoveries (RapidGLM / SCEPTRE) | Jaccard | p-value Spearman |
|---|---:|---|---:|---|---:|---:|
| Gasperini, 50,000 cells | 1 | 258 / 244 / 264 | 851 | 29 / 24 | 0.83 | 0.985 |
| | 8 | 38 / 35 / 53 | 141 | | | |
| Gasperini, 207,324 cells | 8 | 274 / 252 / 333 | 896 | 101 / 101 | 0.92 | 0.986 |

The CLI's times include parsing the MatrixMarket input: 1.4 GB for the 50,000
cells and 6 GB for all cells.
