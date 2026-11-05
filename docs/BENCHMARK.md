# Application benchmarks

RapidGLM **1.6.0** measurements on development tag `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`). All 24 stages passed the timing audit; the release library is unchanged from the measured dev.2 source.

## Host

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5. Exclusive Slurm allocation; schedutil governor, boost enabled and SMT enabled, unchanged. Each timed command starts after two ten-second quiet samples. More than 5% other CPU activity on selected cores or any sibling invalidates the command. Host load, other-user processes and vLLM CPU use are retained per command. From the resumed bulk stage on 2026-11-03, a user-authorized runtime service cpuset confines vLLM to CPUs 0-5,22-29,46-47; the reservation receipt records its application and restoration. Earlier accepted measurements retain their original guarded host conditions.

The previous 1.4.0/1.4.1 receipts used a different workstation. Compare reference/tool ratios measured on the same host; changes in absolute seconds do not isolate code changes.

The [timing audit](../validation/receipts/v1.6.0/TIMING_AUDIT.json) retains all accepted stage exits, quiet-start samples, CPU accounting and policy checks. [Reruns](../validation/receipts/v1.6.0/STAGE_RERUNS.json) retain invalidated attempts. No governor, boost, clock or SMT setting was changed.

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

## Bulk differential expression

Three rotated repeats for airway/thyroid32/thyroid128 and two for thyroid647; minima reported. CLI times include I/O; API times start after loading. Ratios use the same dataset and thread count where available. [Receipt](../validation/APPLICATIONS_BULK.json).

| Dataset | Tool | Threads | Seconds | Reference / tool | Discoveries | Jaccard with reference |
| --- | --- | --- | --- | --- | --- | --- |
| airway | rapidglm-cli | 1 | 0.888 | 5.77× | 4833 | 0.9862 |
| airway | rapidglm-cli | 8 | 0.267 | 9.27× | 4833 | 0.9862 |
| airway | rapidglm-python | 1 | 0.874 | 5.87× | 4833 | 0.9862 |
| airway | rapidglm-python | 8 | 0.28 | 8.83× | 4833 | 0.9862 |
| airway | rapidglm-deseq | 1 | 0.94 | 5.45× | 4833 | 0.9862 |
| airway | rapidglm-deseq | 8 | 0.309 | 8.01× | 4833 | 0.9862 |
| airway | rapidglm-r | 1 | 0.847 | 6.05× | 4833 | 0.9862 |
| airway | rapidglm-r | 8 | 0.244 | 10.1× | 4833 | 0.9862 |
| airway | pydeseq2 | 1 | 41.3 | 0.124× | 4690 | 0.9315 |
| airway | pydeseq2 | 8 | 9.47 | 0.261× | 4690 | 0.9315 |
| airway | pydeseq2-rapidglm | 1 | 2.35 | 2.18× | 4663 | 0.9295 |
| airway | pydeseq2-rapidglm | 8 | 2.01 | 1.23× | 4663 | 0.9295 |
| airway | deseq2 | 1 | 5.12 | 1× | 4834 | 1 |
| airway | deseq2 | 8 | 2.48 | 1× | 4834 | 1 |
| airway | deseq2-glmgampoi | 1 | 12.3 | 0.415× | 8452 | 0.5697 |
| airway | rapidglm-edger-cli | 1 | 0.253 | 2.44× | 6568 | 1 |
| airway | rapidglm-edger-cli | 8 | 0.269 | — | 6568 | 1 |
| airway | edger | 1 | 0.617 | 8.31× | 6568 | 0.7239 |
| thyroid32 | rapidglm-cli | 1 | 4.08 | 6.25× | 1620 | 0.9871 |
| thyroid32 | rapidglm-cli | 8 | 0.736 | 9.36× | 1620 | 0.9871 |
| thyroid32 | rapidglm-python | 1 | 3.92 | 6.5× | 1620 | 0.9871 |
| thyroid32 | rapidglm-python | 8 | 0.727 | 9.48× | 1620 | 0.9871 |
| thyroid32 | rapidglm-deseq | 1 | 4.02 | 6.34× | 1620 | 0.9871 |
| thyroid32 | rapidglm-deseq | 8 | 0.767 | 8.98× | 1620 | 0.9871 |
| thyroid32 | rapidglm-r | 1 | 4 | 6.37× | 1620 | 0.9871 |
| thyroid32 | rapidglm-r | 8 | 0.673 | 10.2× | 1620 | 0.9871 |
| thyroid32 | pydeseq2 | 1 | 79.4 | 0.321× | 1637 | 0.9853 |
| thyroid32 | pydeseq2 | 8 | 15.5 | 0.444× | 1637 | 0.9853 |
| thyroid32 | pydeseq2-rapidglm | 1 | 9.61 | 2.65× | 1642 | 0.9823 |
| thyroid32 | pydeseq2-rapidglm | 8 | 4.57 | 1.51× | 1642 | 0.9823 |
| thyroid32 | deseq2 | 1 | 25.5 | 1× | 1615 | 1 |
| thyroid32 | deseq2 | 8 | 6.89 | 1× | 1615 | 1 |
| thyroid32 | deseq2-glmgampoi | 1 | 31.2 | 0.817× | 2051 | 0.7727 |
| thyroid32 | rapidglm-edger-cli | 1 | 1.14 | 4.44× | 1364 | 0.9834 |
| thyroid32 | rapidglm-edger-cli | 8 | 0.525 | — | 1364 | 0.9834 |
| thyroid32 | edger | 1 | 5.06 | 5.03× | 1387 | 0.7403 |
| thyroid128 | rapidglm-cli | 1 | 8.28 | 11.6× | 790 | 0.9975 |
| thyroid128 | rapidglm-cli | 8 | 1.55 | 17× | 790 | 0.9975 |
| thyroid128 | rapidglm-python | 1 | 7.97 | 12× | 790 | 0.9975 |
| thyroid128 | rapidglm-python | 8 | 1.37 | 19.2× | 790 | 0.9975 |
| thyroid128 | rapidglm-deseq | 1 | 8.12 | 11.8× | 790 | 0.9975 |
| thyroid128 | rapidglm-deseq | 8 | 1.51 | 17.4× | 790 | 0.9975 |
| thyroid128 | rapidglm-r | 1 | 8.27 | 11.6× | 790 | 0.9975 |
| thyroid128 | rapidglm-r | 8 | 1.51 | 17.4× | 790 | 0.9975 |
| thyroid128 | pydeseq2 | 1 | 87.6 | 1.1× | 792 | 0.9975 |
| thyroid128 | pydeseq2 | 8 | 17.1 | 1.53× | 792 | 0.9975 |
| thyroid128 | pydeseq2-rapidglm | 1 | 15.4 | 6.23× | 787 | 0.9912 |
| thyroid128 | pydeseq2-rapidglm | 8 | 6 | 4.38× | 787 | 0.9912 |
| thyroid128 | deseq2 | 1 | 95.9 | 1× | 792 | 1 |
| thyroid128 | deseq2 | 8 | 26.3 | 1× | 792 | 1 |
| thyroid128 | deseq2-glmgampoi | 1 | 71.7 | 1.34× | 841 | 0.9257 |
| thyroid128 | rapidglm-edger-cli | 1 | 2.25 | 7.56× | 822 | 0.9904 |
| thyroid128 | rapidglm-edger-cli | 8 | 0.948 | — | 822 | 0.9904 |
| thyroid128 | edger | 1 | 17 | 5.65× | 830 | 0.7348 |
| thyroid | rapidglm-cli | 1 | 29.4 | 13.6× | 7178 | 0.9999 |
| thyroid | rapidglm-cli | 8 | 5.29 | 13.3× | 7178 | 0.9999 |
| thyroid | rapidglm-python | 1 | 28.9 | 13.9× | 7178 | 0.9999 |
| thyroid | rapidglm-python | 8 | 4.88 | 14.4× | 7178 | 0.9999 |
| thyroid | rapidglm-deseq | 1 | 29.4 | 13.6× | 7178 | 0.9999 |
| thyroid | rapidglm-deseq | 8 | 5.43 | 12.9× | 7178 | 0.9999 |
| thyroid | rapidglm-r | 1 | 30 | 13.4× | 7178 | 0.9999 |
| thyroid | rapidglm-r | 8 | 5.54 | 12.7× | 7178 | 0.9999 |
| thyroid | pydeseq2 | 1 | 127 | 3.15× | 7177 | 1 |
| thyroid | pydeseq2 | 8 | 24.4 | 2.88× | 7177 | 1 |
| thyroid | pydeseq2-rapidglm | 1 | 45.7 | 8.77× | 7176 | 0.9999 |
| thyroid | pydeseq2-rapidglm | 8 | 12.5 | 5.6× | 7176 | 0.9999 |
| thyroid | deseq2 | 1 | 401 | 1× | 7177 | 1 |
| thyroid | deseq2 | 8 | 70.2 | 1× | 7177 | 1 |
| thyroid | deseq2-glmgampoi | 1 | 232 | 1.73× | 7168 | 0.9893 |
| thyroid | rapidglm-edger-cli | 1 | 8.17 | 9.52× | 7481 | 0.9995 |
| thyroid | rapidglm-edger-cli | 8 | 3.34 | — | 7481 | 0.9995 |
| thyroid | edger | 1 | 77.7 | 5.16× | 7479 | 0.9168 |

## High-MOI perturbation

One measured run per tool/thread/mode. The ks variants match SCEPTRE skew-normal acceptance; plain RapidGLM names use the stricter default with confirmation draws. API and complete-command timings are separate. [Receipt](../validation/APPLICATIONS_PERTURBATION.json).

| Dataset | Tool | Threads | Seconds | SCEPTRE / tool | Discoveries | Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| gasperini50k | rapidglm-cli | 1 | 260 | 5.3× | 24 | 0.8462 |
| gasperini50k | rapidglm-cli | 8 | 35.9 | 5.75× | 24 | 0.8462 |
| gasperini50k | rapidglm-python | 1 | 264 | 5.23× | 24 | 0.8462 |
| gasperini50k | rapidglm-python | 8 | 35.6 | 5.8× | 24 | 0.8462 |
| gasperini50k | rapidglm-r | 1 | 300 | 4.6× | 24 | 0.8462 |
| gasperini50k | rapidglm-r | 8 | 42.5 | 4.86× | 24 | 0.8462 |
| gasperini50k | rapidglm-cli-ks | 1 | 222 | 6.21× | 24 | 0.8462 |
| gasperini50k | rapidglm-cli-ks | 8 | 30.9 | 6.68× | 24 | 0.8462 |
| gasperini50k | rapidglm-python-ks | 1 | 223 | 6.2× | 24 | 0.8462 |
| gasperini50k | rapidglm-python-ks | 8 | 30.3 | 6.83× | 24 | 0.8462 |
| gasperini50k | rapidglm-r-ks | 1 | 246 | 5.6× | 24 | 0.8462 |
| gasperini50k | rapidglm-r-ks | 8 | 35.9 | 5.75× | 24 | 0.8462 |
| gasperini50k | sceptre | 1 | 1.38e+03 | 1× | 24 | 1 |
| gasperini50k | sceptre | 8 | 207 | 1× | 24 | 1 |
| gasperini | rapidglm-cli | 8 | 223 | 4.98× | 98 | 0.8952 |
| gasperini | rapidglm-python | 8 | 208 | 5.34× | 98 | 0.8952 |
| gasperini | rapidglm-r | 8 | 238 | 4.65× | 98 | 0.8952 |
| gasperini | rapidglm-cli-ks | 8 | 153 | 7.24× | 98 | 0.8952 |
| gasperini | rapidglm-python-ks | 8 | 152 | 7.27× | 98 | 0.8952 |
| gasperini | rapidglm-r-ks | 8 | 182 | 6.1× | 98 | 0.8952 |
| gasperini | sceptre | 8 | 1.11e+03 | 1× | 101 | 1 |

## Low-MOI perturbation

The Replogle K562 essential pilot uses the same public inputs and covariates for both tools, with sixteen cores. Times cover whole commands; one SCEPTRE run supplies both acceptance comparisons.

| Acceptance | RapidGLM seconds | SCEPTRE seconds | Speedup | RapidGLM discoveries | SCEPTRE discoveries | Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| ks | 226.9 | 2159 | 9.52× | 10982 | 10972 | 0.9492 |
| strict | 1058.1 | 2159 | 2.04× | 10979 | 10972 | 0.9495 |

## CRT panel scaling

Nested target/gene panels from the 207,324-cell Gasperini screen, sixteen threads. [Receipt](../validation/CRT_SCALING.json).

| Targets × genes | Fit check | Pairs tested | Seconds | Draws (millions) | Discoveries |
| --- | --- | --- | --- | --- | --- |
| 50 × 1000 | ks | 48,596 | 54.92 | 29.43 | 5 |
| 50 × 1000 | strict | 48,596 | 55.32 | 31.08 | 6 |
| 200 × 1000 | ks | 193,824 | 166.3 | 116.6 | 7 |
| 200 × 1000 | strict | 193,824 | 178.1 | 122.5 | 5 |
| 100 × 7418 | ks | 721,266 | 645.9 | 437 | 20 |
| 100 × 7418 | strict | 721,266 | 673 | 455.4 | 14 |

## Requests with mostly untestable pairs

Synthetic 20,000-cell panels, twelve cores, sequential_screen enabled and early_stop_p 0.002. The three request forms must produce identical tested records. [Receipt](../validation/UNTESTABLE_PAIRS.json).

| Panel | Pairs requested | Passing QC | All listed (s / GB) | Passing listed (s / GB) | Named targets (s / GB) |
| --- | --- | --- | --- | --- | --- |
| rna_high | 1,860,000 | 101,564 | 25.34 / 1.43 | 21.04 / 0.44 | 20.98 / 0.43 |
| atac_high | 12,400,000 | 645,408 | 116.7 / 7.9 | 91.14 / 1.16 | 91.04 / 1.06 |
| rna_low | 1,860,000 | 60,024 | 7.88 / 1.17 | 3.77 / 0.13 | 3.73 / 0.13 |
| atac_low | 12,400,000 | 373,181 | 52.83 / 7.53 | 25.21 / 0.67 | 25.01 / 0.67 |

## Full-screen applications

Gaussian Geuvadis cis-QTL, 10,000 permutations:

| Scope | RapidGLM seconds | tensorQTL seconds | tensorQTL / RapidGLM |
| --- | --- | --- | --- |
| Chromosome 22 | 35.86 | 107 | 2.983× |
| Genome-wide | 1212 | 3069 | 2.531× |

See [QTL details](QTL_GAUSSIAN_SPEED.md) for precision, eGene identities, budgets and memory.

Full Replogle pseudobulk, sixteen single-thread workers:

| Workflow | Reference / RapidGLM screen time | Median discovery Jaccard |
| --- | --- | --- |
| DESeq2-style bulk DE | 5.123× | 1 |
| edgeR-style quasi-likelihood | 7.544× | 1 |

The [full receipt](../validation/REPLOGLE_PSEUDOBULK_FULL.json) records every perturbation, filtering differences and pilot/full identity. Across-seed CRT agreement is retained in [PERTURBATION_SEEDS.json](../validation/PERTURBATION_SEEDS.json).

## Special functions and division modes

Both slow (default) and fast were measured on Zen 3 core 6, with its sibling idle. C++ accuracy checks are untimed; C++ kernels and Python/R arrays have guarded timings. [SPECIAL_FUNCTIONS.json](../validation/SPECIAL_FUNCTIONS.json) retains both modes; [VMATH.json](../validation/VMATH.json) records formal end-to-end ratios and measured alternatives. Historical Gracemont development measurements remain labelled separately. See the [METHODS backend table](../METHODS.md#choice-of-backends) for selections and rejected alternatives.

## Optimization ablations

| Pair | Threads | Opt-out seconds | Default seconds | Opt-out / default |
| --- | --- | --- | --- | --- |
| Replogle strict: cold / refine | 16 | 1,321.3 | 1,058.1 | 1.249× |
| Replogle ks: cold / refine | 16 | 521.1 | 226.9 | 2.297× |
| CRT ks: batch scoring off / on | 16 | 1,012.7 | 645.9 | 1.568× |
| CRT default: batch scoring off / on | 16 | 1,041.0 | 673.0 | 1.547× |
| NB QTL: separate / lockstep | 12 | 541.5 | 498.1 | 1.087× |

The formal Zen 3 pilot uses 16 threads. With `ks`, cold/refine wall times are 521.1 / 226.9 s (2.297×); with `strict`, 1,321.3 / 1,058.1 s (1.249×). Statuses, empirical p-values and discoveries are identical in each pair and to job 687. Tail approximations can move in their last digits: for `ks`, 72 of 16057 approximated p-values differ by more than 1e-4 relative.

On the formal Zen 3 run (16 threads, 100 targets × 7,418 genes), enabling batch scoring gives 1.568× for the `ks` panel (1,012.7 / 645.9 s off/on) and 1.547× for the default panel (1,041.0 / 673.0 s). Result and QQ tables are byte-identical in both pairs. This compares the current batch pipeline with the pair pipeline; it does not separately time cell-major scoring or the caches across tiles.

On the formal Zen 3 run (12 threads, chr22, 1,000 permutations), NB took 498.1 s with lockstep refits and 541.5 s with separate refits (1.087×). Statuses, leads, statistics and empirical p-values agree with job 687; the largest relative Beta p-value difference is 7.89e-11. Both runs have the same 168 Storey q < 0.05 eGenes and 168 BH q <= 0.05 eGenes. The separate table is byte-identical to job 687. This pair isolates refit batching; both settings use the new block scores.

[Receipt](../validation/OPTIMIZATION_ABLATIONS.json).

## Changes in reference-tool ratios

Each entry divides reference time by RapidGLM time on the same host. The change compares these ratios between the 1.4.0 and 1.6.0 runs; it does not compare seconds across machines.

| Analysis | Reference | Threads | 1.4.0 ratio | 1.6.0 ratio | Ratio change |
| --- | --- | --- | --- | --- | --- |
| airway rapidglm-r | deseq2 | 1 | 6.283× | 6.051× | -3.7% |
| airway rapidglm-r | deseq2 | 8 | 11.055× | 10.148× | -8.2% |
| thyroid32 rapidglm-r | deseq2 | 1 | 6.448× | 6.369× | -1.2% |
| thyroid32 rapidglm-r | deseq2 | 8 | 9.346× | 10.233× | +9.5% |
| thyroid128 rapidglm-r | deseq2 | 1 | 11.996× | 11.593× | -3.4% |
| thyroid128 rapidglm-r | deseq2 | 8 | 17.882× | 17.376× | -2.8% |
| thyroid rapidglm-r | deseq2 | 1 | 13.754× | 13.379× | -2.7% |
| thyroid rapidglm-r | deseq2 | 8 | 12.867× | 12.676× | -1.5% |
| airway rapidglm-edger-cli | edger | 1 | 2.755× | 2.443× | -11.3% |
| thyroid32 rapidglm-edger-cli | edger | 1 | 3.987× | 4.437× | +11.3% |
| thyroid128 rapidglm-edger-cli | edger | 1 | 6.170× | 7.563× | +22.6% |
| thyroid rapidglm-edger-cli | edger | 1 | 6.967× | 9.515× | +36.6% |
| gasperini50k rapidglm-r | sceptre | 1 | 3.558× | 4.599× | +29.3% |
| gasperini50k rapidglm-r | sceptre | 8 | 3.334× | 4.860× | +45.8% |
| gasperini50k rapidglm-r-ks | sceptre | 1 | 4.578× | 5.601× | +22.4% |
| gasperini50k rapidglm-r-ks | sceptre | 8 | 4.212× | 5.747× | +36.4% |
| gasperini rapidglm-r | sceptre | 8 | 3.339× | 4.648× | +39.2% |
| gasperini rapidglm-r-ks | sceptre | 8 | 4.556× | 6.095× | +33.8% |
| matched_gaussian_speedup | matched reference | 12 | 1.293× | 2.983× | +130.7% |
| matched_nb_score_speedup | matched reference | 12 | 2.848× | 4.532× | +59.1% |
| Replogle ks | SCEPTRE | 16 | 4.384× | 9.515× | +117.0% |
| Replogle strict | SCEPTRE | 16 | 1.631× | 2.041× | +25.1% |

GLM engine, solver, Rust and PyDESeq2-backend ratios are also retained in [HEADLINE_COMPARISON.json](../validation/receipts/v1.6.0/HEADLINE_COMPARISON.json). The two full-screen benchmarks first appeared in 1.4.1; their receipts retain that baseline separately.
