# Bulk differential expression

`bulk-de` (C++ `bulk_de`) is a DESeq2-style workflow for bulk and pseudobulk
RNA-seq counts. It follows published methods and DESeq2 documentation, with
permitted MIT PyDESeq2 ports credited in [DEPENDENCIES.md](../DEPENDENCIES.md).
DESeq2 1.48.2 is run through its public interface; no DESeq2 source is read.
The method is in [METHODS.md](../METHODS.md#bulk-differential-expression) and
the interface in [CLI.md](CLI.md#bulk-differential-expression).

Every table below comes from the complete final 1.4.0 run, completed 2026-10-22,
on the library code of `v1.4.0-dev.4`. Sources are
[BULK_DE.json](../validation/BULK_DE.json),
[COOKS_LEGACY.json](../validation/COOKS_LEGACY.json),
[COOKS_THYROID.json](../validation/COOKS_THYROID.json), and
[the release collection](../validation/receipts/v1.4.0/README.md).

## Floored fits and starting dispersion in 1.4.0

The default `min_mean: 0.5` floors fitted counts during the coefficient
iterations and uses the floored means for Wald standard errors. Starting
dispersions use the smaller rough and moments estimates, following PyDESeq2.
Bounded fits retain the covariate-unit invariance of their constraints.
A group of zero counts can receive a finite fold change and Wald test.
Changes to starting dispersions, the shared trend and prior can also move
statistics of genes whose final means exceed the floor.

Independent filtering chooses the first quantile when the maximum rejection
count is at most ten; otherwise it uses the residual RMS over positive
rejection counts. Replacement base means and failed-refit p-values follow the
validated rules in METHODS.md. The PyDESeq2 inference backend uses native
floored fits and delegates failed, bounded and optimizer-fallback cases to
PyDESeq2's DefaultInference. Optimizer differences can still change marginal calls.

## Validation gates

All four gates pass. The thresholds were set after development comparisons;
they document agreement rather than a prediction on held-out data.

- G1: size factors agree within 1e-12 relative, discovery Jaccard ≥0.9,
  and p-value Spearman ≥0.99 on each validation dataset.
- G2: Kang mock-null rejection rates are at most 5.5% and 1.2% at p≤0.05
  and p≤0.01; simulated FDR differs from DESeq2 by at most 0.02.
- G3: output tables are byte-identical at one and eight threads.
- G4: normal-prior shrinkage reproduces the reference prior variance within
  1e-10 relative from the reference estimates, with median coefficient/SD
  differences below 1e-6; end-to-end LFC Pearson correlation is at least 0.99.

## Agreement with DESeq2

Airway has eight samples and design `~ cell + dex`; the Kang monocyte
pseudobulk panel has 16 samples with donor and stimulation. Three simulations
have 4,000 genes, dispersion 0.05 + 2/mean, 10% changed genes and injected
outliers in 2% of genes. sim3/sim3b have three replicates per group; sim8 has eight.

| Dataset | Discoveries (RapidGLM / DESeq2) | Jaccard | p-value Spearman | Trend a0, a1 (RapidGLM / DESeq2) | Prior variance |
|---|---|---|---|---|---|
| airway | 4,833 / 4,834 | 0.986 | 0.999813 | 0.009821, 3.349 / 0.009717, 3.474 | 0.4975 / 0.5285 |
| Kang | 635 / 635 | 1.000 | 0.999999 | 0.05771, 0.4541 / 0.05773, 0.4542 | 0.8799 / 0.8782 |
| sim3 | 239 / 239 | 0.983 | 0.999999 | 0.04744, 3.214 / 0.04716, 3.22 | 0.25 / 0.25 |
| sim3b | 201 / 201 | 1.000 | 0.999999 | 0.04823, 3.434 / 0.04793, 3.439 | 0.25 / 0.25 |
| sim8 | 253 / 253 | 1.000 | 0.999949 | 0.07045, 2.057 / 0.07014, 2.061 | 0.25 / 0.25 |

Maximum relative size-factor difference is 4.11e-15.

| Simulation | FDR (RapidGLM / DESeq2) | Power (RapidGLM / DESeq2) | Flagged / reference / injected | Replaced (RapidGLM / DESeq2) |
|---|---|---|---|---|
| sim3 | 0.121 / 0.117 | 0.506 / 0.508 | 61 / 61 / 69 | 0 / 0 |
| sim3b | 0.085 / 0.085 | 0.437 / 0.437 | 82 / 82 / 97 | 0 / 0 |
| sim8 | 0.059 / 0.059 | 0.652 / 0.652 | 0 / 0 / 79 | 79 / 79 |

With eight replicates per group, both tools replace outlying counts and refit.

## Mock-comparison null

All 70 four-versus-four donor splits of the Kang panel within one condition.

| Tool | Tests | Valid | p≤0.05 | p≤0.01 | p≤0.001 | BH 0.05 discoveries/split |
|---|---|---|---|---|---|---|
| RapidGLM | 70000 | 69995 | 2.367% | 0.536% | 0.146% | 0.471 |
| DESeq2 1.48.2 | 70000 | 69995 | 2.392% | 0.537% | 0.146% | 0.471 |

## Speed

One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.
The runs below use one or eight efficiency cores. Quiet-start samples are
retained. These timings are final, not provisional. RapidGLM measures the
complete CLI command; DESeq2 measures `DESeq()` and `results()`. The separate
[application benchmark](BENCHMARK.md) compares the R interfaces.

| Dataset | Genes × samples | RapidGLM 1 thread (s) | RapidGLM 8 threads (s) | DESeq2 (s) |
|---|---|---|---|---|
| airway | 22,369 × 8 | 1.025 | 0.271 | 5.988 |
| Kang | 1,000 × 16 | 0.090 | 0.027 | 1.132 |
| sim3 | 4,000 × 6 | 0.105 | 0.040 | 1.319 |
| sim3b | 4,000 × 6 | 0.112 | 0.033 | 1.313 |
| sim8 | 4,000 × 16 | 0.154 | 0.042 | 2.736 |

## Fold-change shrinkage

`lfc_shrink=normal` adds the normal-prior posterior mode of the tested log2
fold change and its posterior SD, leaving the test unchanged. The prior uses
the weighted-quantile rule, an intercept variance of 1e6, and a sandwich SD
with fitted means floored at 0.5. From DESeq2's own coefficient, dispersion
and size-factor estimates:

| Dataset | Genes | Prior variance relative difference | LFC median (max) difference | SD median (max) relative difference |
|---|---|---|---|---|
| airway | 2000 | 5.55e-16 | 2.49e-09 (6.45e-07) | 2.78e-11 (1.74e-08) |
| Kang | 1000 | 2.22e-15 | 6.41e-09 (5.06e-02) | 1.33e-10 (5.94e-03) |
| sim3 | 2000 | 5.33e-15 | 2.53e-09 (3.50e-02) | 2.29e-11 (2.69e-03) |
| sim3b | 2000 | 1.78e-15 | 2.12e-09 (7.11e-02) | 1.58e-11 (1.93e-03) |
| sim8 | 2000 | 1.03e-14 | 1.53e-09 (6.89e-06) | 1.32e-11 (6.75e-08) |

End to end, with each tool's own estimates:

| Dataset | LFC Pearson | Median absolute difference | 99th percentile |
|---|---|---|---|
| airway | 0.999894 | 0.000162 | 0.0136 |
| Kang | 0.999999 | 8.65e-06 | 0.000499 |
| sim3 | 0.999982 | 4.16e-05 | 0.0069 |
| sim3b | 0.999975 | 9.42e-05 | 0.00271 |
| sim8 | 1.000000 | 2.13e-05 | 0.000432 |

On airway, shrinkage adds 0.192 s to RapidGLM; DESeq2's `lfcShrink` takes 2.720 s.

For replaced counts, RapidGLM shrinks the refitted fold change, consistent
with its test. DESeq2's black-box outputs show `lfcShrink` refitting the
original counts. This is a deliberate workflow difference.

## Larger multi-factor designs

GTEx thyroid uses `~ centre + Hardy scale + age + sex`, testing sex.
Outlying counts may still be replaced when Cook's filtering is disabled.

| Samples | Discoveries (RapidGLM / DESeq2) | Jaccard | Without Cook filtering: Jaccard (Spearman) |
|---|---|---|---|
| 32 | 1,623 / 1,615 | 0.9853 | — |
| 128 | 790 / 792 | 0.9975 | 1.0000 (0.999959) |
| 647 | 7,178 / 7,177 | 0.9999 | 1.0000 (0.999998) |

## Legacy Cook's distances

`legacy_cooks` uses the robust-dispersion formula ported from MIT PyDESeq2;
it has been the default since 1.2.0. `legacy_cooks: false` selects RapidGLM's
own estimator, based on squared MADs within design cells. Small design cells
can lead to smaller robust dispersions and more flagged genes.

| Dataset | DESeq2 discoveries | Default: discoveries, Jaccard | Own estimator: discoveries, Jaccard |
|---|---|---|---|
| airway | 4834 | 4,833, 0.9862 | 4,833, 0.9862 |
| kang | 635 | 635, 1.0000 | 635, 1.0000 |
| sim3 | 239 | 239, 0.9834 | 239, 0.9834 |
| sim3b | 201 | 201, 1.0000 | 200, 0.9950 |
| sim8 | 253 | 253, 1.0000 | 253, 1.0000 |
| thyroid128 | 792 | 790, 0.9975 | 463, 0.5806 |
| thyroid | 7177 | 7,178, 0.9999 | 6,586, 0.9046 |

On thyroid128, the default flags 219
genes and replaces counts in 67;
the own estimator flags 753
and replaces 284.
The direct public-assay comparison finds 217 shared
flags between RapidGLM and DESeq2, out of 219 and
224 respectively
([COOKS_FLAGS.json](../validation/COOKS_FLAGS.json)).

The historical robust-dispersion inversion diagnostics remain in the
[dev.3 review](REVIEW_1.4.0_R2.md). They are not reissued as final-run numerical
validation of equality between the two robust estimators.

## Replogle pseudobulk pilot

Each target is compared with non-targeting cells after aggregation by 10x
lane, with design `~ lane + perturbed`. Small perturbation groups and widely
different size factors exercise the floored-fit and independent-filtering
rules. These are 20 selected targets, not a full Replogle screen. One thread;
times below cover complete commands. Source:
[REPLOGLE_PSEUDOBULK.json](../validation/REPLOGLE_PSEUDOBULK.json).

| Target | Samples | Bulk discoveries (RapidGLM / DESeq2) | Bulk Jaccard | Bulk speedup | QL Jaccard | QL speedup |
|---|---|---|---|---|---|---|
| AARS2 | 96 | 17 / 17 | 1.0000 | 6.27× | 1.0000 | 6.57× |
| EIF6 | 93 | 1020 / 1025 | 0.9951 | 5.62× | 1.0000 | 6.29× |
| EXOC1 | 92 | 3 / 3 | 1.0000 | 5.69× | 1.0000 | 6.16× |
| FAM136A | 79 | 2056 / 2057 | 0.9947 | 4.85× | 0.9998 | 5.69× |
| GATA1 | 92 | 6062 / 6067 | 0.9988 | 4.73× | 1.0000 | 6.63× |
| HIST1H2AE | 91 | 0 / 0 | 1.0000 | 5.40× | 1.0000 | 6.31× |
| HNRNPR | 94 | 2 / 2 | 1.0000 | 5.90× | 1.0000 | 6.47× |
| ISCA2 | 96 | 69 / 69 | 1.0000 | 6.55× | 1.0000 | 6.31× |
| KANSL3 | 95 | 1375 / 1382 | 0.9949 | 6.09× | 1.0000 | 6.35× |
| MRPL16 | 88 | 132 / 132 | 1.0000 | 5.59× | 1.0000 | 6.25× |
| MRPL27 | 90 | 178 / 175 | 0.9721 | 5.70× | 1.0000 | 6.39× |
| MRPS31 | 91 | 111 / 111 | 1.0000 | 5.43× | 1.0000 | 6.30× |
| NEPRO | 75 | 1099 / 1098 | 0.9973 | 5.02× | 1.0000 | 5.56× |
| NUP98 | 96 | 2788 / 2787 | 0.9996 | 7.13× | 1.0000 | 6.42× |
| RPS20 | 96 | 2299 / 2299 | 1.0000 | 6.33× | 1.0000 | 8.87× |
| RRN3 | 74 | 909 / 909 | 0.9978 | 4.81× | 1.0000 | 5.38× |
| SAP18 | 82 | 615 / 623 | 0.9872 | 5.26× | 1.0000 | 6.20× |
| TAF1B | 90 | 31 / 31 | 1.0000 | 5.70× | 1.0000 | 6.43× |
| TRNT1 | 95 | 30 / 30 | 1.0000 | 6.28× | 1.0000 | 5.85× |
| USP8 | 87 | 5 / 5 | 1.0000 | 5.50× | 1.0000 | 6.32× |

## Known limits and checks

Dispersion optimization, the shared trend and prior, and the discrete
independent-filter threshold can change marginal calls relative to DESeq2.
Low-count and near-boundary fits also have numerical sensitivity. Agreement
on these panels does not establish identical results for every dataset.

The `apeglm` and `ashr` shrinkage estimators, expanded-model-matrix `betaPrior`,
contrast shrinkage, `lfcThreshold`, custom filters, observation weights,
locfit trends and variance-stabilizing transforms are not implemented.
Contrasts and joint likelihood-ratio tests are supported.

The final native and forced-portable C++ suites each pass all 172 cases;
C99, 28 CLI cases on both builds, Python/R bindings, Rust, selected ARM64
checks and container suites also pass. See [TESTS.json](../validation/TESTS.json)
and [ARM64.json](../validation/ARM64.json). QEMU checks establish correctness
on their declared scope and do not measure native ARM performance.
