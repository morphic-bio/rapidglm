# Bulk differential expression

`bulk-de` (C++ `bulk_de`) is a DESeq2-style workflow for bulk and pseudobulk
RNA-seq counts. It follows published methods and DESeq2 documentation, with
permitted MIT PyDESeq2 ports credited in [DEPENDENCIES.md](../DEPENDENCIES.md).
DESeq2 1.48.2 is run through its public interface; no DESeq2 source is read.
The method is in [METHODS.md](../METHODS.md#bulk-differential-expression) and
the interface in [CLI.md](CLI.md#bulk-differential-expression).

All current tables below report completed 1.6.0 measurements on `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`). The release library is unchanged from the measured dev.2 source. The [receipt collection](../validation/receipts/v1.6.0/README.md) retains tests and timing evidence.

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

Airway, Kang monocyte pseudobulk and three simulations exercise size factors, dispersion, outlier handling and independent filtering. Sources: [BULK_DE.json](../validation/BULK_DE.json).

| Dataset | Genes | Discoveries (RapidGLM / DESeq2) | Jaccard | P-value Spearman | Median relative dispersion difference |
| --- | --- | --- | --- | --- | --- |
| airway | 22369 | 4833 / 4834 | 0.9862 | 0.999813 | 0.0102 |
| kang | 1000 | 635 / 635 | 1 | 0.999999 | 0.000822 |
| sim3 | 4000 | 239 / 239 | 0.9834 | 0.999999 | 0.00204 |
| sim3b | 4000 | 201 / 201 | 1 | 0.999999 | 0.00218 |
| sim8 | 4000 | 253 / 253 | 1 | 0.999949 | 0.00125 |

## Mock-comparison null

All seventy four-versus-four donor splits within one condition on the Kang panel:

| Tool | Valid tests | P < .05 | P < .01 | P < .001 | Mean BH .05 discoveries per split |
| --- | --- | --- | --- | --- | --- |
| rapidglm | 69995 | 0.023673 | 0.0053575 | 0.0014572 | 0.47143 |
| deseq2 | 69995 | 0.023916 | 0.0053718 | 0.0014572 | 0.47143 |

## Speed

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5.

The following CLI timings include input/output; DESeq2 timings cover DESeq() and results(). The application benchmark separately compares the R interfaces.

| Dataset | Genes | RapidGLM 1 thread (s) | RapidGLM 8 threads (s) | DESeq2 (s) |
| --- | --- | --- | --- | --- |
| airway | 22369 | 0.89199 | 0.25149 | 5.129 |
| kang | 1000 | 0.078312 | 0.025862 | 0.979 |
| sim3 | 4000 | 0.09661 | 0.07124 | 1.148 |
| sim3b | 4000 | 0.096956 | 0.055611 | 1.139 |
| sim8 | 4000 | 0.13904 | 0.046592 | 2.458 |

## Fold-change shrinkage

The normal-prior posterior mode and SD use the validated rules in METHODS. From fixed DESeq2 estimates, the formula comparison is:

| Dataset | Genes | Prior variance relative difference | Median / maximum LFC difference | Median / maximum SD relative difference |
| --- | --- | --- | --- | --- |
| airway | 2000 | 5.5511e-16 | 2.4916e-09 / 6.4455e-07 | 2.7829e-11 / 1.7435e-08 |
| kang | 1000 | 2.2204e-15 | 6.4081e-09 / 0.050622 | 1.33e-10 / 0.005935 |
| sim3 | 2000 | 5.3291e-15 | 2.5326e-09 / 0.034965 | 2.2861e-11 / 0.0026939 |
| sim3b | 2000 | 1.7764e-15 | 2.1216e-09 / 0.071089 | 1.5792e-11 / 0.0019306 |
| sim8 | 2000 | 1.0325e-14 | 1.5277e-09 / 6.8868e-06 | 1.3199e-11 / 6.7486e-08 |

End to end, using each tool’s own estimates:

| Dataset | LFC Pearson | Median absolute difference | 99th percentile |
| --- | --- | --- | --- |
| airway | 0.9998945 | 0.00016244 | 0.013635 |
| kang | 0.9999993 | 8.6546e-06 | 0.00049883 |
| sim3 | 0.9999817 | 4.1555e-05 | 0.0068981 |
| sim3b | 0.9999751 | 9.4164e-05 | 0.0027111 |
| sim8 | 1 | 2.1279e-05 | 0.00043222 |

For replaced counts, RapidGLM shrinks the refitted effect; the DESeq2 black-box output refits the original counts. This deliberate workflow difference remains.

## Larger multi-factor designs and Cook's distances

The leverage-one rule treats h within four double epsilons of one as exactly one, with undefined Cook distance. Saturation alone does not flag a gene; a finite above-cutoff distance in an eligible replicated cell is still required. An undefined saturated maximum prevents the legacy count-rank exemption. All sixteen required regression genes are flagged in the current one- and eight-thread tables and have missing DESeq2 p-values. [Current evidence](../validation/COOKS_REGRESSION.json).

The earlier alternatives and isolated pre-fix comparisons remain in [COOKS_LEVERAGE.json](../validation/COOKS_LEVERAGE.json); those are development evidence.

| Dataset | DESeq2 discoveries | Default discoveries / Jaccard | Own estimator discoveries / Jaccard |
| --- | --- | --- | --- |
| airway | 4834 | 4833 / 0.98623 | 4833 / 0.98623 |
| kang | 635 | 635 / 1 | 635 / 1 |
| sim3 | 239 | 239 / 0.9834 | 239 / 0.9834 |
| sim3b | 201 | 201 / 1 | 200 / 0.99502 |
| sim8 | 253 | 253 / 1 | 253 / 1 |
| thyroid128 | 792 | 790 / 0.99747 | 463 / 0.5806 |
| thyroid | 7177 | 7178 / 0.99986 | 6586 / 0.90465 |

## Replogle pseudobulk

The twenty-target pilot aggregates by lane, compares each perturbation with non-targeting cells and uses design ~ lane + perturbed. Complete-command times use one thread.

| Target | Samples | Bulk discoveries RapidGLM / DESeq2 | Bulk Jaccard | Bulk speedup | QL Jaccard | QL speedup |
| --- | --- | --- | --- | --- | --- | --- |
| AARS2 | 96 | 17 / 17 | 1 | 5.4606× | 1 | 8.1108× |
| EIF6 | 93 | 1020 / 1025 | 0.9951 | 5.1003× | 1 | 7.8563× |
| EXOC1 | 92 | 3 / 3 | 1 | 5.3944× | 1 | 7.6548× |
| FAM136A | 79 | 2056 / 2057 | 0.9947 | 4.4474× | 0.9998 | 6.8193× |
| GATA1 | 92 | 6062 / 6067 | 0.9988 | 4.4867× | 1 | 8.1901× |
| HIST1H2AE | 91 | 0 / 0 | 1 | 5.0602× | 1 | 7.6563× |
| HNRNPR | 94 | 2 / 2 | 1 | 5.2051× | 1 | 7.9399× |
| ISCA2 | 96 | 69 / 69 | 1 | 5.656× | 1 | 7.7579× |
| KANSL3 | 95 | 1375 / 1382 | 0.9949 | 5.2568× | 1 | 7.8344× |
| MRPL16 | 88 | 132 / 132 | 1 | 5.0161× | 1 | 7.7125× |
| MRPL27 | 90 | 178 / 175 | 0.9721 | 4.9614× | 1 | 7.7445× |
| MRPS31 | 91 | 111 / 111 | 1 | 5.1136× | 1 | 7.6× |
| NEPRO | 75 | 1099 / 1098 | 0.9973 | 4.5884× | 1 | 6.7346× |
| NUP98 | 96 | 2788 / 2787 | 0.9996 | 6.2227× | 1 | 7.9111× |
| RPS20 | 96 | 2299 / 2299 | 1 | 5.4809× | 1 | 10.91× |
| RRN3 | 74 | 909 / 909 | 0.9978 | 4.3942× | 1 | 6.4217× |
| SAP18 | 82 | 615 / 623 | 0.9872 | 4.696× | 1 | 7.3301× |
| TAF1B | 90 | 31 / 31 | 1 | 4.9834× | 1 | 7.7764× |
| TRNT1 | 95 | 30 / 30 | 1 | 5.4297× | 1 | 7.1813× |
| USP8 | 87 | 5 / 5 | 1 | 4.9727× | 1 | 7.5737× |

The separate full-screen run uses sixteen single-thread workers per tool. Screen speedups are 5.1228× over DESeq2 and 7.5435× over edgeR; median discovery Jaccards are 1 and 1. [The full receipt](../validation/REPLOGLE_PSEUDOBULK_FULL.json) retains every perturbation, filtering discrepancies and pilot/full identity.

## Known limits and checks

Dispersion optimization, the shared trend and prior, and the discrete
independent-filter threshold can change marginal calls relative to DESeq2.
Low-count and near-boundary fits also have numerical sensitivity. Agreement
on these panels does not establish identical results for every dataset.

The `apeglm` and `ashr` shrinkage estimators, expanded-model-matrix `betaPrior`,
contrast shrinkage, `lfcThreshold`, custom filters, observation weights,
locfit trends and variance-stabilizing transforms are not implemented.
Contrasts and joint likelihood-ratio tests are supported.

The GPU-host native, CLI, Python/R binding and container checks passed in both division modes. See [TESTS.json](../validation/TESTS.json). The runbook records the earlier full functional suite on the same source; the coordinator’s separate full ARM64 run is outside this receipt collection.
