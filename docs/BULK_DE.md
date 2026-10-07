# Bulk differential expression — 2026-09-30

`bulk-de` (C++ `bulk_de`) is a DESeq2-style workflow for bulk and pseudobulk
RNA-seq counts. It was written from the DESeq2 paper (Love, Huber and Anders
2014) and DESeq2's documentation, and validated against DESeq2 1.48.2 run as a
black box. The method is in [METHODS.md](../METHODS.md#bulk-differential-expression),
the interface in [CLI.md](CLI.md#bulk-differential-expression) and the receipt
in [validation/BULK_DE.json](../validation/BULK_DE.json).

The tables of agreement, the mock-comparison null and speed are from version
1.2.0 (2026-10-07), whose default Cook's distances use DESeq2's robust
dispersion as implemented in PyDESeq2
([below](#legacy-cooks-distances-2026-10-01)). Sections with a date record
what was measured then.

## Gates

The thresholds in `scripts/validate_bulk.py` were set after the development
comparisons below had been seen, so they document agreement rather than test a
prediction. All four pass.

- **G1, agreement:** on every dataset, size factors agree within 1e-12
  (relative), BH (0.1) discovery sets have Jaccard ≥ 0.9 and p-values have
  Spearman correlation ≥ 0.99.
- **G2, calibration:** in the Kang mock-comparison null, at most 5.5% and 1.2%
  of tests are rejected at p ≤ 0.05 and 0.01. On the simulations, the FDR at
  padj < 0.1 is within 0.02 of DESeq2's.
- **G3, determinism:** the table is byte-identical with 1 and 8 threads.
- **G4, fold-change shrinkage** (`lfc_shrink=normal` against
  `lfcShrink(type = "normal")`):
  - from DESeq2's own estimates, RapidGLM's rules reproduce its prior variance
    for the tested column within 1e-10 (relative), and its shrunken fold changes
    and posterior SDs with median differences below 1e-6;
  - end to end, shrunken fold changes have Pearson ≥ 0.99 with DESeq2's over
    genes neither tool replaced.

## Agreement with DESeq2

Datasets:
- airway: Bioconductor `airway`, 8 samples, `~ cell + dex`, genes with at least
  10 reads;
- Kang: the CD14+ monocyte pseudobulk panel, 16 samples, donor + stimulation,
  1,000 genes;
- three simulations with 4,000 genes, dispersion 0.05 + 2/mean, 10% of genes
  changed, and a single-sample outlier (×20 + 50) injected in 2% of genes. sim3
  and sim3b have 3 samples per group; sim8 has 8.

| Dataset | BH 0.1 discoveries (RapidGLM / DESeq2) | Jaccard | p-value Spearman | Trend a0, a1 (RapidGLM / DESeq2) | Prior variance |
|---|---:|---:|---:|---|---|
| airway | 5,164 / 4,834 | 0.932 | 0.9988 | 0.0102, 3.03 / 0.0097, 3.47 | 0.70 / 0.53 |
| Kang | 618 / 635 | 0.973 | 0.9980 | 0.0371, 1.35 / 0.0577, 0.45 | 0.82 / 0.88 |
| sim3 | 240 / 239 | 0.979 | 0.9993 | 0.0476, 3.17 / 0.0472, 3.22 | 0.25 / 0.25 |
| sim3b | 201 / 201 | 1.000 | 0.9994 | 0.0483, 3.40 / 0.0479, 3.44 | 0.25 / 0.25 |
| sim8 | 253 / 253 | 1.000 | 0.9999 | 0.0705, 2.05 / 0.0701, 2.06 | 0.25 / 0.25 |

Size factors agree to 4e-15. On the simulations, FDR and power match DESeq2:

| Simulation | FDR (RapidGLM / DESeq2) | Power (RapidGLM / DESeq2) | Outliers caught by Cook's (RapidGLM / DESeq2 / injected) |
|---|---|---|---|
| sim3 | 0.125 / 0.117 | 0.506 / 0.508 | 61 / 61 / 69 |
| sim3b | 0.085 / 0.085 | 0.437 / 0.437 | 82 / 82 / 97 |
| sim8 | 0.059 / 0.059 | 0.652 / 0.652 | none flagged; counts replaced in 79 / 79 genes, of 79 injected |

Neither implementation flagged a gene without an injected outlier. With eight
replicates per group both replace outlier counts and refit instead of
flagging.

## Mock-comparison null

All 70 four-versus-four donor splits of the Kang panel within a condition
(1,000 genes), each run with default settings by both tools:

| 70,000 tests | p ≤ 0.05 | p ≤ 0.01 | p ≤ 0.001 | BH (0.05) discoveries per split |
|---|---:|---:|---:|---:|
| **RapidGLM bulk-de** | 2.38% | 0.53% | 0.15% | 0.56 |
| DESeq2 1.48.2 | 2.39% | 0.54% | 0.15% | 0.47 |

## Speed

The table gives minima of three alternating runs with version 1.2.0, on
efficiency cores of the benchmark workstation
([BENCHMARK.md](BENCHMARK.md#host)). RapidGLM times the whole CLI run,
including reading counts and writing tables. DESeq2 times only `DESeq()` and
`results()`, on one thread. The application benchmark
([BENCHMARK.md](BENCHMARK.md)) times both tools from R.

| Dataset | Genes × samples | RapidGLM, 1 thread | RapidGLM, 8 threads | DESeq2 |
|---|---|---:|---:|---:|
| airway | 22,369 × 8 | 1.11 s | 0.35 s | 6.05 s |
| Kang | 1,000 × 16 | 0.10 s | 0.02 s | 1.15 s |
| sim3 | 4,000 × 6 | 0.11 s | 0.04 s | 1.32 s |
| sim3b | 4,000 × 6 | 0.11 s | 0.04 s | 1.31 s |
| sim8 | 4,000 × 16 | 0.16 s | 0.05 s | 2.75 s |

Version 1.2.0 removed work in the fits that the estimates did not need
([CHANGELOG.md](../CHANGELOG.md)): on one performance core the airway run went
from 0.98 s to 0.63 s, and the 647-sample thyroid run from 49 s to 21 s.

Earlier, six speed-ups on 2026-09-30 cut the airway run from 17.0 to 10.7
billion instructions (−37%). Under the same load, alternating runs took 1.29 s
against 1.92 s of CPU time (minima). The biggest single gain was the NB coefficient
iterations: Newton steps with the observed information replace Fisher scoring,
and the mean number of iterations of the final fits fell from 10.4 to 4.5. The
other five changes:
- the theta-only part of the NB likelihood is cached across steps at a fixed
  theta;
- the Cox-Reid factor is reused between the objective and the next Newton step;
- small factorization scratch stays on the stack;
- MatrixMarket input is parsed with `std::from_chars`;
- tables are written with `std::to_chars`.

The results are byte-identical except for the Newton steps, which move
estimates within the convergence tolerance. The final fit is also warm-started
from the initial one; that saves little. The profile of that build on
airway:
- 43% in the NB fits for the initial and final means;
- 33% in the dispersion Newton steps, half of it the Cox-Reid terms;
- 5% in the prior-variance simulation used with three or fewer residual df,
  now run in parallel;
- about 4% reading and writing files.

## Fold-change shrinkage

`lfc_shrink=normal` adds the posterior mode of the tested log2 fold change under
the paper's normal prior, and its posterior SD, as DESeq2's
`lfcShrink(type = "normal")` does for a user-supplied model matrix. Tests are
unchanged. Three details were calibrated from DESeq2's outputs:
- the weighted-quantile rule for the prior variance;
- the wide prior (variance 1e6) for the intercept. DESeq2 recognizes the
  intercept by its R name, `(Intercept)`, and RapidGLM by a constant column;
- the sandwich form of the posterior SD, with fitted counts floored at 0.5.

Given DESeq2's own maximum likelihood coefficients, dispersions and size
factors, these rules reproduce DESeq2:

| Dataset | Prior variance, relative difference | Shrunken LFC, median (max) difference | Posterior SD, median (max) relative difference |
|---|---:|---:|---:|
| airway (2,000 genes) | 6e-16 | 2.5e-9 (6.4e-7) | 3e-11 (1.7e-8) |
| Kang | 2e-15 | 6.4e-9 (0.051) | 1e-10 (0.006) |
| sim3 (2,000 genes) | 5e-15 | 2.5e-9 (0.035) | 2e-11 (0.003) |
| sim3b (2,000 genes) | 2e-15 | 2.1e-9 (0.071) | 2e-11 (0.002) |
| sim8 (2,000 genes) | 1e-14 | 1.5e-9 (6.9e-6) | 1e-11 (7e-8) |

The largest differences are in genes with only a few counts, where DESeq2's fit
stops short of the exact optimum. End to end, with RapidGLM's own estimates:

| Dataset | Tested-column prior variance (RapidGLM / DESeq2) | Pearson, shrunken LFC | Median (p99) difference |
|---|---|---:|---:|
| airway | 0.279 / 0.278 | 0.9986 | 0.002 (0.044) |
| Kang | 3.62 / 3.24 | 0.9986 | 0.002 (0.14) |
| sim3 | 1.279 / 1.278 | 0.9998 | 1e-4 (0.010) |
| sim3b | 1.337 / 1.344 | 1.0000 | 1e-4 (0.005) |
| sim8 | 0.2954 / 0.2955 | 1.0000 | 2e-5 (5e-4) |

The remaining differences come from the dispersion differences described below.
Shrinkage adds 0.10 s to airway on one thread; DESeq2's `lfcShrink` takes
1.53 s.

**Replaced counts.** For genes whose outlier counts were replaced, RapidGLM
shrinks the fold change fitted to the replaced counts, consistent with its test.
DESeq2's `lfcShrink` refits the original counts, as its outputs show: using the
original counts reproduces its values for all 79 replaced sim8 genes to 2e-7.
Its shrunken fold changes for those genes therefore still carry the outlier:
they differ from RapidGLM's by 1.4 log2 units at the median.

## Known differences

- **Gene-wise dispersions at the floor.** DESeq2 reports many genes at its
  1e-8 floor: 13,385 of 22,369 airway genes and 668 of 1,000 Kang genes.
  RapidGLM keeps the floor for genes whose rough moment estimate is at it,
  which reproduces DESeq2's output for most of them. It still starts 1,512
  airway genes above the floor where DESeq2 reports the floor (35 Kang genes).
  These genes enter the trend fit. On airway that raises the prior variance
  (0.70 against 0.53) and lowers a1 (3.03 against 3.47). On Kang it moves the
  trend (a0 0.037 against 0.058, a1 1.35 against 0.45), though the calls still
  agree with Jaccard 0.973. RapidGLM makes 330 more airway discoveries than
  DESeq2. Matching these genes would need DESeq2's line-search constants,
  which are not documented.
- **Gene-wise optimum.** Where both are above the floor, RapidGLM returns the
  maximizer of the adjusted likelihood. For 2,297 airway genes this differs
  from DESeq2's value by more than 1% (median ratio 1.04). More than half of
  all genes, most of them at the floor, agree exactly.
- **Cook's robust dispersion.** From 1.2.0 the default is DESeq2's robust
  dispersion as implemented in PyDESeq2
  ([below](#legacy-cooks-distances-2026-10-01)); it flags the genes DESeq2
  flags on sim3 and sim3b. RapidGLM's own estimator (`legacy_cooks: false`,
  METHODS.md) was chosen on sim3 and confirmed on sim3b. It catches slightly
  more injected outliers than DESeq2 (63 and 86 against 61 and 82) with no
  false flags.
- **Infinite fold changes.** When the tested coefficient has no finite estimate,
  for example a group of zeros, DESeq2 reports a large finite fold change and a
  Wald p-value. RapidGLM reports the LR test at the final dispersion and a
  missing fold change (`boundary_lrt`): 40 airway genes, 3 of them discoveries
  (DESeq2: 2).
- **Failed fits: none since the Newton steps.** Two airway genes whose
  reference cell line has only zeros used to fail the final fit. The zero cell
  pushes four coefficients to infinity together, and under Fisher scoring the
  fold change converged so slowly that the QR step lost rank first. With Newton
  steps both fit: p = 0.24 and 0.60 (DESeq2: 0.41 and 0.82).
- **Sensitivity at the margin.** With three or fewer residual df, as in airway,
  the prior variance comes from a grid search over simulated densities. Moving
  one trend gene's estimate can move it by a grid step, and the filter
  threshold is a discrete choice. With LAPACK's Cholesky in place of the
  in-house one, which changes the arithmetic only at rounding level, airway's
  prior variance moves from 0.698 to 0.702 and its filter threshold from 3.86 to
  4.50 (DESeq2: 4.50). 39 genes near padj = 0.1 change call. The other four
  datasets keep identical calls. The numerical changes of 1.2.0 had the same
  effect: four near-Poisson gene-wise dispersions settle at other values,
  the prior variance moves from 0.697 to 0.700 and the threshold from 3.86 to
  4.50, and 40 genes change call (31 gained, 9 lost).
- **Not implemented:** the `apeglm` and `ashr` shrinkage estimators,
  `betaPrior` with expanded model matrices, shrinkage of contrasts,
  `lfcThreshold` tests, custom filter statistics, observation weights, the
  locfit trend (the `local` trend is a running median), and variance-stabilizing
  transforms. Contrasts and joint LR tests are supported
  ([HYPOTHESES.md](HYPOTHESES.md)).

## Larger multi-factor designs (2026-10-01)

On GTEx v8 thyroid (`~ centre + Hardy scale + age + sex`, sex tested), agreement
with DESeq2 falls at larger sample sizes when RapidGLM's own estimator of the
robust dispersion is used (`legacy_cooks: false`; the default before 1.2.0).
Data and timings:
[validation/APPLICATIONS_BULK.json](../validation/APPLICATIONS_BULK.json)
([BENCHMARK.md](BENCHMARK.md)).

| Samples | Discoveries (RapidGLM / DESeq2) | Jaccard | Without Cook's filtering in either: Jaccard (p-value Spearman) |
|---:|---|---:|---|
| 32 | 1,590 / 1,615 | 0.963 | — |
| 128 | 463 / 792 | 0.579 | 0.971 (0.993) |
| 647 | 6,577 / 7,177 | 0.904 | 0.964 (0.982) |

The difference is in Cook's-distance flagging alone. On the 128-sample set,
size factors, dispersions, fold changes and p-values agree with DESeq2's
(median relative differences 1e-4 or less) and outlier replacement nearly
matches (281 genes against 283). RapidGLM, however, sets 751 genes aside as
Cook's outliers against DESeq2's 224. Its Cook's distances are often several
times larger.

DESeq2's paper says only that Cook's distance uses "a method-of-moments
estimate α_rob, using a robust estimator of variance s²_rob". Its manual pages
and vignette add that the variance is a "robust estimation of within-group
variance", and that outliers in groups of two samples or fewer do not count.

The values can be recovered from black-box outputs. Inverting DESeq2's Cook's
distances (its `cooks`, `mu` and hat-value assays) gives one α_rob per gene,
consistent across samples to 1e-12. Given those values, RapidGLM's flagging rule
reproduces DESeq2's flagged genes exactly (224 of 224).

What the values show:
- α_rob has a floor of exactly 0.04 (56% of airway genes, 3% of the 128-sample
  thyroid genes).
- No candidate estimator tested reproduces the values above the floor. The
  candidates were moment estimates from all samples or within design cells,
  using variances, MADs, trimmed or scaled variances, residuals about cell means
  or fitted means, and dropping the maximum. The best gives Jaccard 0.48 on
  flagged genes.

RapidGLM's own estimator was chosen on two-group simulations: squared MADs
pooled within design cells of at least three samples. With many small cells it
gives smaller dispersions, so larger distances and more flags. From 1.2.0 it is
no longer the default (next section).

## Legacy Cook's distances (2026-10-01)

PyDESeq2 (MIT; Muzellec et al. 2023) implements DESeq2's robust dispersion for
Cook's distances. With the author's approval, `legacy_cooks` reimplements that
formula and PyDESeq2's flagging rule (METHODS.md step 9; attribution and licence
in DEPENDENCIES.md).

Its dispersions equal the values recovered from DESeq2's outputs to 1e-12 on
airway and on the 128-sample thyroid set. Against DESeq2 1.48.2
([validation/COOKS_LEGACY.json](../validation/COOKS_LEGACY.json)):

| Dataset | DESeq2 discoveries | Own estimator (`legacy_cooks: false`): discoveries, Jaccard | `legacy_cooks`: discoveries, Jaccard |
|---|---:|---|---|
| airway | 4,834 | 5,142, 0.935 | 5,142, 0.935 |
| Kang | 635 | 618, 0.973 | 618, 0.973 |
| sim3 | 239 | 241, 0.975 | 240, 0.979 |
| sim3b | 201 | 200, 0.995 | 201, 1.000 |
| sim8 | 253 | 253, 1.000 | 253, 1.000 |
| GTEx thyroid, 128 samples | 792 | 463, 0.579 | 790, 0.997 |
| GTEx thyroid, 647 samples | 7,177 | 6,577, 0.904 | 7,177, 1.000 |

- **Flagged genes:** the option flags 216 genes against DESeq2's 224 on the
  128-sample set, and 61 and 82 on sim3 and sim3b, as DESeq2 does.
- **Default:** `legacy_cooks` is the default from 1.2.0. The table was
  measured with 1.1.0; with the default of 1.2.0 the airway row reads 5,164
  discoveries and Jaccard 0.932 ([Known differences](#known-differences),
  sensitivity at the margin).
- **Remaining differences:** on airway and Kang no gene is flagged by either
  tool, so the remaining differences there come from the dispersion steps
  described above.

## Engine changes

- `estimate_theta` takes an optional normal prior on log theta, which gives the
  MAP dispersions.
- Systems of at most 16 columns are now factored and solved in-house, with an
  exact reciprocal condition number, instead of through LAPACK
  ([NUMERICAL_BACKEND.md](NUMERICAL_BACKEND.md)). Concurrent small factorizations
  had serialized on OpenBLAS's buffer lock. With LAPACK, bulk-de on airway takes
  1.93 s on one thread and 3.10 s on eight; in-house, it takes 1.40 s and 0.44 s.
  Checks of this change
  ([validation/SMALL_CHOLESKY.json](../validation/SMALL_CHOLESKY.json)):
  - all 12 frozen panel cases keep their discovery sets. Statistics move by at
    most 8e-11 relative, theta in the weakly identified pseudobulk panel by
    7e-9, and p-values by at most 2.4e-12 absolute;
  - one row of the Bonferroni-integrated public S1 case reports a different
    guide. One guide's skew-normal tail probability, 1.1e-56 before, now
    evaluates to 0, which makes it the minimum-p guide in place of one at
    4e-62;
  - E1 is byte-identical;
  - the GLM accuracy suite keeps its maximum coefficient error (1.6e-5
    standard errors) and failure list;
  - the scan and QL receipts are unchanged apart from rounding-level KS
    statistics.

## Other checks

- **Unit tests:** 78 C++ cases pass, 11 of them for this module. They cover:
  - the R conventions of the median, MAD and type-7 quantile;
  - LOWESS and the gamma identity-link fit on exact lines;
  - size factors from exactly proportional genes;
  - the small Cholesky path against the LAPACK path on an embedded system;
  - the MAP prior limits;
  - recovery of a known prior variance by the simulation, at any thread count;
  - thread invariance, calibration and power on a simulated panel, with MAP
    between the gene-wise estimate and the trend;
  - all-zero genes and infinite fold changes;
  - Cook's flagging with 3 replicates and replacement with 7;
  - the weighted quantile on hand-computed cases;
  - the MAP fit's limits: maximum likelihood under a flat prior, zero under a
    tight one;
  - shrinkage toward zero with unchanged p-values, a finite shrunken change
    where the maximum likelihood change is infinite, and supplied prior
    variances.
- **CLI tests:** 18 pass, 2 of them new.
- ASan/UBSan builds are clean with OpenMP on and off.

## Provenance

The method follows published papers: Love, Huber and Anders 2014, Anders and
Huber 2010, Cox and Reid 1987, Cook 1977, Bourgon, Gentleman and Huber 2010
and Cleveland 1979. It also follows DESeq2's manual pages and vignette. No
DESeq2 or PyDESeq2 source was read. DESeq2 (LGPL ≥ 3) ran only in separate R
processes. These details were calibrated from its outputs:
- the LOWESS span (0.2);
- the `>=` rule of the filter;
- the gene set of the trend (dispersion ≥ 1e-7);
- keeping the floor for genes whose rough estimate is at it;
- the three shrinkage details above.

DESeq2's `lfcShrink` refitting the original rather than the replaced counts was
observed the same way; RapidGLM does not follow it.
