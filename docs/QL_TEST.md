# Pseudobulk quasi-likelihood test — 2026-09-30

`pseudobulk-de` has a quasi-likelihood (QL) F test with empirical Bayes
dispersion shrinkage (`test=ql`). It is the default for `family=nb`. It closes
a calibration gap found in review: per-gene ML LR and Wald tests were
anticonservative with eight replicates. The method is
described in [METHODS.md](../METHODS.md) and the interface in [CLI.md](CLI.md).
The receipt is [validation/QL_TEST.json](../validation/QL_TEST.json).

## Gate

The thresholds below were fixed in `scripts/validate_ql.py` before the first QL
run and were not tuned.

- **G1, calibration:** in the Kang mock-comparison null (all 70
  four-versus-four donor splits within a condition, 1,000 genes), QL rejects at
  most 5.5%, 1.2% and 0.15% of null tests at p ≤ 0.05, 0.01 and 0.001. It makes
  at most 0.5 BH (0.05) discoveries per split on average and returns valid
  p-values for at least 99.5% of tests.
- **G2, power:** on the real donor-paired stimulation contrast, QL BH (0.05)
  discoveries agree with glmGamPoi's QL test: Jaccard ≥ 0.8, and the ratio of
  discovery counts lies in [0.8, 1.25].
- **G3, determinism:** the QL table is byte-identical with 1 and 4 threads.

All three pass.

| Mock null (70,000 tests) | p ≤ 0.05 | p ≤ 0.01 | p ≤ 0.001 | BH discoveries per split |
|---|---:|---:|---:|---:|
| **RapidGLM QL** | 4.0% | 0.73% | 0.10% | 0.14 |
| glmGamPoi 1.25.1 QL | 3.9% | 0.72% | 0.11% | 0.14 |
| edgeR 4.6.2 QL | 4.7% | 0.88% | 0.11% | 0.16 |
| RapidGLM ML, LR | 7.4% | 1.6% | 0.19% | 0.14 |
| RapidGLM ML, Wald | 9.6% | 3.3% | 0.90% | 6.1 |
| edgeR LRT | 5.1% | 1.3% | 0.27% | 1.1 |
| DESeq2 1.48.2 Wald | 2.6% | 0.58% | 0.14% | 0.49 |

RapidGLM QL returns valid p-values for 69,962 of the 70,000 tests. The other 38
are count-boundary fits, which get no ordinary LR, as with `test=lr`.

On the real contrast (1,000 genes, 16 samples, donor-paired), RapidGLM QL makes
728 BH discoveries, glmGamPoi 710 and edgeR QL 748. The Jaccard index is 0.97
against each of them, and the Spearman correlation of p-values is 0.997 with
glmGamPoi and 0.996 with edgeR. The fitted prior has 9.5 df and uses the spline
abundance trend. The QL run takes under a second on one thread on a shared host.

## Simulated null

A second check, `scripts/validate_ql_simulation.py`
([receipt](../validation/QL_SIMULATION.json)), simulates null counts with no
effect of the tested column:
- normalized means log-uniform on [5, 5000];
- dispersions 0.05 + 0.5 / mean (near the Kang trend), times exp(N(0, 0.5²));
- size factors exp(N(0, 0.3²)), supplied as offsets.

Each configuration has 20 datasets of 2,000 genes (40,000 tests). The gate,
fixed before the first run, uses the same thresholds as the Kang gate plus a
lower bound of 3.5% at 0.05, on the four NB configurations. It passes.

| Configuration | RapidGLM QL | edgeR QL | glmGamPoi QL | RapidGLM LR | bulk-de | DESeq2 Wald |
|---|---|---|---|---|---|---|
| A: 3 vs 3 | 4.98, 1.00, 0.11 | 5.01, 0.98, 0.11 | 5.03, 1.00, 0.12 | 11.8, 3.65, 0.63 | 4.50, 1.10, 0.25 | 4.52, 1.11, 0.25 |
| B: 5 vs 5 | 4.98, 1.07, 0.09 | 5.08, 1.09, 0.11 | 5.01, 1.09, 0.10 | 8.48, 2.36, 0.38 | 5.01, 1.31, 0.25 | 5.03, 1.32, 0.25 |
| C: 10 vs 10 | 5.18, 1.01, 0.12 | 5.26, 1.07, 0.13 | 5.17, 1.00, 0.12 | 6.91, 1.69, 0.20 | 5.38, 1.26, 0.19 | 5.40, 1.26, 0.19 |
| D: 8 donors, paired | 4.91, 0.94, 0.09 | 4.91, 0.95, 0.08 | 4.94, 0.95, 0.09 | 19.4, 8.61, 2.86 | 4.75, 1.15, 0.21 | 4.75, 1.15, 0.21 |
| E: 5 vs 5, log-normal Poisson | 5.37, 1.18, 0.12 | 5.44, 1.20, 0.13 | 5.37, 1.19, 0.12 | 8.91, 2.47, 0.36 | 5.36, 1.38, 0.25 | 5.39, 1.39, 0.25 |

Entries are rejection rates in percent at p ≤ 0.05, 0.01 and 0.001. The QL test
is calibrated in every configuration and agrees with edgeR and glmGamPoi.
Per-gene ML LR (and Wald, in the receipt) are anticonservative, most with the
paired design's 8 extra donor coefficients. bulk-de matches DESeq2 closely,
including its excess at 0.001. E, whose counts are not NB, is reported without
a gate.

## Contrasts and joint tests

Since 2026-09-30 the QL test also takes a contrast or several columns jointly,
with the F statistic on as many numerator df as tested columns
([HYPOTHESES.md](HYPOTHESES.md)). Against glmGamPoi, p-values correlate at
0.978–0.999 (Spearman) on four hypotheses in airway and Kang. The discovery
counts lie within the range spanned by edgeR and glmGamPoi; on the Kang joint
donor test the two references differ (36 and 11 discoveries) and RapidGLM makes 20.

## Why Cox-Reid

Before implementation, black-box glmGamPoi runs on the same mock null showed
that its QL test is calibrated with Cox-Reid adjusted dispersions (3.9%, 0.72%,
0.11%) but not with plain ML dispersions (6.3%, 1.5%, 0.29%). The QL test
therefore estimates dispersions with the Cox-Reid adjusted profile likelihood,
which is also available on its own as `cox_reid`.

## Other checks

- **Unit tests:** 59 C++ cases passed at this commit, 8 of them new. They cover the weighted
  median against a sorted reference and the running-median trend. They check
  that the natural spline spans lines and is linear at the boundary knots. They
  recover scaled-F prior parameters, including d0 → ∞ and an abundance trend.
  They compare the Cox-Reid value and derivatives with finite differences and a
  direct determinant, and check its fixed point. On simulated panels they check
  QL calibration, power and thread invariance, and that invalid inputs are
  rejected. 14 CLI cases pass, 2 of them new. ASan/UBSan builds are clean with
  OpenMP on and off, and the R and Python binding suites pass.
- **Unchanged outputs:** all 12 frozen panel cases are byte-identical to the
  previous commit. The
  new options are opt-in, and the frozen pseudobulk panel sets `test`
  explicitly.
- **Convergence:** on the real panel, four genes did not converge within 60
  dispersion rounds under the plain Cox-Reid alternation. Where the adjusted
  likelihood is flat, the alternation converges linearly with a ratio of about
  0.9. With Aitken extrapolation all converge. An independent Python check
  confirmed the fixed points: coefficient score about 1e-9, adjusted-likelihood
  derivative about 1e-9, and grid maxima matching the fitted theta.

## Provenance

The design follows published papers: Lund et al. 2012, Smyth 2004, McCarthy et
al. 2012 and Cox and Reid 1987. It also follows the recipe of glmGamPoi
(Ahlmann-Eltze and Huber 2020). Only MIT-licensed glmGamPoi code was read, at
devel commit 95cffb79 (v1.25.1); see [DEPENDENCIES.md](../DEPENDENCIES.md).
The blocks there that declare DESeq2 (LGPL) or edgeR (GPL) origin were not
read. The Cox-Reid derivatives were derived here, not taken from any package.
The C++ was written for this repository. edgeR, DESeq2 and glmGamPoi ran only
as black boxes.

Reproduce with:

```bash
python3 scripts/validate_ql.py --work validation/work/ql-test --output validation/QL_TEST.json \
  --r-library validation/local-data/R_glmgampoi:validation/local-data/R_library
```

glmGamPoi 1.25.1 is installed from its MIT source into the ignored
`validation/local-data/R_glmgampoi`.
