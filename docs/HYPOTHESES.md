# Contrasts and joint tests — 2026-09-30

`pseudobulk-de`, `dose-trend` and `bulk-de` now test linear hypotheses Cβ = 0,
not only a single design column:
- `contrast`: one weight per column, such as one cell line minus another;
- `coefficients`: several columns tested jointly, such as all donor columns.

The method is in [METHODS.md](../METHODS.md#tests-and-effects), the keys in
[CLI.md](CLI.md#hypotheses-contrasts-and-joint-tests), and the receipt is
[validation/HYPOTHESES.json](../validation/HYPOTHESES.json)
(`scripts/validate_hypotheses.py`). Single-coefficient tests are unchanged: all
12 frozen panel cases, the da-lr panel and the airway bulk table are
byte-identical to the previous commit.

## Gates

The thresholds were set before the RapidGLM runs, after seeing that edgeR and
glmGamPoi disagree on the Kang joint test (64 against 19 BH discoveries). Two
of the four gates fail; the causes are below.

| Gate | Result |
|---|---|
| G1: a contrast gives the same p-values as the coefficient test in the reparametrized design (maximum relative difference < 1e-6) | **Fails** on airway for bulk-de and QL; passes on Kang |
| G2: bulk-de against DESeq2, Jaccard ≥ 0.9 and p-value Spearman ≥ 0.99 per hypothesis | **Fails** on both Kang hypotheses; passes on airway |
| G3: pseudobulk QL against glmGamPoi (Spearman ≥ 0.95) with a discovery count within the two references' range | Passes |
| G4: joint tests identical at 1 and 4 threads | Passes |

Hypotheses:
- airway: cellN080611 − cellN061011 (contrast), and the three cell-line columns
  jointly (3 df);
- Kang: donor3 − donor5 (contrast), and the seven donor columns jointly (7 df).

## bulk-de against DESeq2

DESeq2 ran `results(contrast = weights)` for contrasts and
`DESeq(test = "LRT", reduced = ...)` for joint tests.

| Hypothesis | BH 0.1 discoveries (RapidGLM / DESeq2) | Jaccard | p-value Spearman |
|---|---:|---:|---:|
| airway cell-line contrast | 2,566 / 2,368 | 0.912 | 0.998 |
| airway cell lines jointly | 4,351 / 4,009 | 0.908 | 0.998 |
| Kang donor contrast | 3 / 1 | 0 | 0.990 |
| Kang donors jointly | 64 / 64 | 0.829 | 0.981 |

The airway fold changes for the contrast differ from DESeq2's by 2e-4 log2 at the
median. On Kang the donor effects rest on two samples per donor. They are more
sensitive to the dispersions than the stimulation effect is, and the Kang
dispersion trend differs from DESeq2's ([BULK_DE.md](BULK_DE.md#known-differences)).
The contrast has too few discoveries for a Jaccard index to mean much.

## Pseudobulk QL against edgeR and glmGamPoi

Offsets are log library sizes. glmGamPoi ran `test_de(contrast = ...)` or
`test_de(reduced_design = ...)`; edgeR ran `glmQLFTest(contrast = ...)` or
`glmQLFTest(coef = ...)`.

| Hypothesis | BH 0.05 discoveries (RapidGLM / edgeR / glmGamPoi) | Spearman with glmGamPoi | Jaccard with glmGamPoi |
|---|---|---:|---:|
| airway cell-line contrast | 2,094 / 2,375 / 2,113 | 0.999 | 0.943 |
| airway cell lines jointly | 3,527 / 4,041 / 3,628 | 0.999 | 0.941 |
| Kang donor contrast | 0 / 0 / 0 | 0.992 | — |
| Kang donors jointly | 20 / 36 / 11 | 0.978 | 0.55 |

## Parametrization

On Kang, a contrast matches the coefficient test of the reparametrized design
within 2e-8 for every test (bulk-de Wald and LRT; QL, LR and Wald).

On airway, the pseudobulk LR and Wald tests match within 1.1e-6, but whole
pipelines do not:
- **bulk-de:** six low-count genes have flat likelihoods; three of them have a
  coefficient that diverges. Their gene-wise dispersions differ between
  parametrizations by 0.2–13% (three of them near the 1e-8 floor), where the
  optimizer stops at its tolerance. With only 3 residual df, the prior variance
  comes from a grid search, and it moves one step, from 0.6967 to 0.6987. Every
  MAP dispersion and p-value then moves slightly: at most 54% relative for one
  gene, while the discovery sets agree with Jaccard 0.999. One boundary gene
  changes status between a failed reduced fit and a boundary LR test.
- **QL:** for genes with coefficient boundaries, the Cox-Reid adjustment covers
  the columns not at a boundary, and which columns those are depends on the
  parametrization. The dispersion trend shifts; p-values differ by up to 9%
  relative and discoveries agree with Jaccard 0.998.

These are properties of the pipelines on data with many zero cells, not of the
hypothesis code. The same sensitivity to single genes at 3 residual df is
recorded in [BULK_DE.md](BULK_DE.md#known-differences).

## Boundaries

A contrast of two diverging coefficients can itself be finite. In airway, genes
whose reference cell line has only zeros send every cell-line coefficient to
infinity, but the difference between two lines stays finite. The Wald test
therefore fits the contrast as a coefficient of a reparametrized design, where
the engine's boundary check applies to the contrast itself. A first version
that combined the original coefficients reported 343 such pseudobulk Wald tests
as missing.

## Other checks

- **Unit tests:** 82 C++ cases pass, 4 of them new. They cover:
  - the null-space basis and rejection of invalid hypotheses;
  - contrasts against reparametrized coefficients for Wald, LR, QL and bulk-de;
  - joint LR tests against the likelihood ratio of explicit fits;
  - QL joint tests at 1 and 4 threads;
  - refusal of joint Wald tests and of shrinkage for contrasts.
- **CLI tests:** 19 pass, 1 of them new.
- ASan/UBSan builds are clean with OpenMP on and off.
