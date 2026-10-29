# SCEPTRE concordance and staging

The historical investigation began on 2026-09-30. On public S2 the perturbation workflow's discoveries overlapped SCEPTRE 0.99.0's
with a Jaccard index of only 0.846 (13 against 11). This report finds the cause
clean-room: from SCEPTRE's documentation and black-box outputs, without reading
its source. The receipt is
[validation/SCEPTRE_STAGING.json](../validation/SCEPTRE_STAGING.json).

## Cause

SCEPTRE reports, for each pair, the observed statistic (`z_orig`), the stage,
the p-value and, at stage 2, the fitted skew-normal parameters. On S2:
- **Test statistics:** RapidGLM's equal SCEPTRE's `z_orig` to 2e-7 on all 66
  pairs. The null model and score statistic already match.
- **Skew-normal p-values:** from its reported fits, 2 min(F, 1 − F) reproduces
  SCEPTRE's p-values to 1e-9 (relative), as its manual documents. The exception
  is the far tail (|z| > 15), where its values differ.
- **Staging:** stage-1 p-values are multiples of 2/500, so they are empirical
  two-sided p-values from B1 = 499 draws. Stage-2 p-values come from the
  skew-normal. In 1,500 SCEPTRE null pairs, the smallest stage-1 p-value was
  0.024. 31 pairs (2.1%) went to stage 2, as expected under the null if pairs
  with a first-stage p-value of at most 0.02 go on.

RapidGLM instead tried the skew-normal at the first stage for every pair. For
borderline pairs this gives smaller p-values than SCEPTRE's empirical ones, and
so systematically more discoveries.

## Change

(Since 1.0.0 the acceptance of the skew-normal fit is stricter by default than
the one this page describes; `fit_check=ks` gives the acceptance used here.
See METHODS.md, Resampling. With version 1.2.0 the receipt repeats the ten
seeds of each panel under RapidGLM's defaults as well, `defaults_screen_0.02`:
on these panels they give the discoveries of `fit_check=ks` in every seed.)

A new resampling option, `screen`, reproduces that staging. The first budget
always ends with its empirical p-value, which is final unless it is at most the
screen. Only then are more draws made, and the skew-normal fitted. It is the
`sceptre` command's default (0.02; `screen=0` restores the previous policy),
and the run record stores it. Scans keep 0 by default. A scan configured like
`sceptre` still reproduces it exactly on S2
([SCAN.json](../validation/SCAN.json)).

## Concordance at the Monte Carlo floor

These measurements were repeated in the complete final 1.4.0 run on
`v1.4.0-dev.4`. RapidGLM and SCEPTRE use different random draws, so marginal
discoveries can differ. The between-seed agreement of RapidGLM gives the
comparison floor used here. Ten seeds, BH 0.1 over matched tested pairs:

| Panel (SCEPTRE discoveries) | Screen | RapidGLM discoveries | Jaccard with SCEPTRE, mean (min) | Between-seed Jaccard, mean (min) | Stage agreement |
|---|---|---|---|---|---|
| Public S1, low MOI (32) | 0.02 | 30–32 | 0.963 (0.909) | 0.948 (0.879) | 0.968 |
| Public S1, low MOI (32) | 0 | 31–32 | 0.975 (0.969) | 0.972 (0.938) | 0.545 |
| Public S2, high MOI (11) | 0.02 | 11–16 | 0.834 (0.688) | 0.811 (0.647) | 0.955 |
| Public S2, high MOI (11) | 0 | 13–18 | 0.702 (0.611) | 0.895 (0.722) | 0.833 |

Over 200 seeds, Public S1, low MOI averages 31.095 discoveries (SE 0.077; range 29–34), with mean Jaccard 0.9632 against SCEPTRE.


Over 200 seeds, Public S2, high MOI averages 12.315 discoveries (SE 0.113; range 9–17), with mean Jaccard 0.8826 against SCEPTRE.

All final staging gates pass. G1 requires mean reference Jaccard to be
at least mean between-seed Jaccard minus 0.02. G2 uses the 200-seed mean count,
within max(1, 15%) of the reference count; its ten-seed counterpart is retained
separately because it is sensitive to those few draws. G3 reruns screened
resampling on the six scan-null configurations and checks their declared
binomial and KS bounds. These thresholds were chosen after development
exploration; they are descriptive validation gates.

The original changes to frozen panels below are historical development
observations; they are separate from the fresh staging table above.

## Effect on the frozen panels

The `sceptre` defaults change the fixed-seed panel outputs. Discovery counts
move from 23 to 22 (public S2), 127 to 125 (public S1, singleton integration)
and 37 to 36 (public S1, Bonferroni). The pseudobulk, dose and fit
panels are byte-identical.

## Provenance

The staging rule was inferred from SCEPTRE's outputs and its manual
(`?set_analysis_parameters` and the methods chapter of the sceptre book). No
SCEPTRE source was read. The black-box probes are
`scripts/reference_sceptre_stages.R` and `scripts/reference_sceptre_seeds.R`.
