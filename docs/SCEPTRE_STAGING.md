# SCEPTRE concordance and staging — 2026-09-30

On public S2 the perturbation workflow's discoveries overlapped SCEPTRE 0.99.0's
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

RapidGLM's draws differ from SCEPTRE's, so borderline pairs can go either way.
SCEPTRE's results do not change with R's seed, so its own Monte Carlo spread
cannot be measured by reseeding it. The floor is therefore the agreement between
RapidGLM runs with different seeds. Over seeds 1–10 with version 1.3.0 (BH 0.1
over the pairs matched with SCEPTRE):

| Panel (SCEPTRE discoveries) | Screen | RapidGLM discoveries | Jaccard with SCEPTRE, mean (min) | Jaccard between seeds, mean (min) | Stage agreement |
|---|---|---|---|---|---|
| public S1, low MOI (32) | 0.02 | 30–32 | 0.963 (0.909) | 0.948 (0.879) | 0.97 |
| | 0 | 31–32 | 0.975 (0.969) | 0.972 (0.938) | 0.55 |
| public S2, high MOI (11) | 0.02 | 11–16 | 0.834 (0.688) | 0.811 (0.647) | 0.95 |
| | 0 | 13–18 | 0.702 (0.611) | 0.895 (0.722) | 0.83 |

With the screen, agreement with SCEPTRE equals or exceeds the agreement between
two RapidGLM seeds on every panel. Without it, S2 falls well below that floor
and averages 15.8 discoveries against SCEPTRE's 11.

Ten seeds are few for S2. From 1.3.0 a pair's draws are keyed by the pair
itself, so the same ten seeds give other draws than before: with the screen
they average 13.2 discoveries on S2, where the draws of 1.2.0 gave 12.4. Over
200 seeds the two sets of draws agree: 12.3 discoveries (standard error 0.11;
9 to 17 in single runs) and a mean Jaccard index with SCEPTRE of 0.883 with
either. On S1, 200 seeds average 31.1 discoveries (29 to 34) and 0.963. Stage agreement is the fraction of pairs that both tools
take to a second stage, or both do not. The S2 value that prompted this, 0.846
at seed 20260926 without the screen, is one draw from such a distribution. With
the screen, that seed gives 0.769.

Gates, set after exploring S1 and S2; all pass:
- **G1 floor:** with the screen, the mean Jaccard with SCEPTRE is at least the
  mean between-seed Jaccard minus 0.02.
- **G2 counts:** the mean discovery count is within max(1, 15%) of SCEPTRE's.
  From 1.3.0 the mean is taken over 200 seeds. Over ten seeds its standard
  error on S2, about 0.5, is a third of the margin there (1.65), so the outcome
  turns on the seeds: the ten above average 13.2, outside the margin (the
  receipt records this as `G2_counts_ten_seeds`), and the same seeds with the
  draws of 1.2.0 averaged 12.4, inside it. Over 200 seeds the mean is 12.3.
- **G3 null calibration:** screened resampling meets the scan's null bounds on
  six configurations (Poisson and NB responses; binary and continuous
  predictors; permutation and CRT). Rejections at 0.05 run from 98 to 130 of
  2,400; KS statistics are at most 0.027, against a critical value of 0.040.

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
