# SCEPTRE response-null model

Since 2026-09-29 the `sceptre` command defaults to `"null_model": "sceptre"`;
`"null_model": "direct"` selects the joint GLM fit that was the default when
this page's measurements were made. Both choices are explicitly recorded in
`.run.json`, together with the null fit, score numerator, effect fit and
resampling policy. Since 2026-09-30 the command also screens its
first resampling stage as SCEPTRE does
([SCEPTRE_STAGING.md](SCEPTRE_STAGING.md)).

| Setting | Direct | SCEPTRE model (default) |
|---|---|---|
| NB null coefficients | Joint NB maximum likelihood | Poisson GLM estimates, then frozen |
| NB dispersion | Alternates with coefficient fitting | Conditional ML at the frozen means |
| Score numerator | Includes nuisance-score projection | Raw treatment score |
| Score denominator | Covariate-adjusted information | Covariate-adjusted information |
| Full effect log2FC and SE | Direct GLM fit | Direct GLM fit |

The SCEPTRE model requires `family=nb`, `dispersion=ml`, and `ridge=0`.
It applies to both shared complement-control nulls and treatment-plus-NT
subsets. Invalid combinations are rejected. The tagged low-level fit supplies
no coefficient covariance or ordinary Wald, LR or normal-tail score inference;
its statistic is calibrated by resampling. See [CLI/API details](CLI.md) and
the formulas in [METHODS.md](../METHODS.md).

SCEPTRE's [public methods chapter, footnote 1](https://timothy-barry.github.io/sceptre-book/methods.html#computing-the-test-statistics)
explicitly describes its Poisson-first recipe as a speed improvement. The
chapter attributes robustness to resampling calibration; it does not establish
that the Poisson coefficient estimates are themselves more robust than joint
NB estimates. This implementation uses the existing native conditional-theta
optimizer. It reproduces the documented model and statistic, without claiming
identical historical stopping rules. No reference implementation was read or
translated: comparisons use public R APIs and retained black-box outputs.

## Measurements

On the frozen public S2 panel, all 12 response fits converge. Across 66
retained reference pairs, maximum absolute differences are:

| Comparison | Maximum absolute difference |
|---|---:|
| Coefficients versus R Poisson GLM | 6.01e-12 |
| Fitted means versus R Poisson GLM | 1.35e-10 |
| Dispersion versus R `MASS::theta.ml` at those means | 9.40e-13 |
| Score versus independently calculated R recipe | 1.89e-11 |
| Score versus pinned SCEPTRE 0.99.0 output | 2.39e-7 |
| Mean ratio versus pinned SCEPTRE output | 8.05e-9 |

For context, the direct-model scores differ from retained SCEPTRE scores by
up to 0.727. That is a model/score choice, not solver error.

Loaded native null-fit-and-score calls for the 12 responses take a median
**5.090 s with direct NB versus 1.038 s with the SCEPTRE recipe (4.90x)**.
These are three measured repetitions after warmup, alternating mode order,
using one native thread and one BLAS worker. Times include bridge conversion,
score setup and returned vectors; they exclude resampling and full effect
fits. This is a shared-host stage measurement, not an end-to-end speedup.

The mode does **not** select SCEPTRE's RNG, budgets, tail approximation or
staged acceptance policy. Those remain separate RapidGLM settings. S2 still
has 13 discoveries versus the reference's 11 (Jaccard 0.846 at BH 0.1), despite
the closely matching scores. Full p-value equivalence and the proposed 0.9
concordance target are therefore not established. Thresholds remain provisional.

## Verification and reproduction

- All 41 C++ cases and nine CLI cases pass. Address/undefined sanitizers pass
  with OpenMP enabled and disabled; the installed external consumer passes.
- Default scientific tables of the frozen panels are byte-identical to `45c9825`.
  Explicit `direct` and omitted `null_model` also match exactly.
- S2 SCEPTRE-model scientific tables, including effect estimates, are
  byte-identical across 1, 2, 8 and 16 threads. Unit coverage separately checks
  treatment-plus-NT routing, offsets/weights, boundaries, the published score,
  and identical indexed permutation statistics across thread counts.
- All 8,000 original E1 calibration rows remain byte-identical with direct
  fitting. This preserves the original evidence, including its failed check;
  it does not qualify calibration of the new optional model.
- The updated profiling harness builds and reproduces its panel's output exactly.

With the public panel artifacts prepared and the documented numerical
dependencies available, run the numerical comparison with
`Rscript scripts/audit_deviations.R sceptre-model validation/work/sceptre-model/R`
(set `RAPIDGLM_CPPFLAGS` and `RAPIDGLM_LIBS` for the native bridge). Large
artifacts remain local under `validation/work/sceptre-model/`.
