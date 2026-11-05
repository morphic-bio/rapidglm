# Evidence for the preprint

**Historical focused gate, completed 2026-11-01.** The full 24-stage GPU-host
run has now also passed. The 1.6.0 release benchmark ratios and the separate
full-run ablations are in the [benchmark report](BENCHMARK.md) and
[full receipt collection](../validation/receipts/v1.6.0/README.md). The scope
and paired results below describe the earlier focused delivery, whose receipts
remain in [PREPRINT_OPTIMIZATION_ABLATIONS.json](../validation/receipts/v1.6.0/PREPRINT_OPTIMIZATION_ABLATIONS.json)
and [PREPRINT_REPLOGLE_TAIL_REVIEW.json](../validation/receipts/v1.6.0/PREPRINT_REPLOGLE_TAIL_REVIEW.json).
Release publication was separately authorized on 2026-11-05 after the full run passed.

At the focused gate, the preprint took priority over a complete new performance
run. The candidate was RapidGLM 1.6.0 at `4505ef6` (`v1.6.0-dev.2`), which had
not yet been released. That gate did not authorize release or publication. The
release library remains unchanged from that measured source.

## Claims and their evidence

| Claim | Evidence and allowed scope |
|---|---|
| Published reference-tool benchmark ratios | Keep the existing README and benchmark tables explicitly labelled **1.4.0 measurements**, with their original host, inputs and timing scopes. They are not 1.6.0 timings. |
| 1.6.0 functional correctness | Coordinator's completed native/portable functional suite on `4505ef6`, plus the GPU-host native, CLI, binding and container checks in `validation/receipts/v1.6.0/TESTS.json`. |
| Unchanged GLM and bulk/edgeR validation outputs | Seven completed stages from the interrupted full run: GLM accuracy/speed, batch, solvers/control, bulk validation and edgeR validation. Their receipts retain exact scope and provenance. |
| High-MOI batch scoring | Focused CRT `t100_g7418`, `ks` and default, each with batch scoring enabled/disabled; require result and QQ table byte identity and agreement with the retained baseline. |
| Low-MOI refined null starts | Focused Replogle strict/ks pilot, each with refine/cold starts; require identical statuses, empirical p-values and discoveries. Report tail approximation differences separately. |
| Count-model permutation refits | Focused chr22 NB lockstep/separate pair plus Gaussian 1,000 and 10,000 permutations, including double precision. Require the documented numerical bounds and identical NB discovery sets. |
| Poisson Beta-fit behavior and tiled/resumed high-MOI identity | Existing development receipts `NB_QTL_BATCH.json` and `BATCH_SCORING_DEV2.json`, retaining their original binaries, hosts and limitations. No new GPU-host Poisson or tiled performance claim. |
| Full Gasperini cloud run | Separately owned coordinator receipt; this gate does not modify or independently certify it. |

The five paired ablations measure the optimization ratios on one host and the
same inputs. They do not supply new ratios against reference tools, establish
novelty, or justify extrapolating a pilot to a full-screen runtime.
Development timings retained in the 1.6.0 changelog remain development evidence
on their recorded hosts; the full-run measurements are labelled separately.

## Work deferred at the focused gate

The exhaustive full run stopped after seven complete stages and 157 of 198
bulk commands. Partial bulk timings are excluded from headline claims.
Remaining bulk repetitions, Rust timing, the additional perturbation panels,
reference-tool reruns, pseudobulk, special-function/end-to-end backend timing,
full Replogle, genome-wide eQTL and their compression stage were deferred at
that gate and have since completed in the full run.
The missing historical Rust raw tables do not block this focused gate.
All previous artifacts, including the skipped 1.5.0 run, are retained.

The new measurements retain the exclusive Slurm allocation, fixed cores,
quiet-start checks, unchanged CPU-policy checks, and 5% contention rejection
and retry rules. The focused plan, timing audit and agreement receipt live in
`validation/receipts/v1.6.0/`. Passing measurements, input checks and the
assembled receipts are required before marking this focused gate complete.

The manuscript itself still requires a claim-by-claim check against these
sources; no manuscript draft was available when this scope was recorded.

<!-- FOCUSED_RESULTS -->

The focused gate passed all 13 guarded commands and five agreement pairs. These are paired, single-run wall times on the same host and inputs; they are not confidence intervals or new reference-tool speedups.

| Pair | Default (s) | Opt-out (s) | Opt-out / default |
|---|---:|---:|---:|
| qtl_nb_chr22 | 635.663 | 672.242 | 1.058× |
| crt_t100_g7418_ks | 685.412 | 1015.057 | 1.481× |
| crt_t100_g7418_default | 682.112 | 1053.696 | 1.545× |
| replogle_strict | 1046.436 | 1310.035 | 1.252× |
| replogle_ks | 226.931 | 519.224 | 2.288× |

All Gaussian comparison tables are byte-identical to the retained baseline (double precision also matches 1.4.0). NB retains all 168 eGenes under both Storey and BH rules, with maximum relative Beta-p difference 7.88844e-11. CRT result and QQ tables are byte-identical. Replogle statuses, empirical p-values and discoveries are identical; approximated tail p-values can differ and must not be described as byte-identical. `PREPRINT_REPLOGLE_TAIL_REVIEW.json` records decimal-arithmetic diagnostics for those tails.
