# Changelog

## 1.0.0 (2026-10-06)

- Tiled, checkpointed `sceptre` runs (`checkpoint_dir`, `tile_targets`,
  `tile_responses`, `phase`, `tile`): a run resumes from its finished tiles,
  tiles can run as separate jobs, and the merged output is byte-identical to a
  single run. Response null fits are made once and rebuilt from their
  coefficients in each tile (checked to be exact). Library: `SceptreShard`,
  `sceptre_layout()`, `sceptre_nulls()`.
- Faster CRT resampling with identical results
  ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)):
  - thinned CRT draws are generated and scored in batches, over per-cell score
    records stored together, so each batch reads the sampler and the records
    once instead of once per draw;
  - with several threads, all threads work on a ring of active pairs, taking
    chunks of draws, and a target's CRT sampler is shared by its pairs
    (`sampler_cache_mb`);
  - Philox blocks are computed eight at a time, with AVX2 when available.
  On the 207,324-cell Gasperini panel (high MOI, 32 threads) output is
  byte-identical to 0.2.0, and the 100-target x 7,418-gene point takes 1,080 s
  instead of 4,755 s.
- `reuse_draws` (opt-in, high MOI): each target's CRT draws are generated once
  and scored against every gene, with random streams keyed by target. P-values
  differ from the default's and share resamples within a target, so it is
  meant for routine discovery; reported statistics need the default's
  independent draws. On the 100-target x 7,418-gene point it takes about 5
  minutes instead of 18.
- `shared_screen` (opt-in, high MOI) with `shared_screen_p` (default 0.05):
  pairs are screened with their target's shared first-stage draws; pairs above
  the threshold end there, and all others are tested from their own draws
  exactly as by default, so their p-values are the default's (every p-value
  below the threshold and, in practice, every discovery). A pair can end at the shared screen that
  the default would have continued. Engine: `ResampleOptions::prescreen`.
  Batch callables of `staged_null` now take the stream (`test`) as their first
  argument, and stored draws are used only for their own stream.
- Cholesky factorizations, their solves and inverses, and the QR back
  substitution and (R'R)^-1 are now computed in-house at every size (before:
  LAPACK above 16 columns and for the QR triangular solves). Results no longer
  depend on the BLAS's `dtrsm`: the pthread OpenBLAS 0.3.20 of Ubuntu 22.04
  returned wrong triangular solves from order 32 up with its Cooperlake kernel
  (Intel Xeon 6975P-C). Fits that take the QR path change in the last digits
  (relative 1e-13 on airway and GTEx thyroid; calls unchanged), as can fits
  with more than 16 columns. The new CMake option `RAPIDGLM_LAPACK_SOLVES`
  restores the LAPACK path, with a configure-time warning; run records state
  `lapack_solves` (`validation/LAPACK_SOLVES.json`).
- `sequential_screen` (opt-in): screens end as soon as their outcome is
  certain, so the decision to continue, and every continuing pair's record, are
  the default's; pairs that end early report a Besag-Clifford sequential
  p-value from the draws used. Engine: `ResampleOptions::sequential_screen`.
  Such screens are drawn one chunk at a time with about one active pair per
  thread, so no draws are generated beyond a screen's end (before, threads
  drew ahead: on 50 targets x 1,000 genes at 16 threads `sequential_screen`
  took 94 s, now 53 s, against 113 s without it; output unchanged).
- The skew-normal fit to the null statistics is accepted by a stricter check
  by default (`fit_check`, `strict` or `ks`; `ResampleOptions::fit_check`; in
  the C, Python and R interfaces too). The earlier check, the fit's overall KS
  discrepancy, says little about the far tail where the p-value is read: on
  100 targets x 7,418 genes of the Gasperini screen, counting the tail over 1
  to 4 million draws contradicted the accepted fit for seven pairs, all with
  positive statistics, whose p-values were 8 to several thousand times too
  small; six had been discoveries. `strict` also requires that the draws in the
  tail on the statistic's side are no heavier than the fit says (draws beyond
  the statistic, the most extreme draw, counts beyond fitted quantiles, the
  fourth moment), and accepts a p-value beyond the reach of the current budget
  only on the last one. It accepts five of the seven in at most 2% of repeats
  and the two mildest in 42-44%, and 17 of the 19 other pairs in at least 96%;
  draws rise by 5% and that panel's discoveries go from 20 to 15
  ([validation/SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)).
  `fit_check: "ks"` gives the earlier results exactly. P-values of pairs whose
  fit is now rejected, or confirmed on the last budget, change.
- The C interface accepts `RAPIDGLM_CRT_HAZARD`; Python and R accept
  `crt_draw="hazard"`.
- `early_stop_p` (opt-in, default 0): the budgets after the screen also end
  early. A pair that has passed the screen stops drawing once its empirical
  p-value over the whole budget must exceed `early_stop_p`, and reports a
  Besag-Clifford sequential p-value from the draws used, in place of the
  skew-normal one; a pair whose empirical p-value over the budget is at most
  `early_stop_p` keeps its record. Engine: `ResampleOptions::early_stop`;
  `NullStages` now treats screens and later budgets as sequential stages, with
  chunks that end at the next check. With `sequential_screen` and
  `early_stop_p` 0.002 at 16 threads, 100 targets x 7,418 genes of the
  Gasperini panel take 362 s instead of 542 s (65.9M draws instead of 123.1M):
  13,492 of the 15,057 pairs past the screen stop, after 1,314 draws on
  average, and the 22 discoveries and their records are unchanged
  ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)).
- `crt_draw: "hazard"` (opt-in; `CrtDraw::hazard`): CRT labels from the points
  of a unit-rate Poisson process on a line where observation i owns an interval
  of length -log(1 - p_i). The law is the same independent Bernoulli(p_i) as
  `thinned` and `direct`, from a third random stream: one exponential gap
  (256-layer ziggurat) per treated cell, no rejection and no levels to merge,
  in the same 12 bytes per cell. The gaps' random words come from ARS with 10
  rounds, an AES-based counter generator (`AesCounterRng`) keyed per draw from
  the draw's Philox stream and computed with the processor's AES instructions
  (x86 AES-NI, Arm), at about a third of the cost of a Philox word. With
  `sequential_screen` and `early_stop_p` 0.002 at 16 threads, 100 targets x
  7,418 genes of the Gasperini panel take 279 s instead of 362 s. On a
  processor without AES instructions `hazard` draws with the thinned sampler;
  run records state `crt_draw_used`. Tail probabilities of seven Gasperini
  pairs agree with `thinned`'s within Monte Carlo error over 2 to 8 million
  draws each ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)).
- CI job and script for arm64 (`scripts/test_arm64.sh`): the unit tests are
  cross-compiled and the generator and sampler tests run under QEMU user
  emulation, which covers the Arm AES path.
- A sequential screen whose threshold the first budget cannot resolve (every
  test ends at it) now ends once a draw is as extreme as the statistic on the
  deciding side, with that draw's sequential p-value. Before, it ended at the
  first check, and reported 1 for two-sided tests and NaN for one-sided ones.
- `sceptre` reads its responses into a compact count matrix
  (`CountMatrix`: per gene, 4-byte cell and 4-byte count for each nonzero, in
  one buffer with per-gene offsets) instead of the double CSC, with a parallel
  single-pass MatrixMarket loader (`read_market_counts`): each thread parses a
  piece of the text into per-gene lists, which are then copied in file order
  into the buffer, so a cell-ordered file needs no sort. On the Gasperini
  matrix (6 GB, 445M nonzeros, 16 threads) loading and set-up fall from about
  80 s to 8 s and the load's peak memory from 23 GB to 12 GB; outputs are
  unchanged. `sceptre`, `sceptre_layout` and `sceptre_nulls` accept either
  container. The responses' SHA-1 is computed on its own thread.
- A tiled run records its inputs' SHA-1 fingerprints in the checkpoint
  directory (`inputs.tsv`) with each file's size and modification time; later
  phases and tile tasks take a fingerprint from there when the file's size and
  time are unchanged instead of hashing the file again (about 16 s per task
  for the 6 GB Gasperini matrix: a tile task's start falls from 25 s to 9 s).
  A file modified in the last two seconds is not recorded.
- CRT samplers: 12 bytes per cell (4-byte cell index) instead of 16, with the
  same draws. Every target now keeps its sampler by default
  (`SceptreOptions::sampler_cache_bytes` unlimited; the CLI's
  `sampler_cache_mb` defaults to half of the machine's memory, was 4096): with
  the old default only a third of the full Gasperini panel's 3,776 targets
  were kept and the rest rebuilt their sampler for every pair. With a third
  kept, 50 targets x 1,000 genes takes 132 s instead of 111 s. Targets that
  do not fit are counted (`SceptreResult::samplers_rebuilt`, run record
  `samplers`) and the CLI warns; a tiled run without tile sizes then splits the
  targets into blocks that fit, recorded in the plan.
- The perturbation workflow returns freed response state to the system every
  256 responses (glibc `malloc_trim`). Blocks freed by another thread than the
  one that allocated them stayed in per-thread arenas, and resident memory grew
  by a few megabytes per response over a run.
- Rust binding ([bindings/rust](bindings/rust)) over the C interface, with a
  port of PyDESeq2 0.5.2's workflow (MIT, with attribution) whose per-gene fits
  run in RapidGLM. It makes the same calls as PyDESeq2 with the RapidGLM back
  end on airway and GTEx thyroid at 128 and 647 samples (Jaccard 0.984 at 32
  samples, from Cook's distances at leverage-1 samples;
  `validation/RUST_DESEQ.json`). CI runs `cargo test`, clippy and rustfmt.

## 0.2.0 (2026-10-01)

- `legacy_cooks` (CLI, C, Python, R and the PyDESeq2-style API) computes Cook's
  distances with DESeq2's robust dispersion as implemented in PyDESeq2 (MIT;
  Muzellec et al. 2023), with PyDESeq2's flagging rule. Discoveries then match
  DESeq2's on GTEx thyroid: Jaccard 0.997 and 1.000 at 128 and 647 samples,
  against 0.58 and 0.90 by default ([docs/BULK_DE.md](docs/BULK_DE.md)).
  Attribution and licence: DEPENDENCIES.md.

## 0.1.0 (2026-10-01)

First release.

- **Engine.**
  - Gaussian, Poisson, binomial and negative binomial GLMs for many responses
    that share a design, over sparse column matrices, with OpenMP across
    responses and identical results at any thread count.
  - Safeguarded Newton solvers: observed information for the NB log link,
    likelihood step halving, Cholesky with a QR fallback, and R-compatible
    aliasing. Diverging coefficients are detected. NB dispersions are
    estimated jointly or by Cox–Reid adjusted profile likelihood.
  - Wald, likelihood ratio, score and quasi-likelihood F tests of linear
    hypotheses Cβ = 0.
  - Score scans calibrated by asymptotic score, permutation or conditional
    randomization, with staged budgets and a skew-normal tail.
- **Applications.**
  - `bulk_de`: the DESeq2 procedure, with normal-prior fold-change shrinkage.
  - `sceptre`: the SCEPTRE method for single-cell CRISPR screens.
  - Pseudobulk quasi-likelihood DE, differential abundance and dose trends.
- **Interfaces.**
  - CLI with JSON configurations.
  - Stable C ABI (`librapidglm_c`).
  - Python package: `bulk_de`, `sceptre`, `scan`, `fit_batch`; a
    PyDESeq2-style `DeseqDataSet`/`DeseqStats`; `RapidGLMInference` for
    PyDESeq2; a statsmodels drop-in.
  - R package: `bulk_de()`, `sceptre()`, and `glm()`/`glm.nb()` drop-ins.
  - A Dockerfile that builds everything and runs all test suites.
- **Validation.** Receipts under `validation/` for agreement with DESeq2,
  SCEPTRE, R `glm`/`glm.nb`, statsmodels, glmGamPoi and Ceres Solver, and for
  the application benchmark ([docs/BENCHMARK.md](docs/BENCHMARK.md)).
- **Known differences.**
  - DESeq2's robust dispersion for Cook's distances is unspecified; RapidGLM's
    substitute flags more genes in designs with many small groups.
  - The local dispersion trend uses a running median instead of locfit.
  - Fold-change shrinkage uses the normal prior only.
