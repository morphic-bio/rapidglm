# Changelog

## Unreleased

- **NT-control pair nulls start from the response's reference (`null_start`).**
  With NT controls and the SCEPTRE null, each response's null is now also
  fitted once on all retained cells. A pair takes one Fisher-scoring step from
  it, using the NT cells' totals and its target cells' totals at the reference,
  then exact Newton passes over its cells to the cold fit's convergence
  criteria, with step halving. A pair that does not converge, or whose
  response's reference has a boundary, is fitted cold (METHODS.md, Mergeable
  null start).
  - Nulls agree with the cold fit to the convergence tolerance. On the Replogle
    pilot all empirical p-values, statuses and discoveries are unchanged, and
    77 of 16,062 skew-normal tail p-values below 3e-17 differ by more than 1e-4.
  - Null work fell from 20.9 to 5.1 thread-ms per pair. On 8 threads the `ks`
    run took 592 s instead of 1,262 s, and the `strict` run 3,028 s instead of
    3,666 s.
  - `null_start: "cold"` gives the nulls of releases before 1.6.0, byte for
    byte. Select it with the `sceptre` key `null_start`,
    `SceptreOptions::null_start`, `rapidglm_sceptre_options.null_start`
    (`RAPIDGLM_NULL_START_*`) in C, or `null_start` in Python's and R's
    `sceptre()` and `sceptre_grid()`.
  - Run records state `null_start`, `null_start_used`, `null_refined` and
    `null_cold_refits`. The C table carries them as scalars, Python's `attrs`
    and R's `attr(, "run")` by name. A tiled run's plan includes the default.
  - The measured alternatives are in `validation/MERGEABLE_NULL.json`: the
    one-step null alone (all-cells or NT-only reference), an NT-only base with
    a quadratic model, a lockstep batched cold fit, and an index-mapped score
    kernel.
- **Batch scoring of high-MOI CRT tests, on by default** (`batch_scoring`). With
  complement controls each response's pairs are scored together, stage by
  stage: all their draws of a stage in one pass over the response's score
  records, in blocks of 32,768 cells (a sparse × dense product, two draws at a
  time with AVX2 where available). Records are bit for bit those of the pair
  pipeline, which `batch_scoring: false` keeps, on AVX2, portable and ARM64
  builds and at any thread count. On 100 targets × 7,418 genes of the
  Gasperini screen (8 threads) the legacy settings (`reuse_draws`,
  `confirm_draws: 0`) ran 1.92 times as fast and the default 1.22 times as fast;
  extrapolated to the full test set on 16 cores, about 4.6 h become 2.0 h and
  9.6 h become 7.8 h. The option is in the CLI, `SceptreOptions`, the C
  interface (`rapidglm_sceptre_options.batch_scoring`, appended; default 1),
  and `sceptre()` and `sceptre_grid()` in Python and R. Shared screens,
  sequential screens and early stopping keep the pair pipeline. The
  alternatives measured, among them lockstep fits of response nulls and
  propensity models as matrix products (about 4% less null-fit time and under
  1% less wall time, not adopted), are in `validation/BATCH_SCORING.json`.

## 1.5.0 (unreleased)

- **Stable Cook's flags for leverage-one samples.** Leverages within four
  machine epsilons of one are treated as saturated. Their undefined distances
  do not enter the finite maximum, and a saturated sample prevents the legacy
  count-rank exemption. All 16 thyroid32 regression genes agree with stored
  DESeq2 flags; the other bulk datasets are unchanged.
- **Division policy: `slow` by default, with an explicit `fast` mode.**
  Select it through `RAPIDGLM_DIVISION`, the CLI's `division` key, C++ options,
  the C thread-local option, or Python, R and Rust parameters. Explicit call
  choices propagate to workers and restore the prior policy on return or error.
  CMake can set the fallback with `RAPIDGLM_DIVISION_DEFAULT`.
  Fast mode selects measured eligible kernels on Zen 3; the alternatives,
  accuracy limits and reasons are in `validation/DIVISION_CHOICES.json`.
  The numerical kernels retain identical bits across AVX2, portable and ARM64
  paths within each mode. Run records state the resolved mode.
- The special-function end-to-end benchmark now uses the public SCEPTRE
  high-MOI dataset for NB effect fits.

- **CRT labels are drawn with the `hazard` sampler by default** (`crt_draw`),
  in every interface: the `sceptre` and `scan` settings, `SceptreOptions`,
  `ScanOptions` and `crt_sampler()` in C++, `rapidglm_sceptre_options_init`
  and `rapidglm_scan_options_init` (`RAPIDGLM_CRT_HAZARD`) in C, and
  `crt_draw` of `sceptre()`, `sceptre_grid()` and `scan()` in Python and
  `sceptre()` and `sceptre_grid()` in R. `crt_draw: "thinned"`, the default
  until now, gives the results of 1.4.1; `direct` remains.
  - CRT p-values change within Monte Carlo error: `hazard` draws the same
    independent Bernoulli labels from a new random stream.
  - They depend on the hardware: `hazard` needs the processor's AES
    instructions (x86 since about 2010, Apple and server Arm processors) and
    draws with the thinned sampler, from thinned's stream, without them, and in
    builds with `RAPIDGLM_FORCE_PORTABLE`. A run that draws CRT labels then
    warns (before, any run that set `crt_draw: "hazard"` did), and the run
    record states the sampler used (`crt_draw_used`), now for `scan` as well as
    `sceptre`. The C interface's `sceptre` and `scan` tables carry it as the
    scalar `crt_draw_used` (a `RAPIDGLM_CRT_*` value), and Python's `attrs`
    and R's `attr(, "run")` by name.
  - A tiled CRT run's plan includes the default sampler and, without AES
    instructions, the fallback: a checkpoint written by 1.4.1 with the default
    sampler is not resumed, and tiles drawn on processors with and without AES
    instructions are not merged.
  - Speed, measured with the 1.4.1 code (4e2b39a) and one binary for both
    samplers, on 20 targets × 7,418 genes of the Gasperini panel (the 5
    targets with discoveries in the 1.4.0 100-target receipt, 10 more of
    that panel and 5 non-targeting targets; 148,360 pairs, 207,324 cells),
    16 threads on CPUs 16-31 under the timing lock, two runs each: by
    default (independent draws, 199,999 confirmation draws) `hazard` takes
    369.5 and 370.9 s and `thinned` 483.8 and 490.7 s, 1.31 times as fast
    (1.35 times in CPU time; 95.5 and 95.7 million draws).
    With `reuse_draws` and `confirm_draws: 0`, SCEPTRE's practice, the two
    take the same time within run-to-run variation (187.2 and 191.2 s against
    183.2 and 192.9 s, and 179.3 s at a second seed): each target's draws are
    generated once, and scoring takes nearly all the time.
  - On that subset `hazard` agrees with `thinned` as closely as `thinned`
    agrees with itself at another seed, the yardstick. By default the
    correlation of log10 p over the 144,087 tested pairs is 0.9883 (thinned
    at two seeds: 0.9883); 19 of the 20 discoveries at BH 0.1 are shared (19
    of 20 between the seeds); 4.70% and 0.96% of the 35,763 negative-control
    p-values are below 0.05 and 0.01, against 4.71% and 0.95% (4.62% and 0.99%
    at the second seed). With `reuse_draws`: 0.9880 (yardstick 0.9881), 19
    discoveries shared of 20 and 19 (18 of 19 and 19), and 4.70% and 0.94%
    against 4.62% and 0.95%. Statuses and observed statistics are identical
    ([validation/CRT_BATCHING.json](validation/CRT_BATCHING.json)).
- **Special functions (`include/rapidglm/vmath.hpp`).** A standalone header
  (C++17 standard library only) with `log`, `log1p`, `exp`, `lgamma`, `digamma`
  and `trigamma` of double, as arrays (`<f>_array`) and one value at a time.
  `log`, `log1p`, `exp`, `digamma` and `trigamma` are RapidGLM's and give the
  same bits on every processor and path: AVX2 with FMA (chosen at run time),
  AArch64 Advanced SIMD, portable, and the one-value forms. `lgamma` is the C
  library's `lgamma_r` in slow mode; fast mode uses the earlier own kernel for
  positive arguments and libm for negative arguments.
  - In slow mode, `log` and `log1p` use no division: exponent and mantissa from the bits, a
    degree-19 polynomial by Estrin's scheme, exact splits, and log1p's
    rounding correction c/u formed without dividing.
  - `exp` is the AVX2 exp of `kernels.hpp`, moved here with a portable FMA
    path and full-range handling. `detail::exp_array` calls it, with the same
    bits on [-700, 700].
  - Historical development measurements on pikachu's efficiency cores, in ns per element: log 3.3
    (glibc 4.7), log1p 4.7 (glibc 9.5-11.5), exp 2.6 (glibc 7.5), digamma 10.0
    and trigamma 6.5 (boost::math's double-precision versions: 11.2 and 7.0).
  - Largest errors measured: 0.67, 0.68, 0.86, 3.7 and 2.7 ulp.
  - METHODS.md (Numerical backend, Choice of backends) lists every alternative
    measured and the reason for each choice.
- The count paths use the faster backend at each call site, measured on those
  cores:
  - The exponentials of the log-link means, of bulk DE's floored NB fits and
    of edgeR-QL's fixed-dispersion fits are taken as one array.
  - edgeR-QL's unit deviances take their logarithms as arrays.
  - Elsewhere logarithms stay one at a time with glibc's functions, as in
    1.4.1; arrays did not pay there.
  - digamma and trigamma come from boost::math without promotion to long
    double (1.4.1 promoted). They are faster per call; in the workloads the
    choice made no difference beyond noise.
  - The threaded paths (count-likelihood terms, the QTL Beta calibration) call
    `lgamma_r` instead of `std::lgamma`, which writes `signgam`. The values
    are the same.
- Results move at rounding level:
  - The special-function changes alone leave discoveries unchanged on the
    DESeq2 and edgeR agreement datasets; the separate leverage-one Cook rule
    above changes thyroid32 flags. The historical special-function agreement
    metrics equal 1.4.1's to the digits reported. One airway gene's
    gene-wise dispersion, on a flat likelihood at the 1e-8 lower bound, moves
    from 1.07e-8 to 1e-8.
  - On 24 Replogle pseudobulk perturbations (205,512 tests per method) no
    discovery changes. edgeR-QL p-values move by up to 1.5e-4 relative
    (8.3e-5 above p = 1e-20) where a prior df of 1,209 sits on a flat
    objective.
- One thread on an efficiency core, against 1.4.1 (development measurements of
  the final candidate, `validation/VMATH.json`; the 1.5.0 receipts run
  re-measures every workload):
  - edgeR-QL on GTEx thyroid: 12.57 against 13.09 s.
  - NB batch fit: 2.91 against 3.06 s.
  - bulk DE on thyroid (35.18 against 35.10 s) and on a Replogle perturbation
    (35.79 against 35.84 s): unchanged, inside the 1.5% that a rebuild with
    only a different code layout moved.
- A first version took log, log1p, exp and lgamma from SLEEF 3.9.0. It was
  removed because its log divides and was slower than glibc's on these cores.
  The count paths ran 2-13% slower (bulk DE on thyroid 39.9 against 35.2 s),
  although 2-7% faster on a performance core (`validation/VMATH.json`).
- C API version 3 adds `rapidglm_log_array`, `rapidglm_log1p_array`,
  `rapidglm_exp_array`, `rapidglm_lgamma_array`, `rapidglm_digamma_array`,
  `rapidglm_trigamma_array` and `rapidglm_special_path`. It also covers the
  `crt_draw_used` scalar of the `rapidglm_sceptre` and `rapidglm_scan` tables
  (above), which Python reads only from version 3 on.
  - Python gains `rapidglm.special` (NumPy arrays of any shape, `out=`).
  - R gains `special_log()`, `special_log1p()`, `special_exp()`,
    `special_lgamma()`, `special_digamma()`, `special_trigamma()` and
    `special_path()`, with attributes kept and NA and NaN as in base R. One
    C++ entry point allocates the result, copies the attributes and checks for
    NaN in cache-sized blocks.
  - Their tests reproduce the C++ values of RapidGLM's functions bit for bit.
- `benchmarks/special_functions.cpp` and `scripts/benchmark_special_functions.py`
  (a benchmark stage of its own, `validation/SPECIAL_FUNCTIONS.json`) report ns
  per element and the largest ulp error. They compare against glibc,
  boost::math (both policies), a transcription of Boost's double-precision
  digamma and trigamma for AVX2 (`benchmarks/boost_polygamma.hpp`), NumPy and
  SciPy, and base R.

- **Single-precision Gaussian permutation scores (new default).** The
  projected Gaussian cis-QTL permutations (no prior weights, offset, ridge or
  robust covariance) compute their score product in single precision:
  each permuted response is residualized and scaled to unit residual
  variance in double, so the float factors are O(1) whatever the
  phenotype's units, R'G_res is summed in float within 128-sample panels
  (six-by-sixteen AVX2/FMA or portable tiles) and in double across panels,
  and maxima are taken from that double result. Scales, information,
  observed statistics, nominal output, refitted permutations and the Beta
  calibration stay in double precision.
  `CisQtlOptions::permutation_precision` and the `qtl` setting
  `permutation_precision` (`single` or `double`) select it; `double` keeps
  the 1.4.1 arithmetic. The run record states the precision used (null
  without permutations), and `scripts/compare_eqtl.py` copies it into its
  summary. See METHODS.md for the accuracy design and limits.
- Permutation maxima within a relative tolerance below the observed
  statistic now count as extreme for the empirical p-value: 1e-5 for
  single-precision scores and 1e-9 otherwise. Some designs make maxima equal
  to the observed statistic in exact arithmetic (an intercept only and a
  private variant in every sample gives an exact permutation p-value of 1),
  and rounding decided such ties before, giving values from about 0.2 to 0.9
  in either precision, including in 1.4.1. `double` output stays byte for byte
  that of 1.4.1 on Geuvadis chromosome 22 with 1,000 permutations, a 60-gene
  subset with 10,000 permutations and nominal output, and an NB subset.
- On chromosome 22 (580 genes, 1,000 permutations), single precision leaves
  every status, lead and observed statistic unchanged. Twelve empirical
  p-values (all with |z| < 4) rise by 1/1001, each through one maximum within
  the 1e-5 tie tolerance; beta_p moves by at most 2.6e-5 relative. On the
  test shapes, permutation maxima agree with double to within 1.03e-6 of
  max(1, |z|).
- For single-precision maxima, the Beta calibration treats maxima with a
  relative variance up to 1e-8 (a relative SD of 1e-4) as constant. That is
  six orders of magnitude above the float rounding of maxima that are equal
  in exact arithmetic, so such genes still report `ok_beta_failed`, and
  maxima with a relative SD of 1e-3 are still fitted, as in double precision.
- Shared Gaussian permutation overheads are faster without changing results:
  the maxima are taken in lanes (AVX2 where available), and genotypes and
  shuffled responses are gathered in cache-line blocks.
- Micro-benchmark (`benchmarks/qtl_precision.cpp`, built with
  `RAPIDGLM_BENCHMARKS`): one synthetic gene (462 samples, 43 covariates
  including the intercept, 4,500 variants, 10,000 permutations, one thread)
  on efficiency core 31 takes 1.737 s in double and 1.010 s in single
  precision (1.72×), the best of six alternating runs under the timing lock.
  The score product runs at 12.0 and 20.6 GFMA/s end to end, and the packed
  products alone at 13.3 and 25.6 GFMA/s. The portable float kernel runs at
  12.2 GFMA/s against 3.2 for the portable double kernel.
- Timed on twelve efficiency cores under the timing protocol
  (`validation/QTL_SINGLE_PRECISION.json`), with 10,000 permutations, single
  precision takes:
  - Geuvadis chromosome 22: 74.4 s, against 146.5 s for tensorQTL 1.0.10
    the same day (1.97×) and 117.2 s in double precision (1.4.0 receipt);
  - genome-wide (20,427 genes, 6,317,351 variants): 2,515 s at a 3.8 GiB
    peak, against 4,683 s and 7.2 GiB for tensorQTL (1.86×; 2026-10-23,
    same host setting) and 3,987 s and 25.9 GiB in double precision
    (1.4.1).

  Genome-wide, single and double precision give the same 9,575 eGenes and
  identical leads and observed statistics. 2,687 empirical p-values rise by
  1 to 5 counts out of 10,001 through the tie tolerance, and beta_p moves by
  at most 5.5e-5 relative.
- **Faster single-precision permutations** (after the commit that added
  them, 4e2b39a), in four parts; the first three leave every output byte for
  byte as it was:
  - The permutation maxima are taken inside the float product's register
    tiles, so a batch's 512 × m score matrix (18 MB at 4,500 variants) is never
    written and read back. Each six-by-sixteen tile adds its 128-sample
    panels' float sums to double partials in a 768-byte L1 scratch, in sample
    order as before, then scales them by the inverse information, takes |z|
    and folds it into its permutations' running maxima, skipping NaN and
    infinity as before (`detail::fused_maxima`, AVX2/FMA and portable). The
    projected genotypes are packed once per gene in groups of sixteen
    variants, and variants are taken 128 at a time so that a block's packed
    genotypes stay in L2 for every tile. The compiled AVX2 sample loop keeps
    its twelve accumulators in registers without spills; tiles small enough
    to hold double partials in registers ran four times slower.
  - Each worker thread keeps its buffers from gene to gene in a workspace that
    grows to its largest gene, instead of allocating, faulting in and clearing
    them for every gene, and returns glibc's free pages (`malloc_trim`) after
    a gene whose buffers grew by a megabyte or more: without that, the blocks
    that workers outgrew stayed resident and raised the genome-wide peak by
    0.8 GiB.
  - Shuffles compute their generator values in bulk with the vectorized
    Philox and run the Fisher–Yates swaps over them, with rejection
    thresholds from a table and remainders without division (Lemire, Kaser
    and Kurz 2019); a shuffle that rejects a value continues with the
    following blocks, so every shuffle is unchanged. The shuffles of 10,000
    permutations of 462 samples take 32 ms instead of 59 ms on an efficiency
    core, where two 64-bit divisions per swap had taken most of the time.
  - Permutations that the covariates fit poorly (centred sum of squares at
    most twice the residual sum of squares) are scored from their centred
    response instead of their residual, which skips the product Q T' and its
    subtraction; at the Geuvadis dimensions that is every permutation. The
    float rounding bounds grow by at most √2. This part changes
    single-precision results in their last bits: on the test shapes,
    permutation maxima agree with double precision to within 9.7e-7 of
    max(1, |z|) (1.03e-6 before), within the documented bound. On Geuvadis
    chromosome 22 with 10,000 permutations, statuses, leads and observed
    statistics are unchanged, 76 empirical p-values rise by 1 to 3 counts out
    of 10,001 against double precision (77 before), beta_p agrees with double
    to within 2.3e-5 relative (2.5e-5 before; 4.1e-6 wherever
    beta_p > 1e-10), and both precisions find the same 267 eGenes at Storey
    q < 0.05. With 1,000 permutations the same 12 empirical p-values rise by
    1/1001 as before and beta_p agrees to within 3.8e-5 (at beta_p ≈ 2e-49;
    5.0e-6 wherever beta_p > 1e-10).

  After the first three parts, single-precision output was byte-identical to
  that of 4e2b39a on Geuvadis chromosome 22 with 1,000 and 10,000
  permutations, on the genome-wide panel with 10 permutations, and on test
  shapes covering 37–462 samples, 1–4,500 variants, both precisions, NB
  refits, nominal output and 1, 3 and 4 threads, natively and with
  `RAPIDGLM_FORCE_PORTABLE`. Double precision and NB refits remain byte for
  byte as before: the double-precision chromosome 22 tables with 1,000 and
  10,000 permutations equal those of the 1.4.0 receipts.

  Micro-benchmark (`benchmarks/qtl_precision.cpp`, whose new `genes` mode
  times many genes on many threads): synthetic genes of 462 samples, 43
  covariates and 4,500 variants, 10,000 permutations in single precision, on
  efficiency cores under the timing lock, the best of two alternating rounds
  (one thread on CPU 31, twelve on CPUs 16–27):

  | Build | 1 gene, 1 thread | 12 genes, 1 thread | 48 genes, 12 threads |
  |---|---|---|---|
  | 4e2b39a | 1.017 s | 1.017 s per gene, 20.4 GFMA/s | 0.1020 s per gene, 17.0 GFMA/s per thread |
  | first three parts | 0.924 s | 0.923 s per gene, 22.5 GFMA/s | 0.0823 s per gene, 21.1 GFMA/s per thread |
  | all four parts | 0.912 s | 0.907 s per gene, 22.9 GFMA/s | 0.0806 s per gene, 21.5 GFMA/s per thread |

  (GFMA/s counts the multiply-adds of the score product alone.) Each of twelve
  threads now runs at 94% of the one-thread rate, against 83% for 4e2b39a,
  whose score matrices left the caches; per gene, twelve threads are 1.26×
  and one thread 1.12× faster than 4e2b39a. The packed product with its
  maxima runs at 26.0 GFMA/s on one core (25.0 for the product alone before),
  the portable version at 12.3. Double precision is unchanged (1.72 s for the
  one-gene run, against 1.75 s).

  Peak resident memory, untimed with 12 threads and measured the same way
  for both builds: the genome-wide panel with 10 permutations, 3.35 GiB
  against 3.52 GiB for 4e2b39a (every worker then holds buffers for the HLA
  region's windows of up to 26,032 variants); Geuvadis chromosome 22,
  0.65 GiB in single precision with 1,000 or 10,000 permutations (0.90 and
  0.94 GiB), and 0.82 and 0.88 GiB in double precision (0.84 GiB both).
- **Compact genotype storage.** `Genotypes` gains a hard-call form beside the
  dosage form, which keeps working: `hard_calls` holds the two-bit codes of a
  variant-major PLINK 1 .bed, missing code included, in `call_bytes()` bytes
  per variant, and `validate()` accepts either form. `rapidglm qtl` keeps the
  .bed codes as read (repacked only when the BED's samples are not the .fam's
  in .fam order) instead of expanding every call to a double. `cis_qtl`
  decodes each gene's cis variants into its genotype block as it tests the
  gene, with each variant's mean dosage from its codes
  (`Genotypes::mean_dosage`, four calls at a time through a table of byte
  sums) instead of a mean vector over all variants. Sums
  of whole dosages are exact, so results are bit-identical: the Geuvadis
  chromosome 22 tables (Gaussian with 1,000 permutations in single and in
  double precision, nominal output, NB with 50 permutations, `min_maf` 0.1
  with nominal output) and the genome-wide cis table (10 permutations) are
  byte-identical to those of the previous build, and the double-precision
  chromosome 22 table to that of 1.4.1.
- Other memory reductions, with identical output: the CLI packs variant IDs
  and alleles into one string (31 rather than 128 bytes per Geuvadis variant),
  trims the spare capacity of the .bim vectors and, after `min_maf`
  filtering, of the kept variants' arrays, frees the phenotype table
  once the responses are built; each gene reserves its nominal rows; and
  single-precision permutations no longer hold a batch's scores (about 6.5
  rather than 10 kB per cis variant at 462 samples; see above).
- Peak resident memory, untimed with 12 threads: Geuvadis chromosome 22 with
  1,000 permutations, 1.14 GiB for 1.4.1 → 0.65 GiB in single and 0.82 GiB
  in double precision; the genome-wide panel (20,427 genes, 6,317,351
  variants), measured with 10 permutations, which allocate the same per-gene
  buffers as 10,000, 25.8 → 3.35 GiB (the 1.4.1 run with 10,000 permutations
  peaked at 25.9 GiB; tensorQTL at 7.2 GiB). The calls take 0.68 GiB instead
  of 21.7 GiB; most of the rest is the gene workers' buffers in the HLA
  region, where 143 genes have more than 15,000 cis variants.

## 1.4.1 (2026-10-22)

- Return an empty weighted cross-product matrix immediately when the design
  has zero columns. This avoids undefined behavior from forming row pointers
  into empty storage, including in a Cook's replacement refit when every
  coefficient is at a boundary. Add a regression covering six row counts.
- The focused solver and bulk suites pass ASan/UBSan with OpenMP (29 cases,
  4,553 assertions). All 28 checked nonempty-design cross-products are
  bit-identical to 1.4.0; this patch changes no fitting or resampling rule.
- Benchmark tables and raw receipts remain the completed **1.4.0** run on
  `f0ecfbd` (`v1.4.0-dev.4`). They have not been remeasured for 1.4.1.
  See the [patch validation](validation/receipts/v1.4.1/README.md) and
  [benchmark provenance](validation/receipts/v1.4.0/README.md).

## 1.4.0 (2026-10-22)

The library is the code of `v1.4.0-dev.4`; this release adds the receipts and
documentation measured with it. Every stage in the complete 18-stage release
run was measured afresh. One workstation: Intel Core i9-13900KF, 125 GiB RAM; BIOS 1836, microcode 0x133, 253 W sustained and short-term limits; 4 x 32 GiB at 4,200 MT/s. Turbo is on. Two faulty performance cores (logical CPUs 8-11) are capped at 2.2 GHz and never used for timing. Every timed run is pinned to the 16 identical efficiency cores, one thread per core, under the shared lock after a quiet-host check. Pinning to identical cores makes the comparisons reproducible. The efficiency cores retain their stock 800 MHz-4.3 GHz range. Absolute seconds from other host settings are not directly comparable; use matched speed-up ratios across settings. These timings are final, not provisional.

- **Bulk and pseudobulk DE:** fitted-count flooring (default `min_mean: 0.5`),
  rough/moments starting dispersions, bounded fallback with covariate-unit
  invariance, and independent filtering following the documented and
  black-box-validated rules. Replaced-count base means and failed-refit
  p-values are corrected. Low-count and boundary genes can change results
  relative to 1.3.0; shared trends and priors can also move other genes.
- **Floored-fit interfaces:** C `rapidglm_nb_floored_fits` and Python
  `capi.nb_floored_fits`; the PyDESeq2 inference backend uses native fits and
  delegates unsupported/fallback cases to DefaultInference. MIT PyDESeq2
  ports carry attribution in DEPENDENCIES.md.
- **Quasi-likelihood DE:** `edger-ql` / C++ `edger_ql` implements filtering,
  TMM, common Cox-Reid dispersion, adjusted deviances, moderation and QL
  testing following edgeR 4's publications, documentation and public API
  outputs. Near-Poisson deviances and fit failure handling are stable.
  Inputs must be nonnegative integer counts. Known reference-fit differences
  and small-panel moderation limits are documented in METHODS.md.
- **Cis-QTL mapping:** Gaussian, Poisson and NB score tests, empirical
  permutation calibration and Beta approximation, with PLINK/BED input,
  checked outputs, stable ties, per-gene singular statuses and failed-draw
  accounting. Only phenotype chromosomes are decoded; MAF filtering is
  in-place. Validation covers chromosome 22, not a genome-wide run.
- **Faster Gaussian permutations:** project genotypes once per gene and
  score batches of 512 responses with packed AVX2/FMA or portable kernels.
  On the final matched 10,000-permutation panel, RapidGLM takes
  117.19 s versus tensorQTL's 151.53 s
  (1.29×), with unchanged empirical p-values
  and lead variants relative to dev.3. See
  [QTL_GAUSSIAN_SPEED.md](docs/QTL_GAUSSIAN_SPEED.md).
- **Numerical and input handling:** tiled weighted cross products and
  leverages, vectorized predictor/mean updates, quicker MatrixMarket input
  and SHA-accelerated fingerprints where available; binding, matrix input
  and QTL file validation fixes. Perturbation streams and acceptance-rule
  definitions are unchanged from 1.3.0.
- **ARM64:** cross-build support, selected QEMU checks, and double-precision
  KS CDF checks with unchanged integer streams. Floating-point results can
  differ by architecture; numerical limits are retained explicitly.
- **Validation:** native and portable C++ (172 cases each), C99, CLI (28
  cases on each build), Python/R, Rust, selected ARM64 and container suites
  pass. Final agreements and timing scopes are in the
  [release collection](validation/receipts/v1.4.0/README.md).

## 1.3.0 (2026-10-10)

The perturbation workflow (`sceptre`). Bulk DE, single fits, batch fits and
score scans are unchanged. The library is the code of the development release
`v1.3.0-dev.1`; this release adds the receipts and documents measured with it.

**For a user of 1.2.0 who changes no option:** every resampled p-value of
`sceptre` changes, because a pair's draws are now keyed by the pair itself and
no longer by its place in the list; the new values are as valid as the old and
differ from them as two seeds differ, so discoveries can change at the margin.
Adjusted p-values are now computed over the pairs that pass pair QC, so in a
list that includes pairs failing pair QC they are smaller, more pairs can be
discoveries, and the failing pairs report a missing adjusted p-value instead of
1. A list that holds the same response and target twice within a kind is now
an error. `stream_keys: "position"` with `adjustment_family: "listed"`
reproduces the tables of 1.2.0 byte for byte (checked on six runs).

- **Streams keyed by identity** (`stream_keys`, default `identity`; C++
  `StreamKeys`). A pair's random stream depends only on the seed, its response
  and its target (its guide for per-guide tests), so its record is the same
  wherever it is listed, whatever else is listed, and however the run is
  divided among threads, tiles, shards or separate runs. `response_ids` names
  the responses (gene or peak names); without it a response is identified by
  its column. Reused draws and the shared screen are keyed by the target. The
  keys are FNV-1a hashes of the names, turned into a stream by Philox
  (`name_key`, `keyed_stream`; [METHODS.md](METHODS.md), Resampling).
  `position` keeps the keys of earlier releases.
- **The adjustment family** (`adjustment_family`, default `tested`; C++
  `AdjustmentFamily`): the pairs that pass pair QC, as in SCEPTRE, whose 101
  calls on the Gasperini benchmark are BH over the 9,718 pairs passing its
  pairwise QC and not over the 10,000 listed. `listed` is the family of earlier
  releases. `family_size_*` gives a family's size to a run that lists only some
  of its pairs. The run record stores the pairs listed, passing pair QC and in
  each family.
- **No work for a pair that cannot be tested.** Pair QC is decided first, from
  the target's cells and the response's nonzero cells. A response's null model
  is fitted only when one of its pairs passes (complement controls), a pair's
  cells are gathered only when it passes (non-targeting controls), and the
  start saved for a later effect fit is kept only for tested pairs (352 bytes
  less for every other listed pair). Records are unchanged: with position keys
  the tables are those of 1.2.0.
- **Targets and responses without a list** (`sceptre_grid`, `sceptre_pair_qc`;
  `targets` and `grid_responses` in the CLI; `rapidglm_sceptre_grid` in C;
  `sceptre_grid` in Python and R). Every named target is tested against every
  named response. One pass over each response's nonzero cells decides pair QC
  for all targets; the pairs that pass are tested and get records, and the
  others are tallied by target and kind (too few treated cells with a nonzero
  count, or too few control cells with one). The records are those of listing
  the same pairs.
- Measured on a synthetic screen in which 5% of the requested pairs pass pair
  QC (20,000 cells; 12.4 million peak pairs; 12 threads on efficiency cores,
  with `sequential_screen` and `early_stop_p`;
  [validation/UNTESTABLE_PAIRS.json](validation/UNTESTABLE_PAIRS.json)):

  | Run | 1.2.0, every pair listed | 1.3.0, every pair listed | 1.3.0, targets and responses named |
  |---|---|---|---|
  | High MOI | 202 s, 11.9 GB | 192 s, 7.9 GB | 167 s, 1.1 GB |
  | Low MOI | 87 s, 11.8 GB | 59 s, 7.5 GB | 37 s, 0.7 GB |

  Listing only the pairs that pass takes the time of the request. A listed pair
  that is not tested now costs about 0.57 kB of memory in the CLI (0.92 kB in
  1.2.0), most of it the list itself and the pair's row; in a request it costs
  none. What remains of a list's time is reading it and writing a row for
  every pair.
- Engine: `ResampleOptions::prescreen_seed` (a shared prescreen under its own
  seed), `adjust_pvalues(p, bonferroni, family_size)`, and
  `SceptreResult::null_fits`.
- Every benchmark was measured again with this version
  ([docs/BENCHMARK.md](docs/BENCHMARK.md)). The perturbation workflow is 4.5
  times faster than SCEPTRE on 50,000 cells (one thread) and 4.4 times on
  207,324 cells (eight threads), about a tenth faster than 1.2.0. With the new
  draws the full screen gives 96 discoveries against SCEPTRE's 101 at the
  benchmark's seed (Jaccard 0.89; 101 and 0.92 with the draws of 1.2.0). Over
  six seeds the Jaccard index is 0.89 to 0.96, and 0.90 to 0.96 between two
  RapidGLM seeds
  ([validation/PERTURBATION_SEEDS.json](validation/PERTURBATION_SEEDS.json),
  `scripts/validate_perturbation_seeds.py`).
- The SCEPTRE-agreement receipt takes the mean discovery count of its gate G2
  over 200 seeds (`--gate-seeds`) instead of ten. Over ten seeds the mean's
  standard error is a third of the gate's margin on the public high-MOI panel,
  and the outcome turned on the seeds: 12.4 (pass) with the draws of 1.2.0 and
  13.2 (fail) with those of 1.3.0, for the same ten seeds. Over 200 seeds both
  give 12.3 against SCEPTRE's 11, within the margin of 1.65
  ([docs/SCEPTRE_STAGING.md](docs/SCEPTRE_STAGING.md)).

## 1.2.0 (2026-10-07)

The library is the code of the development release `v1.2.0-dev.1`; this
release adds the receipts and documents measured with it.

- `legacy_cooks` is now the default in bulk differential expression (CLI, C,
  Python, R and the PyDESeq2-style API): Cook's distances use DESeq2's robust
  dispersion as implemented in PyDESeq2 (MIT; Muzellec et al. 2023), with
  PyDESeq2's flagging rule. Default discoveries then agree with DESeq2's on GTEx
  thyroid at Jaccard 0.997 (128 samples) and 1.000 (647 samples), against 0.58
  and 0.90 with RapidGLM's own estimator, which sets aside more genes in
  designs with many small groups
  ([docs/BULK_DE.md](docs/BULK_DE.md#legacy-cooks-distances-2026-10-01)).
  `legacy_cooks: false` selects that estimator and gives the tables of 1.1.0.
  Genes flagged or replaced as Cook's outliers, and so some p-values and
  adjusted p-values, change for designs with replicated cells.
- In the Rust binding, `BulkOptions::legacy_cooks` is an `Option<bool>`; `None`
  keeps the default.
- Fits are faster, with estimates that agree with 1.1.0's to rounding
  ([METHODS.md](METHODS.md)). A fixed-dispersion NB fit of 647 observations and
  12 columns takes 132 to 151 microseconds on one core, against 245; `bulk-de`
  on GTEx thyroid (647 samples, one thread) takes 21 s against 49 s, and 0.63 s
  against 0.98 s on airway. The time went into work that the estimates did not
  need:
  - A further Newton step is no longer taken only to confirm convergence: the
    last step's factor predicts it (the chord step), which is then applied as a
    refinement. Fits report one iteration fewer.
  - Likelihood terms that depend on the response alone used three long double
    `lgamma` calls per count above 64; they now take one logarithm each, by
    Stirling's series in a form without cancellation, and are more accurate
    near the Poisson limit. The polygamma sums of dispersion estimation
    likewise.
  - The deviance that scales the convergence criterion, and the signature that
    identifies a fit's response, are computed in a fraction of the time.
  - Designs with at most half their entries nonzero (designs of factors) form
    their cross-products from the nonzero entries of each row.
  - Fits of many responses to one design share its cross-product, rank check
    and row groups (`FitOptions::design`, from `prepare_design`); `fit_batch`,
    `bulk_de`, score scans and the perturbation workflow do so.
  - The rank check of the positive-count design is skipped when no count is
    zero.
- Results change in the last digits. Where a dispersion estimate lies on a flat
  likelihood (genes close to Poisson), it can settle at another value. On
  airway four gene-wise estimates move. The prior variance, a grid choice,
  moves with them from 0.697 to 0.700, and the independent-filtering threshold
  from 3.86 to 4.50 (DESeq2's value): 5,164 genes are discoveries against
  5,142 (31 gained, 9 lost). On GTEx thyroid and the other four validation
  datasets every column agrees to 3e-10 (relative), with either setting of
  `legacy_cooks`.
- Every benchmark was measured again with this version, each thread on its own
  efficiency core of one workstation
  ([docs/BENCHMARK.md](docs/BENCHMARK.md#host)). Bulk DE is 6.1 to 13.9 times
  faster than DESeq2 on one thread (airway to 647 thyroid samples), and the
  perturbation workflow 3.9 to 4.5 times faster than SCEPTRE. Against Ceres
  Solver, fixed-dispersion fits are 2.1 to 2.7 times faster than
  Levenberg–Marquardt at equal accuracy
  ([docs/SOLVERS.md](docs/SOLVERS.md)); the solver benchmark also times a plain
  `fit` per gene (last argument `shared`, `plain` or `both`).
- New receipt [validation/CRT_SCALING.json](validation/CRT_SCALING.json): the
  conditional randomization test on panels of the Gasperini screen, with
  `fit_check: "ks"` and with the defaults
  (`scripts/prepare_crt_panels.py`). `validation/COOKS_LEGACY.json` and
  `validation/COOKS_THYROID.json` are written by `scripts/validate_cooks.py`,
  and the SCEPTRE agreement receipt repeats its seeds under the defaults.
- Benchmark harnesses can confine runs to listed CPUs and record the share of
  those CPUs that other processes used (`--cpus`).
- The release workflow publishes a pre-release for a tag with a suffix, such as
  `v1.2.0-dev.1`.

## 1.1.0 (2026-10-06)

- Small skew-normal p-values are confirmed on more draws by default
  (`confirm_draws`; `ResampleOptions::confirm`; C, Python and R interfaces).
  With the strict fit check, a fit that is sound on the last budget but whose
  p-value that budget cannot check (fewer than half a draw expected beyond the
  statistic: a p-value below about 5e-5 with 19,999 draws) is no longer
  reported from there. The pair draws on to ten times the last budget in all,
  199,999 for the default budgets, and the fit made on those draws is reported
  if it is sound, the empirical p-value otherwise. A fit rejected on the last
  budget ends there as before, so only candidates for very small p-values draw
  more. `confirm_draws` sets another number; 0 gives the results of 1.0.0. On
  100 targets x 7,418 genes of the Gasperini screen, 55 of 721,266 pairs draw
  the confirmation (14% more draws); 50 fits are confirmed and 5 rejected,
  among them the discovery that direct counts contradict, and the 13
  discoveries that remain are the ones a four-budget run to 199,999 finds
  ([validation/SKEW_NORMAL_CHECK.json](validation/SKEW_NORMAL_CHECK.json)).
  P-values of pairs that draw the confirmation change; a tiled plan includes
  the setting, so a checkpoint written by 1.0.0 is not resumed.
- A stage that can end early is drawn by several threads once it has gone
  1,024 draws without ending (one more chunk in flight per further 1,024
  draws; at most about 3% of draws generated beyond a stage's end). Long
  stages no longer run on one thread: that panel takes 289 s with the
  confirmation draws, against 292 s without them in 1.0.0. Results do not
  depend on it.

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
