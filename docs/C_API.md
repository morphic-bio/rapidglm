# C interface

`librapidglm_c` exposes the engine and the applications through a C ABI,
declared in [include/rapidglm/rapidglm_c.h](../include/rapidglm/rapidglm_c.h).
Other languages (Python, Rust, Julia, R) call it through their foreign-function
interfaces instead of compiling C++. The C++ headers remain the primary API;
the C layer only converts inputs and results, and its outputs are bit-identical
to the C++ calls (`tests/test_c_api.cpp`).

| Function | C++ equivalent | CLI command |
|---|---|---|
| `rapidglm_fit_batch` | `fit_batch` | `fit` |
| `rapidglm_bulk_de` | `bulk_de` | `bulk-de` |
| `rapidglm_edger_ql` | `edger_ql` | `edger-ql` |
| `rapidglm_sceptre` | `sceptre` | `sceptre` |
| `rapidglm_sceptre_grid` | `sceptre_grid` | `sceptre` with `targets` |
| `rapidglm_scan` | `score_scan` | `scan` |
| `rapidglm_qtl` | `cis_qtl` | `qtl` |
| `rapidglm_log_array`, `rapidglm_log1p_array`, `rapidglm_exp_array`, `rapidglm_lgamma_array`, `rapidglm_digamma_array`, `rapidglm_trigamma_array` | `vmath::log_array`, ... (`include/rapidglm/vmath.hpp`) | |

## Building and linking

The normal build produces `librapidglm_c.so` (with a version and soname) and
the static `librapidglm_c.a`; both are installed with the CMake package.

```cmake
find_package(rapidglm CONFIG REQUIRED)
target_link_libraries(app PRIVATE rapidglm::rapidglm_c)
```

Without CMake: `cc app.c -lrapidglm_c`. The shared library carries its BLAS and
LAPACK dependencies. The installed library keeps the directories of a BLAS
outside the system paths in its runpath. `scripts/check_external.sh` builds a C
consumer ([examples/external/main.c](../examples/external/main.c)) against the
installed package.

## Conventions

- **Errors.** Functions return `RAPIDGLM_OK` (0), `RAPIDGLM_ERROR_INVALID` for
  invalid input or `RAPIDGLM_ERROR_FAILED` for other failures. The message is
  in `rapidglm_last_error()`, per thread. No C++ exception crosses the
  interface. On error, `*result` is left unchanged.
- **Matrices.** Designs are dense and row-major (observations × columns).
  Responses, counts and guide assignments are compressed sparse column (CSC)
  with observations as rows and features as columns. Row indices must be sorted
  and unique within each column. Indices are `size_t`. Inputs are copied; the
  caller keeps ownership.
- **Options.** Each option struct begins with `struct_size`. Fill it with its
  `*_init` function, then change fields. New fields are only ever appended, so
  a program built against an older header keeps working. A struct that was not
  initialized is refused.
- **Results.** Each call returns a `rapidglm_table` that the caller frees with
  `rapidglm_table_free`. Columns are numeric (`double`, NaN when missing) or
  text. Their names match the CLI's output tables. Run-level values, such as the
  size factors and prior variance of `bulk_de` or the rejection counts of
  `sceptre`, are named scalars and vectors on the table. Pointers into a table
  stay valid until it is freed.
- **Threads.** `threads` in the options (or the argument of
  `rapidglm_fit_batch`) sets the OpenMP workers; results do not depend on it.
  The calls are reentrant. The BLAS advice in the README applies.

## Example

```c
rapidglm_bulk_options options;
rapidglm_table *table = NULL;
rapidglm_bulk_options_init(&options);          /* tests the last design column */
options.lfc_shrinkage = RAPIDGLM_SHRINK_NORMAL;
if (rapidglm_bulk_de(&design, &counts, &options, &table) != RAPIDGLM_OK) {
    fprintf(stderr, "%s\n", rapidglm_last_error());
    return 1;
}
const double *padj = rapidglm_table_numeric(table, "padj");
size_t samples;
const double *size_factors = rapidglm_table_vector(table, "size_factors", &samples);
rapidglm_table_free(table);
```

`rapidglm_bulk_options.contrasts` takes one row of weights (a contrast) or
several rows (a joint likelihood ratio test), as in
[HYPOTHESES.md](HYPOTHESES.md). `rapidglm_sceptre_options_init(&o, high_moi)`
sets the MOI-dependent defaults for the control group and calibration.

From 1.3.0 `rapidglm_sceptre_options` ends with `stream_keys`, `response_ids`,
`family` and `family_size` ([CLI.md](CLI.md#perturbation-workflow), stream keys
and adjustment family). A caller built against an earlier header passes a
shorter struct and gets their defaults: streams keyed by identity and the
family of the tested pairs. 1.6.0 appends `null_start`
(`RAPIDGLM_NULL_START_REFINE`, the default, or `RAPIDGLM_NULL_START_COLD`;
[CLI.md](CLI.md#perturbation-workflow), null start), and the `rapidglm_sceptre`
table gains the scalars `null_start_used`, `null_refined` and
`null_cold_refits`. `rapidglm_sceptre_grid` takes target names and
response columns instead of a list of pairs and returns a second table, the
tally of the pairs that fail pair QC by target and kind.

1.6.0 then appends `batch_scoring` (default 1): CRT with complement controls
scores each response's pairs together, with the records of scoring them one by
one (0). A caller built against an earlier header gets the defaults of both.

## Cis-QTL mapping (`rapidglm_qtl`, since 1.6.1)

`rapidglm_qtl` runs the analysis of `rapidglm qtl` ([CLI.md](CLI.md#cis-qtl-mapping-qtl))
on inputs in memory, and shares its code with the CLI (`src/qtl_run.hpp`: the
BED anchoring of phenotypes, the `min_maf` filter, the response matrix, library
offsets, the run summary and the column names, then `cis_qtl`). The tables equal
the CLI's for the same inputs.

- **Phenotypes** (`rapidglm_qtl_phenotypes`): a `rapidglm_dense` of phenotypes ×
  samples, row-major (one phenotype's values are contiguous, as in a phenotype
  BED), with `id`, `chrom`, `start` and `end` per phenotype. Coordinates are those
  of a BED (zero-based, half-open; the anchor is `[start + 1, end]`; `start = end`
  anchors at that coordinate).
- **Genotypes** (`rapidglm_qtl_genotypes`): `int8` calls, variants × samples,
  row-major (one variant's calls are contiguous, as in a `.bed`): the number of
  copies of allele 1 (0, 1, 2), negative for missing; any other value is
  refused. Missing calls take the variant's mean. The library keeps the calls as
  PLINK's two-bit codes, as the CLI does, so a call costs a quarter of a byte.
  `chrom`, `position` (1-based) and `id` per variant; `allele1` and `allele2` are
  optional (reported as `NA`). Variants on chromosomes without a phenotype are
  ignored, as in the CLI. Dosages that are not whole numbers are not accepted
  (neither does the CLI).
- **Samples** are the same, in the same order, in every input; nothing is
  matched by ID. `covariates` (samples × covariates, row-major, no intercept; one
  is added; may be NULL) and `library_sizes` (one positive size per sample, count
  families only; may be NULL) are separate arguments.
- **Options** (`rapidglm_qtl_options`): `model` (family gaussian, Poisson or NB;
  dispersion, theta, tolerance, ridge, max_iterations, robust, cox_reid),
  `window`, `permutations`, `nominal`, `seed`, `threads`, `min_maf`,
  `permutation_precision` and `permutation_fits`, with the CLI's defaults.
  `model.offset` and `model.weights` must be NULL.
- **Results**: the cis table (the columns of `PREFIX.cis.tsv`, `NaN` where the CLI
  prints `NA`), the nominal table (those of `PREFIX.nominal.tsv`; no rows unless
  `nominal`; pass NULL to skip it) and scalars (`samples`, `phenotypes`,
  `phenotypes_tested`, `variants`, `covariates`, `window`, `permutations`,
  `min_maf`, `threads`, `permutations_refitted`, `permutation_precision_used`
  and `permutation_fits_used`). The seed is not echoed (a `double` cannot hold
  every 64-bit seed).

## Language bindings

- **Python** ([bindings/python](../bindings/python)) loads the library with
  `ctypes`; nothing is compiled.
- **R** ([bindings/r](../bindings/r)) compiles `src/c_api.cpp` (and the
  `src/qtl_run.hpp` it includes) into the package behind a `cpp11` shim.
- **Rust** ([bindings/rust](../bindings/rust)) declares the functions in an
  `extern "C"` block (`src/ffi.rs`), links `librapidglm_c` from
  `RAPIDGLM_LIB_DIR` in `build.rs`, and wraps result tables in a type that frees
  them on drop. On it sits a Rust port of PyDESeq2's workflow.
- **Julia** can reach the same functions with `ccall`. Julia's
  `SparseMatrixCSC` has the CSC layout but one-based `Int` indices, which must be
  shifted to zero-based `size_t`; dense designs must be passed row-major (a
  transpose of Julia's column-major arrays). No Julia binding is shipped yet.

## Tests

- `rapidglm_c_test` is plain C99 and calls every entry point, including the
  error paths.
- The `c_api_*` Boost cases compare every numeric column with the C++ results
  bit for bit: `fit_batch`; `bulk_de` with shrinkage and a contrast; `score_scan`;
  `sceptre` with each stream key and adjustment family; `sceptre_grid`; and
  `qtl` against `cis_qtl` (with dosage genotypes) in each family.
- Both run under ctest, with and without the sanitizers.
- `vmath_*` Boost cases check the special functions: every path against the
  others bit for bit, known-answer digests, mpmath references and special
  values. The Python (`bindings/python/tests/test_special.py`) and R
  (`bindings/r/tests/special.R`) layers reproduce the C++ values of RapidGLM's
  functions in `tests/fixtures/vmath_known_values.tsv` bit for bit.

C API feature version 5 (1.6.1) adds `rapidglm_qtl`, `rapidglm_qtl_options`,
`rapidglm_qtl_genotypes` and `rapidglm_qtl_phenotypes`; nothing of version 4
changes. C API feature version 2 adds the 1.4.0 entry points and version 4 (1.6.0) the
`null_start` and `batch_scoring` fields of `rapidglm_sceptre_options`, the
`null_start_used`, `null_refined` and `null_cold_refits` scalars of the
`rapidglm_sceptre` table, the `crt_draw_used` scalar of the `rapidglm_sceptre` and `rapidglm_scan` tables
(the `RAPIDGLM_CRT_*` sampler that drew the CRT labels) and the special
functions: `rapidglm_<f>_array(x, out, n)` sets `out[i] = f(x[i])` for
`log`, `log1p`, `exp`, `digamma` and `trigamma` (RapidGLM's; digamma and
trigamma for x >= 0) and `lgamma` (the C library's `lgamma_r` in slow mode,
and RapidGLM's positive-argument kernel in fast mode, with libm for negative
arguments); `out` may be
`x`, and `rapidglm_special_path()` names the path RapidGLM's functions take
(`avx2`, `neon` or `portable`). RapidGLM's results are the same bits on every
processor (METHODS.md, Numerical backend). Version 3 was used only by
development builds. Python checks this version
before resolving symbols; rebuild bindings and libraries together. Existing
legacy entry points remain available.

For `rapidglm_nb_floored_fits`, a nonzero convergence code indicates success:
1 means IRLS convergence, 2 means native bounded-optimizer convergence.
The Python wrapper returns booleans by default; `return_optimizer=True` also
returns the optimizer mask so `RapidGLMInference.irls` can delegate every such
fallback to PyDESeq2's default inference.

`rapidglm_set_division(RAPIDGLM_DIVISION_FAST, &previous)` selects the calling
thread's numerical policy. Restore it with `rapidglm_set_division(previous, NULL)`.
The other choices are `RAPIDGLM_DIVISION_SLOW` and `RAPIDGLM_DIVISION_INHERIT`
(environment/build default). Worker threads inherit the resolved policy.
`rapidglm_division_mode()` returns the resolved 0 (slow) or 1 (fast), or -2 on an
invalid environment setting. Result tables include scalar `division_mode`.
This thread-local option leaves the layout of all existing C option structures
unchanged. Python and R use a scoped `division="slow"`/`"fast"` parameter;
Rust provides `DivisionMode`, `with_division`, and `BulkOptions.division`.
