# C interface

`librapidglm_c` exposes the engine and the two applications through a C ABI,
declared in [include/rapidglm/rapidglm_c.h](../include/rapidglm/rapidglm_c.h).
Other languages (Python, Rust, Julia, R) call it through their foreign-function
interfaces instead of compiling C++. The C++ headers remain the primary API;
the C layer only converts inputs and results, and its outputs are bit-identical
to the C++ calls (`tests/test_c_api.cpp`).

| Function | C++ equivalent | CLI command |
|---|---|---|
| `rapidglm_fit_batch` | `fit_batch` | `fit` |
| `rapidglm_bulk_de` | `bulk_de` | `bulk-de` |
| `rapidglm_sceptre` | `sceptre` | `sceptre` |
| `rapidglm_sceptre_grid` | `sceptre_grid` | `sceptre` with `targets` |
| `rapidglm_scan` | `score_scan` | `scan` |

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
family of the tested pairs. `rapidglm_sceptre_grid` takes target names and
response columns instead of a list of pairs and returns a second table, the
tally of the pairs that fail pair QC by target and kind.

## Language bindings

- **Python** ([bindings/python](../bindings/python)) loads the library with
  `ctypes`; nothing is compiled.
- **R** ([bindings/r](../bindings/r)) compiles `src/c_api.cpp` into the package
  behind a `cpp11` shim.
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
  `sceptre` with each stream key and adjustment family; and `sceptre_grid`.
- Both run under ctest, with and without the sanitizers.
