# rapidglm (Rust)

Rust bindings to RapidGLM through its C interface (`librapidglm_c`,
[docs/C_API.md](../../docs/C_API.md)), and a Rust port of PyDESeq2's workflow
that runs its per-gene fits in RapidGLM.

```rust
use rapidglm::deseq::{DeseqDataSet, DeseqOptions, DeseqStats, StatsOptions};

// counts: samples x genes; design: samples x columns (row-major rapidglm::Matrix)
let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default())?;
dds.deseq2()?;
let res = DeseqStats::new(&dds, contrast, StatsOptions::default())?.summary()?;
// res.base_mean, log2_fold_change, lfc_se, stat, pvalue, padj
```

`rapidglm::bulk_de` runs RapidGLM's own DESeq2-procedure analysis (the CLI's
`bulk-de`) and returns its result table; `nb_fixed_fits` and `nb_dispersions`
are the batch fitters the port is built on.

Division policy defaults to `RAPIDGLM_DIVISION=slow`. Set
`BulkOptions.division` to `DivisionMode::Slow` or `DivisionMode::Fast`,
or use `with_division(mode, || ...)` around synchronous binding calls.
`DivisionMode::Inherit` follows the calling thread's policy, then the
environment or build default. The scope restores the previous policy on
return, error or panic; it does not affect other Rust threads.

## Building

Build the C library, then point Cargo at it:

```sh
cmake -S . -B build $(scripts/local_deps.sh) && cmake --build build --target rapidglm_c
export RAPIDGLM_LIB_DIR=$PWD/build OPENBLAS_NUM_THREADS=1
cd bindings/rust && cargo test --release
cargo run --release --example deseq -- counts.mtx design.tsv contrast.txt results.tsv
```

The binding declares the C functions by hand in `src/ffi.rs` (no `bindgen`);
its only crate dependency is `libm`. `RAPIDGLM_LIB_DIR` is also written into
the runpath of tests and examples.

## The PyDESeq2 port

`rapidglm::deseq` follows PyDESeq2 0.5.2 (MIT, Copyright (c) 2022 Owkin;
Muzellec et al. 2023, Bioinformatics 39:btad547), ported from its source with
the author's approval: the same steps, defaults and result columns as
`DeseqDataSet.deseq2()` and `DeseqStats.summary()`. NB fits and dispersions
run in RapidGLM using unfloored fits; the Python inference backend uses floored
fits from 1.4.0. The
independent-filtering lowess follows PyDESeq2's, which is adapted from
Alexandre Gramfort's (BSD 3-clause). Differences are listed in
[src/deseq/mod.rs](src/deseq/mod.rs); the main ones:

- exact fixed-dispersion MLE and Newton dispersion estimates (RapidGLM) instead
  of PyDESeq2's IRLS and L-BFGS-B; genes whose MLE does not exist (an all-zero
  group) get a bounded fit with PyDESeq2's ridge;
- the dispersion trend's gamma GLM is solved to convergence rather than to
  L-BFGS-B's tolerance;
- Cook's distances at samples with leverage 1 are 0 instead of 0 / 0 in
  floating point (often infinite), which changes which genes are flagged when
  some samples have leverage one; singleton design cells alone do not imply it;
- designs are matrices and contrasts vectors (no formulas); the iterative size
  factors and apeGLM shrinkage are not ported.

`tests/compare_pydeseq2.py` runs PyDESeq2 with the RapidGLM back end on a
dataset and compares the results with `examples/deseq.rs`.

The Rust DESeq workflow currently uses unfloored NB fits. It therefore differs
from the Python inference backend's floored fits introduced in 1.4.0;
low-count and boundary-fit results are not expected to match that backend.
Singleton design cells in the comparison receipt do not imply leverage one
in an additive design.
