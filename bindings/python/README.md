# rapidglm (Python)

Bulk differential expression, SCEPTRE-method perturbation analysis, score scans
and batch GLM fits with the RapidGLM engine, plus statsmodels-compatible GLM
fitting.

## Applications (C interface)

```python
import rapidglm

# counts: samples x genes (array, DataFrame or SciPy sparse); design: samples x columns
res = rapidglm.bulk_de(design, counts, lfc_shrink="normal", threads=8)
res.attrs["size_factors"]

# responses: cells x genes; assignments: cells x guides
res = rapidglm.sceptre(design, responses, assignments, guides, pairs, moi="high", threads=8)
# every target against every response, without a list: rows for the pairs that
# pass pair QC, and a tally of the others by target
res, tally = rapidglm.sceptre_grid(design, responses, assignments, guides,
                                   discovery_targets, negative_targets, moi="high", threads=8)
```

`bulk_de`, `sceptre`, `scan` and `fit_batch` call `librapidglm_c` through
ctypes ([docs/C_API.md](../../docs/C_API.md)); nothing is compiled for Python.
Arguments take the CLI configuration keys and values
([docs/CLI.md](../../docs/CLI.md)). Results are pandas DataFrames with the
CLI's columns; run-level values are in `.attrs`. Gene names come from DataFrame
column names. Build the library and point the module at it:

```sh
cmake -S . -B build $(scripts/local_deps.sh) && cmake --build build --target rapidglm_c
export RAPIDGLM_C_LIBRARY=$PWD/build/librapidglm_c.so PYTHONPATH=$PWD/bindings/python
python3 bindings/python/tests/test_capi.py build/rapidglm
```

The test checks every column against the CLI bit for bit: bulk-de with
shrinkage, contrasts, joint LRTs and local trends; fit; permutation and
asymptotic scans; and three SCEPTRE configurations.

## PyDESeq2-style interface

`rapidglm.deseq` offers PyDESeq2's workflow and argument names on the same
engine. It was designed from PyDESeq2's published documentation; no code is
shared.

```python
from rapidglm.deseq import DeseqDataSet, DeseqStats

dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~batch + condition", n_cpus=8)
dds.deseq2()
ds = DeseqStats(dds, contrast=["condition", "B", "A"])
ds.summary()          # ds.results_df: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
ds.lfc_shrink(coeff="condition[T.B]")  # normal prior
```

Design formulas give patsy's design matrices (factor and numeric main
effects, and interactions of terms present as main effects). A test run that
matches the dataset's own run is reused. Differences from PyDESeq2:
- `lfc_shrink` uses DESeq2's normal prior, not apeGLM;
- `varm["LFC"]` holds no full coefficient matrix;
- VST, plots, `lfc_null`, `alt_hypothesis`, `control_genes` and iterative size
  factors raise `NotImplementedError`.

`tests/test_deseq.py` checks design matrices against patsy and every result
column against `bulk_de`. On airway (`~cell + dex`) it checks them against the
CLI bit for bit.

## PyDESeq2 inference back end

PyDESeq2 (MIT) takes an `inference` object implementing its documented
`Inference` interface. `rapidglm.pydeseq2.RapidGLMInference` replaces its
per-gene NB fits (`irls`), dispersion estimates (`alpha_mle`) and Wald tests
with RapidGLM's batch versions. Everything else is PyDESeq2's own:

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from rapidglm.pydeseq2 import RapidGLMInference

inference = RapidGLMInference(n_cpus=8)
dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition", inference=inference)
dds.deseq2()
DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference).summary()
```

On airway, a PyDESeq2 run took 8.8 s instead of 86 s on one thread (shared
host). Discoveries agree with PyDESeq2's default at Jaccard 0.99. The Wald test
matches PyDESeq2's to rounding. Fits and dispersions differ only where
RapidGLM's optimum is better:
- PyDESeq2's IRLS floors means at `min_mu` while fitting;
- its L-BFGS-B stops early;
- genes that start at the dispersion floor keep it, as in DESeq2.

`tests/test_pydeseq2.py` checks each replaced method against PyDESeq2's on
identical inputs, and whole analyses against each other. The module was
written against PyDESeq2's published interface; PyDESeq2 is run as a black
box.

## statsmodels drop-in

```python
import statsmodels.api as sm
import rapidglm.statsmodels as rsm

result = rsm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(depth)).fit()
result.summary(); result.conf_int(); result.get_prediction(X_new)
```

`rsm.GLM` subclasses `statsmodels.api.GLM` and returns statsmodels
`GLMResults`, including robust covariances via `cov_type`. Gaussian (identity),
Poisson (log), binomial (logit, including trial counts) and negative binomial
with fixed alpha (log) are fitted natively with offsets, exposure, frequency
and variance weights. Other families or links, non-IRLS methods, extra fit
options and rank-deficient designs use statsmodels' own fitting (and so keep its
pseudo-inverse behaviour). `rapidglm.fit()` exposes the engine directly,
including negative binomial with maximum-likelihood theta.

Build from the repository root (pybind11 required):

```sh
cmake -S . -B build -DRAPIDGLM_PYTHON=ON -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
      -DPython_EXECUTABLE=$(which python3) $(scripts/local_deps.sh)
cmake --build build --target _rapidglm
PYTHONPATH=bindings/python python3 bindings/python/tests/test_statsmodels.py
```

The module links the same LAPACKE/BLAS as the C++ library; set
`OPENBLAS_NUM_THREADS=1` when calling it from several threads.
