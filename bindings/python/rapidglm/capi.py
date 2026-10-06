"""Bulk differential expression, the SCEPTRE-method perturbation workflow,
score scans and batch GLM fits through the RapidGLM C interface
(``librapidglm_c``; see docs/C_API.md).

Matrices are observations (samples or cells) by features, as in AnnData:
``counts`` and ``responses`` may be dense arrays, pandas DataFrames or SciPy
sparse matrices. Results are pandas DataFrames whose columns match the CLI
tables; run-level values (size factors, prior variances, rejection counts) are
in ``result.attrs``. Option names and values follow the CLI configuration keys
(docs/CLI.md).

The shared library is found through ``RAPIDGLM_C_LIBRARY``, next to this
package, or on the system library path.
"""

import ctypes
import ctypes.util
import os
import pathlib

import numpy as np
import pandas as pd

__all__ = ["bulk_de", "bulk_adjust", "nb_fixed_fits", "nb_dispersions", "sceptre", "scan", "fit_batch", "library_version"]

_size = ctypes.c_size_t
_double_p = ctypes.POINTER(ctypes.c_double)
_size_p = ctypes.POINTER(ctypes.c_size_t)
_text_p = ctypes.POINTER(ctypes.c_char_p)
_int_p = ctypes.POINTER(ctypes.c_int)


class _Dense(ctypes.Structure):
    _fields_ = [("rows", _size), ("cols", _size), ("values", _double_p)]


class _Sparse(ctypes.Structure):
    _fields_ = [
        ("rows", _size),
        ("cols", _size),
        ("column_ptr", _size_p),
        ("row_index", _size_p),
        ("values", _double_p),
    ]


class _FitOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", _size),
        ("family", ctypes.c_int),
        ("dispersion", ctypes.c_int),
        ("theta", ctypes.c_double),
        ("tolerance", ctypes.c_double),
        ("ridge", ctypes.c_double),
        ("max_iterations", _size),
        ("offset", _double_p),
        ("weights", _double_p),
        ("robust", ctypes.c_int),
        ("cox_reid", ctypes.c_int),
        ("singular_ok", ctypes.c_int),
    ]


class _BulkOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", _size),
        ("coefficient", _size),
        ("contrasts", _double_p),
        ("contrast_rows", _size),
        ("lrt", ctypes.c_int),
        ("size_factor_method", ctypes.c_int),
        ("size_factors", _double_p),
        ("trend", ctypes.c_int),
        ("min_mean", ctypes.c_double),
        ("outlier_sd", ctypes.c_double),
        ("cooks_cutoff", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("keep_minimum_start", ctypes.c_int),
        ("cooks_filter", ctypes.c_int),
        ("independent_filtering", ctypes.c_int),
        ("min_replicates_for_replace", _size),
        ("lfc_shrinkage", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("threads", ctypes.c_int),
        ("min_dispersion", ctypes.c_double),
        ("max_dispersion", ctypes.c_double),
        ("legacy_cooks", ctypes.c_int),
    ]


class _SceptreOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", _size),
        ("model", _FitOptions),
        ("null_model", ctypes.c_int),
        ("high_moi", ctypes.c_int),
        ("nt_controls", ctypes.c_int),
        ("crt", ctypes.c_int),
        ("integration", ctypes.c_int),
        ("side", ctypes.c_int),
        ("approximate", ctypes.c_int),
        ("effect_fits", ctypes.c_int),
        ("budgets", _size_p),
        ("n_budgets", _size),
        ("screen", ctypes.c_double),
        ("cell_qc", ctypes.c_int),
        ("assignment_qc", ctypes.c_int),
        ("min_nonzero", _size),
        ("alpha", ctypes.c_double),
        ("lower_quantile", ctypes.c_double),
        ("upper_quantile", ctypes.c_double),
        ("max_mito", ctypes.c_double),
        ("qc_total", _double_p),
        ("qc_detected", _double_p),
        ("qc_mito", _double_p),
        ("seed", ctypes.c_uint64),
        ("threads", ctypes.c_int),
        ("bonferroni", ctypes.c_int),
        ("crt_draw", ctypes.c_int),
        ("fit_check", ctypes.c_int),
        ("confirm_draws", _size),
    ]


class _ScanOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", _size),
        ("model", _FitOptions),
        ("null_model", ctypes.c_int),
        ("calibration", ctypes.c_int),
        ("side", ctypes.c_int),
        ("approximate", ctypes.c_int),
        ("effect_fits", ctypes.c_int),
        ("budgets", _size_p),
        ("n_budgets", _size),
        ("screen", ctypes.c_double),
        ("strata", _size_p),
        ("alpha", ctypes.c_double),
        ("min_nonzero", _size),
        ("bonferroni", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("threads", ctypes.c_int),
        ("crt_draw", ctypes.c_int),
        ("fit_check", ctypes.c_int),
        ("confirm_draws", _size),
    ]


# Names used by the CLI configuration, mapped to the C enumerations.
_FAMILIES = {"gaussian": 0, "poisson": 1, "binomial": 2, "nb": 3, "negative_binomial": 3}
_DISPERSIONS = {"ml": 0, "moment": 1, "fixed": 2}
_SIZE_FACTORS = {"ratio": 0, "poscounts": 1}
_TRENDS = {"parametric": 0, "local": 1, "mean": 2}
_SHRINKAGE = {"none": 0, "normal": 1}
_PAIR_KINDS = {"discovery": 0, "negative": 1, "positive": 2}
_INTEGRATION = {"union": 0, "singleton": 1, "bonferroni": 2}
_SIDES = {"both": 0, "left": 1, "right": 2}
_NULL_MODELS = {"sceptre": 0, "direct": 1}
_EFFECT_FITS = {"significant": 0, "all": 1, "none": 2}
_CALIBRATIONS = {"asymptotic": 0, "permutation": 1, "crt": 2}
_CRT_DRAWS = {"thinned": 0, "direct": 1, "hazard": 2}
_FIT_CHECKS = {"strict": 0, "ks": 1}
_APPROXIMATIONS = {"skew_normal": 1, "no_approximation": 0}
_ADJUSTMENTS = {"BH": 0, "bonferroni": 1}


def _choice(name, value, choices):
    try:
        return choices[value]
    except KeyError:
        raise ValueError(f"{name} must be one of {'/'.join(choices)}, not {value!r}") from None


def _load():
    candidates = []
    if os.environ.get("RAPIDGLM_C_LIBRARY"):
        candidates.append(os.environ["RAPIDGLM_C_LIBRARY"])
    here = pathlib.Path(__file__).resolve().parent
    candidates += [str(p) for p in sorted(here.glob("librapidglm_c.*"))]
    found = ctypes.util.find_library("rapidglm_c")
    if found:
        candidates.append(found)
    errors = []
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    raise OSError(
        "librapidglm_c not found; build it with CMake and set RAPIDGLM_C_LIBRARY"
        + ("".join(f"\n  {e}" for e in errors))
    )


_lib = None


def _library():
    global _lib
    if _lib is not None:
        return _lib
    lib = _load()
    table_p = ctypes.c_void_p
    signatures = {
        "rapidglm_version": (ctypes.c_char_p, []),
        "rapidglm_c_api_version": (ctypes.c_int, []),
        "rapidglm_last_error": (ctypes.c_char_p, []),
        "rapidglm_table_rows": (_size, [table_p]),
        "rapidglm_table_columns": (_size, [table_p]),
        "rapidglm_table_column_name": (ctypes.c_char_p, [table_p, _size]),
        "rapidglm_table_column_type": (ctypes.c_int, [table_p, _size]),
        "rapidglm_table_numeric": (_double_p, [table_p, ctypes.c_char_p]),
        "rapidglm_table_text": (ctypes.c_char_p, [table_p, ctypes.c_char_p, _size]),
        "rapidglm_table_scalars": (_size, [table_p]),
        "rapidglm_table_scalar_name": (ctypes.c_char_p, [table_p, _size]),
        "rapidglm_table_scalar": (ctypes.c_double, [table_p, ctypes.c_char_p]),
        "rapidglm_table_vectors": (_size, [table_p]),
        "rapidglm_table_vector_name": (ctypes.c_char_p, [table_p, _size]),
        "rapidglm_table_vector": (_double_p, [table_p, ctypes.c_char_p, _size_p]),
        "rapidglm_table_free": (None, [table_p]),
        "rapidglm_fit_options_init": (None, [ctypes.POINTER(_FitOptions)]),
        "rapidglm_bulk_options_init": (None, [ctypes.POINTER(_BulkOptions)]),
        "rapidglm_sceptre_options_init": (None, [ctypes.POINTER(_SceptreOptions), ctypes.c_int]),
        "rapidglm_scan_options_init": (None, [ctypes.POINTER(_ScanOptions)]),
        "rapidglm_fit_batch": (
            ctypes.c_int,
            [
                ctypes.POINTER(_Dense),
                ctypes.POINTER(_Sparse),
                ctypes.POINTER(_FitOptions),
                ctypes.c_int,
                ctypes.POINTER(table_p),
            ],
        ),
        "rapidglm_bulk_de": (
            ctypes.c_int,
            [
                ctypes.POINTER(_Dense),
                ctypes.POINTER(_Sparse),
                ctypes.POINTER(_BulkOptions),
                ctypes.POINTER(table_p),
            ],
        ),
        "rapidglm_bulk_adjust": (
            ctypes.c_int,
            [_double_p, _double_p, _size, ctypes.c_double, ctypes.c_int, _double_p, _double_p],
        ),
        "rapidglm_nb_fixed_fits": (
            ctypes.c_int,
            [ctypes.POINTER(_Dense), ctypes.POINTER(_Dense), _double_p, _double_p, ctypes.c_double, _size,
             ctypes.c_double, ctypes.c_double, ctypes.c_int, _double_p, _double_p, _double_p, _int_p],
        ),
        "rapidglm_nb_dispersions": (
            ctypes.c_int,
            [ctypes.POINTER(_Dense), ctypes.POINTER(_Dense), ctypes.POINTER(_Dense), _double_p,
             ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             _double_p, _int_p],
        ),
        "rapidglm_sceptre": (
            ctypes.c_int,
            [
                ctypes.POINTER(_Dense),
                ctypes.POINTER(_Sparse),
                ctypes.POINTER(_Sparse),
                _size,
                _text_p,
                _text_p,
                _int_p,
                _size,
                _size_p,
                _text_p,
                _int_p,
                ctypes.POINTER(_SceptreOptions),
                ctypes.POINTER(table_p),
            ],
        ),
        "rapidglm_scan": (
            ctypes.c_int,
            [
                ctypes.POINTER(_Dense),
                ctypes.POINTER(_Sparse),
                ctypes.POINTER(_Sparse),
                _size,
                _size_p,
                _size_p,
                _text_p,
                ctypes.POINTER(_ScanOptions),
                ctypes.POINTER(table_p),
            ],
        ),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    if lib.rapidglm_c_api_version() != 1:
        raise OSError(f"librapidglm_c has C API version {lib.rapidglm_c_api_version()}, need 1")
    _lib = lib
    return lib


def library_version():
    """Version of the loaded librapidglm_c."""
    return _library().rapidglm_version().decode()


# Inputs. Each helper returns the C view plus the arrays it points into, which
# must stay alive until the call returns.


def _doubles(values, n=None, name="values"):
    if values is None:
        return None, None
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
    if n is not None and a.size != n:
        raise ValueError(f"{name} needs {n} values, got {a.size}")
    return a.ctypes.data_as(_double_p), a


def _sizes(values, n=None, name="values"):
    if values is None:
        return None, None
    a = np.asarray(values)
    if a.size and (a.min() < 0 or not np.all(np.equal(np.floor(a), a))):
        raise ValueError(f"{name} must be nonnegative integers")
    a = np.ascontiguousarray(a.astype(np.uintp).ravel())
    if n is not None and a.size != n:
        raise ValueError(f"{name} needs {n} values, got {a.size}")
    return a.ctypes.data_as(_size_p), a


def _texts(values):
    encoded = [str(v).encode() for v in values]
    array = (ctypes.c_char_p * len(encoded))(*encoded)
    return array, encoded


def _dense(x, name="design"):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return _Dense(a.shape[0], a.shape[1], a.ctypes.data_as(_double_p)), a


def _sparse(m, name):
    import scipy.sparse as sp

    if isinstance(m, pd.DataFrame):
        m = m.to_numpy()
    c = sp.csc_matrix(m, dtype=np.float64)
    c.sum_duplicates()
    c.sort_indices()
    ptr = np.ascontiguousarray(c.indptr.astype(np.uintp))
    index = np.ascontiguousarray(c.indices.astype(np.uintp))
    data = np.ascontiguousarray(c.data)
    view = _Sparse(
        c.shape[0],
        c.shape[1],
        ptr.ctypes.data_as(_size_p),
        index.ctypes.data_as(_size_p),
        data.ctypes.data_as(_double_p),
    )
    return view, (ptr, index, data)


def _feature_names(m):
    return list(m.columns) if isinstance(m, pd.DataFrame) else None


def _fit_options(o, n, family, dispersion, theta, tolerance, ridge, max_iterations, offset,
                 weights, robust, cox_reid, singular_ok):
    keep = []
    o.family = _choice("family", family, _FAMILIES)
    o.dispersion = _choice("dispersion", dispersion, _DISPERSIONS)
    for field, value in (("theta", theta), ("tolerance", tolerance), ("ridge", ridge)):
        if value is not None:
            setattr(o, field, float(value))
    if max_iterations is not None:
        o.max_iterations = int(max_iterations)
    o.offset, a = _doubles(offset, n, "offset")
    keep.append(a)
    o.weights, a = _doubles(weights, n, "weights")
    keep.append(a)
    o.robust, o.cox_reid, o.singular_ok = int(robust), int(cox_reid), int(singular_ok)
    return keep


def _call(function, *args):
    table = ctypes.c_void_p()
    status = function(*args, ctypes.byref(table))
    if status != 0:
        message = _library().rapidglm_last_error().decode()
        raise (ValueError if status == 1 else RuntimeError)(message)
    try:
        return _frame(table)
    finally:
        _library().rapidglm_table_free(table)


def _frame(table):
    lib = _library()
    rows = lib.rapidglm_table_rows(table)
    data = {}
    for k in range(lib.rapidglm_table_columns(table)):
        name = lib.rapidglm_table_column_name(table, k)
        if lib.rapidglm_table_column_type(table, k) == 0:
            pointer = lib.rapidglm_table_numeric(table, name)
            data[name.decode()] = np.ctypeslib.as_array(pointer, (rows,)).copy() if rows else np.empty(0)
        else:
            data[name.decode()] = [lib.rapidglm_table_text(table, name, i).decode() for i in range(rows)]
    frame = pd.DataFrame(data)
    for k in range(lib.rapidglm_table_scalars(table)):
        name = lib.rapidglm_table_scalar_name(table, k)
        frame.attrs[name.decode()] = lib.rapidglm_table_scalar(table, name)
    for k in range(lib.rapidglm_table_vectors(table)):
        name = lib.rapidglm_table_vector_name(table, k)
        length = _size()
        pointer = lib.rapidglm_table_vector(table, name, ctypes.byref(length))
        frame.attrs[name.decode()] = (
            np.ctypeslib.as_array(pointer, (length.value,)).copy() if length.value else np.empty(0)
        )
    return frame


def _index(frame, column, names):
    """Use feature names (from DataFrame inputs) for an index column."""
    if names is not None and len(frame):
        positions = frame[column].to_numpy().astype(np.int64)
        frame.insert(frame.columns.get_loc(column) + 1, column + "_name", [names[i] for i in positions])
    return frame


def fit_batch(design, responses, family="nb", dispersion="ml", theta=None, tolerance=None,
              ridge=None, max_iterations=None, offset=None, weights=None, robust=False,
              cox_reid=False, singular_ok=False, threads=1):
    """Fit one GLM per response column. Columns: feature, status, theta,
    theta_boundary, log_likelihood, iterations and, per design column j,
    beta<j>, se<j>, wald_p<j>, boundary<j>."""
    lib = _library()
    x, xa = _dense(design)
    y, ya = _sparse(responses, "responses")
    o = _FitOptions()
    lib.rapidglm_fit_options_init(ctypes.byref(o))
    keep = _fit_options(o, x.rows, family, dispersion, theta, tolerance, ridge, max_iterations,
                        offset, weights, robust, cox_reid, singular_ok)
    frame = _call(lib.rapidglm_fit_batch, ctypes.byref(x), ctypes.byref(y), ctypes.byref(o),
                  int(threads))
    del xa, ya, keep
    return _index(frame, "feature", _feature_names(responses))


def bulk_de(design, counts, coefficient=None, contrast=None, coefficients=None, test="wald",
            size_factors="ratio", size_factor_values=None, trend="parametric", min_mean=None,
            keep_minimum_start=None, outlier_sd=None, cooks_filter=None, cooks_cutoff=None,
            min_replicates_for_replace=None, independent_filtering=None, alpha=None,
            lfc_shrink="none", min_dispersion=None, max_dispersion=None, legacy_cooks=False, seed=None,
            threads=1):
    """Bulk differential expression by the DESeq2 procedure; counts are
    samples x genes. Tests the last design column unless ``coefficient``
    (a column index), ``contrast`` (one weight per column) or ``coefficients``
    (several columns tested jointly by LRT) is given.

    Columns: gene, status, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
    dispGeneEst, dispFit, dispMAP, dispersion, dispOutlier, maxCooks, replaced,
    log2FoldChangeShrunk, lfcSEShrunk. ``attrs`` holds size_factors,
    prior_variance and the other run-level values.

    ``legacy_cooks=True`` computes Cook's distances with DESeq2's robust
    dispersion as implemented in PyDESeq2 (MIT; Muzellec et al. 2023), which
    reproduces DESeq2's outlier flags; the default is RapidGLM's estimator."""
    lib = _library()
    x, xa = _dense(design)
    y, ya = _sparse(counts, "counts")
    o = _BulkOptions()
    lib.rapidglm_bulk_options_init(ctypes.byref(o))
    keep = []
    given = [v is not None for v in (coefficient, contrast, coefficients)]
    if sum(given) > 1:
        raise ValueError("give one of coefficient, contrast and coefficients")
    if coefficient is not None:
        o.coefficient = int(coefficient)
    rows = None
    if contrast is not None:
        rows = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    if coefficients is not None:
        rows = np.zeros((len(coefficients), x.cols))
        for r, j in enumerate(coefficients):
            rows[r, int(j)] = 1
    if rows is not None:
        if rows.shape[1] != x.cols:
            raise ValueError(f"contrasts need {x.cols} weights per row")
        o.contrasts, a = _doubles(rows)
        o.contrast_rows = rows.shape[0]
        keep.append(a)
    o.lrt = int(_choice("test", test, {"wald": 0, "lrt": 1}))
    o.size_factor_method = _choice("size_factors", size_factors, _SIZE_FACTORS)
    o.size_factors, a = _doubles(size_factor_values, x.rows, "size_factor_values")
    keep.append(a)
    o.trend = _choice("trend", trend, _TRENDS)
    for field, value in (("min_mean", min_mean), ("min_dispersion", min_dispersion),
                         ("max_dispersion", max_dispersion), ("outlier_sd", outlier_sd),
                         ("cooks_cutoff", cooks_cutoff), ("alpha", alpha)):
        if value is not None:
            setattr(o, field, float(value))
    o.legacy_cooks = int(bool(legacy_cooks))
    for field, value in (("keep_minimum_start", keep_minimum_start), ("cooks_filter", cooks_filter),
                         ("independent_filtering", independent_filtering)):
        if value is not None:
            setattr(o, field, int(bool(value)))
    if min_replicates_for_replace is not None:
        o.min_replicates_for_replace = int(min_replicates_for_replace)
    o.lfc_shrinkage = _choice("lfc_shrink", lfc_shrink, _SHRINKAGE)
    if seed is not None:
        o.seed = int(seed)
    o.threads = int(threads)
    frame = _call(lib.rapidglm_bulk_de, ctypes.byref(x), ctypes.byref(y), ctypes.byref(o))
    del xa, ya, keep
    names = _feature_names(counts)
    if names is not None:
        frame.index = pd.Index(names, name="gene_name")
    return frame


def bulk_adjust(pvalue, base_mean, alpha=0.1, independent_filtering=True):
    """Adjusted p-values of a bulk_de table at another alpha: Benjamini-Hochberg
    after the same independent filtering on baseMean. Returns (padj, threshold)."""
    lib = _library()
    p, pa = _doubles(pvalue)
    n = pa.size
    b, ba = _doubles(base_mean, n, "base_mean")
    padj = np.empty(n)
    threshold = ctypes.c_double()
    status = lib.rapidglm_bulk_adjust(p, b, n, float(alpha), int(bool(independent_filtering)),
                                      padj.ctypes.data_as(_double_p), ctypes.byref(threshold))
    if status != 0:
        raise ValueError(lib.rapidglm_last_error().decode())
    del pa, ba
    return padj, threshold.value


def nb_fixed_fits(design, counts, size_factors, dispersions, tolerance=1e-8, max_iterations=1000,
                  min_beta=-30.0, max_beta=30.0, threads=1):
    """NB GLM fits at a fixed dispersion per feature with log size-factor offsets.
    counts: observations x features (dense). Returns (coefficients [features x
    columns, natural log], fitted [observations x features], hat values
    [observations x features], converged [features])."""
    lib = _library()
    x, xa = _dense(design)
    y, ya = _dense(counts, "counts")
    n, genes, p = x.rows, y.cols, x.cols
    sf, sfa = _doubles(size_factors, n, "size_factors")
    d, da = _doubles(dispersions, genes, "dispersions")
    beta, mu, hat = np.empty((genes, p)), np.empty((n, genes)), np.empty((n, genes))
    converged = np.empty(genes, dtype=np.intc)
    status = lib.rapidglm_nb_fixed_fits(
        ctypes.byref(x), ctypes.byref(y), sf, d, float(tolerance), int(max_iterations), float(min_beta),
        float(max_beta), int(threads), beta.ctypes.data_as(_double_p), mu.ctypes.data_as(_double_p),
        hat.ctypes.data_as(_double_p), converged.ctypes.data_as(_int_p))
    if status != 0:
        raise ValueError(lib.rapidglm_last_error().decode())
    del xa, ya, sfa, da
    return beta, mu, hat, converged.astype(bool)


def nb_dispersions(design, counts, mu, start, min_dispersion=1e-8, max_dispersion=10.0,
                   prior_variance=0.0, cox_reid=True, keep_minimum_start=True, threads=1):
    """Dispersions at fixed means from `start`: Cox-Reid adjusted NB likelihood,
    plus a normal prior on log dispersion centred at log(start) when
    prior_variance > 0. Returns (dispersions, converged)."""
    lib = _library()
    x, xa = _dense(design)
    y, ya = _dense(counts, "counts")
    m, ma = _dense(mu, "mu")
    s, sa = _doubles(start, y.cols, "start")
    out = np.empty(y.cols)
    converged = np.empty(y.cols, dtype=np.intc)
    status = lib.rapidglm_nb_dispersions(
        ctypes.byref(x), ctypes.byref(y), ctypes.byref(m), s, float(min_dispersion), float(max_dispersion),
        float(prior_variance or 0.0), int(bool(cox_reid)), int(bool(keep_minimum_start)), int(threads),
        out.ctypes.data_as(_double_p),
        converged.ctypes.data_as(_int_p))
    if status != 0:
        raise ValueError(lib.rapidglm_last_error().decode())
    del xa, ya, ma, sa
    return out, converged.astype(bool)


def _budgets(o, budgets):
    if budgets is None:
        return None
    o.budgets, a = _sizes(budgets, name="budgets")
    o.n_budgets = a.size
    return a


def sceptre(design, responses, assignments, guides, pairs, moi="low", control_group=None,
            resampling_mechanism=None, integration="union", side="both",
            approximation="skew_normal", budgets=None, screen=None, null_model="sceptre",
            effect_fits="significant", crt_draw="thinned", fit_check="strict",
            confirm_draws=None, cell_qc=True, assignment_qc=True,
            min_nonzero=None, alpha=None, qc_lower=None, qc_upper=None, max_mito=None,
            qc_total=None, qc_detected=None, qc_mito=None, adjustment="BH", family="nb",
            dispersion="ml", theta=None, tolerance=None, max_iterations=None, offset=None,
            seed=1, threads=1):
    """Perturbation screen analysis by the SCEPTRE method.

    responses: cells x genes; assignments: cells x guides (binary calls).
    guides: DataFrame (or records) with id, target and non_targeting, one row
    per assignment column. pairs: DataFrame with response (column index or
    name), target and kind (discovery/negative/positive).

    fit_check: acceptance of the skew-normal fit. "strict" (default) also
    checks the tail where the statistic lies and confirms small p-values on
    confirm_draws draws (default: ten times the last budget, 199,999 for
    the default budgets; 0 turns the confirmation off);
    "ks" is the check of releases before 1.0.0.

    Columns: response, target, guide, kind, status, cells, nonzero_treatment,
    nonzero_control, draws, approximation, statistic, p_value, adjusted_p,
    log2fc, log2fc_se, sceptre_mean_ratio, mean_ratio_log2. ``attrs`` holds
    retained_cells and the rejection counts."""
    lib = _library()
    high = _choice("moi", moi, {"low": 0, "high": 1})
    x, xa = _dense(design)
    y, ya = _sparse(responses, "responses")
    g, ga = _sparse(assignments, "assignments")
    guides = pd.DataFrame(guides)
    pairs = pd.DataFrame(pairs)
    for table, columns in ((guides, ("id", "target", "non_targeting")), (pairs, ("response", "target", "kind"))):
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
    if len(guides) != g.cols:
        raise ValueError(f"{len(guides)} guides for {g.cols} assignment columns")
    names = _feature_names(responses)
    response = pairs["response"]
    if names is not None and not pd.api.types.is_integer_dtype(response):
        position = {n: i for i, n in enumerate(names)}
        response = [position[r] for r in response]
    ids, ids_keep = _texts(guides["id"])
    targets, targets_keep = _texts(guides["target"])
    nt = np.ascontiguousarray(np.asarray(guides["non_targeting"]).astype(np.intc))
    pair_responses, pr = _sizes(response, name="pair responses")
    pair_targets, pt_keep = _texts(pairs["target"])
    kinds = np.ascontiguousarray([_choice("kind", k, _PAIR_KINDS) for k in pairs["kind"]], dtype=np.intc)
    o = _SceptreOptions()
    lib.rapidglm_sceptre_options_init(ctypes.byref(o), high)
    keep = _fit_options(o.model, x.rows, family, dispersion, theta, tolerance, None, max_iterations,
                        offset, None, False, False, False)
    o.null_model = _choice("null_model", null_model, _NULL_MODELS)
    if control_group is not None:
        o.nt_controls = _choice("control_group", control_group, {"complement": 0, "nt_cells": 1})
    if resampling_mechanism is not None:
        o.crt = _choice("resampling_mechanism", resampling_mechanism, {"permutations": 0, "crt": 1})
    o.integration = _choice("integration", integration, _INTEGRATION)
    o.side = _choice("side", side, _SIDES)
    o.approximate = _choice("approximation", approximation, _APPROXIMATIONS)
    keep.append(_budgets(o, budgets))
    if screen is not None:
        o.screen = float(screen)
    o.effect_fits = _choice("effect_fits", effect_fits, _EFFECT_FITS)
    o.crt_draw = _choice("crt_draw", crt_draw, _CRT_DRAWS)
    o.fit_check = _choice("fit_check", fit_check, _FIT_CHECKS)
    if confirm_draws is not None:
        o.confirm_draws = int(confirm_draws)
    o.cell_qc, o.assignment_qc = int(bool(cell_qc)), int(bool(assignment_qc))
    for field, value in (("alpha", alpha), ("lower_quantile", qc_lower),
                         ("upper_quantile", qc_upper), ("max_mito", max_mito)):
        if value is not None:
            setattr(o, field, float(value))
    if min_nonzero is not None:
        o.min_nonzero = int(min_nonzero)
    for field, values in (("qc_total", qc_total), ("qc_detected", qc_detected), ("qc_mito", qc_mito)):
        pointer, a = _doubles(values, x.rows, field)
        setattr(o, field, pointer)
        keep.append(a)
    o.bonferroni = _choice("adjustment", adjustment, _ADJUSTMENTS)
    o.seed = int(seed)
    o.threads = int(threads)
    frame = _call(
        lib.rapidglm_sceptre,
        ctypes.byref(x),
        ctypes.byref(y),
        ctypes.byref(g),
        g.cols,
        ids,
        targets,
        nt.ctypes.data_as(_int_p),
        len(pairs),
        pair_responses,
        pair_targets,
        kinds.ctypes.data_as(_int_p),
        ctypes.byref(o),
    )
    del xa, ya, ga, ids_keep, targets_keep, pr, pt_keep, keep
    frame["kind"] = [list(_PAIR_KINDS)[int(k)] for k in frame["kind"]]
    return _index(frame, "response", names)


def scan(design, responses, predictors, pairs=None, family="nb", dispersion="ml", theta=None,
         tolerance=None, ridge=None, max_iterations=None, offset=None, weights=None,
         null_model="direct", calibration="permutation", side="both", approximation="skew_normal",
         budgets=None, screen=None, effect_fits="significant", crt_draw="thinned",
         fit_check="strict", confirm_draws=None, strata=None,
         alpha=None, min_nonzero=None, adjustment="BH", seed=1, threads=1):
    """Score scan: each response's null model is fitted once and tested against
    many predictors. pairs: None (every response x predictor) or a DataFrame
    with response, predictor and optional family columns.

    Columns: response, predictor, family, status, statistic, p_value,
    adjusted_p, estimate, standard_error, effect, effect_se, draws,
    approximation."""
    lib = _library()
    x, xa = _dense(design)
    y, ya = _sparse(responses, "responses")
    g, ga = _sparse(predictors, "predictors")
    n_pairs, pr, pp, pf = 0, (None, None), (None, None), (None, None)
    if pairs is not None:
        pairs = pd.DataFrame(pairs)
        n_pairs = len(pairs)
        pr = _sizes(pairs["response"], name="pair responses")
        pp = _sizes(pairs["predictor"], name="pair predictors")
        if "family" in pairs:
            pf = _texts(pairs["family"])
    o = _ScanOptions()
    lib.rapidglm_scan_options_init(ctypes.byref(o))
    keep = _fit_options(o.model, x.rows, family, dispersion, theta, tolerance, ridge, max_iterations,
                        offset, weights, False, False, False)
    o.null_model = _choice("null_model", null_model, _NULL_MODELS)
    o.calibration = _choice("calibration", calibration, _CALIBRATIONS)
    o.side = _choice("side", side, _SIDES)
    o.approximate = _choice("approximation", approximation, _APPROXIMATIONS)
    keep.append(_budgets(o, budgets))
    if screen is not None:
        o.screen = float(screen)
    o.effect_fits = _choice("effect_fits", effect_fits, _EFFECT_FITS)
    o.crt_draw = _choice("crt_draw", crt_draw, _CRT_DRAWS)
    o.fit_check = _choice("fit_check", fit_check, _FIT_CHECKS)
    if confirm_draws is not None:
        o.confirm_draws = int(confirm_draws)
    o.strata, a = _sizes(strata, x.rows, "strata")
    keep.append(a)
    if alpha is not None:
        o.alpha = float(alpha)
    if min_nonzero is not None:
        o.min_nonzero = int(min_nonzero)
    o.bonferroni = _choice("adjustment", adjustment, _ADJUSTMENTS)
    o.seed = int(seed)
    o.threads = int(threads)
    frame = _call(lib.rapidglm_scan, ctypes.byref(x), ctypes.byref(y), ctypes.byref(g), n_pairs,
                  pr[0], pp[0], pf[0], ctypes.byref(o))
    del xa, ya, ga, pr, pp, pf, keep
    return _index(frame, "response", _feature_names(responses))
