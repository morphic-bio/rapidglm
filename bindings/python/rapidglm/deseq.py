"""A PyDESeq2-style interface to RapidGLM's bulk differential expression.

``DeseqDataSet`` and ``DeseqStats`` follow the workflow and argument names of
PyDESeq2's published API (designed from its documentation; no code is shared)
and run RapidGLM's DESeq2 procedure (``rapidglm.bulk_de``)::

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~batch + condition")
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()
    ds.results_df          # baseMean, log2FoldChange, lfcSE, stat, pvalue, padj

Differences from PyDESeq2:
- ``lfc_shrink`` uses DESeq2's normal prior (``lfcShrink(type="normal")``), not
  apeGLM, and shrinks a single design column;
- ``varm["LFC"]`` is not provided (only the tested coefficient or contrast is
  estimated); per-gene values are in ``var`` and size factors in ``obs``, as
  in PyDESeq2 0.5; ``vst``, plotting, ``lfc_null`` and ``alt_hypothesis`` are not
  implemented and raise ``NotImplementedError``;
- design formulas support factor and numeric main effects and interactions
  between terms that are also present as main effects (patsy-style column
  names such as ``condition[T.B]``).
"""

import re

import numpy as np
import pandas as pd

from .capi import bulk_adjust, bulk_de

__all__ = ["DeseqDataSet", "DeseqStats", "design_matrix"]


# Design formulas.


def _terms(formula):
    text = formula.strip()
    if not text.startswith("~"):
        raise ValueError(f"design formula must start with '~': {formula!r}")
    text = text[1:].replace(" ", "")
    intercept = True
    terms = []
    for sign, part in re.findall(r"([+-]?)([^+-]+)", text):
        if part in ("0", "1"):
            if sign == "-" and part == "1" or part == "0":
                intercept = False
            continue
        if sign == "-":
            raise ValueError(f"only '-1' may be removed from a design formula: {formula!r}")
        factors = [f for f in re.split(r"[*:]", part)]
        if "*" in part:
            # a*b*c expands to every product of its factors.
            for size in range(1, len(factors) + 1):
                terms += _combinations(factors, size)
        else:
            terms.append(tuple(factors))
    unique = []
    for t in terms:
        if t not in unique:
            unique.append(t)
    return intercept, unique


def _ordered(terms, metadata):
    """Terms in patsy's column order: grouped by their numeric variables (terms
    of factors only first, then groups in order of appearance), lower orders
    first within a group."""
    groups = []
    for t in terms:
        numeric = frozenset(n for n in t if not _categorical(metadata[n]))
        if numeric not in groups:
            groups.append(numeric)
    groups.sort(key=len)  # stable: the factor-only group first
    rank = {g: i for i, g in enumerate(groups)}
    return sorted(
        terms,
        key=lambda t: (rank[frozenset(n for n in t if not _categorical(metadata[n]))], len(t)),
    )


def _combinations(items, size):
    if size == 0:
        return [()]
    if len(items) < size:
        return []
    first, rest = items[0], items[1:]
    return [(first,) + c for c in _combinations(rest, size - 1)] + _combinations(rest, size)


def _levels(column):
    if isinstance(column.dtype, pd.CategoricalDtype):
        return list(column.cat.categories)
    return sorted(column.unique())


def _categorical(column):
    return not pd.api.types.is_numeric_dtype(column) or isinstance(column.dtype, pd.CategoricalDtype)


def design_matrix(formula, metadata):
    """Design matrix for a formula such as ``"~batch + condition"``.

    Factors (non-numeric or categorical columns) are treatment coded against
    their first level (sorted, or the category order); numeric columns enter
    as they are. Without an intercept the first factor is coded with one
    column per level. Column names follow patsy: ``Intercept``,
    ``condition[T.B]``, ``condition[T.B]:batch[T.b2]``."""
    intercept, terms = _terms(formula)
    metadata = pd.DataFrame(metadata)
    for term in terms:
        for name in term:
            if name not in metadata.columns:
                raise ValueError(f"design variable {name!r} is not a metadata column")
    terms = _ordered(terms, metadata)
    for term in terms:
        if len(term) > 1:
            for name in term:
                if (name,) not in terms:
                    raise NotImplementedError(
                        f"interaction {':'.join(term)} needs its main effect {name}"
                    )
    columns = {}
    if intercept:
        columns["Intercept"] = np.ones(len(metadata))
    full_first = not intercept
    coded = {}
    for name in dict.fromkeys(n for t in terms for n in t):
        column = metadata[name]
        if _categorical(column):
            if column.isna().any():
                raise ValueError(f"design variable {name!r} has missing values")
            levels = _levels(column)
            full = full_first and (name,) == next(t for t in terms if len(t) == 1 and _categorical(metadata[t[0]]))
            kept = levels if full else levels[1:]
            label = "{}[{}]" if full else "{}[T.{}]"
            coded[name] = [(label.format(name, lv), (column == lv).to_numpy(float)) for lv in kept]
        else:
            coded[name] = [(name, column.to_numpy(float))]
    for term in terms:
        pieces = [[("", np.ones(len(metadata)))]]
        for name in term:
            pieces.append(coded[name])
        product = pieces[0]
        for part in pieces[1:]:
            product = [
                ((a + ":" if a else "") + b, u * v) for a, u in product for b, v in part
            ]
        for label, values in product:
            columns[label] = values
    return pd.DataFrame(columns, index=metadata.index)


# The dataset and the tests.


class DeseqDataSet:
    """Counts (samples x genes), sample metadata and a design, fitted by the
    DESeq2 procedure. Arguments follow PyDESeq2's DeseqDataSet."""

    def __init__(
        self,
        adata=None,
        counts=None,
        metadata=None,
        design="~condition",
        fit_type="parametric",
        size_factors_fit_type="ratio",
        control_genes=None,
        min_mu=0.5,
        min_disp=1e-8,
        max_disp=10.0,
        refit_cooks=True,
        min_replicates=7,
        beta_tol=1e-8,
        n_cpus=None,
        inference=None,
        quiet=False,
        low_memory=False,
        legacy_cooks=False,
    ):
        if adata is not None:
            if counts is not None or metadata is not None:
                raise ValueError("give adata or counts and metadata, not both")
            x = adata.X
            x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
            counts = pd.DataFrame(x, index=list(adata.obs_names), columns=list(adata.var_names))
            metadata = adata.obs
        if counts is None or metadata is None:
            raise ValueError("counts and metadata are required")
        for name, value, default in (
            ("control_genes", control_genes, None),
            ("inference", inference, None),
            ("beta_tol", beta_tol, 1e-8),
        ):
            if value != default:
                raise NotImplementedError(f"{name} is not supported by RapidGLM's DESeq2 procedure")
        if size_factors_fit_type not in ("ratio", "poscounts"):
            raise NotImplementedError("size_factors_fit_type must be ratio or poscounts")
        if fit_type not in ("parametric", "mean", "local"):
            raise ValueError("fit_type must be parametric, mean or local")
        counts = counts if isinstance(counts, pd.DataFrame) else pd.DataFrame(counts)
        metadata = pd.DataFrame(metadata)
        if len(metadata) != len(counts):
            raise ValueError("metadata needs one row per sample (row of counts)")
        if not metadata.index.equals(counts.index):
            metadata = metadata.reindex(counts.index)
            if metadata.isna().all(axis=1).any():
                raise ValueError("metadata index does not match the samples in counts")
        values = counts.to_numpy(dtype=float)
        if (values < 0).any() or not np.array_equal(values, np.round(values)):
            raise ValueError("counts must be nonnegative integers")
        self.counts = counts
        self.obs = metadata
        self.obs_names = counts.index
        self.var_names = counts.columns
        self.design = design
        if isinstance(design, pd.DataFrame):
            x = design if design.index.equals(counts.index) else design.reindex(counts.index)
        else:
            x = design_matrix(design, metadata)
        # Layout as PyDESeq2 0.5 keeps it: per-sample values in obs, per-gene in var.
        self.obs = self.obs.copy()
        self.var = pd.DataFrame(index=self.var_names)
        self.obsm = {"design_matrix": x.astype(float)}
        self.varm = {}
        self.uns = {}
        self.fit_type = fit_type
        self.size_factors_fit_type = size_factors_fit_type
        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.refit_cooks = refit_cooks
        self.min_replicates = min_replicates
        self.n_processes = n_cpus or 1
        self.legacy_cooks = legacy_cooks
        self.quiet = quiet
        self._fits = {}

    @property
    def variables(self):
        """Metadata variables in the design."""
        if isinstance(self.design, pd.DataFrame):
            return []
        return list(dict.fromkeys(n for t in _terms(self.design)[1] for n in t))

    def _options(self):
        return dict(
            trend=self.fit_type,
            size_factors=self.size_factors_fit_type,
            min_mean=self.min_mu,
            min_dispersion=self.min_disp,
            max_dispersion=0 if self.max_disp is None else self.max_disp,
            min_replicates_for_replace=self.min_replicates if self.refit_cooks else 0,
            threads=self.n_processes,
            legacy_cooks=self.legacy_cooks,
        )

    def _run(self, contrast=None, coefficient=None, alpha=0.05, cooks_filter=True,
             independent_filter=True, lfc_shrink="none"):
        """bulk_de for one test. The fit is reused across tests of the same
        hypothesis; only the adjusted p-values depend on alpha and independent
        filtering, and those are recomputed from the stored p-values."""
        x = self.obsm["design_matrix"]
        if contrast is not None:
            contrast = np.asarray(contrast, dtype=float)
            unit = np.flatnonzero(contrast)
            if len(unit) == 1 and contrast[unit[0]] == 1:
                coefficient, contrast = int(unit[0]), None
        if contrast is None and coefficient is None:
            coefficient = x.shape[1] - 1
        key = (
            None if contrast is None else tuple(contrast),
            coefficient,
            bool(cooks_filter),
            lfc_shrink,
        )
        if key not in self._fits:
            result = bulk_de(
                x.to_numpy(),
                self.counts,
                coefficient=coefficient,
                contrast=contrast,
                alpha=alpha,
                cooks_filter=cooks_filter,
                independent_filtering=independent_filter,
                lfc_shrink=lfc_shrink,
                **self._options(),
            )
            result.attrs["alpha"], result.attrs["independent_filter"] = alpha, independent_filter
            self._fits[key] = result
        result = self._fits[key]
        if (result.attrs["alpha"], result.attrs["independent_filter"]) != (alpha, independent_filter):
            result = result.copy()
            padj, threshold = bulk_adjust(result["pvalue"], result["baseMean"], alpha, independent_filter)
            result["padj"] = padj
            result.attrs.update(alpha=alpha, independent_filter=independent_filter,
                                filter_threshold=threshold)
        return result

    def deseq2(self, fit_type=None):
        """Size factors, dispersions and the fit of the full design."""
        if fit_type is not None:
            self.fit_type = fit_type
            self._fits.clear()
        result = self._run()
        self.obs["size_factors"] = result.attrs["size_factors"]
        for name, column in (
            ("genewise_dispersions", "dispGeneEst"),
            ("fitted_dispersions", "dispFit"),
            ("MAP_dispersions", "dispMAP"),
            ("dispersions", "dispersion"),
            ("_outlier_genes", "dispOutlier"),
            ("replaced", "replaced"),
        ):
            values = result[column].to_numpy()
            if name in ("_outlier_genes", "replaced"):
                values = values.astype(bool)
            self.var[name] = values
        self.var["_normed_means"] = result["baseMean"].to_numpy()
        self.uns["trend_coeffs"] = pd.Series(
            [result.attrs["asymptotic_dispersion"], result.attrs["extra_poisson"]], index=["a0", "a1"]
        )
        self.uns["_squared_logres"] = result.attrs["log_residual_variance"]
        self.uns["prior_disp_var"] = result.attrs["prior_variance"]
        if not self.quiet:
            print(f"Fitted size factors, dispersions and the design ({self.fit_type} trend).")
        return self

    def contrast_vector(self, contrast):
        """Contrast ["variable", "tested level", "reference level"], or a vector
        with one weight per design column, as a vector."""
        columns = list(self.obsm["design_matrix"].columns)
        if len(contrast) == 3 and all(isinstance(c, str) for c in contrast):
            variable, tested, reference = contrast
            if variable not in self.obs.columns:
                raise ValueError(f"{variable!r} is not a metadata column")
            levels = set(self.obs[variable].astype(str))
            for level in (tested, reference):
                if level not in levels:
                    raise ValueError(f"{level!r} is not a level of {variable!r}")
            v = np.zeros(len(columns))
            found = False
            for level, sign in ((tested, 1), (reference, -1)):
                for label in (f"{variable}[T.{level}]", f"{variable}[{level}]"):
                    if label in columns:
                        v[columns.index(label)] += sign
                        found = True
            if not found:
                raise ValueError(f"no design column codes {variable!r}")
            return v
        v = np.asarray(contrast, dtype=float)
        if v.shape != (len(columns),):
            raise ValueError(f"a contrast vector needs {len(columns)} weights")
        return v


class DeseqStats:
    """Wald tests of a contrast for a fitted DeseqDataSet. Arguments follow
    PyDESeq2's DeseqStats."""

    def __init__(
        self,
        dds,
        contrast,
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        prior_LFC_var=None,
        lfc_null=0.0,
        alt_hypothesis=None,
        inference=None,
        quiet=False,
        n_cpus=None,
    ):
        for name, value, default in (
            ("prior_LFC_var", prior_LFC_var, None),
            ("lfc_null", lfc_null, 0.0),
            ("alt_hypothesis", alt_hypothesis, None),
            ("inference", inference, None),
        ):
            if not (value is default or value == default):
                raise NotImplementedError(f"{name} is not supported by RapidGLM's DESeq2 procedure")
        self.dds = dds
        self.contrast = contrast
        self.contrast_vector = dds.contrast_vector(contrast)
        self.design_matrix = dds.obsm["design_matrix"]
        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.quiet = quiet
        if n_cpus:
            dds.n_processes = n_cpus
        self.shrunk_LFCs = False
        self.results_df = None

    def _store(self, result, shrunk):
        genes = self.dds.var_names
        lfc = result["log2FoldChangeShrunk" if shrunk else "log2FoldChange"].to_numpy()
        se = result["lfcSEShrunk" if shrunk else "lfcSE"].to_numpy()
        self.base_mean = pd.Series(result["baseMean"].to_numpy(), index=genes)
        self.statistics = pd.Series(result["stat"].to_numpy(), index=genes)
        self.p_values = pd.Series(result["pvalue"].to_numpy(), index=genes)
        self.padj = pd.Series(result["padj"].to_numpy(), index=genes)
        self.SE = pd.Series(se, index=genes)
        self.results_df = pd.DataFrame(
            {
                "baseMean": self.base_mean,
                "log2FoldChange": lfc,
                "lfcSE": se,
                "stat": self.statistics,
                "pvalue": self.p_values,
                "padj": self.padj,
            },
            index=genes,
        )
        self.shrunk_LFCs = shrunk
        if not self.quiet:
            label = self.contrast if isinstance(self.contrast[0], str) else list(self.contrast_vector)
            print(f"Log2 fold change & Wald test p-value: {label}")
            print(self.results_df)

    def summary(self, **kwargs):
        """Runs the Wald test and stores the table in ``results_df``."""
        for name in ("lfc_null", "alt_hypothesis"):
            if kwargs.get(name) not in (None, 0, 0.0):
                raise NotImplementedError(f"{name} is not supported by RapidGLM's DESeq2 procedure")
        result = self.dds._run(
            contrast=self.contrast_vector,
            alpha=self.alpha,
            cooks_filter=self.cooks_filter,
            independent_filter=self.independent_filter,
        )
        self._store(result, shrunk=False)
        return self.results_df

    def run_wald_test(self):
        """Same as summary(): the Wald test is part of the DESeq2 run."""
        return self.summary()

    def lfc_shrink(self, coeff=None, adapt=True, prior="normal"):
        """Shrinks the log2 fold changes of one design column with DESeq2's
        normal prior (lfcShrink(type="normal")); PyDESeq2 uses apeGLM. coeff is
        a design column name; by default the contrast's column when the
        contrast is a single coefficient. p-values are those of the
        unshrunken test, as in DESeq2."""
        if prior != "normal":
            raise NotImplementedError("only the normal prior is implemented")
        columns = list(self.design_matrix.columns)
        if coeff is None:
            unit = np.flatnonzero(self.contrast_vector)
            if len(unit) != 1 or self.contrast_vector[unit[0]] != 1:
                raise ValueError("give coeff: shrinkage applies to one design column")
            index = int(unit[0])
        else:
            if coeff not in columns:
                raise ValueError(f"{coeff!r} is not a design column; columns are {columns}")
            index = columns.index(coeff)
        result = self.dds._run(
            coefficient=index,
            alpha=self.alpha,
            cooks_filter=self.cooks_filter,
            independent_filter=self.independent_filter,
            lfc_shrink="normal",
        )
        self._store(result, shrunk=True)
        return self.results_df
