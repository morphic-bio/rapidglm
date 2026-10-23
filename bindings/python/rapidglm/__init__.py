"""RapidGLM: fast, numerically stable GLM fitting.

``bulk_de``, ``sceptre``, ``sceptre_grid``, ``scan`` and ``fit_batch`` run the
applications and the batch engine through the C interface (``rapidglm.capi``,
needs librapidglm_c); ``rapidglm.deseq`` wraps bulk_de in PyDESeq2's workflow
(DeseqDataSet, DeseqStats). ``rapidglm.fit`` fits one model through the compiled
extension; ``rapidglm.statsmodels.GLM`` is a drop-in replacement for
``statsmodels.api.GLM`` that returns statsmodels results objects.
"""

import numpy as np

from .capi import bulk_de, edger_ql, fit_batch, library_version, scan, sceptre, sceptre_grid

__all__ = ["bulk_de", "edger_ql", "sceptre", "sceptre_grid", "scan", "fit_batch", "fit", "library_version"]
__version__ = "1.4.1"


def fit(
    exog,
    endog,
    family="gaussian",
    weights=None,
    offset=None,
    start=None,
    theta=10.0,
    estimate_theta=None,
    tolerance=1e-9,
    max_iterations=100,
    singular_ok=True,
    alias_tolerance=1e-11,
):
    """Fit one GLM. family is gaussian, poisson, binomial or nb (log link for
    counts, logit for binomial). For nb, theta is estimated by maximum
    likelihood unless estimate_theta is False (then theta is fixed).
    Returns a dict of coefficients, standard errors, covariance and diagnostics."""
    x = np.ascontiguousarray(exog, dtype=float)
    y = np.asarray(endog, dtype=float)
    n = y.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    s = np.empty(0) if start is None else np.asarray(start, dtype=float)
    if estimate_theta is None:
        estimate_theta = family == "nb"
    from . import _rapidglm

    return _rapidglm.fit(
        x,
        y,
        w,
        off,
        s,
        family,
        float(theta),
        bool(estimate_theta),
        float(tolerance),
        int(max_iterations),
        bool(singular_ok),
        float(alias_tolerance),
    )
