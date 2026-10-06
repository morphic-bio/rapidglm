#!/usr/bin/env python3
"""Black-box Python comparators for the GLM comparison suite.

Usage: python_fit.py problem_dir tool output.json max_seconds
Tools: statsmodels (GLM, IRLS defaults), statsmodels-nb (discrete NegativeBinomial,
default fit), glum (unpenalized GeneralizedLinearRegressor), sklearn (unpenalized
PoissonRegressor / LogisticRegression / LinearRegression, coefficients only).
"""

import json, sys, time, warnings
from pathlib import Path
import numpy as np


def load(d):
    meta = json.loads((d / "meta.json").read_text())
    n, p = meta["n"], meta["p"]
    X = np.fromfile(d / "X.f64", dtype="<f8").reshape(n, p)
    return (
        meta,
        X,
        *(
            np.fromfile(d / f, dtype="<f8")
            for f in ("y.f64", "offset.f64", "weights.f64")
        ),
    )


def main(d, tool, output, budget):
    meta, X, y, off, w = load(Path(d))
    family = meta["family"]
    import statsmodels.api as sm

    # Import every comparator before timing so first-call import cost is excluded.
    import statsmodels.discrete.discrete_model  # noqa: F401
    import sklearn.linear_model  # noqa: F401

    if tool.startswith("glum"):
        import glum  # noqa: F401
    if tool == "py-rapidglm":
        import rapidglm.statsmodels  # noqa: F401

    def fit_once():
        if tool == "statsmodels":
            fam = {
                "gaussian": sm.families.Gaussian(),
                "poisson": sm.families.Poisson(),
                "binomial": sm.families.Binomial(),
            }[family]
            r = sm.GLM(y, X, family=fam, offset=off, var_weights=w).fit()
            return dict(
                b=r.params,
                se=r.bse,
                ll=r.llf,
                converged=bool(r.converged),
                iterations=int(r.fit_history["iteration"]),
            )
        if tool == "py-rapidglm":
            import rapidglm.statsmodels as rsm

            fam = {
                "gaussian": sm.families.Gaussian(),
                "poisson": sm.families.Poisson(),
                "binomial": sm.families.Binomial(),
            }[family]
            r = rsm.GLM(y, X, family=fam, offset=off, var_weights=w).fit()
            return dict(
                b=r.params,
                se=r.bse,
                ll=r.llf,
                converged=bool(r.converged),
                iterations=r.fit_history["iteration"],
                method=getattr(r, "method", None),
            )
        if tool == "statsmodels-nb":
            from statsmodels.discrete.discrete_model import NegativeBinomial

            keep = (
                w > 0
            )  # the discrete model has no prior weights; zero weights are dropped
            r = NegativeBinomial(y[keep], X[keep], offset=off[keep]).fit(disp=0)
            alpha = r.params[-1]
            return dict(
                b=r.params[:-1],
                se=r.bse[:-1],
                ll=r.llf,
                theta=1 / alpha,
                converged=bool(r.mle_retvals.get("converged")),
                iterations=r.mle_retvals.get("iterations"),
                note="weights ignored except zero" if meta["has_weights"] else None,
            )
        if tool in ("glum", "glum-tight"):
            from glum import GeneralizedLinearRegressor

            fam = {"gaussian": "normal", "poisson": "poisson", "binomial": "binomial"}[
                family
            ]
            # Matched-accuracy variant: tolerances near RapidGLM's defaults.
            tight = (
                dict(gradient_tol=1e-10, step_size_tol=1e-14, max_iter=1000)
                if tool == "glum-tight"
                else {}
            )
            m = GeneralizedLinearRegressor(
                family=fam, alpha=0, fit_intercept=False, **tight
            )
            m.fit(X, y, sample_weight=w, offset=off)
            try:
                se = np.sqrt(
                    np.diag(m.covariance_matrix(X, y, sample_weight=w, offset=off))
                )
            except Exception:
                se = None
            return dict(b=m.coef_, se=se, converged=None, iterations=int(m.n_iter_))
        if tool in ("sklearn", "sklearn-tight"):
            from sklearn import linear_model

            tol = 1e-10 if tool == "sklearn-tight" else 1e-4
            if family == "poisson":
                if np.any(off != 0):
                    raise ValueError("sklearn PoissonRegressor has no offset")
                m = linear_model.PoissonRegressor(
                    alpha=0,
                    fit_intercept=False,
                    tol=tol,
                    solver="newton-cholesky",
                    max_iter=1000,
                )
            elif family == "binomial":
                m = linear_model.LogisticRegression(
                    penalty=None,
                    fit_intercept=False,
                    tol=tol,
                    solver="newton-cholesky",
                    max_iter=1000,
                )
            else:
                m = linear_model.LinearRegression(fit_intercept=False)
            m.fit(X, y, sample_weight=w)
            return dict(
                b=np.ravel(m.coef_),
                se=None,
                converged=None,
                iterations=int(np.max(getattr(m, "n_iter_", [0]))),
            )
        raise ValueError(tool)

    seconds, caught, result, error = [], set(), None, None
    while True:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            try:
                result = fit_once()
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        seconds.append(time.perf_counter() - start)
        caught |= {f"{x.category.__name__}: {str(x.message)[:200]}" for x in ws}
        if error or len(seconds) >= 7 or sum(seconds) >= budget:
            break
    out = dict(tool=tool, seconds=seconds, warnings=sorted(caught), error=error)
    if result:

        def lst(v):
            return (
                None
                if v is None
                else [float(x) if np.isfinite(x) else None for x in np.ravel(v)]
            )

        out.update(
            coefficients=lst(result["b"]),
            standard_errors=lst(result.get("se")),
            loglik=result.get("ll"),
            theta=result.get("theta"),
            converged=result.get("converged"),
            iterations=result.get("iterations"),
            note=result.get("note"),
        )
    import importlib

    versions = {}
    for m in ("numpy", "scipy", "statsmodels", "sklearn", "glum"):
        try:
            versions[m] = importlib.import_module(m).__version__
        except Exception:
            pass
    out["versions"] = versions
    Path(output).write_text(
        json.dumps(out, default=lambda v: float(v) if np.isscalar(v) else str(v)) + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
