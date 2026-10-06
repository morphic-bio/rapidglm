"""statsmodels-compatible GLM with the RapidGLM engine.

    import rapidglm.statsmodels as rsm
    result = rsm.GLM(endog, exog, family=sm.families.Poisson(), offset=off).fit()

``GLM`` subclasses ``statsmodels.api.GLM``; ``fit`` returns statsmodels
``GLMResults`` built from RapidGLM's estimates, so summary(), conf_int(),
predict(), robust covariances and the rest of statsmodels apply. Gaussian
(identity), Poisson (log), binomial (logit) and negative binomial with fixed
alpha (log) are fitted natively. Other families or links, other fit methods,
rank-deficient designs and constrained fits use statsmodels' own fitting.
"""

import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.genmod.generalized_linear_model import GLM as _GLM
from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper

from . import _rapidglm

__all__ = ["GLM"]


def _native(family):
    link = family.link
    if isinstance(family, families.Gaussian) and isinstance(link, links.Identity):
        return "gaussian", None
    if isinstance(family, families.Poisson) and isinstance(link, links.Log):
        return "poisson", None
    if isinstance(family, families.Binomial) and isinstance(link, links.Logit):
        return "binomial", None
    if isinstance(family, families.NegativeBinomial) and isinstance(link, links.Log):
        return "nb", 1.0 / family.alpha
    return None, None


class GLM(_GLM):
    __doc__ = _GLM.__doc__

    def fit(
        self,
        start_params=None,
        maxiter=100,
        method="IRLS",
        tol=1e-8,
        scale=None,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=None,
        full_output=True,
        disp=False,
        max_start_irls=3,
        **kwargs,
    ):
        name, theta = _native(self.family)
        fallback = (
            name is None or method.lower() not in ("irls", "rapidglm") or bool(kwargs)
        )
        if not fallback:
            result = self._fit_rapidglm(
                name,
                theta,
                start_params,
                maxiter,
                tol,
                scale,
                cov_type,
                cov_kwds,
                use_t,
            )
            if result is not None:
                return result
        return super().fit(
            start_params=start_params,
            maxiter=maxiter,
            method=method,
            tol=tol,
            scale=scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
            full_output=full_output,
            disp=disp,
            max_start_irls=max_start_irls,
            **kwargs,
        )

    def _fit_rapidglm(
        self, name, theta, start_params, maxiter, tol, scale, cov_type, cov_kwds, use_t
    ):
        endog = np.asarray(self.endog, dtype=float)
        n = endog.shape[0]
        weights = np.asarray(self.freq_weights, dtype=float) * np.asarray(
            self.var_weights, dtype=float
        )
        if name == "binomial":
            weights = weights * np.asarray(self.n_trials, dtype=float)
        offset = np.zeros(n)
        if getattr(self, "offset", None) is not None:
            offset = offset + np.asarray(self.offset, dtype=float)
        if getattr(self, "exposure", None) is not None:
            offset = offset + np.asarray(
                self.exposure, dtype=float
            )  # stored as log(exposure)
        start = (
            np.empty(0)
            if start_params is None
            else np.asarray(start_params, dtype=float)
        )
        r = _rapidglm.fit(
            np.ascontiguousarray(self.exog, dtype=float),
            endog,
            weights,
            offset,
            start,
            name,
            float(theta or 10.0),
            False,
            float(min(tol, 1e-9)),
            int(max(maxiter, 1)),
            True,
            1e-11,
        )
        if any(r["aliased"]) or r["status"] in (
            "rank_deficient",
            "numerical_failure",
            "boundary_response",
        ):
            return None  # statsmodels' pseudo-inverse semantics for these designs
        params = np.asarray(r["coefficients"])
        normalized = np.asarray(r["covariance"]) / r["scale"]
        mu = self.predict(params)
        self.mu = mu
        if scale is None:
            scale = self.estimate_scale(mu)
        elif isinstance(scale, str):
            self.scaletype = scale
            scale = self.estimate_scale(mu)
        self.scale = scale
        if not r["converged"]:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            warnings.warn(f"RapidGLM fit status {r['status']}", ConvergenceWarning)
        result = GLMResults(
            self,
            params,
            normalized,
            scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )
        result.method = "rapidglm"
        result.fit_history = {
            "iteration": r["iterations"],
            "status": r["status"],
            "used_qr": r["used_qr"],
        }
        result.converged = bool(r["converged"])
        return GLMResultsWrapper(result)
