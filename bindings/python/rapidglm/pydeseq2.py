"""RapidGLM as a PyDESeq2 inference back end.

PyDESeq2 (MIT) accepts an ``inference`` object implementing its documented
``pydeseq2.inference.Inference`` interface. ``RapidGLMInference`` subclasses
PyDESeq2's ``DefaultInference`` and replaces the two per-gene optimizations,
``irls`` (NB GLM fits at fixed dispersion) and ``alpha_mle`` (gene-wise and MAP
dispersions), with RapidGLM's batch fitters, and computes ``wald_test`` for all
genes at once; every other step is PyDESeq2's own. Genes RapidGLM does not converge on are handed to the default methods::

    from pydeseq2.dds import DeseqDataSet
    from rapidglm.pydeseq2 import RapidGLMInference

    inference = RapidGLMInference(n_cpus=8)
    dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition", inference=inference)
    dds.deseq2()
    DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference).summary()

Written against PyDESeq2's published interface only. Differences from the
default inference, all confined to a few genes per dataset:
- coefficients come from the same iterations as PyDESeq2's IRLS (means floored
  at ``min_mu``, ridge 1e-6, bounded coefficients); where PyDESeq2 switches to
  L-BFGS-B after its coefficients leave the bounds, RapidGLM continues the
  iterations on the bounded coefficients;
- dispersions are maximized by Newton steps in log dispersion rather than
  L-BFGS-B and reach at least as high a Cox-Reid adjusted likelihood where they
  differ; genes whose rough starting value is at the dispersion floor keep it
  (as DESeq2 does), whereas L-BFGS-B moves some of them;
- the Wald test gives the default's values to rounding.
"""

import os

import numpy as np
from pydeseq2.default_inference import DefaultInference
from scipy.stats import norm

from . import capi

__all__ = ["RapidGLMInference"]


class RapidGLMInference(DefaultInference):
    """PyDESeq2 inference with RapidGLM's NB fitters (``irls``, ``alpha_mle``).

    n_cpus sets RapidGLM's threads (default: all cores) and PyDESeq2's workers
    for the remaining methods."""

    def __init__(self, n_cpus=None, **kwargs):
        super().__init__(n_cpus=n_cpus, **kwargs)
        self.threads = n_cpus or os.cpu_count()
        self.fallbacks = {"irls": 0, "alpha_mle": 0}

    def irls(self, counts, size_factors, design_matrix, disp, min_mu, beta_tol, min_beta=-30,
             max_beta=30, optimizer="L-BFGS-B", maxiter=250):
        counts = np.asarray(counts, dtype=float)
        design = np.asarray(design_matrix, dtype=float)
        beta, mu, hat, converged = capi.nb_floored_fits(
            design, counts, np.asarray(size_factors, float), np.asarray(disp, float), min_mu=min_mu,
            tolerance=beta_tol, max_iterations=int(maxiter), min_beta=min_beta, max_beta=max_beta,
            threads=self.threads)
        bad = np.flatnonzero(~converged)
        if bad.size:
            self.fallbacks["irls"] += bad.size
            b, m, h, c = super().irls(counts[:, bad], size_factors, design_matrix, np.asarray(disp)[bad],
                                      min_mu, beta_tol, min_beta, max_beta, optimizer, maxiter)
            beta[bad], mu[:, bad], hat[:, bad], converged[bad] = b, m, h, c
        return beta, mu, hat, converged

    def alpha_mle(self, counts, design_matrix, mu, alpha_hat, min_disp, max_disp, prior_disp_var=None,
                  cr_reg=True, prior_reg=False, optimizer="L-BFGS-B"):
        counts = np.asarray(counts, dtype=float)
        variance = float(prior_disp_var) if prior_reg and prior_disp_var else 0.0
        if prior_reg and variance <= 0:
            return super().alpha_mle(counts, design_matrix, mu, alpha_hat, min_disp, max_disp,
                                     prior_disp_var, cr_reg, prior_reg, optimizer)
        dispersions, converged = capi.nb_dispersions(
            np.asarray(design_matrix, float), counts, np.asarray(mu, float), np.asarray(alpha_hat, float),
            float(min_disp), float(max_disp), variance, cr_reg, threads=self.threads)
        bad = np.flatnonzero(~converged)
        if bad.size:
            self.fallbacks["alpha_mle"] += bad.size
            d, c = super().alpha_mle(counts[:, bad], design_matrix, np.asarray(mu)[:, bad],
                                     np.asarray(alpha_hat)[bad], min_disp, max_disp, prior_disp_var, cr_reg,
                                     prior_reg, optimizer)
            dispersions[bad], converged[bad] = d, c
        return dispersions, converged

    def wald_test(self, design_matrix, disp, lfc, mu, ridge_factor, contrast, lfc_null,
                  alt_hypothesis=None):
        """Wald tests of all genes at once: covariance (X'WX + R)^-1 X'WX (X'WX + R)^-1
        with W = mu / (1 + disp mu) and ridge R, as in the DESeq2 paper. Other
        alternative hypotheses use the default implementation."""
        if alt_hypothesis is not None:
            return super().wald_test(design_matrix, disp, lfc, mu, ridge_factor, contrast, lfc_null,
                                     alt_hypothesis)
        x = np.asarray(design_matrix, float)
        mu = np.asarray(mu, float)
        disp = np.asarray(disp, float)
        c = np.asarray(contrast, float)
        w = mu / (1 + disp[None, :] * mu)  # samples x genes
        information = np.einsum("ni,nj,ng->gij", x, x, w, optimize=True)
        inverse = np.linalg.inv(information + np.asarray(ridge_factor, float)[None, :, :])
        covariance = inverse @ information @ inverse
        se = np.sqrt(np.einsum("i,gij,j->g", c, covariance, c))
        estimate = np.asarray(lfc, float) @ c
        statistic = (estimate - np.log(2) * float(lfc_null)) / se
        return 2 * norm.sf(np.abs(statistic)), statistic, se
