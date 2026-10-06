"""Parity of rapidglm.statsmodels.GLM with statsmodels.api.GLM."""

import numpy as np
import statsmodels.api as sm
import rapidglm.statsmodels as rsm

rng = np.random.default_rng(7)
n = 800
X = sm.add_constant(np.column_stack([rng.normal(size=n), rng.integers(0, 3, n)]))
off = rng.normal(0, 0.3, n)
w = rng.uniform(0.5, 2, n)
cases = {
    "poisson_offset": (
        rng.poisson(np.exp(0.3 + 0.4 * X[:, 1] + off)),
        sm.families.Poisson(),
        dict(offset=off),
    ),
    "poisson_exposure": (
        rng.poisson(np.exp(0.3 + 0.4 * X[:, 1]) * np.exp(off)),
        sm.families.Poisson(),
        dict(exposure=np.exp(off)),
    ),
    "poisson_var_weights": (
        rng.poisson(np.exp(0.3 + 0.4 * X[:, 1])),
        sm.families.Poisson(),
        dict(var_weights=w),
    ),
    "poisson_freq_weights": (
        rng.poisson(np.exp(0.3 + 0.4 * X[:, 1])),
        sm.families.Poisson(),
        dict(freq_weights=np.round(w * 2)),
    ),
    "binomial": (
        rng.binomial(1, 1 / (1 + np.exp(-0.5 * X[:, 1]))),
        sm.families.Binomial(),
        {},
    ),
    "binomial_trials": (
        np.column_stack([s := rng.binomial(10, 0.4, n), 10 - s]),
        sm.families.Binomial(),
        {},
    ),
    "gaussian": (1 + X[:, 1] + rng.normal(size=n), sm.families.Gaussian(), {}),
    "gaussian_weights": (
        1 + X[:, 1] + rng.normal(size=n),
        sm.families.Gaussian(),
        dict(var_weights=w),
    ),
    "nb_fixed_alpha": (
        rng.negative_binomial(2, 2 / (2 + np.exp(1 + 0.4 * X[:, 1]))),
        sm.families.NegativeBinomial(alpha=0.5),
        {},
    ),
}
for name, (y, fam, kw) in cases.items():
    ref = sm.GLM(y, X, family=fam, **kw).fit()
    new = rsm.GLM(y, X, family=fam, **kw).fit()
    assert new.model.__class__ is rsm.GLM and new.method == "rapidglm", name
    d = dict(
        params=np.max(np.abs(new.params - ref.params)),
        bse=np.max(np.abs(new.bse - ref.bse) / ref.bse),
        pvalues=np.max(np.abs(new.pvalues - ref.pvalues)),
        llf=abs(new.llf - ref.llf),
        deviance=abs(new.deviance - ref.deviance),
        scale=abs(new.scale - ref.scale),
        aic=abs(new.aic - ref.aic),
    )
    robust = rsm.GLM(y, X, family=fam, **kw).fit(cov_type="HC0")
    d["HC0_bse"] = np.max(
        np.abs(robust.bse - sm.GLM(y, X, family=fam, **kw).fit(cov_type="HC0").bse)
    )
    print(f"{name:22s} " + " ".join(f"{k} {v:.1e}" for k, v in d.items()))
    assert d["params"] < 1e-6 and d["bse"] < 1e-5 and d["llf"] < 1e-6, name
    new.summary()
    new.conf_int()
    new.predict(X[:3])
    new.get_prediction(X[:3]).summary_frame()
gamma = rsm.GLM(
    np.exp(X[:, 1]) + 1, X, family=sm.families.Gamma(sm.families.links.Log())
).fit()
assert (
    getattr(gamma, "method", "IRLS") != "rapidglm"
)  # unsupported family: statsmodels fitting
alias = np.column_stack([X, X[:, 1] * 2])
res = rsm.GLM(cases["poisson_offset"][0], alias, family=sm.families.Poisson()).fit()
assert (
    getattr(res, "method", "IRLS") != "rapidglm"
)  # rank deficient: statsmodels fitting
print("ok")
