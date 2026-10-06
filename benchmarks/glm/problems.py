#!/usr/bin/env python3
"""Generate GLM comparison problems as little-endian float64 binaries.

Each problem directory holds X.f64 (row-major n x p), y.f64, offset.f64,
weights.f64 and meta.json. The accuracy suite probes conditioning, scaling,
aliasing, boundaries and weights at moderate n; the speed suite varies n and p.
All tools read identical bytes.
"""

import argparse, json
from pathlib import Path
import numpy as np


def write(
    root,
    name,
    suite,
    family,
    X,
    y,
    description,
    offset=None,
    weights=None,
    theta=None,
    estimate_theta=False,
    truth=None,
):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    n, p = X.shape
    np.ascontiguousarray(X, dtype="<f8").tofile(d / "X.f64")
    np.asarray(y, dtype="<f8").tofile(d / "y.f64")
    np.asarray(np.zeros(n) if offset is None else offset, dtype="<f8").tofile(
        d / "offset.f64"
    )
    np.asarray(np.ones(n) if weights is None else weights, dtype="<f8").tofile(
        d / "weights.f64"
    )
    meta = dict(
        name=name,
        suite=suite,
        family=family,
        n=n,
        p=p,
        description=description,
        theta=theta,
        estimate_theta=estimate_theta,
        has_offset=offset is not None,
        has_weights=weights is not None,
        truth=None if truth is None else [float(v) for v in truth],
    )
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def draw(rng, family, eta, theta=None):
    if family == "gaussian":
        return eta + rng.normal(size=eta.size)
    if family == "binomial":
        return (rng.random(eta.size) < 1 / (1 + np.exp(-eta))).astype(float)
    mu = np.exp(eta)
    if family == "poisson":
        return rng.poisson(mu).astype(float)
    return rng.poisson(rng.gamma(theta, mu / theta)).astype(float)


def accuracy_suite(root, rng):
    n = 2000
    for family in ("gaussian", "poisson", "binomial", "nb"):
        theta = 2.0 if family == "nb" else None
        est = family == "nb"
        z = rng.normal(size=(n, 4))
        base = np.column_stack([np.ones(n), z])
        beta = np.array([0.3, 0.4, -0.3, 0.2, 0.1])
        eta = base @ beta
        y = draw(rng, family, eta, theta)
        write(
            root,
            f"{family}-baseline",
            "accuracy",
            family,
            base,
            y,
            "intercept + four N(0,1) covariates",
            theta=theta,
            estimate_theta=est,
            truth=beta,
        )
        scale = np.array([1, 1e-6, 1e6, 1e3, 1e-3])
        write(
            root,
            f"{family}-scaled",
            "accuracy",
            family,
            base * scale,
            y,
            "baseline columns scaled by 1, 1e-6, 1e6, 1e3, 1e-3",
            theta=theta,
            estimate_theta=est,
            truth=beta / scale,
        )
        unc = base.copy()
        unc[:, 1:] += 1e4
        write(
            root,
            f"{family}-uncentered",
            "accuracy",
            family,
            unc,
            y,
            "covariates shifted by 1e4 (uncentered, like calendar years)",
            theta=theta,
            estimate_theta=est,
        )
        for exponent in (6, 9, 12):
            col = base.copy()
            col[:, 2] = col[:, 1] + 10.0**-exponent * rng.normal(size=n)
            yc = draw(rng, family, col @ np.array([0.3, 0.4, 0.0, 0.2, 0.1]), theta)
            write(
                root,
                f"{family}-collinear-1e-{exponent}",
                "accuracy",
                family,
                col,
                yc,
                f"x2 = x1 + 1e-{exponent} N(0,1)",
                theta=theta,
                estimate_theta=est,
            )
        alias = np.column_stack([base, base[:, 1] + base[:, 2]])
        write(
            root,
            f"{family}-aliased",
            "accuracy",
            family,
            alias,
            y,
            "sixth column = x1 + x2 exactly (rank deficient)",
            theta=theta,
            estimate_theta=est,
        )
        w = rng.choice([0, 0.5, 1, 2, 3.5], size=n)
        write(
            root,
            f"{family}-weights",
            "accuracy",
            family,
            base,
            y,
            "prior weights in {0, 0.5, 1, 2, 3.5}",
            weights=w,
            theta=theta,
            estimate_theta=est,
        )
        if family in ("poisson", "nb"):
            off = rng.normal(np.log(1e4), 0.5, size=n)
            yo = draw(
                rng, family, base @ np.array([-6.0, 0.4, -0.3, 0.2, 0.1]) + off, theta
            )
            write(
                root,
                f"{family}-offset",
                "accuracy",
                family,
                base,
                yo,
                "log-depth offset around log(1e4)",
                offset=off,
                theta=theta,
                estimate_theta=est,
            )
            yr = draw(rng, family, base @ np.array([-4.5, 0.4, -0.3, 0.2, 0.1]), theta)
            write(
                root,
                f"{family}-sparse",
                "accuracy",
                family,
                base,
                yr,
                "mean about 0.01 (mostly zeros)",
                theta=theta,
                estimate_theta=est,
            )
            yl = draw(rng, family, base @ np.array([11.5, 0.4, -0.3, 0.2, 0.1]), theta)
            write(
                root,
                f"{family}-large-counts",
                "accuracy",
                family,
                base,
                yl,
                "mean about 1e5",
                theta=theta,
                estimate_theta=est,
            )
            group = np.zeros((n, 1))
            group[:40, 0] = 1
            yz = y.copy()
            yz[:40] = 0
            write(
                root,
                f"{family}-zero-group",
                "accuracy",
                family,
                np.column_stack([base, group]),
                yz,
                "indicator of 40 observations that are all zero (no finite MLE for it)",
                theta=theta,
                estimate_theta=est,
            )
        if family == "binomial":
            yr = draw(rng, family, base @ np.array([-7.0, 0.4, -0.3, 0.2, 0.1]))
            yr[:3] = 1
            write(
                root,
                f"{family}-rare",
                "accuracy",
                family,
                base,
                yr,
                "event rate about 1e-3",
                truth=None,
            )
            sep = base.copy()
            ys = (sep[:, 1] > 0).astype(float)
            write(
                root,
                f"{family}-separated",
                "accuracy",
                family,
                sep,
                ys,
                "x1 > 0 perfectly separates the response (no finite MLE)",
            )
        if family == "nb":
            for th, label in ((0.05, "overdispersed"), (1e6, "near-poisson")):
                yt = draw(rng, "nb", eta, th)
                write(
                    root,
                    f"nb-{label}",
                    "accuracy",
                    "nb",
                    base,
                    yt,
                    f"true theta {th:g}",
                    theta=th,
                    estimate_theta=True,
                )


def speed_suite(root, rng, largest):
    for family in ("poisson", "binomial", "gaussian", "nb"):
        for n in (10_000, 100_000, 1_000_000):
            if n > largest or (family == "nb" and n > 100_000):
                continue
            for p in (5, 20, 50):
                X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
                beta = rng.normal(0, 0.3 / np.sqrt(p), size=p)
                beta[0] = 0.5 if family != "binomial" else 0.0
                y = draw(rng, family, X @ beta, 2.0 if family == "nb" else None)
                write(
                    root,
                    f"speed-{family}-n{n}-p{p}",
                    "speed",
                    family,
                    X,
                    y,
                    f"n={n}, p={p}, N(0,1) covariates",
                    theta=2.0 if family == "nb" else None,
                    estimate_theta=family == "nb",
                    truth=beta,
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--largest", type=int, default=1_000_000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260929)
    accuracy_suite(args.root, rng)
    speed_suite(args.root, rng, args.largest)
