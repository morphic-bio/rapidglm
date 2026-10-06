#!/usr/bin/env python3
"""Score comparison results against the binary128 references and collect timings.

Coefficient error is reported in reference standard errors (the statistically
relevant unit) and relatively. The log-likelihood gap ll(reference) - ll(tool)
is evaluated for every tool by the same float64 code, with aliased coefficients
reported as NA counted as zero. For aliased and zero-group problems only the
identifiable coefficients are scored.
"""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import gammaln

ORDER = [
    "rapidglm",
    "rapidglm-singular",
    "r-rapidglm.fit",
    "r-rapidglm",
    "r-rapidglm.nb",
    "py-rapidglm",
    "glm.fit",
    "glm",
    "glm.nb",
    "fastglm",
    "speedglm",
    "statsmodels",
    "statsmodels-nb",
    "glum",
    "sklearn",
    "glum-tight",
    "sklearn-tight",
]


def loglik(family, X, y, off, w, b, theta):
    b = np.array([0.0 if v is None else v for v in b])
    eta = X @ b + off
    k = w > 0
    a, y, eta = w[k], y[k], eta[k]
    if family == "gaussian":
        rss = np.sum(a * (y - eta) ** 2)
        m = k.sum()
        return -m / 2 * (np.log(2 * np.pi * rss / m) + 1) + 0.5 * np.sum(np.log(a))
    if family == "binomial":
        return float(np.sum(a * (y * eta - np.logaddexp(0, eta))))
    if family == "poisson":
        return float(np.sum(a * (y * eta - np.exp(eta) - gammaln(y + 1))))
    t, mu = theta, np.exp(eta)
    return float(
        np.sum(
            a
            * (
                gammaln(y + t)
                - gammaln(t)
                - gammaln(y + 1)
                - t * np.log1p(mu / t)
                + y * (eta - np.log(t + mu))
            )
        )
    )


def score_problem(d):
    meta = json.loads((d / "meta.json").read_text())
    n, p, family = meta["n"], meta["p"], meta["family"]
    X = np.fromfile(d / "X.f64", "<f8").reshape(n, p)
    y, off, w = (
        np.fromfile(d / f, "<f8") for f in ("y.f64", "offset.f64", "weights.f64")
    )
    ref = (
        json.loads((d / "reference.json").read_text())
        if (d / "reference.json").exists()
        else None
    )
    name = d.name
    comparable = p - 1 if name.endswith(("-aliased", "-zero-group")) else p
    identifiable = (
        ref is not None and ref["converged"] and not name.endswith("-separated")
    )
    theta_ref = float(ref["theta"]) if ref and ref["theta"] is not None else None
    if identifiable:
        bref = [float(v) for v in ref["coefficients"]][:comparable]
        seref = [float(v) for v in ref["standard_errors"]][:comparable]
        full = bref + [0.0] * (p - comparable)
        if name.endswith("-zero-group"):
            full[-1] = -1e6  # the indicator's supremum; its rows then contribute 0
        ll_ref = loglik(family, X, y, off, w, full, theta_ref)
    rows = []
    for tool in ORDER:
        f = d / f"result-{tool}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        # R's auto_unbox writes length-one vectors as scalars.
        for key in ("seconds", "warnings", "coefficients", "standard_errors"):
            if r.get(key) is not None and not isinstance(r[key], list):
                r[key] = [r[key]]
        row = dict(
            problem=name,
            family=family,
            tool=tool,
            error=r.get("error"),
            converged=r.get("converged"),
            status=r.get("status"),
            warnings=r.get("warnings") or [],
            iterations=r.get("iterations"),
            seconds=min(r["seconds"]) if r.get("seconds") else None,
        )
        b = r.get("coefficients")
        if not row["error"] and b:
            finite = [v for v in b if v is not None]
            row["max_abs_coefficient"] = max(abs(v) for v in finite) if finite else None
            row["na_coefficients"] = sum(v is None for v in b)
            if identifiable:
                sub = b[:comparable]
                if any(v is None for v in sub):
                    row["coefficient_error_se"] = None
                    row["missing_identifiable"] = True
                else:
                    diff = [abs(u - v) for u, v in zip(sub, bref)]
                    row["coefficient_error_se"] = max(
                        dv / s for dv, s in zip(diff, seref)
                    )
                    row["coefficient_error_rel"] = max(
                        dv / max(abs(v), 1e-300) for dv, v in zip(diff, bref)
                    )
                theta = r.get("theta") if family == "nb" else None
                theta = theta if theta is not None else theta_ref
                try:
                    row["loglik_gap"] = ll_ref - loglik(family, X, y, off, w, b, theta)
                except Exception:
                    row["loglik_gap"] = None
                se = r.get("standard_errors")
                if se and all(v is not None for v in se[:comparable]):
                    row["se_error_rel"] = max(
                        abs(u - v) / v for u, v in zip(se[:comparable], seref)
                    )
                if family == "nb" and r.get("theta") is not None and theta_ref:
                    row["theta_error_rel"] = abs(r["theta"] - theta_ref) / theta_ref
        rows.append(row)
    return dict(
        problem=name,
        family=family,
        description=meta["description"],
        suite=meta["suite"],
        n=n,
        p=p,
        reference=(
            None
            if ref is None
            else dict(
                converged=ref["converged"], theta=theta_ref, identifiable=identifiable
            )
        ),
        results=rows,
    )


def main(args):
    problems = sorted(p for p in args.root.iterdir() if (p / "meta.json").exists())
    scored = [score_problem(p) for p in problems]
    args.output.write_text(json.dumps(scored, indent=1, default=float) + "\n")
    for s in scored:
        if s["suite"] != args.suite:
            continue
        print(f"== {s['problem']} ({s['description']})")
        for r in s["results"]:

            def fmt(k, spec=".1e"):
                v = r.get(k)
                return "-" if v is None else format(v, spec)

            flag = "ERROR " + r["error"][:60] if r["error"] else ""
            print(
                f"  {r['tool']:15s} err/SE {fmt('coefficient_error_se'):>8s} rel {fmt('coefficient_error_rel'):>8s}"
                f" llgap {fmt('loglik_gap'):>8s} se {fmt('se_error_rel'):>8s} t {fmt('seconds', '.4f'):>8s}"
                f" conv {r['converged']} NA {r.get('na_coefficients', '-')} max|b| {fmt('max_abs_coefficient')}"
                f" {r.get('status') or ''} {flag} {'W:' + r['warnings'][0][:70] if r['warnings'] else ''}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--suite", default="accuracy")
    main(ap.parse_args())


def speed_table(root, label=""):
    """Minimum single-thread fit time per problem and tool, with agreement to glm.fit."""
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        if meta["suite"] != "speed":
            continue
        results = {}
        for f in sorted(d.glob("result-*.json")):
            r = json.loads(f.read_text())
            for key in ("seconds", "coefficients"):
                if r.get(key) is not None and not isinstance(r[key], list):
                    r[key] = [r[key]]
            results[f.stem[len("result-") :]] = r
        anchor = results.get("glm.fit") or results.get("glm.nb")
        for tool, r in results.items():
            agree = None
            if anchor and r.get("coefficients") and anchor.get("coefficients"):
                a = np.array(
                    [np.nan if v is None else v for v in anchor["coefficients"]], float
                )
                b = np.array(
                    [np.nan if v is None else v for v in r["coefficients"]], float
                )
                if a.shape == b.shape:
                    agree = float(
                        np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-8))
                    )
            rows.append(
                dict(
                    problem=d.name,
                    family=meta["family"],
                    n=meta["n"],
                    p=meta["p"],
                    tool=tool,
                    seconds=min(r["seconds"]) if r.get("seconds") else None,
                    repeats=len(r.get("seconds") or []),
                    error=r.get("error"),
                    max_rel_diff_vs_R=agree,
                )
            )
    return rows
