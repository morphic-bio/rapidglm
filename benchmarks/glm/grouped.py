#!/usr/bin/env python3
"""Grouped-design batch benchmark: 1,000 NB responses, 5,000 observations.

Designs: four groups (intercept + three indicators) and intercept only, both
with a log depth offset. Times RapidGLM fit_batch (one thread, minimum of three
runs) for this build and an optional baseline batch binary, compares their
estimates, and times glmGamPoi 1.25.1 (black box, one fit/dispersion/refit
pass with ML dispersions and no shrinkage) when an R library is given.
"""

import argparse, json, os, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
R_SCRIPT = r"""
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[2], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages(library(glmGamPoi))
meta <- jsonlite::fromJSON(file.path(args[1], "meta.json")); n <- meta$n; p <- meta$p; m <- meta$m
X <- matrix(readBin(file.path(args[1], "X.f64"), "double", n * p), n, p, byrow = TRUE)
Y <- t(matrix(readBin(file.path(args[1], "Y.f64"), "double", n * m), n, m))
off <- readBin(file.path(args[1], "offset.f64"), "double", n)
colnames(X) <- paste0("x", seq_len(p))
best <- Inf
for (r in 1:3) {
  t0 <- proc.time()[["elapsed"]]
  fit <- glm_gp(Y, design = X, offset = off, size_factors = FALSE, overdispersion = TRUE,
                overdispersion_shrinkage = FALSE, do_cox_reid_adjustment = FALSE)
  best <- min(best, proc.time()[["elapsed"]] - t0)
}
cat(best, "\n")
"""


def generate(d, design, n=5000, m=1000, seed=20260930):
    rng = np.random.default_rng(seed)
    offset = np.log(rng.lognormal(np.log(5000), 0.5, n))
    offset -= offset.mean()
    g = rng.integers(0, 4, n)
    X = (
        np.column_stack([np.ones(n)] + [(g == k).astype(float) for k in (1, 2, 3)])
        if design == "oneway"
        else np.ones((n, 1))
    )
    Y = np.empty((m, n))
    for j in range(m):
        b = np.concatenate([[rng.uniform(-3, 1.5)], rng.normal(0, 0.5, X.shape[1] - 1)])
        theta = rng.uniform(0.5, 10)
        Y[j] = rng.poisson(rng.gamma(theta, np.exp(X @ b + offset) / theta))
    d.mkdir(parents=True, exist_ok=True)
    X.astype("<f8").tofile(d / "X.f64")
    Y.astype("<f8").tofile(d / "Y.f64")
    offset.astype("<f8").tofile(d / "offset.f64")
    (d / "meta.json").write_text(json.dumps(dict(n=n, p=X.shape[1], m=m)))
    return X.shape[1]


def time_batch(binary, d, p, family, out):
    runs = [
        json.loads(
            subprocess.run(
                [str(binary), str(d), "5000", str(p), "1000", family, "1", str(out)],
                check=True, capture_output=True, text=True, env=ENV,
            ).stdout
        )["seconds"]
        for _ in range(3)
    ]
    return min(runs)


def main(args):
    subprocess.run(
        ["cmake", "--build", str(ROOT / "build"), "--target", "rapidglm_glm_batch"],
        check=True, stdout=subprocess.DEVNULL,
    )
    current = ROOT / "build/rapidglm_glm_batch"
    results = []
    for design in ("oneway", "intercept"):
        d = args.work / design
        p = generate(d, design)
        for family in ("nb", "poisson"):
            row = dict(design=design, family=family, p=p)
            row["rapidglm_seconds"] = time_batch(current, d, p, family, d / f"{family}.tsv")
            if args.baseline_batch:
                row["baseline_seconds"] = time_batch(
                    args.baseline_batch, d, p, family, d / f"{family}.baseline.tsv"
                )
                a = np.loadtxt(d / f"{family}.tsv").reshape(1000, -1)
                b = np.loadtxt(d / f"{family}.baseline.tsv").reshape(1000, -1)
                row["max_rel_coefficient_difference"] = float(
                    np.max(np.abs(a[:, :p] - b[:, :p]) / (1 + np.abs(b[:, :p])))
                )
                if family == "nb":
                    row["max_abs_log_theta_difference"] = float(
                        np.max(np.abs(np.log(a[:, p] / b[:, p])))
                    )
            if args.r_library and family == "nb":
                script = args.work / "glmgampoi.R"
                script.write_text(R_SCRIPT)
                row["glmgampoi_seconds"] = float(
                    subprocess.run(
                        ["Rscript", str(script), str(d), args.r_library],
                        check=True, capture_output=True, text=True, env=ENV, cwd=ROOT,
                    ).stdout.split()[0]
                )
            results.append(row)
            print(row)
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        command="python3 benchmarks/glm/grouped.py --work DIR --output validation/GROUPED_SOLVER.json "
        "--baseline-batch BASELINE --r-library validation/local-data/R_glmgampoi:validation/local-data/R_library",
        baseline=args.baseline_note,
        timing="minimum of three runs, one thread, shared host",
        results=results,
        load_average=os.getloadavg(),
    )
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline-batch", type=Path)
    ap.add_argument("--baseline-note", default="")
    ap.add_argument("--r-library")
    main(ap.parse_args())
