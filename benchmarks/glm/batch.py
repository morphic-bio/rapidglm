#!/usr/bin/env python3
"""Many-response benchmark: m responses share one n x p design (genes x cells).

RapidGLM fit_batch (1..16 threads) against loops over R glm.fit / MASS::glm.nb,
the rapidglm R drop-ins, statsmodels GLM / NegativeBinomial and rapidglm.fit,
and (NB) glmGamPoi::glm_gp run as a black box on the whole batch (ML
overdispersion, no shrinkage, no Cox-Reid adjustment; best of three runs).
Single-thread timings are sequential; thread counts apply only to RapidGLM.
"""

import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONPATH": f"{ROOT / 'bindings/python'}:{ROOT / 'validation/local-data/python'}",
}
LIBS = f"{ROOT / 'validation/local-data/R_rapidglm'}:{ROOT / 'validation/local-data/R_library'}"
GGP_LIBS = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"
GGP = r"""
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[2], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages(library(glmGamPoi))
meta <- jsonlite::fromJSON(file.path(args[1], "meta.json")); n <- meta$n; p <- meta$p; m <- meta$m
X <- matrix(readBin(file.path(args[1], "X.f64"), "double", n * p), n, p, byrow = TRUE)
Y <- t(matrix(readBin(file.path(args[1], "Y.f64"), "double", n * m), n, m))
colnames(X) <- paste0("x", seq_len(p))
best <- Inf
for (r in 1:3) {
  t0 <- proc.time()[["elapsed"]]
  fit <- glm_gp(Y, design = X, size_factors = FALSE, overdispersion = TRUE,
                overdispersion_shrinkage = FALSE, do_cox_reid_adjustment = FALSE)
  best <- min(best, proc.time()[["elapsed"]] - t0)
}
write.table(cbind(fit$Beta, 1 / fit$overdispersions), file.path(args[1], "coef-glmgampoi-nb.tsv"),
            sep = "\t", row.names = FALSE, col.names = FALSE)
cat(jsonlite::toJSON(list(seconds = best, version = as.character(packageVersion("glmGamPoi"))), auto_unbox = TRUE), "\n")
"""


def generate(d, n, m, seed=20260929):
    rng = np.random.default_rng(seed)
    depth = rng.lognormal(np.log(5000), 0.5, n)
    X = np.column_stack(
        [np.ones(n), np.log(depth) - np.log(5000), rng.normal(size=(n, 3))]
    )
    Y = np.empty((m, n))
    for j in range(m):
        mu = np.exp(rng.uniform(-3, 1.5) + X[:, 1] + X[:, 2:] @ rng.normal(0, 0.2, 3))
        theta = rng.uniform(0.5, 10)
        y = rng.poisson(rng.gamma(theta, mu / theta))
        Y[j] = y
    d.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(X).astype("<f8").tofile(d / "X.f64")
    np.ascontiguousarray(Y).astype("<f8").tofile(d / "Y.f64")  # feature-major
    (d / "meta.json").write_text(json.dumps(dict(n=n, p=X.shape[1], m=m)))
    return X, Y


def python_loop(tool, family, X, Y):
    import sys

    sys.path.insert(0, str(ROOT / "bindings/python"))
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial
    import rapidglm

    start = time.perf_counter()
    out = []
    for y in Y:
        if tool == "statsmodels" and family == "poisson":
            r = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            out.append(list(r.params) + [np.nan])
        elif tool == "statsmodels" and family == "nb":
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = NegativeBinomial(y, X).fit(disp=0)
            out.append(list(r.params[:-1]) + [1 / r.params[-1]])
        else:
            r = rapidglm.fit(X, y, family=family)
            out.append(list(r["coefficients"]) + [r["theta"]])
    return time.perf_counter() - start, np.array(out)


def main(args):
    d = args.work
    X, Y = generate(d, args.n, args.m)
    n, p = X.shape
    results = []
    subprocess.run(
        ["cmake", "--build", str(ROOT / "build"), "--target", "rapidglm_glm_batch"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for family in ("poisson", "nb"):
        coefs = {}
        for threads in (1, 4, 16):
            r = subprocess.run(
                [
                    str(ROOT / "build/rapidglm_glm_batch"),
                    str(d),
                    str(n),
                    str(p),
                    str(args.m),
                    family,
                    str(threads),
                    str(d / "coef.tsv"),
                ],
                env=ENV,
                capture_output=True,
                text=True,
                check=True,
            )
            results.append(
                dict(
                    tool="rapidglm fit_batch",
                    family=family,
                    load=os.getloadavg()[0],
                    **json.loads(r.stdout),
                )
            )
            coefs["rapidglm"] = np.loadtxt(d / "coef.tsv")
            print(results[-1], flush=True)
        for tool in (
            ["glm.fit", "rapidglm"] if family == "poisson" else ["glm.nb", "rapidglm"]
        ):
            r = subprocess.run(
                [
                    "Rscript",
                    str(ROOT / "benchmarks/glm/batch_r.R"),
                    str(d),
                    tool,
                    family,
                    LIBS,
                ],
                env=ENV,
                capture_output=True,
                text=True,
                check=True,
            )
            label = (
                f"R {tool}"
                if tool != "rapidglm"
                else f"R rapidglm::{'glm_fit' if family == 'poisson' else 'glm_nb'}"
            )
            results.append(
                dict(
                    tool=label,
                    family=family,
                    threads=1,
                    load=os.getloadavg()[0],
                    **json.loads(r.stdout),
                )
            )
            coefs[label] = np.genfromtxt(
                d / f"coef-{tool}-{family}.tsv",
                missing_values="NA",
                filling_values=np.nan,
            )
            print(results[-1], flush=True)
        if family == "nb":
            script = d / "glmgampoi.R"
            script.write_text(GGP)
            r = subprocess.run(["Rscript", str(script), str(d), GGP_LIBS], env=ENV, capture_output=True,
                               text=True, check=True)
            info = json.loads(r.stdout.strip().splitlines()[-1])
            results.append(dict(tool="R glmGamPoi::glm_gp", family=family, threads=1, load=os.getloadavg()[0],
                                seconds=info["seconds"], glmgampoi_version=info["version"]))
            coefs["R glmGamPoi::glm_gp"] = np.loadtxt(d / "coef-glmgampoi-nb.tsv")
            print(results[-1], flush=True)
        for tool in ("statsmodels", "rapidglm.fit"):
            if tool == "statsmodels" and family == "nb" and args.m > 200:
                sub = Y[
                    :200
                ]  # the discrete NB model is slow; timed on a subset and scaled
            else:
                sub = Y
            seconds, c = python_loop(tool, family, X, sub)
            results.append(
                dict(
                    tool=f"Python {tool}",
                    family=family,
                    threads=1,
                    load=os.getloadavg()[0],
                    seconds=seconds * len(Y) / len(sub),
                    responses_timed=len(sub),
                )
            )
            coefs[f"Python {tool}"] = c
            print(results[-1], flush=True)
        ref = coefs["R glm.fit" if family == "poisson" else "R glm.nb"]
        for label, c in coefs.items():
            k = min(len(c), len(ref))
            diff = np.abs(c[:k, :p] - ref[:k, :p]) / np.maximum(
                np.abs(ref[:k, :p]), 1e-8
            )
            for r in results:
                if r["family"] == family and (
                    r["tool"] == label
                    or (label == "rapidglm" and r["tool"] == "rapidglm fit_batch")
                ):
                    r["max_rel_coef_diff_vs_R"] = float(np.nanmax(diff))
    (args.output).write_text(
        json.dumps(dict(n=n, p=p, m=args.m, results=results), indent=1) + "\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work", type=Path, default=ROOT / "validation/work/glm-compare/batch"
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--m", type=int, default=1000)
    main(ap.parse_args())
