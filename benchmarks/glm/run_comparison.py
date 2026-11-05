#!/usr/bin/env python3
"""Run the GLM comparison: binary128 references, then every tool on every problem.

Accuracy problems run concurrently (their timings are not reported); speed
problems run one process at a time with one BLAS/OpenMP thread per tool.
Results are written beside each problem as result-<tool>.json.
"""

import argparse, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = {
    "rapidglm": ("cpp", ["gaussian", "poisson", "binomial", "nb"]),
    # R-compatible options: aliased columns dropped and reported as NA.
    "rapidglm-singular": ("cpp", ["gaussian", "poisson", "binomial", "nb"]),
    "glm.fit": ("R", ["gaussian", "poisson", "binomial"]),
    "glm": ("R", ["gaussian", "poisson", "binomial"]),
    "glm.nb": ("R", ["nb"]),
    "fastglm": ("R", ["gaussian", "poisson", "binomial"]),
    "speedglm": ("R", ["gaussian", "poisson", "binomial"]),
    "statsmodels": ("py", ["gaussian", "poisson", "binomial"]),
    "statsmodels-nb": ("py", ["nb"]),
    "glum": ("py", ["gaussian", "poisson", "binomial"]),
    # Drop-in paths: R glm.fit / glm(method=) / glm.nb and statsmodels GLM replacements.
    "r-rapidglm.fit": ("R", ["gaussian", "poisson", "binomial"]),
    "r-rapidglm": ("R", ["gaussian", "poisson", "binomial"]),
    "r-rapidglm.nb": ("R", ["nb"]),
    "py-rapidglm": ("py", ["gaussian", "poisson", "binomial"]),
    "sklearn": ("py", ["gaussian", "poisson", "binomial"]),
    # Matched-accuracy variants: gradient/step tolerances near RapidGLM's.
    "glum-tight": ("py", ["gaussian", "poisson", "binomial"]),
    "sklearn-tight": ("py", ["gaussian", "poisson", "binomial"]),
}
ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONPATH": f"{ROOT / 'bindings/python'}:{ROOT / 'validation/local-data/python'}",
}


RAPIDGLM_BINARY = ROOT / "build/rapidglm_glm_compare"
REFERENCE_BINARY = ROOT / "validation/work/glm-compare/bin/glm_reference"


def reduced(problem, drop_rows, drop_last_column):
    """Identifiable sub-problem whose MLE the identifiable coefficients must match."""
    meta = json.loads((problem / "meta.json").read_text())
    n, p = meta["n"], meta["p"]
    X = np.fromfile(problem / "X.f64", "<f8").reshape(n, p)
    arrays = {
        f: np.fromfile(problem / f, "<f8")
        for f in ("y.f64", "offset.f64", "weights.f64")
    }
    rows = np.arange(drop_rows, n)
    cols = p - 1 if drop_last_column else p
    target = problem / "reduced"
    target.mkdir(exist_ok=True)
    np.ascontiguousarray(X[rows, :cols]).astype("<f8").tofile(target / "X.f64")
    for f, v in arrays.items():
        v[rows].astype("<f8").tofile(target / f)
    (target / "meta.json").write_text(json.dumps(dict(meta, n=len(rows), p=cols)))
    return target


def command(tool, problem, output, budget):
    kind = TOOLS[tool][0]
    if kind == "cpp":
        flags = ["singular_ok"] if tool == "rapidglm-singular" else []
        return [str(RAPIDGLM_BINARY), str(problem), str(output), str(budget), *flags]
    if kind == "R":
        return [
            "Rscript",
            str(ROOT / "benchmarks/glm/r_fit.R"),
            str(problem),
            tool,
            str(output),
            str(budget),
            f"{ROOT / 'validation/local-data/R_rapidglm'}:{ROOT / 'validation/local-data/R_library'}",
        ]
    return [
        "python3",
        str(ROOT / "benchmarks/glm/python_fit.py"),
        str(problem),
        tool,
        str(output),
        str(budget),
    ]


def run_tool(tool, problem, budget, timeout, label=""):
    output = problem / f"result-{tool}{label}.json"
    try:
        r = subprocess.run(
            command(tool, problem, output, budget),
            env=ENV,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode:
            output.write_text(
                json.dumps(dict(tool=tool, error=r.stderr.strip()[-2000:]))
            )
    except subprocess.TimeoutExpired:
        output.write_text(
            json.dumps(dict(tool=tool, error=f"timeout after {timeout} s"))
        )
    return tool, problem.name


def run_reference(problem):
    name = problem.name
    target = problem
    if name.endswith("-aliased"):
        target = reduced(problem, 0, True)
    elif name.endswith("-zero-group"):
        target = reduced(problem, 40, True)
    subprocess.run(
        [
            str(REFERENCE_BINARY),
            str(target),
            str(problem / "reference.json"),
        ],
        check=True,
    )
    return name


def main(args):
    problems = sorted(p for p in args.root.iterdir() if (p / "meta.json").exists())
    meta = {p: json.loads((p / "meta.json").read_text()) for p in problems}
    accuracy = [p for p in problems if meta[p]["suite"] == "accuracy"]
    speed = [p for p in problems if meta[p]["suite"] == "speed"]
    tools = args.tools.split(",") if args.tools else list(TOOLS)
    if args.reference:
        with ThreadPoolExecutor(args.jobs) as pool:
            for name in pool.map(
                run_reference,
                [p for p in accuracy if not (p / "reference.json").exists()],
            ):
                print("reference", name, flush=True)
    if args.accuracy:
        jobs = [
            (t, p) for p in accuracy for t in tools if meta[p]["family"] in TOOLS[t][1]
        ]
        with ThreadPoolExecutor(args.jobs) as pool:
            for tool, name in pool.map(lambda j: run_tool(j[0], j[1], 0.5, 600), jobs):
                print("accuracy", name, tool, flush=True)
    if args.speed:
        for p in speed:
            for t in tools:
                if meta[p]["family"] in TOOLS[t][1]:
                    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
                        sys.path.insert(0, str(ROOT / "scripts"))
                        from quiet_run import wait_quiet
                        wait_quiet(os.sched_getaffinity(0), label=f"GLM {p.name} {t}")
                    run_tool(t, p, args.budget, args.timeout, args.label)
                    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
                        from quiet_run import finish_measurement
                        finish_measurement()
                    print("speed", p.name, t, os.getloadavg()[0], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tools")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--rapidglm-binary", type=Path)
    ap.add_argument("--reference-binary", type=Path)
    ap.add_argument("--label", default="", help="suffix for result files")
    args = ap.parse_args()
    if args.rapidglm_binary:
        RAPIDGLM_BINARY = args.rapidglm_binary.resolve()
    if args.reference_binary:
        REFERENCE_BINARY = args.reference_binary.resolve()
    main(args)
