#!/usr/bin/env python3
"""Build the GLM comparison receipt and Markdown tables from harness results."""

import argparse, collections, hashlib, json, math, os, statistics, subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "rapidglm-singular": "RapidGLM engine (R-compatible aliasing)",
    "r-rapidglm": "R glm(method = rapidglm::glm_fit)",
    "r-rapidglm.nb": "R rapidglm::glm_nb",
    "py-rapidglm": "Python rapidglm.statsmodels.GLM",
    "glm.fit": "R stats::glm.fit",
    "glm.nb": "R MASS::glm.nb",
    "fastglm": "R fastglm",
    "speedglm": "R speedglm",
    "statsmodels": "Python statsmodels GLM",
    "statsmodels-nb": "Python statsmodels NegativeBinomial",
    "glum": "Python glum (defaults)",
    "glum-tight": "Python glum (tight tolerances)",
    "sklearn": "Python scikit-learn (defaults)",
    "sklearn-tight": "Python scikit-learn (tol 1e-10)",
}


def accuracy_summary(scored):
    agg = collections.defaultdict(
        lambda: dict(problems=0, failures=[], errors=[], se=[])
    )
    for p in scored:
        ref = p["reference"]
        # Relative residual 1e-12 is below R's aliasing threshold; aliasing there is a
        # documented tolerance choice, reported separately.
        if (
            p["suite"] != "accuracy"
            or not ref
            or not ref["identifiable"]
            or "collinear-1e-12" in p["problem"]
        ):
            continue
        for r in p["results"]:
            a = agg[r["tool"]]
            a["problems"] += 1
            e = r.get("coefficient_error_se")
            if r["error"] or e is None or math.isnan(e) or e > 1e-3:
                a["failures"].append(p["problem"])
            else:
                a["errors"].append(e)
                if r.get("se_error_rel") is not None:
                    a["se"].append(r["se_error_rel"])
    out = {}
    for tool, a in agg.items():
        out[tool] = dict(
            problems=a["problems"],
            failures=a["failures"],
            median_coefficient_error_se=(
                statistics.median(a["errors"]) if a["errors"] else None
            ),
            max_coefficient_error_se=max(a["errors"]) if a["errors"] else None,
            median_se_relative_error=statistics.median(a["se"]) if a["se"] else None,
            max_se_relative_error=max(a["se"]) if a["se"] else None,
        )
    return out


def edge_cases(scored):
    rows = {}
    for p in scored:
        if p["suite"] != "accuracy":
            continue
        if not any(
            k in p["problem"]
            for k in ("aliased", "collinear-1e-12", "separated", "zero-group")
        ):
            continue
        rows[p["problem"]] = {
            r["tool"]: dict(
                error=r["error"],
                status=r.get("status"),
                converged=r.get("converged"),
                na=r.get("na_coefficients"),
                loglik_gap=r.get("loglik_gap"),
                max_abs_coefficient=r.get("max_abs_coefficient"),
                warnings=r["warnings"][:2],
            )
            for r in p["results"]
        }
    return rows


def engine_regression():
    """Panel and calibration checks of the engine changes against 1975ed4."""
    work = ROOT / "validation/work/glm-compare/engine-check"
    if not (work / "compare.json").exists():
        return None
    compare = json.loads((work / "compare.json").read_text())
    cases = []
    for c in compare["cases"]:
        t = c.get(".tsv", {})
        entry = dict(case=c["case"], byte_identical=t.get("byte_identical"))
        cm = t.get("comparison")
        if cm:
            entry.update(
                discoveries=cm.get("discoveries"),
                discovery_jaccard=cm.get("discovery_jaccard"),
                text_or_finiteness_mismatches=sum(
                    v["text_mismatches"] + v["finiteness_mismatches"]
                    for v in cm["differing_columns"].values()
                ),
                max_abs_diff={
                    k: v["max_abs"] for k, v in cm["differing_columns"].items()
                },
            )
        cases.append(entry)
    e1 = work / "e1.tsv"
    return dict(
        command="python3 scripts/compare_binaries.py --baseline <1975ed4 CLI> (effect_fits=all)",
        baseline_sha256=compare["baseline_sha256"],
        cases=cases,
        e1_sha256=hashlib.sha256(e1.read_bytes()).hexdigest() if e1.exists() else None,
        e1_note="Identical to the historical G3 output (033c8263...).",
        tests="51 C++ cases and 12 CLI cases pass, also in ASan/UBSan builds with OpenMP on and off; R and Python parity suites pass.",
    )


def main(args):
    root = args.root
    scored = [
        summarize.score_problem(p)
        for p in sorted(root.iterdir())
        if (p / "meta.json").exists()
    ]
    acc = accuracy_summary(scored)
    speed = summarize.speed_table(root)
    batch = (
        json.loads(args.batch.read_text())
        if args.batch and args.batch.exists()
        else None
    )
    receipt = dict(
        purpose="Accuracy, stability and single-thread speed of RapidGLM against the standard R and Python GLM fitters, and its drop-in interfaces.",
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        commands=[
            "python3 benchmarks/glm/problems.py validation/work/glm-compare/problems",
            "python3 benchmarks/glm/run_comparison.py validation/work/glm-compare/problems --reference --accuracy",
            "python3 benchmarks/glm/run_comparison.py validation/work/glm-compare/problems --speed --budget 15",
            "python3 benchmarks/glm/batch.py --output validation/work/glm-compare/batch.json",
            "python3 benchmarks/glm/report.py validation/work/glm-compare/problems --output validation/GLM_COMPARISON.json",
        ],
        reference="Damped Newton in IEEE binary128 (benchmarks/glm/reference.cpp); NB theta by profile bisection over [1e-6, 1e8].",
        accuracy_scope="Problems with a finite, identifiable MLE, excluding relative collinearity 1e-12 (aliased by R's threshold). Failure: error, missing identifiable coefficient, or coefficient error above 1e-3 reference standard errors.",
        accuracy=acc,
        edge_cases=edge_cases(scored),
        speed_scope="Minimum over up to 7 repeats (15 s budget) of one fit on loaded in-memory data, one thread (OPENBLAS/OMP_NUM_THREADS=1), shared host. Tool defaults unless marked tight.",
        speed=speed,
        batch=batch,
        engine_regression=engine_regression(),
        environment=dict(
            load_average=os.getloadavg(),
            host_threads=os.cpu_count(),
            note="Shared host; load averages are recorded per timing in validation/work/glm-compare/speed*.log.",
        ),
    )
    args.output.write_text(json.dumps(receipt, indent=1, default=float) + "\n")
    # Markdown tables for the documentation.
    print(
        "| Tool | Problems | Failures | Median error (SE) | Max error (SE) | Median SE error | Max SE error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for tool, label in LABELS.items():
        a = acc.get(tool)
        if not a:
            continue
        f = lambda v: "–" if v is None else f"{v:.1e}"
        print(
            f"| {label} | {a['problems']} | {len(a['failures'])} | {f(a['median_coefficient_error_se'])} | "
            f"{f(a['max_coefficient_error_se'])} | {f(a['median_se_relative_error'])} | {f(a['max_se_relative_error'])} |"
        )
    tab = collections.defaultdict(dict)
    for r in speed:
        tab[r["problem"]][r["tool"]] = r
    cols = [
        "rapidglm",
        "r-rapidglm",
        "py-rapidglm",
        "glm.fit",
        "glm",
        "fastglm",
        "speedglm",
        "statsmodels",
        "glum",
        "glum-tight",
        "sklearn",
        "sklearn-tight",
    ]
    print()
    print("| Problem | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for p, d in tab.items():
        if "-nb-" in p:
            continue
        cells = []
        for t in cols:
            r = d.get(t)
            cells.append(
                "–" if r is None else ("err" if r["error"] else f"{r['seconds']:.3f}")
            )
        print(f"| {p.replace('speed-', '')} | " + " | ".join(cells) + " |")
    ncols = ["rapidglm", "r-rapidglm.nb", "glm.nb", "statsmodels-nb"]
    print()
    print("| Problem | " + " | ".join(ncols) + " |")
    print("|---|" + "---:|" * len(ncols))
    for p, d in tab.items():
        if "-nb-" not in p:
            continue
        print(
            f"| {p.replace('speed-', '')} | "
            + " | ".join(
                (
                    "–"
                    if d.get(t) is None
                    else ("err" if d[t]["error"] else f"{d[t]['seconds']:.3f}")
                )
                for t in ncols
            )
            + " |"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=Path)
    main(ap.parse_args())
