#!/usr/bin/env python3
import argparse, json, os, pathlib, subprocess, time
import numpy as np
from scipy import stats
from validate_glm import ROOT, sha

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", type=int, default=1000)
ap.add_argument("--draws", type=int, default=499)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--crt-draw", choices=["thinned", "direct"], default="thinned")
ap.add_argument(
    "--work", type=pathlib.Path, default=ROOT / "validation/work/calibration"
)
# The historical G3 receipt is CALIBRATION.json; later runs should use a new path.
ap.add_argument(
    "--output", type=pathlib.Path, default=ROOT / "validation/CALIBRATION.json"
)
args = ap.parse_args()
work = args.work
work.mkdir(parents=True, exist_ok=True)
command = [
    "build/rapidglm_calibration",
    str(work / "null.tsv"),
    str(args.datasets),
    str(args.draws),
    str(args.threads),
    args.crt_draw,
]
start = time.monotonic()
load = os.getloadavg()
subprocess.run(command, cwd=ROOT, check=True)
elapsed = time.monotonic() - start
rows = np.genfromtxt(work / "null.tsv", names=True, delimiter="\t")
scenarios = []
for s in range(8):
    data = rows[rows["scenario"] == s]
    valid = data[data["valid"] == 1]
    p = valid["p_value"]
    n = len(p)
    rejections = int(sum(p <= 0.05))
    interval = stats.binom.interval(1 - 0.05 / 8, n, 0.05)
    ks = stats.kstest(p, "uniform")
    ks_bound = float(np.sqrt(-0.5 * np.log(0.05 / (2 * 8)) / n))
    passed = (
        n == args.datasets
        and interval[0] <= rejections <= interval[1]
        and ks.statistic <= ks_bound
    )
    scenarios.append(
        dict(
            scenario=s,
            family=["gaussian", "poisson", "binomial", "nb"][s % 4],
            law="known_propensity_crt" if s >= 4 else "permutation",
            datasets=len(data),
            valid=n,
            rejections_05=rejections,
            rejection_rate_05=rejections / n,
            simultaneous_binomial_bounds=[int(v) for v in interval],
            ks_statistic=float(ks.statistic),
            ks_dkw_bound=ks_bound,
            within_mc_bounds=bool(passed),
        )
    )
receipt = dict(
    gate="G3",
    scope="E1 independent null datasets; fixed empirical budget; 95% familywise Monte Carlo bounds across eight scenarios",
    command=command,
    crt_draw=args.crt_draw,
    seed=20260926,
    seconds=elapsed,
    host_load_start=load,
    binary_sha256=sha(ROOT / "build/rapidglm_calibration"),
    output_sha256=sha(work / "null.tsv"),
    scenarios=scenarios,
    all_within_mc_bounds=all(s["within_mc_bounds"] for s in scenarios),
    thread_equality="unit test compares entire null arrays at 1,2,8,16 threads",
)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(scenarios, indent=2))
