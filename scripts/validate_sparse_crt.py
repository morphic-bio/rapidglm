#!/usr/bin/env python3
"""Evaluate the sparse known-propensity CRT calibration (thinned and direct draws).

Uses the G3 bound formulas with a 0.05/8 allocation over four families x two
draw methods. This is supplementary to, and does not replace, CALIBRATION.json.
"""

import argparse, json, pathlib
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("table", type=pathlib.Path)
ap.add_argument("--output", type=pathlib.Path, required=True)
args = ap.parse_args()
rows = np.genfromtxt(args.table, names=True, delimiter="\t")
results = []
for family in range(4):
    data = rows[rows["family"] == family]
    for method in ("thinned", "direct"):
        p = data["p_" + method]
        p = p[np.isfinite(p)]
        n = len(p)
        rejections = int(np.sum(p <= 0.05))
        interval = stats.binom.interval(1 - 0.05 / 8, n, 0.05)
        ks = stats.kstest(p, "uniform").statistic
        bound = float(np.sqrt(-0.5 * np.log(0.05 / (2 * 8)) / n))
        results.append(
            dict(
                family=["gaussian", "poisson", "binomial", "nb"][family],
                crt_draw=method,
                datasets=int(len(data)),
                valid=n,
                mean_treated=float(np.mean(data["treated"])),
                rejections_05=rejections,
                simultaneous_binomial_bounds=[int(v) for v in interval],
                ks_statistic=float(ks),
                ks_dkw_bound=bound,
                within_mc_bounds=bool(
                    interval[0] <= rejections <= interval[1] and ks <= bound
                ),
            )
        )
receipt = dict(
    scope="n=2000 cells, assignment probability logistic(-4 + 0.5 z) (about 2%), 499 empirical draws, identical datasets for both methods",
    results=results,
    all_within_mc_bounds=all(r["within_mc_bounds"] for r in results),
)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
for r in results:
    print(
        r["family"],
        r["crt_draw"],
        r["valid"],
        r["rejections_05"],
        r["simultaneous_binomial_bounds"],
        round(r["ks_statistic"], 4),
        round(r["ks_dkw_bound"], 4),
        r["within_mc_bounds"],
    )
