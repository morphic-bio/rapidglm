#!/usr/bin/env python3
"""How much the agreement with SCEPTRE on the Gasperini benchmark turns on the
seed (validation/PERTURBATION_SEEDS.json).

The application benchmark (scripts/benchmark_applications.py) compares one
RapidGLM run with SCEPTRE 0.99.0. Both are Monte Carlo tests, so the overlap of
their discoveries depends on RapidGLM's draws. This script repeats the full
screen's run (207,324 cells; fit_check ks, as in the comparison) under several
seeds and reports, over the discovery pairs both tools tested, each run's BH
(0.1) discoveries and Jaccard index with SCEPTRE's, and the Jaccard indices
between RapidGLM's runs: the floor that two runs of one method reach.

    python3 scripts/validate_perturbation_seeds.py --runs validation/work/benchmark/runs \\
        --work DIR --output validation/PERTURBATION_SEEDS.json
"""

import argparse
import itertools
import json
import os
import pathlib
import subprocess

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    q = np.empty(len(p))
    previous = 1.0
    for k in range(len(p), 0, -1):
        previous = min(previous, p[order[k - 1]] * len(p) / k)
        q[order[k - 1]] = previous
    return q


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.runs / args.dataset / "rapidglm-cli-ks-t8/config.json").read_text())
    base.update(args.option)
    reference = pd.read_csv(args.runs / args.dataset / "sceptre-t8/discovery.tsv", sep="\t")
    reference = reference[reference.pass_qc.astype(str).str.upper() == "TRUE"]
    called = reference.significant.astype(str).str.upper() == "TRUE"
    reference_calls = set(zip(reference.response_id[called], reference.grna_target[called]))
    responses = (ROOT / "validation/work/benchmark" / args.dataset / "responses.txt").read_text().split()
    seeds = [base["seed"]] + [s for s in args.seeds if s != base["seed"]]
    calls, rows = {}, []
    for seed in seeds:
        prefix = work / f"seed{seed}"
        if not pathlib.Path(f"{prefix}.tsv").exists():
            config = dict(base, seed=seed, threads=args.threads, output=str(prefix))
            pathlib.Path(f"{prefix}.json").write_text(json.dumps(config))
            subprocess.run([str(args.binary), "sceptre", f"{prefix}.json"], check=True,
                           env={**os.environ, "OPENBLAS_NUM_THREADS": "1"}, stdout=subprocess.DEVNULL)
        table = pd.read_csv(f"{prefix}.tsv", sep="\t", na_values="NA", keep_default_na=False)
        table = table[table.kind.astype(str).isin(["0", "discovery"]) & table.p_value.notna()].copy()
        table["response_id"] = [responses[i] for i in table.response]
        both = table.merge(reference[["response_id", "grna_target"]], left_on=["response_id", "target"],
                           right_on=["response_id", "grna_target"])
        q = bh(both.p_value.to_numpy())
        calls[seed] = set(zip(both.response_id[q < 0.1], both.target[q < 0.1]))
        rows.append({"seed": seed, "pairs_tested_by_both": int(len(both)), "discoveries": len(calls[seed]),
                     "jaccard_with_sceptre": jaccard(calls[seed], reference_calls)})
        print(rows[-1], flush=True)
    between = [jaccard(calls[a], calls[b]) for a, b in itertools.combinations(seeds, 2)]
    version = subprocess.run([str(args.binary), "--version"], capture_output=True, text=True).stdout.strip()
    receipt = {
        "purpose": "Agreement of the perturbation workflow with SCEPTRE 0.99.0 on the Gasperini benchmark "
                   f"({args.dataset}) under several seeds, and between the seeds",
        "dataset": args.dataset,
        "options": args.option,
        "command": "python3 scripts/validate_perturbation_seeds.py --runs RUNS --work DIR --output OUT",
        "version": version,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "configuration": "the benchmark's run with fit_check ks (scripts/benchmark_applications.py, tool "
                         "rapidglm-cli-ks), with its seed and with the others; discoveries are BH at 0.1 over "
                         "the discovery pairs both tools tested",
        "sceptre_discoveries": len(reference_calls),
        "runs": rows,
        "jaccard_with_sceptre": {"min": min(r["jaccard_with_sceptre"] for r in rows),
                                 "mean": float(np.mean([r["jaccard_with_sceptre"] for r in rows])),
                                 "max": max(r["jaccard_with_sceptre"] for r in rows)},
        "jaccard_between_seeds": {"min": min(between), "mean": float(np.mean(between)), "max": max(between),
                                  "pairs_of_seeds": len(between)},
    }
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--runs", type=pathlib.Path, required=True,
                    help="work directory of scripts/benchmark_applications.py")
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", default="gasperini", help="benchmark dataset: gasperini or gasperini50k")
    ap.add_argument("--option", default={}, metavar="JSON",
                    type=json.loads, help='options added to the run\'s configuration, e.g. \'{"reuse_draws": true}\'')
    main(ap.parse_args())
