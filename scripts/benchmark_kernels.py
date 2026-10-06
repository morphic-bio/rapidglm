#!/usr/bin/env python3
"""Repeat isolated kernels, optionally alternating with a baseline executable."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="build/rapidglm_benchmarks")
    ap.add_argument("--baseline")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", default="validation/work/kernel-benchmarks.json")
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("use at least two repetitions")
    binaries = {"current": args.binary}
    if args.baseline:
        binaries["baseline"] = args.baseline
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    runs = []
    for repeat in range(args.repeats):
        labels = list(binaries)
        if repeat % 2:
            labels.reverse()
        for label in labels:
            load = os.getloadavg()
            result = subprocess.run(
                [binaries[label]], env=env, check=True, text=True, capture_output=True
            )
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            for row in rows:
                for k in ("n", "p", "iterations"):
                    row[k] = int(row[k])
                for k in ("seconds", "probability", "checksum"):
                    row[k] = float(row[k])
            runs.append(dict(binary=label, repeat=repeat, load=load, rows=rows))
    groups = {}
    for run in runs:
        for row in run["rows"]:
            key = (row["kernel"], row["n"], row["p"], row["probability"])
            groups.setdefault(key, {}).setdefault(run["binary"], []).append(row)
    summary = []
    for key, groups_by_binary in groups.items():
        row = dict(zip(("kernel", "n", "p", "probability"), key))
        row["timings"] = {
            label: {
                "median_seconds": statistics.median(r["seconds"] for r in rows),
                "min_seconds": min(r["seconds"] for r in rows),
                "max_seconds": max(r["seconds"] for r in rows),
                "iterations": rows[0]["iterations"],
                "checksum": rows[0]["checksum"],
                "checksum_repeatable": len({r["checksum"] for r in rows}) == 1,
            }
            for label, rows in groups_by_binary.items()
        }
        if args.baseline:
            old, new = row["timings"]["baseline"], row["timings"]["current"]
            row["speedup"] = old["median_seconds"] / new["median_seconds"]
            row["checksum_absolute_difference"] = abs(old["checksum"] - new["checksum"])
        summary.append(row)
    record = dict(
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=os.sys.argv,
        host=platform.uname()._asdict(),
        compiler=subprocess.check_output(["c++", "--version"], text=True).splitlines()[
            0
        ],
        star_processes=subprocess.run(
            ["pgrep", "-ax", "STAR"], capture_output=True, text=True
        ).stdout,
        thread_environment={
            k: env[k]
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        binaries={
            k: {
                "path": str(Path(v).resolve()),
                "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest(),
            }
            for k, v in binaries.items()
        },
        summary=summary,
        runs=runs,
        limitations=[
            "Synthetic fixed inputs; setup excluded; not end-to-end timings.",
            "No acceptance or speedup threshold is imposed.",
            "Host load and timing ranges must be considered when interpreting small differences.",
        ],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)
    for row in summary:
        if "speedup" in row:
            print(
                f'{row["kernel"]:16} n={row["n"]:6} p={row["p"]:3} '
                f'prob={row["probability"]:4.2f}: {row["speedup"]:.3f}x'
            )


if __name__ == "__main__":
    main()
