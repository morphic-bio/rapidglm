#!/usr/bin/env python3
"""Generate independent R references, run the native CLI, and record every difference."""

import argparse, csv, hashlib, json, os, pathlib, subprocess, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    p = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout + p.stderr


def table(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/rapidglm")
    ap.add_argument("--sceptre-library")
    ap.add_argument("--public-data")
    # Reuse R outputs already in validation/work/glm and write a separate receipt,
    # leaving GLM_REFERENCE.json as the historical record.
    ap.add_argument("--reuse-reference", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--output", type=pathlib.Path, default=ROOT / "validation/GLM_REFERENCE.json"
    )
    args = ap.parse_args()
    suffix = f"-{args.label}" if args.label else ""
    work = ROOT / "validation/work/glm"
    work.mkdir(parents=True, exist_ok=True)
    command = ["Rscript", "scripts/reference_glm.R", work]
    if args.sceptre_library:
        command.append(args.sceptre_library)
    if args.public_data:
        command.append(args.public_data)
    host = {
        "load": os.getloadavg(),
        "uptime": run(["uptime"]).strip(),
        "STAR": subprocess.run(
            ["pgrep", "-af", "STAR"], capture_output=True, text=True
        ).stdout,
    }
    if args.reuse_reference:
        reference_log = (work / "reference.log").read_text()
    else:
        reference_log = run(command)
        (work / "reference.log").write_text(reference_log)
    panels = []
    for directory in sorted(work.iterdir()):
        if not directory.is_dir():
            continue
        family = (directory / "family.txt").read_text().strip()
        config = {
            k: str(directory / (k + ext))
            for k, ext in [
                ("responses", ".mtx"),
                ("design", ".tsv"),
                ("offset", ".tsv"),
                ("weights", ".tsv"),
            ]
        }
        config.update(
            family=family, output=str(directory / f"native{suffix}"), threads=1
        )
        cfg = directory / "config.json"
        cfg.write_text(json.dumps(config, indent=2))
        run([args.binary, "fit", cfg])
        native = table(directory / f"native{suffix}.tsv")
        reference = table(directory / "reference.tsv")
        config.update(output=str(directory / f"lr{suffix}"), test="lr")
        cfg.write_text(json.dumps(config, indent=2))
        run([args.binary, "dose-trend", cfg])
        lr = table(directory / f"lr{suffix}.tsv")
        columns = [
            k for k in reference[0] if k.startswith(("beta", "se", "wald_p"))
        ] + (["theta"] if family == "nb" else [])
        errors = {}
        differences = []
        for col in (
            columns + [k for k in reference[0] if k.startswith("final_se")] + ["lr_p"]
        ):
            a = np.array(
                [
                    float(
                        r["p_value"]
                        if col == "lr_p"
                        else r[col.replace("final_se", "se")]
                    )
                    for r in (lr if col == "lr_p" else native)
                ]
            )
            b = np.array([float(r[col]) for r in reference])
            absolute = np.abs(a - b)
            relative = absolute / np.maximum(np.abs(b), 1e-12)
            errors[col] = {
                "max_absolute": float(np.nanmax(absolute)),
                "max_relative": float(np.nanmax(relative)),
            }
            for i in np.flatnonzero(relative > 1e-6):
                differences.append(
                    {
                        "feature": int(i),
                        "field": col,
                        "native": float(a[i]),
                        "reference": float(b[i]),
                        "relative_error": float(relative[i]),
                    }
                )
        statuses = {
            s: sum(r["status"] == s for r in native)
            for s in {r["status"] for r in native}
        }
        panels.append(
            {
                "panel": directory.name,
                "features": len(native),
                "statuses": statuses,
                "errors": errors,
                "differences_above_provisional_tolerance": differences,
                "native_run": json.loads(
                    (directory / f"native{suffix}.run.json").read_text()
                ),
                "R_full_and_null_seconds": float(
                    (directory / "reference_seconds.txt").read_text()
                ),
                "input_sha256": {
                    p.name: sha(p)
                    for p in directory.iterdir()
                    if p.suffix in (".mtx", ".tsv")
                    and not p.name.startswith(("native", "lr"))
                },
            }
        )
    receipt = {
        "gates": ["G1", "G2"],
        "thresholds": "provisional; author requests measured reporting",
        "commands": [
            list(map(str, command)),
            f"{args.binary} fit/dose-trend per-panel config.json",
        ],
        "host": host,
        "versions": (work / "versions.txt").read_text().splitlines(),
        "binary_sha256": sha(args.binary),
        "panels": panels,
        "reference_log": reference_log,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    for p in panels:
        print(
            p["panel"],
            p["statuses"],
            "differences:",
            len(p["differences_above_provisional_tolerance"]),
            "max beta/se error:",
            max(
                v["max_relative"]
                for k, v in p["errors"].items()
                if k.startswith(("beta", "se"))
            ),
        )


if __name__ == "__main__":
    main()
