#!/usr/bin/env python3
"""Timing of the perturbation workflow when most requested pairs cannot be
tested (validation/UNTESTABLE_PAIRS.json).

A synthetic screen with many targets of few cells each and many sparse
features: 20,000 cells, 6,000 genes and 40,000 peaks (most nonzero in 0.2-3% of
cells), 300 targets and 10 non-targeting groups of two guides each, about 1.5
guides per cell at high MOI and at most one at low MOI. Every target and
non-targeting group against every feature is 1.86 million gene pairs and 12.4
million peak pairs, of which about 5% pass pair QC (seven cells with a nonzero
count in each arm).

For a binary the script times `rapidglm sceptre` three ways on the same pairs:
  listed    every pair listed in a pairs file;
  testable  only the pairs that pass pair QC listed (found here by their
            definition, from the matrices);
  grid      the targets and responses named, without a list (from 1.3.0).
With the keys of 1.3.0 the tested pairs have the same records in all three,
which is checked.

    python3 scripts/benchmark_untestable_pairs.py --work DIR --binary build/rapidglm \\
        --label 1.3.0 --output validation/UNTESTABLE_PAIRS.json
    python3 scripts/benchmark_untestable_pairs.py --work DIR --binary OLD/rapidglm \\
        --label 1.2.0 --modes listed testable --output validation/UNTESTABLE_PAIRS.json

Runs of several binaries accumulate in the output file under their labels.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 20261009
N_NT = 10


def make_screen(work, n=20000, n_genes=6000, n_peaks=40000, n_targets=300):
    """Writes the screen's files into `work` (once)."""
    if (work / "screen.json").exists():
        return json.loads((work / "screen.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def counts(n_features, dense_fraction):
        # Most features are nonzero in 0.2-3% of cells; a few in 20-70%.
        dense = rng.random(n_features) < dense_fraction
        rate = np.where(dense, rng.uniform(0.2, 0.7, n_features), rng.uniform(0.002, 0.03, n_features))
        k = rng.binomial(n, rate)
        rows = np.concatenate([rng.choice(n, size=int(c), replace=False) for c in k])
        cols = np.repeat(np.arange(n_features), k)
        data = (1 + rng.poisson(1.5, size=rows.size)).astype(np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_features)), dense

    x, dense_genes = counts(n_genes, 0.03)
    y, _ = counts(n_peaks, 0.03)
    genes = [f"G{j}" for j in range(n_genes)]
    peaks = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(n_peaks)]
    # The first third of the targets are named after a gene nonzero in many cells
    # (their own gene is knocked down in their cells); the rest have no gene.
    dense_names = [genes[j] for j in np.flatnonzero(dense_genes)]
    targets = [dense_names[t] if t < min(n_targets // 3, len(dense_names)) else f"T{t}" for t in range(n_targets)]
    guide_ids, guide_target, guide_nt = [], [], []
    for t in targets:
        for r in (1, 2):
            guide_ids.append(f"{t}_g{r}"); guide_target.append(t); guide_nt.append(0)
    for g in range(N_NT):
        for r in (1, 2):
            guide_ids.append(f"NTC{g}_g{r}"); guide_target.append(f"NTC{g}"); guide_nt.append(1)
    guide_ids.append("unlisted_g1"); guide_target.append("unlisted"); guide_nt.append(0)
    # Guide abundance is uneven, so targets range from a handful of cells to a few hundred.
    weight = rng.gamma(1.0, 1.0, len(guide_ids)); weight /= weight.sum()
    per_cell = rng.poisson(1.5, n)
    high, low = [], []
    for i in range(n):
        picked = rng.choice(len(guide_ids), size=min(int(per_cell[i]), 8), replace=False, p=weight)
        high.append(sorted(int(j) for j in picked))
        low.append([int(rng.choice(len(guide_ids), p=weight))] if rng.random() < 0.9 else [])
    x = x.tolil()
    for t in targets:
        if t in dense_names:
            mine = {j for j, name in enumerate(guide_target) if name == t}
            j = genes.index(t)
            for i in range(n):
                if mine.intersection(high[i]) and x[i, j] and rng.random() < 0.7:
                    x[i, j] = 0
    x = x.tocsr(); x.eliminate_zeros()

    def write_market(path, matrix):
        m = sparse.coo_matrix(matrix)
        with open(path, "w") as out:
            out.write("%%MatrixMarket matrix coordinate integer general\n")
            out.write(f"{m.shape[0]} {m.shape[1]} {m.nnz}\n")
            np.savetxt(out, np.column_stack([m.row + 1, m.col + 1, m.data.astype(np.int64)]), fmt="%d")

    assignments = {}
    for moi, calls in (("high", high), ("low", low)):
        rows = np.concatenate([np.array(c, dtype=np.int64) for c in calls])
        cols = np.repeat(np.arange(n), [len(c) for c in calls])
        a = sparse.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(len(guide_ids), n))
        assignments[moi] = a
        write_market(work / f"assignments_{moi}.mtx", a)
    (work / "guides.tsv").write_text("id\ttarget\tnon_targeting\n" +
                                     "".join(f"{i}\t{t}\t{k}\n" for i, t, k in zip(guide_ids, guide_target, guide_nt)))
    names = targets + [f"NTC{g}" for g in range(N_NT)]
    kinds = ["discovery"] * len(targets) + ["negative"] * N_NT
    (work / "targets.tsv").write_text("target\tkind\n" + "".join(f"{t}\t{k}\n" for t, k in zip(names, kinds)))
    guide_target = np.array(guide_target)
    nt_guides = np.flatnonzero(np.array(guide_nt) == 1)
    screen = {"cells": n, "targets": len(targets), "negative_control_groups": N_NT, "guides": len(guide_ids),
              "seed": SEED, "min_nonzero": 7, "modalities": {}}
    for modality, matrix, ids in (("rna", x, genes), ("atac", y, peaks)):
        features = matrix.T.tocsr()                       # features x cells
        write_market(work / f"{modality}.mtx", features)
        (work / f"{modality}_ids.txt").write_text("\n".join(ids) + "\n")
        total = np.asarray(matrix.sum(axis=1)).ravel()
        detected = np.asarray((matrix != 0).sum(axis=1)).ravel()
        entry = {"features": len(ids), "nonzeros": int(matrix.nnz), "pairs_listed": len(names) * len(ids)}
        nonzero = (matrix != 0).astype(np.int32).tocsr()   # cells x features
        for moi in ("high", "low"):
            a = assignments[moi]
            guides_per_cell = np.asarray(a.sum(axis=0)).ravel()
            design = np.column_stack([np.ones(n), np.log1p(total), np.log1p(detected), np.log1p(guides_per_cell)])
            np.savetxt(work / f"design_{modality}_{moi}.tsv", design[:, :4 if moi == "high" else 3],
                       delimiter="\t", fmt="%.17g")
            # Pair QC by its definition. High MOI: complement controls, every cell.
            # Low MOI: cells with exactly one guide; controls are the non-targeting cells.
            kept = np.ones(n, dtype=bool) if moi == "high" else guides_per_cell == 1
            nt_cells = (np.asarray(a[nt_guides].sum(axis=0)).ravel() > 0) & kept
            kept_nonzero = np.asarray(nonzero[kept].sum(axis=0)).ravel()
            nt_nonzero = np.asarray(nonzero[nt_cells].sum(axis=0)).ravel()
            passing = 0
            with open(work / f"pairs_testable_{modality}_{moi}.tsv", "w") as out:
                out.write("response\ttarget\tkind\n")
                for name, kind in zip(names, kinds):
                    treated = (np.asarray(a[np.flatnonzero(guide_target == name)].sum(axis=0)).ravel() > 0) & kept
                    treatment = np.asarray(nonzero[treated].sum(axis=0)).ravel()
                    if moi == "high":
                        control = kept_nonzero - treatment
                    else:
                        control = nt_nonzero - np.asarray(nonzero[treated & nt_cells].sum(axis=0)).ravel()
                    ok = np.flatnonzero((treatment >= 7) & (control >= 7))
                    passing += ok.size
                    out.write("".join(f"{j}\t{name}\t{kind}\n" for j in ok))
            entry[f"pairs_passing_qc_{moi}"] = passing
        with open(work / f"pairs_listed_{modality}.tsv", "w") as out:
            out.write("response\ttarget\tkind\n")
            for name, kind in zip(names, kinds):
                out.write("".join(f"{j}\t{name}\t{kind}\n" for j in range(len(ids))))
        screen["modalities"][modality] = entry
    (work / "screen.json").write_text(json.dumps(screen, indent=1) + "\n")
    return screen


def run(binary, work, name, config, cpus):
    out = work / "runs" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {**config, "output": str(out)}
    (out.parent / f"{name}.json").write_text(json.dumps(config))
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        from quiet_run import wait_quiet
        wait_quiet(cpus, label=f"untestable {name}")
    load_before = os.getloadavg()[0]
    start = time.time()
    r = subprocess.run(["/usr/bin/time", "-v", "taskset", "-c", cpus, str(binary), "sceptre",
                        str(out.parent / f"{name}.json")], capture_output=True, text=True,
                       env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    wall = time.time() - start
    (out.parent / f"{name}.time.log").write_text(r.stderr)
    if r.returncode:
        raise SystemExit(f"{name} failed: {r.stderr[-2000:]}")
    record = json.loads(pathlib.Path(str(out) + ".run.json").read_text())
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr).group(1))
    cpu = sum(float(re.search(rf"{k} time \(seconds\): (\S+)", r.stderr).group(1)) for k in ("User", "System"))
    return {"load_before": load_before, "process_wall_seconds": round(wall, 2), "workflow_wall_seconds": round(record["wall_seconds"], 2),
            "cpu_seconds": round(cpu, 1), "peak_rss_gb": round(rss / 1e6, 2),
            "null_fits": record.get("null_fits"), "load_average": os.getloadavg()[0]}


def tested_rows(path):
    t = pd.read_csv(path, sep="\t", na_values="NA", keep_default_na=False)
    t = t[t.status != "failed_pair_qc"]
    return t.sort_values(["response", "target", "kind"]).reset_index(drop=True)


def main(args):
    work = args.work.resolve()
    screen = make_screen(work)
    version = subprocess.run([str(args.binary), "--version"], capture_output=True, text=True).stdout.strip()
    results = {}
    for moi in args.moi:
        for modality in args.modalities:
            base = {"responses": str(work / f"{modality}.mtx"), "assignments": str(work / f"assignments_{moi}.mtx"),
                    "design": str(work / f"design_{modality}_{moi}.tsv"), "guides": str(work / "guides.tsv"),
                    "moi": moi, "cell_qc": False, "threads": args.threads, "seed": 7, **args.options}
            entry = {"pairs_listed": screen["modalities"][modality]["pairs_listed"],
                     "pairs_passing_qc": screen["modalities"][modality][f"pairs_passing_qc_{moi}"]}
            tables = {}
            for mode in args.modes:
                name = f"{args.label}_{modality}_{moi}_{mode}"
                config = dict(base)
                if mode == "grid":
                    config["targets"] = str(work / "targets.tsv")
                else:
                    config["pairs"] = str(work / (f"pairs_listed_{modality}.tsv" if mode == "listed"
                                                  else f"pairs_testable_{modality}_{moi}.tsv"))
                entry[mode] = run(args.binary, work, name, config, args.cpus)
                tables[mode] = tested_rows(work / "runs" / f"{name}.tsv")
                entry[mode]["rows_tested"] = int(len(tables[mode]))
                print(name, entry[mode], flush=True)
            first = next(iter(tables))
            entry["tested_set_is_the_pairs_passing_qc"] = bool(len(tables[first]) == entry["pairs_passing_qc"])
            # Records of the tested pairs, apart from the response's row number in the output.
            same = {}
            for mode, table in tables.items():
                if mode != first:
                    same[f"{mode}_equals_{first}"] = bool(table.equals(tables[first]))
            entry["tested_records"] = same
            results[f"{modality}_{moi}"] = entry
    receipt = json.loads(args.output.read_text()) if args.output.exists() else {
        "purpose": "Time and memory of `rapidglm sceptre` when most requested pairs fail pair QC: every pair listed, "
                   "only the pairs that pass listed, and (from 1.3.0) the targets and responses named without a list",
        "command": "python3 scripts/benchmark_untestable_pairs.py --work DIR --binary BINARY --label LABEL "
                   "--output validation/UNTESTABLE_PAIRS.json",
        "columns": "workflow_wall_seconds: the run's wall time, reading the inputs and writing the table included; "
                   "peak_rss_gb: the process's peak resident memory; rows_tested: rows that passed pair QC; "
                   "tested_records: whether the rows of the tested pairs are identical between two ways of "
                   "requesting them",
        "screen": screen, "runs": {}}
    if args.host:
        receipt["host"] = args.host
    receipt["runs"][args.label] = {
        "version": version, "threads": args.threads, "cpus": args.cpus, "date": time.strftime("%Y-%m-%d"),
        "options": {"moi": "as named", "cell_qc": False, "seed": 7, **args.options,
                    "otherwise": "the binary's defaults"},
        "note": args.note, "results": results}
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--binary", type=pathlib.Path, required=True)
    ap.add_argument("--label", required=True, help="name of this binary's runs in the receipt")
    ap.add_argument("--modes", nargs="+", default=["listed", "testable", "grid"],
                    choices=["listed", "testable", "grid"])
    ap.add_argument("--moi", nargs="+", default=["high", "low"], choices=["high", "low"])
    ap.add_argument("--modalities", nargs="+", default=["rna", "atac"], choices=["rna", "atac"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cpus", default="20-31", help="CPUs the runs are confined to")
    ap.add_argument("--options", type=json.loads, default={},
                    help="further settings of every run, as a JSON object")
    ap.add_argument("--note", default="")
    ap.add_argument("--host", default="", help="a description of the host, stored in the receipt")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    main(ap.parse_args())
