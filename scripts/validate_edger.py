#!/usr/bin/env python3
"""Agreement of `rapidglm edger-ql` with edgeR 4.6.2's default quasi-likelihood
workflow (filterByExpr, normLibSizes, glmQLFit, glmQLFTest; the reference tables
of scripts/reference_edger.R in validation/work/edger/<dataset>/).

For each dataset: the genes kept, TMM factors, average log CPM, the NB
dispersion, the average quasi-dispersion, the prior df, quasi-dispersions,
p-values and the discoveries at FDR 0.1, and the timing of the CLI. p-values are
compared over the genes whose edgeR fits are consistent (fit_check.tsv from
scripts/reference_edger_fits.R): for a few genes edgeR's glmFit reports a
deviance that is not that of its fitted values, and its F statistic uses it.

    python3 scripts/validate_edger.py --binary build/rapidglm --output validation/EDGER_QL.json
"""

import argparse
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "validation/work"
DATASETS = {
    "airway": "bulk/airway", "kang": "bulk/kang", "sim3": "bulk/sim3", "sim3b": "bulk/sim3b",
    "sim8": "bulk/sim8", "thyroid32": "benchmark/thyroid32", "thyroid128": "benchmark/thyroid128",
    "thyroid": "benchmark/thyroid",
}
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def run(binary, data, out, threads, cpus):
    out.parent.mkdir(parents=True, exist_ok=True)
    config = out.with_suffix(".json")
    config.write_text(json.dumps({"responses": str(data / "counts.mtx"), "design": str(data / "x.tsv"),
                                  "threads": threads, "output": str(out)}))
    command = ([] if cpus is None else ["taskset", "-c", cpus]) + [str(binary), "edger-ql", str(config)]
    start = time.perf_counter()
    subprocess.run(command, check=True, env=ENV, stdout=subprocess.DEVNULL)
    return time.perf_counter() - start


def bh_hits(p, level=0.1):
    """Index labels of the BH discoveries at `level`."""
    p = p.dropna().sort_values()
    m = len(p)
    passed = np.flatnonzero(p.to_numpy() <= level * np.arange(1, m + 1) / m)
    return set(p.index[: passed[-1] + 1]) if passed.size else set()


def compare(name, out):
    ref_dir = WORK / "edger" / name
    reference = json.loads((ref_dir / "reference.json").read_text())
    record = json.loads(pathlib.Path(str(out) + ".run.json").read_text())
    ours = pd.read_csv(str(out) + ".tsv", sep="\t", na_values="NA", keep_default_na=False)
    filt = pd.read_csv(ref_dir / "filter.tsv", sep="\t")
    theirs = pd.read_csv(ref_dir / "genes_ql.tsv", sep="\t").set_index("gene")
    samples = pd.read_csv(ref_dir / "samples.tsv", sep="\t")
    ours_samples = pd.read_csv(str(out) + ".samples.tsv", sep="\t")
    kept = ours.status != "filtered"
    both = ours[kept].set_index("gene").join(theirs, rsuffix="_edger", how="inner")
    rel = lambda a, b: float(np.nanmax(np.abs(a / b - 1)))
    # Genes whose edgeR fits are inconsistent (scripts/reference_edger_fits.R): a
    # fit flagged as failed, or a reported deviance that is not the deviance of its
    # fitted values, for the full or the reduced design. p-values are compared
    # over the other genes.
    inconsistent = set()
    check_path = ref_dir / "fit_check.tsv"
    if check_path.exists():
        check = pd.read_csv(check_path, sep="\t")
        flag = np.zeros(len(check), dtype=bool)
        for design in ("full", "reduced"):
            if f"{design}_failed" in check:
                flag |= check[f"{design}_failed"].astype(str).str.upper() == "TRUE"
            reported, fitted = check[f"{design}_reported"], check[f"{design}_at_fitted"]
            flag |= ((reported - fitted).abs() > 1e-6 * (1 + fitted.abs())).to_numpy()
        inconsistent = set(check.gene[flag])
    same = both[~both.index.isin(inconsistent)]
    lp, le = np.log10(same.PValue), np.log10(same.PValue_edger)
    hits_ours = set(ours.gene[ours.FDR <= 0.1])
    hits_edger = set(theirs.index[theirs.FDR <= 0.1])
    ql = reference["ql"]
    return {
        "genes": int(len(ours)), "kept": int(kept.sum()), "kept_edger": int(filt.kept.sum()),
        "filter_mismatches": int((kept.to_numpy() != filt.kept.astype(bool).to_numpy()).sum()),
        "norm_factor_max_rel": rel(ours_samples.norm_factor.to_numpy(), samples.norm_factor.to_numpy()),
        "ave_log_cpm_max_abs": float(np.max(np.abs(both.AveLogCPM - both.AveLogCPM_edger))),
        "dispersion": record["dispersion"], "dispersion_edger": ql["dispersion"],
        "top_proportion": record["top_proportion"], "top_proportion_edger": ql["top_proportion"],
        "average_ql_dispersion": record["average_ql_dispersion"],
        "average_ql_dispersion_edger": ql["average_ql_dispersion"],
        "df_prior": record["df_prior"], "df_prior_edger": ql["df_prior"],
        "edger_fits_inconsistent": len(inconsistent),
        "s2_median_rel": float(np.nanmedian(np.abs(both.s2 / both.s2_edger - 1))),
        "s2_post_max_rel": rel(both.s2_post.to_numpy(), both.s2_post_edger.to_numpy()),
        "s2_prior_max_rel": rel(both.s2_prior.to_numpy(), both.s2_prior_edger.to_numpy()),
        "log10_p_max_abs": float(np.nanmax(np.abs(lp - le))),
        "log10_p_median_abs": float(np.nanmedian(np.abs(lp - le))),
        "logfc_max_abs": float(np.nanmax(np.abs(both.logFC - both.logFC_edger))),
        "discoveries": len(hits_ours), "discoveries_edger": len(hits_edger),
        "discoveries_shared": len(hits_ours & hits_edger),
        # BH within the genes whose edgeR fits are consistent, for both tools.
        "discoveries_consistent_genes": len(bh_hits(same.PValue)),
        "discoveries_edger_consistent_genes": len(bh_hits(same.PValue_edger)),
        "discoveries_shared_consistent_genes": len(bh_hits(same.PValue) & bh_hits(same.PValue_edger)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", default=None, help="taskset CPU list for the runs")
    parser.add_argument("--work", type=pathlib.Path, default=WORK / "edger/rapidglm")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    results = {}
    for name in args.datasets:
        out = args.work / name / f"ql_t{args.threads}"
        seconds = run(args.binary, WORK / DATASETS[name], out, args.threads, args.cpus)
        results[name] = compare(name, out)
        results[name]["seconds"] = round(seconds, 3)
        print(name, json.dumps(results[name]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
