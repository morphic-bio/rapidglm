#!/usr/bin/env python3
"""Why bulk_de and DESeq2 1.48.2 set aside different genes as Cook's outliers on the
128-sample GTEx thyroid set, from black-box outputs.

DESeq2 is run through its public functions (scripts/reference_cooks_flags.R), keeping its
per-sample Cook's distances before and after outlier replacement (assays cooks and
replaceCooks) and the replaced counts. bulk_de runs with its defaults (legacy_cooks). The
receipt records:
- whether DESeq2's flags follow from its own distances under RapidGLM's rule (largest
  distance over samples in design cells of 3 to 6, against the cutoff) and under the
  three-larger-counts exception of legacy_cooks;
- whether DESeq2's distances change when counts are replaced;
- each gene set aside by one tool only, with the reason: a design level whose counts are all
  zero (the fit has a coefficient without a finite estimate, and bulk_de computes no
  distances for it); counts replaced in a cell of 7 or more samples (bulk_de flags on the
  distances of the refit); or distances on either side of the cutoff.
"""

import argparse, json, os, pathlib, subprocess
from collections import Counter
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import f

ROOT = pathlib.Path(__file__).resolve().parents[1]
R_LIBS = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}


def main(args):
    data, work = args.data.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(data / "design.tsv", sep="\t")
    coefficient = "sex[T.female]"
    index = list(design.columns).index(coefficient)
    ref = work / "deseq2"
    if not (ref / "cooks.tsv").exists():
        subprocess.run(["Rscript", str(ROOT / "scripts/reference_cooks_flags.R"), str(data), coefficient, R_LIBS,
                        str(ref)], check=True, env=ENV)
    prefix = work / "rapidglm"
    config = dict(responses=str(data / "counts.mtx"), design=str(data / "x.tsv"), coefficient=index,
                  threads=args.threads, output=str(prefix))
    (work / "rapidglm.config.json").write_text(json.dumps(config))
    subprocess.run([str(args.binary), "bulk-de", str(work / "rapidglm.config.json")], check=True, env=ENV,
                   stdout=subprocess.DEVNULL)
    ours = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")

    X = design.to_numpy(float)
    m, p = X.shape
    Y = np.asarray(mmread(str(data / "counts.mtx")).todense())
    cells = Counter(map(tuple, X))
    reps = np.array([cells[tuple(r)] for r in X])
    small = (reps >= 3) & (reps < 7)
    cutoff = f.ppf(0.99, p, m - p)
    genes = pd.read_csv(ref / "genes.tsv", sep="\t")
    cooks = np.loadtxt(ref / "cooks.tsv")
    after = np.loadtxt(ref / "replaceCooks.tsv")
    replaced_counts = np.loadtxt(ref / "replaceCounts.tsv")

    def largest(row, mask):
        v = row[mask]
        return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")

    theirs = set(np.flatnonzero(genes.pvalue.isna().to_numpy() & (genes.baseMean.to_numpy() > 0)))
    ours_flag = set(np.flatnonzero((ours.status == "cooks_outlier").to_numpy()))
    rule = {g for g in range(len(genes)) if largest(cooks[g], small) > cutoff}
    exception = {g for g in rule if (Y[g] > Y[g][int(np.nanargmax(cooks[g]))]).sum() < 3}
    changed_deseq2 = set(np.flatnonzero((replaced_counts != Y).any(axis=1)))

    def zero_level(g):
        return [design.columns[c] for c in range(1, p)
                if set(np.unique(X[:, c])) <= {0, 1} and (X[:, c] == 1).any() and (Y[g][X[:, c] == 1] == 0).all()]

    def reason(g):
        if zero_level(g):
            return ("design level with all-zero counts: no finite bulk_de distance"
                    if not np.isfinite(ours.maxCooks[g]) else
                    "design level with all-zero counts: floored fit distances differ")
        if g in changed_deseq2 or ours.replaced[g]:
            return "counts replaced: bulk_de flags on the refit, DESeq2 on the fit before replacement"
        return "distances on either side of the cutoff"

    def detail(g):
        return {"gene": int(g), "base_mean": float(genes.baseMean[g]), "reason": reason(g),
                "all_zero_levels": zero_level(g),
                "deseq2_max_cooks_over_cutoff": largest(cooks[g], small) / cutoff,
                "rapidglm_max_cooks_over_cutoff": float(ours.maxCooks[g] / cutoff),
                "replaced": {"deseq2": g in changed_deseq2, "rapidglm": bool(ours.replaced[g])},
                "dispersion_rapidglm": float(ours.dispersion[g])}

    only_theirs = [detail(g) for g in sorted(theirs - ours_flag)]
    only_ours = [detail(g) for g in sorted(ours_flag - theirs)]
    version = subprocess.run([str(args.binary), "--version"], capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    receipt = {
        "purpose": __doc__.split("\n\n")[0].replace("\n", " "),
        "command": "python3 scripts/validate_cooks_flags.py --output validation/COOKS_FLAGS.json",
        "version": version, "commit": commit,
        "dataset": "thyroid128 (validation/work/benchmark/thyroid128), coefficient sex[T.female]",
        "cutoff": cutoff,
        "samples_by_cell_size": {"1-2": int((reps < 3).sum()), "3-6": int(small.sum()), "7+": int((reps >= 7).sum())},
        "deseq2": {
            "flagged": len(theirs),
            "genes_with_replaced_counts": len(changed_deseq2),
            "distances_unchanged_by_replacement": bool(np.allclose(cooks, after, equal_nan=True)),
            "flags_from_own_distances_cells_3_to_6": {"genes": len(rule), "equal": rule == theirs},
            "with_three_larger_counts_exception": {"genes": len(exception), "equal": exception == theirs},
        },
        "rapidglm": {"flagged": len(ours_flag), "genes_with_replaced_counts": int(ours.replaced.sum())},
        "flagged_by_both": len(theirs & ours_flag),
        "deseq2_only": {"genes": len(only_theirs), "by_reason": dict(Counter(d["reason"] for d in only_theirs)),
                        "detail": only_theirs},
        "rapidglm_only": {"genes": len(only_ours), "by_reason": dict(Counter(d["reason"] for d in only_ours)),
                          "detail": only_ours},
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("deseq2", "rapidglm", "flagged_by_both")}, indent=1))
    print("DESeq2 only:", receipt["deseq2_only"]["by_reason"])
    print("RapidGLM only:", receipt["rapidglm_only"]["by_reason"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--data", type=pathlib.Path, default=ROOT / "validation/work/benchmark/thyroid128")
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "validation/work/cooks_flags")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    main(ap.parse_args())
