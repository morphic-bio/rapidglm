#!/usr/bin/env python3
"""Cook's-distance receipts for bulk DE against DESeq2 1.48.2 (black box).

COOKS_THYROID.json: agreement on GTEx thyroid (128 and 647 samples) with
Cook's-distance filtering disabled in both tools.
COOKS_LEGACY.json: agreement by default (legacy_cooks: DESeq2's robust
dispersion as implemented in PyDESeq2) and with RapidGLM's own estimator
(legacy_cooks false, the default before 1.2.0), on the validation datasets of
scripts/validate_bulk.py and on the two thyroid sets.

DESeq2's tables come from earlier runs: validation/work/bulk/<dataset>/deseq2
(scripts/validate_bulk.py) and, for the thyroid sets, the application
benchmark (<runs>/<dataset>/deseq2-t1, scripts/benchmark_applications.py).
DESeq2 without the Cook's cutoff is run here (scripts/reference_bulk_nocooks.R)
and kept in the work directory.

The findings on DESeq2's robust dispersion (alpha_rob), which come from DESeq2's
outputs alone, are carried over from the receipts being replaced.
"""

import argparse, json, os, pathlib, subprocess
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
BENCH = ROOT / "validation/work/benchmark"
R_LIBS = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"


def bulk_de(binary, data, index, prefix, threads, **extra):
    config = dict(responses=str(data / "counts.mtx"), design=str(data / "x.tsv"), coefficient=index,
                  threads=threads, output=str(prefix), **extra)
    path = pathlib.Path(str(prefix) + ".json")
    path.write_text(json.dumps(config))
    subprocess.run([str(binary), "bulk-de", str(path)], check=True, env=ENV, stdout=subprocess.DEVNULL)
    return pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")


def calls(table):
    return set(np.flatnonzero(np.nan_to_num(table.padj.to_numpy(float), nan=1) < 0.1))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name in ("airway", "kang", "sim3", "sim3b", "sim8"):
        d = ROOT / "validation/work/bulk" / name
        header = (d / "design.tsv").read_text().splitlines()[0].split("\t")
        datasets[name] = (d, header[-1], len(header) - 1, d / "deseq2/genes.tsv")
    for name in ("thyroid128", "thyroid"):
        d = BENCH / name
        header = (d / "design.tsv").read_text().splitlines()[0].split("\t")
        datasets[name] = (d, "sex[T.female]", header.index("sex[T.female]"),
                          args.runs / name / "deseq2-t1/genes.tsv")
    version = subprocess.run([str(args.binary), "--version"], capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    legacy = {}
    for name, (d, _, index, reference) in datasets.items():
        ref = calls(pd.read_csv(reference, sep="\t"))
        row = {"deseq2_discoveries": len(ref)}
        for key, extra in (("default", {}), ("own_estimator", {"legacy_cooks": False})):
            t = bulk_de(args.binary, d, index, work / f"{name}.{key}", args.threads, **extra)
            run = json.loads((work / f"{name}.{key}.run.json").read_text())
            assert run["cooks_dispersion"] == ("rapidglm" if extra else "legacy_pydeseq2")
            row[key] = {"discoveries": len(calls(t)), "jaccard": jaccard(calls(t), ref),
                        "cooks_flagged": int((t.status == "cooks_outlier").sum()),
                        "replaced_genes": run["replaced_genes"]}
        legacy[name] = row
        print(name, row, flush=True)
    receipt = {
        "purpose": "bulk_de against DESeq2 1.48.2 with its default Cook's distances (legacy_cooks: DESeq2's robust "
                   "dispersion as implemented in PyDESeq2, MIT) and with RapidGLM's own estimator "
                   "(legacy_cooks false, the default before 1.2.0)",
        "command": "python3 scripts/validate_cooks.py --thyroid-output validation/COOKS_THYROID.json "
                   "--legacy-output validation/COOKS_LEGACY.json",
        "version": version, "commit": commit,
        "datasets": legacy,
    }
    if args.legacy_output.exists():
        old = json.loads(args.legacy_output.read_text())
        if "alpha_rob_check" in old:
            receipt["alpha_rob_check"] = {**old["alpha_rob_check"],
                                          "carried_over": "analysis of 2026-10-01, not recomputed"}
    args.legacy_output.write_text(json.dumps(receipt, indent=2) + "\n")

    thyroid = {}
    for name in ("thyroid128", "thyroid"):
        d, coefficient, index, _ = datasets[name]
        reference = work / f"deseq2_nocooks_{name}.tsv"
        if not reference.exists():
            subprocess.run(["Rscript", str(ROOT / "scripts/reference_bulk_nocooks.R"), str(d), coefficient, R_LIBS,
                            str(reference), str(args.threads)], check=True, env=ENV, stdout=subprocess.DEVNULL)
        b = pd.read_csv(reference, sep="\t")
        a = bulk_de(args.binary, d, index, work / f"{name}.nocooks", args.threads, cooks_filter=False)
        ok = a.pvalue.notna() & b.pvalue.notna()
        thyroid[name] = {"discoveries": [len(calls(a)), len(calls(b))], "jaccard": jaccard(calls(a), calls(b)),
                         "pvalue_spearman": float(a.pvalue[ok].corr(b.pvalue[ok], method="spearman"))}
        print(name, "no Cook's filtering", thyroid[name], flush=True)
    receipt = {
        "purpose": "Bulk DE agreement with DESeq2 1.48.2 on GTEx thyroid with Cook's-distance filtering disabled "
                   "in both tools (docs/BULK_DE.md, larger multi-factor designs)",
        "commands": {"rapidglm": "rapidglm bulk-de with cooks_filter=false, coefficient sex[T.female] "
                                 "(scripts/validate_cooks.py)",
                     "deseq2": "DESeq(full = design, betaPrior = FALSE) then results(name = 'sex.T.female.', "
                               "cooksCutoff = FALSE) (scripts/reference_bulk_nocooks.R)"},
        "version": version, "commit": commit,
        "datasets": thyroid,
    }
    if args.thyroid_output.exists():
        old = json.loads(args.thyroid_output.read_text())
        if "alpha_rob_black_box" in old:
            block = {**old["alpha_rob_black_box"], "carried_over": "analysis of 2026-10-01, not recomputed"}
            if "flags_thyroid128" in block:
                flags = dict(block["flags_thyroid128"])
                flags.pop("rapidglm", None)
                block["flags_thyroid128"] = {**flags,
                                             "rapidglm_default": legacy["thyroid128"]["default"]["cooks_flagged"],
                                             "rapidglm_own_estimator":
                                                 legacy["thyroid128"]["own_estimator"]["cooks_flagged"]}
            receipt["alpha_rob_black_box"] = block
    args.thyroid_output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "validation/work/cooks")
    ap.add_argument("--runs", type=pathlib.Path, default=BENCH / "runs",
                    help="work directory of scripts/benchmark_applications.py (DESeq2's thyroid tables)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--thyroid-output", type=pathlib.Path, required=True)
    ap.add_argument("--legacy-output", type=pathlib.Path, required=True)
    main(ap.parse_args())
