#!/usr/bin/env python3
"""The K562 essential low-MOI pilot on the panel of scripts/prepare_replogle.py.

  sc         `rapidglm sceptre` at low MOI (non-targeting controls, permutations)
             on the single-cell panel: every listed discovery, positive and
             negative pair; --fit-check ks accepts the skew-normal fit as SCEPTRE
             does, without RapidGLM's confirmation of small p-values.
  sceptre    SCEPTRE 0.99.0 at low MOI on the same panel, as a black box
             (scripts/reference_sceptre_low_moi.R).
  pseudobulk for each drawn target, the pseudobulk samples of its cells and of
             the non-targeting cells in each lane (design ~ lane + perturbed),
             tested by `rapidglm bulk-de` and DESeq2 (scripts/reference_bulk.R),
             and by `rapidglm edger-ql` and edgeR's QL workflow
             (scripts/reference_edger.R).
  compare    agreement of the two single-cell analyses and of the two
             pseudobulk analyses, and the timings, into --summary.

Every stage pins its processes to --cpus; take the timing lock around timed
stages. Both single-cell analyses use the formula --formula over covariates.tsv
(the CLI's factor() names a categorical column; SCEPTRE gets batch as a factor).
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The R library holding SCEPTRE 0.99.0 (set SCEPTRE_LIBRARY when it is not the default one).
SCEPTRE_LIBRARY = os.environ.get("SCEPTRE_LIBRARY", str(ROOT / "validation/local-data/R_library"))
R_LIBRARY = os.environ.get("DESEQ2_LIBRARY", str(ROOT / "validation/local-data/R_library"))
EDGER_LIBRARY = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def timed(command, cpus):
    """Runs `command` pinned to `cpus` under /usr/bin/time; wall seconds, CPU
    seconds and peak RSS (GB)."""
    start = time.time()
    r = subprocess.run(["/usr/bin/time", "-v", "taskset", "-c", cpus, *command], capture_output=True,
                       text=True, env=ENV)
    wall = time.time() - start
    if r.returncode:
        raise SystemExit(f"{command[:2]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr).group(1))
    cpu = sum(float(re.search(rf"{k} time \(seconds\): (\S+)", r.stderr).group(1)) for k in ("User", "System"))
    return {"wall_seconds": round(wall, 2), "cpu_seconds": round(cpu, 1), "peak_rss_gb": round(rss / 1e6, 2),
            "load_average": round(os.getloadavg()[0], 2)}


def cli_formula(formula):
    return re.sub(r"\bbatch\b", "factor(batch)", formula)


def sc_name(args):
    return f"rapidglm_t{args.threads}" + ("" if args.fit_check == "strict" else f"_{args.fit_check}")


def run_sc(args):
    panel = args.pilot / "sc"
    out = args.pilot / "runs" / sc_name(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "responses": str(panel / "responses.mtx"), "assignments": str(panel / "assignments.mtx"),
        "guides": str(panel / "guides.tsv"), "pairs": str(panel / "pairs.tsv"),
        "covariates": str(panel / "covariates.tsv"), "formula": cli_formula(args.formula),
        "response_ids": str(panel / "responses.txt"), "moi": "low", "cell_qc": False,
        "seed": args.seed, "threads": args.threads, "fit_check": args.fit_check, "output": str(out),
    }
    path = out.parent / f"{sc_name(args)}.json"
    path.write_text(json.dumps(config, indent=1) + "\n")
    timing = timed([str(args.binary), "sceptre", str(path)], args.cpus)
    record = json.loads(pathlib.Path(str(out) + ".run.json").read_text())
    timing["workflow_wall_seconds"] = record.get("wall_seconds")
    timing["null_fits"] = record.get("null_fits")
    (out.parent / f"{sc_name(args)}.timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))


def run_sceptre(args):
    out = args.pilot / "runs" / f"sceptre_w{args.threads}"
    timing = timed(["Rscript", str(ROOT / "scripts/reference_sceptre_low_moi.R"), str(args.pilot / "sc"),
                    SCEPTRE_LIBRARY, str(out), str(args.threads), args.formula], args.cpus)
    timing.update(json.loads((out / "timing.json").read_text()))
    (out / "process.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))


def write_market(path, counts):
    """counts: genes x samples."""
    rows, cols = np.nonzero(counts)
    with open(path, "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{counts.shape[0]} {counts.shape[1]} {rows.size}\n")
        np.savetxt(out, np.column_stack([rows + 1, cols + 1, counts[rows, cols]]), fmt="%d")


def pseudobulk_dataset(pilot, target, counts, samples):
    """The samples of `target` and of the non-targeting cells, ~ lane + perturbed."""
    d = pilot / "pseudobulk" / "datasets" / target
    if (d / "counts.mtx").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    keep = samples.index[samples.group.isin([target, "non-targeting"])]
    s = samples.loc[keep]
    lanes = sorted(s.gem_group.unique())
    x = pd.DataFrame({"intercept": 1.0}, index=keep)
    for lane in lanes[1:]:
        x[f"lane{lane}"] = (s.gem_group == lane).astype(float)
    x["perturbed"] = (s.group == target).astype(float)
    write_market(d / "counts.mtx", counts[keep].T)
    x.to_csv(d / "design.tsv", sep="\t", index=False)
    x.to_csv(d / "x.tsv", sep="\t", index=False, header=False)
    s.to_csv(d / "samples.tsv", sep="\t", index=False)
    return d


def run_pseudobulk(args):
    pb = args.pilot / "pseudobulk"
    samples = pd.read_csv(pb / "samples.tsv", sep="\t")
    counts = np.load(pb / "counts.npy", mmap_mode="r")
    targets = json.loads((args.pilot / "provenance.json").read_text())["targets"]
    timings = {}
    for target in targets:
        d = pseudobulk_dataset(args.pilot, target, counts, samples)
        x = pd.read_csv(d / "design.tsv", sep="\t")
        config = d / "rapidglm.json"
        config.write_text(json.dumps({"responses": str(d / "counts.mtx"), "design": str(d / "x.tsv"),
                                      "coefficient": x.shape[1] - 1, "threads": args.threads,
                                      "output": str(d / "rapidglm")}))
        entry = {"samples": int(x.shape[0]), "columns": int(x.shape[1]),
                 "perturbed_samples": int(x.perturbed.sum()),
                 "rapidglm": timed([str(args.binary), "bulk-de", str(config)], args.cpus)}
        entry["deseq2"] = timed(["Rscript", str(ROOT / "scripts/reference_bulk.R"), str(d), "perturbed",
                                 str(d / "deseq2"), R_LIBRARY], args.cpus)
        entry["deseq2"]["deseq_seconds"] = json.loads((d / "deseq2/reference.json").read_text())["seconds"]
        ql_config = d / "edger_ql.json"
        ql_config.write_text(json.dumps({"responses": str(d / "counts.mtx"), "design": str(d / "x.tsv"),
                                         "coefficient": x.shape[1] - 1, "threads": args.threads,
                                         "output": str(d / "edger_ql")}))
        entry["rapidglm_edger_ql"] = timed([str(args.binary), "edger-ql", str(ql_config)], args.cpus)
        entry["edger"] = timed(["Rscript", str(ROOT / "scripts/reference_edger.R"), str(d), "perturbed",
                                str(d / "edger"), EDGER_LIBRARY], args.cpus)
        entry["edger"]["ql_seconds"] = json.loads((d / "edger/timing.json").read_text())["ql_seconds"]
        timings[target] = entry
        print(target, json.dumps(entry), flush=True)
    (pb / f"timing_t{args.threads}.json").write_text(json.dumps(timings, indent=1) + "\n")


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def compare(args):
    pilot = args.pilot
    summary = {"provenance": json.loads((pilot / "provenance.json").read_text())}
    responses = (pilot / "sc/responses.txt").read_text().split()
    # Single cell.
    rapid_path = pilot / "runs" / f"{sc_name(args)}.tsv"
    sceptre_dir = pilot / "runs" / f"sceptre_w{args.threads}"
    if rapid_path.exists() and (sceptre_dir / "discovery.tsv").exists():
        r = pd.read_csv(rapid_path, sep="\t", na_values="NA", keep_default_na=False)
        r["response_id"] = [responses[i] for i in r.response]
        r["kind"] = r.kind.map({0: "discovery", 1: "negative", 2: "positive"})
        s = pd.read_csv(sceptre_dir / "discovery.tsv", sep="\t")
        sc = {}
        for kind in ("discovery", "positive"):
            rk = r[(r.kind == kind) & (r.status == "ok")]
            sk = (s if kind == "discovery" else pd.read_csv(sceptre_dir / "power.tsv", sep="\t"))
            sk = sk[sk.pass_qc] if "pass_qc" in sk else sk
            m = rk.merge(sk, left_on=["response_id", "target"], right_on=["response_id", "grna_target"],
                         suffixes=("", "_sceptre"))
            m = m[np.isfinite(m.p_value_sceptre)] if "p_value_sceptre" in m else m
            entry = {"rapidglm_tested": int(len(rk)), "sceptre_tested": int(np.isfinite(sk.p_value).sum()),
                     "both": int(len(m))}
            if len(m) and kind == "discovery":
                lr, ls = -np.log10(m.p_value), -np.log10(m.p_value_sceptre)
                entry["spearman_p"] = float(pd.Series(lr).corr(pd.Series(ls), method="spearman"))
                entry["pearson_log10_p"] = float(np.corrcoef(lr, ls)[0, 1])
                hits_r = set(rk.loc[rk.adjusted_p <= 0.1, ["response_id", "target"]].itertuples(index=False))
                hits_s = set(sk.loc[sk.significant, ["response_id", "grna_target"]].itertuples(index=False)) \
                    if "significant" in sk else set()
                entry.update({"rapidglm_discoveries": len(hits_r), "sceptre_discoveries": len(hits_s),
                              "jaccard": round(jaccard(hits_r, hits_s), 4)})
                by_target = {}
                for t, g in m.groupby("target"):
                    by_target[t] = {"pairs": int(len(g)),
                                    "spearman": float(g.p_value.corr(g.p_value_sceptre, method="spearman"))}
                entry["by_target"] = by_target
            if len(m) and kind == "positive":
                entry["pairs"] = m[["target", "p_value", "p_value_sceptre", "log2fc"]].to_dict("records")
            sc[kind] = entry
        neg = r[(r.kind == "negative") & (r.status == "ok")].p_value.sort_values().to_numpy()
        if neg.size:
            sc["negative"] = {"rapidglm_tested": int(neg.size),
                              "rapidglm_fraction_below_0.05": float((neg <= 0.05).mean()),
                              "rapidglm_fraction_below_0.01": float((neg <= 0.01).mean())}
            cal = sceptre_dir / "calibration.tsv"
            if cal.exists():
                c = pd.read_csv(cal, sep="\t")
                cp = c.p_value[np.isfinite(c.p_value)]
                sc["negative"].update({"sceptre_tested": int(cp.size),
                                       "sceptre_fraction_below_0.05": float((cp <= 0.05).mean()),
                                       "sceptre_fraction_below_0.01": float((cp <= 0.01).mean())})
        for name, path in (("rapidglm", pilot / "runs" / f"{sc_name(args)}.timing.json"),
                           ("sceptre", sceptre_dir / "process.json")):
            if path.exists():
                sc[f"{name}_timing"] = json.loads(path.read_text())
        summary["single_cell"] = sc
    # Pseudobulk.
    pb = pilot / "pseudobulk"
    timing_path = pb / f"timing_t{args.threads}.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())
        rows = {}
        for target, t in timings.items():
            d = pb / "datasets" / target
            a = pd.read_csv(d / "rapidglm.tsv", sep="\t", na_values="NA", keep_default_na=False)
            b = pd.read_csv(d / "deseq2/genes.tsv", sep="\t")
            ok = np.isfinite(a.pvalue) & np.isfinite(b.pvalue)
            ra, rb = set(np.flatnonzero(a.padj <= 0.1)), set(np.flatnonzero(b.padj <= 0.1))
            ql_a = pd.read_csv(d / "edger_ql.tsv", sep="\t", na_values="NA", keep_default_na=False)
            ql_b = pd.read_csv(d / "edger/genes_ql.tsv", sep="\t")
            qa = set(ql_a.gene[ql_a.FDR <= 0.1])
            qb = set(ql_b.gene[ql_b.FDR <= 0.1])
            merged = ql_a.merge(ql_b, on="gene", suffixes=("", "_edger"))
            rows[target] = {
                "edger_ql_hits": len(qa), "edger_hits": len(qb), "edger_jaccard": round(jaccard(qa, qb), 4),
                "edger_max_abs_log10_p_difference": float(np.nanmax(np.abs(
                    np.log10(merged.PValue) - np.log10(merged.PValue_edger)))) if len(merged) else None,
                "rapidglm_edger_ql_seconds": t["rapidglm_edger_ql"]["wall_seconds"],
                "edger_seconds": t["edger"]["wall_seconds"], "edger_ql_seconds": t["edger"]["ql_seconds"],
                "samples": t["samples"], "perturbed_samples": t["perturbed_samples"],
                "rapidglm_seconds": t["rapidglm"]["wall_seconds"], "deseq2_seconds": t["deseq2"]["wall_seconds"],
                "deseq2_fit_seconds": t["deseq2"]["deseq_seconds"],
                "rapidglm_hits": len(ra), "deseq2_hits": len(rb), "jaccard": round(jaccard(ra, rb), 4),
                "max_abs_log10_p_difference": float(np.max(np.abs(np.log10(a.pvalue[ok]) - np.log10(b.pvalue[ok]))))
                if ok.any() else None,
                "max_abs_lfc_difference": float(np.nanmax(np.abs(a.log2FoldChange[ok] - b.log2FoldChange[ok])))
                if ok.any() else None,
            }
        totals = {key: sum(v[key] for v in rows.values())
                  for key in ("rapidglm_seconds", "deseq2_seconds", "deseq2_fit_seconds",
                              "rapidglm_edger_ql_seconds", "edger_seconds", "edger_ql_seconds")}
        summary["pseudobulk"] = {"targets": rows, "totals": totals}
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=1)[:6000])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["sc", "sceptre", "pseudobulk", "compare"])
    parser.add_argument("--pilot", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    parser.add_argument("--cpus", default="16-31")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--formula", default="~ log_umi + log_detected + mito + batch")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--fit-check", default="strict", choices=["strict", "ks"],
                        help="ks: SCEPTRE's acceptance of the skew-normal fit (the like-for-like run)")
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()
    args.pilot = args.pilot.resolve()
    {"sc": run_sc, "sceptre": run_sceptre, "pseudobulk": run_pseudobulk, "compare": compare}[args.stage](args)


if __name__ == "__main__":
    main()
