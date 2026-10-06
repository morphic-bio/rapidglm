#!/usr/bin/env python3
"""Validation of contrasts and joint (multi-df) hypotheses in bulk-de and pseudobulk-de.

Datasets: airway (8 samples, ~ cell + dex; genes with >= 10 reads) and the Kang CD14
pseudobulk panel (16 samples, donors + stimulation, 1,000 genes), prepared by
scripts/validate_bulk.py. References run as black boxes (scripts/reference_hypotheses.R):
DESeq2 1.48.2 for bulk-de, edgeR and glmGamPoi for the pseudobulk QL test.

Hypotheses: airway cellN080611 - cellN061011 (contrast) and the three cell-line
columns jointly (3 df); Kang donor3 - donor5 (contrast) and the seven donor columns
jointly (7 df). Pseudobulk offsets are log library sizes.

Gates (set before the RapidGLM runs, after seeing that edgeR and glmGamPoi
disagree on the Kang joint test: 64 against 19 BH discoveries):
  G1 equivalence: each contrast gives the same p-values as the coefficient test in
     the reparametrized design (maximum relative difference < 1e-6) for bulk-de
     (wald, lrt) and pseudobulk-de (ql, lr, wald).
  G2 bulk-de against DESeq2: for each hypothesis, BH (0.1) discoveries agree with
     Jaccard >= 0.9 (when either has any) and p-values with Spearman >= 0.99.
  G3 pseudobulk QL against glmGamPoi and edgeR: for each hypothesis, p-value Spearman
     >= 0.95 with glmGamPoi, and the BH (0.05) discovery count within
     [0.8 x the smaller, 1.25 x the larger] of the two references.
  G4 determinism: joint tests give identical tables at 1 and 4 threads.
"""

import argparse, json, os, pathlib, subprocess
import numpy as np
import pandas as pd
from scipy import io, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
HYPOTHESES = {
    # name: (dataset, kind, spec) - spec is contrast weights or dropped columns.
    "airway_cell_contrast": ("airway", "contrast", [0, -1, 1, 0, 0]),
    "airway_cells_joint": ("airway", "lrt", [1, 2, 3]),
    "kang_donor_contrast": ("kang", "contrast", [0, 0, 0, 1, 0, -1, 0, 0, 0]),
    "kang_donors_joint": ("kang", "lrt", [1, 2, 3, 4, 5, 6, 7]),
}


def bh(p, alpha):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    return set(np.flatnonzero(stats.false_discovery_control(p) < alpha))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else None


def run(binary, command, prefix, threads=1, **config):
    config = dict(output=str(prefix), threads=threads, family="nb", **config)
    path = pathlib.Path(str(prefix) + ".json")
    path.write_text(json.dumps(config))
    subprocess.run([str(binary), command, str(path)], check=True, env=ENV, stdout=subprocess.DEVNULL)
    return pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")


def hypothesis_config(kind, spec):
    return dict(contrast=spec) if kind == "contrast" else dict(coefficients=spec)


def reparametrize(x, weights):
    """For weights e_a - e_b, the design whose column a has coefficient beta_a - beta_b."""
    a, b = int(np.flatnonzero(np.array(weights) == 1)[0]), int(np.flatnonzero(np.array(weights) == -1)[0])
    out = x.copy()
    out[:, b] = x[:, a] + x[:, b]
    return out, a


def main(args):
    work = args.work.resolve()
    out_dir = work / "hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: work / name for name in ("airway", "kang")}
    for d in data.values():
        if not (d / "counts.mtx").exists():
            raise SystemExit(f"missing {d}; run scripts/validate_bulk.py with the same --work first")
    # Pseudobulk panels: the same counts, a header-less design and log library sizes.
    panels = {}
    for name, d in data.items():
        panel = out_dir / f"{name}_panel"
        panel.mkdir(exist_ok=True)
        counts = io.mmread(str(d / "counts.mtx")).tocsr()
        (panel / "responses.mtx").write_bytes((d / "counts.mtx").read_bytes())
        np.savetxt(panel / "design.tsv", np.loadtxt(d / "x.tsv"), delimiter="\t", fmt="%.17g")
        np.savetxt(panel / "offset.tsv", np.log(np.asarray(counts.sum(axis=0)).ravel()), fmt="%.17g")
        panels[name] = panel
    results, g1, g2, g3, g4 = {}, True, True, True, True
    for name, (dataset, kind, spec) in HYPOTHESES.items():
        d, panel = data[dataset], panels[dataset]
        row = dict(dataset=dataset, kind=kind, spec=spec)
        text = ",".join(str(v) for v in spec)
        # References.
        deseq2 = out_dir / f"{name}.deseq2.tsv"
        ql_ref = out_dir / f"{name}.ql_reference.tsv"
        if not deseq2.exists():
            subprocess.run(["Rscript", str(ROOT / "scripts/reference_hypotheses.R"), "deseq2", str(d), kind, text,
                            str(deseq2), args.r_library], check=True, env=ENV, stdout=subprocess.DEVNULL)
        if not ql_ref.exists():
            subprocess.run(["Rscript", str(ROOT / "scripts/reference_hypotheses.R"), "ql", str(panel), kind, text,
                            str(ql_ref), args.ql_library], check=True, env=ENV, stdout=subprocess.DEVNULL)
        a = pd.read_csv(deseq2, sep="\t", na_values="NA")
        refs = pd.read_csv(ql_ref, sep="\t", na_values="NA")
        # bulk-de against DESeq2.
        test = "wald" if kind == "contrast" else "lrt"
        b = run(args.binary, "bulk-de", out_dir / f"{name}.bulk", responses=str(d / "counts.mtx"),
                design=str(d / "x.tsv"), test=test, **hypothesis_config(kind, spec))
        pa, pb = a.pvalue.to_numpy(float), b.pvalue.to_numpy(float)
        ok = np.isfinite(pa) & np.isfinite(pb)
        called = lambda t: set(np.flatnonzero(np.nan_to_num(t.padj.to_numpy(float), nan=1) < 0.1))
        da, db = called(a), called(b)
        row["bulk"] = dict(discoveries=[len(db), len(da)], jaccard=jaccard(da, db),
                           pvalue_spearman=float(stats.spearmanr(pa[ok], pb[ok]).statistic))
        if kind == "contrast":
            fa, fb = a.log2FoldChange.to_numpy(float), b.log2FoldChange.to_numpy(float)
            okf = np.isfinite(fa) & np.isfinite(fb)
            row["bulk"]["log2fc_median_abs_diff"] = float(np.median(np.abs(fa - fb)[okf]))
        g2 &= (row["bulk"]["jaccard"] is None or row["bulk"]["jaccard"] >= 0.9) and row["bulk"]["pvalue_spearman"] >= 0.99
        # pseudobulk QL against edgeR and glmGamPoi.
        q = run(args.binary, "pseudobulk-de", out_dir / f"{name}.ql", responses=str(panel / "responses.mtx"),
                design=str(panel / "design.tsv"), offset=str(panel / "offset.tsv"), test="ql",
                **hypothesis_config(kind, spec))
        mine = q.p_value.to_numpy(float)
        counts = dict(rapidglm=len(bh(mine, 0.05)))
        for tool in ("edgeR_ql", "glmGamPoi_ql"):
            counts[tool] = len(bh(refs[tool], 0.05))
        okq = np.isfinite(mine) & np.isfinite(refs.glmGamPoi_ql.to_numpy(float))
        row["ql"] = dict(discoveries=counts,
                         spearman_glmgampoi=float(stats.spearmanr(mine[okq], refs.glmGamPoi_ql[okq]).statistic),
                         spearman_edger=float(stats.spearmanr(mine[okq], refs.edgeR_ql[okq]).statistic),
                         jaccard_glmgampoi=jaccard(bh(mine, 0.05), bh(refs.glmGamPoi_ql, 0.05)),
                         jaccard_edger=jaccard(bh(mine, 0.05), bh(refs.edgeR_ql, 0.05)))
        low, high = min(counts["edgeR_ql"], counts["glmGamPoi_ql"]), max(counts["edgeR_ql"], counts["glmGamPoi_ql"])
        g3 &= row["ql"]["spearman_glmgampoi"] >= 0.95 and 0.8 * low <= counts["rapidglm"] <= 1.25 * high
        # G1: contrasts against reparametrized coefficients.
        if kind == "contrast":
            x = np.loadtxt(panel / "design.tsv")
            xr, column = reparametrize(x, spec)
            np.savetxt(out_dir / f"{name}.xr.tsv", xr, delimiter="\t", fmt="%.17g")
            diffs = {}
            for command, t in (("bulk-de", "wald"), ("bulk-de", "lrt"), ("pseudobulk-de", "ql"),
                               ("pseudobulk-de", "lr"), ("pseudobulk-de", "wald")):
                extra = dict(responses=str(panel / "responses.mtx"), test=t)
                if command == "pseudobulk-de":
                    extra["offset"] = str(panel / "offset.tsv")
                c = run(args.binary, command, out_dir / f"{name}.eq.c", design=str(panel / "design.tsv"),
                        contrast=spec, **extra)
                r = run(args.binary, command, out_dir / f"{name}.eq.r", design=str(out_dir / f"{name}.xr.tsv"),
                        coefficient=column, **extra)
                key = "pvalue" if command == "bulk-de" else "p_value"
                pc, pr = c[key].to_numpy(float), r[key].to_numpy(float)
                both = np.isfinite(pc) & np.isfinite(pr)
                if command == "bulk-de":
                    calls = lambda t: set(np.flatnonzero(np.nan_to_num(t.padj.to_numpy(float), nan=1) < 0.1))
                else:
                    calls = lambda t: bh(t.p_value, 0.05)
                diffs[f"{command}:{t}"] = dict(
                    max_rel_diff=float(np.max(np.abs(pc[both] - pr[both]) / np.maximum(pr[both], 1e-300))),
                    finite_mismatches=int((np.isfinite(pc) != np.isfinite(pr)).sum()),
                    discovery_jaccard=jaccard(calls(c), calls(r)))
                if command == "bulk-de":
                    runs = [json.loads(pathlib.Path(str(out_dir / f"{name}.eq.{x}") + ".run.json").read_text())
                            for x in ("c", "r")]
                    genewise = np.abs(c.dispGeneEst - r.dispGeneEst) / r.dispGeneEst
                    diffs[f"{command}:{t}"].update(
                        prior_variance=[runs[0]["prior_variance"], runs[1]["prior_variance"]],
                        genewise_dispersions_differing=int((genewise > 1e-6).sum()))
            row["equivalence"] = diffs
            g1 &= all(v["max_rel_diff"] < 1e-6 and v["finite_mismatches"] == 0 for v in diffs.values())
        else:
            # G4: joint tests at 1 and 4 threads.
            same = {}
            for command, t, source in (("bulk-de", "lrt", d), ("pseudobulk-de", "ql", panel)):
                extra = dict(test=t, coefficients=spec)
                if command == "bulk-de":
                    extra.update(responses=str(d / "counts.mtx"), design=str(d / "x.tsv"))
                else:
                    extra.update(responses=str(panel / "responses.mtx"), design=str(panel / "design.tsv"),
                                 offset=str(panel / "offset.tsv"))
                tables = []
                for threads in (1, 4):
                    run(args.binary, command, out_dir / f"{name}.t{threads}", threads=threads, **extra)
                    tables.append((out_dir / f"{name}.t{threads}.tsv").read_bytes())
                same[command] = tables[0] == tables[1]
            row["identical_1_and_4_threads"] = same
            g4 &= all(same.values())
        results[name] = row
        print(name, json.dumps({k: row[k] for k in ("bulk", "ql")}), flush=True)
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        worktree_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()),
        command="python3 scripts/validate_hypotheses.py --work DIR --output validation/HYPOTHESES.json",
        hypotheses=results,
        gates=dict(G1_equivalence=g1, G2_bulk_deseq2=g2, G3_ql_references=g3, G4_threads=g4),
        passed=g1 and g2 and g3 and g4,
    )
    args.output.write_text(json.dumps(receipt, indent=1, default=float) + "\n")
    print("gates", receipt["gates"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=pathlib.Path, required=True, help="the validate_bulk.py work directory")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--r-library", default=str(ROOT / "validation/local-data/R_library"))
    ap.add_argument("--ql-library",
                    default=f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}")
    main(ap.parse_args())
