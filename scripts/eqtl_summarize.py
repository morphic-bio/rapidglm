#!/usr/bin/env python3
"""Gene-level summary of the chr22 reference runs: Storey q-values (R qvalue, default smoother pi0)
on each method's gene-level p-value, eGenes at q < 0.05, pairwise overlaps and timings."""
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

W = os.environ.get("EQTL_WORK", "/mnt/pikachu/rapidglm/validation/work/eqtl")
OUT = os.path.join(W, "runs", "summary")
os.makedirs(OUT, exist_ok=True)
genes = pd.read_csv(os.path.join(W, "data/expression/chr22.genes.txt"), header=None)[0].tolist()
gtype = pd.read_csv(os.path.join(W, "data/gencode_v12_genes.tsv"), sep="\t", header=None,
                    usecols=[0, 7, 8], names=["gene_id", "gene_type", "gene_name"]).set_index("gene_id")

FQ_PERM_COLS = ["phenotype_id", "num_var", "beta_shape1", "beta_shape2", "true_df", "pval_true_df",
                "variant_id", "tss_distance", "ma_samples", "ma_count", "maf", "ref_factor",
                "pval_nominal", "slope", "slope_se", "pval_perm", "pval_beta"]

methods = {}  # name -> DataFrame(phenotype_id, p, lead, valid, nominal_p, num_var, extra...)


def load_fastqtl():
    f = os.path.join(W, "runs/fastqtl/permute/chr22.permute.txt.gz")
    d = pd.read_csv(f, sep="\t", header=None, names=FQ_PERM_COLS)
    d.to_csv(os.path.join(W, "runs/fastqtl/permute/chr22.permute.with_header.txt.gz"), sep="\t", index=False)
    return pd.DataFrame({"phenotype_id": d.phenotype_id, "p": d.pval_beta, "lead": d.variant_id,
                         "nominal_p": d.pval_nominal, "num_var": d.num_var, "valid": np.isfinite(d.pval_beta),
                         "p_perm_direct": d.pval_perm})


def load_jaxqtl(run, fname):
    f = os.path.join(W, "runs/jaxqtl", run, fname)
    if not os.path.exists(f):
        return None
    d = pl.read_parquet(f).to_pandas()
    valid = d.result_valid.fillna(False) & d.model_converged.fillna(False) & np.isfinite(d.pvalue_adj) \
        & d.pvalue_adj.between(0, 1)
    if "perm_converged" in d:
        valid &= d.perm_converged.fillna(False)
    out = pd.DataFrame({"phenotype_id": d.phenotype_id, "p": d.pvalue_adj, "lead": d.snp,
                        "nominal_p": d.pvalue, "num_var": d.num_var, "valid": valid})
    if "nb_alpha" in d:
        out["nb_alpha"] = d.nb_alpha
    return out


methods["FastQTL_perm_beta"] = load_fastqtl()
_tq = os.path.join(W, "runs/tensorqtl/cis/chr22.cis_qtl.txt.gz")
if os.path.exists(_tq):
    _d = pd.read_csv(_tq, sep="\t")
    methods["tensorQTL_perm10000_beta"] = pd.DataFrame({
        "phenotype_id": _d.phenotype_id, "p": _d.pval_beta, "lead": _d.variant_id,
        "nominal_p": _d.pval_nominal, "num_var": _d.num_var, "valid": np.isfinite(_d.pval_beta)})
for name, run, fname in [
    ("jaxQTL_NB_score_SPA_ACAT", "nb_score_spa_acat", "chr22.cis.score.spa.acat.parquet.gz"),
    ("jaxQTL_NB_score_perm1000", "nb_score_perm1000", "chr22.cis.score.perm.parquet.gz"),
    ("jaxQTL_Gauss_score_ACAT", "gaussian_score_acat", "chr22.cis.score.acat.parquet.gz"),
    ("jaxQTL_Gauss_score_perm1000_ncx2", "gaussian_score_perm1000_ncx2", "chr22.cis.score.perm.parquet.gz"),
    ("jaxQTL_Gauss_score_perm1000_tdist", "gaussian_score_perm1000_tdist", "chr22.cis.score.perm.parquet.gz"),
]:
    d = load_jaxqtl(run, fname)
    if d is not None:
        methods[name] = d

# same-configuration reproducibility: 1 process x 12 CPUs vs 4 processes x 3 CPUs
a = load_jaxqtl("nb_score_spa_acat", "chr22.cis.score.spa.acat.parquet.gz")
b = load_jaxqtl("nb_score_spa_acat_4x3", "chr22.cis.score.spa.acat.parquet.gz")
repro = None
if a is not None and b is not None:
    m = a.merge(b, on="phenotype_id")
    repro = {"genes": int(len(m)), "max_abs_diff_pvalue_adj": float(np.nanmax(np.abs(m.p_x - m.p_y))),
             "same_lead": int((m.lead_x == m.lead_y).sum())}

summary = {"genes_tested_input": len(genes), "fdr_method": "Storey q-value, Bioconductor qvalue 2.40.0 "
           "defaults (lambda = 0.05..0.95, smoother pi0), on valid genes of each method; eGene: q < 0.05",
           "methods": {}, "reproducibility_nb_spa_acat_1x12_vs_4x3": repro}
egenes = {}
for name, d in methods.items():
    v = d[d.valid].copy()
    tmp_in, tmp_out = os.path.join(OUT, f"_{name}_p.tsv"), os.path.join(OUT, f"_{name}_q.tsv")
    info = {"rows": int(len(d)), "valid": int(len(v)),
            "invalid": int((~d.valid).sum()), "genes_missing_from_output": int(len(set(genes) - set(d.phenotype_id)))}
    if len(v) > 0:
        v[["phenotype_id", "p"]].rename(columns={"phenotype_id": "id"}).to_csv(tmp_in, sep="\t", index=False)
        r = subprocess.run(["Rscript", os.path.join(os.path.dirname(__file__), "eqtl_qvalue.R"), tmp_in, tmp_out,
                            os.path.join(W, "tools/Rlib")], capture_output=True, text=True, check=True)
        q = pd.read_csv(tmp_out, sep="\t")
        os.remove(tmp_in)
        os.remove(tmp_out)
        v["q"] = q.q.values
        m = re.search(r"pi0 ([\d.]+) method (\S+)", r.stdout)
        info["pi0"] = float(m.group(1))
        info["pi0_method"] = m.group(2)
        info["egenes_q05"] = int((v.q < 0.05).sum())
        # Benjamini-Hochberg for reference
        p = np.sort(v.p.values)
        bh = p * len(p) / np.arange(1, len(p) + 1)
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        info["egenes_bh05"] = int((bh < 0.05).sum())
        egenes[name] = set(v.phenotype_id[v.q < 0.05])
        d = d.merge(v[["phenotype_id", "q"]], on="phenotype_id", how="left")
    else:
        d["q"] = np.nan
        egenes[name] = set()
    methods[name] = d
    summary["methods"][name] = info
    with open(os.path.join(OUT, f"egenes.{name}.txt"), "w") as fh:
        fh.write("\n".join(sorted(egenes[name])) + ("\n" if egenes[name] else ""))

# Geuvadis published EUR373 gene eQTLs (FDR 5%), chr22 genes, for orientation only
geu = pd.read_csv(os.path.join(W, "downloads/geuvadis/EUR373.gene.cis.FDR5.best.rs137.txt.gz"), sep="\t",
                  header=None)
geu22 = set(geu[geu[5].astype(str) == "22"][2])
summary["geuvadis_EUR373_published_chr22_egenes"] = len(geu22)
summary["geuvadis_EUR373_published_chr22_egenes_in_tested_set"] = len(geu22 & set(genes))

# pairwise overlaps and gene-level p-value agreement
names = list(methods)
ov = []
for x, y in itertools.combinations(names, 2):
    ex, ey = egenes[x], egenes[y]
    mx = methods[x][methods[x].valid][["phenotype_id", "p", "lead"]]
    my = methods[y][methods[y].valid][["phenotype_id", "p", "lead"]]
    m = mx.merge(my, on="phenotype_id")
    rho = spearmanr(m.p_x, m.p_y).statistic if len(m) > 2 else np.nan
    ov.append({"a": x, "b": y, "egenes_a": len(ex), "egenes_b": len(ey), "both": len(ex & ey),
               "only_a": len(ex - ey), "only_b": len(ey - ex),
               "jaccard": (len(ex & ey) / len(ex | ey)) if (ex | ey) else np.nan,
               "genes_valid_both": int(len(m)), "spearman_gene_p": rho,
               "same_lead_variant": int((m.lead_x == m.lead_y).sum())})
for x in names:
    ex = egenes[x]
    ov.append({"a": x, "b": "Geuvadis_EUR373_published", "egenes_a": len(ex), "egenes_b": len(geu22 & set(genes)),
               "both": len(ex & geu22), "only_a": len(ex - geu22), "only_b": len((geu22 & set(genes)) - ex),
               "jaccard": len(ex & geu22) / len(ex | (geu22 & set(genes))) if ex else np.nan})
ov = pd.DataFrame(ov)
ov.to_csv(os.path.join(OUT, "overlaps.tsv"), sep="\t", index=False, float_format="%.4g")
summary["overlaps"] = ov.replace({np.nan: None}).to_dict(orient="records")

# gene table (one row per gene; p, q, lead per method)
tab = pd.DataFrame({"phenotype_id": genes})
tab = tab.join(gtype, on="phenotype_id")
for name, d in methods.items():
    cols = d[["phenotype_id", "p", "q", "lead", "valid"]].rename(
        columns={c: f"{name}.{c}" for c in ["p", "q", "lead", "valid"]})
    tab = tab.merge(cols, on="phenotype_id", how="left")
tab.to_csv(os.path.join(OUT, "gene_level_chr22.tsv.gz"), sep="\t", index=False, float_format="%.6g")

# eGenes by gene type for FastQTL and the NB methods
bytype = {}
for name in names:
    e = tab[tab.phenotype_id.isin(egenes[name])]
    bytype[name] = e.gene_type.value_counts().to_dict()
summary["egenes_by_gene_type"] = bytype


# timings
def parse_time(path):
    s = open(path).read()
    g = lambda k: float(re.search(k + r": ([\d.]+)", s).group(1))
    el = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", s).group(1).split(":")
    sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(el)))
    return {"wall_s": sec, "user_s": g(r"User time \(seconds\)"), "sys_s": g(r"System time \(seconds\)"),
            "max_rss_kb": g(r"Maximum resident set size \(kbytes\)")}


timing = {}
for mode in ["nominal", "permute"]:
    p = os.path.join(W, "runs/fastqtl", mode, "time.txt")
    if os.path.exists(p):
        t = parse_time(p)
        t["layout"] = "12 fastQTL processes (--chunk k 12), 1 thread each"
        timing[f"FastQTL_{mode}"] = t
for mode in ["cis", "cis_nominal"]:
    p = os.path.join(W, "runs/tensorqtl", mode, "time.txt")
    if os.path.exists(p):
        t = parse_time(p)
        t["layout"] = "1 process, torch CPU, CPUs 6,7,12-15 (6 logical)"
        timing[f"tensorQTL_{mode}"] = t
for run in sorted(os.listdir(os.path.join(W, "runs/jaxqtl"))):
    d = os.path.join(W, "runs/jaxqtl", run)
    if run.startswith(("smoke", "aborted")):
        continue
    if os.path.exists(os.path.join(d, "time.txt")):
        t = parse_time(os.path.join(d, "time.txt"))
        cmdline = open(os.path.join(d, "command.txt")).read()
        m = re.match(r"CPUS=(\S+)", cmdline)
        t["layout"] = f"1 process, CPUs {m.group(1) if m else '0-7,12-15'}"
        timing[f"jaxQTL_{run}"] = t
    elif os.path.exists(os.path.join(d, "cpu.txt")):
        w = open(os.path.join(d, "wall.txt")).read()
        c = open(os.path.join(d, "cpu.txt")).read()
        parts = [parse_time(os.path.join(d, f"part{k}", "time.txt")) for k in range(4)
                 if os.path.exists(os.path.join(d, f"part{k}", "time.txt"))]
        timing[f"jaxQTL_{run}"] = {
            "wall_s": float(re.search(r"wall_seconds ([\d.]+)", w).group(1)),
            "user_s": float(re.search(r"user_seconds ([\d.]+)", c).group(1)),
            "sys_s": float(re.search(r"system_seconds ([\d.]+)", c).group(1)),
            "max_rss_kb": max(p["max_rss_kb"] for p in parts) if parts else None,
            "layout": "4 processes x 3 logical CPUs (genes split round-robin)"}
summary["timing"] = timing
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

# console
print(json.dumps({k: v for k, v in summary["methods"].items()}, indent=1))
print(ov.to_string(index=False))
print(pd.DataFrame(timing).T.to_string())
