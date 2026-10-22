#!/usr/bin/env python3
"""Prepare Geuvadis gene-level expression for the chr22 cis-eQTL benchmark.

Inputs (E-GEUV-1, ArrayExpress/BioStudies):
  GD660.GeneQuantCount.txt.gz   raw gene read counts, 660 QC-passed libraries
  GD660.GeneQuantRPKM.txt.gz    RPKM for the same libraries
  E-GEUV-1.sdrf.txt             maps each of the 462 unique individuals to its library

Outputs (in --out):
  samples.txt                   462 individual IDs (sorted), the sample order everywhere
  sample_info.tsv               individual, library, population, sex, lab, phase1 flag
  library_size.tsv              iid, library (assay), libsize (sum of rounded counts over all
                                53,934 genes), log_libsize, tmm_factor, effective_libsize
  offset_log_libsize.tsv        iid, log_libsize (jaxQTL --offset)
  genes_expressed.tsv           genome-wide expressed autosomal gene set with TSS
  expr_int_allchr.tsv.gz        TMM-CPM -> per-gene inverse-normal transform, all expressed genes
  expression_pcs.tsv            top expression PCs (from expr_int_allchr)
  chr{CHR}.counts.bed           integer counts (rounded), TSS BED, expressed genes on CHR
  chr{CHR}.expr_int.bed         inverse-normal TMM-CPM, expressed genes on CHR
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

p = argparse.ArgumentParser()
p.add_argument("--geuv", required=True, help="directory with Geuvadis downloads")
p.add_argument("--out", required=True)
p.add_argument("--chr", default="22")
p.add_argument("--n-expr-pcs", type=int, default=20)
p.add_argument("--min-rpkm", type=float, default=0.1)
p.add_argument("--min-count", type=float, default=6)
p.add_argument("--min-frac", type=float, default=0.2)
p.add_argument("--tmm-script", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
summary = {}

# --- samples: one library per individual from the SDRF ---------------------------------
sdrf = pd.read_csv(os.path.join(a.geuv, "E-GEUV-1.sdrf.txt"), sep="\t", dtype=str)
info = sdrf[["Source Name", "Assay Name", "Characteristics[ancestry category]",
             "Characteristics[sex]", "Characteristics[laboratory]",
             "Comment[1000g Phase1 Genotypes]"]].drop_duplicates()
info.columns = ["iid", "library", "population", "sex", "lab", "phase1_genotypes"]
assert info.iid.is_unique and len(info) == 462, len(info)
info = info.sort_values("iid").reset_index(drop=True)
info.to_csv(os.path.join(a.out, "sample_info.tsv"), sep="\t", index=False)
info.iid.to_csv(os.path.join(a.out, "samples.txt"), index=False, header=False)
lib2iid = dict(zip(info.library, info.iid))

# --- counts ------------------------------------------------------------------------------
meta_cols = ["TargetID", "Gene_Symbol", "Chr", "Coord"]
cnt = pd.read_csv(os.path.join(a.geuv, "GD660.GeneQuantCount.txt.gz"), sep="\t")
rpkm = pd.read_csv(os.path.join(a.geuv, "GD660.GeneQuantRPKM.txt.gz"), sep="\t")
assert (cnt.TargetID.values == rpkm.TargetID.values).all()
missing = [l for l in info.library if l not in cnt.columns]
assert not missing, missing
meta = cnt[meta_cols].copy()
meta["Chr"] = meta["Chr"].astype(str)
C = cnt[info.library.tolist()].to_numpy(dtype=float)
R = rpkm[info.library.tolist()].to_numpy(dtype=float)
Cr = np.rint(C)
dev = np.abs(C - Cr)
summary["count_matrix_genes"] = int(C.shape[0])
summary["count_entries_noninteger_frac"] = float(np.mean(dev > 0))
summary["count_entries_off_integer_gt_0.01_frac"] = float(np.mean(dev > 0.01))
summary["count_entries_off_integer_gt_0.1_frac"] = float(np.mean(dev > 0.1))
summary["count_rounding_abs_change_over_total"] = float(dev.sum() / C.sum())

libsize = Cr.sum(axis=0)
# --- gene filter (GTEx v8 thresholds, RPKM in place of TPM), autosomes only -------------
auto = meta.Chr.isin([str(i) for i in range(1, 23)]).to_numpy()
n = Cr.shape[1]
keep = auto & ((R >= a.min_rpkm).sum(1) >= a.min_frac * n) & ((Cr >= a.min_count).sum(1) >= a.min_frac * n)
summary["genes_autosomal"] = int(auto.sum())
summary["genes_expressed_autosomal"] = int(keep.sum())
gm = meta.loc[keep].reset_index(drop=True)
Ck = Cr[keep]

# --- TMM factors via edgeR::calcNormFactors (method TMM) on the expressed genes ---------
tmp_counts = os.path.join(a.out, "_tmm_input.tsv")
pd.DataFrame(Ck, columns=info.iid).to_csv(tmp_counts, sep="\t", index=False)
tmp_out = os.path.join(a.out, "_tmm_factors.tsv")
subprocess.run(["Rscript", a.tmm_script, tmp_counts, tmp_out], check=True)
tmm = pd.read_csv(tmp_out, sep="\t")
assert (tmm.iid.values == info.iid.values).all()
os.remove(tmp_counts)
os.remove(tmp_out)
tmm_f = tmm.norm_factor.to_numpy()
eff = Ck.sum(axis=0) * tmm_f  # edgeR effective library size (lib.size = colSums of input)
cpm = Ck / eff * 1e6


def inverse_normal(x):
    r = rankdata(x, method="average")
    return norm.ppf(r / (len(x) + 1))


Z = np.apply_along_axis(inverse_normal, 1, cpm)

ls = pd.DataFrame({"iid": info.iid, "library": info.library, "libsize": libsize.astype(np.int64),
                   "log_libsize": np.log(libsize), "tmm_factor": tmm_f,
                   "effective_libsize_expressed_genes": eff})
ls.to_csv(os.path.join(a.out, "library_size.tsv"), sep="\t", index=False, float_format="%.10g")
ls[["iid", "log_libsize"]].to_csv(os.path.join(a.out, "offset_log_libsize.tsv"), sep="\t",
                                   index=False, float_format="%.12g")
summary["libsize_min_median_max"] = [int(libsize.min()), float(np.median(libsize)), int(libsize.max())]

gm_out = gm.rename(columns={"TargetID": "gene_id", "Gene_Symbol": "gene_symbol", "Chr": "chr",
                            "Coord": "tss"})
gm_out.to_csv(os.path.join(a.out, "genes_expressed.tsv"), sep="\t", index=False)
zdf = pd.concat([gm_out[["gene_id"]], pd.DataFrame(Z, columns=info.iid)], axis=1)
zdf.to_csv(os.path.join(a.out, "expr_int_allchr.tsv.gz"), sep="\t", index=False, float_format="%.6g")

# --- expression PCs: SVD of samples x genes, genes centred and scaled -------------------
X = Z.T.copy()
X -= X.mean(0)
X /= X.std(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
var = S ** 2 / (S ** 2).sum()
k = a.n_expr_pcs
pcs = pd.DataFrame(U[:, :k] * S[:k] / np.sqrt(n - 1), columns=[f"ePC{i+1}" for i in range(k)])
pcs.insert(0, "iid", info.iid)
pcs.to_csv(os.path.join(a.out, "expression_pcs.tsv"), sep="\t", index=False, float_format="%.8g")
summary["expr_pc_var_explained_first30"] = [round(float(v), 5) for v in var[:30]]
summary["expr_pc_var_explained_cum_k"] = float(var[:k].sum())

# Buja-Eyuboglu style parallel analysis (20 column-permuted replicates), recorded only
rng = np.random.default_rng(20261016)
perm_top = []
for _ in range(20):
    Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    sp = np.linalg.svd(Xp, compute_uv=False)
    perm_top.append(sp[:60] ** 2 / (sp ** 2).sum())
perm_top = np.array(perm_top)
thr = np.quantile(perm_top, 0.95, axis=0)
be_k = int(np.argmax(var[:60] <= thr)) if (var[:60] <= thr).any() else 60
summary["parallel_analysis_k_95pct"] = be_k

# --- chromosome BEDs ---------------------------------------------------------------------
sel = (gm_out.chr == a.chr).to_numpy()
order = np.argsort(gm_out.tss.to_numpy()[sel], kind="stable")
g = gm_out.loc[sel].iloc[order]
bed_meta = pd.DataFrame({"#chr": g.chr.values, "start": g.tss.values - 1, "end": g.tss.values,
                         "gene_id": g.gene_id.values})
cnt_bed = pd.concat([bed_meta, pd.DataFrame(Ck[sel][order].astype(np.int64), columns=info.iid)], axis=1)
int_bed = pd.concat([bed_meta, pd.DataFrame(Z[sel][order], columns=info.iid)], axis=1)
cnt_bed.to_csv(os.path.join(a.out, f"chr{a.chr}.counts.bed"), sep="\t", index=False)
int_bed.to_csv(os.path.join(a.out, f"chr{a.chr}.expr_int.bed"), sep="\t", index=False, float_format="%.6g")
summary[f"chr{a.chr}_genes"] = int(sel.sum())
summary[f"chr{a.chr}_genes_in_GD660"] = int((meta.Chr == a.chr).sum())

json.dump(summary, open(os.path.join(a.out, "prep_expression_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
