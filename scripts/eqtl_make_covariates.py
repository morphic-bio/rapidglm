#!/usr/bin/env python3
"""Build the covariate tables shared by every tool.

Covariates: genotype PCs 1-5 (plink2 --pca on LD-pruned chr22 SNVs), expression PCs 1-K
(SVD of the TMM-CPM inverse-normal matrix of all 20,427 expressed autosomal genes; K chosen by
parallel analysis, see prep_expression_summary.json) and sex (female = 1).
Writes FastQTL/tensorQTL layout (covariates x samples, first row "id") and jaxQTL layout
(samples x covariates, first column "iid").
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--eigenvec", required=True)
p.add_argument("--expr-pcs", required=True)
p.add_argument("--sample-info", required=True)
p.add_argument("--n-geno-pcs", type=int, default=5)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

info = pd.read_csv(a.sample_info, sep="\t")
ev = pd.read_csv(a.eigenvec, sep="\t")
ev = ev.rename(columns={"#IID": "iid", "IID": "iid", "#FID": "fid"})
ev = ev.set_index("iid").reindex(info.iid)
assert ev.notna().all().all()
g = ev[[f"PC{i+1}" for i in range(a.n_geno_pcs)]]
g.columns = [f"gPC{i+1}" for i in range(a.n_geno_pcs)]
e = pd.read_csv(a.expr_pcs, sep="\t").set_index("iid").reindex(info.iid)
assert e.notna().all().all()
sex = pd.Series((info.sex.values == "female").astype(int), index=info.iid, name="sex_female")
cov = pd.concat([g, e, sex], axis=1)
cov.index.name = "iid"
cov.to_csv(os.path.join(a.out, "covariates.jaxqtl.tsv"), sep="\t", float_format="%.8g")
t = cov.T
t.index.name = "id"
t.to_csv(os.path.join(a.out, "covariates.fastqtl.txt"), sep="\t", float_format="%.8g")
print(cov.shape, "rank", np.linalg.matrix_rank(np.column_stack([np.ones(len(cov)), cov.values])))
