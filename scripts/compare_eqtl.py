#!/usr/bin/env python3
"""Gene-level agreement of `rapidglm qtl` with the cis-eQTL reference runs on
Geuvadis chromosome 22 (validation/work/eqtl/: FastQTL, tensorQTL and jaxQTL
runs and their gene-level table runs/summary/gene_level_chr22.tsv.gz).

For each RapidGLM run (a `qtl` output prefix and a label), the gene-level
p-value is the Beta-approximated permutation p-value (beta_p). q-values come
from Bioconductor qvalue with its defaults (run as a black box, as for the
references). Reported per run and reference: eGenes at q < 0.05 and their
overlap, the Spearman correlation of gene-level p-values, and the share of
genes with the same lead variant.

    python3 scripts/compare_eqtl.py --run nb validation/work/eqtl/runs/rapidglm/chr22_nb \\
        --run gaussian validation/work/eqtl/runs/rapidglm/chr22_gaussian --output OUT.json
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
EQTL = pathlib.Path(os.environ.get("EQTL_WORK", ROOT / "validation/work/eqtl"))
REFERENCES = ["FastQTL_perm_beta", "tensorQTL_perm10000_beta", "jaxQTL_NB_score_perm1000",
              "jaxQTL_NB_score_SPA_ACAT", "jaxQTL_Gauss_score_perm1000_ncx2", "jaxQTL_Gauss_score_ACAT"]


def qvalues(p):
    """Storey q-values (qvalue defaults) for a Series of p-values; (q, pi0)."""
    with tempfile.TemporaryDirectory() as tmp:
        source, target = pathlib.Path(tmp) / "p.tsv", pathlib.Path(tmp) / "q.tsv"
        pd.DataFrame({"id": p.index, "p": p.to_numpy()}).to_csv(source, sep="\t", index=False)
        r = subprocess.run(["Rscript", str(ROOT / "scripts/eqtl_qvalue.R"), str(source), str(target),
                            str(EQTL / "tools/Rlib")], capture_output=True, text=True, check=True)
        q = pd.read_csv(target, sep="\t", dtype={"id": str})
        pi0 = float(re.search(r"pi0 ([\d.]+)", r.stdout).group(1))
    return pd.Series(q.q.to_numpy(), index=q.id), pi0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "PREFIX"), required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path,
                        default=EQTL / "runs/summary/gene_level_chr22.tsv.gz")
    args = parser.parse_args()
    reference = pd.read_csv(args.reference, sep="\t").set_index("phenotype_id")
    summary = {}
    for label, prefix in args.run:
        ours = pd.read_csv(prefix + ".cis.tsv", sep="\t", na_values="NA", keep_default_na=False)
        ours = ours.set_index("phenotype_id")
        record = json.loads(pathlib.Path(prefix + ".run.json").read_text())
        valid = ours.status.str.startswith("ok") & np.isfinite(ours.beta_p)
        q, pi0 = qvalues(ours.beta_p[valid])
        hits = set(q.index[q < 0.05])
        entry = {"genes": int(len(ours)), "valid": int(valid.sum()),
                 "statuses": ours.status.value_counts().to_dict(), "pi0": pi0, "egenes": len(hits),
                 "beta_vs_permutation_spearman": float(spearmanr(ours.beta_p[valid],
                                                                 ours.permutation_p[valid]).statistic),
                 "wall_seconds": record.get("wall_seconds"), "threads": record.get("threads"),
                 "permutations": record.get("permutations"), "references": {}}
        for name in REFERENCES:
            if f"{name}.p" not in reference:
                continue
            ref = reference[reference[f"{name}.valid"].astype(str) == "True"]
            ref_hits = set(ref.index[ref[f"{name}.q"] < 0.05])
            both = ref.index.intersection(q.index)
            entry["references"][name] = {
                "egenes": len(ref_hits), "shared": len(hits & ref_hits),
                "jaccard": round(len(hits & ref_hits) / len(hits | ref_hits), 4) if hits | ref_hits else 1.0,
                "spearman_p": float(spearmanr(ours.beta_p[both], ref.loc[both, f"{name}.p"]).statistic),
                "same_lead": float(np.mean(ours.lead_variant[both].to_numpy() ==
                                           ref.loc[both, f"{name}.lead"].to_numpy())),
            }
        summary[label] = entry
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
