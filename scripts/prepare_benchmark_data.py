#!/usr/bin/env python3
"""Prepare the large public datasets of the application benchmark.

thyroid: GTEx v8 thyroid from recount3 (public domain; https://rna.recount.bio),
  files under validation/local-data/recount3:
    gtex.gene_sums.THYROID.G026.gz, gtex.gtex.THYROID.MD.gz, gtex.recount_qc.THYROID.MD.gz
  (https://duffel.rail.bio/recount3/human/data_sources/gtex/{gene_sums,metadata}/ID/THYROID/).
  recount3 stores base-level coverage sums; read counts are the sums divided by
  the sample's average mapped read length (star.average_mapped_length), rounded,
  as recount3's documentation describes for compute_read_counts(). Samples: GTEx's
  RNA-seq release set (SMAFRZE = RNASEQ) with known death circumstances (DTHHRDY)
  from the two large centres. Genes: at least 10 reads in at least 5% of samples.
  Design: ~ centre + Hardy scale + age bracket + sex; the tested column is sex
  (female against male).

gasperini: the Gasperini et al. (2019) at-scale CRISPRi enhancer screen, GEO
  GSE120861 (K562, high MOI; 207,324 cells x 13,135 genes), files under
  validation/local-data/gasperini2019 (GSE120861_at_scale_screen.{exprs.mtx.gz,
  genes.txt.gz, cells.txt.gz, phenoData.txt.gz}, GSE120861_grna_groups.at_scale.txt.gz,
  GSE120861_gene_gRNAgroup_pair_table.at_scale.txt.gz and
  GSE120861_zero_inflated_outlier_genes_to_exclude.atscale.txt.gz; the "exprs.mtx.gz"
  file is served as plain MatrixMarket text). Guides are the 20-nt spacers listed
  per cell in phenoData, grouped by grna_groups (group -> target; NTC groups are
  non-targeting). Pairs, drawn once with a fixed seed from the published pair
  table and excluding the zero-inflated outlier genes: every positive_ctrl pair,
  10,000 candidate-enhancer pairs (test_5k-*) and 5,000 NTC pairs. Responses are
  the genes in those pairs; covariates come from every gene: log total UMIs, log
  detected genes, log(1 + guides called), percent mitochondrial (phenoData) and
  the preparation batch.

Writes validation/work/benchmark/<dataset>/ with counts.mtx (feature by sample,
the CLI layout), x.tsv (design), design.tsv (design with column names),
samples.tsv, genes.txt and provenance.json (input hashes and filters).
"""

import argparse
import gzip
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL = ROOT / "validation/local-data"
WORK = ROOT / "validation/work/benchmark"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_market(path, counts):
    """counts: features x samples integer array, written in coordinate form."""
    rows, cols = np.nonzero(counts)
    with open(path, "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{counts.shape[0]} {counts.shape[1]} {len(rows)}\n")
        values = counts[rows, cols]
        for chunk in range(0, len(rows), 1_000_000):
            s = slice(chunk, chunk + 1_000_000)
            out.write("".join(f"{r + 1} {c + 1} {v}\n" for r, c, v in zip(rows[s], cols[s], values[s])))


def thyroid(output):
    source = LOCAL / "recount3"
    files = {
        "gene_sums": source / "gtex.gene_sums.THYROID.G026.gz",
        "metadata": source / "gtex.gtex.THYROID.MD.gz",
        "qc": source / "gtex.recount_qc.THYROID.MD.gz",
    }
    md = pd.read_csv(files["metadata"], sep="\t", low_memory=False)
    qc = pd.read_csv(files["qc"], sep="\t", low_memory=False)
    with gzip.open(files["gene_sums"], "rt") as f:
        sums = pd.read_csv(f, sep="\t", comment="#", index_col=0)
    md = md.set_index("external_id")
    qc = qc.set_index("external_id")
    keep = md[
        (md.SMAFRZE == "RNASEQ") & md.DTHHRDY.notna() & md.SMCENTER.isin(["B1", "C1"])
    ].index
    length = qc.loc[keep, "star.average_mapped_length"]
    counts = np.rint(sums[keep].to_numpy(float) / length.to_numpy()[None, :]).astype(np.int64)
    expressed = (counts >= 10).sum(axis=1) >= 0.05 * len(keep)
    counts = counts[expressed]
    genes = sums.index[expressed]
    samples = pd.DataFrame(
        {
            "center": md.loc[keep, "SMCENTER"],
            "hardy": "hardy" + md.loc[keep, "DTHHRDY"].astype(int).astype(str),
            "age": "age" + md.loc[keep, "AGE"].str.replace("-", "_"),
            "sex": np.where(md.loc[keep, "SEX"] == 2, "female", "male"),
        },
        index=keep,
    )
    samples["sex"] = pd.Categorical(samples["sex"], categories=["male", "female"])
    columns = {"Intercept": np.ones(len(samples))}
    for name in ("center", "hardy", "age", "sex"):
        column = samples[name]
        levels = list(column.cat.categories) if hasattr(column, "cat") else sorted(column.unique())
        for level in levels[1:]:
            columns[f"{name}[T.{level}]"] = (column == level).to_numpy(float)
    design = pd.DataFrame(columns, index=samples.index)
    output.mkdir(parents=True, exist_ok=True)
    write_market(output / "counts.mtx", counts)
    np.savetxt(output / "x.tsv", design.to_numpy(), fmt="%g", delimiter="\t")
    design.to_csv(output / "design.tsv", sep="\t", index=False)
    samples.to_csv(output / "samples.tsv", sep="\t")
    (output / "genes.txt").write_text("\n".join(genes) + "\n")
    provenance = {
        "dataset": "GTEx v8 thyroid (recount3, public domain)",
        "inputs": {k: {"file": str(v.relative_to(ROOT)), "sha256": sha256(v)} for k, v in files.items()},
        "counts": "round(coverage sum / star.average_mapped_length)",
        "samples": int(len(keep)),
        "genes": int(len(genes)),
        "sample_filter": "SMAFRZE == RNASEQ, DTHHRDY known, SMCENTER in B1/C1",
        "gene_filter": ">= 10 reads in >= 5% of samples",
        "design": list(design.columns),
        "tested_coefficient": int(design.columns.get_loc("sex[T.female]")),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ("samples", "genes", "tested_coefficient")}))


def gasperini(output, discovery=10_000, negative=5_000, seed=20260930):
    source = LOCAL / "gasperini2019"
    name = lambda f: source / f"GSE120861_{f}"
    files = {
        "matrix": name("at_scale_screen.exprs.mtx.gz"),
        "genes": name("at_scale_screen.genes.txt.gz"),
        "cells": name("at_scale_screen.cells.txt.gz"),
        "phenodata": name("at_scale_screen.phenoData.txt.gz"),
        "groups": name("grna_groups.at_scale.txt.gz"),
        "pairs": name("gene_gRNAgroup_pair_table.at_scale.txt.gz"),
        "outliers": name("zero_inflated_outlier_genes_to_exclude.atscale.txt.gz"),
    }
    genes = pd.read_csv(files["genes"], header=None)[0].tolist()
    cells = pd.read_csv(files["cells"], header=None)[0].tolist()
    groups = pd.read_csv(files["groups"], sep="\t", header=None, names=["group", "spacer"])
    table = pd.read_csv(files["pairs"], sep="\t", usecols=["gRNAgroup", "general_group", "ENSG.targetgene"])
    outliers = set(pd.read_csv(files["outliers"], sep="\t")["id"])
    pheno = pd.read_csv(files["phenodata"], sep=r"\s+", header=None, engine="c")
    if pheno[1].tolist() != cells:
        raise SystemExit("phenoData cells differ from the cells file")
    # Guides: spacers, one column each, grouped into targets.
    ntc = set(table.loc[table.general_group == "NTC", "gRNAgroup"])
    groups = groups.drop_duplicates("spacer")
    spacer_index = {s: k for k, s in enumerate(groups.spacer)}
    rows, cols, unknown = [], [], 0
    for c, field in enumerate(pheno[6]):
        for spacer in str(field).split("_"):
            k = spacer_index.get(spacer)
            if k is None:
                unknown += 1
            else:
                rows.append(k)
                cols.append(c)
    # Pairs.
    rng = np.random.default_rng(seed)
    gene_index = {g: i for i, g in enumerate(genes)}
    table = table[table["ENSG.targetgene"].isin(gene_index) & ~table["ENSG.targetgene"].isin(outliers)]
    table = table.drop_duplicates(["gRNAgroup", "ENSG.targetgene"])
    positive = table[table.general_group == "positive_ctrl"]
    tests = table[table.general_group.str.startswith("test_5k")]
    tests = tests.iloc[np.sort(rng.choice(len(tests), min(discovery, len(tests)), replace=False))]
    negatives = table[table.general_group == "NTC"]
    negatives = negatives.iloc[np.sort(rng.choice(len(negatives), min(negative, len(negatives)), replace=False))]
    pairs = pd.concat([positive.assign(kind="positive"), tests.assign(kind="discovery"),
                       negatives.assign(kind="negative")])
    responses = sorted(set(pairs["ENSG.targetgene"]), key=gene_index.get)
    response_row = np.full(len(genes), -1)
    response_row[[gene_index[g] for g in responses]] = np.arange(len(responses))
    used_groups = set(pairs.gRNAgroup)
    # One pass over the matrix: per-cell totals from every gene, entries of the responses.
    totals = np.zeros(len(cells))
    detected = np.zeros(len(cells))
    kept = []
    with open(files["matrix"]) as f:
        f.readline()
        n_genes, n_cells, nnz = map(int, f.readline().split())
        assert (n_genes, n_cells) == (len(genes), len(cells))
        for chunk in pd.read_csv(f, sep=" ", header=None, names=["g", "c", "v"], dtype=np.int64,
                                 chunksize=50_000_000, engine="c"):
            g, c, v = chunk.g.to_numpy() - 1, chunk.c.to_numpy() - 1, chunk.v.to_numpy()
            totals += np.bincount(c, weights=v, minlength=len(cells))
            detected += np.bincount(c, minlength=len(cells))
            r = response_row[g]
            mask = r >= 0
            kept.append(np.column_stack([r[mask], c[mask], v[mask]]))
            print(f"read {int(detected.sum()):,} of {nnz:,} entries", flush=True)
    entries = np.concatenate(kept)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "responses.mtx", "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{len(responses)} {len(cells)} {len(entries)}\n")
        np.savetxt(out, entries + np.array([1, 1, 0]), fmt="%d")
    # Guides of the pairs' groups only (NTC groups included for the negative pairs).
    guide_rows = [k for k, g in enumerate(groups.group) if g in used_groups]
    guide_new = np.full(len(groups), -1)
    guide_new[guide_rows] = np.arange(len(guide_rows))
    rows, cols = np.array(rows), np.array(cols)
    mask = guide_new[rows] >= 0
    with open(output / "assignments.mtx", "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{len(guide_rows)} {len(cells)} {int(mask.sum())}\n")
        np.savetxt(out, np.column_stack([guide_new[rows[mask]] + 1, cols[mask] + 1,
                                         np.ones(int(mask.sum()), dtype=int)]), fmt="%d")
    guide_table = groups.iloc[guide_rows]
    pd.DataFrame({"id": guide_table.spacer, "target": guide_table.group,
                  "non_targeting": guide_table.group.isin(ntc).astype(int)}).to_csv(
        output / "guides.tsv", sep="\t", index=False)
    pd.DataFrame({"response": [responses.index(g) for g in pairs["ENSG.targetgene"]],
                  "target": pairs.gRNAgroup.to_numpy(), "kind": pairs.kind.to_numpy()}).to_csv(
        output / "pairs.tsv", sep="\t", index=False)
    covariates = pd.DataFrame({
        "log_umi": np.log(totals), "log_detected": np.log(detected),
        # Guides called per cell (all spacers, not only the pairs' groups); phenoData
        # lists no guides for 1,527 cells.
        "log_guides": np.log1p(np.bincount(cols, minlength=len(cells))), "percent_mito": pheno[17].to_numpy(float),
        "batch2": (pheno[14] == "prep_batch_2").astype(float).to_numpy(),
    })
    covariates.to_csv(output / "covariates.tsv", sep="\t", index=False)
    np.savetxt(output / "design.tsv", np.column_stack([np.ones(len(cells)), covariates.to_numpy()]),
               fmt="%.17g", delimiter="\t")
    (output / "responses.txt").write_text("\n".join(responses) + "\n")
    provenance = {
        "dataset": "Gasperini et al. 2019 at-scale screen, GEO GSE120861",
        "inputs": {k: {"file": str(v.relative_to(ROOT)), "bytes": v.stat().st_size} for k, v in files.items()},
        "cells": len(cells), "responses": len(responses), "guides": len(guide_rows),
        "pairs": {k: int(v) for k, v in pairs.kind.value_counts().items()},
        "unmatched_spacers": unknown, "seed": seed,
        "covariates": list(covariates.columns),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ("cells", "responses", "guides", "pairs", "unmatched_spacers")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["thyroid", "gasperini"])
    ap.add_argument("--output", type=pathlib.Path)
    a = ap.parse_args()
    {"thyroid": thyroid, "gasperini": gasperini}[a.dataset](a.output or WORK / a.dataset)
