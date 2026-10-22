#!/usr/bin/env python3
"""Prepare the K562 essential-gene low-MOI Perturb-seq screen of Replogle et al.
(2022, Cell 185:2559) for the low-MOI pilot.

Input: K562_essential_raw_singlecell_01.h5ad from the authors' figshare release
(https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387,
file 35773219, CC BY 4.0): 310,385 cells x 8,563 genes of raw UMI counts (dense
float32), one guide pair (sgID_AB) per cell, its target gene, the 10x lane
(gem_group) and the mitochondrial fraction. The file is read with h5py, in
blocks of cells, once.

Writes, under --output:

sc/  a single-cell panel in the format of scripts/reference_sceptre_low_moi.R
  and the CLI's sceptre command: responses.mtx (genes x cells, ordered by cell),
  assignments.mtx (guide pairs x cells, binary), guides.tsv (id, target,
  non_targeting; a non-targeting guide pair is its own target so it can serve as
  a negative control against the other non-targeting cells), pairs.tsv
  (response, target, kind), responses.txt (Ensembl IDs, one per response),
  covariates.tsv (log_umi and log_detected over the matrix's genes, mito from the
  authors' table, batch = gem_group) and cells.tsv. Cells: every non-targeting
  cell and every cell of --targets targeting perturbations, drawn with --seed in
  equal numbers from the quintiles of cells per perturbation, among the
  perturbations whose target gene is in the matrix. Pairs: each drawn target
  against every gene (discovery), against its own gene (positive), and
  --negative-pairs distinct (non-targeting guide pair, gene) pairs (negative).

pseudobulk/  counts.npy (int32, samples x genes) and samples.tsv (group, gem_group,
  cells): the sums of every perturbation's cells, and of the non-targeting cells,
  in each lane where it has cells, for every perturbation of the screen.

provenance.json  the input's MD5, the draw, and the counts.
"""

import argparse
import hashlib
import json
import pathlib

import h5py
import numpy as np


def text(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def categorical(group, name):
    return np.asarray(text(group["__categories"][name][:]), dtype=object)[group[name][:]]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def draw_targets(target, measured, count, seed):
    """`count` targeting perturbations, equal numbers from each quintile of cells
    per perturbation, among those whose target gene is measured."""
    names, cells = np.unique(target[target != "non-targeting"], return_counts=True)
    keep = np.array([measured(n) for n in names])
    names, cells = names[keep], cells[keep]
    order = np.argsort(cells, kind="stable")
    rng = np.random.default_rng(seed)
    chosen = []
    for k, part in enumerate(np.array_split(order, 5)):
        take = count // 5 + (k < count % 5)
        chosen += list(names[rng.choice(part, size=take, replace=False)])
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5ad", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--negative-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--block", type=int, default=8192, help="cells read at a time")
    args = parser.parse_args()
    sc = args.output / "sc"
    pb = args.output / "pseudobulk"
    sc.mkdir(parents=True, exist_ok=True)
    pb.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5ad, "r") as f:
        obs, var, X = f["obs"], f["var"], f["X"]
        n, genes = X.shape
        target = categorical(obs, "gene")
        target_id = categorical(obs, "gene_id")
        guide = categorical(obs, "sgID_AB")
        gem = obs["gem_group"][:]
        mito = obs["mitopercent"][:].astype(float)
        barcodes = np.asarray(text(obs["cell_barcode"][:]), dtype=object)
        gene_ids = text(var["gene_id"][:])
        gene_names = categorical(var, "gene_name")
        gene_index = {g: j for j, g in enumerate(gene_ids)}
        symbol_to_id = dict(zip(target, target_id))

        nt = target == "non-targeting"
        chosen = draw_targets(target, lambda t: symbol_to_id[t] in gene_index, args.targets,
                              args.seed)
        selected = nt | np.isin(target, chosen)
        cells = np.flatnonzero(selected)
        column = np.full(n, -1)
        column[cells] = np.arange(cells.size)

        # Pseudobulk samples: (group, lane) with cells, group = perturbation or NT.
        groups, group_of = np.unique(target, return_inverse=True)
        lanes, lane_of = np.unique(gem, return_inverse=True)
        sample_key = group_of * lanes.size + lane_of
        samples, sample_of, sample_cells = np.unique(sample_key, return_inverse=True,
                                                     return_counts=True)
        sums = np.zeros((samples.size, genes), dtype=np.int64)

        total = np.zeros(n)
        detected = np.zeros(n)
        entries = 0
        temporary = sc / "responses.mtx.body"
        with open(temporary, "w") as body:
            for start in range(0, n, args.block):
                stop = min(n, start + args.block)
                block = X[start:stop, :]
                counts = np.rint(block).astype(np.int64)
                if not np.array_equal(counts, block):
                    raise SystemExit(f"non-integer counts in cells {start}..{stop}")
                total[start:stop] = counts.sum(axis=1)
                detected[start:stop] = (counts > 0).sum(axis=1)
                # Pseudobulk sums: sort the block by sample and reduce.
                order = np.argsort(sample_of[start:stop], kind="stable")
                keys = sample_of[start:stop][order]
                firsts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
                sums[keys[firsts]] += np.add.reduceat(counts[order], firsts, axis=0)
                # Single-cell panel entries, ordered by cell.
                rows = np.flatnonzero(selected[start:stop])
                if rows.size:
                    part = counts[rows]
                    c, g = np.nonzero(part)
                    lines = np.column_stack([g + 1, column[start + rows[c]] + 1, part[c, g]])
                    np.savetxt(body, lines, fmt="%d", delimiter=" ")
                    entries += lines.shape[0]
                print(f"cells {stop}/{n}", flush=True)
    with open(sc / "responses.mtx", "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{genes} {cells.size} {entries}\n")
        with open(temporary) as body:
            for chunk in iter(lambda: body.read(1 << 24), ""):
                out.write(chunk)
    temporary.unlink()

    # Guides: one per guide pair present among the panel's cells.
    panel_guides = sorted(set(guide[cells]))
    guide_column = {g: k for k, g in enumerate(panel_guides)}
    guide_target = {}
    for g, t in zip(guide[cells], target[cells]):
        guide_target[g] = g if t == "non-targeting" else t
    with open(sc / "guides.tsv", "w") as out:
        out.write("id\ttarget\tnon_targeting\n")
        for g in panel_guides:
            out.write(f"{g}\t{guide_target[g]}\t{int(guide_target[g] == g)}\n")
    with open(sc / "assignments.mtx", "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{len(panel_guides)} {cells.size} {cells.size}\n")
        for k, i in enumerate(cells):
            out.write(f"{guide_column[guide[i]] + 1} {k + 1} 1\n")
    with open(sc / "responses.txt", "w") as out:
        out.write("".join(f"{g}\n" for g in gene_ids))
    with open(sc / "covariates.tsv", "w") as out:
        out.write("log_umi\tlog_detected\tmito\tbatch\n")
        for i in cells:
            out.write(f"{float(np.log(total[i]))!r}\t{float(np.log(detected[i]))!r}\t"
                      f"{float(mito[i])!r}\t{gem[i]}\n")
    with open(sc / "cells.tsv", "w") as out:
        out.write("cell\ttarget\tguide\tgem_group\n")
        for b, i in zip(barcodes[cells], cells):
            out.write(f"{b}\t{target[i]}\t{guide[i]}\t{gem[i]}\n")
    rng = np.random.default_rng(args.seed + 1)
    nt_targets = [g for g in panel_guides if guide_target[g] == g]
    negative = set()
    while len(negative) < args.negative_pairs:
        negative.add((int(rng.integers(genes)), nt_targets[rng.integers(len(nt_targets))]))
    with open(sc / "pairs.tsv", "w") as out:
        out.write("response\ttarget\tkind\n")
        for t in chosen:
            for j in range(genes):
                out.write(f"{j}\t{t}\tdiscovery\n")
        for t in chosen:
            out.write(f"{gene_index[symbol_to_id[t]]}\t{t}\tpositive\n")
        for j, t in sorted(negative, key=lambda p: (p[1], p[0])):
            out.write(f"{j}\t{t}\tnegative\n")

    np.save(pb / "counts.npy", sums.astype(np.int32))
    if sums.max() > np.iinfo(np.int32).max:
        raise SystemExit("pseudobulk sums overflow int32")
    with open(pb / "samples.tsv", "w") as out:
        out.write("group\tgem_group\tcells\n")
        for key, c in zip(samples, sample_cells):
            out.write(f"{groups[key // lanes.size]}\t{lanes[key % lanes.size]}\t{c}\n")
    with open(pb / "genes.tsv", "w") as out:
        out.write("gene_id\tgene_name\n")
        out.write("".join(f"{a}\t{b}\n" for a, b in zip(gene_ids, gene_names)))

    per_target = {t: int((target == t).sum()) for t in chosen}
    provenance = {
        "input": str(args.h5ad), "input_md5": md5(args.h5ad),
        "source": "Replogle et al. 2022, figshare file 35773219 (CC BY 4.0)",
        "seed": args.seed, "targets": per_target,
        "cells": int(cells.size), "non_targeting_cells": int(nt.sum()),
        "genes": genes, "entries": entries,
        "nt_guides": len(nt_targets), "negative_pairs": args.negative_pairs,
        "pseudobulk_samples": int(samples.size), "pseudobulk_groups": int(groups.size),
        "lanes": int(lanes.size),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
