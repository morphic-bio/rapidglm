#!/usr/bin/env python3
"""Panels for the CRT scaling receipt (validation/CRT_SCALING.json): every pair
of T targets and G response genes of the Gasperini benchmark panel written by
scripts/prepare_benchmark_data.py.

Targets are the first T of a random order of the panel's discovery targets
(their sorted names), and genes the first G of a random order of its response
genes, both from numpy's default_rng(20261001), targets first. The sets are
therefore nested. For each TxG the script writes pairs_tT_gG.tsv and, when G is
below the panel's gene count, responses_gG.mtx (the selected rows in their
original order, renumbered) and responses_gG.txt. Pairs index the rows of that
matrix.

    python3 scripts/prepare_crt_panels.py --output DIR 50x1000 200x1000 100x7418
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20261001


def subset_rows(source, target, rows):
    """Rows `rows` (0-based, sorted) of a MatrixMarket file, renumbered, by streaming."""
    with open(source) as f:
        f.readline()
        n_rows, n_cols, _ = map(int, f.readline().split())
        new = np.full(n_rows, -1)
        new[rows] = np.arange(len(rows))
        kept = []
        for chunk in pd.read_csv(f, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64,
                                 chunksize=50_000_000):
            r = new[chunk.r.to_numpy() - 1]
            mask = r >= 0
            kept.append(np.column_stack([r[mask] + 1, chunk.c.to_numpy()[mask], chunk.v.to_numpy()[mask]]))
    entries = np.concatenate(kept)
    with open(target, "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{len(rows)} {n_cols} {len(entries)}\n")
        np.savetxt(out, entries, fmt="%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panels", nargs="+", help="TxG: targets x genes")
    ap.add_argument("--source", type=pathlib.Path, default=ROOT / "validation/work/benchmark/gasperini")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    pairs = pd.read_csv(args.source / "pairs.tsv", sep="\t")
    responses = (args.source / "responses.txt").read_text().split()
    eligible = sorted(set(pairs.loc[pairs.kind == "discovery", "target"]))
    rng = np.random.default_rng(SEED)
    target_order = [eligible[i] for i in rng.permutation(len(eligible))]
    gene_order = [int(i) for i in rng.permutation(len(responses))]
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in args.panels:
        t, g = map(int, spec.split("x"))
        genes = sorted(gene_order[:g])
        if g < len(responses) and not (args.output / f"responses_g{g}.mtx").exists():
            subset_rows(args.source / "responses.mtx", args.output / f"responses_g{g}.mtx", np.array(genes))
            (args.output / f"responses_g{g}.txt").write_text("\n".join(responses[i] for i in genes) + "\n")
        with open(args.output / f"pairs_t{t}_g{g}.tsv", "w") as out:
            out.write("response\ttarget\tkind\n")
            for r in range(g):
                for target in target_order[:t]:
                    out.write(f"{r}\t{target}\tdiscovery\n")
        print(f"{spec}: {t * g:,} pairs")


if __name__ == "__main__":
    main()
