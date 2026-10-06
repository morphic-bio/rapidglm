#!/usr/bin/env python3
"""Read-only S5 export and target-direction comparison, using frozen guide calls."""

import gzip, json, re, pathlib, subprocess
import h5py, numpy as np, pandas as pd
from scipy import sparse, io
from validate_glm import ROOT, sha

BASE = pathlib.Path("/mnt/pikachu/catatac_gse288996")
source = BASE / "qc_endpoint/catatac_dmso1_trimodal_fdr.h5mu"
parity = BASE / "biological_parity/dmso1_20260616"
out = ROOT / "validation/work/multimodal"
out.mkdir(parents=True, exist_ok=True)


def strings(node):
    if isinstance(node, h5py.Group):
        cats = np.array(node["categories"]).astype(str)
        codes = np.array(node["codes"])
        return np.array([cats[i] if i >= 0 else "" for i in codes])
    return np.array(node).astype(str)


def matrix(g):
    return sparse.csr_matrix(
        (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g.attrs["shape"])
    )


with h5py.File(source) as f:
    genes = strings(f["mod/rna/var/_index"])
    peaks = strings(f["mod/atac/var/_index"])
    rna = matrix(f["mod/rna/X"])
    atac = matrix(f["mod/atac/X"])
    guide = f["mod/guide/obs"]
    assigned = strings(guide["assigned_gene"])
    status = strings(guide["call_status"])
    # Preserve the existing target-level singlet policy of the endpoint.
    valid = status == "singlet"
    targets = pd.read_csv(
        parity / "target_gene_expression_knockdown.tsv", sep="\t"
    ).target.tolist()
    target_names = sorted(set(assigned[valid]) | set(targets))
    nt = {v for v in target_names if v.startswith("NTC")}
    guide_rows = []
    guide_cols = []
    for j, target in enumerate(target_names):
        cells = np.where(valid & (assigned == target))[0]
        guide_rows.extend([j] * len(cells))
        guide_cols.extend(cells)
    calls = sparse.coo_matrix(
        (np.ones(len(guide_rows), dtype=int), (guide_rows, guide_cols)),
        shape=(len(target_names), rna.shape[0]),
    )
    io.mmwrite(out / "assignments.mtx", calls, field="integer")
    pd.DataFrame(
        [dict(id=t, target=t, non_targeting=int(t in nt)) for t in target_names]
    ).to_csv(out / "guides.tsv", sep="\t", index=False)
    totals = {
        "rna": np.array(rna.sum(axis=1)).ravel(),
        "atac": np.array(atac.sum(axis=1)).ravel(),
    }
# Target TSS windows from the exact annotation and window width in the prior receipt.
provenance = json.loads((parity / "provenance.json").read_text())
windows = {t: [] for t in targets}
with gzip.open(provenance["gtf"], "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip().split("\t")
        if fields[2] != "gene":
            continue
        m = re.search(r'gene_name "([^"]+)"', fields[8])
        target = m.group(1) if m else ""
        if target in windows:
            tss = int(fields[3]) - 1 if fields[6] == "+" else int(fields[4]) - 1
            windows[target].append((fields[0], max(0, tss - 50000), tss + 50000))
peak_coords = [
    (chrom, int(start), int(end))
    for chrom, start, end in (re.split("[:-]", p) for p in peaks)
]
loci = []
locus_sizes = []
for target in targets:
    indices = [
        i
        for i, (c, s, e) in enumerate(peak_coords)
        if any(c == wc and s < we and e > ws for wc, ws, we in windows[target])
    ]
    locus_sizes.append(len(indices))
    loci.append(sparse.csr_matrix(atac[:, indices].sum(axis=1)))
# Each response is a raw target-gene count or a sum of raw peaks in its TSS window.
matrices = {
    "rna": rna[:, [np.where(genes == t)[0][0] for t in targets]],
    "atac": sparse.hstack(loci).tocsr(),
}
results = []
for modality, y in matrices.items():
    d = out / modality
    d.mkdir(exist_ok=True)
    io.mmwrite(d / "responses.mtx", y.T, field="integer")
    np.savetxt(
        d / "design.tsv",
        np.column_stack([np.ones(y.shape[0]), np.log1p(totals[modality])]),
        delimiter="\t",
    )
    pairs = [dict(response=j, target=t, kind="positive") for j, t in enumerate(targets)]
    pd.DataFrame(pairs).to_csv(d / "pairs.tsv", sep="\t", index=False)
    config = dict(
        responses=str(d / "responses.mtx"),
        design=str(d / "design.tsv"),
        assignments=str(out / "assignments.mtx"),
        guides=str(out / "guides.tsv"),
        pairs=str(d / "pairs.tsv"),
        output=str(d / "native"),
        moi="low",
        cell_qc=False,
        family="nb",
        threads=2,
        seed=20260926,
        budgets=[499, 4999],
        effect_fits="all",
    )
    (d / "config.json").write_text(json.dumps(config, indent=2))
    subprocess.run([ROOT / "build/rapidglm", "sceptre", d / "config.json"], check=True)
    native = pd.read_csv(d / "native.tsv", sep="\t")
    reference = pd.read_csv(
        parity
        / (
            "target_gene_expression_knockdown.tsv"
            if modality == "rna"
            else "target_locus_accessibility_summary.tsv"
        ),
        sep="\t",
    )
    merged = native.merge(reference, on="target")
    merged["native_paper_direction_agree"] = np.sign(merged.log2fc) == np.sign(
        merged.paper_delta
    )
    merged.to_csv(d / "direction_comparison.tsv", sep="\t", index=False)
    results.append(
        dict(
            modality=modality,
            tested=len(merged),
            valid=int(merged.log2fc.notna().sum()),
            paper_direction_agreement=int(merged.native_paper_direction_agree.sum()),
            differences=merged.loc[
                ~merged.native_paper_direction_agree,
                ["target", "status", "log2fc", "paper_delta", "suite_delta"],
            ]
            .replace({np.nan: None})
            .to_dict("records"),
            run=json.loads((d / "native.run.json").read_text()),
            result_sha256=sha(d / "direction_comparison.tsv"),
        )
    )
# DA panel: high-support peaks overlapping tested target loci, group BRD2 vs explicit NT cells.
group_target = "BRD2"
keep = valid & ((assigned == group_target) | np.isin(assigned, list(nt)))
peak_total = np.array(atac[keep].sum(axis=0)).ravel()
selected = np.argsort(-peak_total, kind="stable")[:100]
d = out / "da"
d.mkdir(exist_ok=True)
io.mmwrite(d / "responses.mtx", atac[keep][:, selected].T, field="integer")
np.savetxt(d / "group.tsv", (assigned[keep] == group_target).astype(int))
np.savetxt(
    d / "design.tsv",
    np.column_stack([np.ones(sum(keep)), np.log1p(totals["atac"][keep])]),
    delimiter="\t",
)
config = dict(
    responses=str(d / "responses.mtx"),
    design=str(d / "design.tsv"),
    group=str(d / "group.tsv"),
    output=str(d / "native"),
    family="binomial",
    threads=2,
)
(d / "config.json").write_text(json.dumps(config, indent=2))
subprocess.run([ROOT / "build/rapidglm", "da-lr", d / "config.json"], check=True)
receipt = dict(
    gate="G7",
    source=str(source),
    source_sha256=sha(source),
    annotation=provenance["gtf"],
    annotation_sha256=sha(provenance["gtf"]),
    window_bp=50000,
    targets=targets,
    peaks_per_locus=locus_sizes,
    command="python3 scripts/validate_multimodal.py",
    scope="DMSO1 frozen target-singlet calls; depth-adjusted NB effect vs published normalized descriptive direction; no DASA claim",
    results=results,
)
(ROOT / "validation/MULTIMODAL.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n"
)
print(json.dumps(results, indent=2))
