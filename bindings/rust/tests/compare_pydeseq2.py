"""Compare the Rust port of PyDESeq2's workflow (examples/deseq.rs) with
PyDESeq2 itself, run as a black box with RapidGLM's inference back end
(rapidglm.pydeseq2.RapidGLMInference), on one dataset directory holding
counts.mtx (genes x samples) and design.tsv (header, one row per sample). The
contrast is the last design column.

Usage (from the repository root, after cargo build --release --examples):

    RAPIDGLM_C_LIBRARY=build/librapidglm_c.so PYTHONPATH=bindings/python \\
        python3 bindings/rust/tests/compare_pydeseq2.py DATA_DIR [--json OUT]

Prints, per result column, the largest relative difference and the numbers of
genes differing by more than 1e-6 and 1e-3, and the agreement of calls at
padj < 0.05. Most differences of 1e-6 come from the dispersion trend, which
PyDESeq2 fits with L-BFGS-B to a looser tolerance.

--deseq2 TABLE adds the genes each tool drops as Cook's outliers, against R
DESeq2 run as a black box on the same design (TABLE: its baseMean and pvalue
per gene, in the same gene order).
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from rapidglm.pydeseq2 import RapidGLMInference

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[3]
COLUMNS = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


def pydeseq2(data, threads):
    counts = scipy.io.mmread(data / "counts.mtx").T.toarray().astype(int)
    design = pd.read_csv(data / "design.tsv", sep="\t")
    index = [f"s{i}" for i in range(counts.shape[0])]
    design.index = index
    frame = pd.DataFrame(counts, index=index, columns=[f"g{j}" for j in range(counts.shape[1])])
    inference = RapidGLMInference(n_cpus=threads)
    affinity = os.sched_getaffinity(0)
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        sys.path.insert(0, str(ROOT / "scripts"))
        from quiet_run import wait_quiet
        cpus = set(range(16, 16 + threads))
        wait_quiet(cpus, label=f"Rust comparison {data.name} PyDESeq2")
        os.sched_setaffinity(0, cpus)
    start = time.perf_counter()
    dds = DeseqDataSet(counts=frame, metadata=pd.DataFrame(index=index), design=design,
                       inference=inference, quiet=True)
    dds.deseq2()
    contrast = np.zeros(design.shape[1])
    contrast[-1] = 1
    stats = DeseqStats(dds, contrast=contrast, inference=inference, quiet=True)
    stats.summary()
    seconds = time.perf_counter() - start
    os.sched_setaffinity(0, affinity)
    return stats.results_df.reset_index(drop=True), dds, seconds, contrast


def rust(data, contrast, workdir):
    path = pathlib.Path(workdir)
    (path / "contrast.txt").write_text(" ".join(str(c) for c in contrast))
    binary = ROOT / "bindings/rust/target/release/examples/deseq"
    command = [str(binary), str(data / "counts.mtx"), str(data / "design.tsv"),
               str(path / "contrast.txt"), str(path / "results.tsv")]
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        sys.path.insert(0, str(ROOT / "scripts"))
        from quiet_run import wait_quiet
        wait_quiet("16-23", label=f"Rust comparison {data.name} native")
        command = ["taskset", "-c", "16-23", *command]
    start = time.perf_counter()
    run = subprocess.run(command,
                         capture_output=True, text=True, check=True)
    seconds = time.perf_counter() - start
    return pd.read_csv(path / "results.tsv", sep="\t"), seconds, run.stderr.strip()


def cooks_outliers(ours, ref, dds, deseq2):
    """Genes whose p-value is dropped as a Cook's outlier (missing, with baseMean > 0)."""
    def dropped(table):
        return set(np.flatnonzero(table["pvalue"].isna().to_numpy() & (table["baseMean"].to_numpy() > 0)))

    def jaccard(a, b):
        return len(a & b) / max(len(a | b), 1)

    r, p, d = dropped(ours), dropped(ref), dropped(deseq2)
    # Samples alone in their design cell have leverage 1.
    design = dds.obsm["design_matrix"]
    cells = design.value_counts()
    single = np.array([cells[tuple(row)] == 1 for row in design.to_numpy()])
    cooks = np.asarray(dds.layers["cooks"])
    largest = np.argmax(np.where(np.isnan(cooks), -np.inf, cooks), axis=0)
    differ = np.array(sorted(r ^ p), dtype=int)
    return {"deseq2": len(d), "rust": len(r), "pydeseq2_rapidglm": len(p),
            "jaccard_rust_deseq2": round(jaccard(r, d), 3), "jaccard_pydeseq2_deseq2": round(jaccard(p, d), 3),
            "singleton_design_cell_samples": int(single.sum()),
            "pydeseq2_infinite_distances": int(np.isinf(cooks).sum()),
            "mismatched_genes": {"total": len(differ), "rust_only": len(r - p), "pydeseq2_only": len(p - r),
                                 "pydeseq2_max_at_singleton_design_cell_sample": int(single[largest[differ]].sum())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=pathlib.Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--deseq2", type=pathlib.Path)
    args = parser.parse_args()
    ref, dds, py_seconds, contrast = pydeseq2(args.data, args.threads)
    with tempfile.TemporaryDirectory() as workdir:
        ours, rust_seconds, log = rust(args.data, contrast, workdir)
    report = {"data": str(args.data), "samples": int(dds.n_obs), "genes": int(dds.n_vars),
              "pydeseq2_rapidglm_seconds": round(py_seconds, 2), "rust_seconds": round(rust_seconds, 2),
              "rust_log": log, "columns": {}}
    for column in COLUMNS:
        a, b = ours[column].to_numpy(), ref[column].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a - b)[both] / np.maximum(np.abs(b[both]), 1e-300)
        report["columns"][column] = {"max_relative": float(rel.max()) if rel.size else 0.0,
                                     "over_1e-6": int((rel > 1e-6).sum()),
                                     "over_1e-3": int((rel > 1e-3).sum()),
                                     "missing_mismatch": int((np.isnan(a) != np.isnan(b)).sum())}
    a, b = set(np.flatnonzero(ours.padj < 0.05)), set(np.flatnonzero(ref.padj < 0.05))
    report["calls"] = {"rust": len(a), "pydeseq2": len(b), "both": len(a & b),
                       "jaccard": len(a & b) / max(len(a | b), 1)}
    if args.deseq2:
        report["cooks_outliers"] = cooks_outliers(ours, ref, dds, pd.read_csv(args.deseq2, sep="\t"))
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["calls"]["jaccard"] > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
