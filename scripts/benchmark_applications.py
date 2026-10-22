#!/usr/bin/env python3
"""Application benchmark for the app note: RapidGLM's bulk DE and perturbation
workflows against DESeq2 and SCEPTRE (black-box runs of their public APIs).

Bulk datasets (validation/work/benchmark/<name>/, see prepare_benchmark_data.py):
  airway (8 samples), thyroid subsamples of 32 and 128 samples, and the whole
  GTEx thyroid set (647 samples, 28,769 genes). Subsamples are drawn once with a
  fixed seed from the thyroid samples and keep every gene.
Tools:
  rapidglm-cli      the CLI (`bulk-de`), whole process including input parsing;
  rapidglm-python   rapidglm.bulk_de through the C interface (call only);
  rapidglm-deseq    the PyDESeq2-style API (DeseqDataSet + DeseqStats, calls only);
  rapidglm-r        rapidglm::bulk_de in R through the C interface (call only;
                    the package in validation/local-data/R_rapidglm, R's BLAS);
  pydeseq2          PyDESeq2 0.5 (black box; its own environment under
                    validation/local-data/pydeseq2-venv): DeseqDataSet (with
                    its input validation) + deseq2() + DeseqStats.summary();
  pydeseq2-rapidglm the same with inference = rapidglm.pydeseq2.RapidGLMInference;
  deseq2            DESeq2 DESeq() + results(), parallel = MulticoreParam(threads);
  deseq2-glmgampoi  the same with fitType = "glmGamPoi" (DESeq2 does not
                    parallelize it, so it runs on one worker only);
  rapidglm-edger-cli the CLI (`edger-ql`), whole process including input parsing;
  edger             edgeR's default quasi-likelihood workflow (DGEList,
                    filterByExpr, normLibSizes, glmQLFit, glmQLFTest, topTags;
                    one thread, scripts/reference_edger_timing.R).
Perturbation datasets: the Gasperini et al. (2019) at-scale screen panel
  (gasperini: 207,324 cells; positive controls, 10,000 enhancer-gene pairs and
  5,000 non-targeting pairs) and a 50,000-cell subsample (gasperini50k).
Tools:
  rapidglm-cli      `sceptre` (high MOI, complement controls, CRT), whole process;
  rapidglm-python   rapidglm.sceptre through the C interface (call only);
  rapidglm-r        rapidglm::sceptre in R through the C interface (call only);
  rapidglm-*-ks     the same three with fit_check = "ks", which accepts a
                    skew-normal fit as SCEPTRE does (the like-for-like runs; the
                    plain names run RapidGLM's defaults);
  sceptre           SCEPTRE 0.99.0 run_calibration_check + run_power_check +
                    run_discovery_analysis (parallel = TRUE, n_processors = threads).
Agreement with SCEPTRE (1 worker) over discovery pairs: Jaccard of BH
discoveries at 0.1 and Spearman correlation of p-values.

Each tool runs at 1 and 8 threads (where supported), --repeats times in rotation
(tools alternate within a repeat); the minimum wall time is reported with the
host load before each run. Runs with a 1-minute load above a quarter of the
cores are marked contended. --cpus THREADS:LIST confines the runs at a thread
count to those CPUs (for example 1:16 8:16-23), which matters on processors
whose cores differ in speed. A confined run starts once other processes use
less than a fifth of its CPUs (or after --idle-timeout seconds), and the share
of those CPUs that other processes used during it is recorded (others_cpu; an
upper bound, since a tool's unreaped worker processes count as others). Such
runs are marked contended when that share exceeds a fifth, instead of by the
load. Agreement with DESeq2 (parametric, 1 thread):
Jaccard index of BH discoveries (padj < 0.1) and Spearman correlation of
p-values over genes both tested.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "validation/work/benchmark"
R_LIBS = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
CORES = os.cpu_count()


def load():
    return os.getloadavg()


ALLOWED = os.sched_getaffinity(0)


def cpu_list(text):
    """THREADS:LIST, for example 8:0-7 or 1:4,5."""
    threads, _, spec = text.partition(":")
    cpus = set()
    for part in spec.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return int(threads), cpus


TICKS = os.sysconf("SC_CLK_TCK")


def busy_seconds(cpus):
    """Seconds the given CPUs have spent busy (all but idle and I/O wait), from /proc/stat."""
    jiffies = 0
    with open("/proc/stat") as f:
        for line in f:
            name, *values = line.split()
            if name[:3] == "cpu" and name[3:].isdigit() and int(name[3:]) in cpus:
                values = [int(v) for v in values]
                jiffies += sum(values[:3]) + sum(values[5:8])
    return jiffies / TICKS


def timed_run(run, threads, args, *arguments):
    """Runs a tool on the CPUs for its thread count (inherited from this process's
    affinity). Returns the tool's result and a record of the host around the run."""
    cpus = dict(args.cpus).get(threads)
    os.sched_setaffinity(0, cpus or ALLOWED)
    record = {"load": load()[0]}
    if cpus:
        # Wait for other processes to leave these CPUs.
        start = time.time()
        while True:
            before = busy_seconds(cpus)
            time.sleep(3)
            if (busy_seconds(cpus) - before) / (3 * len(cpus)) < 0.2 or time.time() - start > args.idle_timeout:
                break
        record["waited_for_idle_seconds"] = round(time.time() - start - 3, 1)
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        from quiet_run import wait_quiet, wait_memory
        if os.environ.get("RAPIDGLM_MIN_AVAILABLE_GIB"):
            wait_memory(float(os.environ["RAPIDGLM_MIN_AVAILABLE_GIB"]))
        record["waited_for_quiet_host_seconds"] = wait_quiet(cpus, label=f"{run.__name__} threads={threads}")
    record["load"] = load()[0]
    if cpus:
        busy, own = busy_seconds(cpus), os.times()
    result = run(*arguments)
    if cpus:
        after = os.times()
        ours = sum(after[:4]) - sum(own[:4])  # this process and the children it has waited for
        wall = after.elapsed - own.elapsed
        record["others_cpu"] = round(max(0.0, busy_seconds(cpus) - busy - ours) / (wall * len(cpus)), 4)
        record["contended"] = record["others_cpu"] > 0.2
    else:
        record["contended"] = record["load"] > CORES / 4
    return result, record


# Datasets.


def airway():
    d = ROOT / "validation/work/bulk/airway"
    return dict(dir=d, coefficient="dextrt", index=4)


def thyroid_subsample(n, seed=20260930):
    source = WORK / "thyroid"
    d = WORK / f"thyroid{n}"
    if not (d / "counts.mtx").exists():
        import scipy.io

        counts = scipy.io.mmread(source / "counts.mtx").tocsc()
        design = pd.read_csv(source / "design.tsv", sep="\t")
        rng = np.random.default_rng(seed + n)
        pick = np.sort(rng.choice(counts.shape[1], n, replace=False))
        x = design.iloc[pick]
        keep = x.columns[(x != 0).any(axis=0) & ~((x == 1).all(axis=0) & (x.columns != "Intercept"))]
        x = x[keep]
        d.mkdir(parents=True, exist_ok=True)
        scipy.io.mmwrite(d / "counts.mtx", counts[:, pick].astype(np.int64), field="integer")
        x.to_csv(d / "design.tsv", sep="\t", index=False)
        np.savetxt(d / "x.tsv", x.to_numpy(), fmt="%g", delimiter="\t")
        (d / "samples.txt").write_text("\n".join(map(str, pick)) + "\n")
    design = pd.read_csv(d / "design.tsv", sep="\t", nrows=0)
    return dict(dir=d, coefficient="sex[T.female]", index=list(design.columns).index("sex[T.female]"))


def thyroid():
    d = WORK / "thyroid"
    design = pd.read_csv(d / "design.tsv", sep="\t", nrows=0)
    return dict(dir=d, coefficient="sex[T.female]", index=list(design.columns).index("sex[T.female]"))


BULK = {
    "airway": airway,
    "thyroid32": lambda: thyroid_subsample(32),
    "thyroid128": lambda: thyroid_subsample(128),
    "thyroid": thyroid,
}

# Tools. Each returns (seconds, table of gene, pvalue, padj).

PYTHON_BULK = r"""
import json, sys, time, numpy as np, pandas as pd, scipy.io
sys.path.insert(0, sys.argv[1])
import rapidglm
d, index, threads, mode, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
counts = scipy.io.mmread(f"{d}/counts.mtx").T.tocsc()
design = pd.read_csv(f"{d}/design.tsv", sep="\t")
if mode == "bulk_de":
    t = time.perf_counter()
    r = rapidglm.bulk_de(design.to_numpy(), counts, coefficient=index, threads=threads)
    seconds = time.perf_counter() - t
else:
    from rapidglm.deseq import DeseqDataSet, DeseqStats
    frame = pd.DataFrame(counts.toarray(), index=[f"s{i}" for i in range(counts.shape[0])])
    x = design.set_index(frame.index)
    contrast = np.zeros(x.shape[1]); contrast[index] = 1
    t = time.perf_counter()
    dds = DeseqDataSet(counts=frame, metadata=x, design=x, max_disp=None, n_cpus=threads, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=contrast, alpha=0.1, quiet=True)
    r = ds.summary()
    seconds = time.perf_counter() - t
pd.DataFrame({"pvalue": r["pvalue"].to_numpy(), "padj": r["padj"].to_numpy()}).to_csv(out, sep="\t", index=False)
print(json.dumps({"seconds": seconds}))
"""


PYDESEQ2_VENV = ROOT / "validation/local-data/pydeseq2-venv/bin/python"
PYTHON_PYDESEQ2 = r"""
import json, sys, time, warnings, numpy as np, pandas as pd, scipy.io
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
d, index, threads, backend, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
counts = scipy.io.mmread(f"{d}/counts.mtx").T.toarray().astype(np.int64)
design = pd.read_csv(f"{d}/design.tsv", sep="	")
samples = [f"s{i}" for i in range(counts.shape[0])]
frame = pd.DataFrame(counts, index=samples, columns=[f"g{j}" for j in range(counts.shape[1])])
x = design.set_index(pd.Index(samples))
if backend == "rapidglm":
    from rapidglm.pydeseq2 import RapidGLMInference
    inference = RapidGLMInference(n_cpus=threads)
else:
    inference = DefaultInference(n_cpus=threads)
contrast = np.zeros(x.shape[1]); contrast[index] = 1
t = time.perf_counter()
dds = DeseqDataSet(counts=frame, metadata=x, design=x, inference=inference, quiet=True)
dds.deseq2()
ds = DeseqStats(dds, contrast=contrast, alpha=0.1, inference=inference, quiet=True)
ds.summary()
seconds = time.perf_counter() - t
r = ds.results_df
pd.DataFrame({"pvalue": r["pvalue"].to_numpy(), "padj": r["padj"].to_numpy()}).to_csv(out, sep="	", index=False)
print(json.dumps({"seconds": seconds}))
"""


def run_pydeseq2(backend):
    def run(binary, data, threads, out):
        r = subprocess.run(
            [str(PYDESEQ2_VENV), "-c", PYTHON_PYDESEQ2, str(ROOT / "bindings/python"), str(data["dir"]),
             str(data["index"]), str(threads), backend, str(out / "pydeseq2.tsv")],
            check=True, env={**ENV, "RAPIDGLM_C_LIBRARY": str(binary.parent / "librapidglm_c.so")},
            capture_output=True, text=True,
        )
        return json.loads(r.stdout.splitlines()[-1])["seconds"], pd.read_csv(out / "pydeseq2.tsv", sep="\t")
    return run


def run_cli(binary, data, threads, out):
    config = out / "config.json"
    config.write_text(json.dumps({
        "responses": str(data["dir"] / "counts.mtx"), "design": str(data["dir"] / "x.tsv"),
        "coefficient": data["index"], "threads": threads, "output": str(out / "rapid"),
    }))
    t = time.perf_counter()
    subprocess.run([binary, "bulk-de", config], check=True, env=ENV, capture_output=True)
    seconds = time.perf_counter() - t
    return seconds, pd.read_csv(out / "rapid.tsv", sep="\t", usecols=["pvalue", "padj"])


def run_python(mode):
    def run(binary, data, threads, out):
        library = binary.parent / "librapidglm_c.so"
        r = subprocess.run(
            [sys.executable, "-c", PYTHON_BULK, str(ROOT / "bindings/python"), str(data["dir"]),
             str(data["index"]), str(threads), mode, str(out / "python.tsv")],
            check=True, env={**ENV, "RAPIDGLM_C_LIBRARY": str(library)}, capture_output=True, text=True,
        )
        return json.loads(r.stdout.splitlines()[-1])["seconds"], pd.read_csv(out / "python.tsv", sep="\t")
    return run


def run_deseq2(fit_type):
    def run(binary, data, threads, out):
        subprocess.run(
            ["Rscript", str(ROOT / "scripts/reference_bulk_timing.R"), str(data["dir"]), data["coefficient"],
             str(out), R_LIBS, fit_type, str(threads)],
            check=True, env=ENV, capture_output=True,
        )
        seconds = json.loads((out / "timing.json").read_text())["seconds"]
        return seconds, pd.read_csv(out / "genes.tsv", sep="\t", usecols=["pvalue", "padj"])
    return run


def run_edger_cli(binary, data, threads, out):
    config = out / "config.json"
    config.write_text(json.dumps({
        "responses": str(data["dir"] / "counts.mtx"), "design": str(data["dir"] / "x.tsv"),
        "coefficient": data["index"], "threads": threads, "output": str(out / "rapid"),
    }))
    t = time.perf_counter()
    subprocess.run([binary, "edger-ql", config], check=True, env=ENV, capture_output=True)
    seconds = time.perf_counter() - t
    table = pd.read_csv(out / "rapid.tsv", sep="\t", usecols=["PValue", "FDR"], na_values="NA",
                        keep_default_na=False)
    return seconds, table.rename(columns={"PValue": "pvalue", "FDR": "padj"})


def run_edger(binary, data, threads, out):
    subprocess.run(["Rscript", str(ROOT / "scripts/reference_edger_timing.R"), str(data["dir"]),
                    data["coefficient"], str(out), R_LIBS], check=True, env=ENV, capture_output=True)
    seconds = json.loads((out / "timing.json").read_text())["seconds"]
    return seconds, pd.read_csv(out / "genes.tsv", sep="\t", usecols=["pvalue", "padj"])


RAPIDGLM_R_LIBS = f"{ROOT / 'validation/local-data/R_rapidglm'}:{ROOT / 'validation/local-data/R_library'}"


def run_rapidglm_r(binary, data, threads, out):
    subprocess.run(["Rscript", str(ROOT / "scripts/rapidglm_r_timing.R"), "bulk", str(data["dir"]),
                    str(data["index"]), str(threads), str(out), RAPIDGLM_R_LIBS],
                   check=True, env=ENV, capture_output=True)
    seconds = json.loads((out / "timing.json").read_text())["seconds"]
    return seconds, pd.read_csv(out / "r.tsv", sep="\t")


TOOLS = {
    "rapidglm-cli": (run_cli, (1, 8)),
    "rapidglm-python": (run_python("bulk_de"), (1, 8)),
    "rapidglm-deseq": (run_python("deseq"), (1, 8)),
    "rapidglm-r": (run_rapidglm_r, (1, 8)),
    "pydeseq2": (run_pydeseq2("default"), (1, 8)),
    "pydeseq2-rapidglm": (run_pydeseq2("rapidglm"), (1, 8)),
    "deseq2": (run_deseq2("parametric"), (1, 8)),
    "deseq2-glmgampoi": (run_deseq2("glmGamPoi"), (1,)),
    "rapidglm-edger-cli": (run_edger_cli, (1, 8)),
    "edger": (run_edger, (1,)),
}


def agreement(table, reference, alpha=0.1):
    from scipy.stats import spearmanr

    a = set(np.flatnonzero(table["padj"].to_numpy() < alpha))
    b = set(np.flatnonzero(reference["padj"].to_numpy() < alpha))
    both = table["pvalue"].notna().to_numpy() & reference["pvalue"].notna().to_numpy()
    return {
        "discoveries": len(a),
        "jaccard": len(a & b) / len(a | b) if a | b else 1.0,
        "spearman": float(spearmanr(table["pvalue"][both], reference["pvalue"][both]).statistic),
    }


def bulk(args):
    results = []
    for name in args.datasets:
        data = BULK[name]()
        runs = [(tool, threads) for tool in args.tools for threads in TOOLS[tool][1]
                if threads <= args.max_threads and threads in args.threads]
        tables, timings = {}, {run: [] for run in runs}
        for repeat in range(args.repeats):
            for tool, threads in runs:
                out = args.work / name / f"{tool}-t{threads}"
                out.mkdir(parents=True, exist_ok=True)
                (seconds, table), record = timed_run(TOOLS[tool][0], threads, args, args.binary, data, threads, out)
                timings[(tool, threads)].append({"seconds": seconds, **record})
                tables[(tool, threads)] = table
                print(f"{name} {tool} t{threads} repeat {repeat}: {seconds:.2f} s {record}", flush=True)
        reference = tables.get(("deseq2", 1))
        edger_reference = tables.get(("edger", 1))
        for (tool, threads), runs_ in timings.items():
            best = min(runs_, key=lambda r: r["seconds"])
            entry = {"dataset": name, "tool": tool, "threads": threads, "seconds": best["seconds"],
                     "all_seconds": [r["seconds"] for r in runs_], "load": best["load"],
                     "contended": best["contended"], "runs": runs_}
            if reference is not None:
                entry["agreement_with_deseq2"] = agreement(tables[(tool, threads)], reference)
            if edger_reference is not None:
                entry["agreement_with_edger"] = agreement(tables[(tool, threads)], edger_reference)
            results.append(entry)
    return results


# Perturbation panels.

# The R library that holds SCEPTRE 0.99.0 (the comparator, run as a black box).
SCEPTRE_LIBRARY = os.environ.get("SCEPTRE_LIBRARY", str(ROOT / "validation/local-data/R_library"))


def subsample_market(source, target, cells):
    """Columns (cells) `cells` of a MatrixMarket file, renumbered, by streaming."""
    new = np.full(cells.max() + 1 if len(cells) else 0, -1)
    new[cells] = np.arange(len(cells))
    kept = []
    with open(source) as f:
        f.readline()
        rows, cols, _ = map(int, f.readline().split())
        new = np.concatenate([new, np.full(cols + 1 - len(new), -1)])
        for chunk in pd.read_csv(f, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64,
                                 chunksize=50_000_000):
            c = new[chunk.c.to_numpy() - 1]
            mask = c >= 0
            kept.append(np.column_stack([chunk.r.to_numpy()[mask], c[mask] + 1, chunk.v.to_numpy()[mask]]))
    entries = np.concatenate(kept)
    with open(target, "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{rows} {len(cells)} {len(entries)}\n")
        np.savetxt(out, entries, fmt="%d")


def gasperini_subsample(n, seed=20260930):
    source = WORK / "gasperini"
    d = WORK / f"gasperini{n // 1000}k"
    if not (d / "pairs.tsv").exists():
        d.mkdir(parents=True, exist_ok=True)
        design = np.loadtxt(source / "design.tsv")
        rng = np.random.default_rng(seed)
        cells = np.sort(rng.choice(len(design), n, replace=False))
        for name in ("responses.mtx", "assignments.mtx"):
            subsample_market(source / name, d / name, cells)
        np.savetxt(d / "design.tsv", design[cells], fmt="%.17g", delimiter="\t")
        pd.read_csv(source / "covariates.tsv", sep="\t").iloc[cells].to_csv(d / "covariates.tsv", sep="\t",
                                                                           index=False)
        (d / "cells.txt").write_text("\n".join(map(str, cells)) + "\n")
        for name in ("guides.tsv", "responses.txt", "pairs.tsv"):  # pairs last: marks completion
            (d / name).write_bytes((source / name).read_bytes())
    return d


SCEPTRE_DATA = {
    "gasperini50k": lambda: gasperini_subsample(50_000),
    "gasperini": lambda: WORK / "gasperini",
}

PYTHON_SCEPTRE = r"""
import json, sys, time, numpy as np, pandas as pd, scipy.io
sys.path.insert(0, sys.argv[1])
import rapidglm
d, threads, out, fit_check = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
y = scipy.io.mmread(f"{d}/responses.mtx").T.tocsc()
g = scipy.io.mmread(f"{d}/assignments.mtx").T.tocsc()
x = np.loadtxt(f"{d}/design.tsv")
guides = pd.read_csv(f"{d}/guides.tsv", sep="	")
pairs = pd.read_csv(f"{d}/pairs.tsv", sep="	")
t = time.perf_counter()
r = rapidglm.sceptre(x, y, g, guides, pairs, moi="high", cell_qc=False, threads=threads, seed=20260930,
                     fit_check=fit_check)
seconds = time.perf_counter() - t
r.to_csv(out, sep="	", index=False)
print(json.dumps({"seconds": seconds}))
"""


def sceptre_cli(fit_check):
    def run(binary, d, threads, out):
        config = out / "config.json"
        config.write_text(json.dumps({
            "responses": str(d / "responses.mtx"), "assignments": str(d / "assignments.mtx"),
            "design": str(d / "design.tsv"), "guides": str(d / "guides.tsv"), "pairs": str(d / "pairs.tsv"),
            "moi": "high", "cell_qc": False, "threads": threads, "seed": 20260930, "fit_check": fit_check,
            "output": str(out / "rapid"),
        }))
        t = time.perf_counter()
        subprocess.run([binary, "sceptre", config], check=True, env=ENV, capture_output=True)
        seconds = time.perf_counter() - t
        return seconds, pd.read_csv(out / "rapid.tsv", sep="\t")
    return run


def sceptre_python(fit_check):
    def run(binary, d, threads, out):
        r = subprocess.run(
            [sys.executable, "-c", PYTHON_SCEPTRE, str(ROOT / "bindings/python"), str(d), str(threads),
             str(out / "python.tsv"), fit_check],
            check=True, env={**ENV, "RAPIDGLM_C_LIBRARY": str(binary.parent / "librapidglm_c.so")},
            capture_output=True, text=True,
        )
        table = pd.read_csv(out / "python.tsv", sep="\t")
        table["kind"] = table["kind"].map({"discovery": 0, "negative": 1, "positive": 2})
        return json.loads(r.stdout.splitlines()[-1])["seconds"], table
    return run


def sceptre_r(binary, d, threads, out):
    subprocess.run(["Rscript", str(ROOT / "scripts/reference_sceptre_timing.R"), str(d), SCEPTRE_LIBRARY,
                    str(out), str(threads)], check=True, env=ENV, capture_output=True)
    timing = json.loads((out / "timing.json").read_text())
    return timing["seconds"], pd.read_csv(out / "discovery.tsv", sep="\t")


def sceptre_rapidglm_r(fit_check):
    def run(binary, d, threads, out):
        subprocess.run(["Rscript", str(ROOT / "scripts/rapidglm_r_timing.R"), "sceptre", str(d), str(threads),
                        str(out), fit_check, RAPIDGLM_R_LIBS], check=True, env=ENV, capture_output=True)
        seconds = json.loads((out / "timing.json").read_text())["seconds"]
        return seconds, pd.read_csv(out / "r.tsv", sep="\t")
    return run


SCEPTRE_TOOLS = {
    "rapidglm-cli": (sceptre_cli("strict"), (1, 8)),
    "rapidglm-python": (sceptre_python("strict"), (1, 8)),
    "rapidglm-r": (sceptre_rapidglm_r("strict"), (1, 8)),
    "rapidglm-cli-ks": (sceptre_cli("ks"), (1, 8)),
    "rapidglm-python-ks": (sceptre_python("ks"), (1, 8)),
    "rapidglm-r-ks": (sceptre_rapidglm_r("ks"), (1, 8)),
    "sceptre": (sceptre_r, (1, 8)),
}


def discovery_pvalues(table, d):
    """Discovery-pair p-values keyed by (response id, target)."""
    responses = (d / "responses.txt").read_text().split()
    if "response_id" in table:  # SCEPTRE
        return table.set_index(["response_id", "grna_target"])["p_value"]
    t = table[table["kind"] == 0]
    return pd.Series(t["p_value"].to_numpy(),
                     index=pd.MultiIndex.from_arrays([[responses[int(i)] for i in t["response"]],
                                                      t["target"]]))


def sceptre_agreement(table, reference, d, alpha=0.1):
    from scipy.stats import spearmanr

    a, b = discovery_pvalues(table, d), discovery_pvalues(reference, d)
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    ok = a.notna() & b.notna()

    def bh(p):
        p = p.to_numpy()
        order = np.argsort(p)
        q = p[order] * len(p) / np.arange(1, len(p) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q)
        out[order] = np.minimum(q, 1)
        return out

    da = set(np.flatnonzero(bh(a[ok]) < alpha))
    db = set(np.flatnonzero(bh(b[ok]) < alpha))
    return {"pairs": int(ok.sum()), "discoveries": len(da),
            "jaccard": len(da & db) / len(da | db) if da | db else 1.0,
            "spearman": float(spearmanr(a[ok], b[ok]).statistic)}


def perturbation(args):
    results = []
    for name in args.sceptre_datasets:
        d = SCEPTRE_DATA[name]()
        runs = [(tool, threads) for tool in args.sceptre_tools for threads in SCEPTRE_TOOLS[tool][1]
                if threads <= args.max_threads and threads in args.threads]
        tables, timings = {}, {run: [] for run in runs}
        for repeat in range(args.repeats):
            for tool, threads in runs:
                out = args.work / name / f"{tool}-t{threads}"
                out.mkdir(parents=True, exist_ok=True)
                (seconds, table), record = timed_run(SCEPTRE_TOOLS[tool][0], threads, args, args.binary, d, threads,
                                                     out)
                timings[(tool, threads)].append({"seconds": seconds, **record})
                tables[(tool, threads)] = table
                print(f"{name} {tool} t{threads} repeat {repeat}: {seconds:.2f} s {record}", flush=True)
        # SCEPTRE at one thread, else at the fewest threads run.
        sceptre_runs = sorted(k for k in tables if k[0] == "sceptre")
        reference = tables[sceptre_runs[0]] if sceptre_runs else None
        for (tool, threads), runs_ in timings.items():
            best = min(runs_, key=lambda r: r["seconds"])
            entry = {"dataset": name, "tool": tool, "threads": threads, "seconds": best["seconds"],
                     "all_seconds": [r["seconds"] for r in runs_], "load": best["load"],
                     "contended": best["contended"], "runs": runs_}
            if reference is not None:
                entry["agreement_with_sceptre"] = sceptre_agreement(tables[(tool, threads)], reference, d)
            results.append(entry)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--datasets", nargs="*", default=list(BULK), choices=list(BULK))
    ap.add_argument("--tools", nargs="+", default=list(TOOLS), choices=list(TOOLS))
    ap.add_argument("--sceptre-datasets", nargs="*", default=list(SCEPTRE_DATA), choices=list(SCEPTRE_DATA))
    ap.add_argument("--sceptre-tools", nargs="+", default=list(SCEPTRE_TOOLS), choices=list(SCEPTRE_TOOLS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-threads", type=int, default=8)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8],
                    help="thread counts to run (each tool's supported subset)")
    ap.add_argument("--cpus", type=cpu_list, nargs="*", default=[], metavar="THREADS:LIST",
                    help="CPUs for the runs at a thread count, e.g. 1:16 8:16-23 (default: all allowed)")
    ap.add_argument("--idle-timeout", type=float, default=1200,
                    help="longest wait, in seconds, for other processes to leave a run's CPUs")
    ap.add_argument("--work", type=pathlib.Path, default=WORK / "runs")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.binary = args.binary.resolve()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = bulk(args)
    perturbation_results = perturbation(args)
    receipt = {
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"cores": CORES, "cpus_allowed": sorted(ALLOWED),
                 "cpus_by_threads": {str(t): sorted(c) for t, c in args.cpus}, "uname": " ".join(os.uname())},
        "binary": str(args.binary),
        "version": subprocess.run([args.binary, "--version"], capture_output=True, text=True).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "repeats": args.repeats,
        "bulk": results,
        "perturbation": perturbation_results,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    frame = pd.DataFrame(results + perturbation_results)
    if len(frame):
        print(frame[["dataset", "tool", "threads", "seconds", "contended"]].to_string(index=False))


if __name__ == "__main__":
    main()
