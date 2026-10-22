#!/usr/bin/env python3
"""Validation of the bulk DE module (rapidglm bulk-de) against DESeq2 1.48.2.

Datasets: airway (Bioconductor, 8 samples, ~ cell + dex, genes with >= 10
reads), the Kang CD14 pseudobulk panel (16 samples, donors + stimulation) and
three simulations with injected single-sample outliers (3 replicates per group
twice, 8 replicates per group once). DESeq2 runs as a black box
(scripts/reference_bulk.R).

Gates (set after the development comparisons; every value behind them is recorded):
  G1 agreement: size factors within 1e-12 (relative); on every dataset the BH
     (0.1) discovery sets have Jaccard >= 0.9 and p-values Spearman >= 0.99.
  G2 calibration: in the Kang mock-comparison null (70 four-versus-four donor
     splits, 1000 genes) bulk-de rejects at most 5.5% at p <= 0.05 and 1.2% at
     p <= 0.01; on the simulations its FDR at padj < 0.1 is within 0.02 of DESeq2's.
  G3 determinism: output identical with 1 and 8 threads.
  G4 fold-change shrinkage (lfc_shrink=normal against lfcShrink(type="normal")):
     (a) from DESeq2's own estimates, the prior-variance rule reproduces its prior
     variance for the tested column within 1e-10 (relative), and the MAP fit
     reproduces its shrunken LFC (median |difference| < 1e-6 log2) and posterior
     SD (median relative difference < 1e-6), using its original counts;
     (b) end to end, shrunken LFCs have Pearson >= 0.99 with DESeq2's over genes
     neither tool replaced.
Timings are minima of three alternating runs on a shared host (RapidGLM: the whole
CLI run at 1 and 8 threads; DESeq2: DESeq() and results() only), reported, not gated.
Shrinkage timings: RapidGLM's extra time over the plain run at one thread, and
DESeq2's lfcShrink() call.
"""

import argparse, json, os, pathlib, subprocess, time
import numpy as np
import pandas as pd
from scipy import io, sparse, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
WORK_DEFAULT = ROOT / "validation/work/bulk"


def prepare(work, bulk_library):
    work.mkdir(parents=True, exist_ok=True)
    airway = work / "airway"
    if not (airway / "counts.mtx").exists():
        airway.mkdir(exist_ok=True)
        script = f"""
suppressMessages({{library(airway); library(Matrix)}}); data(airway)
counts <- SummarizedExperiment::assay(airway, "counts"); counts <- counts[rowSums(counts) >= 10, ]
cd <- as.data.frame(SummarizedExperiment::colData(airway)); cd$dex <- relevel(cd$dex, "untrt")
X <- model.matrix(~ cell + dex, cd)
writeMM(as(Matrix(counts, sparse=TRUE), "generalMatrix"), "{airway}/counts.mtx")
write.table(X, "{airway}/design.tsv", sep="\\t", quote=FALSE, row.names=FALSE)"""
        subprocess.run(["Rscript", "-e", script], check=True, env={**ENV, "R_LIBS": bulk_library})
    kang = work / "kang"
    if not (kang / "counts.mtx").exists():
        kang.mkdir(exist_ok=True)
        panel = ROOT / "validation/work/applications/pseudobulk"
        (kang / "counts.mtx").write_bytes((panel / "responses.mtx").read_bytes())
        X = np.loadtxt(panel / "design.tsv")
        names = ["intercept"] + [f"donor{k}" for k in range(1, X.shape[1] - 1)] + ["stim"]
        np.savetxt(kang / "design.tsv", X, delimiter="\t", header="\t".join(names), comments="", fmt="%g")
    for name, reps, seed in (("sim3", 3, 3), ("sim3b", 3, 33), ("sim8", 8, 8)):
        d = work / name
        if (d / "counts.mtx").exists():
            continue
        rng = np.random.default_rng(seed)
        genes, n = 4000, 2 * reps
        g = np.repeat([0, 1], reps)
        sf = np.exp(rng.normal(0, 0.25, n))
        X = np.column_stack([np.ones(n), g])
        base = np.exp(rng.uniform(np.log(2), np.log(5000), genes))
        disp = 0.05 + 2.0 / base
        lfc = np.where(rng.random(genes) < 0.1, rng.normal(0, 1.2, genes), 0.0)
        mu = base[:, None] * np.exp(lfc[:, None] * g[None, :]) * sf[None, :]
        Y = rng.poisson(rng.gamma(1 / disp[:, None], mu * disp[:, None]))
        out = rng.random(genes) < 0.02
        for i in np.flatnonzero(out):
            j = rng.integers(n)
            Y[i, j] = int(Y[i, j] * 20 + 50)
        d.mkdir(exist_ok=True)
        io.mmwrite(str(d / "counts.mtx"), sparse.coo_matrix(Y), field="integer")
        np.savetxt(d / "design.tsv", X, delimiter="\t", header="intercept\tgroup", comments="", fmt="%g")
        np.savetxt(d / "truth.tsv", np.column_stack([lfc / np.log(2), out]), delimiter="\t",
                   header="log2fc\toutlier", comments="")
    datasets = {}
    for name in ("airway", "kang", "sim3", "sim3b", "sim8"):
        d = work / name
        header = (d / "design.tsv").read_text().splitlines()[0].split("\t")
        np.savetxt(d / "x.tsv", np.loadtxt(d / "design.tsv", skiprows=1), delimiter="\t", fmt="%.17g")
        datasets[name] = (d, header[-1], len(header) - 1)
    return datasets


def run_bulk(binary, d, coefficient, threads, prefix, **extra):
    config = dict(responses=str(d / "counts.mtx"), design=str(d / "x.tsv"), coefficient=coefficient,
                  threads=threads, output=str(prefix), **extra)
    path = pathlib.Path(str(prefix) + ".json")
    path.write_text(json.dumps(config))
    command = [str(binary), "bulk-de", str(path)]
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        from quiet_run import wait_quiet
        cpus = "16" if threads == 1 else "16-23"
        wait_quiet(cpus, label=f"bulk validation {prefix.name} threads={threads}")
        command = ["taskset", "-c", cpus, *command]
    t0 = time.perf_counter()
    subprocess.run(command, check=True, env=ENV, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t0


def compare(d, prefix):
    ref = json.loads((d / "deseq2/reference.json").read_text())
    run = json.loads(pathlib.Path(str(prefix) + ".run.json").read_text())
    a = pd.read_csv(d / "deseq2/genes.tsv", sep="\t")
    b = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")
    rel = lambda x, y: float(np.nanmedian(np.abs(y - x) / np.maximum(np.abs(x), 1e-12)))
    pa, pb = a.pvalue.to_numpy(float), b.pvalue.to_numpy(float)
    ok = np.isfinite(pa) & np.isfinite(pb)
    da = set(np.flatnonzero(np.nan_to_num(a.padj.to_numpy(float), nan=1) < 0.1))
    db = set(np.flatnonzero(np.nan_to_num(b.padj.to_numpy(float), nan=1) < 0.1))
    row = dict(
        genes=len(a),
        size_factor_max_rel_diff=float(np.max(np.abs(np.array(run["size_factors"]) / np.array(ref["size_factors"]) - 1))),
        trend=[run["asymptotic_dispersion"], run["extra_poisson"]],
        trend_deseq2=[ref["trend_coefficients"].get("asymptDisp"), ref["trend_coefficients"].get("extraPois")],
        prior_variance=[run["prior_variance"], ref["prior_variance"]],
        median_rel_diff=dict(dispGeneEst=rel(a.dispGeneEst, b.dispGeneEst), dispersion=rel(a.dispersion, b.dispersion),
                             log2FoldChange=rel(a.log2FoldChange, b.log2FoldChange), pvalue=rel(a.pvalue, b.pvalue)),
        pvalue_spearman=float(stats.spearmanr(pa[ok], pb[ok]).statistic),
        discoveries=[len(db), len(da)],
        jaccard=len(da & db) / max(1, len(da | db)),
        filter_threshold=[run["filter_threshold"], ref["filter_threshold"]],
        statuses=b.status.value_counts().to_dict(),
    )
    truth = d / "truth.tsv"
    if truth.exists():
        t = np.loadtxt(truth, skiprows=1)
        de, outlier = t[:, 0] != 0, t[:, 1] > 0
        for name, tab in (("rapidglm", b), ("deseq2", a)):
            call = np.nan_to_num(tab.padj.to_numpy(float), nan=1) < 0.1
            row[f"fdr_{name}"] = float((call & ~de).sum() / max(1, call.sum()))
            row[f"power_{name}"] = float((call & de).sum() / de.sum())
        flags_b = b.status == "cooks_outlier"
        flags_a = a.pvalue.isna() & (a.baseMean > 0)
        row["cooks_flags"] = dict(rapidglm=int(flags_b.sum()), deseq2=int(flags_a.sum()), both=int((flags_a & flags_b).sum()),
                                  injected_outliers=int(outlier.sum()), caught_rapidglm=int((flags_b & outlier).sum()),
                                  caught_deseq2=int((flags_a & outlier).sum()), replaced_rapidglm=int(b.replaced.sum()),
                                  replaced_deseq2=int(a["replace"].astype(str).str.upper().eq("TRUE").sum()) if "replace" in a else 0)
    return row


def weighted_quantile(x, w, p):
    """The weighted quantile rule of include/rapidglm/bulk.hpp (weights scaled to n)."""
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order] * len(w) / w.sum()
    cumulative = np.cumsum(w)
    position = 1 + (len(x) - 1) * p
    low = np.floor(position)
    at = lambda t: x[min(np.searchsorted(cumulative, t, side="left"), len(x) - 1)]
    return at(low) if position == low else (1 - (position - low)) * at(low) + (position - low) * at(low + 1)


def shrinkage_formula(d, genes_limit=2000):
    """Reproduce DESeq2's normal-prior shrinkage from its own estimates (black-box
    outputs only): prior variances from its MLE coefficients, then the MAP fit and
    sandwich posterior SD at its final dispersions, size factors and original counts."""
    ref = json.loads((d / "deseq2/reference.json").read_text())
    genes = pd.read_csv(d / "deseq2/genes.tsv", sep="\t", na_values="NA")
    mle = pd.read_csv(d / "deseq2/mle.tsv", sep="\t", na_values="NA")
    X = np.loadtxt(d / "x.tsv")
    Y = io.mmread(str(d / "counts.mtx")).tocsr()
    sf = np.array(ref["size_factors"])
    target = np.array(ref["lfc_prior_variance"])
    w = (1 / (1 / genes.baseMean + genes.dispFit)).to_numpy()
    k = X.shape[1] - 1
    b = mle.iloc[:, k].to_numpy(float)
    ok = np.isfinite(b) & np.isfinite(w)
    q = weighted_quantile(np.abs(b[ok]), w[ok], 0.95)
    prior = (q / stats.norm.ppf(0.975)) ** 2
    lam = 1 / (target * np.log(2) ** 2)
    rng = np.random.default_rng(1)
    usable = np.flatnonzero(np.isfinite(genes.dispersion.to_numpy()) & (genes.baseMean.to_numpy() > 0))
    chosen = usable if len(usable) <= genes_limit else np.sort(rng.choice(usable, genes_limit, replace=False))
    lfc_err, se_err = [], []
    for g in chosen:
        y, a = Y[g].toarray().ravel().astype(float), genes.dispersion[g]
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(np.mean(y / sf), 1e-3))
        for _ in range(500):
            eta = X @ beta
            mu = sf * np.exp(eta)
            wt = mu / (1 + a * mu)
            new = np.linalg.solve(X.T @ (X * wt[:, None]) + np.diag(lam), X.T @ (wt * (eta + (y - mu) / mu)))
            done = np.max(np.abs(new - beta)) < 1e-12
            beta = new
            if done:
                break
        m = np.maximum(sf * np.exp(X @ beta), 0.5)
        wt = m / (1 + a * m)
        H = X.T @ (X * wt[:, None])
        Ai = np.linalg.inv(H + np.diag(lam))
        lfc_err.append(abs(beta[k] / np.log(2) - genes.shrunkLFC[g]))
        se_err.append(abs(np.sqrt((Ai @ H @ Ai)[k, k]) / np.log(2) / genes.shrunkSE[g] - 1))
    lfc_err, se_err = np.array(lfc_err), np.array(se_err)
    return dict(prior_variance_rel_diff=float(abs(prior / target[k] - 1)), genes=int(len(chosen)),
                lfc_abs_diff=dict(median=float(np.median(lfc_err)), p99=float(np.quantile(lfc_err, 0.99)),
                                  max=float(lfc_err.max())),
                se_rel_diff=dict(median=float(np.median(se_err)), p99=float(np.quantile(se_err, 0.99)),
                                 max=float(se_err.max())))


def compare_shrinkage(d, prefix):
    ref = json.loads((d / "deseq2/reference.json").read_text())
    run = json.loads(pathlib.Path(str(prefix) + ".run.json").read_text())
    a = pd.read_csv(d / "deseq2/genes.tsv", sep="\t", na_values="NA")
    b = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")
    replaced_a = a["replace"].astype(str).str.upper().eq("TRUE") if "replace" in a else pd.Series(False, index=a.index)
    replaced = replaced_a.to_numpy() | b.replaced.astype(bool).to_numpy()
    ok = np.isfinite(a.shrunkLFC.to_numpy(float)) & np.isfinite(b.log2FoldChangeShrunk.to_numpy(float))
    diff = np.abs(a.shrunkLFC - b.log2FoldChangeShrunk).to_numpy()
    kept = ok & ~replaced
    row = dict(prior_variance=[run["lfc_prior_variance"], ref["lfc_prior_variance"]],
               genes_compared=int(kept.sum()), genes_replaced=int(replaced.sum()),
               missing=[int(b.log2FoldChangeShrunk.isna().sum()), int(a.shrunkLFC.isna().sum())],
               lfc_pearson=float(np.corrcoef(a.shrunkLFC[kept], b.log2FoldChangeShrunk[kept])[0, 1]),
               lfc_abs_diff=dict(median=float(np.median(diff[kept])), p99=float(np.quantile(diff[kept], 0.99))),
               se_rel_diff_median=float(np.median(np.abs(b.lfcSEShrunk / a.shrunkSE - 1)[kept])))
    if (ok & replaced).any():
        row["replaced_lfc_abs_diff_median"] = float(np.median(diff[ok & replaced]))
    return row


def mock_null(binary, work, library):
    W = ROOT / "validation/work/ql-test/null"
    splits = json.loads((W / "splits.json").read_text())
    out = work / "mocknull"
    out.mkdir(exist_ok=True)
    rows = []
    for s in splits:
        label, idx = s["condition"], s["split"]
        prefix = out / f"{label}.{idx}"
        config = dict(responses=str(W / f"{label}.mtx"), design=str(W / f"{label}.{idx}.design.tsv"),
                      coefficient=1, threads=4, output=str(prefix))
        pathlib.Path(str(prefix) + ".json").write_text(json.dumps(config))
        subprocess.run([str(binary), "bulk-de", str(prefix) + ".json"], check=True, env=ENV, stdout=subprocess.DEVNULL)
        t = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")
        rows.append(t.assign(condition=label, split=idx))
    # DESeq2 with its default (median-of-ratios) size factors on the same splits.
    script = r"""
args <- commandArgs(TRUE); W <- args[1]; .libPaths(c(args[2], .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(jsonlite)})
splits <- fromJSON(file.path(W, "splits.json"), simplifyVector = FALSE); out <- list()
for (s in splits) {
  counts <- as.matrix(readMM(file.path(W, paste0(s$condition, ".mtx")))); storage.mode(counts) <- "integer"
  X <- as.matrix(read.table(file.path(W, paste0(s$condition, ".", s$split, ".design.tsv"))))
  colnames(X) <- c("intercept", "group"); colnames(counts) <- paste0("s", seq_len(ncol(counts)))
  dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
  dds <- DESeq(dds, full = X, quiet = TRUE); res <- results(dds, name = "group")
  out[[length(out) + 1]] <- data.frame(condition = s$condition, split = s$split, p = res$pvalue, padj = res$padj)
}
write.table(do.call(rbind, out), args[3], sep = "\t", row.names = FALSE, quote = FALSE)
"""
    path = out / "deseq2_null.R"
    path.write_text(script)
    subprocess.run(["Rscript", str(path), str(W), library, str(out / "deseq2.tsv")], check=True, env=ENV,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = {}
    for name, data, pcol, qcol in (("rapidglm", pd.concat(rows), "pvalue", "padj"),
                                   ("deseq2", pd.read_csv(out / "deseq2.tsv", sep="\t", na_values="NA"), "p", "padj")):
        p = data[pcol].to_numpy(float)
        v = np.isfinite(p)
        bh = [int((np.nan_to_num(g[qcol].to_numpy(float), nan=1) <= 0.05).sum()) for _, g in data.groupby(["condition", "split"])]
        summary[name] = dict(tests=int(len(p)), valid=int(v.sum()), rate_0_05=float(np.mean(p[v] <= 0.05)),
                             rate_0_01=float(np.mean(p[v] <= 0.01)), rate_0_001=float(np.mean(p[v] <= 0.001)),
                             bh_0_05_per_split=float(np.mean(bh)))
    return summary


def main(args):
    work = args.work.resolve()
    datasets = prepare(work, args.bulk_library)
    results = {}
    for name, (d, coef, index) in datasets.items():
        reference = d / "deseq2/reference.json"
        if not reference.exists() or "lfc_prior_variance" not in json.loads(reference.read_text()):
            subprocess.run(["Rscript", str(ROOT / "scripts/reference_bulk.R"), str(d), coef, str(d / "deseq2"),
                            args.r_library], check=True, env=ENV, stdout=subprocess.DEVNULL)
        # Alternate RapidGLM and DESeq2 timing runs on the shared host; keep minima.
        seconds, deseq2_seconds, other = {1: [], 8: []}, [], {1: [], 8: []}
        shrink_seconds, deseq2_shrink_seconds = [], []
        for _ in range(3):
            for t in (1, 8):
                seconds[t].append(run_bulk(args.binary, d, index, t, d / f"rapid.t{t}"))
                if args.compare_binary:
                    other[t].append(run_bulk(args.compare_binary, d, index, t, d / f"other.t{t}"))
            shrink_seconds.append(run_bulk(args.binary, d, index, 1, d / "rapid.shrink", lfc_shrink="normal"))
            command = ["Rscript", str(ROOT / "scripts/reference_bulk.R"), str(d), coef,
                       str(d / "deseq2_timing"), args.r_library]
            if os.environ.get("RAPIDGLM_QUIET_LOAD"):
                from quiet_run import wait_quiet
                wait_quiet("16", label=f"bulk validation {name} DESeq2")
                command = ["taskset", "-c", "16", *command]
            subprocess.run(command, check=True, env=ENV, stdout=subprocess.DEVNULL)
            timing = json.loads((d / "deseq2_timing/reference.json").read_text())
            deseq2_seconds.append(timing["seconds"])
            deseq2_shrink_seconds.append(timing["shrink_seconds"])
        row = compare(d, d / "rapid.t1")
        row["rapidglm_seconds"] = {t: min(v) for t, v in seconds.items()}
        row["deseq2_seconds"] = min(deseq2_seconds)
        row["shrinkage"] = compare_shrinkage(d, d / "rapid.shrink")
        row["shrinkage"]["formula_from_deseq2_estimates"] = shrinkage_formula(d)
        row["shrinkage"]["seconds"] = dict(rapidglm_extra=max(0.0, min(shrink_seconds) - row["rapidglm_seconds"][1]),
                                           deseq2_lfcshrink=min(deseq2_shrink_seconds))
        if args.compare_binary:
            mine = pd.read_csv(d / "rapid.t1.tsv", sep="\t", na_values="NA")
            theirs = pd.read_csv(d / "other.t1.tsv", sep="\t", na_values="NA")
            calls = lambda t: set(np.flatnonzero(np.nan_to_num(t.padj.to_numpy(float), nan=1) < 0.1))
            row["compare_binary"] = dict(seconds={t: min(v) for t, v in other.items()},
                                         identical_table=(d / "rapid.t1.tsv").read_bytes() == (d / "other.t1.tsv").read_bytes(),
                                         identical_discoveries=calls(mine) == calls(theirs),
                                         statuses_equal=bool((mine.status == theirs.status).all()))
        row["identical_1_and_8_threads"] = (d / "rapid.t1.tsv").read_bytes() == (d / "rapid.t8.tsv").read_bytes()
        results[name] = row
        print(name, {k: row[k] for k in ("jaccard", "pvalue_spearman", "discoveries", "rapidglm_seconds", "deseq2_seconds")}, flush=True)
    null = mock_null(args.binary, work, args.r_library)
    print("mock null", null, flush=True)
    g1 = all(r["size_factor_max_rel_diff"] < 1e-12 and r["jaccard"] >= 0.9 and r["pvalue_spearman"] >= 0.99
             for r in results.values())
    g2 = (null["rapidglm"]["rate_0_05"] <= 0.055 and null["rapidglm"]["rate_0_01"] <= 0.012 and
          all(abs(r["fdr_rapidglm"] - r["fdr_deseq2"]) <= 0.02 for r in results.values() if "fdr_rapidglm" in r))
    g3 = all(r["identical_1_and_8_threads"] for r in results.values())
    g4 = all(r["shrinkage"]["formula_from_deseq2_estimates"]["prior_variance_rel_diff"] < 1e-10 and
             r["shrinkage"]["formula_from_deseq2_estimates"]["lfc_abs_diff"]["median"] < 1e-6 and
             r["shrinkage"]["formula_from_deseq2_estimates"]["se_rel_diff"]["median"] < 1e-6 and
             r["shrinkage"]["lfc_pearson"] >= 0.99 for r in results.values())
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        worktree_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()),
        command="python3 scripts/validate_bulk.py --work DIR --output validation/BULK_DE.json"
                + (" --compare-binary " + args.compare_binary_note if args.compare_binary else ""),
        datasets=results,
        mock_null=null,
        gates=dict(G1_agreement=g1, G2_calibration=g2, G3_threads=g3, G4_shrinkage=g4),
        passed=g1 and g2 and g3 and g4,
        load_average=os.getloadavg(),
    )
    args.output.write_text(json.dumps(receipt, indent=1, default=float) + "\n")
    print("gates", receipt["gates"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=pathlib.Path, default=WORK_DEFAULT)
    ap.add_argument("--compare-binary", type=pathlib.Path,
                    help="another rapidglm build to time alternately (e.g. a baseline)")
    ap.add_argument("--compare-binary-note", default="BASELINE",
                    help="how the receipt describes --compare-binary")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--r-library", default=str(ROOT / "validation/local-data/R_library"))
    ap.add_argument("--bulk-library", default=f"{ROOT / 'validation/local-data/R_bulkdata'}:{ROOT / 'validation/local-data/R_library'}")
    main(ap.parse_args())
