#!/usr/bin/env python3
"""The K562 essential low-MOI pilot on the panel of scripts/prepare_replogle.py.

  sc         `rapidglm sceptre` at low MOI (non-targeting controls, permutations)
             on the single-cell panel: every listed discovery, positive and
             negative pair; --fit-check ks accepts the skew-normal fit as SCEPTRE
             does, without RapidGLM's confirmation of small p-values.
  sceptre    SCEPTRE 0.99.0 at low MOI on the same panel, as a black box
             (scripts/reference_sceptre_low_moi.R).
  pseudobulk for each drawn target, the pseudobulk samples of its cells and of
             the non-targeting cells in each lane (design ~ lane + perturbed),
             tested by `rapidglm bulk-de` and DESeq2 (scripts/reference_replogle_bulk.R),
             and by `rapidglm edger-ql` and edgeR's QL workflow
             (scripts/reference_replogle_edger.R).
  compare    agreement of the two single-cell analyses and of the two
             pseudobulk analyses, and the timings, into --summary.

The whole screen (every perturbation's pseudobulk dataset, built and tested as
in the pseudobulk stage), in a --pilot folder whose pseudobulk/ holds counts.npy
and samples.tsv:
  full-prepare  build the datasets, untimed (--jobs processes on the caller's
                CPUs); --targets FILE limits them to the listed perturbations.
                Perturbations whose dataset cannot be built are skipped and
                recorded with the reason in pseudobulk/full_targets.json.
  full          one tool (--tool: rapidglm, deseq2, rapidglm_edger_ql, edger)
                on every prepared dataset with the pilot's command, over one
                single-thread worker per CPU of --cpus: worker k is pinned to the
                k-th CPU and takes the next perturbation from a shared queue.
                Per perturbation: wall, CPU and peak memory, the tool-internal
                seconds, and the other processes' share of its CPU; for the
                screen: wall time and the other processes' share of --cpus
                (pseudobulk/full_TOOL.json). A failing perturbation is recorded
                and the screen continues.
  full-compare  the compare stage's pseudobulk rows for every perturbation,
                with medians and worst cases, into --summary; --identical-to
                PILOT checks the outputs against an earlier pilot folder.
Take the timing lock around `full` and start it on a quiet host.

Every stage pins its processes to --cpus; take the timing lock around timed
stages. Both single-cell analyses use the formula --formula over covariates.tsv
(the CLI's factor() names a categorical column; SCEPTRE gets batch as a factor).
"""

import argparse
import datetime
import json
import os
import pathlib
import queue
import re
import subprocess
import threading
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The R library holding SCEPTRE 0.99.0 (set SCEPTRE_LIBRARY when it is not the default one).
SCEPTRE_LIBRARY = os.environ.get("SCEPTRE_LIBRARY", str(ROOT / "validation/local-data/R_library"))
R_LIBRARY = os.environ.get("DESEQ2_LIBRARY", str(ROOT / "validation/local-data/R_library"))
EDGER_LIBRARY = f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


class ToolFailed(SystemExit):
    """A measured command exited non-zero."""

    def __init__(self, command, result):
        super().__init__(f"{command[:2]} failed:\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}")
        self.returncode = result.returncode
        self.stderr_tail = result.stderr[-3000:]


def measure(command, cpus):
    """Runs `command` pinned to `cpus` under /usr/bin/time; wall seconds, CPU
    seconds and peak RSS (GB)."""
    load_before = os.getloadavg()[0]
    start = time.time()
    r = subprocess.run(["/usr/bin/time", "-v", "taskset", "-c", cpus, *command], capture_output=True,
                       text=True, env=ENV)
    wall = time.time() - start
    if r.returncode:
        raise ToolFailed(command, r)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr).group(1))
    cpu = sum(float(re.search(rf"{k} time \(seconds\): (\S+)", r.stderr).group(1)) for k in ("User", "System"))
    return {"load_before": round(load_before, 2), "wall_seconds": round(wall, 2), "cpu_seconds": round(cpu, 1), "peak_rss_gb": round(rss / 1e6, 2),
            "load_average": round(os.getloadavg()[0], 2)}


def timed(command, cpus):
    """measure() after a quiet host and quiet `cpus` (when RAPIDGLM_QUIET_LOAD is set)."""
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        from quiet_run import wait_quiet
        wait_quiet(cpus)
    result = measure(command, cpus)
    if os.environ.get("RAPIDGLM_QUIET_LOAD"):
        from quiet_run import finish_measurement
        finish_measurement()
    return result


def cli_formula(formula):
    return re.sub(r"\bbatch\b", "factor(batch)", formula)


def sc_name(args):
    return f"rapidglm_t{args.threads}" + ("" if args.fit_check == "strict" else f"_{args.fit_check}")


def run_sc(args):
    panel = args.pilot / "sc"
    out = args.pilot / "runs" / sc_name(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "responses": str(panel / "responses.mtx"), "assignments": str(panel / "assignments.mtx"),
        "guides": str(panel / "guides.tsv"), "pairs": str(panel / "pairs.tsv"),
        "covariates": str(panel / "covariates.tsv"), "formula": cli_formula(args.formula),
        "response_ids": str(panel / "responses.txt"), "moi": "low", "cell_qc": False,
        "seed": args.seed, "threads": args.threads, "fit_check": args.fit_check, "output": str(out),
    }
    path = out.parent / f"{sc_name(args)}.json"
    path.write_text(json.dumps(config, indent=1) + "\n")
    timing = timed([str(args.binary), "sceptre", str(path)], args.cpus)
    record = json.loads(pathlib.Path(str(out) + ".run.json").read_text())
    timing["workflow_wall_seconds"] = record.get("wall_seconds")
    timing["null_fits"] = record.get("null_fits")
    (out.parent / f"{sc_name(args)}.timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))


def run_sceptre(args):
    out = args.pilot / "runs" / f"sceptre_w{args.threads}"
    timing = timed(["Rscript", str(ROOT / "scripts/reference_sceptre_low_moi.R"), str(args.pilot / "sc"),
                    SCEPTRE_LIBRARY, str(out), str(args.threads), args.formula], args.cpus)
    timing.update(json.loads((out / "timing.json").read_text()))
    (out / "process.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))


def write_market(path, counts):
    """counts: genes x samples."""
    rows, cols = np.nonzero(counts)
    with open(path, "w") as out:
        out.write("%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{counts.shape[0]} {counts.shape[1]} {rows.size}\n")
        np.savetxt(out, np.column_stack([rows + 1, cols + 1, counts[rows, cols]]), fmt="%d")


def pseudobulk_dataset(pilot, target, counts, samples):
    """The samples of `target` and of the non-targeting cells, ~ lane + perturbed."""
    d = pilot / "pseudobulk" / "datasets" / target
    if (d / "counts.mtx").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    keep = samples.index[samples.group.isin([target, "non-targeting"])]
    s = samples.loc[keep]
    lanes = sorted(s.gem_group.unique())
    x = pd.DataFrame({"intercept": 1.0}, index=keep)
    for lane in lanes[1:]:
        x[f"lane{lane}"] = (s.gem_group == lane).astype(float)
    x["perturbed"] = (s.group == target).astype(float)
    write_market(d / "counts.mtx", counts[keep].T)
    x.to_csv(d / "design.tsv", sep="\t", index=False)
    x.to_csv(d / "x.tsv", sep="\t", index=False, header=False)
    s.to_csv(d / "samples.tsv", sep="\t", index=False)
    return d


# The pilot's four pseudobulk tools, in its order.
TOOLS = ("rapidglm", "deseq2", "rapidglm_edger_ql", "edger")


def pseudobulk_tool(tool, d, x, args):
    """The pilot's command for `tool` on dataset `d` (design `x`), and a function
    returning the seconds the tool records itself ({} for RapidGLM, whose run
    record the pilot does not time)."""
    if tool in ("rapidglm", "rapidglm_edger_ql"):
        stem, command = ("rapidglm", "bulk-de") if tool == "rapidglm" else ("edger_ql", "edger-ql")
        config = d / f"{stem}.json"
        config.write_text(json.dumps({"responses": str(d / "counts.mtx"), "design": str(d / "x.tsv"),
                                      "coefficient": x.shape[1] - 1, "threads": args.threads,
                                      "output": str(d / stem)}))
        return [str(args.binary), command, str(config)], dict
    if tool == "deseq2":
        return (["Rscript", str(ROOT / "scripts/reference_replogle_bulk.R"), str(d), "perturbed",
                 str(d / "deseq2"), R_LIBRARY],
                lambda: {"deseq_seconds": json.loads((d / "deseq2/reference.json").read_text())["seconds"]})
    if tool == "edger":
        return (["Rscript", str(ROOT / "scripts/reference_replogle_edger.R"), str(d), "perturbed",
                 str(d / "edger"), EDGER_LIBRARY],
                lambda: {"ql_seconds": json.loads((d / "edger/timing.json").read_text())["ql_seconds"]})
    raise ValueError(tool)


def run_pseudobulk(args):
    pb = args.pilot / "pseudobulk"
    samples = pd.read_csv(pb / "samples.tsv", sep="\t")
    counts = np.load(pb / "counts.npy", mmap_mode="r")
    targets = json.loads((args.pilot / "provenance.json").read_text())["targets"]
    timings = {}
    for target in targets:
        d = pseudobulk_dataset(args.pilot, target, counts, samples)
        x = pd.read_csv(d / "design.tsv", sep="\t")
        entry = {"samples": int(x.shape[0]), "columns": int(x.shape[1]),
                 "perturbed_samples": int(x.perturbed.sum())}
        for tool in TOOLS:
            command, internal = pseudobulk_tool(tool, d, x, args)
            entry[tool] = timed(command, args.cpus)
            entry[tool].update(internal())
        timings[target] = entry
        print(target, json.dumps(entry), flush=True)
    (pb / f"timing_t{args.threads}.json").write_text(json.dumps(timings, indent=1) + "\n")


# The whole screen.


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpu_list(text):
    cpus = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def busy_jiffies(cpus):
    """Jiffies each CPU has spent busy (all but idle and I/O wait), from /proc/stat,
    as in benchmark_applications.py and the release guards."""
    result = {}
    with open("/proc/stat") as f:
        for line in f:
            name, *values = line.split()
            if name[:3] == "cpu" and name[3:].isdigit() and int(name[3:]) in cpus:
                values = [int(v) for v in values]
                result[int(name[3:])] = sum(values[:3]) + sum(values[5:8])
    return result


TICKS = os.sysconf("SC_CLK_TCK")


def full_targets(args):
    """The screen's perturbations (every group but non-targeting), or those in --targets."""
    samples = pd.read_csv(args.pilot / "pseudobulk/samples.tsv", sep="\t")
    groups = sorted(set(samples.group) - {"non-targeting"})
    if args.targets:
        listed = args.targets.read_text().split()
        missing = sorted(set(listed) - set(groups))
        if missing:
            raise SystemExit(f"--targets lists groups absent from samples.tsv: {missing[:10]}")
        groups = sorted(set(listed))
    return groups


def _prepare_one(target):
    try:
        d = pseudobulk_dataset(_PREPARE["pilot"], target, _PREPARE["counts"], _PREPARE["samples"])
        x = pd.read_csv(d / "design.tsv", sep="\t")
        return target, {"samples": int(x.shape[0]), "columns": int(x.shape[1]),
                        "perturbed_samples": int(x.perturbed.sum())}, None
    except Exception as error:  # recorded as a skip with its reason
        return target, None, f"{type(error).__name__}: {error}"


_PREPARE = {}


def full_prepare(args):
    import multiprocessing
    pb = args.pilot / "pseudobulk"
    targets = full_targets(args)
    _PREPARE.update(pilot=args.pilot, samples=pd.read_csv(pb / "samples.tsv", sep="\t"),
                    counts=np.load(pb / "counts.npy", mmap_mode="r"))
    start = time.time()
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = pool.map(_prepare_one, targets, chunksize=4)
    record = {"created_utc": utc_now(), "targets": {t: e for t, e, _ in results if e is not None},
              "skipped": {t: reason for t, _, reason in results if reason is not None},
              "untimed_seconds": round(time.time() - start, 1)}
    (pb / "full_targets.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"targets": len(record["targets"]), "skipped": record["skipped"],
                      "seconds": record["untimed_seconds"]}))


def run_full(args):
    pb = args.pilot / "pseudobulk"
    if (pb / f"full_{args.tool}.json").exists():
        raise SystemExit(f"archive the earlier measurement before a rerun: {pb / f'full_{args.tool}.json'}")
    prepared = json.loads((pb / "full_targets.json").read_text())
    targets = sorted(prepared["targets"])
    cpus = cpu_list(args.cpus)
    jobs = queue.Queue()
    for target in targets:
        jobs.put(target)
    results, failures = {}, {}
    lock = threading.Lock()
    progress = pb / f"full_{args.tool}.progress.jsonl"
    progress.unlink(missing_ok=True)
    try:  # an audit record when the release timing audit is configured
        from quiet_run import audit
    except ImportError:
        audit = None
    begin = time.monotonic()

    def worker(k):
        cpu = cpus[k]
        while True:
            try:
                target = jobs.get_nowait()
            except queue.Empty:
                return
            d = pb / "datasets" / target
            x = pd.read_csv(d / "design.tsv", sep="\t")
            command, internal = pseudobulk_tool(args.tool, d, x, args)
            offset = time.monotonic() - begin
            before = busy_jiffies({cpu})[cpu]
            try:
                entry = measure(command, str(cpu))
                entry.update(internal())
            except ToolFailed as failure:
                entry = {"exit": failure.returncode, "stderr_tail": failure.stderr_tail}
            busy = (busy_jiffies({cpu})[cpu] - before) / TICKS
            entry.update(worker=k, cpu=cpu, start_offset_seconds=round(offset, 2))
            if "wall_seconds" in entry:
                entry["others_cpu"] = round(max(0.0, busy - entry["cpu_seconds"]) / entry["wall_seconds"], 4)
            with lock:
                (results if "exit" not in entry else failures)[target] = entry
                with progress.open("a") as out:
                    out.write(json.dumps({"target": target, **entry}) + "\n")

    if audit:
        audit({"event": "full_screen_started", "utc": utc_now(), "tool": args.tool, "cpus": cpus,
               "targets": len(targets)})
    start_utc, load_before = utc_now(), os.getloadavg()
    busy_before, own_before = busy_jiffies(set(cpus)), os.times()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(cpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - begin
    own_after, busy_after = os.times(), busy_jiffies(set(cpus))
    # CPU of the measured commands (children this process has waited for) against the CPUs' busy time.
    ours = (own_after.children_user + own_after.children_system
            - own_before.children_user - own_before.children_system)
    busy = sum(busy_after[c] - busy_before[c] for c in cpus) / TICKS
    others = max(0.0, busy - ours) / (wall * len(cpus))
    record = {
        "tool": args.tool, "binary": str(args.binary),
        "version": subprocess.check_output([str(args.binary), "--version"], text=True).strip(),
        "threads_per_job": args.threads, "workers": len(cpus), "cpus": cpus,
        "started_utc": start_utc, "finished_utc": utc_now(), "load_before": load_before,
        "load_after": os.getloadavg(), "wall_seconds": round(wall, 2),
        "command_cpu_seconds": round(ours, 1), "cpus_busy_seconds": round(busy, 1),
        "others_cpu": round(others, 4), "contended": others > 0.05,
        "targets": len(targets), "succeeded": len(results), "failed": failures,
        "skipped_at_prepare": prepared["skipped"],
        "per_target": {t: results[t] for t in sorted(results)},
    }
    (pb / f"full_{args.tool}.json").write_text(json.dumps(record, indent=1) + "\n")
    if audit:
        audit({"event": "full_screen_finished", "utc": utc_now(), "tool": args.tool, "cpus": cpus,
               "wall_seconds": record["wall_seconds"], "others_cpu": record["others_cpu"],
               "contended": record["contended"], "failed": len(failures)})
    print(json.dumps({k: v for k, v in record.items() if k != "per_target"}, indent=1))
    if failures:
        raise SystemExit(f"{len(failures)} perturbations failed with {args.tool}")


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def common_negative_pairs(rapid, reference):
    if "pass_qc" in reference:
        reference = reference[reference.pass_qc]
    paired = rapid.merge(reference, left_on=["response_id", "target"],
                         right_on=["response_id", "grna_target"], suffixes=("", "_sceptre"))
    return paired[np.isfinite(paired.p_value) & np.isfinite(paired.p_value_sceptre)]


def pseudobulk_row(d, t):
    """The compare stage's agreement and timing row for one dataset `d` and its timings `t`
    (the pilot's per-target timing entry)."""
    a = pd.read_csv(d / "rapidglm.tsv", sep="\t", na_values="NA", keep_default_na=False)
    b = pd.read_csv(d / "deseq2/genes.tsv", sep="\t")
    ok = np.isfinite(a.pvalue) & np.isfinite(b.pvalue)
    ra, rb = set(np.flatnonzero(a.padj <= 0.1)), set(np.flatnonzero(b.padj <= 0.1))
    ql_a = pd.read_csv(d / "edger_ql.tsv", sep="\t", na_values="NA", keep_default_na=False)
    ql_b = pd.read_csv(d / "edger/genes_ql.tsv", sep="\t")
    qa = set(ql_a.gene[ql_a.FDR <= 0.1])
    qb = set(ql_b.gene[ql_b.FDR <= 0.1])
    merged = ql_a.merge(ql_b, on="gene", suffixes=("", "_edger"))
    return {
        "edger_ql_hits": len(qa), "edger_hits": len(qb), "edger_jaccard": round(jaccard(qa, qb), 4),
        "edger_max_abs_log10_p_difference": float(np.nanmax(np.abs(
            np.log10(merged.PValue) - np.log10(merged.PValue_edger)))) if len(merged) else None,
        "rapidglm_edger_ql_seconds": t["rapidglm_edger_ql"]["wall_seconds"],
        "edger_seconds": t["edger"]["wall_seconds"], "edger_ql_seconds": t["edger"]["ql_seconds"],
        "samples": t["samples"], "perturbed_samples": t["perturbed_samples"],
        "rapidglm_seconds": t["rapidglm"]["wall_seconds"], "deseq2_seconds": t["deseq2"]["wall_seconds"],
        "deseq2_fit_seconds": t["deseq2"]["deseq_seconds"],
        "rapidglm_hits": len(ra), "deseq2_hits": len(rb), "jaccard": round(jaccard(ra, rb), 4),
        "max_abs_log10_p_difference": float(np.max(np.abs(np.log10(a.pvalue[ok]) - np.log10(b.pvalue[ok]))))
        if ok.any() else None,
        "max_abs_lfc_difference": float(np.nanmax(np.abs(a.log2FoldChange[ok] - b.log2FoldChange[ok])))
        if ok.any() else None,
    }


def compare(args):
    pilot = args.pilot
    summary = {"provenance": json.loads((pilot / "provenance.json").read_text()),
               "version": subprocess.check_output([str(args.binary), "--version"],text=True).strip(),
               "commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "single_cell_fit_check": args.fit_check}
    responses = (pilot / "sc/responses.txt").read_text().split()
    # Single cell.
    rapid_path = pilot / "runs" / f"{sc_name(args)}.tsv"
    sceptre_dir = pilot / "runs" / f"sceptre_w{args.threads}"
    if rapid_path.exists() and (sceptre_dir / "discovery.tsv").exists():
        r = pd.read_csv(rapid_path, sep="\t", na_values="NA", keep_default_na=False)
        r["response_id"] = [responses[i] for i in r.response]
        r["kind"] = r.kind.map({0: "discovery", 1: "negative", 2: "positive"})
        s = pd.read_csv(sceptre_dir / "discovery.tsv", sep="\t")
        sc = {}
        for kind in ("discovery", "positive"):
            rk = r[(r.kind == kind) & r.status.str.startswith("ok")]
            sk = (s if kind == "discovery" else pd.read_csv(sceptre_dir / "power.tsv", sep="\t"))
            sk = sk[sk.pass_qc] if "pass_qc" in sk else sk
            m = rk.merge(sk, left_on=["response_id", "target"], right_on=["response_id", "grna_target"],
                         suffixes=("", "_sceptre"))
            m = m[np.isfinite(m.p_value_sceptre)] if "p_value_sceptre" in m else m
            entry = {"rapidglm_tested": int(len(rk)), "sceptre_tested": int(np.isfinite(sk.p_value).sum()),
                     "both": int(len(m))}
            if len(m) and kind == "discovery":
                lr, ls = -np.log10(m.p_value), -np.log10(m.p_value_sceptre)
                entry["spearman_p"] = float(pd.Series(lr).corr(pd.Series(ls), method="spearman"))
                entry["pearson_log10_p"] = float(np.corrcoef(lr, ls)[0, 1])
                hits_r = set(rk.loc[rk.adjusted_p <= 0.1, ["response_id", "target"]].itertuples(index=False))
                hits_s = set(sk.loc[sk.significant, ["response_id", "grna_target"]].itertuples(index=False)) \
                    if "significant" in sk else set()
                entry.update({"rapidglm_discoveries": len(hits_r), "sceptre_discoveries": len(hits_s),
                              "jaccard": round(jaccard(hits_r, hits_s), 4)})
                by_target = {}
                for t, g in m.groupby("target"):
                    by_target[t] = {"pairs": int(len(g)),
                                    "spearman": float(g.p_value.corr(g.p_value_sceptre, method="spearman"))}
                entry["by_target"] = by_target
            if len(m) and kind == "positive":
                entry["pairs"] = m[["target", "p_value", "p_value_sceptre", "log2fc"]].to_dict("records")
            sc[kind] = entry
        cal = sceptre_dir / "calibration.tsv"
        if cal.exists():
            c = pd.read_csv(cal, sep="\t")
            rn = r[(r.kind == "negative") & r.status.str.startswith("ok")]
            paired = common_negative_pairs(rn, c)
            neg, cp = paired.p_value.to_numpy(), paired.p_value_sceptre.to_numpy()
            sc["negative"] = {"common_tested": int(len(paired)),
                "rapidglm_fraction_below_0.05": float((neg <= .05).mean()) if len(neg) else None,
                "rapidglm_fraction_below_0.01": float((neg <= .01).mean()) if len(neg) else None,
                "sceptre_fraction_below_0.05": float((cp <= .05).mean()) if len(cp) else None,
                "sceptre_fraction_below_0.01": float((cp <= .01).mean()) if len(cp) else None}
        for name, path in (("rapidglm", pilot / "runs" / f"{sc_name(args)}.timing.json"),
                           ("sceptre", sceptre_dir / "process.json")):
            if path.exists():
                sc[f"{name}_timing"] = json.loads(path.read_text())
        summary["single_cell"] = sc
    # Pseudobulk.
    pb = pilot / "pseudobulk"
    timing_path = pb / f"timing_t{args.threads}.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())
        rows = {target: pseudobulk_row(pb / "datasets" / target, t) for target, t in timings.items()}
        totals = {key: sum(v[key] for v in rows.values())
                  for key in ("rapidglm_seconds", "deseq2_seconds", "deseq2_fit_seconds",
                              "rapidglm_edger_ql_seconds", "edger_seconds", "edger_ql_seconds")}
        summary["pseudobulk"] = {"targets": rows, "totals": totals}
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=1)[:6000])


# Run records and configurations name their own paths, versions and timings.
VOLATILE_KEYS = {"version", "config", "config_path", "wall_seconds", "peak_rss_kib", "load_average",
                 "seconds", "ql_seconds"}


def normalized_record(path):
    """A JSON record without volatile keys; paths reduced to file names, and input
    fingerprints of configuration files (which hold paths) dropped."""
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        return record
    record = {k: v for k, v in record.items() if k not in VOLATILE_KEYS}
    if "input_sha1" in record:
        record["input_sha1"] = {pathlib.Path(k).name: v for k, v in record["input_sha1"].items()
                                if not k.endswith(".json")}
    for key in ("parameters", None):
        target = record.get(key, {}) if key else record
        for k, v in list(target.items()):
            if isinstance(v, str) and v.startswith("/"):
                target[k] = pathlib.Path(v).name
    return record


def identity(new, reference):
    """Each file of the reference dataset folder (except the untimed edgeR fit check)
    against the new one: byte-identical tables, or equal JSON records after
    normalized_record()."""
    result = {}
    for path in sorted(reference.rglob("*")):
        name = str(path.relative_to(reference))
        if path.is_dir() or name.startswith("edger/fit_check"):
            continue
        other = new / name
        if not other.exists():
            result[name] = "missing"
        elif other.read_bytes() == path.read_bytes():
            result[name] = "identical"
        elif path.suffix == ".json" and normalized_record(other) == normalized_record(path):
            result[name] = "equal_record"
        else:
            result[name] = "different"
    return result


def _full_row(job):
    target, d, t, reference = job
    row = pseudobulk_row(d, t)
    files = identity(d, reference / "pseudobulk/datasets" / target) if reference else None
    return target, row, files


def distribution(values):
    v = pd.Series(values, dtype=float)
    finite = v[np.isfinite(v)]
    return {"n": int(v.notna().sum()), "non_finite": int((~np.isfinite(v.dropna())).sum()),
            "median": float(v.median()) if v.notna().any() else None,
            "min": float(v.min()) if v.notna().any() else None, "max": float(v.max()) if v.notna().any() else None,
            "quantiles": {str(q): float(finite.quantile(q)) for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)}
            if len(finite) else {}}


def full_compare(args):
    import multiprocessing
    pb = args.pilot / "pseudobulk"
    prepared = json.loads((pb / "full_targets.json").read_text())
    runs = {tool: json.loads((pb / f"full_{tool}.json").read_text()) for tool in TOOLS}
    reference = args.identical_to.resolve() if args.identical_to else None
    jobs, incomplete = [], {}
    for target, info in sorted(prepared["targets"].items()):
        absent = [tool for tool in TOOLS if target not in runs[tool]["per_target"]]
        if absent:
            incomplete[target] = {tool: runs[tool]["failed"].get(target, "not run") for tool in absent}
            continue
        jobs.append((target, pb / "datasets" / target,
                     {**info, **{tool: runs[tool]["per_target"][target] for tool in TOOLS}}, reference))
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = pool.map(_full_row, jobs, chunksize=4)
    rows = {target: row for target, row, _ in results}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    summary = {"version": runs["rapidglm"]["version"], "binary": runs["rapidglm"]["binary"],
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "perturbations": len(prepared["targets"]) + len(prepared["skipped"]),
               "compared": len(rows), "skipped_at_prepare": prepared["skipped"],
               "incomplete": incomplete, "agreement": {}, "timing": {}}
    # Agreement, as in the compare stage: bulk-de against DESeq2, edger-ql against edgeR QL.
    for metric, worse in (("jaccard", "low"), ("max_abs_log10_p_difference", "high"),
                          ("max_abs_lfc_difference", "high"), ("edger_jaccard", "low"),
                          ("edger_max_abs_log10_p_difference", "high")):
        column = frame[metric].astype(float)
        order = column.sort_values(ascending=worse == "low", na_position="first")
        entry = distribution(column)
        entry["worst"] = [{"target": t, metric: None if pd.isna(order[t]) else float(order[t]),
                           **{k: rows[t][k] for k in ("samples", "perturbed_samples", "rapidglm_hits",
                                                      "deseq2_hits", "edger_ql_hits", "edger_hits")}}
                          for t in order.index[:args.worst]]
        summary["agreement"][metric] = entry
    for key in ("rapidglm_hits", "deseq2_hits", "edger_ql_hits", "edger_hits"):
        summary["agreement"][key] = {**distribution(frame[key]), "total": int(frame[key].sum())}
    for metric, threshold in (("jaccard", 0.9), ("edger_jaccard", 0.9)):
        summary["agreement"][f"{metric}_at_least_{threshold}"] = int((frame[metric] >= threshold).sum())
    # Timing: per-perturbation seconds (16 single-thread jobs at once) and the screen's wall time.
    for tool in TOOLS:
        run = runs[tool]
        per = [run["per_target"][t] for t in rows]
        entry = {"screen_wall_seconds": run["wall_seconds"], "workers": run["workers"], "cpus": run["cpus"],
                 "started_utc": run["started_utc"], "finished_utc": run["finished_utc"],
                 "others_cpu": run["others_cpu"], "contended": run["contended"],
                 "load_before": run["load_before"], "succeeded": run["succeeded"], "failed": run["failed"],
                 "wall_seconds": {**distribution([p["wall_seconds"] for p in per]),
                                  "sum": round(sum(p["wall_seconds"] for p in per), 1)},
                 "peak_rss_gb": distribution([p["peak_rss_gb"] for p in per]),
                 "job_others_cpu": distribution([p["others_cpu"] for p in per])}
        for internal in ("deseq_seconds", "ql_seconds"):
            if per and internal in per[0]:
                entry[internal] = {**distribution([p[internal] for p in per]),
                                   "sum": round(sum(p[internal] for p in per), 1)}
        summary["timing"][tool] = entry
    if reference:
        files = {target: f for target, _, f in results}
        flat = [state for f in files.values() for state in f.values()]
        groups = {"inputs": ("counts.mtx", "design.tsv", "x.tsv", "samples.tsv"),
                  "rapidglm": ("rapidglm.", "edger_ql."), "deseq2": ("deseq2/",), "edger": ("edger/",)}
        summary["identity"] = {
            "reference": str(reference),
            "files_checked": len(flat), "states": pd.Series(flat).value_counts().to_dict(),
            "by_group": {g: {s: sum(1 for f in files.values() for n, st in f.items() if n.startswith(p) and st == s)
                             for s in ("identical", "equal_record", "different", "missing")}
                         for g, p in groups.items()},
            "not_identical": {t: {n: s for n, s in f.items() if s not in ("identical", "equal_record")}
                              for t, f in files.items()
                              if any(s not in ("identical", "equal_record") for s in f.values())},
            "files": files}
        pilot_timing = reference / "pseudobulk/timing_t1.json"
        if pilot_timing.exists():
            alone = json.loads(pilot_timing.read_text())
            summary["beside_pilot"] = {
                t: {tool: {"pilot_alone_seconds": alone[t][tool]["wall_seconds"],
                           "sixteen_at_once_seconds": runs[tool]["per_target"][t]["wall_seconds"],
                           **{f"pilot_{k}": alone[t][tool][k] for k in ("deseq_seconds", "ql_seconds")
                              if k in alone[t][tool]},
                           **{k: runs[tool]["per_target"][t][k] for k in ("deseq_seconds", "ql_seconds")
                              if k in runs[tool]["per_target"][t]}}
                    for tool in TOOLS} for t in rows if t in alone}
    summary["targets"] = rows
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("targets", "identity", "beside_pilot")},
                     indent=1)[:8000])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["sc", "sceptre", "pseudobulk", "compare",
                                          "full-prepare", "full", "full-compare"])
    parser.add_argument("--pilot", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    parser.add_argument("--cpus", default="16-31")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--formula", default="~ log_umi + log_detected + mito + batch")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--fit-check", default="strict", choices=["strict", "ks"],
                        help="ks: SCEPTRE's acceptance of the skew-normal fit (the like-for-like run)")
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--tool", choices=TOOLS, help="full: the tool to run over the screen")
    parser.add_argument("--targets", type=pathlib.Path, help="full-prepare: only these perturbations")
    parser.add_argument("--jobs", type=int, default=8, help="full-prepare, full-compare: processes")
    parser.add_argument("--worst", type=int, default=10, help="full-compare: perturbations listed per metric")
    parser.add_argument("--identical-to", type=pathlib.Path,
                        help="full-compare: an earlier pilot folder whose outputs must match")
    args = parser.parse_args()
    args.pilot = args.pilot.resolve()
    if args.stage == "full" and not args.tool:
        parser.error("full needs --tool")
    {"sc": run_sc, "sceptre": run_sceptre, "pseudobulk": run_pseudobulk, "compare": compare,
     "full-prepare": full_prepare, "full": run_full, "full-compare": full_compare}[args.stage](args)


if __name__ == "__main__":
    main()
