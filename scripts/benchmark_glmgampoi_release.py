#!/usr/bin/env python3
"""glmGamPoi's development version, as timed in ENGINE_BATCH.json and APPLICATIONS_BULK.json
(1.25.1), against its Bioconductor 3.21 release (1.20.0), the version that ships with
DESeq2 1.48.2. Both run on the same inputs with the harness's own pieces: the glm_gp script of
benchmarks/glm/batch.py (best of three fits of 1,000 NB responses) and DESeq2 with
fitType = "glmGamPoi" on the four bulk datasets (scripts/reference_bulk_timing.R, one worker).
Within each repeat the two versions run back to back; the receipt keeps the fastest repeat.

Run under the shared timing lock on one efficiency core, for example
  flock LOCK taskset -c 16 python3 scripts/benchmark_glmgampoi_release.py \\
      --release-library DIR --work DIR --output validation/GLMGAMPOI_RELEASE.json
where DIR holds glmGamPoi 1.20.0 (BiocManager::install("glmGamPoi") under Bioconductor 3.21).
--summarize RESULTS.json writes the receipt from an earlier run's results.json."""
import argparse, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/glm"))
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
BASE = ROOT / "validation/local-data/R_library"
BULK = {
    "airway": (ROOT / "validation/work/bulk/airway", "dextrt", 3),
    "thyroid32": (ROOT / "validation/work/benchmark/thyroid32", "sex[T.female]", 3),
    "thyroid128": (ROOT / "validation/work/benchmark/thyroid128", "sex[T.female]", 3),
    "thyroid": (ROOT / "validation/work/benchmark/thyroid", "sex[T.female]", 2),
}


def run(args):
    import batch  # RapidGLM's harness: the data generator and the glm_gp script
    libs = {"1.25.1": f"{ROOT / 'validation/local-data/R_glmgampoi'}:{BASE}",
            "1.20.0": f"{args.release_library.resolve()}:{BASE}"}
    out = args.work.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def record(entry):
        entry["load"] = os.getloadavg()[0]
        results.append(entry)
        print(json.dumps(entry), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))

    d = out / "batch"
    batch.generate(d, 5000, 1000)
    script = d / "glmgampoi.R"
    script.write_text(batch.GGP)
    for version, lib in libs.items():
        r = subprocess.run(["Rscript", str(script), str(d), lib], env=ENV, capture_output=True, text=True, check=True)
        record(dict(task="batch glm_gp", requested=version, **json.loads(r.stdout.strip().splitlines()[-1])))
    for name, (data, coef, repeats) in BULK.items():
        for rep in range(repeats):
            for version, lib in libs.items():
                o = out / f"{name}-{version}-{rep}"
                subprocess.run(["Rscript", str(ROOT / "scripts/reference_bulk_timing.R"), str(data), coef, str(o), lib,
                                "glmGamPoi", "1"], env=ENV, check=True, capture_output=True)
                t = json.loads((o / "timing.json").read_text())
                record(dict(task=f"DESeq2+glmGamPoi {name}", requested=version, repeat=rep, seconds=t["seconds"],
                            glmGamPoi=t["glmGamPoi"], deseq2=t["deseq2"]))
    return results


def summarize(results, output):
    rows = {}
    for r in results:
        version = r.get("version", r.get("glmGamPoi"))
        assert version == r["requested"], r
        best = rows.setdefault(r["task"], {}).get(version)
        if best is None or r["seconds"] < best:
            rows[r["task"]][version] = r["seconds"]
    tasks = [{"task": t, "seconds": v, "release_over_development": v["1.20.0"] / v["1.25.1"]} for t, v in rows.items()]
    ratios = [t["release_over_development"] for t in tasks]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    receipt = {
        "purpose": __doc__.split("\n\n")[0].replace("\n", " "),
        "command": "flock LOCK taskset -c 16 python3 scripts/benchmark_glmgampoi_release.py --release-library DIR "
                   "--work DIR --output validation/GLMGAMPOI_RELEASE.json",
        "commit": commit,
        "host_note": "One workstation (Intel Core i9-13900KF, 125 GiB), 2026-10-14; CPU 16 (an efficiency core), "
                     "one thread, under the shared timing lock. Other sessions ran untimed jobs on other CPUs; the "
                     "load average before each run is in runs.",
        "versions": {"development": "1.25.1", "release": "1.20.0 (Bioconductor 3.21)", "deseq2": "1.48.2"},
        "tasks": tasks,
        "release_over_development": {"min": min(ratios), "max": max(ratios)},
        "runs": results,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("tasks", "release_over_development")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-library", type=pathlib.Path)
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "validation/work/glmgampoi_release")
    ap.add_argument("--summarize", type=pathlib.Path, help="results.json of an earlier run")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    summarize(json.loads(a.summarize.read_text()) if a.summarize else run(a), a.output)
