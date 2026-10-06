#!/usr/bin/env python3
"""Gate for the pseudobulk quasi-likelihood test (test=ql).

G1 calibration: in the Kang mock-comparison null (scripts/pseudobulk_null.py,
   70 four-versus-four donor splits x 1000 genes) native QL rejects at most
   5.5%, 1.2% and 0.15% at p <= 0.05, 0.01 and 0.001, makes at most 0.5 BH
   (0.05) discoveries per split on average, and returns valid p-values for at
   least 99.5% of tests.
G2 power: on the real donor-paired stimulation contrast, native QL BH (0.05)
   discoveries agree with glmGamPoi's QL test (Jaccard >= 0.8, discovery count
   ratio in [0.8, 1.25]); edgeR QL is reported for reference.
G3 determinism: the native QL table is byte-identical with 1 and 4 threads.
Thresholds were fixed before the first QL run and are not tuned.
"""

import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "validation/work/applications/pseudobulk"
GATE = dict(
    null_rate_0_05=0.055,
    null_rate_0_01=0.012,
    null_rate_0_001=0.0015,
    null_bh_discoveries_per_split=0.5,
    null_valid_fraction=0.995,
    contrast_jaccard_glmgampoi=0.8,
    contrast_count_ratio=(0.8, 1.25),
)


def run_native(binary, work, threads):
    config = dict(
        responses=str(PANEL / "responses.mtx"),
        design=str(PANEL / "design.tsv"),
        offset=str(PANEL / "offset.tsv"),
        family="nb",
        coefficient=8,
        test="ql",
        threads=threads,
        output=str(work / f"contrast.ql.t{threads}"),
    )
    path = work / f"contrast.ql.t{threads}.json"
    path.write_text(json.dumps(config))
    subprocess.run(
        [str(binary), "pseudobulk-de", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    table = Path(config["output"] + ".tsv")
    record = json.loads(Path(config["output"] + ".run.json").read_text())
    return table, record


def discoveries(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1)
    return set(np.flatnonzero(stats.false_discovery_control(p) <= 0.05))


def main(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # G1: mock-comparison null, with black-box edgeR/DESeq2/glmGamPoi references.
    null_json = work / "null.json"
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/pseudobulk_null.py"),
            "--binary",
            str(args.binary),
            "--work",
            str(work / "null"),
            "--output",
            str(null_json),
            "--r-library",
            args.r_library,
            "--threads",
            str(args.threads),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    null = {s["method"]: s for s in json.loads(null_json.read_text())["summary"]}
    q = null["native_ql"]
    g1 = dict(
        rate_0_05=q["rate_p_le_0_05"] <= GATE["null_rate_0_05"],
        rate_0_01=q["rate_p_le_0_01"] <= GATE["null_rate_0_01"],
        rate_0_001=q["rate_p_le_0_001"] <= GATE["null_rate_0_001"],
        bh=q["bh_0_05_discoveries_per_split_mean"]
        <= GATE["null_bh_discoveries_per_split"],
        valid=q["valid"] >= GATE["null_valid_fraction"] * q["tests"],
    )
    # G2/G3: real contrast.
    t1, record = run_native(args.binary, work, 1)
    t4, _ = run_native(args.binary, work, 4)
    identical = t1.read_bytes() == t4.read_bytes()
    reference = work / "contrast.reference.tsv"
    subprocess.run(
        [
            "Rscript",
            str(ROOT / "scripts/reference_ql_contrast.R"),
            str(PANEL),
            "8",
            str(reference),
            args.r_library,
        ],
        check=True,
        cwd=ROOT,
    )
    native = pd.read_csv(t1, sep="\t", na_values="NA")
    ref = pd.read_csv(reference, sep="\t")
    sets = dict(
        native_ql=discoveries(native.p_value),
        edgeR_ql=discoveries(ref.edgeR_ql),
        glmGamPoi_ql=discoveries(ref.glmGamPoi_ql),
    )
    jaccard = lambda a, b: len(a & b) / max(1, len(a | b))
    ratio = len(sets["native_ql"]) / max(1, len(sets["glmGamPoi_ql"]))
    concordance = {
        f"{a}_vs_{b}": dict(jaccard=jaccard(sets[a], sets[b]), shared=len(sets[a] & sets[b]))
        for a, b in (
            ("native_ql", "glmGamPoi_ql"),
            ("native_ql", "edgeR_ql"),
            ("glmGamPoi_ql", "edgeR_ql"),
        )
    }
    g2 = dict(
        jaccard=concordance["native_ql_vs_glmGamPoi_ql"]["jaccard"]
        >= GATE["contrast_jaccard_glmgampoi"],
        count_ratio=GATE["contrast_count_ratio"][0] <= ratio <= GATE["contrast_count_ratio"][1],
    )
    valid = native.p_value.notna()
    spearman = {
        k: float(stats.spearmanr(native.p_value[valid], ref[k][valid]).statistic)
        for k in ("edgeR_ql", "glmGamPoi_ql")
    }
    passed = all(g1.values()) and all(g2.values()) and identical
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        gate=GATE,
        commands=[
            "python3 scripts/validate_ql.py --work DIR --output validation/QL_TEST.json "
            "--r-library validation/local-data/R_glmgampoi:validation/local-data/R_library"
        ],
        G1_null=dict(checks=g1, native_ql=q, references={k: v for k, v in null.items() if k != "native_ql"}),
        G2_contrast=dict(
            checks=g2,
            discoveries={k: len(v) for k, v in sets.items()},
            count_ratio_native_over_glmgampoi=ratio,
            concordance=concordance,
            spearman_p_native_vs=spearman,
            native_run=dict(
                df_prior=record["ql_df_prior"],
                abundance_trend=record["ql_abundance_trend"],
                trend_window=record["ql_trend_window"],
                dispersion_features=record["ql_dispersion_features"],
                wall_seconds=record["wall_seconds"],
            ),
        ),
        G3_threads=dict(identical=identical, sha256=hashlib.sha256(t1.read_bytes()).hexdigest()),
        passed=passed,
        versions=(work / "null/versions.txt").read_text().splitlines(),
        load_average=os.getloadavg(),
    )
    args.output.write_text(json.dumps(receipt, indent=2, default=float) + "\n")
    print(json.dumps(dict(G1=g1, G2=g2, G3=identical, passed=passed), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--r-library", required=True)
    ap.add_argument("--threads", type=int, default=4)
    main(ap.parse_args())
