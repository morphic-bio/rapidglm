#!/usr/bin/env python3
"""Mock-comparison null for pseudobulk DE on the Kang CD14 monocyte panel.

Within one condition (8 donors), every balanced 4-versus-4 donor split is a
comparison with no designed effect. The native CLI and, optionally, black-box
edgeR/DESeq2 (and glmGamPoi, when installed) runs
(scripts/reference_pseudobulk_null.R) test the split label
with the same library-size offsets. Donor differences are real biological
variation that a calibrated method must absorb, so rejection rates measure
calibration under realistic between-replicate variance.
"""

import argparse, itertools, json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import io, stats

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "validation/work/applications/pseudobulk"
VARIANTS = {
    "native_ml_wald": dict(test="wald"),
    "native_ml_lr": dict(test="lr"),
    "native_shared_wald": dict(test="wald", dispersion="shared"),
    "native_ql": dict(test="ql"),
}


def main(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    counts = io.mmread(PANEL / "responses.mtx").tocsc()
    design = np.loadtxt(PANEL / "design.tsv")
    offset = np.loadtxt(PANEL / "offset.tsv")
    splits = []
    rows = []
    for condition, label in ((0, "ctrl"), (1, "stim")):
        samples = np.flatnonzero(design[:, -1] == condition)
        sub = counts[:, samples]
        mtx = work / f"{label}.mtx"
        io.mmwrite(str(mtx), sub.astype(int), field="integer")
        np.savetxt(work / f"{label}.offset.tsv", offset[samples])
        # Fixing the first donor in group 0 enumerates each partition once.
        for index, rest in enumerate(itertools.combinations(range(1, 8), 3)):
            group = np.ones(8, int)
            group[[0, *rest]] = 0
            splits.append(dict(condition=label, split=index, group=group.tolist()))
            x = np.column_stack([np.ones(8), group])
            dfile = work / f"{label}.{index}.design.tsv"
            np.savetxt(dfile, x, fmt="%d", delimiter="\t")
            for name, extra in VARIANTS.items():
                prefix = work / f"{label}.{index}.{name}"
                config = dict(
                    responses=str(mtx),
                    design=str(dfile),
                    offset=str(work / f"{label}.offset.tsv"),
                    family="nb",
                    coefficient=1,
                    threads=args.threads,
                    output=str(prefix),
                    **extra,
                )
                path = prefix.with_suffix(".json")
                path.write_text(json.dumps(config))
                subprocess.run(
                    [str(args.binary), "pseudobulk-de", str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
                )
                table = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")
                rows.append(
                    pd.DataFrame(
                        dict(
                            condition=label,
                            split=index,
                            feature=table.feature,
                            method=name,
                            p=table.p_value,
                        )
                    )
                )
    (work / "splits.json").write_text(json.dumps(splits))
    native = pd.concat(rows)
    frames = [native]
    if args.r_library:
        subprocess.run(
            [
                "Rscript",
                str(ROOT / "scripts/reference_pseudobulk_null.R"),
                str(work),
                str(args.r_library),
            ],
            check=True,
            cwd=ROOT,
        )
        frames.append(pd.read_csv(work / "reference.tsv", sep="\t"))
    data = pd.concat(frames)
    summary = []
    for method, d in data.groupby("method"):
        p = d.p.to_numpy(float)
        valid = np.isfinite(p)
        per_split = []
        for _, s in d.groupby(["condition", "split"]):
            q = stats.false_discovery_control(np.nan_to_num(s.p.to_numpy(float), nan=1))
            per_split.append(int(np.sum(q <= 0.05)))
        summary.append(
            dict(
                method=method,
                tests=int(len(p)),
                valid=int(valid.sum()),
                rate_p_le_0_05=float(np.mean(p[valid] <= 0.05)),
                rate_p_le_0_01=float(np.mean(p[valid] <= 0.01)),
                rate_p_le_0_001=float(np.mean(p[valid] <= 0.001)),
                ks_statistic=float(stats.kstest(p[valid], "uniform").statistic),
                bh_0_05_discoveries_per_split_mean=float(np.mean(per_split)),
                splits_with_any_bh_0_05_discovery=int(np.sum(np.array(per_split) > 0)),
                splits=len(per_split),
            )
        )
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        panel=str(PANEL),
        design="intercept + 4-vs-4 split label; offsets log library size; all 35 partitions of 8 donors in each condition",
        genes=int(counts.shape[0]),
        load_average=os.getloadavg(),
        summary=summary,
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    for s in summary:
        print(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--r-library", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    main(ap.parse_args())
