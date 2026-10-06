#!/usr/bin/env python3
"""Second calibration check for the pseudobulk QL test: simulated null data.

Counts are simulated with no effect of the tested column, so every p-value is null.
Per gene: normalized mean log-uniform on [5, 5000]; dispersion 0.05 + 0.5 / mean
(close to the Kang pseudobulk trend) times exp(N(0, 0.5^2)); sample size factors
exp(N(0, 0.3^2)), supplied as offsets. Configurations (20 datasets x 2,000 genes each):
  A  two groups, 3 samples each      B  two groups, 5 each      C  two groups, 10 each
  D  donor-paired: 8 donors x 2 conditions, donor effects exp(N(0, 0.3^2)) per gene
  E  as B, but log-normal Poisson counts with the same variance (misspecified model)
References run as black boxes (scripts/reference_simulated_null.R): edgeR QL,
glmGamPoi QL and DESeq2 Wald. RapidGLM also reports its LR and Wald tests and bulk-de.

Gates, fixed before the first run and applied to test=ql on A-D (the NB model):
  rejection rate at most 5.5%, 1.2% and 0.15% at p <= 0.05, 0.01 and 0.001, and at
  least 3.5% at 0.05 (not grossly conservative); valid p-values for >= 99.5% of tests.
E is reported without a gate.
"""

import argparse, json, os, pathlib, subprocess
import numpy as np
import pandas as pd
from scipy import io, sparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
GENES, DATASETS = 2000, 20
CONFIGS = {"A": ("groups", 3, "nb"), "B": ("groups", 5, "nb"), "C": ("groups", 10, "nb"),
           "D": ("paired", 8, "nb"), "E": ("groups", 5, "lognormal")}
GATE = dict(rate_0_05=(0.035, 0.055), rate_0_01=0.012, rate_0_001=0.0015, valid=0.995)


def simulate(directory, config, seed):
    layout, n, law = config
    rng = np.random.default_rng(seed)
    if layout == "groups":
        condition = np.repeat([0, 1], n)
        X = np.column_stack([np.ones(2 * n), condition])
        donor_effect = np.zeros((GENES, 2 * n))
    else:
        donors = np.repeat(np.arange(n), 2)
        condition = np.tile([0, 1], n)
        X = np.column_stack([np.ones(2 * n)] + [(donors == d).astype(float) for d in range(1, n)] + [condition])
        donor_effect = rng.normal(0, 0.3, (GENES, n))[:, donors]
    m = X.shape[0]
    sf = np.exp(rng.normal(0, 0.3, m))
    base = np.exp(rng.uniform(np.log(5), np.log(5000), GENES))
    dispersion = (0.05 + 0.5 / base) * np.exp(rng.normal(0, 0.5, GENES))
    mu = base[:, None] * sf[None, :] * np.exp(donor_effect)
    if law == "nb":
        Y = rng.poisson(rng.gamma(1 / dispersion[:, None], mu * dispersion[:, None]))
    else:
        s2 = np.log1p(dispersion)[:, None]
        Y = rng.poisson(mu * np.exp(rng.normal(-s2 / 2, np.sqrt(s2), mu.shape)))
    directory.mkdir(parents=True, exist_ok=True)
    io.mmwrite(str(directory / "counts.mtx"), sparse.coo_matrix(Y), field="integer")
    np.savetxt(directory / "design.tsv", X, delimiter="\t", fmt="%g")
    np.savetxt(directory / "offset.tsv", np.log(sf), fmt="%.17g")


def native(binary, directory, command, test):
    prefix = directory / f"rapid.{command}.{test}"
    config = dict(responses=str(directory / "counts.mtx"), design=str(directory / "design.tsv"),
                  family="nb", test=test, output=str(prefix))
    if command == "pseudobulk-de":
        config["offset"] = str(directory / "offset.tsv")
    path = pathlib.Path(str(prefix) + ".json")
    path.write_text(json.dumps(config))
    subprocess.run([str(binary), command, str(path)], check=True, env=ENV, stdout=subprocess.DEVNULL)
    table = pd.read_csv(str(prefix) + ".tsv", sep="\t", na_values="NA")
    return table["pvalue" if command == "bulk-de" else "p_value"].to_numpy(float)


def rates(p):
    p = np.asarray(p, float)
    valid = np.isfinite(p)
    return dict(tests=int(p.size), valid=float(valid.mean()),
                rate_0_05=float(np.mean(p[valid] <= 0.05)), rate_0_01=float(np.mean(p[valid] <= 0.01)),
                rate_0_001=float(np.mean(p[valid] <= 0.001)))


def main(args):
    work = args.work.resolve()
    results, passed = {}, True
    for label, config in CONFIGS.items():
        root = work / label
        names = [f"d{k:02d}" for k in range(DATASETS)]
        for k, name in enumerate(names):
            if not (root / name / "counts.mtx").exists():
                simulate(root / name, config, seed=1000 * (ord(label) - ord("A") + 1) + k)
        (root / "datasets.txt").write_text("\n".join(names) + "\n")
        mine = {key: [] for key in ("ql", "lr", "wald", "bulk-de")}
        for name in names:
            d = root / name
            for test in ("ql", "lr", "wald"):
                mine[test].append(native(args.binary, d, "pseudobulk-de", test))
            mine["bulk-de"].append(native(args.binary, d, "bulk-de", "wald"))
        reference = root / "reference.tsv"
        if not reference.exists():
            subprocess.run(["Rscript", str(ROOT / "scripts/reference_simulated_null.R"), str(root), str(reference),
                            args.r_library], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ref = pd.read_csv(reference, sep="\t", na_values="NA")
        row = {f"rapidglm_{k}": rates(np.concatenate(v)) for k, v in mine.items()}
        for tool in ("edgeR_ql", "glmGamPoi_ql", "DESeq2_wald"):
            row[tool] = rates(ref[tool])
        row["design"] = dict(layout=config[0], per_group_or_donors=config[1], counts=config[2])
        ql = row["rapidglm_ql"]
        if label != "E":
            ok = (GATE["rate_0_05"][0] <= ql["rate_0_05"] <= GATE["rate_0_05"][1] and
                  ql["rate_0_01"] <= GATE["rate_0_01"] and ql["rate_0_001"] <= GATE["rate_0_001"] and
                  ql["valid"] >= GATE["valid"])
            row["gate_passed"] = ok
            passed &= ok
        results[label] = row
        print(label, {k: (round(v["rate_0_05"], 4), round(v["rate_0_01"], 4), round(v["rate_0_001"], 5))
                      for k, v in row.items() if isinstance(v, dict) and "rate_0_05" in v}, flush=True)
    receipt = dict(
        purpose=__doc__.strip().splitlines()[0],
        commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        worktree_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()),
        command="python3 scripts/validate_ql_simulation.py --work DIR --output validation/QL_SIMULATION.json",
        gate=GATE, configurations=results, passed=passed)
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("passed", passed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/rapidglm")
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--r-library",
                    default=f"{ROOT / 'validation/local-data/R_glmgampoi'}:{ROOT / 'validation/local-data/R_library'}")
    main(ap.parse_args())
