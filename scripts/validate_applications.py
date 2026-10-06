#!/usr/bin/env python3
import json, pathlib, subprocess
import numpy as np, pandas as pd
from scipy import sparse, io, stats
from validate_glm import ROOT, sha

work = ROOT / "validation/work/applications"
work.mkdir(exist_ok=True, parents=True)


def compare(
    native_p,
    reference_p,
    native_stat,
    reference_stat,
    native_fc=None,
    reference_fc=None,
):
    good = (
        np.isfinite(native_p)
        & np.isfinite(reference_p)
        & np.isfinite(native_stat)
        & np.isfinite(reference_stat)
    )
    a = np.asarray(native_p)[good]
    b = np.asarray(reference_p)[good]
    na = stats.false_discovery_control(np.nan_to_num(native_p, nan=1)) <= 0.05
    nb = stats.false_discovery_control(np.nan_to_num(reference_p, nan=1)) <= 0.05
    result = dict(
        pairs=len(native_p),
        usable=int(sum(good)),
        statistic_spearman=float(
            stats.spearmanr(
                np.asarray(native_stat)[good], np.asarray(reference_stat)[good]
            ).statistic
        ),
        native_discoveries=int(sum(na)),
        reference_discoveries=int(sum(nb)),
        discovery_jaccard=float(sum(na & nb) / sum(na | nb)) if sum(na | nb) else None,
        max_p_absolute_difference=float(max(abs(a - b))),
    )
    if native_fc is not None:
        valid = np.isfinite(native_fc) & np.isfinite(reference_fc)
        result["log2fc_correlation"] = float(
            stats.pearsonr(
                np.asarray(native_fc)[valid], np.asarray(reference_fc)[valid]
            ).statistic
        )
    return result


p = work / "pseudobulk"
subprocess.run(
    [ROOT / "build/rapidglm", "pseudobulk-de", p / "config.json"], check=True
)
native = pd.read_csv(p / "native.tsv", sep="\t")
ref = pd.read_csv(p / "reference.tsv", sep="\t")
m = native.merge(ref, on="feature")
comparisons = {}
for name in ["edgeR", "DESeq2"]:
    stat = m[name + "_statistic"]
    stat = np.sign(m[name + "_log2fc"]) * np.sqrt(stat) if name == "edgeR" else stat
    comparisons[name] = compare(
        m.p_value, m[name + "_p"], m.statistic, stat, m.log2fc, m[name + "_log2fc"]
    )
# Known continuous dose, balanced dose groups, prespecified 100 null and 100 effect features.
rng = np.random.default_rng(20260926)
dose = np.repeat(np.arange(4), 20)
x = np.column_stack([np.ones(len(dose)), dose])
effect = np.r_[np.zeros(100), np.full(100, 0.6)]
mu = np.exp(1 + dose[:, None] * effect)
y = rng.negative_binomial(3, 3 / (3 + mu))
d = work / "dose"
d.mkdir(exist_ok=True)
io.mmwrite(d / "responses.mtx", sparse.csr_matrix(y.T), field="integer")
np.savetxt(d / "design.tsv", x, delimiter="\t")
config = dict(
    responses=str(d / "responses.mtx"),
    design=str(d / "design.tsv"),
    output=str(d / "native"),
    family="nb",
    coefficient=1,
    threads=2,
)
(d / "config.json").write_text(json.dumps(config, indent=2))
subprocess.run([ROOT / "build/rapidglm", "dose-trend", d / "config.json"], check=True)
dose_result = pd.read_csv(d / "native.tsv", sep="\t")
disc = dose_result.adjusted_p.to_numpy() <= 0.05
da = ROOT / "validation/work/multimodal/da"
dn = pd.read_csv(da / "native.tsv", sep="\t")
dr = pd.read_csv(da / "reference.tsv", sep="\t")
dm = dn.merge(dr, on="feature")
da_comparison = compare(dm.p_value, dm.p_val, dm.statistic, stats.chi2.isf(dm.p_val, 1))
receipt = dict(
    gate="G8",
    thresholds="provisional",
    command="python3 scripts/validate_applications.py",
    pseudobulk=dict(
        reference=json.loads((p / "reference.json").read_text()),
        comparisons=comparisons,
        native_run=json.loads((p / "native.run.json").read_text()),
        limitations="Native feature-wise ML dispersion differs from edgeR and DESeq2 moderated dispersion; same explicit library offsets, no automatic shrinkage.",
    ),
    dose=dict(
        features=200,
        null_features=100,
        effect_features=100,
        dose_coefficient=0.6,
        seed=20260926,
        detected_effects=int(sum(disc[100:])),
        false_discoveries=int(sum(disc[:100])),
        effect_sign_recovery=int(sum(dose_result.coefficient.to_numpy()[100:] > 0)),
        run=json.loads((d / "native.run.json").read_text()),
    ),
    da=dict(
        comparison=da_comparison,
        reference=json.loads((da / "reference.json").read_text()),
        run=json.loads((da / "native.run.json").read_text()),
    ),
    input_sha256={
        "Kang_EH2259": sha(ROOT / "validation/local-data/Kang18_EH2259.rds"),
        "pseudobulk_counts": sha(p / "responses.mtx"),
        "dose_counts": sha(d / "responses.mtx"),
        "DA_counts": sha(da / "responses.mtx"),
    },
)
(ROOT / "validation/APPLICATIONS.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {"pseudobulk": comparisons, "da": da_comparison, "dose": receipt["dose"]},
        indent=2,
    )
)
