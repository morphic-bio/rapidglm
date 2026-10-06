#!/usr/bin/env python3
import json, pathlib, subprocess
import numpy as np, pandas as pd
from scipy import stats
from validate_glm import ROOT, sha

root = ROOT / "validation/work/sceptre"
results = []
for d in sorted(root.iterdir()):
    if not d.is_dir() or not (d / "native.tsv").exists():
        continue
    ref = d / "reference.tsv"
    if not ref.exists():
        ref = d / "reference_cached.tsv"
    if not ref.exists():
        continue
    native = pd.read_csv(d / "native.tsv", sep="\t").merge(
        pd.read_csv(d / "response_ids.tsv", sep="\t"), on="response"
    )
    reference = pd.read_csv(ref, sep="\t")
    m = native.merge(
        reference,
        left_on=["response_id", "target"],
        right_on=["response_id", "grna_target"],
        suffixes=("_native", "_reference"),
    )
    usable = m[np.isfinite(m.p_value_native) & np.isfinite(m.p_value_reference)].copy()
    pn = usable.p_value_native.to_numpy()
    pr = usable.p_value_reference.to_numpy()
    qn = stats.false_discovery_control(pn) if len(pn) else []
    qr = stats.false_discovery_control(pr) if len(pr) else []
    nd = qn <= 0.1
    rd = qr <= 0.1
    shared = nd & rd
    union = nd | rd

    def corr(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        return (
            float(stats.pearsonr(np.asarray(a)[valid], np.asarray(b)[valid]).statistic)
            if sum(valid) > 2
            else None
        )

    result = dict(
        panel=d.name,
        reference_scope=(
            "fresh black-box SCEPTRE 0.99.0"
            if ref.name == "reference.tsv"
            else "cached official results; matched rows only"
        ),
        requested_rows=len(native),
        reference_rows=len(reference),
        matched_rows=len(m),
        usable_rows=len(usable),
        qc_treatment_equal=bool(np.all(m.nonzero_treatment == m.n_nonzero_trt)),
        qc_control_equal=bool(np.all(m.nonzero_control == m.n_nonzero_cntrl)),
        p_spearman=float(stats.spearmanr(pn, pr).statistic) if len(pn) > 2 else None,
        log2fc_correlation=corr(usable.log2fc, usable.log_2_fold_change),
        mean_ratio_log2_correlation=corr(
            usable.mean_ratio_log2, usable.log_2_fold_change
        ),
        native_discoveries=int(sum(nd)),
        reference_discoveries=int(sum(rd)),
        discovery_jaccard=float(sum(shared) / sum(union)) if sum(union) else None,
        shared_discovery_sign_agreement=(
            float(
                np.mean(
                    np.sign(usable.log2fc.to_numpy()[shared])
                    == np.sign(usable.log_2_fold_change.to_numpy()[shared])
                )
            )
            if sum(shared)
            else None
        ),
        native_run=json.loads((d / "native.run.json").read_text()),
        input_sha256={
            p.name: sha(p)
            for p in d.iterdir()
            if p.suffix == ".mtx"
            or p.name in ("design.tsv", "covariates.tsv", "guides.tsv", "pairs.tsv")
        },
        reference_sha256=sha(ref),
    )
    if (d / "reference.json").exists():
        result["reference_run"] = json.loads((d / "reference.json").read_text())
    results.append(result)
receipt = dict(
    gate="G4",
    thresholds="provisional",
    status="PARTIAL_CONCORDANCE; complete calibration/power control panels not yet qualified",
    command="python3 scripts/compare_sceptre.py",
    panels=results,
    limitations=[
        "Packaged 0.99.0 examples are 1000/500-cell subsets, not full sceptredata S1/S2.",
        "P-value and effect estimators differ as documented in METHODS.md.",
        "Empty discovery sets have undefined Jaccard, not a passing value.",
        "No G6 speed target is qualified by cached comparator timing.",
    ],
)
(ROOT / "validation/SCEPTRE_CONCORDANCE.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n"
)
for r in results:
    print(r["panel"], r["p_spearman"], r["discovery_jaccard"], r["log2fc_correlation"])
