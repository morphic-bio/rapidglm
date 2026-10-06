#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from scipy import stats
from validate_glm import ROOT, sha

results = []
for panel in ["S1_public", "S2_public"]:
    d = ROOT / "validation/work/public-sceptre" / panel
    native = pd.read_csv(d / "native.tsv", sep="\t").merge(
        pd.read_csv(d / "response_ids.tsv", sep="\t"), on="response"
    )
    record = {
        "panel": panel,
        "reference": json.loads((d / "reference.json").read_text()),
        "native_run": json.loads((d / "native.run.json").read_text()),
        "input_sha256": {
            p.name: sha(p)
            for p in d.iterdir()
            if p.suffix == ".mtx" or p.name in ("pairs.tsv", "guides.tsv", "design.tsv")
        },
    }
    for kind, name, file in [
        (0, "discovery", "reference.tsv"),
        (2, "power", "reference_power.tsv"),
    ]:
        ref = pd.read_csv(d / file, sep="\t")
        m = native[native.kind == kind].merge(
            ref,
            left_on=["response_id", "target"],
            right_on=["response_id", "grna_target"],
            suffixes=("_native", "_reference"),
        )
        good = m[
            np.isfinite(m.p_value_native) & np.isfinite(m.p_value_reference)
        ].copy()
        pn = np.nan_to_num(m.p_value_native.to_numpy(), nan=1)
        pr = np.nan_to_num(m.p_value_reference.to_numpy(), nan=1)
        dn = stats.false_discovery_control(pn) <= 0.1
        dr = stats.false_discovery_control(pr) <= 0.1
        fc = good[np.isfinite(good.log2fc) & np.isfinite(good.log_2_fold_change)]
        record[name] = {
            "requested_native": int(sum(native.kind == kind)),
            "reference_pairs": len(ref),
            "matched_pairs": len(m),
            "usable": len(good),
            "p_spearman": float(
                stats.spearmanr(good.p_value_native, good.p_value_reference).statistic
            ),
            "log2fc_correlation": float(
                stats.pearsonr(fc.log2fc, fc.log_2_fold_change).statistic
            ),
            "native_discoveries": int(sum(dn)),
            "reference_discoveries": int(sum(dr)),
            "discovery_jaccard": (
                float(sum(dn & dr) / sum(dn | dr)) if sum(dn | dr) else None
            ),
            "shared_sign_agreement": (
                float(
                    np.mean(
                        np.sign(m.log2fc.to_numpy()[dn & dr])
                        == np.sign(m.log_2_fold_change.to_numpy()[dn & dr])
                    )
                )
                if sum(dn & dr)
                else None
            ),
            "qc_treatment_equal": bool(np.all(m.nonzero_treatment == m.n_nonzero_trt)),
            "qc_control_equal": bool(np.all(m.nonzero_control == m.n_nonzero_cntrl)),
        }
    nr = native[native.kind == 1]
    ref = pd.read_csv(d / "reference_calibration.tsv", sep="\t")
    qq = pd.read_csv(d / "native.qq.tsv", sep="\t")
    record["calibration"] = {
        "scope": "different negative-pair samples; diagnostic, not matched-pair qualification",
        "native_pairs": len(nr),
        "reference_pairs": len(ref),
        "native_valid": int(nr.p_value.notna().sum()),
        "reference_valid": int(ref.p_value.notna().sum()),
        "native_BH_01": int(sum(nr.adjusted_p <= 0.1)),
        "reference_BH_01": int(
            sum(stats.false_discovery_control(ref.p_value.fillna(1)) <= 0.1)
        ),
        "native_Bonferroni_01": int(sum(nr.p_value <= 0.1 / len(nr))),
        "reference_Bonferroni_01": int(sum(ref.p_value <= 0.1 / len(ref))),
        "native_QQ_points_outside_pointwise_95": int(
            sum((qq.observed < qq.lower_95) | (qq.observed > qq.upper_95))
        ),
    }
    results.append(record)
provenance = json.loads(
    (ROOT / "validation/local-data/sceptredata/provenance.json").read_text()
)
provenance["data_sha256"] = {
    p.name: sha(p) for p in (ROOT / "validation/local-data/sceptredata").glob("*.rda")
}
receipt = {
    "gate": "G4",
    "thresholds": "provisional",
    "command": "python3 scripts/compare_public.py",
    "data": provenance,
    "panels": results,
    "limitations": [
        "Discovery and power panels matched; negative-control panels differ.",
        "Continuous cell QC clipping disabled identically; low-MOI assignment QC enabled.",
        "No complete matched negative and positive control families on the other panels; no full G4 qualification.",
    ],
}
(ROOT / "validation/PUBLIC_SCEPTRE.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n"
)
for r in results:
    print(r["panel"], r["discovery"], r["power"], r["calibration"])
