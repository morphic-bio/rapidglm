#!/usr/bin/env python3
"""Compare native and R convergence using shared data and coefficient starts."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from scipy.special import expit, gammaln, ndtr, softmax, xlogy

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def paired_poisson_oracle(x, y, offset):
    """Profile donor intercepts analytically; solve the one remaining score."""
    assert np.all(x[:, 0] == 1)
    _, groups = np.unique(x[:, 1:-1], axis=0, return_inverse=True)
    pairs = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    assert all(len(ids) == 2 and np.array_equal(x[ids, -1], [0, 1]) for ids in pairs)
    totals = np.array([y[ids].sum() for ids in pairs])
    offset_difference = np.array([offset[ids[1]] - offset[ids[0]] for ids in pairs])
    observed_treatment = y[x[:, -1] == 1].sum()
    effect = brentq(
        lambda b: observed_treatment - totals @ expit(b + offset_difference),
        -100,
        100,
        xtol=5e-15,
    )
    mu = np.zeros(len(y))
    for ids, total in zip(pairs, totals):
        mu[ids] = total * softmax(offset[ids] + effect * x[ids, -1])
    ll = np.sum(xlogy(y, mu) - mu - gammaln(y + 1))
    probability = expit(effect + offset_difference)
    se = 1 / np.sqrt(np.sum(totals * probability * (1 - probability)))
    return dict(
        treatment_coefficient=float(effect),
        treatment_standard_error=float(se),
        treatment_wald_p=float(2 * ndtr(-abs(effect / se))),
        log_likelihood_supremum=float(ll),
        fitted_means=mu.tolist(),
        zero_total_donors=int(sum(totals == 0)),
    )


def prefix_metrics(frame, start):
    pieces = []
    beta_columns = [f"beta{j}" for j in range(len(start))]
    for model, block in frame.groupby("model", sort=False):
        block = block.sort_values("maxit").copy()
        block["relative_coefficient_change"] = np.nan
        block["relative_deviance_change"] = np.nan
        previous = None
        for idx, row in block.iterrows():
            # A larger budget after natural convergence is the same final fit,
            # not another algorithm iteration. Do not plot it as a zero step.
            if row.iterations != row.maxit:
                continue
            if row.maxit == 1:
                prior_beta = start
            elif previous is not None and row.maxit == previous.maxit + 1:
                prior_beta = previous[beta_columns].to_numpy(dtype=float)
                block.at[idx, "relative_deviance_change"] = abs(
                    row.deviance - previous.deviance
                ) / (0.1 + abs(row.deviance))
            else:
                prior_beta = None
            if prior_beta is not None:
                block.at[idx, "relative_coefficient_change"] = np.max(
                    np.abs(row[beta_columns].to_numpy(dtype=float) - prior_beta)
                    / (1 + np.abs(prior_beta))
                )
            previous = row
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="build/rapidglm_boundary_diagnostic")
    ap.add_argument("--cli", default="build/rapidglm")
    ap.add_argument("--pseudobulk", default="validation/work/applications/pseudobulk")
    ap.add_argument("--work", default="validation/work/boundary-diagnostic")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--receipt")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = [
        [
            "Rscript",
            "scripts/reference_boundary.R",
            str(work),
            str(Path(args.pseudobulk).resolve()),
            str(args.seed),
        ],
        [str(Path(args.binary).resolve()), str(work / "manifest.json")],
    ]
    for i, command in enumerate(commands):
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True
        )
        (work / f"command-{i}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    manifest = json.loads((work / "manifest.json").read_text())
    reference = json.loads((work / "reference_environment.json").read_text())
    summary = []
    data_hashes = {}
    for case in manifest["cases"]:
        d = Path(case["directory"])
        binary = {
            k: np.fromfile(p, dtype="<f8") for k, p in case["binary_inputs"].items()
        }
        x = binary["design"].reshape(
            (case["observations"], case["parameters"]), order="F"
        )
        y, offset, start = (binary[k] for k in ("response", "offset", "start"))
        canonical = (
            np.concatenate((x.ravel(), y, offset, binary["weights"], start))
            .astype("<f8")
            .tobytes()
        )
        assert (d / "native_inputs.f64le").read_bytes() == canonical, case["name"]
        oracle = paired_poisson_oracle(x, y, offset)
        frames = {}
        runs = {}
        for engine, prefix in (("native", "native"), ("R", "r")):
            frame = pd.read_csv(d / f"{prefix}_prefix.tsv", sep="\t")
            frame = prefix_metrics(frame, start)
            frame.to_csv(d / f"{prefix}_metrics.tsv", sep="\t", index=False)
            frames[engine] = frame
            runs[engine] = pd.read_csv(d / f"{prefix}_runs.tsv", sep="\t")
        r_converged = (
            frames["R"].query("model == 'poisson' and status == 'converged'").iloc[0]
        )
        k = int(r_converged.iterations)
        native_at_k = (
            frames["native"].query("model == 'poisson' and maxit == @k").iloc[0]
        )
        compared = [f"beta{j}" for j in range(x.shape[1])]
        selected = [
            "maxit",
            "iterations",
            "status",
            "boundary_beta",
            "boundary_mu",
            "log_likelihood",
            "relative_coefficient_change",
            "relative_deviance_change",
        ]
        detail = dict(
            case=case["name"],
            origin=case["origin"],
            poisson_oracle=oracle,
            native_input_bitwise_verified=True,
            R_first_poisson_convergence=clean(r_converged[selected].to_dict()),
            native_at_same_budget=clean(native_at_k[selected].to_dict()),
            same_budget_max_coefficient_difference=float(
                np.max(
                    np.abs(
                        native_at_k[compared].to_numpy(dtype=float)
                        - r_converged[compared].to_numpy(dtype=float)
                    )
                )
            ),
            same_budget_max_mean_difference=float(
                np.max(
                    np.abs(
                        native_at_k[[f"mu{i}" for i in range(len(y))]].to_numpy(
                            dtype=float
                        )
                        - r_converged[[f"mu{i}" for i in range(len(y))]].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),
            native_runs=clean(runs["native"].to_dict("records")),
            R_runs=clean(runs["R"].to_dict("records")),
        )
        final_poisson = (
            runs["native"]
            .query("model == 'poisson' and start == 'shared' and maxit == 100")
            .iloc[0]
        )
        detail["native_final_treatment_error_against_profile_oracle"] = (
            final_poisson[f"beta{x.shape[1] - 1}"] - oracle["treatment_coefficient"]
        )
        detail["native_final_treatment_se_error_against_profile_oracle"] = (
            final_poisson[f"se{x.shape[1] - 1}"] - oracle["treatment_standard_error"]
        )
        detail["native_final_treatment_p_error_against_profile_oracle"] = (
            final_poisson[f"wald_p{x.shape[1] - 1}"] - oracle["treatment_wald_p"]
        )
        detail["native_final_max_mean_error_against_profile_oracle"] = float(
            np.max(
                np.abs(
                    final_poisson[[f"mu{i}" for i in range(len(y))]].to_numpy(
                        dtype=float
                    )
                    - np.asarray(oracle["fitted_means"])
                )
            )
        )
        boundary_column = case["boundary_column"]
        expected_boundary = "positive_control" not in case["name"]
        checked = runs["native"].query("maxit == 100")
        detail["native_boundary_inference_checks"] = dict(
            expected_boundary=expected_boundary,
            statuses_match=bool(
                (
                    checked.status
                    == ("converged_boundary" if expected_boundary else "converged")
                ).all()
            ),
            flags_match=bool(
                (checked[f"boundary{boundary_column}"] == int(expected_boundary)).all()
            ),
            donor_inference_available=bool(
                checked[f"wald_p{boundary_column}"].notna().all()
            ),
            donor_inference_all_missing=bool(
                checked[f"wald_p{boundary_column}"].isna().all()
                and checked[f"se{boundary_column}"].isna().all()
            ),
            treatment_inference_available=bool(
                checked[f"wald_p{x.shape[1]-1}"].notna().all()
            ),
        )
        for engine, row in (("R", r_converged), ("native", native_at_k)):
            detail[f"{engine}_gap_to_poisson_likelihood_supremum"] = (
                oracle["log_likelihood_supremum"] - row.log_likelihood
            )
            detail[f"{engine}_treatment_error_against_profile_oracle"] = (
                row[f"beta{x.shape[1]-1}"] - oracle["treatment_coefficient"]
            )
        summary.append(detail)
        data_hashes[case["name"]] = {
            k: sha(case[k])
            for k in ("design", "response", "offset", "weights", "start")
        }
        data_hashes[case["name"]]["binary_inputs"] = {
            k: sha(p) for k, p in case["binary_inputs"].items()
        }

    # Preserve the original text-input CLI reproduction separately from the
    # bit-matched diagnostic executable. Repeated concurrent fits also expose
    # unsafe numerical-library builds independently of any random draws.
    real_cases = [c for c in manifest["cases"] if c["name"] in ("NME7", "MRPS25")]
    responses_path = work / "original_responses.mtx"
    mmwrite(
        responses_path,
        csr_matrix(
            np.stack(
                [
                    np.fromfile(c["binary_inputs"]["response"], dtype="<f8")
                    for c in real_cases
                ]
            )
        ),
    )
    cli_replays = []
    for threads in (1, 2, 8, 16):
        for repeat in range(3):
            prefix = work / f"original_cli_t{threads}_r{repeat}"
            cfg = dict(
                design=str(Path(args.pseudobulk).resolve() / "design.tsv"),
                offset=str(Path(args.pseudobulk).resolve() / "offset.tsv"),
                responses=str(responses_path),
                output=str(prefix),
                family="nb",
                max_iterations=1000,
                threads=threads,
                tolerance=1e-9,
            )
            config_path = prefix.with_suffix(".json")
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            command = [str(Path(args.cli).resolve()), "fit", str(config_path)]
            result = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True
            )
            prefix.with_suffix(".log").write_text(result.stdout + result.stderr)
            rows = pd.read_csv(prefix.with_suffix(".tsv"), sep="\t")
            cli_replays.append(
                dict(
                    threads=threads,
                    repeat=repeat,
                    command=command,
                    output_sha256=sha(prefix.with_suffix(".tsv")),
                    rows=clean(rows.to_dict("records")),
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, gene in enumerate(("NME7", "MRPS25")):
        for prefix, name in (("native", "RapidGLM"), ("r", "R glm.fit")):
            frame = pd.read_csv(work / gene / f"{prefix}_metrics.tsv", sep="\t")
            frame = frame.query(
                "model == 'poisson' and maxit <= 40 and iterations == maxit"
            )
            for j, field in enumerate(
                (
                    "boundary_beta",
                    "relative_deviance_change",
                    "relative_coefficient_change",
                )
            ):
                axes[i, j].plot(frame.maxit, frame[field], label=name)
                axes[i, j].set_xlabel("Iteration")
                if j:
                    axes[i, j].set_yscale("log")
                    axes[i, j].axhline(
                        1e-9, linestyle=":", color="black", label="Tolerance"
                    )
        axes[i, 0].set_ylabel(gene)
    for j, title in enumerate(
        (
            "Zero-count donor coefficient",
            "Relative deviance change",
            "Coefficient change, including boundary",
        )
    ):
        axes[0, j].set_title(title)
    axes[0, 0].legend()
    fig.suptitle(
        "Same data and starts: R and RapidGLM convergence on zero-count donors"
    )
    fig.savefig(work / "convergence.png", dpi=160)
    fig.savefig(work / "convergence.svg")
    plt.close(fig)

    record = dict(
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commands=commands,
        engine_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        production_model_sha256=sha(ROOT / "include/rapidglm/model.hpp"),
        diagnostic_sources={
            p: sha(ROOT / p)
            for p in (
                "tests/diagnose_boundary.cpp",
                "scripts/reference_boundary.R",
                "scripts/diagnose_boundary.py",
            )
        },
        executable_sha256=sha(args.binary),
        seed=args.seed,
        reference_environment=reference,
        inputs=data_hashes,
        cases=summary,
        original_cli_replays=cli_replays,
        original_cli_all_outputs_identical=len(
            {r["output_sha256"] for r in cli_replays}
        )
        == 1,
        cli_sha256=sha(args.cli),
        native_dynamic_dependencies=subprocess.check_output(
            ["ldd", str(Path(args.binary).resolve())], text=True
        ),
        environment_overrides={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        scope="Paired comparison of the current production fitter. Historical calibration gates and provisional thresholds are unchanged.",
        method="Each prefix is a fresh production API call from the identical start with a truncated iteration budget. No replacement fitter is used in the harness.",
        rng="Real data are deterministic. Synthetic data are generated once by R's recorded RNG/seed. Both engines read identical IEEE binary64 inputs, with native inputs echoed and verified bit-for-bit. Every R fit verifies unchanged RNG state; the native fitter uses no RNG.",
        oracle="Poisson paired-donor intercepts profiled analytically: group total times softmax(offset + treatment effect). The remaining treatment score is solved by Brent's method. Zero-total donors have limiting means zero, so this gives a likelihood supremum without an infinite intercept in the numerical solver.",
        limitations=[
            "R glm.nb couples its inner and outer limits through maxit; native outer limit remains 60. Exact iteration comparison isolates Poisson and fixed-theta NB fits.",
            "Native and R automatic initializations differ; shared-start runs isolate that difference.",
            "R warnings, errors and convergence flags are all retained.",
            "No package implementation was copied or inspected; reference code uses public modelling and tracing APIs.",
        ],
        sources=[
            "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.control.html",
            "https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/glm.nb.html",
        ],
        work_directory=str(work),
    )
    receipt = Path(args.receipt).resolve() if args.receipt else work / "summary.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
    print(receipt)
    for c in summary:
        print(
            c["case"],
            "R stops at",
            c["R_first_poisson_convergence"]["iterations"],
            "native:",
            c["native_at_same_budget"]["status"],
            "max beta difference",
            c["same_budget_max_coefficient_difference"],
        )


if __name__ == "__main__":
    main()
