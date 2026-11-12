#!/usr/bin/env python3
"""Time the steps that made CRT null draws faster, on the library as built.

The two executables of benchmarks/crt_steps.cpp are run on one panel in five
configurations, each of which changes one thing from the one before it:

    first     portable build, mode first     the draws as first implemented
    single    portable build, mode single    the released draw code, one draw at a time
    batched   portable build, mode batched   draws in batches, scored in blocks of cells
    pipeline  portable build, mode pipeline  the pair pipeline, one sampler per target
    vector    native build,   mode pipeline  Philox blocks eight at a time with AVX2

and, with --hazard, a sixth: `hazard`, the native pipeline with the default
sampler of `rapidglm sceptre` (crt_draw hazard), which draws different random
numbers and so is compared with nothing but sceptre() itself.

Only the null draws are timed (seconds.draws of each run's record); the fits and
the other preparation are the same in every configuration and are reported
beside it. Runs of the same build must write the same table byte for byte. The
two builds may round their fits differently, so between them every pair's
statistic and p-value must agree to --tolerance, and pairs that ended at
another stage are counted. One run of each build, in the first pass, also
compares its table with sceptre().

    python3 scripts/benchmark_crt_steps.py --config panel.json --threads 16

panel.json holds the inputs and settings in the keys of `rapidglm sceptre`
(docs/CLI.md); threads, output, crt_draw and verify are set here. With --cpus
every command runs on those processors, and the processor time that other
processes used on them and on the other hardware threads of their cores is
recorded beside its timings. See docs/CRT_STEPS.md.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

# name: (executable, mode, crt_draw), in the order of the steps.
CONFIGURATIONS = {
    "first": ("rapidglm_crt_steps_portable", "first", "thinned"),
    "single": ("rapidglm_crt_steps_portable", "single", "thinned"),
    "batched": ("rapidglm_crt_steps_portable", "batched", "thinned"),
    "pipeline": ("rapidglm_crt_steps_portable", "pipeline", "thinned"),
    "vector": ("rapidglm_crt_steps", "pipeline", "thinned"),
    "hazard": ("rapidglm_crt_steps", "pipeline", "hazard"),
}
# Ratios of median draw seconds reported when both configurations were run.
GAINS = [
    ("first", "single"),
    ("single", "batched"),
    ("first", "batched"),
    ("batched", "pipeline"),
    ("pipeline", "vector"),
    ("first", "vector"),
    ("vector", "hazard"),
]
# Columns that must be equal between the two builds; the rest are numbers.
EXACT = ["response", "target", "guide", "kind", "cells", "nonzero_treatment", "nonzero_control"]
STAGE = ["status", "draws", "approximation"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative(a, b):
    if a == b:  # also NA with NA
        return 0.0
    if "NA" in (a, b):
        return math.inf
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b))


def compare_builds(reference, other, tolerance):
    """Differences between the tables of the two builds."""
    a, b = rows(reference), rows(other)
    if len(a) != len(b):
        raise RuntimeError(f"the builds tested different pairs: {reference} {other}")
    staged = 0
    worst = {"statistic": 0.0, "p_value": 0.0}
    for index, (x, y) in enumerate(zip(a, b)):
        if any(x[column] != y[column] for column in EXACT):
            raise RuntimeError(f"pair {index} differs between the builds: {reference} {other}")
        if any(x[column] != y[column] for column in STAGE):
            staged += 1
            continue
        for column in worst:
            difference = relative(x[column], y[column])
            if difference > tolerance:
                raise RuntimeError(
                    f"pair {index}: {column} {x[column]} and {y[column]} differ between the "
                    f"builds by more than {tolerance}: {reference} {other}"
                )
            worst[column] = max(worst[column], difference)
    return {
        "pairs": len(a),
        "pairs_ending_at_another_stage": staged,
        "max_relative_difference_statistic": worst["statistic"],
        "max_relative_difference_p_value": worst["p_value"],
    }


def cpu_list(text):
    """The processors of a list such as 6-21 or 0,2,4-7."""
    cpus = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def siblings(cpus):
    """The other hardware threads of the cores of `cpus`."""
    others = set()
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            others |= cpu_list(path.read_text().strip())
        except (OSError, ValueError):
            pass
    return others - cpus


def busy_seconds(cpus):
    """Seconds since boot that the processors `cpus` were not idle, summed."""
    ticks = 0
    for line in Path("/proc/stat").read_text().splitlines():
        name, *fields = line.split()
        if name[:3] == "cpu" and name[3:].isdigit() and int(name[3:]) in cpus:
            # user nice system idle iowait irq softirq steal
            values = [int(field) for field in fields[:8]]
            ticks += sum(values) - values[3] - values[4]
    return ticks / os.sysconf("SC_CLK_TCK")


def child_seconds():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def host():
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"node": platform.node(), "cpu": model, "logical_cpus": os.cpu_count()}


def commit():
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return head + ("+changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def benchmark(args):
    base = json.loads(Path(args.config).read_text())
    for key in ("threads", "output", "crt_draw", "verify"):
        if key in base:
            raise SystemExit(f"{key} is set by this script: remove it from {args.config}")
    names = [name for name in CONFIGURATIONS if name in args.configurations]
    if args.hazard and "hazard" not in names:
        names.append("hazard")
    if not names:
        raise SystemExit("no configuration to run")
    build = Path(args.build)
    executables = {}
    for name in names:
        path = build / CONFIGURATIONS[name][0]
        if not path.is_file():
            raise SystemExit(f"{path} is missing: build with -DRAPIDGLM_BENCHMARKS=ON")
        executables[str(path)] = sha256(path)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    runs_path = work / "runs.json"
    records = json.loads(runs_path.read_text()) if args.resume and runs_path.exists() else []
    done = {(record["configuration"], record["pass"]) for record in records}
    prefix = []
    if args.cpus:
        prefix = ["taskset", "-c", args.cpus]
        if args.quiet:
            prefix = [sys.executable, str(ROOT / "scripts/quiet_run.py"), "--cpus", args.cpus,
                      "--", *prefix]
    elif args.quiet:
        raise SystemExit("--quiet needs --cpus")
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    # With --cpus on Linux, what else ran on the timed processors and on the
    # other hardware threads of their cores is recorded for every command.
    timed_cpus = cpu_list(args.cpus) if args.cpus and Path("/proc/stat").exists() else set()
    sibling_cpus = siblings(timed_cpus)

    def activity():
        return busy_seconds(timed_cpus), busy_seconds(sibling_cpus), child_seconds()

    # The table every later run of the same build and sampler must reproduce,
    # and whether sceptre() has been compared with that build and sampler.
    reference, verified = {}, set()
    for record in records:
        group = (record["executable"], record["crt_draw"])
        reference.setdefault(group, (record["output_sha256"], record["table"]))
        if record["verified_against_sceptre"]:
            verified.add(group)
    cross = {}

    def compare():
        # The portable and native tables of the same sampler, once both exist.
        for crt_draw in {draw for _, draw in reference} - set(cross):
            portable = reference.get(("rapidglm_crt_steps_portable", crt_draw))
            native = reference.get(("rapidglm_crt_steps", crt_draw))
            if portable and native:
                cross[crt_draw] = compare_builds(portable[1], native[1], args.tolerance)

    # Pass 0 is the warm-up: it loads the inputs into the page cache and is left
    # out of the medians. The first pass made runs the fastest configuration
    # first, so that the comparisons are made early; later passes alternate the
    # order, so that no configuration always follows the same one.
    passes = range(0 if args.warmup else 1, args.repetitions + 1)
    for position, number in enumerate(passes):
        for name in names[::-1] if position % 2 == 0 else names:
            if (name, number) in done:
                continue
            executable, mode, crt_draw = CONFIGURATIONS[name]
            group = (executable, crt_draw)
            directory = work / name
            directory.mkdir(exist_ok=True)
            output = directory / f"pass{number}"
            verify = args.verify and group not in verified
            config = dict(base, threads=args.threads, crt_draw=crt_draw, verify=verify,
                          output=str(output))
            config_path = directory / f"pass{number}.config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            command = [*prefix, str(build / executable), mode, str(config_path)]
            load = os.getloadavg()
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            before = activity() if timed_cpus else None
            start = time.perf_counter()
            subprocess.run(command, check=True, env=environment)
            wall = time.perf_counter() - start
            others = None
            if timed_cpus:
                busy, sibling, own = (b - a for a, b in zip(before, activity()))
                # Processor time on the timed processors that was not the
                # command's (a difference of two tick counts, so only roughly),
                # and all processor time on their cores' other hardware threads.
                others = {"busy_cpu_seconds": busy, "command_cpu_seconds": own,
                          "other_cpu_seconds": busy - own, "sibling_cpu_seconds": sibling}
            run = json.loads(Path(f"{output}.run.json").read_text())
            # The step between the two executables is the AVX2 generator: the
            # native one must have used it and the portable one must not.
            if run["philox_avx2"] != (executable == "rapidglm_crt_steps"):
                raise RuntimeError(
                    f"{executable} ran with philox_avx2 {run['philox_avx2']}: the two builds "
                    "differ only on an x86-64 host with AVX2, with a library configured "
                    "without RAPIDGLM_FORCE_PORTABLE")
            table = Path(f"{output}.tsv")
            digest = sha256(table)
            if group not in reference:
                reference[group] = (digest, str(table))
            elif digest != reference[group][0]:
                raise RuntimeError(f"scientific output changed: {table} {reference[group][1]}")
            else:
                table.unlink()  # the same bytes as the reference table
            verified |= {group} if run["verified_against_sceptre"] else set()
            records.append(
                {
                    "configuration": name,
                    "pass": number,
                    "warmup": number == 0,
                    "executable": executable,
                    "mode": mode,
                    "crt_draw": crt_draw,
                    "threads": args.threads,
                    "command": command,
                    "started_utc": started,
                    "load_before": load,
                    "wall_seconds": wall,
                    "others": others,
                    "seconds": run["seconds"],
                    "null_draws": run["null_draws"],
                    "pairs": run["pairs"],
                    "pairs_tested": run["pairs_tested"],
                    "philox_avx2": run["philox_avx2"],
                    "verified_against_sceptre": run["verified_against_sceptre"],
                    "output_sha256": digest,
                    "table": reference[group][1],
                }
            )
            runs_path.write_text(json.dumps(records, indent=2) + "\n")
            print(f"pass {number} {name}: {run['seconds']['draws']:.2f} s of draws", flush=True)
            compare()
    compare()

    summary = {
        "benchmark": "crt_steps",
        "rapidglm_commit": commit(),
        "host": host(),
        "threads": args.threads,
        "cpus": args.cpus,
        "quiet_run": args.quiet,
        "config": base,
        "executables_sha256": executables,
        "timed_passes": args.repetitions,
        "configurations": {},
        "gains": {},
        "builds_compared": cross,
    }
    medians = {}
    for name in names:
        timed = [r for r in records if r["configuration"] == name and not r["warmup"]]
        if not timed:
            continue
        draws = [r["seconds"]["draws"] for r in timed]
        medians[name] = statistics.median(draws)
        summary["configurations"][name] = {
            "executable": CONFIGURATIONS[name][0],
            "mode": CONFIGURATIONS[name][1],
            "crt_draw": CONFIGURATIONS[name][2],
            "draw_seconds": draws,
            "median_draw_seconds": medians[name],
            "median_prepare_seconds": statistics.median(r["seconds"]["prepare"] for r in timed),
            "wall_seconds": [r["wall_seconds"] for r in timed],
            "other_cpu_seconds": [(r.get("others") or {}).get("other_cpu_seconds") for r in timed],
            "sibling_cpu_seconds": [
                (r.get("others") or {}).get("sibling_cpu_seconds") for r in timed
            ],
            "null_draws": timed[0]["null_draws"],
            "pairs": timed[0]["pairs"],
            "pairs_tested": timed[0]["pairs_tested"],
            "microseconds_per_draw_per_thread": (
                1e6 * medians[name] * args.threads / timed[0]["null_draws"]
                if timed[0]["null_draws"] else None
            ),
            "output_sha256": timed[0]["output_sha256"],
        }
    for slower, faster in GAINS:
        if slower in medians and faster in medians:
            summary["gains"][f"{slower}_to_{faster}"] = medians[slower] / medians[faster]
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="inputs and settings, in the keys of `rapidglm sceptre`")
    parser.add_argument("--build", default=ROOT / "build",
                        help="directory holding rapidglm_crt_steps and rapidglm_crt_steps_portable")
    parser.add_argument("--work", default=ROOT / "validation/work/crt-steps")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3, help="timed passes")
    parser.add_argument("--configurations", nargs="+",
                        default=["first", "single", "batched", "pipeline", "vector"],
                        choices=list(CONFIGURATIONS))
    parser.add_argument("--hazard", action="store_true",
                        help="also time the default sampler (crt_draw hazard) in the native build")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="no untimed pass first (for panels whose first mode takes hours)")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="do not compare a run of each build with sceptre()")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="largest relative difference of a statistic or p-value between "
                             "the two builds")
    parser.add_argument("--cpus", help="run under taskset -c on these processors")
    parser.add_argument("--quiet", action="store_true",
                        help="run each command through scripts/quiet_run.py, which waits for a "
                             "quiet host and holds the timing lock named by RAPIDGLM_TIMING_LOCK")
    parser.add_argument("--resume", action="store_true",
                        help="keep the runs recorded in the work directory and make the rest")
    print(json.dumps(benchmark(parser.parse_args()), indent=2))
