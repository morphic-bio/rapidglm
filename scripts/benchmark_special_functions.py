#!/usr/bin/env python3
"""Special-function benchmark stage: accuracy and speed of RapidGLM's special
functions (include/rapidglm/vmath.hpp: RapidGLM's log, log1p, exp, lgamma, digamma and
trigamma in the selected division mode) against glibc's libm and boost::math (C++),
NumPy and SciPy (Python) and base R, single-threaded.

Stages (each timed one under the shared lock after the quiet-host check,
pinned to one physical core, --cpu):
  cpp     benchmarks/special_functions.cpp: the largest error in ulps against
          long double references over standard ranges and the NB likelihood
          arguments y + 1/alpha (alpha from 1e-8 to 1e3), ns per element for
          glibc, boost::math (default policy and without promotion to long
          double) and RapidGLM's arrays (AVX2, portable) and one-value form, and
          ns per call of digamma and trigamma in the count paths' patterns;
  python  numpy.log/log1p/exp, scipy.special.gammaln/digamma/polygamma(1, x)
          against rapidglm.special at 1e3, 1e5 and 1e7 elements;
  r       base R's log/log1p/exp/lgamma/digamma/trigamma against
          rapidglm::special_* at the same sizes.

    python3 scripts/benchmark_special_functions.py --build build \
        --r-library LIB --output validation/SPECIAL_FUNCTIONS.json
"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZES = (1000, 100000, 10000000)
PYTHON = r"""
import json, sys, time
import numpy as np, scipy, scipy.special as sc
sys.path.insert(0, sys.argv[1])
from rapidglm import special
sizes = [int(s) for s in sys.argv[2].split(",")]
rng = np.random.default_rng(1)
pairs = {"log": (np.log, special.log), "log1p": (np.log1p, special.log1p), "exp": (np.exp, special.exp),
         "lgamma": (sc.gammaln, special.lgamma), "digamma": (sc.digamma, special.digamma),
         "trigamma": (lambda x: sc.polygamma(1, x), special.trigamma)}
rows = []
for n in sizes:
    # NB likelihood arguments y + 1/alpha (exp: uniform on (-50, 50)).
    x = rng.geometric(1 / 21, n) - 1 + 1 / 10.0 ** rng.uniform(-8, 3, n)
    e = rng.uniform(-50, 50, n)
    for name, (theirs, ours) in pairs.items():
        arg = e if name == "exp" else x
        reps = max(3, min(200, int(2e7 // n)))
        for label, f in (("numpy_scipy", theirs), ("rapidglm", ours)):
            f(arg[:16])
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                f(arg)
                best = min(best, time.perf_counter() - t0)
            rows.append(dict(function=name, n=n, implementation=label, ns_per_element=1e9 * best / n, repeats=reps))
print(json.dumps(dict(rows=rows, versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                                                python=sys.version.split()[0], path=special.path()))))
"""
R = r"""
args <- setdiff(commandArgs(TRUE), "--args")
.libPaths(c(strsplit(args[1], ":")[[1]], .libPaths()))
suppressPackageStartupMessages(library(rapidglm))
sizes <- as.numeric(strsplit(args[2], ",")[[1]])
set.seed(1)
pairs <- list(log = list(log, special_log), log1p = list(log1p, special_log1p), exp = list(exp, special_exp),
              lgamma = list(lgamma, special_lgamma), digamma = list(digamma, special_digamma),
              trigamma = list(trigamma, special_trigamma))
rows <- list()
for (n in sizes) {
  x <- rgeom(n, 1 / 21) + 1 / 10^runif(n, -8, 3)
  e <- runif(n, -50, 50)
  for (name in names(pairs)) {
    arg <- if (name == "exp") e else x
    reps <- max(3, min(200, floor(2e7 / n)))
    for (k in 1:2) {
      f <- pairs[[name]][[k]]
      invisible(f(arg[1:16]))
      best <- Inf
      for (r in seq_len(reps)) { t0 <- proc.time()[["elapsed"]]; invisible(f(arg)); best <- min(best, proc.time()[["elapsed"]] - t0) }
      # elapsed has millisecond resolution: time a batch of calls at small sizes
      if (best < 0.01) {
        batch <- ceiling(0.05 / max(best, 1e-4))
        t0 <- proc.time()[["elapsed"]]; for (b in seq_len(batch)) invisible(f(arg)); best <- (proc.time()[["elapsed"]] - t0) / batch
      }
      rows[[length(rows) + 1]] <- list(`function` = name, n = n, implementation = c("base_r", "rapidglm")[k],
                                       ns_per_element = 1e9 * best / n, repeats = reps)
    }
  }
}
cat(jsonlite::toJSON(list(rows = rows, versions = list(r = paste(R.version$major, R.version$minor, sep = "."),
                                                        rapidglm = as.character(packageVersion("rapidglm")),
                                                        path = special_path())), auto_unbox = TRUE, digits = NA))
"""


def timed(cpu, command, env):
    """A command pinned to `cpu` under the shared lock after the quiet check."""
    wrapped = [sys.executable, str(ROOT / "scripts/quiet_run.py"), "--cpus", cpu, "--",
               "taskset", "-c", cpu, *command]
    started = time.time()
    out = subprocess.run(wrapped, env=env, check=True, capture_output=True, text=True)
    return out.stdout, round(time.time() - started, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=pathlib.Path, default=ROOT / "build",
                    help="CMake build with RAPIDGLM_BENCHMARKS=ON (rapidglm_special_functions, librapidglm_c)")
    ap.add_argument("--r-library", default="", help="R library (with jsonlite) holding this rapidglm package")
    ap.add_argument("--cpu", default="16")
    ap.add_argument("--untimed-cpus", default="0-7,12-15", help="CPUs for the untimed accuracy stage")
    ap.add_argument("--accuracy-n", type=int, default=1000000)
    ap.add_argument("--speed-seconds", type=float, default=120)
    ap.add_argument("--stages", nargs="+", default=["cpp", "python", "r"])
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--merge", action="store_true", help="keep the other stages of an existing --output")
    args = ap.parse_args()
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "RAPIDGLM_C_LIBRARY": str((args.build / "librapidglm_c.so").resolve())}
    harness = str((args.build / "rapidglm_special_functions").resolve())
    receipt = dict(purpose=__doc__.strip().splitlines()[0], cpu=args.cpu,
                   commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(),
                   worktree_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                                      capture_output=True, text=True).stdout.strip()),
                   host=dict(machine=platform.machine(), processor=next(
                       (l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")),
                   sizes=list(SIZES))
    if args.merge and args.output.exists():
        receipt = {**json.loads(args.output.read_text()), **{k: v for k, v in receipt.items()
                                                              if k in ("commit", "worktree_dirty")}}

    receipt["division"] = os.environ.get("RAPIDGLM_DIVISION", "slow")

    def save():  # after every stage, so that a later failure keeps the earlier ones
        receipt["load_average"] = os.getloadavg()
        args.output.write_text(json.dumps(receipt, indent=1) + "\n")

    if "cpp" in args.stages:
        accuracy = subprocess.run(["taskset", "-c", args.untimed_cpus, "nice", "-n", "10", harness,
                                   "accuracy", str(args.accuracy_n)], env=env, check=True,
                                  capture_output=True, text=True).stdout  # untimed
        receipt["cpp_accuracy"] = json.loads(accuracy)["accuracy"]
        speed, wall = timed(args.cpu, [harness, "speed", str(args.speed_seconds)], env)
        receipt["cpp_speed"] = json.loads(speed)
        receipt["cpp_speed"]["wall_seconds"] = wall
        save()
    if "python" in args.stages:
        out, wall = timed(args.cpu, [sys.executable, "-c", PYTHON, str(ROOT / "bindings/python"),
                                     ",".join(map(str, SIZES))], env)
        receipt["python"] = json.loads(out)
        receipt["python"]["wall_seconds"] = wall
        save()
    if "r" in args.stages:
        out, wall = timed(args.cpu, ["Rscript", "-e", R, "--args", args.r_library, ",".join(map(str, SIZES))], env)
        receipt["r"] = json.loads(out)
        receipt["r"]["wall_seconds"] = wall
        save()


if __name__ == "__main__":
    main()
