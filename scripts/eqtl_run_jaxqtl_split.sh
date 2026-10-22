#!/usr/bin/env bash
# Run one jaxQTL configuration as 4 processes (genes split round-robin), each pinned to 3 logical
# CPUs inside 0-7,12-15, then merge the parquet outputs. usage: run_jaxqtl_split.sh NAME [jaxqtl args]
set -euo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
name=$1; shift
CPUSETS=(0-2 3-5 6,7,12 13-15)
start=$(date +%s.%N)
pids=()
for k in 0 1 2 3; do
  CPUS=${CPUSETS[$k]} bash $(dirname "$0")/eqtl_run_jaxqtl.sh $name/part$k --gene-list $W/data/genelists/chr22.part$k.txt "$@" > /dev/null &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
end=$(date +%s.%N)
cd $W/runs/jaxqtl/$name
echo "wall_seconds $(echo "$end - $start" | bc)" > wall.txt
echo "exit $rc" >> wall.txt
$W/venv/bin/python -I - "$PWD" <<'PY'
import sys, glob, os, re, polars as pl
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "part*", "chr22.*.parquet.gz")))
if files:
    out = os.path.join(d, os.path.basename(files[0]))
    pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").write_parquet(out)
    print("merged", len(files), "->", out)
tot_u = tot_s = 0.0
for t in glob.glob(os.path.join(d, "part*", "time.txt")):
    s = open(t).read()
    tot_u += float(re.search(r"User time \(seconds\): ([\d.]+)", s).group(1))
    tot_s += float(re.search(r"System time \(seconds\): ([\d.]+)", s).group(1))
with open(os.path.join(d, "cpu.txt"), "w") as fh:
    fh.write(f"user_seconds {tot_u:.1f}\nsystem_seconds {tot_s:.1f}\n")
PY
cat wall.txt cpu.txt
