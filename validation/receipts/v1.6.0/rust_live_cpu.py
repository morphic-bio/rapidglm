"""Rust-only live descendant CPU accounting; original guard and threshold stay fixed.

The libc clock_getcpuclockid(pid, clockid_t*) API returns a process CPU clock.
time.clock_gettime_ns(clockid) reads its total CPU time across process threads.
Only verified live descendants with every thread confined to the selected CPUs
are credited. Two observations around the original final snapshot bound credit
conservatively; exited/reaped children stay in the original getrusage channel.
No external implementation or program example was consulted.
"""
import ctypes
import errno
import json
import os
import time
from pathlib import Path

_libc = ctypes.CDLL(None)
_clock_id = _libc.clock_getcpuclockid
_clock_id.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_clock_id.restype = ctypes.c_int


def identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=int(pid), ppid=int(fields[1]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants():
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            row = identity(int(path.name))
            if row is not None:
                rows[row["pid"]] = row
    owned = {os.getpid()}
    while True:
        new = {pid for pid, row in rows.items() if row["ppid"] in owned} - owned
        if not new:
            break
        owned.update(new)
    return [rows[pid] for pid in sorted(owned - {os.getpid()})]


def process_sample(row):
    pid = row["pid"]
    clock = ctypes.c_int()
    code = _clock_id(pid, ctypes.byref(clock))
    if code in (errno.ESRCH, errno.EINVAL):
        return None
    if code:
        raise OSError(code, os.strerror(code))
    try:
        tasks = sorted(int(p.name) for p in Path(f"/proc/{pid}/task").iterdir())
        affinity = {str(tid): sorted(os.sched_getaffinity(tid)) for tid in tasks}
        cpu_ns = time.clock_gettime_ns(clock.value)
        if identity(pid) != row:
            return None
        if tasks != sorted(int(p.name) for p in Path(f"/proc/{pid}/task").iterdir()):
            return None
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ESRCH):
            return None
        raise
    return dict(row, cpu_ns=cpu_ns, thread_affinities=affinity)


def sample():
    return [value for row in descendants() if (value := process_sample(row)) is not None]


def off_core_sample(identities_only=False):
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, set(range(6)))
    start = time.thread_time_ns()
    try:
        rows = descendants() if identities_only else sample()
    finally:
        seconds = (time.thread_time_ns() - start) / 1e9
        os.sched_setaffinity(0, affinity)
    return rows, seconds


def install():
    import measurement_guard_v3 as guard
    assert not getattr(guard, "_rust_live_installed", False)
    guard._rust_live_installed = True
    original_begin, original_snapshot = guard.begin, guard.precise_snapshot
    state = None

    def begin(cpus, label):
        nonlocal state
        state = None
        if label.endswith(" PyDESeq2"):
            assert os.environ.get("RAPIDGLM_RUST_ATTEMPT_ID"), "fresh interpreter token required"
            state = dict(label=label, started=False)
        return original_begin(cpus, label)

    def precise(cpus):
        if state is None:
            return original_snapshot(cpus)
        if not state["started"]:
            before, _ = off_core_sample(identities_only=True)
            assert not before, "PyDESeq2 must begin without existing worker processes"
            result = original_snapshot(cpus)
            state.update(started=True, start_descendants=before, process_identity=identity(os.getpid()))
            result["rust_live_worker_start"] = dict(descendants=before, attempt_id=os.environ["RAPIDGLM_RUST_ATTEMPT_ID"])
            return result
        before, sampler_cpu = off_core_sample()
        result = original_snapshot(cpus)
        # This sampler ran inside the parent CPU interval, but on CPUs 0-5.
        result["off_cpu_before_usage_seconds"] += sampler_cpu
        after, _ = off_core_sample()
        later = {(x["pid"], x["start_ticks"], x["ppid"]): x for x in after}
        credited, excluded = [], []
        selected = set(cpus)
        for row in before:
            end = later.get((row["pid"], row["start_ticks"], row["ppid"]))
            confined = end is not None and all(set(a) <= selected for item in (row, end) for a in item["thread_affinities"].values())
            if not confined:
                excluded.append(dict(row, reason="not alive at both observations or not confined to selected CPUs"))
                continue
            assert end["cpu_ns"] >= row["cpu_ns"]
            credited.append(dict(pid=row["pid"], ppid=row["ppid"], start_ticks=row["start_ticks"], cpu_ns=min(row["cpu_ns"], end["cpu_ns"])))
        external = dict(provider="rust-live-descendants-v1", cpu_seconds=sum(x["cpu_ns"] for x in credited)/1e9,
                        attempt_id=os.environ["RAPIDGLM_RUST_ATTEMPT_ID"], process_identity=state["process_identity"],
                        start_descendants=state["start_descendants"], before=before, after=after,
                        credited=credited, excluded=excluded, selected_cpus=sorted(cpus),
                        sampler_affinity=list(range(6)), parent_off_core_sampler_cpu_seconds=sampler_cpu,
                        method="Fresh-process baseline is zero. Credit the smaller lifetime process CPU clock for descendants alive with identical PID/start/parent at both final observations and with every thread confined to the selected CPUs. Reaped children remain solely in getrusage; transient or unconfined processes receive no live credit.")
        path = Path(os.environ["RAPIDGLM_EXTERNAL_CPU_AUDIT"])
        path.write_text(json.dumps(external, indent=2) + "\n")
        result["rust_live_worker_end"] = dict(provider=external["provider"], live_cpu_seconds=external["cpu_seconds"])
        return result

    guard.begin, guard.precise_snapshot = begin, precise
