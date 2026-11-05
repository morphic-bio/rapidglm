#!/usr/bin/env python3
"""Run a timed command under the shared lock after quiet host and CPU samples.

The lock path is RAPIDGLM_TIMING_LOCK (default: pikachu's shared lock). With
RAPIDGLM_QUIET_SIBLINGS=1 the per-CPU idle check also covers the SMT siblings of
the selected CPUs (from sysfs), which must be idle for one thread per core.
"""
import argparse
import importlib
import fcntl
import json
import os
import subprocess
import sys
import time

LOCK = os.environ.get('RAPIDGLM_TIMING_LOCK', '/mnt/pikachu/e2e_bench_20260926/pikachu_timed.lock')


def cpu_set(value):
    if not isinstance(value, str):
        return set(value)
    result = set()
    for part in value.split(','):
        limits = [int(v) for v in part.split('-')]
        result.update(range(limits[0], limits[-1] + 1))
    return result


def thread_siblings(cpus):
    """The SMT siblings of the selected CPUs (sysfs thread_siblings_list), not the CPUs themselves."""
    result = set()
    for cpu in cpus:
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list') as source:
                result |= cpu_set(source.read().strip())
        except OSError:
            pass
    return result - set(cpus)


def cpu_snapshot(cpus):
    result = {}
    with open('/proc/stat') as source:
        for line in source:
            name, *values = line.split()
            if name.startswith('cpu') and name[3:].isdigit() and int(name[3:]) in cpus:
                values = [int(v) for v in values[:8]]
                total = sum(values)
                result[int(name[3:])] = (total, total - values[3] - values[4])
    return result


def audit(record):
    """Append release timing evidence when a caller supplies an audit destination."""
    destination = os.environ.get('RAPIDGLM_TIMING_AUDIT')
    if destination:
        record = dict(record, stage=os.environ.get('RAPIDGLM_STAGE'), pid=os.getpid())
        with open(destination, 'a') as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(record) + '\n')


def finish_measurement():
    """Close an optional release contention audit after a timed command."""
    module = os.environ.get('RAPIDGLM_MEASUREMENT_GUARD')
    return importlib.import_module(module).finish() if module else None


def wait_quiet(cpus=None, label=None):
    if os.environ.get('RAPIDGLM_MIN_AVAILABLE_GIB'):
        wait_memory(float(os.environ['RAPIDGLM_MIN_AVAILABLE_GIB']))
    threshold = float(os.environ.get('RAPIDGLM_QUIET_LOAD', '4'))
    cpus = cpu_set(cpus) if cpus is not None else set()
    siblings = thread_siblings(cpus) if os.environ.get('RAPIDGLM_QUIET_SIBLINGS') == '1' else set()
    checked = cpus | siblings
    quiet = 0
    start = time.monotonic()
    samples = []
    while quiet < 2:
        load_before = os.getloadavg()[0]
        before = cpu_snapshot(checked) if checked else None
        time.sleep(10)
        load_after = os.getloadavg()[0]
        idle = True
        fractions = {}
        if checked:
            after = cpu_snapshot(checked)
            for cpu in sorted(checked):
                if cpu not in before or cpu not in after:
                    idle = False
                    break
                elapsed = after[cpu][0] - before[cpu][0]
                used = after[cpu][1] - before[cpu][1]
                fractions[str(cpu)] = used / elapsed if elapsed > 0 else None
                if not (elapsed > 0 and used / elapsed < .2):
                    idle = False
                    break
        quiet = quiet + 1 if max(load_before, load_after) <= threshold and idle else 0
        samples.append(dict(load_before=load_before, load_after=load_after,
                            busy_fractions=fractions, passed=quiet > 0))
        samples = samples[-2:]
    waited = round(time.monotonic() - start, 2)
    record = dict(event='quiet_ready', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  label=label or ' '.join(sys.argv), cpus=sorted(cpus),
                  waited_seconds=waited, load_one_minute=load_after, samples=samples)
    if siblings:
        record['sibling_cpus'] = sorted(siblings)
    audit(record)
    module = os.environ.get("RAPIDGLM_MEASUREMENT_GUARD")
    if module:
        importlib.import_module(module).begin(cpus, record["label"])
    return waited


def wait_memory(minimum_gib):
    """Wait for release full-screen memory headroom and absence of other large jobs."""
    own = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        own.add(parent)
        try:
            with open(f'/proc/{parent}/status') as source:
                parent = int(next(line.split()[1] for line in source if line.startswith('PPid:')))
        except (FileNotFoundError, StopIteration):
            break
    while True:
        with open('/proc/meminfo') as source:
            available = int(next(line.split()[1] for line in source if line.startswith('MemAvailable:'))) / 2**20
        heavy = []
        for name in os.listdir('/proc'):
            if not name.isdigit() or int(name) in own:
                continue
            try:
                with open(f'/proc/{name}/status') as source:
                    rss = int(next(line.split()[1] for line in source if line.startswith('VmRSS:'))) / 2**20
                if rss >= 8:
                    heavy.append(dict(pid=int(name), rss_gib=rss))
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                pass
        if available >= minimum_gib and not heavy:
            audit(dict(event='memory_ready', available_gib=available, other_heavy_jobs=heavy))
            return
        print(json.dumps(dict(memory_wait=True, available_gib=available, other_heavy_jobs=heavy)),
              file=sys.stderr, flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-only', action='store_true')
    parser.add_argument('--cpus')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    cpus = args.cpus
    if cpus is None and len(command) >= 3 and command[:2] == ['taskset', '-c']:
        cpus = command[2]

    def ready():
        waited = wait_quiet(cpus)
        print(json.dumps({'timed_command_ready_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'load_average': os.getloadavg(), 'cpus': cpus,
                          'waited_for_quiet_seconds': waited}), file=sys.stderr, flush=True)

    if args.wait_only:
        os.environ.pop("RAPIDGLM_MEASUREMENT_GUARD", None)
        ready()
        return
    if not command:
        parser.error('a command is required')
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready()
        code = subprocess.call(command)
        finish_measurement()
        raise SystemExit(code)


if __name__ == '__main__':
    main()
