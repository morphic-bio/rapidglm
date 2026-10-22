#!/usr/bin/env python3
"""cpu_guard.py wait CPUS [TIMEOUT]: returns once other processes leave the CPUs idle (under 20% busy over
3 s) or after TIMEOUT seconds (default 1200); prints the seconds waited.
cpu_guard.py busy CPUS: prints the seconds those CPUs have spent busy since boot (all but idle and I/O wait)."""
import os, sys, time


def cpu_set(text):
    cpus = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def busy_seconds(cpus):
    jiffies = 0
    with open("/proc/stat") as f:
        for line in f:
            name, *values = line.split()
            if name[:3] == "cpu" and name[3:].isdigit() and int(name[3:]) in cpus:
                values = [int(v) for v in values]
                jiffies += sum(values[:3]) + sum(values[5:8])
    return jiffies / os.sysconf("SC_CLK_TCK")


mode, cpus = sys.argv[1], cpu_set(sys.argv[2])
if mode == "busy":
    print(f"{busy_seconds(cpus):.2f}")
else:
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 1200
    start = time.time()
    while True:
        before = busy_seconds(cpus)
        time.sleep(3)
        if (busy_seconds(cpus) - before) / (3 * len(cpus)) < 0.2 or time.time() - start > timeout:
            break
    print(f"{time.time() - start - 3:.0f}")
