#!/usr/bin/env python3
"""Run one release measurement with a lock, quiet samples and retained evidence."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from quiet_run import LOCK, cpu_set, wait_quiet, audit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpus', required=True)
parser.add_argument('--directory', required=True, type=Path)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('a command is required')
selected = cpu_set(args.cpus)
if not selected <= set(range(16, 32)):
    parser.error('release timings must use efficiency CPUs 16-31')
d = args.directory
if (d / 'host.json').exists():
    parser.error(f'existing measurement must be archived before a rerun: {d}')
d.mkdir(parents=True, exist_ok=True)

def busy():
    return float(subprocess.check_output([sys.executable, str(Path(__file__).with_name('cpu_guard.py')),
                                          'busy', args.cpus], text=True))

with open(LOCK, 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    waited = float(subprocess.check_output([sys.executable, str(Path(__file__).with_name('cpu_guard.py')),
                                           'wait', args.cpus], text=True))
    quiet = wait_quiet(args.cpus, label=' '.join(command))
    record = dict(command=command, cpus=args.cpus, load_before=os.getloadavg(),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  waited_for_idle_seconds=waited, waited_for_quiet_seconds=quiet,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    before, started = busy(), time.monotonic()
    with (d/'stdout.log').open('w') as stdout, (d/'stderr.log').open('w') as stderr:
        result = subprocess.run(['/usr/bin/time', '-v', '-o', str(d/'time.log'),
                                 'taskset', '-c', args.cpus, *command], stdout=stdout, stderr=stderr)
    record.update(exit=result.returncode, cpus_busy_seconds=busy()-before,
                  outer_wall_seconds=time.monotonic()-started, load_after=os.getloadavg())
    text = (d/'time.log').read_text()
    own = sum(float(re.search(rf'{kind} time \(seconds\):\s*(\S+)', text)[1])
              for kind in ('User', 'System'))
    record['command_cpu_seconds'] = own
    record['others_cpu'] = max(0., record['cpus_busy_seconds']-own)/(record['outer_wall_seconds']*len(selected))
    record['contended'] = record['others_cpu'] > .05
    (d/'host.json').write_text(json.dumps(record, indent=2)+'\n')
    audit(dict(event='command_finished', directory=str(d), **record))
raise SystemExit(result.returncode)
