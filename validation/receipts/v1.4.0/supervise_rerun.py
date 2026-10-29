#!/usr/bin/env python3
"""Append a guarded eQTL-stage rerun after a completed, contended attempt."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from cpu_policy import check, digest, emit, read_policy, utc
from supervise import stop_child

R = Path(__file__).resolve().parent
ROOT = R.parents[2]


def main():
    source = json.loads((R / 'source.json').read_text())
    expected = source['cpu_policy']['expected']
    assert read_policy() == expected, 'CPU policy differs; stop without changing it'
    prior = json.loads((R / 'run-complete.json').read_text())
    assert prior['exit'] == 0 and prior['cpu_policy_unchanged']
    assert not (R / 'CPU_POLICY_VIOLATION.json').exists()
    for name, value in {**source['cpu_policy']['controller_files_sha256'],
                        **source['cpu_policy']['supplemental_guard_files_sha256']}.items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == value, name
    subprocess.run(['python3', str(R / 'verify_source.py')], cwd=ROOT, check=True)
    audit = R / 'timing-audit.jsonl'
    position = audit.stat().st_size
    started = utc()
    emit('started', commit=source['commit'], policy=expected, policy_sha256=digest(expected),
         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         poll_interval_seconds=1, maximum_allowed_poll_gap_seconds=5,
         scope='eqtl stage rerun after contention; earlier 17 stages retained')
    (R / 'run-complete.json').write_text(json.dumps(dict(exit=None, started_utc=prior['started_utc'],
        status='eqtl contention rerun in progress'), indent=2) + '\n')
    interrupted = []
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda number, frame: interrupted.append(number))
    child = None
    sample_count = quiet_count = 0
    maximum_gap = 0.
    code = 125
    error = None
    last_sample = last_heartbeat = time.monotonic()
    try:
        check(expected, event='launch_check')
        child = subprocess.Popen(['bash', str(R / 'run.sh'), 'eqtl'], cwd=ROOT, start_new_session=True)
        emit('runner_started', runner_pid=child.pid, process_group=child.pid)
        print('Started eQTL-stage contention rerun:', started, flush=True)
        while True:
            now = time.monotonic()
            gap = now - last_sample
            maximum_gap = max(maximum_gap, gap)
            if interrupted:
                raise RuntimeError(f'Supervisor interrupted by signal {interrupted[-1]}')
            if (R / 'CPU_POLICY_VIOLATION.json').exists():
                raise RuntimeError('A per-command CPU-policy check failed')
            if gap > 5:
                emit('monitor_gap', seconds=gap)
                raise RuntimeError('CPU-policy monitor gap exceeded five seconds')
            sample_count += 1
            check(expected, sample=sample_count)
            last_sample = now
            with audit.open() as stream:
                stream.seek(position)
                for line in stream:
                    if not line.endswith('\n'):
                        break
                    position += len(line.encode())
                    event = json.loads(line)
                    if event['event'] == 'quiet_ready':
                        assert event.get('cpu_policy_check_id')
                        assert event['stage'] == 'eqtl'
                        quiet_count += 1
                        emit('quiet_start_observed', stage=event['stage'], quiet_start_utc=event['utc'],
                             quiet_start_pid=event['pid'], cpus=event['cpus'],
                             pre_command_check_id=event['cpu_policy_check_id'], policy_sha256=digest(expected))
            if now - last_heartbeat >= 15:
                emit('heartbeat', samples=sample_count, observed_quiet_starts=quiet_count,
                     maximum_poll_gap_seconds=maximum_gap, policy_sha256=digest(expected))
                last_heartbeat = now
            if child.poll() is not None:
                check(expected, event='final_check')
                code = child.returncode
                break
            time.sleep(1)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        emit('aborted', error=error)
        stop_child(child)
        code = 125
    finally:
        summary = dict(exit=code, started_utc=started, finished_utc=utc(),
                       cpu_policy_unchanged=error is None, cpu_policy_samples=sample_count,
                       cpu_policy_maximum_poll_gap_seconds=maximum_gap,
                       cpu_policy_observed_quiet_starts=quiet_count, error=error)
        emit('finished', **summary)
        (R / 'eqtl-rerun-complete.json').write_text(json.dumps(summary, indent=2) + '\n')
        combined = dict(summary, started_utc=prior['started_utc'],
                        cpu_policy_samples=prior['cpu_policy_samples'] + sample_count,
                        cpu_policy_maximum_poll_gap_seconds=max(prior['cpu_policy_maximum_poll_gap_seconds'], maximum_gap),
                        cpu_policy_observed_quiet_starts=prior['cpu_policy_observed_quiet_starts'] + quiet_count,
                        stage_reruns=['eqtl'], initial_attempt_exit=prior['exit'])
        (R / 'run-complete.json').write_text(json.dumps(combined, indent=2) + '\n')
        print('eQTL-stage contention rerun completed:', summary, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
