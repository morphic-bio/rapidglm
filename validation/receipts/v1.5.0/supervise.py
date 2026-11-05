#!/usr/bin/env python3
"""Supervise the 1.5.0 receipts runner and audit the CPU policy; never change CPU settings.

    setsid nohup taskset -c 0-5 nice -n 10 python3 supervise.py LABEL STAGE... &

Runs `bash run.sh STAGE...` from the measured source tree. As the 1.4.0 supervisor: the policy is checked
synchronously at every quiet start (policy_hook) and once per second throughout; a mismatch, a failed
quiet-start check or a monitoring gap over five seconds stops only owned processes by PID (exit 125).
Each launch has its own LABEL and summary (supervisor-LABEL.json), so a stopped chain can be continued
from the next stage. Every five seconds it also records the concurrent host load (HOST_LOAD.jsonl: the
one-minute load, every process, and cumulative CPU jiffies on all 48 logical CPUs).
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
MAIN = R.parents[2]
CLONE = MAIN
os.environ['RAPIDGLM_CPU_POLICY_AUDIT'] = str(R / 'CPU_POLICY_AUDIT.jsonl')
os.environ['RAPIDGLM_CPU_POLICY_VIOLATION'] = str(R / 'CPU_POLICY_VIOLATION.json')
os.environ['RAPIDGLM_TIMING_AUDIT'] = str(R / 'timing-audit.jsonl')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(R))
from cpu_policy import EXPECTED, check, digest, emit, host_sample, read_policy, utc  # noqa: E402

CONTROLLER = ['check_run.py', 'chain.py', 'docker_measure.py', 'measurement_guard.py', 'CPU_POLICY_EXPECTED.json', 'supervise.py', 'cpu_policy.py', 'policy_hook/sitecustomize.py', 'policy_hook/usercustomize.py',
              'guarded.py', 'cpu_guard.py', 'verify_source.py', 'run.sh', 'vmath_e2e.py', 'jaxqtl_split.sh']


def stop_child(child):
    if child is not None and child.poll() is None:
        from cpu_policy import stop_owned
        stop_owned(child)
        child.wait()


def main():
    label, stages = sys.argv[1], sys.argv[2:]
    assert os.environ.get('SLURM_JOB_ID'), 'exclusive Slurm allocation required'
    assert stages, 'name the stages to run'
    smoke = stages == ['smoke']
    source = dict(commit='runner-smoke', cpu_policy=dict(expected=EXPECTED, controller_files_sha256={})) if smoke else json.loads((R / 'source.json').read_text())
    assert source['cpu_policy']['expected'] == EXPECTED, 'source.json policy differs from cpu_policy.EXPECTED'
    actual = read_policy()
    if actual != EXPECTED:
        emit('launch_refused', expected=EXPECTED, actual=actual, label=label)
        raise SystemExit('CPU policy differs from the expected setting; stopping without changing it')
    assert not (R / 'CPU_POLICY_VIOLATION.json').exists(), 'an earlier policy violation is unresolved'
    for name, value in source['cpu_policy']['controller_files_sha256'].items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == value, f'controller file changed: {name}'
    summary_path = R / f'supervisor-{label}.json'
    assert not summary_path.exists(), f'label already used: {summary_path}'
    if not smoke:
        subprocess.run(['python3', str(R / 'verify_source.py')], cwd=CLONE, check=True)
    started = utc()
    emit('started', label=label, stages=stages, commit=source['commit'], policy=EXPECTED,
         policy_sha256=digest(EXPECTED), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         poll_interval_seconds=1, maximum_allowed_poll_gap_seconds=5)
    interrupted = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: interrupted.append(number))
    audit = R / 'timing-audit.jsonl'
    position = audit.stat().st_size if audit.exists() else 0
    env = dict(os.environ, PYTHONPATH=str(R / 'policy_hook')+':'+str(R), RAPIDGLM_CPU_POLICY_ENABLED='1')
    child = None
    samples = quiet = 0
    maximum_gap = 0.
    code, error = 125, None
    last_sample = last_heartbeat = last_host = time.monotonic()
    host_log = (R / 'HOST_LOAD.jsonl').open('a')
    try:
        check(EXPECTED, event='launch_check', label=label)
        child = subprocess.Popen(['bash', str(R / 'run.sh'), *stages], cwd=CLONE, env=env, start_new_session=True)
        emit('runner_started', label=label, runner_pid=child.pid, ownership="runner descendants, checked by PID and start time")
        (R / 'runner.pid').write_text(f'{child.pid}\n')
        print('started', label, stages, started, flush=True)
        while True:
            now = time.monotonic()
            gap = now - last_sample
            maximum_gap = max(maximum_gap, gap)
            if interrupted:
                raise RuntimeError(f'supervisor interrupted by signal {interrupted[-1]}')
            if (R / 'CPU_POLICY_VIOLATION.json').exists():
                raise RuntimeError('a quiet-start CPU-policy check failed')
            if gap > 5:
                emit('monitor_gap', seconds=gap)
                raise RuntimeError('CPU-policy monitor gap exceeded five seconds')
            samples += 1
            check(EXPECTED, sample=samples)
            last_sample = now
            if now - last_host >= 5 or samples == 1:
                host_log.write(json.dumps(host_sample(), separators=(',', ':')) + '\n')
                host_log.flush()
                last_host = now
            if audit.exists():
                with audit.open() as stream:
                    stream.seek(position)
                    for line in stream:
                        if not line.endswith('\n'):
                            break
                        position += len(line.encode())
                        event = json.loads(line)
                        if event.get('event') == 'quiet_ready':
                            assert event.get('cpu_policy_check_id'), 'quiet start without a synchronous policy check'
                            quiet += 1
                            emit('quiet_start_observed', stage=event.get('stage'), quiet_start_utc=event['utc'],
                                 quiet_start_pid=event['pid'], cpus=event['cpus'],
                                 load_one_minute=event.get('load_one_minute'),
                                 pre_command_check_id=event['cpu_policy_check_id'], policy_sha256=digest(EXPECTED))
            if now - last_heartbeat >= 15:
                emit('heartbeat', label=label, samples=samples, observed_quiet_starts=quiet,
                     maximum_poll_gap_seconds=maximum_gap, policy_sha256=digest(EXPECTED))
                last_heartbeat = now
            if child.poll() is not None:
                check(EXPECTED, event='final_check', label=label)
                code = child.returncode
                break
            time.sleep(1)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        emit('aborted', label=label, error=error)
        stop_child(child)
        code = 125
    finally:
        host_log.write(json.dumps(host_sample(), separators=(',', ':')) + '\n')
        host_log.close()
        summary = dict(label=label, stages=stages, exit=code, started_utc=started, finished_utc=utc(),
                       cpu_policy_unchanged=error is None, cpu_policy_samples=samples,
                       cpu_policy_maximum_poll_gap_seconds=maximum_gap,
                       cpu_policy_observed_quiet_starts=quiet, error=error)
        emit('finished', **summary)
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        print('finished', summary, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
