#!/usr/bin/env python3
"""Guard a command; archive and retry only verified contention rejections."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data/lhhung/rapidglm-1.6.0')
R = ROOT / 'validation/receipts/v1.6.0'
W = ROOT / 'validation/work/release-1.6.0-gpu'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(R))
from quiet_run import LOCK, cpu_set, wait_quiet, finish_measurement

REJECTION = 'measurement exceeded 5% contention; rerun required: '


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def snapshot(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('command output directory contains a symlink: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(directory))] = sha(path)
    return result


def verified_rejection(error, command, cpus, began_ns, audit_directory):
    if not str(error).startswith(REJECTION):
        return None
    path = Path(str(error)[len(REJECTION):]).resolve()
    if path.parent != audit_directory.resolve() or not path.name.startswith(str(os.getpid()) + '-'):
        return None
    # Linux filesystem mtimes may use a coarse clock. The guard's filename
    # carries time.time_ns(); validate that precise creation token as well.
    try:
        written_ns = int(path.stem.split('-', 1)[1])
    except (ValueError, IndexError):
        return None
    if not began_ns <= written_ns <= time.time_ns() + 1_000_000_000:
        return None
    if path.suffix != '.json' or not path.is_file() or path.stat().st_mtime_ns < began_ns - 1_000_000_000:
        return None
    row = json.loads(path.read_text())
    if (row.get('pid') != os.getpid() or row.get('label') != ' '.join(command)
            or row.get('cpus') != sorted(cpu_set(cpus))
            or row.get('stage') != os.environ.get('RAPIDGLM_STAGE')
            or row.get('slurm_job_id') != os.environ.get('SLURM_JOB_ID')
            or row.get('policy_unchanged') is not True or row.get('contended') is not True):
        return None
    if not (row['others_cpu'] > .05 or any(v > .05 for v in row['sibling_busy_fraction'].values())):
        return None
    started = datetime.datetime.fromisoformat(row['started_utc'].replace('Z', '+00:00')).timestamp()
    finished = datetime.datetime.fromisoformat(row['finished_utc'].replace('Z', '+00:00')).timestamp()
    # Audit ISO timestamps have microsecond precision; began_ns has nanoseconds.
    if not began_ns / 1e9 - 1e-6 <= started <= finished <= time.time() + 1:
        return None
    return path, row


def verify_retry_record(record, actual, root, command_reruns):
    """Recheck exclusion evidence independently when assembling accepted timings."""
    assert record['command_exit'] == 0
    assert record['audit_sha256'] == actual['raw_audit_sha256']
    for key in ['label', 'pid', 'stage', 'cpus', 'slurm_job_id', 'started_utc', 'finished_utc']:
        assert record[key] == actual[key], key
    assert record['label'] == ' '.join(record['command'])
    assert actual['contended'] is True and actual['policy_unchanged'] is True
    assert actual['others_cpu'] > .05 or any(v > .05 for v in actual['sibling_busy_fraction'].values())
    assert record['archive_files']
    for item in record['archive_files']:
        assert sha(root / item['copy']) == item['sha256'], item['copy']
    baseline = record['baseline_files_sha256']
    archived = {item['relative_output']: item['sha256'] for item in record['archive_files']}
    assert all(archived.get(name) == value for name, value in baseline.items())
    assert any(x.get('audit_sha256') == record['audit_sha256'] and x == record for x in command_reruns)


def execute(command, cpus, directory):
    with (directory / 'stdout.log').open('w') as out, (directory / 'stderr.log').open('w') as err:
        return subprocess.run(['/usr/bin/time', '-v', '-o', str(directory / 'time.log'),
                               'taskset', '-c', cpus, *command], stdout=out, stderr=err)


def run_guarded(command, cpus, directory, *, quiet=wait_quiet, finish=finish_measurement,
                run=execute, audit_directory=None, retry_path=None, reruns_path=None,
                archive=None):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ['host.json', 'host.failed.json', 'time.log', 'stdout.log', 'stderr.log']):
        raise RuntimeError('archive previous command outputs before rerunning: ' + str(directory))
    baseline = snapshot(directory)
    audit_directory = audit_directory or Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR', str(R / 'commands')))
    retry_path = retry_path or R / 'GUARDED_COMMAND_RETRIES.json'
    reruns_path = reruns_path or R / 'STAGE_RERUNS.json'
    archive = archive or W / 'guarded-command-retries'
    commit = json.loads((R / 'source.json').read_text())['commit']
    while True:
        began_ns = time.time_ns()
        quiet(cpus, label=' '.join(command))
        result = run(command, cpus, directory)
        rejection = None
        try:
            row = finish() or {}
        except RuntimeError as error:
            rejection = verified_rejection(error, command, cpus, began_ns, audit_directory)
            if rejection is None:
                raise
            audit_path, row = rejection
        host = dict(row, exit=result.returncode, command=command)
        if result.returncode:
            save(directory / 'host.failed.json', host)
            return result.returncode
        if rejection is None:
            if row.get('contended') is not False or row.get('policy_unchanged') is not True:
                raise RuntimeError('measurement did not pass its guard')
            save(directory / 'host.json', host)
            return 0

        save(directory / 'host.failed.json', host)
        target = archive / (str(os.environ.get('SLURM_JOB_ID', 'untimed')) + '-' + str(os.getpid()) + '-' + str(time.time_ns()))
        target.mkdir(parents=True, exist_ok=False)
        files = snapshot(directory)
        copies = []
        for name, value in files.items():
            original = directory / name
            copied = target / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, copied)
            assert sha(copied) == value, str(copied)
            copies.append(dict(original=os.path.relpath(original, ROOT), copy=os.path.relpath(copied, ROOT),
                               relative_output=name, sha256=value))
        save(target / 'MANIFEST.json', dict(files=copies, all_sha256_verified=True))
        if any(files.get(name) != value for name, value in baseline.items()):
            raise RuntimeError('command changed a pre-existing input; output archived, automatic retry refused')
        record = {key: row[key] for key in ['stage', 'label', 'pid', 'cpus', 'slurm_job_id',
                                            'started_utc', 'finished_utc', 'others_cpu', 'sibling_busy_fraction']}
        record.update(commit=commit, command=list(command), command_exit=0,
                      audit_file=os.path.relpath(audit_path, R), audit_sha256=sha(audit_path),
                      archive_files=copies, baseline_files_sha256=baseline,
                      reason='Successful command rejected by the unchanged 5% contention predicate. All outputs archived and SHA-256 verified; identical command retried after the full quiet predicate.')
        proof = json.loads(retry_path.read_text()) if retry_path.exists() else dict(commit=commit, retries=[])
        assert proof['commit'] == commit
        proof['retries'].append(record)
        save(retry_path, proof)
        reruns = json.loads(reruns_path.read_text()) if reruns_path.exists() else dict(commit=commit, reruns=[], command_reruns=[])
        assert reruns['commit'] == commit
        reruns.setdefault('command_reruns', []).append(record)
        save(reruns_path, reruns)
        for name, value in files.items():
            if name not in baseline:
                path = directory / name
                assert sha(path) == value
                path.unlink()
        assert snapshot(directory) == baseline
        print(json.dumps(dict(event='guarded_command_retry', label=row['label'],
                              audit_file=record['audit_file'])), flush=True)


def install_policy_hook():
    # This external controller is not one of the frozen sitecustomize entrypoints.
    # Apply the same pre-command CPU-policy check at quiet_run's audit boundary.
    if not os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED'):
        return
    import quiet_run
    from cpu_policy import check_quiet_start
    original = quiet_run.audit

    def audited(record):
        if record.get('event') == 'quiet_ready':
            check_quiet_start(record)
        return original(record)

    quiet_run.audit = audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required')
    hold = W / 'TIMED_RUN_HOLD.json'
    if hold.exists() and json.loads(hold.read_text()).get('hold_active'):
        parser.error('timed receipts are on hold; await the updated plan')
    install_policy_hook()
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_guarded(command, args.cpus, args.directory)


if __name__ == '__main__':
    raise SystemExit(main())
