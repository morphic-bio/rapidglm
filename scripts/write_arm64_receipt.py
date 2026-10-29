#!/usr/bin/env python3
"""Collect completed native, portable and ARM64 tests and application comparisons.

Run scripts/build_arm64.sh all (or its recorded CMake/QEMU commands), then
scripts/validate_arm64.py. Supply the native and portable Boost summary logs.
This collects verification results; elapsed QEMU times are not hardware benchmarks.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=pathlib.Path, default=ROOT / 'build-arm64-r2')
p.add_argument('--native-log', type=pathlib.Path, required=True)
p.add_argument('--portable-log', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, default=ROOT / 'validation/ARM64.json')
a = p.parse_args()


def summary(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    match = re.search(r'(\d+) test cases? out of (\d+) passed', text)
    if not match:
        raise ValueError('no completed Boost summary: ' + str(path))
    passed, total = map(int, match.groups())
    skipped = re.search(r'(\d+) test cases? out of \d+ skipped', text)
    failed = re.search(r'(\d+) test cases? out of \d+ failed', text)
    result = {'run': total - (int(skipped[1]) if skipped else 0), 'passed': passed,
              'skipped': int(skipped[1]) if skipped else 0, 'failed': int(failed[1]) if failed else 0,
              'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if result['failed'] or result['skipped'] or result['run'] != result['passed']:
        raise ValueError((path, result))
    return result


unit = a.work / 'full-direct-ks-tests.log'
cli = a.work / 'direct-ks-cli.log'
c99 = a.work / 'direct-ks-c99.log'
cli_text = cli.read_text()
match = re.search(r'Ran (\d+) tests?', cli_text)
if not match or not cli_text.rstrip().endswith('OK'):
    raise ValueError('ARM64 CLI suite did not pass')
if '100% tests passed' not in c99.read_text():
    raise ValueError('ARM64 C99 caller did not pass')
apps = json.loads((a.work / 'application_comparison.json').read_text())
if len(apps['applications']) != 10:
    raise ValueError('incomplete ARM64 application comparison')
failed_relative = {}
for name, case in apps['applications'].items():
    if not case['discoveries_equal'] or not case['status_equal']:
        raise ValueError((name, 'application classification differs'))
    failed_relative[name] = {column: value for column, value in case['numeric_columns'].items()
                             if value['max_relative'] > 1e-8 or value['nonfinite_mismatches']}
receipt = {
    'version': subprocess.check_output([str(ROOT / 'build/rapidglm'), '--version'], text=True).strip(),
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'platform': 'Linux ARM64 under QEMU on Linux x86-64; no native ARM64 timing claim',
    'cpus': '0-7, untimed, without the timing lock', 'emulated_cpu': 'max',
    'compiler': (a.work / 'compiler.txt').read_text().strip(),
    'qemu': (a.work / 'qemu.txt').read_text().strip(),
    'unit_tests': {'x86_dispatch': summary(a.native_log),
                   'x86_forced_portable': summary(a.portable_log),
                   'arm64': summary(unit)},
    'arm64_c99': 'passed', 'arm64_cli': {'run': int(match[1]), 'passed': int(match[1]), 'skipped': 0},
    'arm64_ks_cdf': 'Double-precision Phi(z)-2T(z,a) for KS only; promoted distribution CDF for inferential tails. Reference test covers 7 shapes and 201 points per shape, absolute error <2e-14.',
    'applications': apps['applications'],
    'relative_comparison_criterion': 1e-8,
    'columns_exceeding_relative_criterion': failed_relative,
    'all_numeric_columns_meet_relative_criterion': not any(failed_relative.values()),
    'note': 'Exact empirical and fitted-tail comparisons are retained separately. Near-zero values can have large relative differences; those failures are reported, with the original 1e-8 criterion retained.'}
a.output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'unit_tests': receipt['unit_tests'], 'arm64_cli': receipt['arm64_cli'],
                  'relative_criterion_met': receipt['all_numeric_columns_meet_relative_criterion']}, indent=2))
