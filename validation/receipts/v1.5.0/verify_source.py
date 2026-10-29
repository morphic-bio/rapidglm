#!/usr/bin/env python3
"""Reject a release stage if its measured source or installed artifacts changed (1.5.0).

The receipts run from the clean source tree tagged locally as v1.5.0-dev.3.
Every recorded artifact and container image must remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import subprocess

R = Path(__file__).resolve().parent
MAIN = Path('/data/lhhung/rapidglm')
CLONE = MAIN
source = json.loads((R / 'source.json').read_text())
for root in {MAIN, CLONE}:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    tag = subprocess.check_output(['git', 'rev-parse', f"{source['tag']}^{{}}"], cwd=root, text=True).strip()
    assert head == tag == source['commit'], f'measured commit changed: {root}'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=root, text=True).strip(), f'tracked source changed: {root}'
for name, expected in source['binaries_sha256'].items():
    path = Path(name) if name.startswith('/') else CLONE / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'artifact changed: {name}'
for name, expected in source['container_images'].items():
    actual = subprocess.check_output(['docker', 'image', 'inspect', name, '--format', '{{.Id}}'], text=True).strip()
    assert actual == expected, f'container changed: {name}'
