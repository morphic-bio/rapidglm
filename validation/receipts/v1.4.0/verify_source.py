#!/usr/bin/env python3
"""Reject a release stage if its measured source or installed artifacts changed."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
source = json.loads(Path(__file__).with_name('source.json').read_text())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
tag = subprocess.check_output(['git', 'rev-parse', 'v1.4.0-dev.4^{}'], cwd=root, text=True).strip()
assert head == tag == source['commit'], 'measured commit changed'
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                   cwd=root, text=True).strip(), 'tracked source changed'
for name, expected in source['binaries_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, f'artifact changed: {name}'
for name, expected in source['container_images'].items():
    actual = subprocess.check_output(['docker', 'image', 'inspect', name, '--format', '{{.Id}}'], text=True).strip()
    assert actual == expected, f'container changed: {name}'
