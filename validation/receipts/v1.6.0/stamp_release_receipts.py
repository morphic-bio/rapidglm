#!/usr/bin/env python3
"""Attach source provenance to top-level review receipts without changing historical receipts."""
import argparse,json,pathlib,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);a=p.parse_args()
root=pathlib.Path('/data/lhhung/rapidglm-1.6.0')
source=json.loads((a.directory/'source.json').read_text())
for path in a.directory.glob('*.json'):
    if path.name.startswith(('GASPERINI_', 'supervisor-', 'PREPRINT_', 'preprint-')) or path.name in ('source.json', 'CPU_POLICY_EXPECTED.json', 'TESTS.json', 'STAGE_RERUNS.json', 'INVALIDATED_RUNS.json', 'IMAGE_SETUP_ATTEMPTS.json', 'chain-status.json'): continue
    value=json.loads(path.read_text())
    if isinstance(value,dict):
        value['_provenance']={k:value.get(k,source[k]) for k in ('version','commit','worktree_dirty')}
        path.write_text(json.dumps(value,indent=2)+'\n')
