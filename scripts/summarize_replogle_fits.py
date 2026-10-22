#!/usr/bin/env python3
"""Summarize public edgeR fit diagnostics for the twenty-target pilot."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/replogle_pilot.py').exists())


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((ROOT / 'validation/receipts/v1.4.0-r2/source.json').read_text())
    targets = {}
    for directory in sorted((args.pilot / 'pseudobulk/datasets').iterdir()):
        if not directory.is_dir():
            continue
        path = directory / 'edger/fit_check.tsv'
        with path.open() as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            columns = reader.fieldnames
            rows = list(reader)
        record = {'genes': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for model in ['full', 'reduced']:
            delta = [abs(number(r[model + '_reported']) - number(r[model + '_at_fitted']))
                     for r in rows if number(r[model + '_reported']) is not None and number(r[model + '_at_fitted']) is not None]
            flags = [r.get(model + '_failed') for r in rows]
            iterations = [number(r.get(model + '_iter')) for r in rows]
            known = [f for f in flags if f in ['TRUE', 'FALSE']]
            record[model] = {
                'finite_deviance_pairs': len(delta),
                'exact_reported_vs_at_fitted_mismatches': sum(d != 0 for d in delta),
                'maximum_absolute_reported_vs_at_fitted_difference': max(delta) if delta else None,
                'failure_metadata_present': model + '_failed' in columns,
                'known_failure_flags': len(known),
                'failed': sum(f == 'TRUE' for f in known) if known else None,
                'iteration_metadata_present': model + '_iter' in columns,
                'maximum_iterations': max((x for x in iterations if x is not None), default=None)}
        targets[directory.name] = record
    if len(targets) != 20:
        raise ValueError(f'Expected all twenty public target diagnostics; found {len(targets)}')
    result = {'purpose': __doc__, 'timed': False, 'version': source['version'],
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'reference_api': 'scripts/reference_edger_fits.R: glmFit and nbinomUnitDeviance through public functions',
              'targets': targets,
              'scope': 'Exact numeric comparisons are descriptive. Missing public iteration/failure metadata stays missing; it is not converted to successful convergence. GATA1_FIT_DIAGNOSTIC.json separately checks the reduced objective.',
              'recipe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'recipe_source': Path(__file__).read_text()}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('targets', len(targets), 'genes', sum(t['genes'] for t in targets.values()))
    print('maximum reported/at-fitted differences', {model: max(t[model]['maximum_absolute_reported_vs_at_fitted_difference'] or 0 for t in targets.values()) for model in ['full', 'reduced']})
    print('public failure flags', {model: {n: t[model]['failed'] for n, t in targets.items()} for model in ['full', 'reduced']})


if __name__ == '__main__':
    main()
