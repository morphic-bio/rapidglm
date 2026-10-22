#!/usr/bin/env python3
"""Untimed matched-pair calibration check using public SCEPTRE pilot outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/replogle_pilot.py').exists())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=ROOT / 'build/rapidglm')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    if not set(os.sched_getaffinity(0)) <= set(range(8)):
        p.error('Run untimed on CPUs 0-7 with taskset -c 0-7')
    pilot, work = args.pilot.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    refpath = pilot / 'runs/sceptre_w16/calibration.tsv'
    ref = pd.read_csv(refpath, sep='\t')
    keys = ['response_id', 'grna_target']
    if ref.duplicated(keys).any():
        raise ValueError('Reference calibration has duplicate response/target pairs')
    responses = (pilot / 'sc/responses.txt').read_text().split()
    index = {name: i for i, name in enumerate(responses)}
    guides = pd.read_csv(pilot / 'sc/guides.tsv', sep='\t')
    targets = set(guides.loc[guides.non_targeting == 1, 'target'])
    if not ref.response_id.isin(index).all() or not ref.grna_target.isin(targets).all():
        raise ValueError('A reference pair is outside the prepared public native panel')
    pairs = pd.DataFrame({'response': ref.response_id.map(index), 'target': ref.grna_target, 'kind': 'negative'})
    pairpath = work / 'pairs.tsv'
    pairs.sort_values(['target', 'response']).to_csv(pairpath, sep='\t', index=False)
    base = ref[keys + ['p_value', 'pass_qc']].rename(columns={'p_value': 'reference_p'})
    native = {}
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    for mode, suffix in [('strict', ''), ('ks', '_ks')]:
        config = json.loads((pilot / f'runs/rapidglm_t16{suffix}.json').read_text())
        prefix = work / mode
        config.update(pairs=str(pairpath), threads=args.threads, output=str(prefix), fit_check=mode)
        cfgpath = work / f'{mode}.json'
        cfgpath.write_text(json.dumps(config, indent=2) + '\n')
        subprocess.run([str(args.binary), 'sceptre', str(cfgpath)], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        tablepath = Path(str(prefix) + '.tsv')
        table = pd.read_csv(tablepath, sep='\t', na_values='NA')
        table['response_id'] = table.response.map(lambda i: responses[i])
        ok = table[table.status.str.startswith('ok') & np.isfinite(table.p_value)]
        selected = ok[['response_id', 'target', 'p_value']].rename(columns={'target': 'grna_target', 'p_value': mode + '_p'})
        base = base.merge(selected, on=keys, how='inner', validate='one_to_one')
        native[mode] = {'rows': len(table), 'statuses': table.status.value_counts().to_dict(),
                        'finite_ok': len(ok), 'table_sha256': sha(tablepath)}
    base = base[base.pass_qc & np.isfinite(base.reference_p)].copy()
    rows = {}
    for label in ['reference', 'strict', 'ks']:
        values = base[label + '_p'].to_numpy()
        rows[label] = {f'count_below_{cutoff}': int((values <= cutoff).sum()) for cutoff in [.05, .01]}
        rows[label].update({f'fraction_below_{cutoff}': float((values <= cutoff).mean()) for cutoff in [.05, .01]})
    source = json.loads((ROOT / 'validation/receipts/v1.4.0-r2/source.json').read_text())
    result = {'purpose': __doc__, 'timed': False, 'cpus': sorted(os.sched_getaffinity(0)),
              'version': subprocess.check_output([str(args.binary), '--version'], text=True).strip(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'measured_engine_commit': source['commit'],
              'recipe_sha256': sha(Path(__file__)),
              'recipe_source': Path(__file__).read_text(),
              'reference_calibration_sha256': sha(refpath), 'pair_file_sha256': sha(pairpath),
              'reference_pairs': len(ref), 'common_tested_all_three': len(base),
              'native': native, 'same_pair_fractions': rows,
              'scope': 'Independent untimed calibration check on the reference-selected control pairs. Pilot timing and original-panel agreement remain separate.'}
    if not len(base):
        raise ValueError('No finite common calibration pairs')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
