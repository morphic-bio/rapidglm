#!/usr/bin/env python3
"""Independent one-factor reduced-model check for the public GATA1 pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.optimize import brentq
from scipy.special import xlogy
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--gene', type=int, default=2779)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p = args.dataset.resolve()
    with threadpool_limits(limits=1):
        counts = mmread(p / 'counts.mtx').tocsr()
    y = counts[args.gene].toarray().ravel()
    design = pd.read_csv(p / 'design.tsv', sep='\t').drop(columns='perturbed').to_numpy()
    _, groups = np.unique(design, axis=0, return_inverse=True)
    with threadpool_limits(limits=1):
        rank = np.linalg.matrix_rank(design)
    if rank != len(np.unique(groups)):
        raise ValueError('Reduced design is not a complete one-factor group basis')
    library = pd.read_csv(p / 'edger/samples.tsv', sep='\t').effective_lib_size.to_numpy()
    shared = json.loads((p / 'edger/reference.json').read_text())['ql']
    run = json.loads((p / 'edger_ql.run.json').read_text())
    native = pd.read_csv(p / 'edger_ql.tsv', sep='\t').set_index('gene').loc[args.gene]
    reference = pd.read_csv(p / 'edger/genes_ql.tsv', sep='\t').set_index('gene').loc[args.gene]
    check = pd.read_csv(p / 'edger/fit_check.tsv', sep='\t').set_index('gene').loc[args.gene]
    def solve(phi):
        mu = np.zeros_like(y, dtype=float)
        residuals = []
        for group in np.unique(groups):
            mask = groups == group
            if y[mask].sum() == 0:
                continue
            start = np.log(y[mask].sum() / library[mask].sum())
            def score(beta):
                fitted = library[mask] * np.exp(beta)
                return np.sum((y[mask] - fitted) / (1 + phi * fitted))
            beta = brentq(score, start - 50, start + 50, xtol=1e-13)
            mu[mask] = library[mask] * np.exp(beta)
            residuals.append(abs(score(beta)))
        keep = mu > 0
        if phi > 0:
            k = 1 / phi
            deviance = 2 * np.sum(xlogy(y[keep], y[keep] / mu[keep]) +
                                 (y[keep] + k) * np.log1p((mu[keep] - y[keep]) / (y[keep] + k)))
        else:
            deviance = 2 * np.sum(xlogy(y[keep], y[keep] / mu[keep]) - y[keep] + mu[keep])
        return {'working_dispersion': phi, 'reduced_deviance': float(deviance),
                'maximum_absolute_group_score': max(residuals)}
    root_native = solve(run['dispersion'] / run['average_ql_dispersion'])
    root_reference = solve(shared['dispersion'] / shared['average_ql_dispersion'])
    implied_native = float(native.deviance + native.F * native.s2_post)
    result = {'purpose': __doc__, 'timed': False, 'version': run['version'],
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'gene': args.gene, 'samples': len(y), 'reduced_design_groups': len(np.unique(groups)),
              'native': {'F': float(native.F), 'PValue': float(native.PValue), 'full_deviance': float(native.deviance),
                         's2_post': float(native.s2_post), 'implied_reduced_deviance': implied_native},
              'reference': {'F': float(reference.F), 'PValue': float(reference.PValue),
                            'fit_check': {k: (v.item() if hasattr(v, 'item') else v) for k, v in check.items()}},
              'independent_at_native_dispersion': root_native,
              'independent_at_reference_dispersion': root_reference,
              'native_deviance_difference_from_independent': implied_native - root_native['reduced_deviance'],
              'reference_reduced_difference_from_independent': float(check.reduced_reported) - root_reference['reduced_deviance'],
              'method': 'Drop perturbed; each distinct remaining design row is a lane. Solve each nonzero lane score sum((y-mu)/(1+phi*mu))=0 by bracketed root search; zero-count lanes use their mu=0 likelihood limit. Evaluate the NB unit deviance with log1p. Both checks use public reference effective-library offsets.',
              'scope': 'Reported and at-fitted reference deviances agree in this example. The independently solved reduced objective differs from that public fit; reference reduced iteration/failure metadata is absent, so no failure flag is inferred.',
              'input_sha256': {name: hashlib.sha256((p / name).read_bytes()).hexdigest()
                               for name in ['counts.mtx', 'design.tsv', 'edger/samples.tsv', 'edger/reference.json', 'edger_ql.run.json', 'edger/fit_check.tsv']},
              'recipe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'recipe_source': Path(__file__).read_text()}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['recipe_source', 'input_sha256']}, indent=2))


if __name__ == '__main__':
    main()
