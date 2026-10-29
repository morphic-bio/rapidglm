#!/usr/bin/env python3
"""Compare the same 1,000-permutation native QTL inputs before/after review."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    receipts = root / 'validation/receipts/v1.4.0-r2'
    source = json.loads((receipts / 'source.json').read_text())
    output = {'purpose': __doc__, 'version': source['version'], 'commit': source['commit'], 'models': {}}
    for family in ['gaussian', 'nb']:
        paths = [root / f'validation/work/benchmark/runs-{version}/eqtl/rapidglm_{family}/chr22.cis.tsv'
                 for version in ['v1.4.0', 'v1.4.0-r2']]
        old, new = [pd.read_csv(path, sep='\t', na_values='NA').set_index('phenotype_id') for path in paths]
        assert old.index.equals(new.index)
        row = {'rows': len(new), 'old_statuses': old.status.value_counts().to_dict(),
               'new_statuses': new.status.value_counts().to_dict(),
               'old_table_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
               'new_table_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest(),
               'new_columns': sorted(set(new) - set(old)), 'common_numeric': {}}
        for column in old:
            if column not in new or not pd.api.types.is_numeric_dtype(old[column]):
                continue
            a, b = old[column].to_numpy(float), new[column].to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            equal = (a == b) | (np.isnan(a) & np.isnan(b))
            delta = np.abs(a[finite] - b[finite])
            row['common_numeric'][column] = {
                'exact_mismatches': int((~equal).sum()), 'finite_pairs': int(finite.sum()),
                'max_absolute_difference': float(delta.max()) if len(delta) else None,
                'max_relative_to_old': float((delta / np.maximum(np.abs(a[finite]), 1e-300)).max()) if len(delta) else None}
        changed = old.index[old.lead_variant.fillna('NA') != new.lead_variant.fillna('NA')]
        row['changed_leads'] = [{ 'phenotype_id': g, 'old': old.loc[g, 'lead_variant'], 'new': new.loc[g, 'lead_variant']}
                                for g in changed]
        if 'permutations_failed' in new:
            row['failed_permutations_total'] = int(new.permutations_failed.sum())
        output['models'][family] = row
    destination = args.output or receipts / 'QTL_RESULT_CHANGES.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
