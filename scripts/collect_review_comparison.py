#!/usr/bin/env python3
"""Collect old/new review metrics without modifying either set of raw results."""
import argparse
import datetime
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def load(path):
    return json.loads(path.read_text())


def bulk_rows(old, new):
    previous = {(r['dataset'], r['tool'], r['threads']): r for r in old['bulk']}
    rows = []
    for r in new['bulk']:
        key = r['dataset'], r['tool'], r['threads']
        before = previous[key]
        fields = ['seconds', 'all_seconds', 'agreement_with_deseq2', 'agreement_with_edger']
        runs = r['runs']
        chosen = min(range(len(runs)), key=lambda i: runs[i]['seconds'])
        rows.append(dict(dataset=key[0], tool=key[1], threads=key[2],
                         old={k: before[k] for k in fields if k in before},
                         new={k: r[k] for k in fields if k in r},
                         seconds_ratio_new_over_old=r['seconds'] / before['seconds'],
                         selected_repeat=chosen, selected_run=runs[chosen]))
    return rows



def replogle_table_changes(root, targets):
    """Retain native call membership, statuses and input hashes for each target."""
    result = {}
    for target in targets:
        result[target] = {}
        for workflow, column in [('rapidglm', 'padj'), ('edger_ql', 'FDR')]:
            records = []
            for version in ['v1.4.0', 'v1.4.0-r2']:
                path = root / f'validation/work/replogle/k562-essential/pilot-{version}/pseudobulk/datasets/{target}/{workflow}.tsv'
                with path.open() as stream:
                    rows = list(csv.DictReader(stream, delimiter='\t'))
                calls = set()
                for row in rows:
                    try:
                        if float(row[column]) <= .1:
                            calls.add(row['gene'])
                    except ValueError:
                        pass
                records.append((rows, calls, hashlib.sha256(path.read_bytes()).hexdigest()))
            before, after = records
            if [r['gene'] for r in before[0]] != [r['gene'] for r in after[0]]:
                raise ValueError(f'Changed gene ordering: {target}/{workflow}')
            result[target][workflow] = dict(
                genes=len(after[0]), old_table_sha256=before[2], new_table_sha256=after[2],
                old_statuses=dict(Counter(r['status'] for r in before[0])),
                new_statuses=dict(Counter(r['status'] for r in after[0])),
                old_calls=len(before[1]), new_calls=len(after[1]),
                gained_calls=sorted(after[1] - before[1], key=int),
                lost_calls=sorted(before[1] - after[1], key=int))
        filters = []
        for version in ['v1.4.0', 'v1.4.0-r2']:
            path = root / f'validation/work/replogle/k562-essential/pilot-{version}/pseudobulk/datasets/{target}/rapidglm.run.json'
            record = load(path)
            filters.append({k: record[k] for k in ['filter_theta', 'filter_threshold', 'replaced_genes']})
        result[target]['bulk_filter'] = dict(old=filters[0], new=filters[1])
    return result


def elapsed(path):
    text = path.read_text()
    match = re.search(r'Elapsed \(wall clock\) time .*?:\s*([\d:.]+)\s*$', text, re.M)
    if not match:
        raise ValueError(f'No wall time in {path}')
    parts = [float(x) for x in match[1].split(':')]
    seconds = 0.0
    for p in parts:
        seconds = 60 * seconds + p
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text)
    return dict(wall_seconds=seconds, peak_rss_kbytes=int(rss[1]) if rss else None,
                time_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                time_log=text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quiet-log', type=Path, help='Captured run.sh stderr containing QTL readiness summaries')
    parser.add_argument('--partial', action='store_true', help='Permit stages that are still running')
    args = parser.parse_args()
    root = args.root.resolve()
    old = root / 'validation/work/benchmark/receipts/v1.4.0'
    new = root / 'validation/receipts/v1.4.0-r2'
    source = load(new / 'source.json')
    result = dict(purpose='Combined review-fix effects against the untouched v1.4.0 receipts; no per-fix ablation is claimed.',
                  version=source['version'], commit=source['commit'],
                  engine_change_commit=source['engine_change_commit'],
                  collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  baseline='validation/work/benchmark/receipts/v1.4.0',
                  bulk=[], pending=[])
    required = ['BULK_DE.json', 'EDGER_QL.json', 'applications_bulk_small.json',
                'applications_bulk_large.json', 'COOKS_LEGACY.json', 'COOKS_THYROID.json',
                'COOKS_FLAGS.json', 'replogle_sc_strict.json', 'replogle_sc_ks.json',
                'replogle_pseudobulk.json', 'EQTL_COMPARE.json']
    for name in required:
        if not (new / name).exists():
            result['pending'].append(name)
            continue
        a = load(new / name)
        b = load(old / name) if (old / name).exists() else None
        if name.startswith('applications_bulk'):
            result['bulk'].extend(bulk_rows(b, a))
        elif name in ['BULK_DE.json', 'COOKS_LEGACY.json', 'COOKS_THYROID.json']:
            result[name] = {k: dict(old=b['datasets'][k], new=v) for k, v in a['datasets'].items()}
        elif name == 'EDGER_QL.json':
            result[name] = {k: dict(old=b[k], new=v) for k, v in a.items() if not k.startswith('_')}
        elif name.startswith('replogle_'):
            section = 'single_cell' if 'single_cell' in a else 'pseudobulk'
            result[name] = dict(old=b[section], new=a[section],
                                timing_comparability='SC: same fit-check mode; pseudobulk: old reference commands included extra shrinkage/classic steps, new commands use matching minimal outputs.')
        else:
            result[name] = dict(new=a, old=b,
                                baseline_note='The v1.4.0 stage has no agreement receipt.' if b is None else '')
    runs = [run for name in ['applications_bulk_small.json', 'applications_bulk_large.json']
            if (new / name).exists() for row in load(new / name)['bulk'] for run in row['runs']]
    result['bulk_timing_quality'] = dict(
        definition='Seconds are the minimum of all repeats, as in the original harness. others_cpu is the selected-core fraction attributed to other work; its contended threshold is 0.2. Quiet waits are excluded.',
        measured_commands=len(runs),
        maximum_start_load=max((r['load'] for r in runs), default=None),
        maximum_others_cpu=max((r['others_cpu'] for r in runs), default=None),
        contended_commands=sum(r['contended'] for r in runs),
        selected_minima_others_above_point05=[dict(dataset=r['dataset'], tool=r['tool'], threads=r['threads'], selected_run=r['selected_run'])
                                             for r in result['bulk'] if r['selected_run']['others_cpu'] > .05])
    result['replogle_timing_records'] = {}
    for version in ['v1.4.0', 'v1.4.0-r2']:
        path = root / f'validation/work/replogle/k562-essential/pilot-{version}/pseudobulk/timing_t1.json'
        if path.exists():
            result['replogle_timing_records'][version] = load(path)
    pilot = result['replogle_timing_records'].get('v1.4.0-r2', {})
    records = [v[k] for v in pilot.values()
               for k in ['rapidglm', 'deseq2', 'rapidglm_edger_ql', 'edger']]
    if records:
        result['replogle_timing_quality'] = dict(
            measured_commands=len(records), repeats_per_tool_and_target=1,
            maximum_start_load=max(v['load_before'] for v in records),
            maximum_end_load=max(v['load_average'] for v in records),
            definition='Each whole command starts after the host/core quiet predicate. End load can increase during a run. Per-core other-work fractions were not collected for the pilot; CPU, wall time and start/end load are retained.')
    pseudobulk = result.get('replogle_pseudobulk.json', {}).get('new', {}).get('targets')
    if pseudobulk:
        result['replogle_native_table_changes'] = replogle_table_changes(root, pseudobulk)
    result['qtl_timings'] = []
    names = [('rapidglm_gaussian', 'Gaussian', 1000), ('rapidglm_gaussian10000', 'Gaussian', 10000),
             ('rapidglm_nb', 'NB score', 1000), ('tensorqtl', 'Gaussian', 10000),
             ('fastqtl', 'Gaussian', 'adaptive 1000-10000'),
             ('jaxqtl_nb_score_perm1000', 'NB score', 1000),
             ('jaxqtl_nb_score_spa_acat', 'NB SPA+ACAT', 'none')]
    for name, model, permutations in names:
        p = root / 'validation/work/benchmark/runs-v1.4.0-r2/eqtl' / name
        q = root / 'validation/work/benchmark/runs-v1.4.0/eqtl' / name
        if (p / 'host.json').exists():
            row = dict(tool=name, model=model, permutations=permutations, threads=12,
                       new={**elapsed(p / 'time.log'), **load(p / 'host.json')})
            if (q / 'time.log').exists():
                row['old'] = {**elapsed(q / 'time.log'), **load(q / 'host.json')}
            if (p / 'chr22.run.json').exists():
                row['run'] = load(p / 'chr22.run.json')
            result['qtl_timings'].append(row)
    result['qtl_timing_quality'] = dict(
        repeats_per_tool_and_budget=1,
        quiet_start='guarded() enforces scripts/quiet_run.py --wait-only with the selected CPUs after cpu_guard.py wait; individual successful quiet samples are not persisted.',
        waiting='host.json waited_for_idle_seconds covers only cpu_guard.py wait, not the additional global/core quiet wait. All waits are excluded from command wall seconds.',
        cpu_accounting='cpus_busy_seconds includes the measured command; it is not other-work CPU time.',
        peak_rss='GNU time maximum RSS is a per-process maximum, not aggregate process-tree memory for multi-process commands.')
    if args.quiet_log:
        quiet = []
        for line in args.quiet_log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('timed_command_ready_utc') and record.get('cpus') == '16-27':
                quiet.append(record)
        order = ['rapidglm_gaussian', 'rapidglm_nb', 'rapidglm_gaussian10000',
                 'fastqtl', 'tensorqtl', 'jaxqtl_nb_score_spa_acat', 'jaxqtl_nb_score_perm1000']
        if len(quiet) > len(order) or (not args.partial and len(quiet) != len(order)):
            raise ValueError(f'Expected seven QTL readiness summaries; found {len(quiet)}')
        by_tool = dict(zip(order, quiet))
        for row in result['qtl_timings']:
            row['new']['quiet_ready'] = by_tool.get(row['tool'])
        result['qtl_timing_quality']['readiness_summaries'] = by_tool
        result['qtl_timing_quality']['readiness_log_sha256'] = hashlib.sha256(args.quiet_log.read_bytes()).hexdigest()
        result['qtl_timing_quality']['readiness_scope'] = 'Final readiness time, load averages, selected CPUs and quiet-wait duration are retained; individual ten-second core/load samples are not.'
    if not args.partial and len(result['qtl_timings']) != len(names):
        raise ValueError('Expected all seven completed QTL command timings')
    if result['pending'] and not args.partial:
        raise RuntimeError('Missing final stages: ' + ', '.join(result['pending']))
    output = args.output or new / 'COMPARISON.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{output}: {len(result["bulk"])} bulk comparisons, {len(result["qtl_timings"])} QTL timings, pending {result["pending"]}')


if __name__ == '__main__':
    main()
