#!/usr/bin/env python3
"""Prepare, execute and check the focused preprint evidence gate."""
import csv, datetime, hashlib, itertools, json, math, os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
ROOT = R.parents[2]
W = ROOT / 'validation/work/release-1.6.0-gpu'
OLD = Path('/data/lhhung/rapidglm')
OLD_RUNS = OLD / 'validation/work/benchmark/runs-v1.5.0-gpu'
WORK = ROOT / 'validation/work/benchmark/runs-v1.6.0-gpu/preprint'
ORDER = ['preprint_qtl', 'preprint_crt', 'preprint_replogle']
FIELDS = ['null_start_used', 'null_refined', 'null_cold_refits', 'batch_scoring', 'permutation_fits', 'permutations_refitted', 'crt_draw_used']

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp'); q.write_text(json.dumps(d, indent=2) + '\n'); q.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def run(*args): subprocess.run(list(map(str, args)), cwd=ROOT, check=True)
def milestone(s):
    with Path('<status-work>/STATUS.md').open('a') as f: f.write(utc() + ' — ' + s + '\n')
def plan(): return load(R / 'PREPRINT_PLAN.json')
def jobs(stage): return [x for x in plan()['commands'] if x['stage'] == stage]

def prepare():
    assert not (R / 'PREPRINT_PLAN.json').exists(), 'plan already frozen'
    commands = []; inputs = {}; baseline = {}
    def add(name, stage, old_config, subcommand, updates=None, pair=None, role=None):
        original = load(old_config); cfg = dict(original)
        for k, v in original.items():
            if k == 'output' or not isinstance(v, str): continue
            if v.startswith(str(OLD) + '/'):
                value = v.replace(str(OLD) + '/', str(ROOT) + '/', 1).replace('v1.5.0-gpu', 'v1.6.0-gpu')
                cfg[k] = value
                suffixes = ['.bed', '.bim', '.fam'] if k == 'genotypes' else ['']
                for suffix in suffixes:
                    a, b = Path(v + suffix), Path(value + suffix)
                    assert a.is_file() and b.is_file(), (a, b)
                    if str(b) not in inputs:
                        h = sha(b); assert sha(a) == h, (a, b)
                        inputs[str(b)] = dict(reference=str(a), sha256=h)
        directory = WORK / name; directory.mkdir(parents=True, exist_ok=False)
        cfg['output'] = str(directory / 'result')
        cfg.update(updates or {})
        path = directory / 'config.json'; save(path, cfg)
        expected = Path(original['output']); expected = expected if expected.is_absolute() else OLD / expected
        suffixes = ['.cis.tsv'] if subcommand == 'qtl' else ['.tsv', '.qq.tsv']
        refs = {}
        for suffix in suffixes:
            p = Path(str(expected) + suffix); assert p.is_file(), p
            refs[suffix] = dict(path=str(p), sha256=sha(p))
        command = ['build/rapidglm', subcommand, str(path)]
        commands.append(dict(name=name, stage=stage, command=command, config=str(path), config_sha256=sha(path),
                             original_config=str(old_config), original_config_sha256=sha(old_config),
                             directory=str(directory), prefix=cfg['output'], cpus='6-17' if subcommand == 'qtl' else '6-21',
                             threads=cfg['threads'], references=refs, pair=pair, role=role))
    for name in ['rapidglm_gaussian', 'rapidglm_gaussian10000', 'rapidglm_gaussian10000_double', 'rapidglm_nb']:
        add(name, ORDER[0], OLD_RUNS / 'eqtl' / name / 'config.json', 'qtl',
            {'permutation_fits': 'lockstep'} if name == 'rapidglm_nb' else {},
            'qtl_nb_chr22' if name == 'rapidglm_nb' else None, 'default')
    add('qtl_nb_separate', ORDER[0], OLD_RUNS / 'eqtl/rapidglm_nb/config.json', 'qtl', {'permutation_fits': 'separate'}, 'qtl_nb_chr22', 'opt_out')
    for variant in ['ks', 'default']:
        for role, enabled in [('default', True), ('opt_out', False)]:
            add('crt_' + variant + '_' + role, ORDER[1], OLD_RUNS / 'crt' / ('t100_g7418_' + variant) / 'config.json',
                'sceptre', {'batch_scoring': enabled}, 'crt_t100_g7418_' + variant, role)
    pilot = OLD / 'validation/work/replogle/k562-essential/pilot-v1.5.0-gpu/runs'
    for fc in ['strict', 'ks']:
        for role, mode in [('default', 'refine'), ('opt_out', 'cold')]:
            add('replogle_' + fc + '_' + role, ORDER[2], pilot / ('rapidglm_t16' + ('_ks' if fc == 'ks' else '') + '.json'),
                'sceptre', {'null_start': mode}, 'replogle_' + fc, role)
    assert len(commands) == 13
    for n in ['NB_QTL_BATCH', 'MERGEABLE_NULL', 'BATCH_SCORING', 'BATCH_SCORING_DEV2', 'GASPERINI_FULL_CLOUD']:
        baseline[n] = sha(ROOT / 'validation' / (n + '.json'))
    old_double = OLD / 'validation/work/benchmark/runs-v1.4.0-final/eqtl/rapidglm_gaussian10000/chr22.cis.tsv'
    save(R / 'PREPRINT_PLAN.json', dict(version='1.6.0', commit=load(R / 'source.json')['commit'], prepared_utc=utc(),
         stage_order=ORDER, commands=commands, inputs=inputs, protected_receipts_sha256=baseline,
         gaussian_double_1_4_reference=dict(path=str(old_double), sha256=sha(old_double)),
         scope='Focused changed-path agreement and five paired ablations. This is not a complete performance run.',
         deferred=['remaining bulk timing', 'cooks', 'rust', 'staging', 'pert50k', 'pertfull', 'seeds', 'untestable',
                   'reference-tool reruns', 'other CRT panels', 'Replogle pseudobulk', 'special', 'vmath_e2e', 'replogle_full', 'eqtl_gw', 'compress'],
         poisson_evidence='Existing NB_QTL_BATCH.json development receipt and coordinator functional checks, with its documented Beta-fit differences; no new Poisson timing claim.',
         publication_authorized=False))
    print('Prepared 13 commands and verified', len(inputs), 'input files')

def execute(stage):
    assert stage in ORDER and os.environ.get('SLURM_JOB_ID')
    assert os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED') == '1'
    assert set(os.sched_getaffinity(0)) <= set(range(6))
    for row in jobs(stage):
        assert sha(row['config']) == row['config_sha256']
        milestone('Preprint: ' + row['name'] + ' queued with unchanged quiet/contention guards.')
        env = dict(os.environ)
        if stage == 'preprint_crt': env['OMP_NUM_THREADS'] = '1'
        subprocess.run(['python3', str(R / 'guarded_retry.py'), '--cpus', row['cpus'], '--directory', row['directory'], '--', *row['command']], cwd=ROOT, env=env, check=True)
        save(W / 'preprint-command-status.json', dict(recorded_utc=utc(), last_completed=row['name'], stage=stage, job=os.environ['SLURM_JOB_ID']))
        milestone('Preprint: ' + row['name'] + ' measurement accepted.')

def coverage(stage, records, specification=None):
    specification = specification if specification is not None else jobs(stage)
    expected = {' '.join(x['command']): x for x in specification}
    errors = []
    if len(records) != len(expected) or {x['label'] for x in records} != set(expected): errors.append(['preprint command coverage', stage, len(records), len(expected)])
    for record in records:
        row = expected.get(record['label'])
        if row:
            lo, hi = map(int, row['cpus'].split('-'))
            if record['cpus'] != list(range(lo, hi + 1)) or record['division'] != 'slow': errors.append(['preprint execution layout', record['label']])
    return errors

def nb_agreement(a, b):
    keys = ['phenotype_id','status','lead_variant','statistic','p_value','estimate','standard_error','permutation_p','permutations','permutations_failed']
    maximum = 0.; count = 0
    with Path(a).open() as af, Path(b).open() as bf:
        for x, y in itertools.zip_longest(csv.DictReader(af, delimiter='\t'), csv.DictReader(bf, delimiter='\t')):
            assert x is not None and y is not None, 'NB row count differs'
            for k in keys: assert x[k] == y[k], (count, k, x[k], y[k])
            if x['beta_p'] != y['beta_p']:
                assert x['beta_p'] != 'NA' and y['beta_p'] != 'NA', 'NB Beta missing mismatch'
                p, q = float(x['beta_p']), float(y['beta_p']); assert math.isfinite(p) and math.isfinite(q)
                maximum = max(maximum, abs(p-q) / max(abs(p), 1e-300))
            count += 1
    assert count and maximum <= 7.9e-11, maximum
    return dict(passed=True, rows=count, required_columns_identical=True, max_beta_p_relative_difference=maximum, beta_p_relative_bound=7.9e-11)

def check(stage):
    from ablations import low_moi_agreement
    report_path = R / 'PREPRINT_AGREEMENT.json'
    report = load(report_path) if report_path.exists() else dict(commit=plan()['commit'], stages={})
    checked = []
    for row in jobs(stage):
        record_path = Path(row['prefix'] + '.run.json'); record = load(record_path)
        cfg = load(row['config']); assert sha(row['config']) == row['config_sha256']
        assert record.get('division') == 'slow', record.get('division')
        if stage == 'preprint_replogle':
            assert record['null_start_used'] == cfg['null_start']
            assert all(k in record for k in ['null_refined', 'null_cold_refits', 'crt_draw_used'])
        if stage == 'preprint_crt': assert record['batch_scoring'] == cfg['batch_scoring'] and record['crt_draw_used'] == 'hazard'
        if row['pair'] == 'qtl_nb_chr22':
            assert record['permutation_fits'] == cfg['permutation_fits']; assert record['permutations_refitted'] >= 0
            if row['role'] == 'opt_out': assert record['permutations_refitted'] == 0
        checks = []
        for suffix, ref in row['references'].items():
            current = Path(row['prefix'] + suffix); assert sha(ref['path']) == ref['sha256']
            if stage == 'preprint_replogle' and suffix == '.tsv': agreement = low_moi_agreement(Path(ref['path']), current)
            elif stage == 'preprint_replogle': continue
            elif row['name'] == 'rapidglm_nb': agreement = nb_agreement(ref['path'], current)
            else:
                assert sha(current) == ref['sha256'], ('result differs from baseline', row['name'], suffix)
                agreement = dict(passed=True, byte_identical=True)
            checks.append(dict(file=str(current.relative_to(ROOT)), sha256=sha(current), reference_sha256=ref['sha256'], agreement=agreement))
        if row['name'] == 'rapidglm_gaussian10000_double':
            ref = plan()['gaussian_double_1_4_reference']; assert sha(ref['path']) == ref['sha256'] == sha(row['prefix'] + '.cis.tsv')
        checked.append(dict(name=row['name'], checks=checks, record_sha256=sha(record_path), fields={k:record[k] for k in FIELDS if k in record}))
    report['stages'][stage] = dict(passed=True, commands=checked, checked_utc=utc()); save(report_path, report)
    print(stage, 'scientific agreement passed')

def assemble():
    import numpy as np
    import pandas as pd
    from ablations import low_moi_agreement
    sys.path.insert(0, str(ROOT / 'scripts')); from compare_eqtl import qvalues
    specification = plan(); state = load(R / 'preprint-chain-status.json')
    assert list(state['stages']) == ORDER and all(x['exit'] == 0 for x in state['stages'].values())
    run('python3', R / 'check_run_accepted.py', '--preprint')
    for stage in ORDER: check(stage)
    for p, row in specification['inputs'].items(): assert sha(p) == sha(row['reference']) == row['sha256'], p
    for n, value in specification['protected_receipts_sha256'].items(): assert sha(ROOT / 'validation' / (n + '.json')) == value, n
    pairs = []
    for name in dict.fromkeys(x['pair'] for x in specification['commands'] if x['pair']):
        pair = {x['role']:x for x in specification['commands'] if x['pair'] == name}; a, b = pair['default'], pair['opt_out']
        ah, bh = load(Path(a['directory']) / 'host.json'), load(Path(b['directory']) / 'host.json')
        assert not ah['contended'] and not bh['contended'] and ah['policy_unchanged'] and bh['policy_unchanged']
        suffix = '.cis.tsv' if name == 'qtl_nb_chr22' else '.tsv'
        ap, bp = Path(a['prefix'] + suffix), Path(b['prefix'] + suffix)
        if name.startswith('replogle'):
            agreement = low_moi_agreement(ap, bp)
            x, y = pd.read_csv(ap, sep='\t'), pd.read_csv(bp, sep='\t')
            finite = np.isfinite(x.p_value) & np.isfinite(y.p_value) & (y.p_value > 0)
            tail = (x.approximation != 0) & (y.approximation != 0) & finite
            rel = np.abs(x.p_value[finite] - y.p_value[finite]) / y.p_value[finite]
            agreement['tail_p_diagnostics'] = dict(tail_rows=int(tail.sum()), tail_values_changed=int(((x.p_value != y.p_value) & tail).sum()), max_relative_p_difference=float(rel.max()) if len(rel) else 0.)
        elif name.startswith('crt'):
            assert sha(ap) == sha(bp); assert sha(a['prefix'] + '.qq.tsv') == sha(b['prefix'] + '.qq.tsv')
            agreement = dict(passed=True, byte_identical=True, qq_byte_identical=True)
        else:
            agreement = nb_agreement(bp, ap)
            def discoveries(path, bh=False):
                frame = pd.read_csv(path, sep='\t').set_index('phenotype_id'); valid = frame.status.str.startswith('ok') & np.isfinite(frame.beta_p)
                values = frame.beta_p[valid]
                if bh:
                    values = values.sort_values(); qs = np.minimum.accumulate((values.to_numpy()*len(values)/np.arange(1,len(values)+1))[::-1])[::-1]
                    return sorted(values.index[qs <= .05])
                qs, _ = qvalues(values); return sorted(qs.index[qs < .05])
            for use_bh in [False, True]:
                sets = [discoveries(p, use_bh) for p in [ap, bp, a['references']['.cis.tsv']['path']]]; assert sets[0] == sets[1] == sets[2]
                agreement['bh_egenes' if use_bh else 'storey_egenes'] = dict(identical=True, count=len(sets[0]), genes=sets[0], threshold='<= 0.05' if use_bh else '< 0.05')
            agreement['separate_byte_identical_to_job_687'] = sha(bp) == a['references']['.cis.tsv']['sha256']; assert agreement['separate_byte_identical_to_job_687']
        ar, br = load(a['prefix']+'.run.json'), load(b['prefix']+'.run.json')
        pairs.append(dict(name=name, default_seconds=ah['wall_seconds'], ablation_seconds=bh['wall_seconds'],
                          speedup_default_over_opt_out=bh['wall_seconds']/ah['wall_seconds'], agreement=agreement,
                          threads=a['threads'], cpus=a['cpus'], default_sha256=sha(ap), ablation_sha256=sha(bp),
                          default_record={k:ar[k] for k in FIELDS if k in ar}, ablation_record={k:br[k] for k in FIELDS if k in br}))
    receipt = dict(version='1.6.0', commit=specification['commit'], scope=specification['scope'], host='biocloudgpu1', pairs=pairs, all_agreements_passed=True, finished_utc=utc())
    save(R / 'OPTIMIZATION_ABLATIONS.json', receipt); save(ROOT / 'validation/OPTIMIZATION_ABLATIONS.json', receipt)
    result = dict(recorded_utc=utc(), measured_commit=specification['commit'], focused_checks_passed=True, commands=13, paired_ablations=5,
                  full_run_complete=False, prior_complete_stages=7, partial_bulk_used_for_claims=False,
                  deferred=specification['deferred'], inputs_checked=len(specification['inputs']), protected_receipts_unchanged=True,
                  documentation='docs/PREPRINT_EVIDENCE.md', remaining=['review and package evidence', 'public-tree check', 'private fast-forward push', 'user review before any release or publication'])
    save(R / 'PREPRINT_GATE.json', result)
    doc = ROOT / 'docs/PREPRINT_EVIDENCE.md'; text = doc.read_text(); marker = '<!-- FOCUSED_RESULTS -->'
    assert marker in text
    table = '\n\nThe focused gate passed all 13 guarded commands and five agreement pairs. These are paired, single-run wall times on the same host and inputs; they are not confidence intervals or new reference-tool speedups.\n\n| Pair | Default (s) | Opt-out (s) | Opt-out / default |\n|---|---:|---:|---:|\n'
    for x in pairs: table += f"| {x['name']} | {x['default_seconds']:.3f} | {x['ablation_seconds']:.3f} | {x['speedup_default_over_opt_out']:.3f}× |\n"
    text = text.split(marker)[0] + marker + table; doc.write_text(text)
    save(W/'preprint-assembly-status.json', dict(recorded_utc=utc(), state='assembled_pending_review_and_private_push', gate=result))
    milestone('Preprint focused gate passed: 13 accepted commands, five paired ablations, scientific agreement and input rechecks complete; evidence review, public-tree check and private push remain.')

if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    elif action == 'run': execute(sys.argv[2])
    elif action == 'check': check(sys.argv[2])
    elif action == 'assemble': assemble()
    else: raise SystemExit('unknown action')
