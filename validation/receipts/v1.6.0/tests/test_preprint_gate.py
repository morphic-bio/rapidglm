#!/usr/bin/env python3
"""Exercise real focused acceptance and numerical rejection checks without timing fits."""
import copy, datetime, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import preprint_gate as gate
import check_run_accepted as checker
spec=gate.plan();real_r=R;W=gate.W
base=Path(tempfile.mkdtemp(prefix='preprint-checker-',dir=W));r=base/'validation/receipts/v1.6.0';w=base/'work';r.mkdir(parents=True);w.mkdir()
checker.R=r;checker.W=w;gate.R=r
gate.save(r/'PREPRINT_PLAN.json',spec)
commit=spec['commit'];t=datetime.datetime(2026,10,31,tzinfo=datetime.timezone.utc)
def utc(s):return (t+datetime.timedelta(seconds=s)).isoformat()
gate.save(r/'source.json',dict(commit=commit,cpu_policy=dict(expected={},controller_files_sha256={})))
gate.save(r/'preprint-chain-status.json',dict(stages={'preprint_qtl':dict(exit=0,label='fixture',started_utc=utc(0),finished_utc=utc(100),slurm_job_id='fixture-no-job')}))
gate.save(r/'supervisor-fixture.json',dict(exit=0,cpu_policy_unchanged=True,cpu_policy_maximum_poll_gap_seconds=1,cpu_policy_samples=100))
digest=hashlib.sha256(json.dumps({},sort_keys=True).encode()).hexdigest();rows=[];quiet=[];policy=[]
for i,specrow in enumerate(x for x in spec['commands'] if x['stage']=='preprint_qtl'):
    when=1+10*i;pid=1000+i;label=' '.join(specrow['command']);cpus=list(range(6,18));siblings=list(range(30,42))
    rows.append(dict(stage='preprint_qtl',label=label,pid=pid,slurm_job_id='fixture-no-job',cpus=cpus,division='slow',started_utc=utc(when),finished_utc=utc(when+1),contended=False,policy_unchanged=True,others_cpu=0.,sibling_busy_fraction={str(c):0. for c in siblings},host_start={'processes':[{'pid':pid}]},host_end={'processes':[{'pid':pid}]}))
    quiet.append(dict(event='quiet_ready',stage='preprint_qtl',pid=pid,label=label,utc=utc(when),cpus=cpus,sibling_cpus=siblings,load_one_minute=0.,cpu_policy_check_id=str(i),samples=[dict(passed=True,load_before=0.,load_after=0.,busy_fractions={str(c):0. for c in cpus+siblings})]*2))
    policy.append(dict(check_id=str(i),event='policy_check',utc=utc(when),matched=True,policy_sha256=digest))
(r/'CPU_POLICY_AUDIT.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in policy));(r/'commands').mkdir()
sys.argv=['check_run_accepted.py','--preprint','--partial'];results=[]
for mode in ['valid','missing_command','wrong_cores','wrong_division','contention','missing_policy','bad_quiet_sample']:
    current=copy.deepcopy(rows);q=copy.deepcopy(quiet)
    for p in (r/'commands').glob('*.json'):p.unlink()
    if mode=='missing_command':current.pop()
    if mode=='wrong_cores':current[0]['cpus']=[6]
    if mode=='wrong_division':current[0]['division']='fast'
    if mode=='contention':current[0]['others_cpu']=.06;current[0]['contended']=True
    if mode=='missing_policy':q[0]['cpu_policy_check_id']='absent'
    if mode=='bad_quiet_sample':q[0]['samples'][0]['busy_fractions']['6']=.2
    for i,row in enumerate(current):gate.save(r/'commands'/f'{i}.json',row)
    (r/'timing-audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in q))
    code=checker.main();assert (code==0)==(mode=='valid'),mode
    results.append(dict(case=mode,passed=True))
# Bounds must reject a scientific change even when a measurement itself passed.
keys=['phenotype_id','status','lead_variant','statistic','p_value','estimate','standard_error','permutation_p','permutations','permutations_failed','beta_p']
a=base/'reference.tsv';b=base/'candidate.tsv';original=['g','ok','v','1','0.1','0.1','0.1','0.2','1000','0','0.5']
def table(path,values):path.write_text('\t'.join(keys)+'\n'+'\t'.join(values)+'\n')
table(a,original)
for mode in ['identity','allowed_rounding','beta_outside_bound','status_change','missing_row']:
    values=original.copy()
    if mode=='allowed_rounding':values[-1]=str(.5*(1+7e-11))
    if mode=='beta_outside_bound':values[-1]=str(.5*(1+8e-11))
    if mode=='status_change':values[1]='failed'
    table(b,values)
    if mode=='missing_row':b.write_text('\t'.join(keys)+'\n')
    try:gate.nb_agreement(a,b);passed=True
    except AssertionError:passed=False
    assert passed==(mode in ['identity','allowed_rounding']),mode
    results.append(dict(case=mode,passed=True))
gate.save(W/'PREPRINT_RUNNER_TESTS.json',dict(result='PASS',untimed=True,synthetic=True,cpus=sorted(os.sched_getaffinity(0)),tests=results,tests_passed=len(results),fixture=str(base)))
print('Focused gate: 12 acceptance/scientific checks passed')
