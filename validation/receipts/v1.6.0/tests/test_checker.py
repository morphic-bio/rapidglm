#!/usr/bin/env python3
"""Exercise the actual acceptance checker against isolated synthetic audit records."""
import copy,datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
D=R.parents[2]/'validation/work/release-1.6.0-gpu/runner-tests'
D.mkdir(parents=True,exist_ok=True)
ROOT=Path('/data/lhhung/rapidglm-1.6.0')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
g=module('guarded_retry',R/'guarded_retry.py')
check=module('acceptance_checker_candidate',R/'check_run_accepted.py')
base=D/'checker-fixture-final';r=base/'validation/receipts/v1.6.0';w=base/'validation/work/fixture'
r.mkdir(parents=True);w.mkdir(parents=True)
check.R=r;check.W=w;check.ORDER=['qtl']
commit=json.loads((g.R/'source.json').read_text())['commit']
proof=json.loads((D/'tests-5/public_qtl_integration/retries.json').read_text())
record=copy.deepcopy(proof['retries'][0])
t=datetime.datetime(2026,10,29,tzinfo=datetime.timezone.utc)
def utc(seconds):return (t+datetime.timedelta(seconds=seconds)).isoformat()
record.update(stage='qtl',slurm_job_id='synthetic-no-job',cpus=[6],started_utc=utc(1),finished_utc=utc(10),
              label='synthetic qtl command',command=['synthetic','qtl','command'],
              others_cpu=0.,sibling_busy_fraction={'30':.05435891450480667},audit_file='commands/rejected.json')
for item in record['archive_files']:item['copy']=os.path.relpath(ROOT/item['copy'],base)
rejected={k:record[k] for k in ['pid','label','stage','slurm_job_id','cpus','started_utc','finished_utc','others_cpu','sibling_busy_fraction']}
rejected.update(contended=True,policy_unchanged=True,host_start={'processes':[{'pid':1}]},host_end={'processes':[{'pid':1}]})
g.save(r/record['audit_file'],rejected);record['audit_sha256']=g.sha(r/record['audit_file'])
clean=copy.deepcopy(rejected);clean.update(started_utc=utc(40),finished_utc=utc(50),contended=False,sibling_busy_fraction={'30':0.})
g.save(r/'commands/accepted.json',clean)
g.save(r/'source.json',dict(commit=commit,cpu_policy=dict(expected={},controller_files_sha256={})))
g.save(r/'chain-status.json',dict(stages={'qtl':dict(exit=0,label='fixture',started_utc=utc(0),finished_utc=utc(60),slurm_job_id='synthetic-no-job')}))
g.save(r/'supervisor-fixture.json',dict(exit=0,cpu_policy_unchanged=True,cpu_policy_maximum_poll_gap_seconds=1,cpu_policy_samples=2))
digest=hashlib.sha256(json.dumps({},sort_keys=True).encode()).hexdigest()
policy=[];quiet=[]
for i,when in enumerate([1,40]):
 policy.append(dict(check_id=str(i),event='policy_check',utc=utc(when),matched=True,policy_sha256=digest))
 quiet.append(dict(event='quiet_ready',stage='qtl',pid=record['pid'],label=record['label'],utc=utc(when),
                   cpus=[6],sibling_cpus=[30],samples=[dict(passed=True,load_before=0,load_after=0,busy_fractions={'6':0.,'30':0.})]*2,
                   load_one_minute=0,cpu_policy_check_id=str(i)))
(r/'CPU_POLICY_AUDIT.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in policy))
(r/'timing-audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in quiet))
sys.argv=['check_run_accepted.py','--partial']
outcomes=[]
for mode in ['valid','audit_hash_changed','source_changed','missing_rerun','archive_missing']:
 p=copy.deepcopy(record);rr=[copy.deepcopy(record)]
 if mode=='audit_hash_changed':p['audit_sha256']='0'*64
 if mode=='source_changed':p['commit']='different'
 if mode=='missing_rerun':rr=[]
 if mode=='archive_missing':p['archive_files'][0]['copy']='nonexistent'
 g.save(r/'GUARDED_COMMAND_RETRIES.json',dict(commit=commit,retries=[p]))
 g.save(r/'STAGE_RERUNS.json',dict(commit=commit,reruns=[],command_reruns=rr))
 exit_code=check.main();result=json.loads((w/'timing-audit-progress.json').read_text())
 if mode=='valid':
  assert exit_code==0 and not result['problems']
  assert result['stages'][0]['commands']==1 and result['stages'][0]['quiet_starts']==1
 else:
  assert exit_code!=0 and any(x[0]=='guarded retry evidence invalid' for x in result['problems'])
  assert result['stages'][0]['commands']==2, 'unverified rejection was improperly excluded'
 outcomes.append(dict(case=mode,passed=True,commands=result['stages'][0]['commands'],problems=result['problems']))
g.save(D/'CHECKER_TEST_RESULTS.json',dict(result='PASS',untimed=True,synthetic_fixture=True,slurm_job_submitted=False,
       tests_passed=len(outcomes),tests=outcomes,cpus=sorted(os.sched_getaffinity(0)),checker_sha256=g.sha(R/'check_run_accepted.py')))
print('Acceptance checker tests PASS')
