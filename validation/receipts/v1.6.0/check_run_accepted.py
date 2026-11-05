#!/usr/bin/env python3
"""Check accepted stage exits, quiet starts, policy and per-command contention evidence."""
import argparse,datetime,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/data/lhhung/rapidglm-1.6.0/validation/receipts/v1.6.0')
from chain import ORDER,R,W
sys.path.insert(0,'/data/lhhung/rapidglm-1.6.0/validation/work/release-1.6.0-gpu')
from guarded_retry import verify_retry_record
UNTIMED={'glmacc','edgerval','cooks','staging','seeds','compress'}
def load(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def lines(p):
 if not p.exists():p=Path(str(p)+'.gz')
 if not p.exists():return []
 text=gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text()
 return [json.loads(s) for s in text.splitlines() if s.strip()]
def ts(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def inner_command_problems(stage,c):
 errors=[]
 if stage in {'batch','bulkval','untestable','replogle','ablations'}:
  if any('quiet_run.py' in x['label'] for x in c):errors.append(['driver-wide measurement must not include inner quiet waits',stage])
 expected_inner={'batch':15,'bulkval':60,'untestable':12,'replogle':83,'ablations':5}
 if stage in expected_inner and len(c)!=expected_inner[stage]:errors.append(['inner command coverage',stage,len(c),expected_inner[stage]])
 if stage=='batch':
  labels={'batch '+family+' native threads='+str(t) for family in ['poisson','nb'] for t in [1,4,16]}
  labels|={'batch '+family+' Python '+tool for family in ['poisson','nb'] for tool in ['statsmodels','rapidglm.fit']}
  labels|={'batch poisson R glm.fit','batch poisson R rapidglm','batch nb R glm.nb','batch nb R rapidglm','batch nb glmGamPoi'}
  if {x['label'] for x in c}!=labels:errors.append(['batch command identity coverage'])
 return errors
def main():
 global ORDER
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--preprint',action='store_true');a=ap.parse_args()
 if a.preprint:
  from preprint_gate import ORDER as focused_order
  ORDER=list(focused_order)
 source=load(R/'source.json');state=load(R/('preprint-chain-status.json' if a.preprint else 'chain-status.json'));policy=lines(R/'CPU_POLICY_AUDIT.jsonl')
 quiet=lines(R/'timing-audit.jsonl');allcommands=[]
 reruns=load(R/'STAGE_RERUNS.json')['reruns'] if (R/'STAGE_RERUNS.json').exists() else []
 command_reruns=load(R/'STAGE_RERUNS.json').get('command_reruns',[]) if (R/'STAGE_RERUNS.json').exists() else []
 for p in sorted((R/'commands').glob('*.json*')):
  if p.suffix not in ['.json','.gz']:continue
  row=load(p)
  if 'host_start' in row:
   raw=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
   row['raw_audit_file']=str(p.relative_to(R));row['raw_audit_sha256']=hashlib.sha256(raw).hexdigest()
   allcommands.append(row)
 allcommands=[x for x in allcommands if 'host_start' in x]
 errors=[];stages=[];commands=[];starts=[];segments=[];accepted_windows=[]
 checks={x['check_id']:x for x in policy if 'check_id' in x}
 digest=hashlib.sha256(json.dumps(source['cpu_policy']['expected'],sort_keys=True).encode()).hexdigest()
 for name,expected in source['cpu_policy']['controller_files_sha256'].items():
  if hashlib.sha256((R/name).read_bytes()).hexdigest()!=expected:errors.append(['controller changed',name])
 for stage in ORDER:
  row=state['stages'].get(stage,{})
  done=row.get('exit')==0
  if not done:
   if not a.partial:errors.append(['stage incomplete',stage,row])
   stages.append(dict(stage=stage,complete=False,exits=[row.get('exit')] if row else [],attempts_queued=int(bool(row))))
   continue
  supervisor=load(R/('supervisor-'+row['label']+'.json'));segments.append(supervisor)
  if supervisor['exit'] or not supervisor['cpu_policy_unchanged'] or supervisor['cpu_policy_maximum_poll_gap_seconds']>5:
   errors.append(['supervisor failed',stage])
  low,high=ts(row['started_utc']),ts(row['finished_utc']);accepted_windows.append((low,high))
  prior=[x for x in reruns if x['stage']==stage]
  for attempt in prior:
   if attempt['commit']!=source['commit'] or not attempt.get('reason'):errors.append(['invalid rerun record',stage])
  accepted=lambda x:x.get('stage')==stage and low<=ts(x.get('started_utc',x.get('utc')))<=high
  q=[x for x in quiet if x['event']=='quiet_ready' and accepted(x)]
  c=[x for x in allcommands if accepted(x)]
  policy_samples=supervisor['cpu_policy_samples'];maximum_gap=supervisor['cpu_policy_maximum_poll_gap_seconds']
  stage_jobs=[row['slurm_job_id']];retained_count=0
  if stage=='glmspeed':
   proof=load(R/'GLM_SPEED_COMPLETION.json')
   spec_path=W/'glm_speed_jobs.json'
   if hashlib.sha256(spec_path.read_bytes()).hexdigest()!=source['cpu_policy']['command_resume']['job_manifest_sha256']:
    errors.append(['GLM input/job manifest changed'])
   specification=load(spec_path)
   if not proof['complete'] or proof['commit']!=source['commit'] or any(proof[k]!=specification[k] for k in ['commit','jobs','input_sha256']):
    errors.append(['GLM resume specification mismatch'])
   expected_labels=[x['label'] for x in proof['jobs']]
   if len(expected_labels)!=408 or set(expected_labels)!=set(proof['accepted']):
    errors.append(['GLM accepted command coverage'])
   for name,value in proof['input_sha256'].items():
    if hashlib.sha256((R.parents[2]/name).read_bytes()).hexdigest()!=value:errors.append(['GLM input changed',name])
   windows=[(low,high,row['slurm_job_id'])]
   for attempt in proof['supporting_attempts']:
    p=R/attempt['supervisor_file'];support=load(p)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=attempt['supervisor_sha256']:errors.append(['prior supervisor changed'])
    if not support['cpu_policy_unchanged'] or support['cpu_policy_maximum_poll_gap_seconds']>5:errors.append(['prior command policy failed'])
    if not any(x['invalidated_label']==attempt['label'] and x.get('retained_clean_commands')==attempt['accepted_commands'] for x in prior):
     errors.append(['retained attempt lacks inspected record'])
    lo,hi=ts(attempt['started_utc']),ts(attempt['finished_utc'])
    accepted_windows.append((lo,hi));windows.append((lo,hi,attempt['slurm_job_id']))
    stage_jobs.append(attempt['slurm_job_id']);policy_samples+=support['cpu_policy_samples']
    maximum_gap=max(maximum_gap,support['cpu_policy_maximum_poll_gap_seconds'])
    segments.append(dict(support,partial_command_acceptance=attempt['accepted_commands']))
   by_file={x['raw_audit_file'].removesuffix('.gz'):x for x in allcommands}
   c=[]
   for label in expected_labels:
    record=proof['accepted'][label];actual=by_file[record['audit_file']]
    if actual['raw_audit_sha256']!=record['audit_sha256'] or any(actual[k]!=record[k] for k in ['label','pid','slurm_job_id','started_utc','finished_utc']):
     errors.append(['accepted GLM audit changed',label])
    if actual['cpus']!=[6] or actual['division']!='slow' or actual['stage']!='glmspeed':errors.append(['GLM execution scope changed',label])
    if not any(lo<=ts(actual['started_utc'])<=ts(actual['finished_utc'])<=hi and job==actual['slurm_job_id'] for lo,hi,job in windows):
     errors.append(['GLM audit outside supervised attempt',label])
    if hashlib.sha256((R.parents[2]/record['result_file']).read_bytes()).hexdigest()!=record['result_sha256']:
     errors.append(['accepted GLM result changed',label])
    retained_count+=actual['slurm_job_id']!=row['slurm_job_id'];c.append(actual)
   for rejected in proof['invalidated']:
    actual=by_file[rejected['audit_file']]
    if actual['raw_audit_sha256']!=rejected['audit_sha256'] or not actual['contended']:
     errors.append(['invalidated GLM evidence mismatch',rejected['label']])
    if hashlib.sha256((R.parents[2]/rejected['archive_result']).read_bytes()).hexdigest()!=rejected['result_sha256']:
     errors.append(['invalidated GLM output archive changed',rejected['label']])
    if not any(x.get('audit_sha256')==rejected['audit_sha256'] for x in command_reruns):
     errors.append(['invalidated GLM command lacks rerun record',rejected['label']])
   measured={x['raw_audit_file'].removesuffix('.gz') for x in allcommands if x.get('stage')=='glmspeed' and x.get('slurm_job_id') in stage_jobs and x['label'].startswith('GLM ')}
   accounted={x['audit_file'] for x in proof['accepted'].values()}|{x['audit_file'] for x in proof['invalidated']}
   if measured!=accounted:errors.append(['unaccounted GLM command audits',sorted(measured-accounted),sorted(accounted-measured)])
   q=[x for x in quiet if x['event']=='quiet_ready' and x.get('stage')==stage and any(z['pid']==x['pid'] and z['label']==x['label'] and abs(ts(z['started_utc'])-ts(x['utc']))<3 for z in c)]
  if stage in {'bulk','pert50k','pertfull','rust'}:
   retry_path=R/('RUST_COMMAND_RETRIES.json' if stage=='rust' else 'APPLICATION_COMMAND_RETRIES.json')
   proof=load(retry_path) if retry_path.exists() else dict(commit=source['commit'],retries=[])
   if proof['commit']!=source['commit']:errors.append(['application retry source mismatch'])
   rejected=[x for x in proof['retries'] if x['stage']==stage and x['slurm_job_id']==row['slurm_job_id'] and low<=ts(x['started_utc'])<=high]
   by_file={x['raw_audit_file'].removesuffix('.gz'):x for x in c};rejected_rows=[]
   for record in rejected:
    actual=by_file.get(record['audit_file'])
    if actual is None:
     errors.append(['application retry missing command audit',record['audit_file']]);continue
    if actual['raw_audit_sha256']!=record['audit_sha256'] or any(actual[k]!=record[k] for k in ['label','pid','slurm_job_id','started_utc','finished_utc']):
     errors.append(['application retry audit changed',record['audit_file']])
    if not actual['contended'] or not actual['policy_unchanged']:errors.append(['application retry is not a guarded rejection',record['audit_file']])
    for copied in record['archive_files']:
     if hashlib.sha256((R.parents[2]/copied['copy']).read_bytes()).hexdigest()!=copied['sha256']:errors.append(['application rejected output archive changed',copied['copy']])
    if not record['archive_files'] or not any(x.get('audit_sha256')==record['audit_sha256'] for x in command_reruns):
     errors.append(['application retry lacks output or rerun evidence',record['audit_file']])
    rejected_rows.append(actual)
   excluded={x['raw_audit_file'] for x in rejected_rows}
   c=[x for x in c if x['raw_audit_file'] not in excluded]
   q=[x for x in q if not any(z['pid']==x['pid'] and z['label']==x['label'] and abs(ts(z['started_utc'])-ts(x['utc']))<3 for z in rejected_rows)]
   expected_commands={'bulk':198,'pert50k':14,'pertfull':7,'rust':8}[stage]
   if len(c)!=expected_commands:errors.append(['application accepted command coverage',stage,len(c),expected_commands])
  if stage=='bulkval':
   proof=load(R/'BULK_COMMAND_RETRIES.json') if (R/'BULK_COMMAND_RETRIES.json').exists() else dict(commit=source['commit'],retries=[])
   if proof['commit']!=source['commit']:errors.append(['bulk retry source mismatch'])
   rejected=[x for x in proof['retries'] if x['slurm_job_id']==row['slurm_job_id'] and low<=ts(x['started_utc'])<=high]
   by_file={x['raw_audit_file'].removesuffix('.gz'):x for x in c}
   rejected_rows=[]
   for record in rejected:
    actual=by_file.get(record['audit_file'])
    if actual is None:
     errors.append(['bulk retry missing command audit',record['audit_file']]);continue
    if actual['raw_audit_sha256']!=record['audit_sha256'] or any(actual[k]!=record[k] for k in ['label','pid','slurm_job_id','started_utc','finished_utc']):
     errors.append(['bulk retry audit changed',record['audit_file']])
    if not actual['contended'] or not actual['policy_unchanged']:errors.append(['bulk retry is not a guarded rejection',record['audit_file']])
    if not actual['label'].startswith('bulk validation rapid.'):errors.append(['bulk retry scope changed',record['label']])
    for copied in record['archive_files']:
     if hashlib.sha256((R.parents[2]/copied['copy']).read_bytes()).hexdigest()!=copied['sha256']:
      errors.append(['bulk rejected output archive changed',copied['copy']])
    if not record['archive_files'] or not any(x.get('audit_sha256')==record['audit_sha256'] for x in command_reruns):
     errors.append(['bulk retry lacks output or rerun evidence',record['audit_file']])
    rejected_rows.append(actual)
   excluded={x['raw_audit_file'] for x in rejected_rows}
   c=[x for x in c if x['raw_audit_file'] not in excluded]
   q=[x for x in q if not any(z['pid']==x['pid'] and z['label']==x['label'] and abs(ts(z['started_utc'])-ts(x['utc']))<3 for z in rejected_rows)]
  if stage=='rust':
   expected_rust_labels={'Rust comparison '+data+' '+tool for data in ['airway','thyroid32','thyroid128','thyroid'] for tool in ['PyDESeq2','native']}
   if len(c)!=8 or {x['label'] for x in c}!=expected_rust_labels:errors.append(['Rust comparison dataset/tool coverage'])
   sys.path.insert(0,str(R))
   from verify_rust_live_cpu import verify as verify_rust_cpu
   for x in c:
    if x['label'].endswith(' PyDESeq2'):
     for problem in verify_rust_cpu(x):errors.append([problem,x['raw_audit_file']])
   for record in rejected:
    if record['label'].endswith(' PyDESeq2'):
     cleanup=record.get('fresh_process_cleanup',{})
     if record.get('retry_strategy')!='fresh_interpreter' or cleanup.get('all_rejected_descendants_gone') is not True or cleanup.get('rejected_interpreter_exit')!=75 or cleanup.get('attempt_id')!=record.get('attempt_id'):
      errors.append(['Rust retry did not restore fresh process start',record['audit_file']])
     if any(x['pid']==record['pid'] or (x.get('container_cpu_audit') or {}).get('attempt_id')==record.get('attempt_id') for x in c if x['label']==record['label']):
      errors.append(['Rust retry reused rejected process',record['audit_file']])
   for x in c:
    if x['label'].endswith(' PyDESeq2') and (not x['parent_cpu_counted'] or x['accounting_start']['affinity']!=x['cpus'] or x['accounting_end']['affinity']!=x['cpus']):errors.append(['Rust comparison parent CPU accounting',x['raw_audit_file']])
  generic_path=R/'GUARDED_COMMAND_RETRIES.json'
  generic=load(generic_path) if generic_path.exists() else dict(commit=source['commit'],retries=[])
  if generic['commit']!=source['commit']:errors.append(['guarded retry source mismatch'])
  rejected=[x for x in generic['retries'] if x['stage']==stage and x['slurm_job_id']==row['slurm_job_id'] and low<=ts(x['started_utc'])<=high]
  by_file={x['raw_audit_file'].removesuffix('.gz'):x for x in c};rejected_rows=[]
  for record in rejected:
   actual=by_file.get(record['audit_file'])
   try:
    assert record['commit']==source['commit']
    assert actual is not None, 'missing command audit'
    verify_retry_record(record,actual,R.parents[2],command_reruns)
   except (AssertionError,KeyError,TypeError,ValueError,OSError) as error:
    errors.append(['guarded retry evidence invalid',record['audit_file'],str(error)])
    continue
   rejected_rows.append(actual)
  excluded={x['raw_audit_file'] for x in rejected_rows}
  c=[x for x in c if x['raw_audit_file'] not in excluded]
  q=[x for x in q if not any(z['pid']==x['pid'] and z['label']==x['label'] and abs(ts(z['started_utc'])-ts(x['utc']))<3 for z in rejected_rows)]
  errors.extend(inner_command_problems(stage,c))
  if a.preprint:
   from preprint_gate import coverage
   errors.extend(coverage(stage,c))
  for x in q:
   cpus=set(x['cpus']);siblings={v+24 for v in cpus}
   if not cpus or not cpus<=set(range(6,22)) or set(x.get('sibling_cpus',[]))!=siblings:errors.append(['quiet CPU layout',stage,x['utc']])
   if len(x['samples'])!=2 or x['load_one_minute']>8:errors.append(['quiet predicate',stage,x['utc']])
   for sample in x['samples']:
    if not(sample['passed'] and max(sample['load_before'],sample['load_after'])<=8 and set(map(int,sample['busy_fractions']))==cpus|siblings and all(v is not None and v<.2 for v in sample['busy_fractions'].values())):errors.append(['quiet sample',stage,x['utc']])
   check=checks.get(x.get('cpu_policy_check_id'),{})
   if not check.get('matched') or check.get('policy_sha256')!=digest:errors.append(['quiet policy check',stage,x['utc']])
   if not any(z.get('pid')==x['pid'] and z['label']==x['label'] and abs(ts(z['started_utc'])-ts(x['utc']))<3 for z in c):errors.append(['quiet start missing completed command audit',stage,x['utc']])
  for x in c:
   if x['contended'] or not x['policy_unchanged'] or x['others_cpu']>.05 or max(x['sibling_busy_fraction'].values(),default=0)>.05:errors.append(['contention',stage,x['started_utc']])
   if not x['host_start'].get('processes') or not x['host_end'].get('processes'):errors.append(['process evidence missing',stage])
  if stage not in UNTIMED and not c:errors.append(['timed stage has no command records',stage])
  if len(q)!=len(c):errors.append(['command count mismatch',stage,len(q),len(c)])
  stages.append(dict(stage=stage,timed=stage not in UNTIMED,complete=True,exits=[x['prior_exit'] for x in prior]+[0],attempts_queued=len(prior)+1,rerun_reasons=[x['reason'] for x in prior],quiet_starts=len(q),commands=len(c),start_load_range=[min(x['load_one_minute'] for x in q),max(x['load_one_minute'] for x in q)] if q else None,cpu_policy_checks=policy_samples,maximum_poll_gap_seconds=maximum_gap,slurm_job_id=row['slurm_job_id'],slurm_job_ids=sorted(set(stage_jobs)),retained_prior_commands=retained_count,command_reruns=sum(x['stage']==stage for x in command_reruns)))
  for item in c:
   summary={k:v for k,v in item.items() if k not in ['host_start','host_end']}
   for key in ['host_start','host_end']:
    summary[key]={k:item[key][k] for k in ['utc','t','load1','load'] if k in item[key]}
   commands.append(summary)
  starts.extend(q)
 accepted_checks=[x for x in checks.values() if any(lo<=ts(x['utc'])<=hi for lo,hi in accepted_windows)]
 allgood=all(x.get('matched',True) and x.get('policy_sha256',digest)==digest for x in accepted_checks)
 if not allgood:errors.append(['CPU policy check mismatch in accepted attempt'])
 invalidated_policy_checks=[x for x in checks.values() if not x.get('matched',True) or x.get('policy_sha256',digest)!=digest]
 for x in invalidated_policy_checks:
  if not any(ts(z['started_utc'])<=ts(x['utc'])<=ts(z['finished_utc']) for z in reruns):errors.append(['policy failure lacks rerun record',x['utc']])
 memory=[x for x in quiet if x['event']=='memory_ready']
 if any(x['available_gib']<100 or x['other_heavy_jobs'] for x in memory):errors.append(['memory gate'])
 result=dict(commit=source['commit'],stages=stages,stage_order=list(state['stages']),quiet_starts=len(starts),concurrent_load_by_command=commands,full_screen_memory_checks=memory,cpu_policy=dict(expected=source['cpu_policy']['expected'],every_check_passed=allgood,scope='Accepted stage attempts and hash-verified clean GLM commands and the complete small/large-bulk blocks retained from inspected interrupted attempts; invalidated stages and commands remain in STAGE_RERUNS.json.',invalidated_checks=invalidated_policy_checks,periodic_checks=sum(x['event']=='policy_check' for x in policy),pre_command_checks=len(starts),segments=segments),reruns=reruns,command_reruns=command_reruns,problems=errors,load_note='Drivers use CPUs 0-5. Timed cores 6-21 have siblings 30-45 idle. Two ten-second quiet samples precede each command; selected-core unowned CPU and each sibling are limited to 5% over the command. Host and process snapshots and CPU deltas include other users and vLLM.')
 if result['stage_order']!=ORDER[:len(result['stage_order'])]:errors.append(['stage order'])
 dest=(W/'preprint-timing-audit-progress.json' if a.partial else R/'PREPRINT_TIMING_AUDIT.json') if a.preprint else (W/'timing-audit-progress.json' if a.partial else R/'TIMING_AUDIT.json');dest.write_text(json.dumps(result,indent=1)+'\n')
 print(json.dumps(dict(stages_complete=sum(x['complete'] for x in stages),commands=len(commands),problems=errors[:20]),indent=2))
 return bool(errors)
if __name__=='__main__':raise SystemExit(main())
