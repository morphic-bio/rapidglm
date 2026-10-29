#!/usr/bin/env python3
"""Check accepted stage exits, quiet starts, policy and per-command contention evidence."""
import argparse,datetime,gzip,hashlib,json
from pathlib import Path
from chain import ORDER,R,W
UNTIMED={'glmacc','edgerval','cooks','staging','seeds','compress'}
def load(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def lines(p):
 if not p.exists():p=Path(str(p)+'.gz')
 if not p.exists():return []
 text=gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text()
 return [json.loads(s) for s in text.splitlines() if s.strip()]
def ts(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 source=load(R/'source.json');state=load(R/'chain-status.json');policy=lines(R/'CPU_POLICY_AUDIT.jsonl')
 quiet=lines(R/'timing-audit.jsonl');allcommands=[]
 reruns=load(R/'STAGE_RERUNS.json')['reruns'] if (R/'STAGE_RERUNS.json').exists() else []
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
  stages.append(dict(stage=stage,timed=stage not in UNTIMED,complete=True,exits=[x['prior_exit'] for x in prior]+[0],attempts_queued=len(prior)+1,rerun_reasons=[x['reason'] for x in prior],quiet_starts=len(q),commands=len(c),start_load_range=[min(x['load_one_minute'] for x in q),max(x['load_one_minute'] for x in q)] if q else None,cpu_policy_checks=supervisor['cpu_policy_samples'],maximum_poll_gap_seconds=supervisor['cpu_policy_maximum_poll_gap_seconds'],slurm_job_id=row['slurm_job_id']))
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
 result=dict(commit=source['commit'],stages=stages,stage_order=list(state['stages']),quiet_starts=len(starts),concurrent_load_by_command=commands,full_screen_memory_checks=memory,cpu_policy=dict(expected=source['cpu_policy']['expected'],every_check_passed=allgood,scope='Accepted stage attempts; invalidated attempts are retained in STAGE_RERUNS.json.',invalidated_checks=invalidated_policy_checks,periodic_checks=sum(x['event']=='policy_check' for x in policy),pre_command_checks=len(starts),segments=segments),reruns=reruns,problems=errors,load_note='Drivers use CPUs 0-5. Timed cores 6-21 have siblings 30-45 idle. Two ten-second quiet samples precede each command; selected-core unowned CPU and each sibling are limited to 5% over the command. Host and process snapshots and CPU deltas include other users and vLLM.')
 if result['stage_order']!=ORDER[:len(result['stage_order'])]:errors.append(['stage order'])
 dest=W/'timing-audit-progress.json' if a.partial else R/'TIMING_AUDIT.json';dest.write_text(json.dumps(result,indent=1)+'\n')
 print(json.dumps(dict(stages_complete=sum(x['complete'] for x in stages),commands=len(commands),problems=errors[:20]),indent=2))
 return bool(errors)
if __name__=='__main__':raise SystemExit(main())
