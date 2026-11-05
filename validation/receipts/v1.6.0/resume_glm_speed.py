#!/usr/bin/env python3
"""Resume verified GLM speed commands; retry only measured CPU contention."""
import argparse,datetime,fcntl,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/data/lhhung/rapidglm-1.6.0');R=ROOT/'validation/receipts/v1.6.0';W=ROOT/'validation/work/release-1.6.0-gpu'
# Register only the two new driver entry points with the existing synchronous
# policy hook; keep its implementation and the numerical source unchanged.
sys.path.insert(0,str(R/'policy_hook'))
import sitecustomize
sitecustomize.ENTRIES.update({R/'resume_glm_speed.py',R/'smoke_glm_resume.py'})
sitecustomize.install()
sys.path.insert(0,str(ROOT/'benchmarks/glm'));import run_comparison as original
sys.path.insert(0,str(ROOT/'scripts'));from quiet_run import LOCK,wait_quiet,finish_measurement
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(p,d):
 temporary=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 temporary.write_text(json.dumps(d,indent=2)+'\n');temporary.replace(p)
def check_result(row):
 assert sha(ROOT/row['result_file'])==row['result_sha256'],row['label']
 assert sha(R/row['audit_file'])==row['audit_sha256'],row['label']
 d=json.loads((R/row['audit_file']).read_text())
 assert d['label']==row['label'] and d['pid']==row['pid'] and d['slurm_job_id']==row['slurm_job_id']
 assert d['cpus']==[6] and d['division']=='slow'
 assert not d['contended'] and d['policy_unchanged'] and d['others_cpu']<=.05 and max(d['sibling_busy_fraction'].values())<=.05
def latest_audit(label):
 folder=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands')))
 p=max(folder.glob(str(os.getpid())+'-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
 d=json.loads(p.read_text());assert d['label']==label
 return p,d
def archive_output(output,directory):
 directory.mkdir(parents=True,exist_ok=True);target=directory/output.name
 assert not target.exists();raw=output.read_bytes();h=hashlib.sha256(raw).hexdigest()
 target.write_bytes(raw);assert sha(target)==h;output.unlink()
 return dict(archive_result=str(target.relative_to(ROOT)),result_sha256=h)
def execute(state,path,reruns_path,archive,job):
 label=job['label']
 if label in state['accepted']:
  check_result(state['accepted'][label]);return False
 problem=ROOT/job['problem'];output=problem/('result-'+job['tool']+'.json')
 assert not output.exists(),'Unverified output needs inspection before overwrite: '+str(output)
 while True:
  os.sched_setaffinity(0,{6})
  wait_quiet({6},label=label)
  original.run_tool(job['tool'],problem,15.,1800,'')
  try:
   row=finish_measurement();assert row and row['label']==label
  except RuntimeError:
   p,row=latest_audit(label)
   if not row['contended'] or not row['policy_unchanged']:raise
   os.sched_setaffinity(0,set(range(6)))
   rec=dict(stage='glmspeed',commit=state['commit'],label=label,reason='Measured CPU contention exceeded the unchanged 5% selected-core/sibling limit.',audit_file=os.path.relpath(p,R),audit_sha256=sha(p),result_file=str(output.relative_to(ROOT)),slurm_job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),started_utc=row['started_utc'],finished_utc=row['finished_utc'],others_cpu=row['others_cpu'],sibling_busy_fraction=row['sibling_busy_fraction'])
   rec.update(archive_output(output,archive/(os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()))))
   state['invalidated'].append(rec);save(path,state)
   reruns=json.loads(reruns_path.read_text()) if reruns_path.exists() else dict(reruns=[],command_reruns=[])
   reruns.setdefault('command_reruns',[]).append(rec);save(reruns_path,reruns)
   print(json.dumps(dict(event='retry_required',label=label,audit=rec['audit_file'],utc=utc())),flush=True)
   continue
  os.sched_setaffinity(0,set(range(6)))
  p,record=latest_audit(label)
  assert record==json.loads(json.dumps(row)) and not row['contended'] and row['policy_unchanged']
  accepted=dict(label=label,audit_file=os.path.relpath(p,R),audit_sha256=sha(p),result_file=str(output.relative_to(ROOT)),result_sha256=sha(output),slurm_job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),started_utc=row['started_utc'],finished_utc=row['finished_utc'])
  check_result(accepted);state['accepted'][label]=accepted;save(path,state)
  print(json.dumps(dict(event='accepted',label=label,completed=len(state['accepted']),total=len(state['jobs']),utc=utc())),flush=True)
  return True
def main():
 assert os.environ.get('SLURM_JOB_ID') and os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')=='1'
 assert set(os.sched_getaffinity(0))<=set(range(6))
 path=R/'GLM_SPEED_RESUME.json';state=json.loads(path.read_text())
 source=json.loads((R/'source.json').read_text())
 assert state['commit']==source['commit']
 spec_path=W/'glm_speed_jobs.json'
 assert sha(spec_path)==source['cpu_policy']['command_resume']['job_manifest_sha256']
 specification=json.loads(spec_path.read_text())
 assert all(state[k]==specification[k] for k in ['commit','jobs','input_sha256'])
 for name,value in state['input_sha256'].items():assert sha(ROOT/name)==value,name
 for row in state['accepted'].values():check_result(row)
 assert len({x['label'] for x in state['jobs']})==len(state['jobs'])
 with open(LOCK,'a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for job in state['jobs']:execute(state,path,R/'STAGE_RERUNS.json',W/'command-resume-2/invalidated',job)
 assert set(state['accepted'])=={x['label'] for x in state['jobs']}
 for name,value in state['input_sha256'].items():assert sha(ROOT/name)==value,name
 state.update(complete=True,finished_utc=utc());save(path,state);save(R/'GLM_SPEED_COMPLETION.json',state)
 print(json.dumps(dict(event='complete',commands=len(state['accepted']),invalidated=len(state['invalidated']))),flush=True)
if __name__=='__main__':main()
