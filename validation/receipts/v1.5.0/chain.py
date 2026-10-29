#!/usr/bin/env python3
"""Resume the ordered release measurements under one exclusive Slurm allocation."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
W=ROOT/'validation/work/release-1.5.0-gpu'
ORDER=['glmacc','glmspeed','batch','solvers','solvers_control','bulkval','edgerval','bulk','cooks','rust','staging','pert50k','pertfull','seeds','untestable','crt','replogle','qtl','special','vmath_e2e','replogle_full','eqtl_gw','compress']
P=R/'chain-status.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def milestone(text):
 with Path(os.environ.get('RAPIDGLM_PROGRESS_FILE',str(W/'STATUS.md'))).open('a') as f:f.write(utc()+' | '+text+'\n')
def main():
 assert os.environ.get('SLURM_JOB_ID'),'Slurm allocation required'
 assert set(os.sched_getaffinity(0))<=set(range(6))
 job=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
 assert 'NumCPUs=48' in job and 'OverSubscribe=NO' in job,'exclusive allocation on all logical CPUs required'
 identity=json.loads((R/'GASPERINI_IDENTITY.json').read_text())
 assert identity['byte_identical'] and identity['instance_terminated'],'cloud identity gate not passed'
 source=json.loads((R/'source.json').read_text())
 assert identity['measured_commit']==source['commit']
 state=json.loads(P.read_text()) if P.exists() else dict(commit=source['commit'],stages={})
 assert state['commit']==source['commit']
 for stage in ORDER:
  if state['stages'].get(stage,{}).get('exit')==0:continue
  if stage in state['stages']:raise RuntimeError('failed stage needs an inspected rerun record: '+stage)
  label=stage+'-'+str(time.time_ns())
  row=dict(started_utc=utc(),label=label,slurm_job_id=os.environ['SLURM_JOB_ID'],exit=None)
  state['stages'][stage]=row;P.write_text(json.dumps(state,indent=2)+'\n');milestone('H: '+stage+' started under exclusive Slurm job '+os.environ['SLURM_JOB_ID']+'.')
  with (W/'logs'/('stage-'+label+'.log')).open('w') as log:
   result=subprocess.run(['python3',str(R/'supervise.py'),label,stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  row.update(exit=result.returncode,finished_utc=utc());P.write_text(json.dumps(state,indent=2)+'\n')
  milestone('H: '+stage+' exit '+str(result.returncode)+'.')
  if result.returncode:return result.returncode
 state['finished_utc']=utc();P.write_text(json.dumps(state,indent=2)+'\n');return 0
if __name__=='__main__':
 code=main();(W/'chain.exit').write_text(str(code)+'\n');raise SystemExit(code)
