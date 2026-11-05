#!/usr/bin/env python3
"""Run all 24 stages in order under the approved exclusive allocation."""
import datetime,json,os,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2];W=ROOT/'validation/work/release-1.6.0-gpu'
ORDER=['glmacc','glmspeed','batch','solvers','solvers_control','bulkval','edgerval','bulk','cooks','rust','staging','pert50k','pertfull','seeds','untestable','crt','replogle','qtl','special','vmath_e2e','ablations','replogle_full','eqtl_gw','compress']
P=R/'chain-status.json'
STATUS=Path('<status-work>/STATUS.md')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def milestone(text):
 with STATUS.open('a') as f:f.write(utc()+' | '+text+'\n')
def save(state):
 temp=P.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(P)
def main():
 assert os.environ.get('SLURM_JOB_ID'),'Slurm allocation required'
 assert set(os.sched_getaffinity(0))<=set(range(6))
 job=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
 assert 'NumCPUs=48' in job and 'OverSubscribe=NO' in job,'exclusive allocation on all logical CPUs required'
 subprocess.run(['python3',str(R/'verify_source.py')],cwd=ROOT,check=True)
 source=json.loads((R/'source.json').read_text());state=json.loads(P.read_text()) if P.exists() else dict(commit=source['commit'],stages={})
 assert state['commit']==source['commit']
 from retry_stage import retry_untestable
 for stage in ORDER:
  if state['stages'].get(stage,{}).get('exit')==0:continue
  if stage in state['stages']:raise RuntimeError('failed stage needs an inspected rerun record: '+stage)
  while True:
   label=stage+'-'+str(time.time_ns());row=dict(started_utc=utc(),label=label,slurm_job_id=os.environ['SLURM_JOB_ID'],exit=None)
   state['stages'][stage]=row;save(state);milestone('N: '+stage+' started under exclusive Slurm job '+os.environ['SLURM_JOB_ID']+'.')
   with (W/'logs'/('stage-'+label+'.log')).open('w') as log:
    result=subprocess.run(['python3',str(R/'supervise_accepted.py'),label,stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row.update(exit=result.returncode,finished_utc=utc());save(state);milestone('N: '+stage+' exit '+str(result.returncode)+'.')
   if not result.returncode:break
   subprocess.run(['python3',str(R/'verify_source.py')],cwd=ROOT,check=True)
   if stage=='untestable' and retry_untestable(row,source['commit']):
    milestone('N: untestable rejected attempt archived and SHA-256 verified; repeating unchanged stage under the same allocation.')
    del state['stages'][stage];save(state);continue
   return result.returncode
  with (W/'logs'/('audit-after-'+stage+'.log')).open('w') as log:
   result=subprocess.run(['python3',str(R/'check_run_accepted.py'),'--partial'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('incremental acceptance audit failed after '+stage)
  with (W/'logs'/('agreement-after-'+stage+'.log')).open('w') as log:
   result=subprocess.run(['python3',str(R/'agreement_job687.py'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('scientific agreement check failed after '+stage)
 state['finished_utc']=utc();save(state);return 0
if __name__=='__main__':
 try:code=main()
 except BaseException:
  traceback.print_exc();code=1
 (W/'chain.exit').write_text(str(code)+'\n')
 milestone('N: detached chain '+('completed; assembly required.' if not code else 'stopped; inspect stage and audit logs before any rerun.'))
 raise SystemExit(code)
