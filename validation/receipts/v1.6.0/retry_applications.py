#!/usr/bin/env python3
"""Run the unchanged application benchmarks; retry only audited contention rejections."""
import datetime,functools,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/data/lhhung/rapidglm-1.6.0');W=ROOT/'validation/work/release-1.6.0-gpu';R=ROOT/'validation/receipts/v1.6.0'
from retry_bulk_validation import install_hook,save,sha

def retry_applications(fn,record_path=None,reruns_path=None,archive=None,commit=None):
 record_path=record_path or R/'APPLICATION_COMMAND_RETRIES.json'
 reruns_path=reruns_path or R/'STAGE_RERUNS.json'
 archive=archive or W/'application-retries-6/invalidated'
 commit=commit or json.loads((R/'source.json').read_text())['commit']
 @functools.wraps(fn)
 def wrapped(run,threads,args,*arguments):
  assert os.environ.get('SLURM_JOB_ID') and os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')=='1'
  expected=run.__name__+' threads='+str(threads)
  output=Path(arguments[-1]).resolve()
  assert output.is_relative_to(ROOT/'validation/work') and output.is_dir()
  while True:
   began=time.time()
   try:return fn(run,threads,args,*arguments)
   except RuntimeError as exc:
    marker='measurement exceeded 5% contention; rerun required: '
    if not str(exc).startswith(marker):raise
    p=Path(str(exc)[len(marker):]).resolve()
    folder=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands'))).resolve()
    if p.parent!=folder or not p.name.startswith(str(os.getpid())+'-'):raise
    row=json.loads(p.read_text())
    if p.stat().st_mtime<began or row['label']!=expected or row['pid']!=os.getpid() or row['stage']!=os.environ['RAPIDGLM_STAGE'] or row['slurm_job_id']!=os.environ['SLURM_JOB_ID'] or not row['contended'] or not row['policy_unchanged']:raise
    target=archive/(os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()));target.mkdir(parents=True,exist_ok=False)
    copies=[]
    for source in sorted(output.rglob('*')):
     if not source.is_file():continue
     to=target/source.relative_to(output);to.parent.mkdir(parents=True,exist_ok=True)
     raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest();to.write_bytes(raw);assert sha(to)==h
     copies.append(dict(original=str(source.relative_to(ROOT)),copy=str(to.relative_to(ROOT)),sha256=h))
    assert copies,'rejected application command did not leave an output to archive'
    rec=dict(stage=os.environ['RAPIDGLM_STAGE'],commit=commit,label=expected,slurm_job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),started_utc=row['started_utc'],finished_utc=row['finished_utc'],audit_file=os.path.relpath(p,R),audit_sha256=sha(p),archive_files=copies,others_cpu=row['others_cpu'],sibling_busy_fraction=row['sibling_busy_fraction'],reason='Unchanged guard rejected this application measurement. Outputs retained by SHA-256; repeat the identical original timed_run call after its complete original quiet predicate.')
    proof=json.loads(record_path.read_text()) if record_path.exists() else dict(commit=commit,retries=[])
    assert proof['commit']==commit;proof['retries'].append(rec);save(record_path,proof)
    reruns=json.loads(reruns_path.read_text()) if reruns_path.exists() else dict(commit=commit,reruns=[],command_reruns=[])
    assert reruns['commit']==commit;reruns.setdefault('command_reruns',[]).append(rec);save(reruns_path,reruns)
    print(json.dumps(dict(event='application_command_retry',label=expected,audit=rec['audit_file'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
 return wrapped

def main():
 install_hook(__file__)
 path=ROOT/'scripts/benchmark_applications.py';text=path.read_text();anchor='if __name__ == "__main__":'
 assert text.count(anchor)==1
 text=text.replace(anchor,'timed_run = __retry_applications(timed_run)\n\n'+anchor,1)
 scope=dict(__name__='__main__',__file__=str(path),__retry_applications=retry_applications)
 exec(compile(text,str(path),'exec'),scope)
if __name__=='__main__':main()
