#!/usr/bin/env python3
"""Run the unchanged bulk validator, retrying only guarded run_bulk rejections."""
import datetime,functools,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/data/lhhung/rapidglm-1.6.0');W=ROOT/'validation/work/release-1.6.0-gpu';R=ROOT/'validation/receipts/v1.6.0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 temporary=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 temporary.write_text(json.dumps(d,indent=2)+'\n');temporary.replace(p)
def install_hook(entry):
 sys.path.insert(0,str(R/'policy_hook'))
 import sitecustomize
 sitecustomize.ENTRIES.add(Path(entry).resolve());sitecustomize.install()
def retry_bulk(fn,record_path=None,reruns_path=None,archive=None,commit=None):
 record_path=record_path or R/'BULK_COMMAND_RETRIES.json'
 reruns_path=reruns_path or R/'STAGE_RERUNS.json'
 archive=archive or W/'bulk-command-retries-5/invalidated'
 commit=commit or json.loads((R/'source.json').read_text())['commit']
 @functools.wraps(fn)
 def wrapped(binary,d,coefficient,threads,prefix,**extra):
  assert os.environ.get('SLURM_JOB_ID') and os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')=='1'
  prefix=Path(prefix);expected='bulk validation '+prefix.name+' threads='+str(threads)
  while True:
   began=time.time()
   try:return fn(binary,d,coefficient,threads,prefix,**extra)
   except RuntimeError:
    folder=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands')))
    matches=list(folder.glob(str(os.getpid())+'-*.json'))
    if not matches:raise
    p=max(matches,key=lambda p:p.stat().st_mtime_ns)
    row=json.loads(p.read_text())
    if p.stat().st_mtime<began or row['label']!=expected or not row['contended'] or not row['policy_unchanged']:raise
    target=archive/(os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()));target.mkdir(parents=True,exist_ok=False)
    copies=[]
    for output in sorted(prefix.parent.glob(prefix.name+'.*')):
     if not output.is_file():continue
     to=target/output.name;raw=output.read_bytes();h=hashlib.sha256(raw).hexdigest()
     to.write_bytes(raw);assert sha(to)==h
     copies.append(dict(original=str(output.relative_to(ROOT)),copy=str(to.relative_to(ROOT)),sha256=h))
    assert copies
    rec=dict(stage='bulkval',commit=commit,label=expected,slurm_job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),started_utc=row['started_utc'],finished_utc=row['finished_utc'],audit_file=os.path.relpath(p,R),audit_sha256=sha(p),archive_files=copies,others_cpu=row['others_cpu'],sibling_busy_fraction=row['sibling_busy_fraction'],reason='The unchanged contention predicate rejected this measurement, including conservative counter uncertainty. Its output is archived; retry the identical run_bulk call after its full quiet predicate.')
    proof=json.loads(record_path.read_text()) if record_path.exists() else dict(commit=commit,retries=[])
    assert proof['commit']==commit;proof['retries'].append(rec);save(record_path,proof)
    reruns=json.loads(reruns_path.read_text()) if reruns_path.exists() else dict(commit=commit,reruns=[],command_reruns=[])
    reruns.setdefault('command_reruns',[]).append(rec);save(reruns_path,reruns)
    print(json.dumps(dict(event='bulk_command_retry',label=expected,audit=rec['audit_file'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
 return wrapped
def main():
 install_hook(__file__)
 path=ROOT/'scripts/validate_bulk.py';text=path.read_text()
 anchor='if __name__ == "__main__":'
 assert text.count(anchor)==1
 text=text.replace(anchor,'run_bulk = __retry_bulk(run_bulk)\n\n'+anchor,1)
 scope=dict(__name__='__main__',__file__=str(path),__retry_bulk=retry_bulk)
 exec(compile(text,str(path),'exec'),scope)
if __name__=='__main__':main()
