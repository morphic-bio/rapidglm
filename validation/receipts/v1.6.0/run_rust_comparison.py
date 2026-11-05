#!/usr/bin/env python3
"""Preserve the Rust comparison calculations; align guard affinity and retry contention."""
import functools,json,os,sys,time,hashlib,subprocess,tempfile,uuid
from pathlib import Path
from retry_bulk_validation import install_hook,save,sha
ROOT=Path('/data/lhhung/rapidglm-1.6.0');W=ROOT/'validation/work/release-1.6.0-gpu';R=ROOT/'validation/receipts/v1.6.0'

def retry_call(fn,record_path=None,reruns_path=None,archive=None,commit=None):
 record_path=record_path or R/'RUST_COMMAND_RETRIES.json'
 reruns_path=reruns_path or R/'STAGE_RERUNS.json'
 archive=archive or W/'rust-live-workers-9/invalidated'
 commit=commit or json.loads((R/'source.json').read_text())['commit']
 @functools.wraps(fn)
 def wrapped(*args):
  assert os.environ.get('SLURM_JOB_ID') and os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')=='1'
  affinity=set(os.sched_getaffinity(0));assert affinity<=set(range(6))
  expected='Rust comparison '+Path(args[0]).name+(' PyDESeq2' if fn.__name__=='pydeseq2' else ' native')
  while True:
   began=time.time()
   try:
    result=fn(*args)
    destination=R/'rust_tables'/Path(args[0]).name;destination.mkdir(parents=True,exist_ok=True)
    if fn.__name__=='pydeseq2':result[0].to_csv(destination/'pydeseq2.tsv',sep='\t',index=False)
    else:
     source=Path(args[2])/'results.tsv';target=destination/'native.tsv';target.write_bytes(source.read_bytes());assert sha(target)==sha(source)
    return result
   except RuntimeError as exc:
    marker='measurement exceeded 5% contention; rerun required: '
    if not str(exc).startswith(marker):raise
    p=Path(str(exc)[len(marker):]).resolve();folder=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands'))).resolve()
    if p.parent!=folder or not p.name.startswith(str(os.getpid())+'-'):raise
    row=json.loads(p.read_text())
    if p.stat().st_mtime<began or row['label']!=expected or row['pid']!=os.getpid() or row['stage']!=os.environ['RAPIDGLM_STAGE'] or row['slurm_job_id']!=os.environ['SLURM_JOB_ID'] or not row['contended'] or not row['policy_unchanged']:raise
    target=archive/(os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()));target.mkdir(parents=True,exist_ok=False)
    evidence=[]
    if fn.__name__=='pydeseq2':
     tb=exc.__traceback__
     while tb is not None and tb.tb_frame.f_code is not fn.__code__:tb=tb.tb_next
     assert tb is not None and 'stats' in tb.tb_frame.f_locals
     output=target/'results.tsv';tb.tb_frame.f_locals['stats'].results_df.to_csv(output,sep='\t',index=False)
     evidence.append(dict(copy=str(output.relative_to(ROOT)),sha256=sha(output),origin='Public results_df of the rejected own comparison call'))
    else:
     for source in Path(args[2]).iterdir():
      if not source.is_file():continue
      output=target/source.name;raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest();output.write_bytes(raw);assert sha(output)==h
      evidence.append(dict(copy=str(output.relative_to(ROOT)),sha256=h,origin='Rejected native comparison temporary output'))
    assert evidence
    rec_identity=__import__('rust_live_cpu').descendants()
    rec=dict(stage=os.environ['RAPIDGLM_STAGE'],commit=commit,label=expected,slurm_job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),started_utc=row['started_utc'],finished_utc=row['finished_utc'],audit_file=os.path.relpath(p,R),audit_sha256=sha(p),archive_files=evidence,others_cpu=row['others_cpu'],sibling_busy_fraction=row['sibling_busy_fraction'],reason='Existing unchanged guard rejected this exact Rust comparison call. Rejected output is retained. PyDESeq2 restarts in a fresh interpreter with new workers; native retries its original fresh subprocess after the complete quiet check.',attempt_id=os.environ.get('RAPIDGLM_RUST_ATTEMPT_ID'),retry_strategy='fresh_interpreter' if fn.__name__=='pydeseq2' else 'fresh_native_subprocess',live_descendants_at_rejection=rec_identity)
    proof=json.loads(record_path.read_text()) if record_path.exists() else dict(commit=commit,retries=[])
    assert proof['commit']==commit;proof['retries'].append(rec);save(record_path,proof)
    reruns=json.loads(reruns_path.read_text()) if reruns_path.exists() else dict(commit=commit,reruns=[],command_reruns=[])
    assert reruns['commit']==commit;reruns.setdefault('command_reruns',[]).append(rec);save(reruns_path,reruns)
    print(json.dumps(dict(event='rust_command_retry',label=expected,audit=rec['audit_file'])),flush=True)
    if fn.__name__=='pydeseq2':
     restart=Path(os.environ['RAPIDGLM_RUST_RESTART_RECORD'])
     save(restart,dict(record_path=str(record_path),reruns_path=str(reruns_path),audit_sha256=rec['audit_sha256'],attempt_id=rec['attempt_id'],descendants=rec_identity))
     raise SystemExit(75)
   finally:os.sched_setaffinity(0,affinity)
 return wrapped

def load_benchmark(factory=retry_call,entry=__file__):
 install_hook(entry)
 from rust_live_cpu import install
 install()
 path=ROOT/'bindings/rust/tests/compare_pydeseq2.py';text=path.read_text()
 anchor='        wait_quiet(cpus, label=f"Rust comparison {data.name} PyDESeq2")'
 assert text.count(anchor)==1
 # Data loading remains on driver CPUs. Set the already specified timed affinity
 # before the guard begins; the original timed interval and calculations are unchanged.
 text=text.replace(anchor,'        os.sched_setaffinity(0, cpus)\n'+anchor,1)
 scope=dict(__name__='measured_rust_comparison',__file__=str(path))
 exec(compile(text,str(path),'exec'),scope)
 scope['pydeseq2']=factory(scope['pydeseq2']);scope['rust']=factory(scope['rust'])
 return scope
def coordinate(command):
 from rust_live_cpu import identity
 assert set(os.sched_getaffinity(0))<=set(range(6))
 with tempfile.TemporaryDirectory(prefix='rust-restart-',dir=W) as work:
  restart=Path(work)/'restart.json'
  while True:
   attempt=uuid.uuid4().hex
   env=dict(os.environ,RAPIDGLM_RUST_ATTEMPT_ID=attempt,RAPIDGLM_RUST_RESTART_RECORD=str(restart))
   result=subprocess.run(command,env=env)
   if result.returncode!=75:return result.returncode
   item=json.loads(restart.read_text());assert item['attempt_id']==attempt
   deadline=time.monotonic()+10
   while any((current:=identity(p['pid'])) is not None and current['start_ticks']==p['start_ticks'] for p in item['descendants']):
    if time.monotonic()>deadline:raise RuntimeError('rejected interpreter left live owned workers; restart refused')
    time.sleep(.05)
   cleanup=dict(rejected_interpreter_exit=75,all_rejected_descendants_gone=True,attempt_id=attempt,verified_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
   for key,field in [('record_path','retries'),('reruns_path','command_reruns')]:
    p=Path(item[key]);proof=json.loads(p.read_text())
    matches=[x for x in proof[field] if x.get('audit_sha256')==item['audit_sha256']]
    assert len(matches)==1;matches[0]['fresh_process_cleanup']=cleanup;save(p,proof)
   restart.unlink()

def main():
 if os.environ.get('RAPIDGLM_RUST_ATTEMPT_ID'):return load_benchmark()['main']()
 return coordinate([sys.executable,__file__,*sys.argv[1:]])
if __name__=='__main__':raise SystemExit(main())
