"""Per-command contention audit; the supervisor polls policy throughout."""
import json,os,time
from pathlib import Path
from cpu_policy import R,EXPECTED,check,host_sample,utc
active=None

def begin(cpus,label):
    global active
    if active is not None:raise RuntimeError('previous measurement was not closed')
    cpus=set(cpus)
    if not cpus or not cpus<=set(range(6,22)):raise RuntimeError('timed cores must be within 6-21')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('missing Slurm allocation')
    if os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')!='1':raise RuntimeError('policy supervisor required')
    check(event='measurement_begin',label=label,cpus=sorted(cpus))
    start=host_sample()
    directory=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands')));directory.mkdir(parents=True,exist_ok=True)
    external=directory/f'external-cpu-{os.getpid()}-{time.time_ns()}.json'
    os.environ['RAPIDGLM_EXTERNAL_CPU_AUDIT']=str(external)
    active=dict(label=label,cpus=cpus,start=start,usage=os.times(),clock=time.perf_counter(),external=external)

def finish():
    global active
    if active is None:return None
    item,active=active,None
    elapsed=time.perf_counter()-item['clock'];usage=os.times();end=host_sample();cpus=item['cpus'];start=item['start']
    check(event='measurement_end',label=item['label'],cpus=sorted(cpus))
    hz=os.sysconf('SC_CLK_TCK')
    busy={c:(end['cpus'][str(c)][1]-start['cpus'][str(c)][1])/hz for c in cpus|{c+24 for c in cpus}}
    external=json.loads(item['external'].read_text()) if item['external'].exists() else None
    external_cpu=external['cpu_seconds'] if external else 0.
    own=sum(usage[:4])-sum(item['usage'][:4])+external_cpu;other=max(0,sum(busy[c] for c in cpus)-own)/max(elapsed*len(cpus),1e-6)
    siblings={str(c+24):busy[c+24]/max(elapsed,1e-6) for c in cpus}
    contended=other>.05 or any(v>.05 for v in siblings.values())
    old_processes={p['pid']:p for p in start.get('processes',[])}
    process_cpu=[]
    for p in end.get('processes',[]):
        before=old_processes.get(p['pid'])
        if before and before['started_ticks']==p['started_ticks']:
            ticks=p['ticks']-before['ticks']
            if ticks>0:process_cpu.append(dict(pid=p['pid'],user=p['user'],comm=p['comm'],cpu_seconds=ticks/hz))
    row=dict(pid=os.getpid(),process_cpu_seconds=process_cpu,stage=os.environ.get('RAPIDGLM_STAGE'),label=item['label'],division=os.environ.get('RAPIDGLM_DIVISION','slow'),cpus=sorted(cpus),started_utc=start['utc'],finished_utc=end['utc'],wall_seconds=elapsed,command_cpu_seconds=own,container_cpu_audit=external,others_cpu=other,sibling_busy_fraction=siblings,contended=contended,policy_unchanged=True,host_start=start,host_end=end,slurm_job_id=os.environ['SLURM_JOB_ID'])
    directory=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands')));directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{os.getpid()}-{time.time_ns()}.json';path.write_text(json.dumps(row,indent=1)+'\n')
    if contended:raise RuntimeError('measurement exceeded 5% contention; rerun required: '+str(path))
    return row
