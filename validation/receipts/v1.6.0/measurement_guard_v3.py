"""Per-command contention audit; the supervisor polls policy throughout."""
import json,os,time,resource
from pathlib import Path
from cpu_policy import R,EXPECTED,check,host_sample,utc
active=None

def precise_snapshot(cpus):
    """Read documented nanosecond scheduler counters; no kernel settings changed.
    https://docs.kernel.org/scheduler/sched-stats.html (CPU field 7).
    Expensive full process snapshots are kept outside these matched boundaries.
    """
    def usage():
        a=resource.getrusage(resource.RUSAGE_SELF)
        b=resource.getrusage(resource.RUSAGE_CHILDREN)
        return (a.ru_utime+a.ru_stime,b.ru_utime+b.ru_stime)
    affinity=set(os.sched_getaffinity(0));moved=bool(affinity & set(cpus))
    if moved:os.sched_setaffinity(0,set(range(6)))
    off_start=time.thread_time_ns()
    before=time.monotonic_ns();ua=usage();thread_before=time.thread_time_ns()
    runtimes={}
    with open('/proc/schedstat') as f:
        version=int(next(f).split()[1]);assert version>=15
        for line in f:
            v=line.split()
            if v and v[0].startswith('cpu') and v[0][3:].isdigit():
                runtimes[int(v[0][3:])]=int(v[7])
    irq={};busy_ticks={}
    with open('/proc/stat') as f:
        for line in f:
            v=line.split()
            if v and v[0].startswith('cpu') and v[0][3:].isdigit():
                # Retain IRQ/softirq/steal as a conservative extra activity term.
                irq[int(v[0][3:])]=sum(map(int,v[6:9]))
                values=list(map(int,v[1:9]));busy_ticks[int(v[0][3:])]=sum(values)-values[3]-values[4]
    thread_after=time.thread_time_ns();ub=usage();after=time.monotonic_ns()
    off_end=time.thread_time_ns()
    if moved:os.sched_setaffinity(0,affinity)
    assert runtimes and any(runtimes.values())
    return dict(monotonic_ns=(before+after)//2,read_span_ns=after-before,
                runtime_ns=runtimes,irq_ticks=irq,busy_ticks=busy_ticks,
                self_cpu_seconds=(ua[0]+ub[0])/2,children_cpu_seconds=(ua[1]+ub[1])/2,
                affinity=sorted(affinity),sampler_affinity=list(range(6)) if moved else sorted(affinity),
                off_cpu_before_usage_seconds=((thread_before+thread_after)/2-off_start)/1e9 if moved else 0,
                off_cpu_after_usage_seconds=(off_end-(thread_before+thread_after)/2)/1e9 if moved else 0,
                schedstat_version=version)

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
    active=dict(label=label,cpus=cpus,start=start,precise=precise_snapshot(cpus),external=external)

def finish():
    global active
    if active is None:return None
    item,active=active,None
    final=precise_snapshot(item['cpus']);initial=item['precise'];elapsed=(final['monotonic_ns']-initial['monotonic_ns'])/1e9;end=host_sample();cpus=item['cpus'];start=item['start']
    check(event='measurement_end',label=item['label'],cpus=sorted(cpus))
    hz=os.sysconf('SC_CLK_TCK')
    busy={c:(final['runtime_ns'][c]-initial['runtime_ns'][c])/1e9+(final['irq_ticks'][c]-initial['irq_ticks'][c])/hz for c in cpus|{c+24 for c in cpus}}
    # Compare independent accounting estimates at the scope of the predicate.
    # Selected-core activity is an aggregate predicate: sum each counter first,
    # then take the maximum. Per-core maxima would select each counter's upward
    # quantization error and inflate the sum for short parallel commands.
    selected_sched_seconds=sum(busy[c] for c in cpus)
    selected_tick_seconds=sum(final['busy_ticks'][c]-initial['busy_ticks'][c] for c in cpus)/hz
    selected_busy_seconds=selected_sched_seconds
    if len(cpus)>1 or elapsed>=.5:
        selected_busy_seconds=max(selected_sched_seconds,selected_tick_seconds)
    # Siblings are individually limited; retain the independent tick estimate
    # there to detect live tasks whose scheduler slices have not completed.
    for c in busy:
        if c not in cpus:
            busy[c]=max(busy[c],(final['busy_ticks'][c]-initial['busy_ticks'][c])/hz)
    external=json.loads(item['external'].read_text()) if item['external'].exists() else None
    external_cpu=external['cpu_seconds'] if external else 0.
    own=final['children_cpu_seconds']-initial['children_cpu_seconds']+external_cpu
    parent_on_timed=set(initial['affinity'])<=cpus and set(final['affinity'])<=cpus
    if parent_on_timed:own+=final['self_cpu_seconds']-initial['self_cpu_seconds']-initial['off_cpu_after_usage_seconds']-final['off_cpu_before_usage_seconds']
    other=max(0,selected_busy_seconds-own)/max(elapsed*len(cpus),1e-6)
    siblings={str(c+24):busy[c+24]/max(elapsed,1e-6) for c in cpus}
    contended=other>.05 or any(v>.05 for v in siblings.values())
    old_processes={p['pid']:p for p in start.get('processes',[])}
    process_cpu=[]
    for p in end.get('processes',[]):
        before=old_processes.get(p['pid'])
        if before and before['started_ticks']==p['started_ticks']:
            ticks=p['ticks']-before['ticks']
            if ticks>0:process_cpu.append(dict(pid=p['pid'],user=p['user'],comm=p['comm'],cpu_seconds=ticks/hz))
    row=dict(pid=os.getpid(),process_cpu_seconds=process_cpu,stage=os.environ.get('RAPIDGLM_STAGE'),label=item['label'],division=os.environ.get('RAPIDGLM_DIVISION','slow'),cpus=sorted(cpus),started_utc=start['utc'],finished_utc=end['utc'],wall_seconds=elapsed,command_cpu_seconds=own,container_cpu_audit=external,accounting_start=initial,accounting_end=final,selected_sched_seconds=selected_sched_seconds,selected_tick_seconds=selected_tick_seconds,selected_busy_seconds=selected_busy_seconds,parent_cpu_counted=parent_on_timed,accounting_method='schedstat nanosecond task runtime plus conservative IRQ/softirq/steal ticks; matched getrusage CPU interval; full process snapshots outside the accounting interval',others_cpu=other,sibling_busy_fraction=siblings,contended=contended,policy_unchanged=True,host_start=start,host_end=end,slurm_job_id=os.environ['SLURM_JOB_ID'])
    directory=Path(os.environ.get('RAPIDGLM_COMMAND_AUDIT_DIR',str(R/'commands')));directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{os.getpid()}-{time.time_ns()}.json';path.write_text(json.dumps(row,indent=1)+'\n')
    if contended:raise RuntimeError('measurement exceeded 5% contention; rerun required: '+str(path))
    return row
