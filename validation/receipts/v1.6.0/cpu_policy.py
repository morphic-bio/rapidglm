#!/usr/bin/env python3
"""Read-only CPU policy, process and load audit for biocloudgpu1."""
import datetime,fcntl,hashlib,json,os,pwd,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
PCORES=range(0,6)
ECORES=range(6,22)
EXPECTED=json.loads((R/'CPU_POLICY_EXPECTED.json').read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_policy():
    root=Path('/sys/devices/system/cpu')
    d={k:(root/k).read_text().strip() for k in ['online','cpufreq/boost','smt/control','smt/active']}
    d['cpus']={str(c):{k:(root/f'cpu{c}/cpufreq'/k).read_text().strip() for k in ['scaling_driver','scaling_governor','scaling_min_freq','scaling_max_freq']} for c in range(48)}
    d['siblings']={str(c):(root/f'cpu{c}/topology/thread_siblings_list').read_text().strip() for c in range(48)}
    return d
def processes():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text();end=s.rfind(')');v=s[end+2:].split();comm=s[s.index('(')+1:end]
            result.append(dict(pid=int(p.name),user=pwd.getpwuid(p.stat().st_uid).pw_name,comm=comm,ppid=int(v[1]),ticks=int(v[11])+int(v[12]),cpu=int(v[36]),rss_bytes=int(v[21])*os.sysconf('SC_PAGESIZE'),started_ticks=int(v[19])))
        except (OSError,ValueError,KeyError,IndexError):pass
    return result
def stop_owned(child):
    # Children started by this controller, and explicitly labelled containers
    # started by this Slurm job. Never use a process-group or name-based kill.
    items=processes();roots={p['pid'] for p in items if p['pid']==child.pid and p['ppid']==os.getpid()}
    job=os.environ.get('SLURM_JOB_ID')
    if job:
        found=subprocess.run(['docker','ps','-q','--filter','label=rapidglm.slurm_job='+job],capture_output=True,text=True)
        for container in found.stdout.split():
            result=subprocess.run(['docker','inspect','--format','{{.State.Pid}}',container],capture_output=True,text=True)
            if result.returncode==0 and result.stdout.strip().isdigit():roots.add(int(result.stdout))
    parents=set(roots)
    while True:
        more={p['pid'] for p in items if p['ppid'] in parents}-parents
        if not more:break
        parents|=more
    targets=[p for p in items if p['pid'] in parents]
    for sig in (signal.SIGTERM,signal.SIGKILL):
        live={p['pid']:p for p in processes()}
        for p in reversed(targets):
            q=live.get(p['pid'])
            if q and q['started_ticks']==p['started_ticks']:
                try:os.kill(p['pid'],sig)
                except ProcessLookupError:pass
                except PermissionError:
                    subprocess.run(['sudo','-n','kill','-'+str(int(sig)),str(p['pid'])],check=False,capture_output=True)
        if sig==signal.SIGTERM:time.sleep(1)

def digest(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
def emit(event,**values):
    path=Path(os.environ.get('RAPIDGLM_CPU_POLICY_AUDIT',str(R/'CPU_POLICY_AUDIT.jsonl')))
    row=dict(event=event,utc=utc(),pid=os.getpid(),**values)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,separators=(',',':'))+'\n')
    return row
def check(expected=EXPECTED,event='policy_check',**context):
    actual=read_policy();identifier=f'{os.getpid()}:{time.monotonic_ns()}'
    row=emit(event,check_id=identifier,matched=actual==expected,policy_sha256=digest(actual),**context)
    if actual!=expected:
        (R/'CPU_POLICY_VIOLATION.json').write_text(json.dumps(dict(row,expected=expected,actual=actual),indent=2)+'\n')
        raise RuntimeError('CPU policy changed; no settings were changed by this runner')
    return identifier
def host_sample():
    sums={'p_busy':0,'p_total':0,'e_busy':0,'e_total':0,'s_busy':0,'s_total':0};cpus={}
    with open('/proc/stat') as f:
        for line in f:
            name,*values=line.split()
            if name.startswith('cpu') and name[3:].isdigit():
                cpu=int(name[3:]);v=list(map(int,values[:8]));total=sum(v);busy=total-v[3]-v[4]
                cpus[str(cpu)]=[total,busy]
                kind='p' if cpu in PCORES else 'e' if cpu in ECORES else 's'
                sums[kind+'_busy']+=busy;sums[kind+'_total']+=total
    return dict(t=time.time(),utc=utc(),load1=os.getloadavg()[0],load=os.getloadavg(),cpus=cpus,processes=processes(),**sums)
def check_quiet_start(record):
    selected=set(record.get('cpus',[]))
    if not selected or not selected<=set(ECORES):raise RuntimeError('timed starts require cores 6-21')
    if set(record.get('sibling_cpus',[]))!={c+24 for c in selected}:raise RuntimeError('SMT sibling quiet check missing')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('timed command requires a Slurm allocation')
    record['cpu_policy_check_id']=check(event='pre_command_check',stage=os.environ.get('RAPIDGLM_STAGE'),cpus=sorted(selected),label=record.get('label'))
    record['host_at_start']=dict(load=os.getloadavg(),processes=processes())
