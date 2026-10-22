#!/usr/bin/env python3
"""Audit the final CPU policy without changing host settings."""
import datetime,fcntl,hashlib,json,os,time
from pathlib import Path
R=Path(__file__).resolve().parent

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read_policy():
    root=Path('/sys/devices/system/cpu');cpus={}
    for cpu in [8,9,10,11,*range(16,32)]:
        names=['scaling_max_freq'] if cpu<16 else ['scaling_min_freq','scaling_max_freq']
        cpus[str(cpu)]={k:(root/f'cpu{cpu}/cpufreq'/k).read_text().strip() for k in names}
    return dict(no_turbo=(root/'intel_pstate/no_turbo').read_text().strip(),cpus=cpus)

def digest(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()

def emit(event,**values):
    path=Path(os.environ.get('RAPIDGLM_CPU_POLICY_AUDIT',str(R/'CPU_POLICY_AUDIT.jsonl')))
    row=dict(event=event,utc=utc(),pid=os.getpid(),**values)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(json.dumps(row,separators=(',',':'))+'\n')
    return row

def check(expected,event='policy_check',**context):
    actual=read_policy();identifier=f'{os.getpid()}:{time.monotonic_ns()}'
    row=emit(event,check_id=identifier,matched=actual==expected,policy_sha256=digest(actual),**context)
    if actual!=expected:
        path=Path(os.environ.get('RAPIDGLM_CPU_POLICY_VIOLATION',str(R/'CPU_POLICY_VIOLATION.json')))
        path.write_text(json.dumps(dict(row,expected=expected,actual=actual),indent=2)+'\n')
        raise RuntimeError('CPU policy differs from the final setting; host settings were not changed')
    return identifier

def check_quiet_start(record):
    expected=json.loads((R/'source.json').read_text())['cpu_policy']['expected']
    selected=record.get('cpus',[])
    if not selected or not set(selected)<=set(range(16,32)):
        raise RuntimeError('Timed starts must select only efficiency CPUs 16-31')
    record['cpu_policy_check_id']=check(expected,event='pre_command_check',
        stage=os.environ.get('RAPIDGLM_STAGE'),quiet_start_utc=record['utc'],
        quiet_start_pid=os.getpid(),cpus=selected,label=record.get('label'))
