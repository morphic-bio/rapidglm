#!/usr/bin/env python3
"""Run an owned container and retain CPU usage that Docker does not return to wait4."""
import argparse,json,os,subprocess,sys
from pathlib import Path
INNER = """
import json,resource,subprocess,sys
try:
    result=subprocess.run(sys.argv[1:])
    code=result.returncode
finally:
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    with open('/measurement/cpu.json','w') as f:
        json.dump(dict(cpu_seconds=r.ru_utime+r.ru_stime,method='container wait4 child rusage'),f)
raise SystemExit(code)
"""
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cpus',required=True);p.add_argument('--image',required=True)
p.add_argument('--data',type=Path);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command:p.error('a command is required')
audit=Path(os.environ['RAPIDGLM_EXTERNAL_CPU_AUDIT']).resolve()
folder=audit.with_suffix('');folder.mkdir()
cmd=['docker','run','--rm','--label','rapidglm.slurm_job='+os.environ['SLURM_JOB_ID'],
     '--cpuset-cpus',a.cpus,'-e','OPENBLAS_NUM_THREADS=1',
     '-e','RAPIDGLM_DIVISION='+os.environ.get('RAPIDGLM_DIVISION','slow'),
     '-v',str(folder)+':/measurement']
if a.data:cmd+=['-v',str(a.data.resolve())+':/data:ro']
result=subprocess.run(cmd+[a.image,'python3','-c',INNER,*command])
record=json.loads((folder/'cpu.json').read_text())
record.update(image=a.image,cpus=a.cpus,exit=result.returncode)
audit.write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(result.returncode)
