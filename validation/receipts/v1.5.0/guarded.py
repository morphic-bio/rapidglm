#!/usr/bin/env python3
"""One command under the shared lock, quiet checks, policy and contention audit."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(R))
from quiet_run import LOCK,wait_quiet,finish_measurement
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpus',required=True);p.add_argument('--directory',required=True,type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command:p.error('a command is required')
a.directory.mkdir(parents=True,exist_ok=True)
if (a.directory/'host.json').exists():p.error('archive the previous command before rerunning')
with open(LOCK,'a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);wait_quiet(a.cpus,label=' '.join(command))
 with (a.directory/'stdout.log').open('w') as out,(a.directory/'stderr.log').open('w') as err:
  result=subprocess.run(['/usr/bin/time','-v','-o',str(a.directory/'time.log'),'taskset','-c',a.cpus,*command],stdout=out,stderr=err)
 row=finish_measurement() or {};row['exit']=result.returncode;row['command']=command
 (a.directory/'host.json').write_text(json.dumps(row,indent=1)+'\n')
raise SystemExit(result.returncode)
