#!/usr/bin/env python3
"""Run the three focused groups under the unchanged exclusive timing policy."""
import json, os, subprocess, time, traceback
from preprint_gate import R, ROOT, W, ORDER, utc, save, load, run, milestone

def main():
    job = os.environ['SLURM_JOB_ID']; info = subprocess.check_output(['scontrol','show','job',job],text=True)
    assert 'NumCPUs=48' in info and 'OverSubscribe=NO' in info
    assert set(os.sched_getaffinity(0)) <= set(range(6))
    run('python3', R/'verify_source.py')
    path=R/'preprint-chain-status.json'; assert not path.exists(), 'Inspect existing focused run before restarting'
    state=dict(commit=load(R/'source.json')['commit'], scope='preprint', started_utc=utc(), job=job, stages={});save(path,state)
    for stage in ORDER:
        label=stage+'-'+str(time.time_ns());row=dict(started_utc=utc(),label=label,slurm_job_id=job,exit=None)
        state['stages'][stage]=row;save(path,state);milestone('Preprint: '+stage+' started under exclusive Slurm job '+job+'.')
        with (W/'logs'/('stage-'+label+'.log')).open('w') as log:
            result=subprocess.run(['python3',str(R/'supervise_accepted.py'),label,stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row.update(exit=result.returncode,finished_utc=utc());save(path,state)
        if result.returncode: raise RuntimeError(stage+' stopped with exit '+str(result.returncode))
        run('python3',R/'verify_source.py')
        run('python3',R/'check_run_accepted.py','--preprint','--partial')
        run('python3',R/'preprint_gate.py','check',stage)
        milestone('Preprint: '+stage+' passed timing acceptance and scientific agreement.')
    state['finished_utc']=utc();save(path,state)
    run('python3',R/'preprint_gate.py','assemble')
    return 0
if __name__=='__main__':
    try: code=main()
    except BaseException:
        traceback.print_exc();code=1
        milestone('Preprint focused gate stopped; inspect preprint job log. No incomplete measurements accepted; no release/publication performed.')
    (W/'preprint-chain.exit').write_text(str(code)+'\n');raise SystemExit(code)
