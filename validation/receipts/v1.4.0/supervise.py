#!/usr/bin/env python3
"""Supervise every stage and audit the final policy; never change CPU settings."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
from cpu_policy import check,digest,emit,read_policy,utc
R=Path(__file__).resolve().parent;ROOT=R.parents[2]

def stop_child(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
    source=json.loads((R/'source.json').read_text());expected=source['cpu_policy']['expected']
    assert expected['no_turbo']=='0'
    assert all(expected['cpus'][str(c)]=={'scaling_min_freq':'800000','scaling_max_freq':'4300000'} for c in range(16,32))
    assert all(expected['cpus'][str(c)]=={'scaling_max_freq':'2200000'} for c in range(8,12))
    assert read_policy()==expected,'CPU policy differs before launch; stop without changing it'
    for name in ['status.txt','CPU_POLICY_AUDIT.jsonl','CPU_POLICY_VIOLATION.json']:
        assert not (R/name).exists(),f'Archive an earlier attempt first: {name}'
    for name,value in source['cpu_policy']['controller_files_sha256'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value,name
    subprocess.run(['python3',str(R/'verify_source.py')],cwd=ROOT,check=True)
    started=utc()
    emit('started',commit=source['commit'],policy=expected,policy_sha256=digest(expected),
         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         poll_interval_seconds=1,maximum_allowed_poll_gap_seconds=5)
    interrupted=[]
    for signum in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(signum,lambda number,frame:interrupted.append(number))
    child=None;sample_count=quiet_count=0;maximum_gap=0.;code=125;error=None
    last_sample=last_heartbeat=time.monotonic();position=0
    try:
        check(expected,event='launch_check')
        child=subprocess.Popen(['bash',str(R/'run.sh')],cwd=ROOT,start_new_session=True)
        emit('runner_started',runner_pid=child.pid,process_group=child.pid)
        print('Started complete final-policy release run:',started,flush=True)
        while True:
            now=time.monotonic();gap=now-last_sample;maximum_gap=max(maximum_gap,gap)
            if interrupted:raise RuntimeError(f'Supervisor interrupted by signal {interrupted[-1]}')
            if (R/'CPU_POLICY_VIOLATION.json').exists():raise RuntimeError('A per-command CPU-policy check failed')
            if gap>5:
                emit('monitor_gap',seconds=gap);raise RuntimeError('CPU-policy monitor gap exceeded five seconds')
            sample_count+=1;check(expected,sample=sample_count);last_sample=now
            audit=R/'timing-audit.jsonl'
            if audit.exists():
                with audit.open() as stream:
                    stream.seek(position)
                    for line in stream:
                        if not line.endswith('\n'):break
                        position+=len(line.encode());event=json.loads(line)
                        if event['event']=='quiet_ready':
                            assert event.get('cpu_policy_check_id'),'Quiet start lacks synchronous CPU-policy check'
                            quiet_count+=1
                            emit('quiet_start_observed',stage=event['stage'],quiet_start_utc=event['utc'],
                                 quiet_start_pid=event['pid'],cpus=event['cpus'],
                                 pre_command_check_id=event['cpu_policy_check_id'],policy_sha256=digest(expected))
            if now-last_heartbeat>=15:
                emit('heartbeat',samples=sample_count,observed_quiet_starts=quiet_count,
                     maximum_poll_gap_seconds=maximum_gap,policy_sha256=digest(expected));last_heartbeat=now
            if child.poll() is not None:
                check(expected,event='final_check');code=child.returncode;break
            time.sleep(1)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';emit('aborted',error=error);stop_child(child);code=125
    finally:
        summary=dict(exit=code,started_utc=started,finished_utc=utc(),cpu_policy_unchanged=error is None,
                     cpu_policy_samples=sample_count,cpu_policy_maximum_poll_gap_seconds=maximum_gap,
                     cpu_policy_observed_quiet_starts=quiet_count,error=error)
        emit('finished',**summary);(R/'run-complete.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('Release receipt runner completed:',summary,flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
