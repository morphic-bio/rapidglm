#!/usr/bin/env python3
"""Untimed runner regression checks. Synthetic contention is never a receipt."""
import copy, datetime, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
D=R.parents[2]/'validation/work/release-1.6.0-gpu/runner-tests'
D.mkdir(parents=True,exist_ok=True)
ROOT=Path('/data/lhhung/rapidglm-1.6.0')
spec=importlib.util.spec_from_file_location('guarded_candidate',R/'guarded_retry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
T=D/'tests-5';T.mkdir(exist_ok=False)
os.environ.pop('SLURM_JOB_ID',None)
os.environ['RAPIDGLM_STAGE']='untimed-qtl-runner-regression'
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1'
passed=[];cases={}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def passed_case(name,**data):
    passed.append(name);cases[name]=data
    print(json.dumps(dict(test=name,result='PASS',**data)),flush=True)
def expect_exception(function):
    try:function()
    except (RuntimeError,AssertionError,ValueError,KeyError,FileNotFoundError):return
    raise AssertionError('Expected refusal')
class Fixture:
    def __init__(self,name,mode='retry',exitcode=0,real=False):
        self.base=T/name;self.out=self.base/'output';self.out.mkdir(parents=True)
        self.audit=self.base/'commands';self.audit.mkdir()
        self.config=self.out/'config.json';self.config.write_text('{"input":"unchanged"}\n')
        self.command=[sys.executable,'-c','pass'];self.mode=mode;self.exitcode=exitcode
        self.real=real;self.quiets=0;self.runs=0;self.finishes=0;self.pids=[]
        self.options=dict(quiet=self.quiet,finish=self.finish,run=self.run,audit_directory=self.audit,
                          retry_path=self.base/'retries.json',reruns_path=self.base/'reruns.json',archive=self.base/'archives')
    def quiet(self,cpus,label):
        assert cpus=='0-5' and label==' '.join(self.command)
        self.quiets+=1;self.started=utc()
    def run(self,command,cpus,directory):
        assert command==self.command and directory==self.out
        self.runs+=1
        if self.real:
            with (directory/'stdout.log').open('w') as out,(directory/'stderr.log').open('w') as err:
                p=subprocess.Popen(['taskset','-c',cpus,*command],cwd=ROOT,stdout=out,stderr=err)
                self.pids.append(p.pid);status=p.wait()
            assert status==0,(directory/'stderr.log').read_text()
        else:
            (directory/'stdout.log').write_text('test output\n')
            (directory/'stderr.log').write_text('')
            (directory/'result.tsv').write_text('value\n1\n')
            status=self.exitcode
        if self.mode=='changed_input':self.config.write_text('changed\n')
        return subprocess.CompletedProcess(command,status)
    def finish(self):
        self.finishes+=1
        row=dict(pid=os.getpid(),label=' '.join(self.command),stage=os.environ['RAPIDGLM_STAGE'],
                 cpus=list(range(6)),slurm_job_id=None,policy_unchanged=True,contended=False,
                 others_cpu=0.,sibling_busy_fraction={'24':0.},started_utc=self.started,finished_utc=utc())
        if self.finishes>1 or self.mode=='clean':return row
        if self.mode=='unrelated':raise RuntimeError('unrelated measurement failure')
        if self.mode=='empty_success':return {}
        row.update(contended=True,sibling_busy_fraction={'24':.05435891450480667})
        if self.mode=='policy_changed':row['policy_unchanged']=False
        if self.mode=='wrong_pid':row['pid']+=1
        if self.mode=='wrong_label':row['label']='different command'
        if self.mode=='wrong_cpus':row['cpus']=[6]
        if self.mode=='wrong_stage':row['stage']='qtl'
        if self.mode=='wrong_job':row['slurm_job_id']='687'
        if self.mode=='below_threshold':row['sibling_busy_fraction']={'24':.01}
        if self.mode=='stale_time':row['started_utc']='2000-01-01T00:00:00+00:00'
        p=self.audit/(str(os.getpid())+'-'+str(time.time_ns())+'.json')
        g.save(p,row);self.audit_path=p;self.audit_row=row
        if self.mode=='stale_file':os.utime(p,(1,1))
        if self.mode=='outside_directory':
            p=self.base/p.name;g.save(p,row)
        if self.mode=='malformed':p.write_text('not json')
        raise RuntimeError(g.REJECTION+str(p))
    def go(self):return g.run_guarded(self.command,'0-5',self.out,**self.options)
    def proof(self):
        record=json.loads((self.base/'retries.json').read_text())['retries'][0]
        actual=dict(self.audit_row,raw_audit_sha256=g.sha(self.audit_path))
        reruns=json.loads((self.base/'reruns.json').read_text())['command_reruns']
        return record,actual,reruns

try:
    assert set(os.sched_getaffinity(0))==set(range(6))
    assert json.loads((g.W/'TIMED_RUN_HOLD.json').read_text())['hold_active']
    f=Fixture('public_qtl_integration',real=True)
    original=Path('/data/lhhung/rapidglm/validation/work/benchmark/runs-v1.5.0-gpu/eqtl/rapidglm_gaussian10000_double/config.json')
    config=json.loads(original.read_text());bed=Path(config['phenotypes']);mini=f.out/'three_genes.bed'
    with bed.open() as source:
        rows=[next(source) for _ in range(4)]
    assert rows[0].startswith('#')
    mini.write_text(''.join(rows))
    config.update(phenotypes=str(mini),permutations=100,threads=6,output=str(f.out/'chr22'))
    g.save(f.config,config);f.command=[str(ROOT/'build/rapidglm'),'qtl',str(f.config)]
    external=[Path(config['covariates']),bed]+sorted(Path(config['genotypes']).parent.glob(Path(config['genotypes']).name+'.*'))
    input_hashes={str(p):g.sha(p) for p in external if p.is_file()}
    baseline=g.snapshot(f.out)
    assert f.go()==0 and f.quiets==2 and f.runs==2 and f.finishes==2 and len(set(f.pids))==2
    assert all(g.sha(f.out/name)==value for name,value in baseline.items())
    assert all(g.sha(Path(name))==value for name,value in input_hashes.items())
    record,actual,reruns=f.proof();g.verify_retry_record(record,actual,ROOT,reruns)
    result=f.out/'chr22.cis.tsv'
    archived=next(x for x in record['archive_files'] if x['relative_output']=='chr22.cis.tsv')
    assert g.sha(result)==archived['sha256']
    assert len(result.read_text().splitlines())==4
    host=json.loads((f.out/'host.json').read_text())
    assert host['exit']==0 and host['contended'] is False and not (f.out/'host.failed.json').exists()
    passed_case('public_qtl_retry_integration',command=f.command,permutations=100,phenotypes=3,
                precision='double',actual_cli_runs=2,pids=f.pids,quiet_gate_calls=f.quiets,
                result_sha256=g.sha(result),archived_result_byte_identical=True,
                external_input_sha256=input_hashes,untimed=True,synthetic_contention=True)
    for key,value in [('audit_sha256','0'*64),('command_exit',1),('pid',0),('label','wrong'),('stage','wrong')]:
        bad=copy.deepcopy(record);bad[key]=value
        expect_exception(lambda:g.verify_retry_record(bad,actual,ROOT,reruns))
        passed_case('reject_tampered_record_'+key)
    expect_exception(lambda:g.verify_retry_record(record,actual,ROOT,[]))
    passed_case('reject_missing_rerun_evidence')
    archived_file=ROOT/archived['copy'];saved=archived_file.read_bytes();archived_file.write_bytes(saved+b'changed')
    expect_exception(lambda:g.verify_retry_record(record,actual,ROOT,reruns))
    archived_file.write_bytes(saved);assert g.sha(archived_file)==archived['sha256']
    passed_case('reject_tampered_archive')
    for name in ['policy_changed','wrong_pid','wrong_label','wrong_cpus','wrong_stage','wrong_job',
                 'below_threshold','stale_time','stale_file','outside_directory','malformed','unrelated','empty_success','changed_input']:
        x=Fixture(name,mode=name)
        expect_exception(x.go)
        assert x.runs==1 and x.quiets==1
        assert not (x.out/'host.json').exists() and not (x.base/'retries.json').exists()
        passed_case('no_retry_'+name)
    for name,mode in [('nonzero_clean','clean'),('nonzero_contended','retry')]:
        x=Fixture(name,mode=mode,exitcode=7)
        assert x.go()==7 and x.runs==1 and x.quiets==1
        assert json.loads((x.out/'host.failed.json').read_text())['exit']==7
        assert not (x.base/'retries.json').exists()
        passed_case(name)
    x=Fixture('prior_output')
    (x.out/'time.log').write_text('prior\n');expect_exception(x.go);assert x.runs==0
    passed_case('prior_outputs_not_overwritten')
    probe=T/'hold-probe'
    p=subprocess.run([sys.executable,str(R/'guarded_retry.py'),'--cpus','0-5','--directory',str(probe),'--',sys.executable,'-c','raise SystemExit(99)'],capture_output=True,text=True)
    assert p.returncode==2 and 'on hold' in p.stderr and not probe.exists()
    passed_case('hold_blocks_timed_entrypoint',exit=p.returncode)
    import quiet_run,cpu_policy
    original_audit=quiet_run.audit;original_check=cpu_policy.check_quiet_start
    old_flag=os.environ.pop('RAPIDGLM_CPU_POLICY_ENABLED',None)
    try:
        g.install_policy_hook()
        assert quiet_run.audit is original_audit
        passed_case('policy_hook_disabled_without_policy_flag')
        events=[]
        quiet_run.audit=lambda record:events.append(('audit',record['event']))
        cpu_policy.check_quiet_start=lambda record:events.append(('policy',record['event']))
        os.environ['RAPIDGLM_CPU_POLICY_ENABLED']='1'
        g.install_policy_hook()
        quiet_run.audit({'event':'quiet_ready'});quiet_run.audit({'event':'waiting'})
        assert events==[('policy','quiet_ready'),('audit','quiet_ready'),('audit','waiting')]
        passed_case('policy_check_before_quiet_ready_audit')
        events=[]
        quiet_run.audit=lambda record:events.append(('audit',record['event']))
        def changed_policy(record):raise RuntimeError('policy changed')
        cpu_policy.check_quiet_start=changed_policy;g.install_policy_hook()
        expect_exception(lambda:quiet_run.audit({'event':'quiet_ready'}))
        assert events==[]
        passed_case('policy_failure_blocks_quiet_ready')
    finally:
        quiet_run.audit=original_audit;cpu_policy.check_quiet_start=original_check
        if old_flag is None:os.environ.pop('RAPIDGLM_CPU_POLICY_ENABLED',None)
        else:os.environ['RAPIDGLM_CPU_POLICY_ENABLED']=old_flag
    launch=subprocess.run(['bash',str(R/'run_accepted.sh'),'qtl'],capture_output=True,text=True,cwd=ROOT)
    assert launch.returncode==1 and 'on hold' in launch.stderr
    passed_case('full_launcher_blocks_active_hold',exit=launch.returncode)
    result=dict(result='PASS',finished_utc=utc(),untimed=True,cpus=sorted(os.sched_getaffinity(0)),
                slurm_job_submitted=False,tests_passed=len(passed),tests=cases,
                guarded_sha256=g.sha(R/'guarded_retry.py'),checker_sha256=g.sha(R/'check_run_accepted.py'))
    g.save(D/'TEST_RESULTS.json',result);g.save(D/'state.json',dict(state='DONE',exit=0,utc=utc()))
except BaseException:
    traceback.print_exc()
    g.save(D/'state.json',dict(state='FAILED',exit=1,utc=utc(),tests_passed=passed))
    raise
