#!/usr/bin/env python3
"""Freeze completed build/check artifacts and the fully prepared receipt harness."""
import datetime,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2];W=ROOT/'validation/work/release-1.6.0-gpu'
COMMIT='4505ef6afaa00a38c5c871ae0e12d531c91ec1cf'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def out(*cmd):return subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
def main():
 for name in ['build-checks','runner-tests','prepare-inputs']:assert (W/(name+'.exit')).read_text().strip()=='0',name
 f=json.loads((W/'functional-checks.json').read_text());c=json.loads((W/'container-checks.json').read_text())
 assert all(v['exit']==0 for v in f.values() if isinstance(v,dict) and 'exit' in v)
 assert all(v['exit']==0 for v in c.values() if isinstance(v,dict) and 'exit' in v)
 assert out('git','rev-parse','v1.6.0-dev.2^{}')==COMMIT
 protected=['include','src','CMakeLists.txt','cmake','bindings']
 assert not out('git','diff',COMMIT,'--',*protected)
 files=out('git','ls-tree','-r','--name-only',COMMIT,'--',*protected).splitlines()
 binaries={str(Path(p).relative_to(ROOT)):h for p,h in f['binaries_sha256'].items()}
 extra=[ROOT/'build/librapidglm_c.a',ROOT/'bindings/rust/target/release/examples/deseq',ROOT/'validation/local-data/R_rapidglm/rapidglm/libs/rapidglm.so',ROOT/'validation/work/benchmark/runs-v1.6.0-gpu/bin/glm_reference',*list((ROOT/'bindings/python/rapidglm').glob('_rapidglm*.so'))]
 for p in extra:assert p.is_file(),str(p);binaries[str(p.relative_to(ROOT))]=sha(p)
 base=Path('/data/lhhung/rapidglm/validation/work/release-1.5.0/base-1.4.1/build')
 for n in ['rapidglm','rapidglm_glm_batch','rapidglm_effect_benchmark']:binaries[str(base/n)]=sha(base/n)
 controllers=[p for p in rfiles() if p.suffix in ['.py','.sh','.sbatch'] or p.name=='CPU_POLICY_EXPECTED.json']
 harness={str(p.relative_to(ROOT)):sha(p) for p in controllers}
 for n in out('git','ls-files','scripts','benchmarks').splitlines():
  p=ROOT/n
  if p.is_file():harness[n]=sha(p)
 source=dict(version='1.6.0',commit=COMMIT,tag='v1.6.0-dev.2',runner_commit=out('git','rev-parse','HEAD'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host='biocloudgpu1',worktree_dirty=bool(out('git','status','--porcelain','--untracked-files=no')),library_sources_equal_measured_commit=True,library_sources_sha256={n:sha(ROOT/n) for n in files},binaries_sha256=binaries,harness_sha256=harness,container_images={'rapidglm:1.6.0':c['image']['id'],'rapidglm-solvers:1.6.0':c['solvers']['id']},build=dict(type='Release',native=True,division_default='slow',compiler=out('g++','--version').splitlines()[0],cmake=out('cmake','--version').splitlines()[0],configure=f['native-configure']['command']),cpu_policy=dict(expected=json.loads((R/'CPU_POLICY_EXPECTED.json').read_text()),controller_files_sha256={str(p.relative_to(R)):sha(p) for p in controllers},command_resume=dict(job_manifest_sha256=sha(W/'glm_speed_jobs.json'))),prepared_inputs_manifest_sha256=sha(W/'prepared-inputs-sha256.json'))
 (R/'source.json').write_text(json.dumps(source,indent=2)+'\n')
 tests=dict(version='1.6.0',commit=COMMIT,host='biocloudgpu1',cpus='0-5',untimed=True,native=f,containers={k:v for k,v in c.items() if k!='source'},container_source_archive_sha256=c['source']['archive_sha256'],runner_regressions=json.loads((W/'runner-tests/TEST_RESULTS.json').read_text()),acceptance_checker=json.loads((W/'runner-tests/CHECKER_TEST_RESULTS.json').read_text()),full_functional_suite='Previously passed on uwt-biocloud1 on this measured commit; this run performs the required GPU-host quick checks only.')
 (R/'TESTS.json').write_text(json.dumps(tests,indent=2)+'\n')
 if not (R/'STAGE_RERUNS.json').exists():(R/'STAGE_RERUNS.json').write_text(json.dumps(dict(commit=COMMIT,reruns=[],command_reruns=[]),indent=2)+'\n')
 subprocess.run(['python3',str(R/'verify_source.py')],cwd=ROOT,check=True)
def rfiles():return [p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
if __name__=='__main__':main()
