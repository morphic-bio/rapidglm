#!/usr/bin/env python3
"""One-core end-to-end ratios: 1.4.1, 1.5.0 slow and 1.5.0 fast, public inputs."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
W=ROOT/'validation/work/release-1.5.0-gpu'
sys.path[:0]=[str(ROOT/'scripts'),str(R)]
from quiet_run import LOCK,wait_quiet,finish_measurement
CPU='6'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--work',type=Path,default=W/'vmath-e2e');ap.add_argument('--array-only',type=Path);a=ap.parse_args()
 T=a.work;T.mkdir(parents=True,exist_ok=True)
 variants={'1.4.1':(ROOT/'validation/work/release-1.5.0/base-1.4.1/build','slow'),'1.5.0-slow':(ROOT/'build','slow'),'1.5.0-fast':(ROOT/'build','fast')}
 if a.array_only:variants['1.5.0-fast-arrays-only']=(a.array_only,'fast')
 thyroid=ROOT/'validation/work/benchmark/thyroid';rep=ROOT/'validation/work/release-1.5.0/inputs/replogle_AAAS';batch=ROOT/'validation/work/benchmark/runs-v1.4.0-final/glm-compare/batch-v1.4.0'
 public=ROOT/'validation/work/public-sceptre/S2_public'
 effect={k:str(public/(k+'.tsv' if k in ['design','guides','pairs'] else k+'.mtx')) for k in ['responses','assignments','design','guides','pairs']}
 effect.update(moi='high',family='nb',control_group='complement',cell_qc=False,assignment_qc=False,integration='union',threads=1,seed=20260926)
 effects=T/'public-effects.json';effects.write_text(json.dumps(effect,indent=2)+'\n')
 inputs=[thyroid/'counts.mtx',thyroid/'x.tsv',rep/'counts.mtx',rep/'x.tsv',*[Path(effect[k]) for k in ['responses','assignments','design','guides','pairs']]]
 inputs += [p for p in batch.iterdir() if p.name in ['X.f64','Y.f64','meta.json']]
 def config(name,v,responses,design,coefficient):
  p=T/f'{name}_{v}.json';p.write_text(json.dumps(dict(responses=str(responses),design=str(design),coefficient=coefficient,threads=1,output=str(T/f'{name}_{v}'))));return p
 workloads={
  'bulk_de_thyroid':(2,lambda b,v:[b/'rapidglm','bulk-de',config('thyroid_bulk',v,thyroid/'counts.mtx',thyroid/'x.tsv',11)],lambda v:[T/f'thyroid_bulk_{v}.tsv']),
  'bulk_de_replogle_AAAS':(2,lambda b,v:[b/'rapidglm','bulk-de',config('replogle_bulk',v,rep/'counts.mtx',rep/'x.tsv',48)],lambda v:[T/f'replogle_bulk_{v}.tsv']),
  'edger_ql_thyroid':(3,lambda b,v:[b/'rapidglm','edger-ql',config('thyroid_edger',v,thyroid/'counts.mtx',thyroid/'x.tsv',11)],lambda v:[T/f'thyroid_edger_{v}.tsv']),
  'nb_glm_batch_5000x5x1000':(5,lambda b,v:[b/'rapidglm_glm_batch',batch,5000,5,1000,'nb',1,T/f'nbbatch_{v}.tsv'],lambda v:[T/f'nbbatch_{v}.tsv']),
  'sceptre_nb_effect_fits_public_high_moi':(2,lambda b,v:[b/'rapidglm_effect_benchmark',effects,T/f'effects_{v}.tsv',1,1],lambda v:[T/f'effects_{v}.tsv'])}
 out=dict(purpose=__doc__,cpu=6,sibling_cpu=30,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),variants={v:dict(build=str(b),division=m) for v,(b,m) in variants.items()},inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},public_effect_provenance='validation/PUBLIC_SCEPTRE.json',workloads={})
 out['binaries_sha256']={v:{n:sha(b/n) for n in ['rapidglm','rapidglm_glm_batch','rapidglm_effect_benchmark'] if (b/n).exists()} for v,(b,m) in variants.items()}
 def save():a.output.write_text(json.dumps(out,indent=1)+'\n')
 for name,(repeats,command,outputs) in workloads.items():
  labels=[v for v in variants if not (a.array_only and v.endswith('arrays-only') and name.startswith(('nb_glm','sceptre')))]
  records=[]
  for repeat in range(repeats):
   order=labels[repeat%len(labels):]+labels[:repeat%len(labels)]
   for v in order:
    b,mode=variants[v];os.environ['RAPIDGLM_DIVISION']=mode;cmd=list(map(str,command(b,v)));env=dict(os.environ,RAPIDGLM_DIVISION=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with open(LOCK,'a') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX);wait_quiet(CPU,label=f'vmath_e2e {name} {v} repeat={repeat}')
     start=time.perf_counter();result=subprocess.run(['taskset','-c',CPU,*cmd],env=env,capture_output=True,text=True);elapsed=time.perf_counter()-start
     audit=finish_measurement() or {}
    if result.returncode:raise RuntimeError(f'{name} {v} exit {result.returncode}: {result.stderr[-2000:]}')
    rec=dict(variant=v,division=mode,repeat=repeat,seconds=elapsed,outputs_sha256={p.name:sha(p) for p in outputs(v) if p.exists()},**{k:audit.get(k) for k in ['others_cpu','sibling_busy_fraction','policy_unchanged','contended','started_utc']})
    records.append(rec);print(name,v,elapsed,flush=True)
    entry=dict(measurements=records,min={label:min(z['seconds'] for z in records if z['variant']==label) for label in {z['variant'] for z in records}})
    if all(v in entry['min'] for v in ['1.4.1','1.5.0-slow','1.5.0-fast']):
     entry['speedup_vs_1.4.1']={m:entry['min']['1.4.1']/entry['min']['1.5.0-'+m] for m in ['slow','fast']};entry['fast_vs_slow']=entry['min']['1.5.0-slow']/entry['min']['1.5.0-fast']
    out['workloads'][name]=entry;save()
 out['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
if __name__=='__main__':main()
