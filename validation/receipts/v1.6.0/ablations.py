#!/usr/bin/env python3
"""Measure the five default/opt-out pairs on the same host, inputs and cores."""
import csv, hashlib, json, os, subprocess, sys, datetime
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
RUNS=ROOT/'validation/work/benchmark/runs-v1.6.0-gpu'
OLD=Path('/data/lhhung/rapidglm')
PILOT=ROOT/'validation/work/replogle/k562-essential/pilot-v1.6.0-gpu'
FIELDS=['null_start_used','null_refined','null_cold_refits','batch_scoring','permutation_fits','permutations_refitted','crt_draw_used']
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def low_moi_agreement(a,b):
 rows=empirical=discovery=0
 with a.open() as af,b.open() as bf:
  ar=csv.DictReader(af,delimiter='\t');br=csv.DictReader(bf,delimiter='\t')
  for x,y in __import__('itertools').zip_longest(ar,br):
   assert x is not None and y is not None,'row count differs'
   for key in ['response','target','guide','kind','status','draws','approximation']:
    assert x[key]==y[key],(key,rows,x.get(key),y.get(key))
   if x['approximation']=='0':
    assert x['p_value']==y['p_value'],('empirical p',rows)
    empirical+=1
   def hit(z):
    return z['kind']=='0' and z['status'].startswith('ok') and z['adjusted_p']!='NA' and float(z['adjusted_p'])<=.1
   assert hit(x)==hit(y),('discovery',rows)
   discovery+=hit(x);rows+=1
 return dict(passed=True,rows=rows,statuses_identical=True,empirical_p_identical=True,empirical_rows=empirical,discoveries_identical=True,discoveries=discovery)
def records(prefix):
 p=Path(str(prefix)+'.run.json');d=load(p)
 return {k:d[k] for k in FIELDS if k in d}
def run(command):subprocess.run(list(map(str,command)),cwd=ROOT,check=True)
def main():
 assert os.environ.get('SLURM_JOB_ID') and os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED')=='1'
 source=load(R/'source.json');receipt=dict(version='1.6.0',commit=source['commit'],host='biocloudgpu1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pairs=[])
 def add(name,default_seconds,ablation_seconds,agreement,**extra):
  receipt['pairs'].append(dict(name=name,default_seconds=default_seconds,ablation_seconds=ablation_seconds,speedup_default_over_opt_out=ablation_seconds/default_seconds,agreement=agreement,**extra))
  (R/'OPTIMIZATION_ABLATIONS.json').write_text(json.dumps(receipt,indent=2)+'\n')
 for fc in ['strict','ks']:
  run(['flock',os.environ['RAPIDGLM_TIMING_LOCK'],'taskset','-c','0-5','python3','scripts/replogle_pilot.py','sc','--pilot',PILOT,'--cpus','6-21','--threads','16','--fit-check',fc,'--null-start','cold'])
  name='rapidglm_t16'+('' if fc=='strict' else '_ks')
  default=PILOT/'runs'/name;ablation=PILOT/'runs'/(name+'_cold')
  a=Path(str(default)+'.tsv');b=Path(str(ablation)+'.tsv')
  agreement=low_moi_agreement(a,b)
  agreement['job_687']=low_moi_agreement(OLD/'validation/work/replogle/k562-essential/pilot-v1.5.0-gpu/runs'/(name+'.tsv'),b)
  add('replogle_'+fc,load(Path(str(default)+'.timing.json'))['wall_seconds'],load(Path(str(ablation)+'.timing.json'))['wall_seconds'],agreement,cpus='6-21',threads=16,opt_out={'null_start':'cold'},default_record=records(default),ablation_record=records(ablation),default_sha256=sha(a),ablation_sha256=sha(b))
 for variant in ['ks','default']:
  default=RUNS/'crt'/('t100_g7418_'+variant);ablation=RUNS/'ablations'/('crt_t100_g7418_'+variant);ablation.mkdir(parents=True,exist_ok=True)
  config=load(default/'config.json');config.update(batch_scoring=False,output=str(ablation/'rapid'));(ablation/'config.json').write_text(json.dumps(config,indent=2)+'\n')
  run(['env','OMP_NUM_THREADS=1','python3',R/'guarded_retry.py','--cpus','6-21','--directory',ablation,'--','build/rapidglm','sceptre',ablation/'config.json'])
  a=default/'rapid.tsv';b=ablation/'rapid.tsv';equal=sha(a)==sha(b)
  add('crt_t100_g7418_'+variant,load(default/'host.json')['wall_seconds'],load(ablation/'host.json')['wall_seconds'],dict(passed=equal,byte_identical=equal),cpus='6-21',threads=16,opt_out={'batch_scoring':False},default_record=records(default/'rapid'),ablation_record=records(ablation/'rapid'),default_sha256=sha(a),ablation_sha256=sha(b))
  assert equal,'CRT ablation result differs'
 default=RUNS/'eqtl/rapidglm_nb';ablation=RUNS/'ablations/qtl_nb_separate';ablation.mkdir(parents=True,exist_ok=True)
 config=load(default/'config.json');config.update(permutation_fits='separate',output=str(ablation/'chr22'));(ablation/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 run(['python3',R/'guarded_retry.py','--cpus','6-17','--directory',ablation,'--','build/rapidglm','qtl',ablation/'config.json'])
 a=default/'chr22.cis.tsv';b=ablation/'chr22.cis.tsv';old=OLD/'validation/work/benchmark/runs-v1.5.0-gpu/eqtl/rapidglm_nb/chr22.cis.tsv'
 equal=sha(b)==sha(old)
 add('qtl_nb_chr22',load(default/'host.json')['wall_seconds'],load(ablation/'host.json')['wall_seconds'],dict(passed=equal,separate_byte_identical_to_job_687=equal,job_687_sha256=sha(old)),cpus='6-17',threads=12,permutations=1000,opt_out={'permutation_fits':'separate'},default_record=records(default/'chr22'),ablation_record=records(ablation/'chr22'),default_sha256=sha(a),ablation_sha256=sha(b))
 assert equal,'NB separate result differs from job 687'
 receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt['all_agreements_passed']=True
 (R/'OPTIMIZATION_ABLATIONS.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
