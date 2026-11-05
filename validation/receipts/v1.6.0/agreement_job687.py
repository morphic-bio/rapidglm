#!/usr/bin/env python3
"""Check new scientific outputs against the preserved accepted 686/687 collection."""
import csv,hashlib,json,math,sys
from pathlib import Path
from ablations import low_moi_agreement
R=Path(__file__).resolve().parent;ROOT=R.parents[2];OLDROOT=Path('/data/lhhung/rapidglm')
NEW=ROOT/'validation/work/benchmark/runs-v1.6.0-gpu';OLD=OLDROOT/'validation/work/benchmark/runs-v1.5.0-gpu';OLDR=OLDROOT/'validation/receipts/v1.5.0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def main():
 stage=sys.argv[1];path=R/'AGREEMENT_JOB_687.json'
 report=load(path) if path.exists() else dict(commit=load(R/'source.json')['commit'],reference_commit='67f5adc3aeb09329a93d10cd608110f3bc0ca72e',reference_collection='Accepted stages through Slurm job 687, including retained earlier stages',stages={})
 checks=[];errors=[]
 def exact(a,b):
  ok=a.exists() and b.exists() and sha(a)==sha(b)
  checks.append(dict(file=str(b.relative_to(ROOT)),reference=str(a),byte_identical=ok,reference_sha256=sha(a) if a.exists() else None,sha256=sha(b) if b.exists() else None))
  if not ok:errors.append(str(b))
 def low(a,b):
  try:agreement=low_moi_agreement(a,b)
  except (AssertionError,KeyError,FileNotFoundError) as error:agreement=dict(passed=False,error=str(error));errors.append(str(b))
  checks.append(dict(file=str(b.relative_to(ROOT)),reference=str(a),agreement=agreement))
 maps={'batch':['glm-compare/batch-v1.5.0'],'bulkval':['bulkval'],'edgerval':['edger'],'bulk':['airway','thyroid32','thyroid128','thyroid'],'cooks':['cooks','cooks_flags'],'pert50k':['gasperini50k'],'pertfull':['gasperini'],'seeds':['seeds'],'crt':['crt']}
 if stage in maps:
  for name in maps[stage]:
   old=OLD/name;new=NEW/name.replace('1.5.0','1.6.0')
   for a in sorted(old.rglob('*.tsv')):exact(a,new/a.relative_to(old))
 elif stage in ['glmacc','glmspeed']:
  old=OLD/'glm-compare/problems-v1.5.0';new=NEW/'glm-compare/problems-v1.6.0'
  for folder in sorted(old.iterdir()):
   meta=load(folder/'meta.json')
   if (meta['suite']=='speed')!=(stage=='glmspeed'):continue
   for a in sorted(folder.glob('result-*.json')):
    b=new/a.relative_to(old)
    x={k:v for k,v in load(a).items() if k!='seconds'};y={k:v for k,v in load(b).items() if k!='seconds'}
    ok=json.dumps(x,sort_keys=True)==json.dumps(y,sort_keys=True)
    checks.append(dict(file=str(b.relative_to(ROOT)),scientific_fields_identical=ok,excluded_timing_field='seconds'))
    if not ok:errors.append(str(b))
 elif stage in ['staging','untestable']:
  base=OLD/('staging' if stage=='staging' else 'untestable/runs')
  for record in sorted(base.rglob('*.run.json')):
   a=Path(str(record).removesuffix('.run.json')+'.tsv');b=NEW/str(a.relative_to(OLD)).replace('1.5.0','1.6.0')
   d=load(record);config=Path(d['config_path']);config=config if config.is_absolute() else OLDROOT/config;c=load(config)
   if c.get('moi')=='low':low(a,b)
   else:
    exact(a,b)
    qq=Path(str(a).removesuffix('.tsv')+'.qq.tsv')
    if qq.exists():exact(qq,Path(str(b).removesuffix('.tsv')+'.qq.tsv'))
 elif stage=='replogle':
  old=OLDROOT/'validation/work/replogle/k562-essential/pilot-v1.5.0-gpu';new=ROOT/'validation/work/replogle/k562-essential/pilot-v1.6.0-gpu'
  for n in ['rapidglm_t16.tsv','rapidglm_t16_ks.tsv']:low(old/'runs'/n,new/'runs'/n)
  for a in sorted((old/'pseudobulk/datasets').rglob('*.tsv')):exact(a,new/a.relative_to(old))
 elif stage=='qtl':
  for n in ['rapidglm_gaussian','rapidglm_gaussian10000','rapidglm_gaussian10000_double']:exact(OLD/'eqtl'/n/'chr22.cis.tsv',NEW/'eqtl'/n/'chr22.cis.tsv')
  exact(OLDROOT/'validation/work/benchmark/runs-v1.4.0-final/eqtl/rapidglm_gaussian10000/chr22.cis.tsv',NEW/'eqtl/rapidglm_gaussian10000_double/chr22.cis.tsv')
  a=OLD/'eqtl/rapidglm_nb/chr22.cis.tsv';b=NEW/'eqtl/rapidglm_nb/chr22.cis.tsv';maximum=0.;count=0
  with a.open() as af,b.open() as bf:
   for x,y in __import__('itertools').zip_longest(csv.DictReader(af,delimiter='\t'),csv.DictReader(bf,delimiter='\t')):
    if x is None or y is None:errors.append('NB row count');break
    for key in ['phenotype_id','status','lead_variant','statistic','p_value','estimate','standard_error','permutation_p','permutations','permutations_failed']:
     if x[key]!=y[key]:errors.append('NB '+x['phenotype_id']+' '+key)
    if x['beta_p']!=y['beta_p']:
     if x['beta_p']=='NA' or y['beta_p']=='NA':errors.append('NB Beta missing mismatch')
     else:maximum=max(maximum,abs(float(x['beta_p'])-float(y['beta_p']))/max(abs(float(x['beta_p'])),1e-300))
    count+=1
  if maximum>8e-11:errors.append('NB Beta relative bound '+str(maximum))
  checks.append(dict(file=str(b.relative_to(ROOT)),rows=count,max_beta_p_relative_difference=maximum,beta_p_relative_bound=8e-11,required_columns_identical=not any(x.startswith('NB ') for x in errors)))
 elif stage=='rust':
  for name in ['airway','thyroid32','thyroid128','thyroid']:
   a=load(OLDR/('rust_'+name+'.json'));b=load(R/('rust_'+name+'.json'))
   keys=['samples','genes','columns','calls']+(['cooks_outliers'] if 'cooks_outliers' in a else [])
   same=all(a[k]==b[k] for k in keys)
   checks.append(dict(dataset=name,retained_scientific_fields_identical=same,fields=keys,raw_table_identity_available=False,note='The historical comparison deleted its temporary result tables; retained scientific fields are compared exactly.'))
   if not same:errors.append('Rust scientific fields '+name)
 else:return
 assert checks,'no agreement checks for '+stage
 report['stages'][stage]=dict(checks=checks,problems=errors,passed=not errors)
 path.write_text(json.dumps(report,indent=2)+'\n')
 if errors:raise SystemExit('agreement mismatch in '+stage+': '+str(errors[:10]))
 print(stage,len(checks),'agreement checks passed')
if __name__=='__main__':main()
