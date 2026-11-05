#!/usr/bin/env python3
"""Index the optimization choices actually recorded by each completed stage."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
RUNS=ROOT/'validation/work/benchmark/runs-v1.6.0-gpu'
PILOT=ROOT/'validation/work/replogle/k562-essential/pilot-v1.6.0-gpu'
FIELDS=['null_start_used','null_refined','null_cold_refits','batch_scoring','permutation_fits','permutations_refitted','crt_draw_used']
ROOTS={'staging':[RUNS/'staging'],'pert50k':[RUNS/'gasperini50k'],'pertfull':[RUNS/'gasperini'],'seeds':[RUNS/'seeds'],'untestable':[RUNS/'untestable/runs'],'crt':[RUNS/'crt'],'replogle':[PILOT/'runs'],'qtl':[RUNS/'eqtl'],'ablations':[RUNS/'ablations',PILOT/'runs'],'eqtl_gw':[ROOT/'validation/work/eqtl/runs-genomewide-v1.6.0-gpu']}
def main():
 stage=sys.argv[1]
 if stage not in ROOTS:return
 path=R/'RUN_RECORD_FIELDS.json';record=json.loads(path.read_text()) if path.exists() else dict(version='1.6.0',commit=json.loads((R/'source.json').read_text())['commit'],stages={})
 rows=[]
 for folder in ROOTS[stage]:
  for p in sorted(folder.rglob('*.run.json')):
   if stage=='ablations' and p.parent==PILOT/'runs' and '_cold.' not in p.name:continue
   d=json.loads(p.read_text());fields={k:d[k] for k in FIELDS if k in d}
   if not fields:continue
   rows.append(dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fields=fields))
 assert rows,'no run record fields collected for '+stage
 record['stages'][stage]=rows;path.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
