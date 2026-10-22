#!/usr/bin/env python3
"""Untimed application comparison of the ARM64 QEMU CLI with x86.

Requires the toolchain container from scripts/build_arm64.sh, kept alive with
docker, its /src source mount and its writable /out build directory.
"""
import argparse,json,os,pathlib,subprocess
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1'}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--container',required=True)
p.add_argument('--resume',action='store_true',help='Reuse completed tables from the same input and binary version')
p.add_argument('--work',type=pathlib.Path,default=ROOT/'build-arm64-r2/apps')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args(); a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)

def translate(value):
    if isinstance(value,str):
        return value.replace(str(ROOT/'build-arm64-r2'),'/out').replace(str(ROOT),'/src')
    return value

configs=[]
for dataset in ('airway','kang'):
    for command in ('bulk-de','edger-ql'):
        configs.append((command+'-'+dataset,command,dict(
            responses=str(ROOT/'validation/work/bulk'/dataset/'counts.mtx'),
            design=str(ROOT/'validation/work/bulk'/dataset/'x.tsv'),threads=1)))
for dataset in ('S1_public','S2_public'):
    c=json.loads((ROOT/'validation/work/public-sceptre'/dataset/'config.json').read_text())
    for check in ('strict','ks'):
        configs.append(('sceptre-'+dataset+'-'+check,'sceptre',{**c,'fit_check':check,'threads':4}))
for family in ('gaussian','nb'):
    source=ROOT/'validation/work/eqtl/runs/rapidglm/inputs'/('chr22.counts.bed' if family=='nb' else 'chr22.expr_int.bed')
    bed=a.work/(family+'.bed');bed.write_text('\n'.join(source.read_text().splitlines()[:21])+'\n')
    c=dict(phenotypes=str(bed),genotypes=str(ROOT/'validation/work/eqtl/data/genotypes/chr22'),
           covariates=str(ROOT/'validation/work/eqtl/data/covariates/covariates.fastqtl.txt'),
           family=family,permutations=100,threads=4,seed=20261016)
    if family=='nb':c['library_sizes']=str(ROOT/'validation/work/eqtl/runs/rapidglm/inputs/library_sizes.tsv')
    configs.append(('qtl-'+family,'qtl',c))

report={'version':subprocess.check_output([str(ROOT/'build/rapidglm'),'--version'],text=True).strip(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'emulated_cpu':'max','applications':{}}
for name,command,c in configs:
    frames={}
    for architecture in ('x86','arm64'):
        prefix=a.work/architecture/name;prefix.parent.mkdir(exist_ok=True)
        config={**c,'output':str(prefix)}
        if architecture=='arm64':config={k:translate(v) for k,v in config.items()}
        path=prefix.with_suffix('.json');path.write_text(json.dumps(config,indent=2)+'\n')
        cmd=([str(ROOT/'build/rapidglm'),command,str(path)] if architecture=='x86' else
             ['docker','exec',a.container,'/out/rapidglm-qemu',command,translate(str(path))])
        suffix='.cis.tsv' if command=='qtl' else '.tsv'
        record_path=pathlib.Path(str(prefix)+'.run.json')
        reuse=(a.resume and pathlib.Path(str(prefix)+suffix).exists() and record_path.exists()
               and json.loads(record_path.read_text()).get('version')==report['version'])
        if not reuse:
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
        frames[architecture]=pd.read_csv(str(prefix)+('.cis.tsv' if command=='qtl' else '.tsv'),sep='\t',na_values='NA')
        print(name,architecture,'complete',flush=True)
    x,y=frames['x86'],frames['arm64'];assert len(x)==len(y)
    result={'rows':len(x),'status_equal':x.status.equals(y.status),'numeric_columns':{}}
    for col in x:
        if not pd.api.types.is_numeric_dtype(x[col]): continue
        left=x[col].to_numpy(float);right=y[col].to_numpy(float)
        finite=np.isfinite(left)&np.isfinite(right);diff=np.abs(left[finite]-right[finite])
        relative=diff/np.maximum(np.abs(left[finite]),1e-300)
        result['numeric_columns'][col]={'max_abs':float(diff.max()) if diff.size else 0.,
            'max_relative':float(relative.max()) if relative.size else 0.,
            'nonfinite_mismatches':int(np.sum(np.isfinite(left)!=np.isfinite(right)))}
        if relative.size:
            worst=int(np.flatnonzero(finite)[np.argmax(relative)])
            result['numeric_columns'][col].update(worst_relative_row=worst,
                x86_at_worst_relative=float(left[worst]),arm64_at_worst_relative=float(right[worst]))
    pcol={'bulk-de':'padj','edger-ql':'FDR','sceptre':'adjusted_p','qtl':'beta_p'}[command]
    result['discoveries_equal']=bool(np.array_equal(x[pcol]<.1,y[pcol]<.1))
    assert result['discoveries_equal'],name
    if command in ('qtl','sceptre'):
        streamcol='permutation_p' if command=='qtl' else 'p_value'
        result['resampling_p_exact']=bool(np.array_equal(x[streamcol],y[streamcol],equal_nan=True))
        if command=='sceptre':
            empirical=(x.approximation==0)&(y.approximation==0)
            result['empirical_p_exact']=bool(np.array_equal(x.p_value[empirical],y.p_value[empirical],equal_nan=True))
            result['draws_exact']=bool(x.draws.equals(y.draws))
            result['approximation_flags_exact']=bool(x.approximation.equals(y.approximation))
            mismatch=(x.p_value!=y.p_value)&~(x.p_value.isna()&y.p_value.isna())
            result['p_mismatches']=int(mismatch.sum())
            result['p_mismatches_using_approximation']=int((mismatch&(x.approximation==1)&(y.approximation==1)).sum())
        # Preserve the exact-comparison result; collect other applications even
        # when a discrepancy needs investigation. Never loosen the criterion.
        if not result['resampling_p_exact']:
            print(name,'resampling p differs: investigation required',flush=True)
    report['applications'][name]=result
    a.output.write_text(json.dumps(report,indent=2)+'\n')
