#!/usr/bin/env python3
"""Build the comparison table from the new, matched chromosome-22 reference outputs."""
import argparse
import pathlib
import pandas as pd
import polars as pl
import numpy as np
from compare_eqtl import qvalues

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--tensorqtl',type=pathlib.Path,
               help='tensorQTL cis table (default: RUNS/tensorqtl/chr22.cis_qtl.txt.gz)')
p.add_argument('--tensorqtl-only',action='store_true',
               help='the tensorQTL reference alone (the genome-wide run has no FastQTL or jaxQTL)')
a=p.parse_args()
frames={}
if not a.tensorqtl_only:
    cols=['phenotype_id','num_var','beta_shape1','beta_shape2','true_df','pval_true_df',
          'variant_id','tss_distance','ma_samples','ma_count','maf','ref_factor',
          'pval_nominal','slope','slope_se','pval_perm','pval_beta']
    parts=sorted((a.runs/'fastqtl/chunks').glob('chunk_*.txt.gz'))
    f=pd.concat([pd.read_csv(x,sep='\t',header=None,names=cols) for x in parts])
    frames['FastQTL_perm_beta']=pd.DataFrame({'id':f.phenotype_id,'p':f.pval_beta,'lead':f.variant_id})
t=pd.read_csv(a.tensorqtl or a.runs/'tensorqtl/chr22.cis_qtl.txt.gz',sep='\t')
frames['tensorQTL_perm10000_beta']=pd.DataFrame({'id':t.phenotype_id,'p':t.pval_beta,'lead':t.variant_id})
for name,folder in ([] if a.tensorqtl_only else
                    [('jaxQTL_NB_score_perm1000','jaxqtl_nb_score_perm1000'),
                     ('jaxQTL_NB_score_SPA_ACAT','jaxqtl_nb_score_spa_acat')]):
    paths=sorted((a.runs/folder).glob('part*/chr22.*.parquet.gz'))
    d=pl.concat([pl.read_parquet(x) for x in paths],how='vertical_relaxed').to_pandas()
    valid=d.result_valid.fillna(False)&d.model_converged.fillna(False)
    if 'perm_converged' in d: valid &= d.perm_converged.fillna(False)
    frames[name]=pd.DataFrame({'id':d.phenotype_id,'p':d.pvalue_adj.where(valid),'lead':d.snp})
result=None
for name,d in frames.items():
    d=d.set_index('id');d.index.name='phenotype_id'
    if not d.index.is_unique: raise ValueError('duplicate phenotype in '+name)
    d['valid']=np.isfinite(d.p)&d.p.between(0,1)
    q,_=qvalues(d.p[d.valid]);d['q']=q
    d=d.rename(columns={k:name+'.'+k for k in d.columns})
    result=d if result is None else result.join(d,how='outer')
result.to_csv(a.output,sep='\t')
