#!/usr/bin/env python3
"""Untimed precision checks against public DESeq2 exports from reference_bulk.R."""
import argparse, hashlib, json, pathlib, subprocess
import numpy as np
import pandas as pd
from scipy.io import mmread
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',type=pathlib.Path,required=True)
p.add_argument('--data',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();w=a.work;r=w/'deseq2'
x=pd.read_csv(w/'native.tsv',sep='\t');y=pd.read_csv(r/'genes.tsv',sep='\t')
assert np.array_equal(x.gene,y.gene)
ref=json.loads((r/'reference.json').read_text())
sf=np.asarray(json.loads((w/'native.run.json').read_text())['size_factors'])
metrics={}
for key,left,right in [('size_factors',sf,np.asarray(ref['size_factors'])),('dispersions',x.dispersion.to_numpy(),y.dispersion.to_numpy()),('fold_changes',x.log2FoldChange.to_numpy(),y.log2FoldChange.to_numpy()),('p_values',x.pvalue.to_numpy(),y.pvalue.to_numpy())]:
 good=np.isfinite(left)&np.isfinite(right)&(right!=0)
 err=np.abs(left[good]-right[good])/np.abs(right[good])
 metrics[key]={'compared':int(good.sum()),'excluded':int((~good).sum()),'median_relative':float(np.median(err)),'q95_relative':float(np.quantile(err,.95)),'max_relative':float(err.max())}
Y=mmread(a.data/'counts.mtx').toarray();mu=np.loadtxt(r/'mu.tsv');H=np.loadtxt(r/'H.tsv');D=np.loadtxt(r/'cooks.tsv')
X=pd.read_csv(a.data/'design.tsv',sep='\t').to_numpy(float);q=X.shape[1]
replaced=y['replace'].fillna(False).to_numpy(bool) if 'replace' in y else np.any(np.loadtxt(r/'replace_counts.tsv')!=Y,axis=1)
with np.errstate(all='ignore'): alpha=((Y-mu)**2*H/(q*D*(1-H)**2)-mu)/(mu**2)
eligible=np.isfinite(alpha)&np.isfinite(H)&np.isfinite(D)&(D>0)&(mu>0)&(H>0)&(H<1)&(~replaced[:,None])
inversion={}
for name,mask in [('genes_without_public_replace_flag',eligible),('no_replace_flag_and_all_means_at_least_0_5',eligible&np.all(mu>=.5,axis=1)[:,None])]:
 ranges=[];absolute=[];observations=0;negative=0
 for i in np.flatnonzero(mask.sum(axis=1)>=2):
  v=alpha[i,mask[i]];med=np.median(v);ranges.append(np.ptp(v)/max(abs(med),1e-300));absolute.append(np.ptp(v));observations+=len(v);negative+=int(np.sum(v<0))
 inversion[name]={'genes':len(ranges),'sample_values':observations,'negative_inverted_values':negative,'median_within_gene_relative_range':float(np.median(ranges)),'q95_within_gene_relative_range':float(np.quantile(ranges,.95)),'max_within_gene_relative_range':float(max(ranges)),'max_within_gene_absolute_range':float(max(absolute)),'all_relative_ranges_at_most_1e_12':bool(max(ranges)<=1e-12)}
out={'version':subprocess.check_output(['build/rapidglm','--version'],text=True).strip(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'reference_version':ref['version'],'untimed':True,'cpus':'0-7','timing_lock':False,'timing_usable':False,'dataset':'GTEx thyroid128','counts_sha256':hashlib.sha256((a.data/'counts.mtx').read_bytes()).hexdigest(),'native_config':json.loads((w/'native.json').read_text()),'reference_command':'taskset -c 0-7 env OPENBLAS_NUM_THREADS=1 Rscript scripts/reference_bulk.R validation/work/benchmark/thyroid128 sex[T.female] WORK/deseq2 LIBS','relative_difference_definition':'abs(native-reference)/abs(reference), finite pairs with nonzero reference; no post-hoc removal of large errors','comparison':metrics,'all_four_medians_at_most_1e_4':bool(all(v['median_relative']<=1e-4 for v in metrics.values())),'cooks_inversion_formula':'alpha_rob=((y-mu)^2*H/(p*D*(1-H)^2)-mu)/mu^2; p is the number of design columns; public assays mu,H,cooks; genes marked by the public replace flag are excluded because cooks are retained from the original fit; that flag need not imply changed counts','replacement_flagged_genes_excluded':int(replaced.sum()),'genes_with_changed_replace_counts':int(np.any(np.loadtxt(r/'replace_counts.tsv')!=Y,axis=1).sum()),'cooks_inversion':inversion,'scope':'Current dev.3 versus DESeq2 1.48.2. This does not recreate the unsaved 2026-10-01 analysis or establish a universal precision bound.'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'comparison':metrics,'cooks_inversion':inversion},indent=2))
