#!/usr/bin/env python3
"""Check filtering rules against saved public DESeq2 black-box result metadata."""
import argparse,json,pathlib,subprocess
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
report={'version':subprocess.check_output([str(ROOT/'build/rapidglm'),'--version'],text=True).strip(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'reference_rules':{}}
for name in ('airway','kang','sim3','sim3b','sim8'):
    path=ROOT/'validation/work/bulk'/name/'deseq2/reference.json';r=json.loads(path.read_text())
    rej=np.array([v['numRej'] for v in r['filter_num_rej']]);theta=np.array([v['theta'] for v in r['filter_num_rej']])
    smooth=np.array(r['lo_fit']['y']);positive=rej>0
    rms=float(np.sqrt(np.mean((rej[positive]-smooth[positive])**2))) if positive.any() else 0.
    chosen=0 if max(rej)<=10 else next((i for i,v in enumerate(rej) if v>max(smooth)-rms),0)
    assert abs(theta[chosen]-r['filter_theta'])<1e-12,(name,theta[chosen],r['filter_theta'])
    report['reference_rules'][name]={'reference_metadata':str(path.relative_to(ROOT)),
        'maximum_rejections':int(max(rej)),'positive_residual_rms':rms,
        'theta_calculated':float(theta[chosen]),'theta_DESeq2':r['filter_theta']}
paths={'old':ROOT/'validation/work/replogle/k562-essential/pilot-v1.4.0/pseudobulk/datasets/HNRNPR/rapidglm.tsv',
       'new':pathlib.Path('/tmp/rapidglm-hnrnpr-new.tsv'),
       'reference':pathlib.Path('/tmp/rapidglm-hnrnpr-reference/genes.tsv')}
hits={name:set((t:=pd.read_csv(path,sep='\t')).gene[t.padj<.1]) for name,path in paths.items()}
ref=json.loads(pathlib.Path('/tmp/rapidglm-hnrnpr-reference/reference.json').read_text())
ours=json.loads(pathlib.Path('/tmp/rapidglm-hnrnpr-new.run.json').read_text())
assert ref['filter_theta']==ours['filter_theta']==0
assert abs(ref['filter_threshold']-ours['filter_threshold'])<1e-12
report['HNRNPR']={'reference_version':ref['version'],'reference_theta':ref['filter_theta'],
    'rapidglm_theta':ours['filter_theta'],'reference_threshold':ref['filter_threshold'],
    'rapidglm_threshold':ours['filter_threshold'],'discoveries':{name:len(v) for name,v in hits.items()},
    'old_jaccard':len(hits['old']&hits['reference'])/len(hits['old']|hits['reference']),
    'new_jaccard':len(hits['new']&hits['reference'])/len(hits['new']|hits['reference'])}
a.output.write_text(json.dumps(report,indent=2)+'\n')
