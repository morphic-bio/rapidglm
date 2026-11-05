#!/usr/bin/env python3
"""Initialize fresh GLM command coverage from the same verified inputs as job 687."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2];W=ROOT/'validation/work/release-1.6.0-gpu'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sys.path.insert(0,str(ROOT/'benchmarks/glm'));import run_comparison as m
 problems=ROOT/'validation/work/benchmark/runs-v1.6.0-gpu/glm-compare/problems-v1.6.0'
 jobs=[];inputs={}
 for p in sorted(problems.iterdir()):
  if not (p/'meta.json').exists():continue
  meta=json.loads((p/'meta.json').read_text())
  if meta['suite']!='speed':continue
  for name in ['meta.json','X.f64','y.f64','offset.f64','weights.f64']:inputs[str((p/name).relative_to(ROOT))]=sha(p/name)
  for tool in m.TOOLS:
   if meta['family'] in m.TOOLS[tool][1]:jobs.append(dict(label='GLM '+p.name+' '+tool,problem=str(p.relative_to(ROOT)),tool=tool))
 assert len(jobs)==408
 spec=dict(commit='4505ef6afaa00a38c5c871ae0e12d531c91ec1cf',jobs=jobs,input_sha256=inputs)
 target=W/'glm_speed_jobs.json';assert not target.exists();target.write_text(json.dumps(spec,indent=2)+'\n')
 state=dict(spec,accepted={},invalidated=[],supporting_attempts=[],complete=False)
 (R/'GLM_SPEED_RESUME.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
