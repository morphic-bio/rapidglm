#!/usr/bin/env python3
"""Check measured library bytes and the frozen artifacts before and after every stage."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
MEASURED='4505ef6afaa00a38c5c871ae0e12d531c91ec1cf'
PROTECTED=['include','src','CMakeLists.txt','cmake','bindings']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 source=json.loads((R/'source.json').read_text());assert source['commit']==MEASURED
 if 'preprint_plan_sha256' in source:assert sha(R/'PREPRINT_PLAN.json')==source['preprint_plan_sha256'],'focused plan changed'
 tag=subprocess.check_output(['git','rev-parse','v1.6.0-dev.2^{}'],cwd=ROOT,text=True).strip();assert tag==MEASURED
 subprocess.run(['git','merge-base','--is-ancestor',MEASURED,'HEAD'],cwd=ROOT,check=True)
 subprocess.run(['git','diff','--exit-code',MEASURED,'--',*PROTECTED],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 names=subprocess.check_output(['git','ls-tree','-r','--name-only',MEASURED,'--',*PROTECTED],cwd=ROOT,text=True).splitlines()
 assert set(names)==set(source['library_sources_sha256'])
 for name,expected in source['library_sources_sha256'].items():assert sha(ROOT/name)==expected,'library source changed: '+name
 for field in ['binaries_sha256','harness_sha256']:
  for name,expected in source[field].items():
   p=Path(name) if name.startswith('/') else ROOT/name
   assert sha(p)==expected,'frozen artifact changed: '+name
 for name,expected in source['container_images'].items():
  actual=subprocess.check_output(['docker','image','inspect',name,'--format','{{.Id}}'],text=True).strip()
  assert actual==expected,'image changed: '+name
if __name__=='__main__':main()
