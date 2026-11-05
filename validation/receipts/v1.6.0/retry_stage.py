"""Preserve and repeat an entire untestable stage after verified contention."""
import hashlib,json,shutil,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2];W=ROOT/'validation/work/release-1.6.0-gpu'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def retry_untestable(row,commit):
 log=R/'untestable.log';marker='RuntimeError: measurement exceeded 5% contention; rerun required: '
 if not log.exists():return False
 lines=log.read_text().splitlines();last=next((x for x in reversed(lines) if x.strip()),'')
 if not last.startswith(marker):return False
 audit=Path(last[len(marker):]);assert audit.parent==R/'commands'
 bad=json.loads(audit.read_text());sup=R/('supervisor-'+row['label']+'.json');supervisor=json.loads(sup.read_text())
 assert supervisor['cpu_policy_unchanged'] and supervisor['cpu_policy_maximum_poll_gap_seconds']<=5
 assert bad['stage']=='untestable' and bad['slurm_job_id']==row['slurm_job_id'] and bad['policy_unchanged'] and bad['contended']
 assert row['started_utc']<=bad['started_utc']<=bad['finished_utc']<=row['finished_utc']
 assert bad['others_cpu']>.05 or any(x>.05 for x in bad['sibling_busy_fraction'].values())
 work=ROOT/'validation/work/benchmark/runs-v1.6.0-gpu/untestable';out=work/'runs'
 inputs={str(p.relative_to(ROOT)):sha(p) for p in work.iterdir() if p.is_file()}
 target=W/'stage-reruns'/row['label'];target.mkdir(parents=True,exist_ok=False)
 outputs=[p for p in out.rglob('*') if p.is_file()]
 files=set(outputs)|{log,sup,audit,R/'chain-status.json',R/'source.json'}
 for name in ['UNTESTABLE_PAIRS.json','STAGE_RERUNS.json']:
  if (R/name).exists():files.add(R/name)
 manifest=[]
 for p in sorted(files):
  rel=p.relative_to(ROOT);to=target/rel;to.parent.mkdir(parents=True,exist_ok=True);before=sha(p)
  shutil.copyfile(p,to);assert sha(to)==before
  manifest.append(dict(original=str(rel),copy=str(to.relative_to(ROOT)),sha256=before))
 m=target/'MANIFEST.json';m.write_text(json.dumps(dict(files=manifest,input_sha256=inputs,all_copies_sha256_verified=True),indent=2)+'\n')
 rec=dict(stage='untestable',commit=commit,prior_exit=row['exit'],invalidated_label=row['label'],started_utc=row['started_utc'],finished_utc=row['finished_utc'],reason='Verified 5% contention rejection. Entire incomplete stage is discarded and repeated with identical source and inputs.',retained_stage_timings=0,archive_path=str(target.relative_to(ROOT)),archive_manifest_sha256=sha(m),invalidated_command=dict(audit_file=str(audit.relative_to(R)),audit_sha256=sha(audit)),numerical_source_inputs_and_thresholds_unchanged=True)
 p=R/'STAGE_RERUNS.json';reruns=json.loads(p.read_text());reruns['reruns'].append(rec);p.write_text(json.dumps(reruns,indent=2)+'\n')
 (R/'UNTESTABLE_RETRY.json').write_text(json.dumps(rec,indent=2)+'\n')
 expected={x['original']:x['sha256'] for x in manifest}
 for p in outputs:
  assert sha(p)==expected[str(p.relative_to(ROOT))];p.unlink()
 assert all(sha(ROOT/n)==h for n,h in inputs.items())
 return True
