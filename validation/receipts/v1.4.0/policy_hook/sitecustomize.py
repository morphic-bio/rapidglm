"""Add policy checks to frozen quiet-start guards without editing measured code.
Profiling is removed at the first audit call, before any measurement starts.
Reference-tool and fit entry points do not install the hook.
"""
import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2]
ENTRIES={ROOT/p for p in ['scripts/quiet_run.py','scripts/benchmark_applications.py',
    'scripts/validate_bulk.py','scripts/benchmark_untestable_pairs.py','scripts/replogle_pilot.py',
    'benchmarks/glm/batch.py','benchmarks/glm/run_comparison.py',
    'bindings/rust/tests/compare_pydeseq2.py','validation/receipts/v1.4.0/guarded.py']}

def install():
    if not os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED'):return
    entry=Path(sys.argv[0]).resolve()
    if entry not in ENTRIES:return
    if entry.name=='run_comparison.py' and '--speed' not in sys.argv:return
    sys.path.insert(0,str(R))
    from cpu_policy import check_quiet_start
    target=ROOT/'scripts/quiet_run.py'
    def attach(frame,event,arg):
        if event!='call' or frame.f_code.co_name!='audit':return
        if Path(frame.f_code.co_filename).resolve()!=target:return
        original=frame.f_globals['audit']
        def audited(record):
            if record.get('event')=='quiet_ready':check_quiet_start(record)
            return original(record)
        frame.f_globals['audit']=audited
        sys.setprofile(None)
        record=frame.f_locals['record']
        if record.get('event')=='quiet_ready':check_quiet_start(record)
    sys.setprofile(attach)
install()
