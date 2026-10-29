"""Add CPU-policy checks to the frozen quiet-start guards without editing measured code (1.5.0).
As in validation/receipts/v1.4.0/policy_hook, with the entry points of the measured clone, which the
1.5.0 runner runs from. Profiling is removed at the first audit call, before any measurement starts.
Reference-tool and fit entry points do not install the hook.
"""
import os, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
CLONE = Path('/data/lhhung/rapidglm')
ENTRIES = {CLONE / p for p in ['scripts/quiet_run.py', 'scripts/benchmark_applications.py',
    'scripts/validate_bulk.py', 'scripts/benchmark_untestable_pairs.py', 'scripts/replogle_pilot.py',
    'benchmarks/glm/batch.py', 'benchmarks/glm/run_comparison.py',
    'bindings/rust/tests/compare_pydeseq2.py']} | {R / 'guarded.py', R / 'vmath_e2e.py'}


def install():
    if not os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED'):
        return
    entry = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if entry not in ENTRIES:
        return
    if entry.name == 'run_comparison.py' and '--speed' not in sys.argv:
        return
    sys.path.insert(0, str(R))
    from cpu_policy import check_quiet_start
    target = CLONE / 'scripts/quiet_run.py'

    def attach(frame, event, arg):
        if event != 'call' or frame.f_code.co_name != 'audit':
            return
        if Path(frame.f_code.co_filename).resolve() != target:
            return
        original = frame.f_globals['audit']

        def audited(record):
            if record.get('event') == 'quiet_ready':
                check_quiet_start(record)
            return original(record)
        frame.f_globals['audit'] = audited
        sys.setprofile(None)
        record = frame.f_locals['record']
        if record.get('event') == 'quiet_ready':
            check_quiet_start(record)
    sys.setprofile(attach)


install()
