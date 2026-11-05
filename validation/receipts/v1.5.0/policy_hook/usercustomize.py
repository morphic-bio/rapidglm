"""Add the full-screen memory gate to the unchanged, untimed seed driver (as in 1.4.0)."""
import os
import sys

if (os.environ.get('RAPIDGLM_CPU_POLICY_ENABLED') == '1'
        and os.path.basename(sys.argv[0]) == 'validate_perturbation_seeds.py'):
    import json
    import subprocess
    import time
    from pathlib import Path
    from quiet_run import audit, wait_memory

    _root = Path('/data/lhhung/rapidglm')
    _run = subprocess.run

    def _guarded_run(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 3
                and Path(os.fspath(command[0])).resolve() == _root / 'build/rapidglm'
                and command[1] == 'sceptre'):
            wait_memory(100)
            config = Path(command[2])
            audit(dict(event='seed_full_screen_command_ready',
                       utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       seed=json.loads(config.read_text())['seed'],
                       config=str(config.resolve()),
                       cpus=sorted(os.sched_getaffinity(0)),
                       minimum_available_gib=100, untimed=True))
        return _run(command, *args, **kwargs)

    subprocess.run = _guarded_run
