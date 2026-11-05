#!/usr/bin/env python3
"""Exercise ablation routing and the pilot's existing inner measurement guard."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
ROOT = Path('/data/lhhung/rapidglm-1.6.0')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

ablations = module('ablation_scope_driver', R / 'ablations.py')
checker = module('ablation_scope_checker', R / 'check_run_accepted.py')
pilot = module('ablation_scope_pilot', ROOT / 'scripts/replogle_pilot.py')

class CapturedCommand(Exception):
    pass

class AblationScope(unittest.TestCase):
    def test_replogle_driver_is_launched_without_outer_measurement(self):
        calls = []
        def capture(command):
            calls.append(command)
            raise CapturedCommand()
        with patch.dict(os.environ, SLURM_JOB_ID='scope-test', RAPIDGLM_CPU_POLICY_ENABLED='1', RAPIDGLM_TIMING_LOCK='/tmp/ablation-scope-fixture.lock'):
            with patch.object(ablations, 'load', return_value={'commit': 'fixture'}):
                with patch.object(ablations, 'run', capture):
                    with self.assertRaises(CapturedCommand):
                        ablations.main()
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[:8], ['flock', '/tmp/ablation-scope-fixture.lock', 'taskset', '-c', '0-5', 'python3', 'scripts/replogle_pilot.py', 'sc'])
        for option, value in [('--cpus', '6-21'), ('--threads', '16'), ('--null-start', 'cold'), ('--fit-check', 'strict')]:
            self.assertEqual(command[command.index(option) + 1], value)

    def guarded_pilot(self, reject=False):
        events = []
        guard = types.ModuleType('quiet_run')
        guard.wait_quiet = lambda cpus: events.append(('quiet', cpus))
        def finish():
            events.append(('finish',))
            if reject:
                raise RuntimeError('injected contention rejection')
        guard.finish_measurement = finish
        def measure(command, cpus):
            events.append(('measure', command, cpus))
            return {'wall_seconds': 1.25}
        with patch.dict(os.environ, RAPIDGLM_QUIET_LOAD='8'):
            with patch.dict(sys.modules, quiet_run=guard):
                with patch.object(pilot, 'measure', measure):
                    if reject:
                        with self.assertRaisesRegex(RuntimeError, 'contention rejection'):
                            pilot.timed(['fixture-fit'], '6-21')
                    else:
                        self.assertEqual(pilot.timed(['fixture-fit'], '6-21'), {'wall_seconds': 1.25})
        return events

    def test_existing_pilot_guard_covers_actual_command(self):
        self.assertEqual(self.guarded_pilot(), [('quiet', '6-21'), ('measure', ['fixture-fit'], '6-21'), ('finish',)])

    def test_existing_pilot_contention_rejection_propagates(self):
        self.assertEqual(self.guarded_pilot(reject=True)[-1], ('finish',))

    def test_five_ablation_commands_required_and_nested_driver_rejected(self):
        rows = [{'label': 'fit ' + str(i)} for i in range(5)]
        self.assertEqual(checker.inner_command_problems('ablations', rows), [])
        self.assertTrue(checker.inner_command_problems('ablations', rows[:-1]))
        outer = {'label': 'scripts/quiet_run.py --cpus 6-21 -- python3 scripts/replogle_pilot.py sc'}
        self.assertTrue(checker.inner_command_problems('ablations', rows + [outer]))
        self.assertTrue(any('driver-wide' in x[0] for x in checker.inner_command_problems('ablations', rows[:-1] + [outer])))

if __name__ == '__main__':
    unittest.main(verbosity=2)
