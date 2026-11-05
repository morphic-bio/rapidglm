import importlib.util
import pathlib
import json
import tempfile
import unittest
import types
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('quiet_run', pathlib.Path(__file__).resolve().parents[1] / 'scripts/quiet_run.py')
quiet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quiet)


class QuietHost(unittest.TestCase):
    def test_release_monitor_has_explicit_begin_and_end(self):
        monitor = types.SimpleNamespace(begin=lambda *args: None, finish=lambda: None)
        with patch.dict(quiet.os.environ, {"RAPIDGLM_MEASUREMENT_GUARD": "audit_for_test"}), \
             patch.object(quiet.importlib, "import_module", return_value=monitor), \
             patch.object(monitor, "begin") as begin, patch.object(monitor, "finish", return_value={"ok": True}) as finish, \
             patch.object(quiet.os, "getloadavg", return_value=(0., 0., 0.)), \
             patch.object(quiet, "cpu_snapshot", side_effect=[{6: (i*100, 0)} for i in range(4)]), \
             patch.object(quiet.time, "sleep"):
            quiet.wait_quiet("6", label="measured fit")
            begin.assert_called_once_with({6}, "measured fit")
            self.assertEqual(quiet.finish_measurement(), {"ok": True})
            finish.assert_called_once()

    def test_busy_selected_cpu_blocks_even_when_global_load_is_low(self):
        # Three sampling windows: one busy, then two consecutive quiet windows.
        samples = [(0, 0), (100, 90), (100, 90), (200, 95), (200, 95), (300, 100)]
        # One saturated core among eight must block, even though aggregate
        # utilization (11.25%) would pass the former 20% average test.
        snapshots = [{cpu: (total, busy if cpu == 16 else 0) for cpu in range(16, 24)}
                     for total, busy in samples]
        with patch.object(quiet.os, 'getloadavg', return_value=(1., 1., 1.)), \
             patch.object(quiet, 'cpu_snapshot', side_effect=snapshots), \
             patch.object(quiet.time, 'sleep') as sleep:
            quiet.wait_quiet('16-23')
            self.assertEqual(sleep.call_count, 3)

    def test_busy_host_blocks_even_when_selected_cpus_are_idle(self):
        loads = [8., 8., 1., 1., 1., 1.]
        with patch.object(quiet.os, 'getloadavg', side_effect=[(v, v, v) for v in loads]), \
             patch.object(quiet.time, 'sleep') as sleep:
            quiet.wait_quiet()
            self.assertEqual(sleep.call_count, 3)


    def test_busy_sibling_blocks_when_siblings_are_checked(self):
        # CPU 6 idle throughout; its sibling 30 busy in the first window only.
        samples = [(0, 0), (100, 90), (100, 90), (200, 95), (200, 95), (300, 100)]
        snapshots = [{6: (total, 0), 30: (total, busy)} for total, busy in samples]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'timing.jsonl'
            with patch.dict(quiet.os.environ, {'RAPIDGLM_QUIET_SIBLINGS': '1',
                                               'RAPIDGLM_TIMING_AUDIT': str(output)}), \
                 patch.object(quiet, 'thread_siblings', return_value={30}), \
                 patch.object(quiet.os, 'getloadavg', return_value=(1., 1., 1.)), \
                 patch.object(quiet, 'cpu_snapshot', side_effect=snapshots), \
                 patch.object(quiet.time, 'sleep') as sleep:
                quiet.wait_quiet('6')
                self.assertEqual(sleep.call_count, 3)
            record = json.loads(output.read_text())
            self.assertEqual(record['cpus'], [6])
            self.assertEqual(record['sibling_cpus'], [30])

    def test_siblings_are_not_checked_by_default(self):
        snapshots = [{16: (i * 100, 0)} for i in range(4)]
        with patch.dict(quiet.os.environ, {}, clear=False), \
             patch.object(quiet, 'thread_siblings', side_effect=AssertionError), \
             patch.object(quiet.os, 'getloadavg', return_value=(1., 1., 1.)), \
             patch.object(quiet, 'cpu_snapshot', side_effect=snapshots), \
             patch.object(quiet.time, 'sleep') as sleep:
            quiet.os.environ.pop('RAPIDGLM_QUIET_SIBLINGS', None)
            quiet.wait_quiet('16')
            self.assertEqual(sleep.call_count, 2)

    def test_audit_retains_both_quiet_samples_and_stage(self):
        snapshots = [{16: (i * 100, i * 5)} for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'timing.jsonl'
            with patch.dict(quiet.os.environ, {'RAPIDGLM_TIMING_AUDIT': str(output),
                                               'RAPIDGLM_STAGE': 'eqtl'}), \
                 patch.object(quiet.os, 'getloadavg', return_value=(1., 1., 1.)), \
                 patch.object(quiet, 'cpu_snapshot', side_effect=snapshots), \
                 patch.object(quiet.time, 'sleep'):
                quiet.wait_quiet('16', label='Gaussian QTL')
            record = json.loads(output.read_text())
            self.assertEqual(record['stage'], 'eqtl')
            self.assertEqual(record['label'], 'Gaussian QTL')
            self.assertEqual(record['cpus'], [16])
            self.assertEqual(len(record['samples']), 2)
            self.assertTrue(all(s['passed'] and s['busy_fractions']['16'] == .05
                                for s in record['samples']))


if __name__ == '__main__':
    unittest.main()
