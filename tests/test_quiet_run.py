import importlib.util
import pathlib
import json
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('quiet_run', pathlib.Path(__file__).resolve().parents[1] / 'scripts/quiet_run.py')
quiet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quiet)


class QuietHost(unittest.TestCase):
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
