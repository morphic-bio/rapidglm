#!/usr/bin/env python3
"""Untimed launcher and acceptance regressions for nested driver measurements."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

R = Path(__file__).resolve().parents[1]
SOURCE = (R / "run_accepted.sh").read_text()
spec = importlib.util.spec_from_file_location("receipt_checker_scope", R / "check_run_accepted.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DriverScope(unittest.TestCase):
    def launch(self, helper, failing=False):
        definition = next(line for line in SOURCE.splitlines() if line.startswith(helper + "()"))
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            (work / "scripts").mkdir()
            (work / "scripts/quiet_run.py").write_text(
                "import json,os,subprocess,sys\n"
                "def log(event):\n"
                " with open(os.environ['EVIDENCE'],'a') as f:f.write(json.dumps({'event':event})+'\\n')\n"
                "log('outer_start')\n"
                "args=sys.argv[sys.argv.index('--')+1:]\n"
                "code=subprocess.call(args)\n"
                "log('outer_end')\n"
                "raise SystemExit(code)\n"
            )
            (work / "inner.py").write_text(
                "import json,os,sys\n"
                "assert set(os.sched_getaffinity(0)) <= set(range(6))\n"
                "def log(event):\n"
                " with open(os.environ['EVIDENCE'],'a') as f:f.write(json.dumps({'event':event})+'\\n')\n"
                "if os.environ.get('WHOLE_SCREEN'):\n"
                " log('screen_work')\n"
                "else:\n"
                " for i in range(2):\n"
                "  log('inner_start')\n"
                "  if i==1 and os.environ.get('INJECT_FAIL'):sys.exit(23)\n"
                "  log('inner_end')\n"
            )
            env = dict(os.environ, EVIDENCE=str(work / "events.jsonl"))
            if failing:
                env["INJECT_FAIL"] = "1"
            if helper == "screen_timed":
                env["WHOLE_SCREEN"] = "1"
            program = (
                'set -euo pipefail\nLOCK="$PWD/lock"\nPCORES=0-5\nALL=0-5\n'
                + definition + '\n' + helper + ' "$1" inner.py\n'
            )
            result = subprocess.run(
                ["bash", "-c", program, "scope-test", sys.executable], cwd=work, env=env,
                capture_output=True, text=True
            )
            events = [json.loads(line)["event"] for line in (work / "events.jsonl").read_text().splitlines()]
            return result, events

    def test_driver_has_only_inner_measurements(self):
        result, events = self.launch("driver")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events, ["inner_start", "inner_end"] * 2)

    def test_inner_failure_propagates(self):
        result, events = self.launch("driver", failing=True)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(events, ["inner_start", "inner_end", "inner_start"])

    def test_full_screen_keeps_sole_outer_guard(self):
        result, events = self.launch("screen_timed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events, ["outer_start", "screen_work", "outer_end"])

    def test_stage_routes(self):
        names = ["batch", "bulkval", "untestable", "replogle"]
        for stage in names:
            body = re.search(r"\n    " + stage + r"\).*?(?=\n    [a-z_]+\)|\n    \*\))",
                             SOURCE, re.S).group()
            self.assertNotIn("quiet_run.py", body, stage)
            self.assertIn("driver python3", body, stage)
        self.assertIn("screen_timed python3 scripts/replogle_pilot.py full ", SOURCE)

    def test_batch_coverage_and_nested_regression(self):
        labels = {f"batch {f} native threads={t}" for f in ["poisson", "nb"] for t in [1, 4, 16]}
        labels |= {f"batch {f} Python {t}" for f in ["poisson", "nb"] for t in ["statsmodels", "rapidglm.fit"]}
        labels |= {"batch poisson R glm.fit", "batch poisson R rapidglm", "batch nb R glm.nb",
                   "batch nb R rapidglm", "batch nb glmGamPoi"}
        rows = [{"label": x} for x in sorted(labels)]
        self.assertEqual(checker.inner_command_problems("batch", rows), [])
        self.assertTrue(checker.inner_command_problems("batch", rows[:-1]))
        self.assertTrue(checker.inner_command_problems("batch", rows[:-1] + [rows[0]]))
        outer = {"label": "scripts/quiet_run.py --cpus 6-21 -- taskset -c 0-5 python3 benchmarks/glm/batch.py"}
        errors = checker.inner_command_problems("batch", rows + [outer])
        self.assertTrue(any("driver-wide" in x[0] for x in errors))

    def test_other_driver_coverage_and_full_screen_exception(self):
        for stage, count in [("bulkval", 60), ("untestable", 12), ("replogle", 83)]:
            rows = [{"label": "inner " + str(i)} for i in range(count)]
            self.assertEqual(checker.inner_command_problems(stage, rows), [])
            self.assertTrue(checker.inner_command_problems(stage, rows[:-1]))
            self.assertTrue(checker.inner_command_problems(
                stage, rows + [{"label": "scripts/quiet_run.py driver"}]))
        self.assertEqual(checker.inner_command_problems(
            "replogle_full", [{"label": "scripts/quiet_run.py full screen"}] * 4), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
