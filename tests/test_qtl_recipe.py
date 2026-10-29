"""The published QTL comparison must work without ignored work/scripts files.

The Rscript boundary is controlled here; qvalue mathematics is tested by the
live reference comparisons, while this test checks dependency and work routing.
"""
import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "scripts/compare_eqtl.py").is_file())

class QTLRecipe(unittest.TestCase):
    def test_qvalues_with_an_isolated_work_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            (work / "tools/Rlib").mkdir(parents=True)
            bindir = work / "bin"
            bindir.mkdir()
            runner = bindir / "Rscript"
            runner.write_text("""#!/usr/bin/env python3
import csv, os, pathlib, sys
script, source, target, library = sys.argv[1:]
assert pathlib.Path(script).is_file(), "comparison wrapper missing"
expected = pathlib.Path(os.environ["EQTL_WORK"]) / "tools/Rlib"
assert pathlib.Path(library).resolve() == expected.resolve(), "configured work library ignored"
with open(source) as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
with open(target, "w") as stream:
    writer = csv.DictWriter(stream, fieldnames=["id", "q"], delimiter="\t")
    writer.writeheader()
    writer.writerows({"id": r["id"], "q": r["p"]} for r in rows)
print("pi0 1")
""")
            runner.chmod(0o755)
            with patch.dict(os.environ, {"EQTL_WORK": str(work),
                             "PATH": str(bindir) + os.pathsep + os.environ["PATH"]}):
                spec = importlib.util.spec_from_file_location("compare_eqtl_recipe", ROOT / "scripts/compare_eqtl.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                p = pd.Series([.01, .2, .8], index=["gene1", "gene2", "gene3"])
                q, pi0 = module.qvalues(p)
            self.assertFalse((work / "scripts").exists())
            self.assertEqual(list(q.index), list(p.index))
            np.testing.assert_array_equal(q.to_numpy(), p.to_numpy())
            self.assertEqual(pi0, 1.)

if __name__ == "__main__":
    unittest.main()
