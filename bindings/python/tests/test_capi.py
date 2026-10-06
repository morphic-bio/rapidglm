"""The Python layer over librapidglm_c must reproduce the CLI's tables exactly.

Usage: RAPIDGLM_C_LIBRARY=build/librapidglm_c.so \
       python3 bindings/python/tests/test_capi.py build/rapidglm
"""

import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

import rapidglm

BINARY = pathlib.Path(sys.argv[1]).resolve()
sys.argv = sys.argv[:1]


def write_market(path, matrix):
    """Observations x features in, feature-by-observation MatrixMarket out (CLI layout)."""
    m = sp.coo_matrix(matrix)
    lines = [f"{j + 1} {i + 1} {v:.17g}" for i, j, v in zip(m.row, m.col, m.data) if v != 0]
    path.write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        f"{m.shape[1]} {m.shape[0]} {len(lines)}\n" + "".join(line + "\n" for line in lines)
    )
    return str(path)


def write_design(path, x):
    path.write_text("".join("\t".join(f"{v:.17g}" for v in row) + "\n" for row in np.atleast_2d(x)))
    return str(path)


class CAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, command, **config):
        config["output"] = str(self.root / "result")
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        r = subprocess.run([BINARY, command, path], text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        table = pd.read_csv(self.root / "result.tsv", sep="\t", dtype=str, keep_default_na=False)
        record = json.loads((self.root / "result.run.json").read_text())
        return table, record

    def assert_same(self, cli, frame, text=()):
        """Every CLI column equals the Python column bit for bit (NA = non-finite)."""
        self.assertEqual(len(cli), len(frame))
        for column in cli.columns:
            self.assertIn(column, frame.columns)
            expected, got = cli[column].tolist(), frame[column].tolist()
            for row, (e, g) in enumerate(zip(expected, got)):
                where = f"{column}[{row}]"
                if column in text:
                    self.assertEqual(e, str(g), where)
                elif e == "NA":
                    self.assertFalse(math.isfinite(g), where)
                else:
                    self.assertEqual(float(e), float(g), where)

    def bulk_data(self):
        rng = np.random.default_rng(11)
        samples, genes = 10, 150
        group = np.repeat([0, 1], samples // 2)
        batch = np.tile([0, 1], samples // 2)
        x = np.column_stack([np.ones(samples), batch, group])
        base = rng.gamma(2, 40, genes)
        effect = np.where(np.arange(genes) % 6 == 0, 2.5, 1.0)
        mu = base[None, :] * np.where(group[:, None] == 1, effect[None, :], 1) * (1 + 0.2 * batch[:, None])
        counts = rng.negative_binomial(8, 8 / (8 + mu)).astype(float)
        counts[:, 3] = 0  # an all-zero gene
        counts[0, 7] = 4000  # an outlier count
        return x, counts

    def test_bulk_de_matches_cli(self):
        x, counts = self.bulk_data()
        files = dict(design=write_design(self.root / "design.tsv", x),
                     responses=write_market(self.root / "counts.mtx", counts), family="nb")
        cases = [
            (dict(coefficient=2, lfc_shrink="normal", threads=3), dict(coefficient=2, lfc_shrink="normal", threads=3)),
            (dict(contrast=[0, 1, -1]), dict(contrast=[0, 1, -1])),
            (dict(coefficients=[1, 2], test="lrt"), dict(coefficients=[1, 2], test="lrt")),
            (dict(coefficient=2, legacy_cooks=True), dict(coefficient=2, legacy_cooks=True)),
            (dict(test="lrt", size_factors="poscounts", trend="local", coefficient=2),
             dict(test="lrt", size_factors="poscounts", trend="local", coefficient=2)),
        ]
        for config, arguments in cases:
            with self.subTest(**{k: str(v) for k, v in arguments.items()}):
                table, record = self.cli("bulk-de", **files, **config)
                frame = rapidglm.bulk_de(x, sp.csr_matrix(counts), **arguments)
                self.assert_same(table, frame, text=("status",))
                np.testing.assert_array_equal(frame.attrs["size_factors"], record["size_factors"])
                self.assertEqual(frame.attrs["prior_variance"], record["prior_variance"])
        # DataFrame input names the genes.
        named = pd.DataFrame(counts, columns=[f"g{j}" for j in range(counts.shape[1])])
        frame = rapidglm.bulk_de(x, named)
        self.assertEqual(frame.index[5], "g5")

    def test_fit_batch_and_scan_match_cli(self):
        x, counts = self.bulk_data()
        files = dict(design=write_design(self.root / "design.tsv", x),
                     responses=write_market(self.root / "counts.mtx", counts[:, :40]), family="nb")
        table, _ = self.cli("fit", **files, threads=2)
        self.assert_same(table, rapidglm.fit_batch(x, counts[:, :40], threads=2), text=("status",))
        rng = np.random.default_rng(4)
        predictors = np.column_stack([x[:, 2], rng.normal(size=len(x))])
        design = x[:, :2]
        files["design"] = write_design(self.root / "design2.tsv", design)
        files["predictors"] = write_market(self.root / "predictors.mtx", predictors)
        for calibration in ("permutation", "asymptotic"):
            with self.subTest(calibration=calibration):
                table, _ = self.cli("scan", **files, calibration=calibration, budgets=[99, 999], seed=5)
                frame = rapidglm.scan(design, counts[:, :40], predictors, calibration=calibration,
                                      budgets=[99, 999], seed=5)
                self.assert_same(table, frame, text=("family", "status"))
                self.assertEqual((frame["status"] == "ok").sum(), 78)  # gene 3 is all zero

    def test_sceptre_matches_cli(self):
        rng = np.random.default_rng(7)
        cells, genes = 600, 12
        guides = pd.DataFrame({
            "id": ["a1", "a2", "b1", "b2", "c1", "nt1", "nt2"],
            "target": ["A", "A", "B", "B", "C", "non-targeting", "non-targeting"],
            "non_targeting": [0, 0, 0, 0, 0, 1, 1],
        })
        assignments = (rng.random((cells, len(guides))) < 0.08).astype(float)
        depth = rng.gamma(20, 0.05, cells)
        x = np.column_stack([np.ones(cells), np.log(depth)])
        mu = 3 * depth[:, None] * rng.gamma(2, 1, genes)[None, :]
        mu[:, 0] *= np.where(assignments[:, :2].any(axis=1), 0.3, 1)  # A knocks down gene 0
        responses = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
        pairs = pd.DataFrame({
            "response": [0, 1, 2, 0, 3, 4, 5],
            "target": ["A", "A", "B", "B", "C", "non-targeting", "A"],
            "kind": ["positive", "discovery", "discovery", "discovery", "discovery", "negative",
                     "discovery"],
        })
        guides_file = self.root / "guides.tsv"
        guides.to_csv(guides_file, sep="\t", index=False)
        pairs_file = self.root / "pairs.tsv"
        pairs.to_csv(pairs_file, sep="\t", index=False)
        files = dict(design=write_design(self.root / "design.tsv", x),
                     responses=write_market(self.root / "responses.mtx", responses),
                     assignments=write_market(self.root / "assignments.mtx", assignments),
                     guides=str(guides_file), pairs=str(pairs_file))
        cases = [
            dict(moi="high", budgets=[199, 1999], seed=3, threads=4),
            dict(moi="high", integration="singleton", approximation="no_approximation",
                 budgets=[199], seed=9, effect_fits="all", adjustment="bonferroni"),
            dict(moi="high", null_model="direct", family="poisson", resampling_mechanism="permutations",
                 budgets=[299], crt_draw="direct", screen=0.1),
        ]
        for config in cases:
            with self.subTest(**{k: str(v) for k, v in config.items()}):
                table, record = self.cli("sceptre", **files, **config)
                frame = rapidglm.sceptre(x, responses, assignments, guides, pairs, **config)
                kinds = {"0": "discovery", "1": "negative", "2": "positive"}
                table["kind"] = [kinds[k] for k in table["kind"]]
                self.assert_same(table, frame, text=("target", "guide", "kind", "status"))
                self.assertGreaterEqual((frame["status"] == "ok").sum(), 5)
                self.assertEqual(frame.attrs["discovery_rejections"], record["discovery_rejections"])

    def test_errors_are_python_exceptions(self):
        x, counts = self.bulk_data()
        with self.assertRaisesRegex(ValueError, "likelihood ratio test"):
            rapidglm.bulk_de(x, counts, coefficients=[1, 2], test="wald")
        with self.assertRaisesRegex(ValueError, "trend must be one of"):
            rapidglm.bulk_de(x, counts, trend="spline")
        with self.assertRaises(ValueError):
            rapidglm.bulk_de(x[:5], counts)
        self.assertEqual(rapidglm.library_version(), rapidglm.__version__)


if __name__ == "__main__":
    unittest.main()
