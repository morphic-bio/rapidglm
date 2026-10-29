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
    def test_floored_optimizer_fallback_metadata(self):
        from rapidglm import capi
        x = np.ones((20, 2)); x[:10, 1] = 0; x[10:, 1] = .01
        y = np.r_[np.zeros(10), 10 + np.arange(10, 20) % 4][:, None]
        *_, converged, optimized = capi.nb_floored_fits(
            x, y, np.ones(20), np.array([.2]), max_iterations=1,
            min_beta=-1, max_beta=1, return_optimizer=True)
        self.assertTrue(converged[0])
        self.assertTrue(optimized[0])

    def test_old_api_checked_before_new_symbols(self):
        from unittest.mock import patch
        from rapidglm import capi
        class Version:
            def __call__(self): return 1
        class Old:
            rapidglm_c_api_version = Version()
            def __getattr__(self, name):
                raise AssertionError("looked up new symbol before version check: " + name)
        with patch.object(capi, "_lib", None), patch.object(capi, "_load", return_value=Old()):
            with self.assertRaisesRegex(OSError, "need 3"): capi._library()

    def test_bad_hypothesis_dimensions(self):
        x = np.ones((8, 2)); y = np.ones((8, 20))
        for method in (rapidglm.bulk_de, rapidglm.edger_ql):
            for options in ({"coefficient": -1}, {"coefficient": 2},
                            {"coefficients": [-1]}, {"contrast": [1]}):
                with self.assertRaises(ValueError): method(x, y, **options)
        with self.assertRaises(ValueError): rapidglm.edger_ql(x, y, norm_factors=[1])

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
            (dict(coefficient=2, legacy_cooks=False), dict(coefficient=2, legacy_cooks=False)),
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

    def test_edger_ql_matches_cli(self):
        x, counts = self.bulk_data()
        files = dict(design=write_design(self.root / "design.tsv", x),
                     responses=write_market(self.root / "counts.mtx", counts))
        cases = [
            (dict(coefficient=2, threads=3), dict(coefficient=2, threads=3)),
            (dict(contrast=[0, 1, -1]), dict(contrast=[0, 1, -1])),
            (dict(coefficients=[1, 2]), dict(coefficients=[1, 2])),
            (dict(filter=False, normalization="none", dispersion=0.1),
             dict(filter=False, normalization="none", dispersion=0.1)),
        ]
        for config, arguments in cases:
            with self.subTest(**{k: str(v) for k, v in arguments.items()}):
                table, record = self.cli("edger-ql", **files, **config)
                frame = rapidglm.edger_ql(x, sp.csr_matrix(counts), **arguments)
                self.assert_same(table, frame, text=("status",))
                self.assertEqual(frame.attrs["df_prior"], record["df_prior"])
                self.assertEqual(frame.attrs["dispersion"], record["dispersion"])
                self.assertEqual(frame.attrs["kept"], record["kept"])

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
                table, record = self.cli("scan", **files, calibration=calibration, budgets=[99, 999],
                                         seed=5)
                frame = rapidglm.scan(design, counts[:, :40], predictors, calibration=calibration,
                                      budgets=[99, 999], seed=5)
                self.assert_same(table, frame, text=("family", "status"))
                self.assertEqual(record["crt_draw"], "hazard")
                self.assertEqual(frame.attrs["crt_draw_used"], record["crt_draw_used"])
                self.assertEqual((frame["status"] == "ok").sum(), 78)  # gene 3 is all zero

    def screen(self):
        """A high-MOI screen and its CLI input files."""
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
        return x, responses, assignments, guides, pairs, files

    def cli_kinds(self, table):
        kinds = {"0": "discovery", "1": "negative", "2": "positive"}
        table["kind"] = [kinds[k] for k in table["kind"]]
        return table

    def assert_same_counts(self, record, frame):
        """The run record's counts by kind equal the frame's run-level values."""
        for name in ("pairs_listed", "pairs_passing_qc", "family_size"):
            for kind, count in record[name].items():
                self.assertEqual(frame.attrs[f"{name}_{kind}"], count, (name, kind))
        self.assertEqual(frame.attrs["null_fits"], record["null_fits"])
        for name in ("discovery_rejections", "calibration_rejections", "power_rejections"):
            self.assertEqual(frame.attrs[name], record[name], name)

    def test_sceptre_matches_cli(self):
        x, responses, assignments, guides, pairs, files = self.screen()
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
                self.assert_same(self.cli_kinds(table), frame, text=("target", "guide", "kind", "status"))
                self.assertGreaterEqual((frame["status"] == "ok").sum(), 5)
                self.assertEqual(frame.attrs["discovery_rejections"], record["discovery_rejections"])
                # hazard by default, thinned without AES instructions, as the CLI says.
                self.assertEqual(record["crt_draw"], config.get("crt_draw", "hazard"))
                self.assertIn(record["crt_draw_used"], ("hazard", "thinned", "direct"))
                self.assertEqual(frame.attrs["crt_draw_used"], record["crt_draw_used"])

    def test_sceptre_null_start_matches_cli(self):
        # Low MOI: NT-control pair nulls, refined from the response's reference or cold.
        x, responses, assignments, guides, pairs, files = self.screen()
        common = dict(moi="low", budgets=[199, 999], seed=5)
        frames = {}
        for start in ("refine", "cold"):
            with self.subTest(null_start=start):
                table, record = self.cli("sceptre", **files, **common, null_start=start)
                frame = rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, null_start=start)
                self.assert_same(self.cli_kinds(table), frame, text=("target", "guide", "kind", "status"))
                self.assertEqual(record["null_start"], start)
                self.assertEqual(record["null_start_used"], start)
                self.assertEqual(frame.attrs["null_start_used"], start)
                for name in ("null_refined", "null_cold_refits", "null_fits"):
                    self.assertEqual(frame.attrs[name], record[name], name)
                frames[start] = frame
        refine, cold = frames["refine"], frames["cold"]
        self.assertGreater(refine.attrs["null_refined"], 0)
        self.assertEqual(cold.attrs["null_refined"], 0)
        self.assertTrue((refine["status"] == cold["status"]).all())
        ok = (cold["status"] == "ok").to_numpy()
        self.assertGreaterEqual(ok.sum(), 3)
        np.testing.assert_allclose(refine["statistic"][ok], cold["statistic"][ok], rtol=1e-6, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "null_start must be one of refine/cold"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, null_start="warm")

    def test_sceptre_stream_keys_and_families_match_cli(self):
        x, responses, assignments, guides, pairs, files = self.screen()
        genes = responses.shape[1]
        # Every target against every gene; with min_nonzero 40 the pairs of C fail pair QC.
        targets = ["A", "B", "C", "non-targeting"]
        pairs = pd.DataFrame(
            [(g, t, "negative" if t == "non-targeting" else "discovery") for g in range(genes) for t in targets]
            + [(0, "A", "positive")], columns=["response", "target", "kind"])
        pairs.to_csv(files["pairs"], sep="\t", index=False)
        ids = [f"gene{j}" for j in range(genes)]
        ids_file = self.root / "ids.txt"
        ids_file.write_text("".join(f"{i}\n" for i in ids))
        common = dict(moi="high", budgets=[199, 999], seed=3, min_nonzero=40, alpha=0.3)
        default = rapidglm.sceptre(x, responses, assignments, guides, pairs, **common)
        failed = default["status"] == "failed_pair_qc"
        self.assertEqual(failed.sum(), 13)
        self.assertTrue(default["adjusted_p"][failed].isna().all())
        self.assertTrue(default["adjusted_p"][~failed].notna().all())
        self.assertEqual((default.attrs["pairs_listed_discovery"], default.attrs["family_size_discovery"]),
                         (36, 24))
        cases = [  # CLI configuration, Python arguments
            (dict(), dict()),
            (dict(stream_keys="position"), dict(stream_keys="position")),
            (dict(response_ids=str(ids_file)), dict(response_ids=ids)),
            (dict(adjustment_family="listed", threads=3), dict(adjustment_family="listed", threads=3)),
            (dict(family_size_discovery=50, family_size_positive=2),
             dict(family_size={"discovery": 50, "positive": 2})),
            (dict(stream_keys="position", adjustment_family="listed", family_size_negative=30,
                  integration="singleton", adjustment="bonferroni"),
             dict(stream_keys="position", adjustment_family="listed", family_size=(0, 30, 0),
                  integration="singleton", adjustment="bonferroni")),
        ]
        for config, arguments in cases:
            with self.subTest(**{k: str(v) for k, v in arguments.items()}):
                table, record = self.cli("sceptre", **files, **common, **config)
                frame = rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, **arguments)
                self.assert_same(self.cli_kinds(table), frame, text=("target", "guide", "kind", "status"))
                self.assert_same_counts(record, frame)
                self.assertEqual(record["stream_keys"], arguments.get("stream_keys", "identity"))
                self.assertEqual(record["adjustment_family"], arguments.get("adjustment_family", "tested"))
                if "integration" not in arguments:
                    # The keys change the draws; the family, the adjusted p-values alone.
                    keys = "stream_keys" in arguments or "response_ids" in arguments
                    self.assertEqual(frame["p_value"].equals(default["p_value"]), not keys)
                    self.assertEqual(frame["adjusted_p"].equals(default["adjusted_p"]), not arguments)
        # Batch scoring (the default) gives the records of scoring pair by pair.
        unbatched = rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, batch_scoring=False)
        pd.testing.assert_frame_equal(unbatched, default, check_exact=True)
        table, record = self.cli("sceptre", **files, **common, batch_scoring=False)
        self.assertFalse(record["batch_scoring"])
        self.assert_same(self.cli_kinds(table), default, text=("target", "guide", "kind", "status"))
        with self.assertRaisesRegex(ValueError, "stream_keys must be one of identity/position"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, stream_keys="name")
        with self.assertRaisesRegex(ValueError, "adjustment_family must be one of tested/listed"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, adjustment_family="all")
        with self.assertRaisesRegex(ValueError, f"response_ids needs {genes} IDs, got 2"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, response_ids=["a", "b"])
        with self.assertRaisesRegex(ValueError, "response IDs must be distinct"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, response_ids=["a"] * genes)
        with self.assertRaisesRegex(ValueError, "family_size kinds must be"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, family_size={"other": 3})
        with self.assertRaisesRegex(ValueError, "family_size needs three sizes"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, family_size=[50])
        with self.assertRaisesRegex(ValueError, "family_size is smaller"):
            rapidglm.sceptre(x, responses, assignments, guides, pairs, **common, family_size=[3, 0, 0])
        with self.assertRaisesRegex(ValueError, "listed twice"):
            rapidglm.sceptre(x, responses, assignments, guides, pd.concat([pairs, pairs[4:5]]), **common)

    def test_sceptre_grid_matches_cli_and_the_list(self):
        x, responses, assignments, guides, _, files = self.screen()
        genes = responses.shape[1]
        discovery, negative = ["A", "B", "C"], ["non-targeting"]
        positive = pd.DataFrame({"response": [0], "target": ["A"], "kind": ["positive"]})
        grid_pairs = pd.DataFrame(
            [(g, t, "negative" if t in negative else "discovery") for g in range(genes)
             for t in discovery + negative], columns=["response", "target", "kind"])
        positive.to_csv(files["pairs"], sep="\t", index=False)
        targets_file = self.root / "targets.tsv"
        pd.DataFrame({"target": discovery + negative,
                      "kind": ["discovery"] * 3 + ["negative"]}).to_csv(targets_file, sep="\t", index=False)
        some = [7, 0, 5]
        some_file = self.root / "grid_responses.txt"
        some_file.write_text("".join(f"{g}\n" for g in some))
        common = dict(moi="high", budgets=[199, 999], seed=3, min_nonzero=40, alpha=0.3)
        cases = [  # CLI configuration, Python arguments
            (dict(), dict()),
            (dict(adjustment_family="listed", threads=3), dict(adjustment_family="listed", threads=3)),
            (dict(grid_responses=str(some_file), integration="bonferroni"),
             dict(grid_responses=some, integration="bonferroni")),
        ]
        for config, arguments in cases:
            with self.subTest(**{k: str(v) for k, v in arguments.items()}):
                table, record = self.cli("sceptre", **files, targets=str(targets_file), **common, **config)
                frame, tally = rapidglm.sceptre_grid(x, responses, assignments, guides, discovery, negative,
                                                     pairs=positive, **common, **arguments)
                self.assert_same(self.cli_kinds(table), frame, text=("target", "guide", "kind", "status"))
                self.assert_same_counts(record, frame)
                self.assertTrue(record["grid"])
                cli_tally = pd.read_csv(self.root / "result.tally.tsv", sep="\t", dtype=str, keep_default_na=False)
                self.assert_same(cli_tally, tally, text=("target", "kind"))
                # The grid's rows are the rows of the listed pairs that pass pair QC.
                responses_used = arguments.get("grid_responses", range(genes))
                options = {k: v for k, v in arguments.items() if k != "grid_responses"}
                listed = pd.concat([grid_pairs[grid_pairs["response"] == g] for g in responses_used] + [positive],
                                   ignore_index=True)
                reference = rapidglm.sceptre(x, responses, assignments, guides, listed, **common, **options)
                passing = reference[reference["status"] != "failed_pair_qc"].reset_index(drop=True)
                self.assertGreater(len(reference), len(passing))
                pd.testing.assert_frame_equal(frame, passing, check_exact=True)
                for name in ("pairs_listed", "pairs_passing_qc", "family_size"):
                    for kind in ("discovery", "negative", "positive"):
                        self.assertEqual(frame.attrs[f"{name}_{kind}"], reference.attrs[f"{name}_{kind}"])
                # Every pair of the grid is in the tally, by target and kind.
                self.assertEqual(tally["target"].tolist(), discovery + negative)
                self.assertEqual(tally["kind"].tolist(), ["discovery"] * 3 + ["negative"])
                self.assertTrue((tally["requested"] == len(responses_used)).all())
                self.assertTrue(
                    (tally["requested"] == tally["passing"] + tally["few_treatment"] + tally["few_control"]).all())
                self.assertEqual(tally["passing"].sum() + 1, len(frame))
                self.assertEqual(tally["few_treatment"][2], len(responses_used))
        # No listed pairs, responses by name, and invalid requests.
        named = pd.DataFrame(responses, columns=[f"gene{j}" for j in range(genes)])
        frame, tally = rapidglm.sceptre_grid(x, named, assignments, guides, ["A"], grid_responses=["gene5", "gene2"],
                                             **common)
        self.assertEqual(frame["response_name"].tolist(), ["gene5", "gene2"])
        self.assertEqual((len(tally), tally["requested"][0]), (1, 2))
        with self.assertRaisesRegex(ValueError, "identity stream keys"):
            rapidglm.sceptre_grid(x, responses, assignments, guides, discovery, negative, **common,
                                  stream_keys="position")
        with self.assertRaisesRegex(ValueError, "repeats a pair of the grid"):
            rapidglm.sceptre_grid(x, responses, assignments, guides, discovery, negative, pairs=grid_pairs[:1],
                                  **common)
        with self.assertRaisesRegex(ValueError, "grid targets must be distinct known targets"):
            rapidglm.sceptre_grid(x, responses, assignments, guides, ["A", "A"], **common)
        with self.assertRaisesRegex(ValueError, "grid_responses is empty"):
            rapidglm.sceptre_grid(x, responses, assignments, guides, ["A"], grid_responses=[], **common)

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
