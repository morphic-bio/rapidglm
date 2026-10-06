"""The PyDESeq2-style interface (rapidglm.deseq) against rapidglm.bulk_de, patsy
design matrices and, when the local airway data are present, the CLI.

Usage: RAPIDGLM_C_LIBRARY=build/librapidglm_c.so \
       python3 bindings/python/tests/test_deseq.py build/rapidglm
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import rapidglm
from rapidglm.deseq import DeseqDataSet, DeseqStats, design_matrix

BINARY = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
sys.argv = sys.argv[:1]
ROOT = pathlib.Path(__file__).resolve().parents[3]
AIRWAY = ROOT / "validation/work/bulk/airway"


def data(seed=3, genes=200):
    rng = np.random.default_rng(seed)
    metadata = pd.DataFrame(
        {
            "condition": list("ABC") * 4,
            "batch": ["b1", "b1", "b1", "b2", "b2", "b2"] * 2,
            "age": rng.normal(40, 8, 12),
        },
        index=[f"s{i}" for i in range(12)],
    )
    effect = {"A": 1.0, "B": 2.0, "C": 0.7}
    base = rng.gamma(2, 50, genes)
    changed = np.arange(genes) % 5 == 0
    mu = np.array(
        [base * np.where(changed, effect[c], 1) * (1.3 if b == "b2" else 1)
         for c, b in zip(metadata.condition, metadata.batch)]
    )
    counts = pd.DataFrame(
        rng.negative_binomial(6, 6 / (6 + mu)), index=metadata.index,
        columns=[f"gene{j}" for j in range(genes)],
    )
    return counts, metadata


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


class Deseq(unittest.TestCase):
    def test_design_matrices_match_patsy(self):
        try:
            import patsy
        except ImportError:
            self.skipTest("patsy not installed")
        _, metadata = data()
        for formula in ("~condition", "~batch + condition", "~batch*condition",
                        "~age + condition + age:condition", "~0 + condition + batch",
                        "~condition - 1"):
            with self.subTest(formula=formula):
                ours = design_matrix(formula, metadata)
                ref = patsy.dmatrix(formula, metadata, return_type="dataframe")
                self.assertEqual(list(ours.columns), list(ref.columns))
                self.assertTrue(np.array_equal(ours.to_numpy(), ref.to_numpy()))
        with self.assertRaises(NotImplementedError):
            design_matrix("~batch + batch:condition", metadata)
        with self.assertRaisesRegex(ValueError, "not a metadata column"):
            design_matrix("~donor", metadata)

    def test_stats_equal_bulk_de(self):
        counts, metadata = data()
        dds = DeseqDataSet(counts=counts, metadata=metadata, design="~batch + condition", quiet=True)
        dds.deseq2()
        x = design_matrix("~batch + condition", metadata).to_numpy()
        # The last column, a contrast of two non-reference levels, and a reversed one.
        cases = [
            (["condition", "C", "A"], dict(coefficient=3)),
            (["condition", "C", "B"], dict(contrast=[0, 0, -1, 1])),
            (["condition", "A", "B"], dict(contrast=[0, 0, -1, 0])),
        ]
        for contrast, arguments in cases:
            with self.subTest(contrast=contrast):
                ds = DeseqStats(dds, contrast=contrast, quiet=True)
                ds.summary()
                expected = rapidglm.bulk_de(x, counts, alpha=0.05, max_dispersion=10, **arguments)
                for column in ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
                    self.assertTrue(same(ds.results_df[column], expected[column]), column)
                self.assertEqual(list(ds.results_df.index), list(counts.columns))
        self.assertTrue(same(dds.obs["size_factors"], expected.attrs["size_factors"]))
        self.assertTrue(same(dds.var["dispersions"], expected["dispersion"]))
        # The default test reuses the deseq2() run; a fresh dataset gives the same table.
        ds = DeseqStats(dds, contrast=["condition", "C", "A"], quiet=True)
        first = ds.summary()
        fresh = DeseqDataSet(counts=counts, metadata=metadata, design="~batch + condition", quiet=True)
        again = DeseqStats(fresh, contrast=["condition", "C", "A"], quiet=True).summary()
        self.assertTrue(same(first.to_numpy(), again.to_numpy()))
        self.assertEqual(len(dds._fits), 3)
        # Another alpha or no filtering reuses the fit and re-adjusts exactly.
        for alpha, independent in ((0.1, True), (0.01, True), (0.05, False)):
            ds = DeseqStats(dds, contrast=["condition", "C", "A"], alpha=alpha,
                            independent_filter=independent, quiet=True)
            ds.summary()
            expected = rapidglm.bulk_de(x, counts, coefficient=3, alpha=alpha,
                                        independent_filtering=independent, max_dispersion=10)
            self.assertTrue(same(ds.results_df["padj"], expected["padj"]), (alpha, independent))
        self.assertEqual(len(dds._fits), 3)

    def test_shrinkage_filters_and_errors(self):
        counts, metadata = data(seed=5)
        dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition", quiet=True,
                           refit_cooks=False, size_factors_fit_type="poscounts", n_cpus=2)
        dds.deseq2()
        x = design_matrix("~condition", metadata).to_numpy()
        ds = DeseqStats(dds, contrast=["condition", "B", "A"], alpha=0.1, independent_filter=False,
                        quiet=True)
        ds.summary()
        shrunk = ds.lfc_shrink(coeff="condition[T.B]")
        expected = rapidglm.bulk_de(x, counts, coefficient=1, alpha=0.1, independent_filtering=False,
                                    lfc_shrink="normal", min_replicates_for_replace=0,
                                    size_factors="poscounts", max_dispersion=10, threads=2)
        self.assertTrue(same(shrunk["log2FoldChange"], expected["log2FoldChangeShrunk"]))
        self.assertTrue(same(shrunk["lfcSE"], expected["lfcSEShrunk"]))
        self.assertTrue(same(shrunk["pvalue"], expected["pvalue"]))
        self.assertTrue(ds.shrunk_LFCs)
        with self.assertRaises(NotImplementedError):
            DeseqStats(dds, contrast=["condition", "B", "A"], lfc_null=1.0)
        with self.assertRaises(NotImplementedError):
            DeseqDataSet(counts=counts, metadata=metadata, size_factors_fit_type="iterative")
        with self.assertRaisesRegex(ValueError, "not a level"):
            DeseqStats(dds, contrast=["condition", "D", "A"])
        with self.assertRaisesRegex(ValueError, "shrinkage applies to one design column"):
            DeseqStats(dds, contrast=["condition", "C", "B"], quiet=True).lfc_shrink()
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            DeseqDataSet(counts=counts + 0.5, metadata=metadata)

    @unittest.skipUnless((AIRWAY / "counts.mtx").exists() and BINARY, "local airway data and CLI")
    def test_airway_matches_cli(self):
        import scipy.io

        genes = (AIRWAY / "genes.txt").read_text().split()
        counts = scipy.io.mmread(AIRWAY / "counts.mtx").T.toarray()
        coded = pd.read_csv(AIRWAY / "design.tsv", sep="\t")
        cells = ["N052611", "N061011", "N080611", "N61311"]
        metadata = pd.DataFrame({
            "cell": [next((c for c in cells[1:] if row[f"cell{c}"] == 1), cells[0])
                     for _, row in coded.iterrows()],
            "dex": np.where(coded["dextrt"] == 1, "trt", "untrt"),
        }, index=[f"sample{i}" for i in range(len(coded))])
        metadata["dex"] = pd.Categorical(metadata["dex"], categories=["untrt", "trt"])
        dds = DeseqDataSet(counts=pd.DataFrame(counts, index=metadata.index, columns=genes),
                           metadata=metadata, design="~cell + dex", max_disp=None, n_cpus=4,
                           quiet=True)
        x = dds.obsm["design_matrix"].to_numpy()
        self.assertTrue(np.array_equal(x, coded.to_numpy(float)))
        ds = DeseqStats(dds, contrast=["dex", "trt", "untrt"], alpha=0.1, quiet=True)
        ds.summary()
        with tempfile.TemporaryDirectory() as tmp:
            config = pathlib.Path(tmp) / "config.json"
            config.write_text(json.dumps({
                "responses": str(AIRWAY / "counts.mtx"), "design": str(AIRWAY / "x.tsv"),
                "coefficient": 4, "threads": 4, "output": f"{tmp}/airway",
            }))
            r = subprocess.run([BINARY, "bulk-de", config], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            cli = pd.read_csv(f"{tmp}/airway.tsv", sep="\t", float_precision="round_trip")
        for column in ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
            self.assertTrue(same(ds.results_df[column], cli[column]), column)


if __name__ == "__main__":
    unittest.main()
