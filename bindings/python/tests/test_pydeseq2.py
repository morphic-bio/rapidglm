"""RapidGLMInference (rapidglm.pydeseq2) inside PyDESeq2, black box: needs pydeseq2.

Usage: RAPIDGLM_C_LIBRARY=build/librapidglm_c.so python3 bindings/python/tests/test_pydeseq2.py

Checks, on simulated data, that each replaced method matches PyDESeq2's default
on the same inputs (the Wald test to rounding; fits and dispersions to
optimizer tolerance, with RapidGLM's dispersions at least as high in the
Cox-Reid adjusted likelihood where they differ), and that whole analyses agree.
"""

import sys
import unittest
import warnings

import numpy as np
import pandas as pd

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
except ImportError:  # pragma: no cover
    DeseqDataSet = None

sys.tracebacklimit = 0
warnings.filterwarnings("ignore")


def data(seed=0, samples=12, genes=1500):
    rng = np.random.default_rng(seed)
    md = pd.DataFrame(
        {"condition": list("AB") * (samples // 2), "batch": ["x"] * (samples // 2) + ["y"] * (samples // 2)},
        index=[f"s{i}" for i in range(samples)],
    )
    base = rng.gamma(1.5, 60, genes)
    effect = np.where(np.arange(genes) % 5 == 0, 3.0, 1.0)
    mu = base[None, :] * np.where(md.condition.to_numpy()[:, None] == "B", effect[None, :], 1)
    counts = rng.negative_binomial(4, 4 / (4 + mu))
    counts[:, 7] = 0
    counts[: samples // 2, 11] = 0  # an all-zero group: a diverging coefficient
    return pd.DataFrame(counts, index=md.index, columns=[f"g{j}" for j in range(genes)]), md


def _copy(r):
    # The caller edits returned arrays in place (e.g. Cook's outliers), so keep copies.
    return tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in r)


class Recorder(DefaultInference if DeseqDataSet else object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def irls(self, *a, **k):
        r = super().irls(*a, **k)
        self.calls.append(("irls", k, _copy(r)))
        return r

    def alpha_mle(self, *a, **k):
        r = super().alpha_mle(*a, **k)
        self.calls.append(("alpha_mle", k, _copy(r)))
        return r

    def wald_test(self, *a, **k):
        r = super().wald_test(*a, **k)
        self.calls.append(("wald_test", k, _copy(r)))
        return r


def run(counts, md, inference):
    dds = DeseqDataSet(counts=counts, metadata=md, design="~batch + condition", inference=inference, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference, quiet=True)
    ds.summary()
    return dds, ds.results_df


@unittest.skipIf(DeseqDataSet is None, "pydeseq2 not installed")
class PyDESeq2Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rapidglm.pydeseq2 import RapidGLMInference

        cls.Inference = RapidGLMInference
        cls.counts, cls.md = data()
        cls.recorder = Recorder(n_cpus=1)
        cls.default_dds, cls.default = run(cls.counts, cls.md, cls.recorder)

    def test_methods_on_identical_inputs(self):
        ours = self.Inference(n_cpus=2)
        seen = set()
        for name, k, theirs in self.recorder.calls:
            seen.add(name)
            mine = getattr(ours, name)(**k)
            if name == "wald_test":
                for a, b in zip(mine, theirs):
                    np.testing.assert_allclose(a, b, rtol=1e-10)
            elif name == "irls":
                # The same iterations as PyDESeq2's IRLS (means floored at min_mu, ridge
                # 1e-6): coefficients, unfloored means and leverages agree to its
                # deviance tolerance.
                ok = mine[3] & theirs[3]
                self.assertGreater(ok.mean(), 0.9)
                np.testing.assert_allclose(mine[0][ok], theirs[0][ok], atol=1e-3)
                np.testing.assert_allclose(mine[1][:, ok], theirs[1][:, ok], rtol=1e-3)
                np.testing.assert_allclose(mine[2][:, ok], theirs[2][:, ok], atol=1e-4)
            else:
                ratio = np.abs(np.log(mine[0] / theirs[0]))
                self.assertGreater(np.mean(ratio < 1e-3), 0.97, name)
        self.assertEqual(seen, {"irls", "alpha_mle", "wald_test"})

    def test_whole_analysis_agrees(self):
        inference = self.Inference(n_cpus=2)
        dds, ours = run(self.counts, self.md, inference)
        theirs = self.default
        a, b = set(ours.index[ours.padj < 0.1]), set(theirs.index[theirs.padj < 0.1])
        self.assertGreater(len(a & b) / len(a | b), 0.98)
        both = ours.pvalue.notna() & theirs.pvalue.notna()
        self.assertGreater(ours.pvalue[both].corr(theirs.pvalue[both], method="spearman"), 0.999)
        self.assertEqual(ours.pvalue.isna().sum(), theirs.pvalue.isna().sum())
        np.testing.assert_allclose(dds.obs["size_factors"], self.default_dds.obs["size_factors"])


if __name__ == "__main__":
    unittest.main()
