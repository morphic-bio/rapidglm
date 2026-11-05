"""rapidglm.special must reproduce the C++ results of RapidGLM's functions bit
for bit (tests/fixtures/vmath_known_values.tsv, written by
``rapidglm_special_functions values``; lgamma, the C library's, is not in it)
and agree with NumPy and SciPy within the documented bounds plus an allowance
for their own error.

Usage: RAPIDGLM_C_LIBRARY=build/librapidglm_c.so \
       python3 bindings/python/tests/test_special.py
"""

import pathlib
import unittest

import numpy as np

import rapidglm
from rapidglm import special

ROOT = pathlib.Path(__file__).resolve().parents[3]
FUNCTIONS = ("log", "log1p", "exp", "lgamma", "digamma", "trigamma")
OURS = ("log", "log1p", "exp", "digamma", "trigamma")
BOUND = dict(log=1, log1p=1, exp=1, lgamma=4, digamma=3.7, trigamma=2.8)


def from_hex(text):
    return np.array([int(t, 16) for t in text], dtype=np.uint64).view(np.float64)


def ulp(v):
    return np.ldexp(1.0, np.maximum(np.frexp(np.abs(v))[1] - 1, -1022) - 52)


class Special(unittest.TestCase):
    def test_bits_equal_the_cpp_results(self):
        rows = [line.split("\t") for line in
                (ROOT / "tests/fixtures/vmath_known_values.tsv").read_text().splitlines()[1:]]
        self.assertEqual(sorted({r[0] for r in rows}), sorted(OURS))
        for name in OURS:
            x = from_hex([r[1] for r in rows if r[0] == name])
            expected = from_hex([r[2].strip() for r in rows if r[0] == name])
            got = getattr(special, name)(x, division="slow")
            self.assertTrue(np.array_equal(got.view(np.uint64), expected.view(np.uint64)), name)

    def test_division_scopes_restore_and_work_in_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        from rapidglm.capi import _library
        lib = _library()
        original = lib.rapidglm_division_mode()
        x = np.linspace(0.1, 5, 1003)
        expected = {mode: special.digamma(x, division=mode) for mode in ("slow", "fast")}
        def run(mode):
            out = x.copy()
            special.digamma(out, out=out, division=mode)
            return np.array_equal(out.view(np.uint64), expected[mode].view(np.uint64))
        with ThreadPoolExecutor(4) as executor:
            self.assertTrue(all(executor.map(run, ["slow", "fast"] * 20)))
        self.assertEqual(lib.rapidglm_division_mode(), original)
        with self.assertRaises(ValueError):
            special.log(x, division="unknown")
        with self.assertRaises(ValueError):
            special.log(x, out=np.empty(3), division="fast")
        self.assertEqual(lib.rapidglm_division_mode(), original)
        for mode in ("slow", "fast"):
            result = rapidglm.fit_batch(np.ones((8, 1)), np.arange(1., 9.)[:, None],
                                       family="poisson", division=mode, threads=2)
            self.assertEqual(result.attrs["division"], mode)

    def test_agrees_with_numpy_and_scipy(self):
        import scipy.special as sc
        reference = dict(log=np.log, log1p=np.log1p, exp=np.exp, lgamma=sc.gammaln,
                         digamma=sc.digamma, trigamma=lambda x: sc.polygamma(1, x))
        # Allowance for the comparator's own error (rapidglm's own errors are
        # checked against mpmath in tests/test_vmath.cpp).
        allowance = dict(log=2, log1p=2, exp=2, lgamma=2, digamma=2, trigamma=8)
        rng = np.random.default_rng(1)
        x = np.concatenate([10.0 ** rng.uniform(-300, -5, 2000), rng.uniform(1e-5, 3, 2000),
                            rng.uniform(3, 100, 2000), 10.0 ** rng.uniform(2, 15, 2000),
                            rng.geometric(1 / 21, 2000) - 1 + 1 / 10.0 ** rng.uniform(-8, 3, 2000)])
        for name in FUNCTIONS:
            arg = {"exp": rng.uniform(-700, 700, 10000), "log1p": np.concatenate([-rng.uniform(0, 1, 1000), x])}.get(name, x)
            ours, ref = getattr(special, name)(arg), reference[name](arg)
            ok = np.isfinite(ref) & (ref != 0)
            # lgamma and digamma near their zeros: in units of the last place of max(|f|, 1).
            scale = np.maximum(np.abs(ref[ok]), 1) if name in ("lgamma", "digamma") else ref[ok]
            difference = np.max(np.abs(ours[ok] - ref[ok]) / ulp(scale))
            print(f"{name:8s} max |rapidglm - numpy/scipy| = {difference:.3f} ulp")
            self.assertLessEqual(difference, BOUND[name] + allowance[name], name)

    def test_shapes_out_nan_and_scalars(self):
        x = np.array([[0.5, np.nan], [4.0, 1e3]])
        for name in FUNCTIONS:
            f = getattr(special, name)
            y = f(x)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(np.isnan(y[0, 1]))
            out = np.empty_like(x)
            self.assertIs(f(x, out=out), out)
            self.assertTrue(np.array_equal(out, y, equal_nan=True))
            self.assertIsInstance(f(2.5), np.float64)
            self.assertEqual(f(2.5), f(np.array([2.5]))[0])
        z = x.copy()
        special.digamma(z, out=z)  # in place
        self.assertTrue(np.array_equal(z, special.digamma(x), equal_nan=True))
        with self.assertRaises(ValueError):
            special.lgamma(x, out=np.empty(3))
        self.assertTrue(np.isnan(special.digamma(-1.0)))
        self.assertIn(special.path(), ("avx2", "neon", "portable"))
        self.assertTrue(hasattr(rapidglm, "special"))


if __name__ == "__main__":
    unittest.main()
