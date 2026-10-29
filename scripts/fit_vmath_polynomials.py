#!/usr/bin/env python3
"""Polynomial coefficients of log1p, digamma and trigamma in include/rapidglm/vmath.hpp.

Chebyshev interpolation at 60 digits (mpmath), converted to the monomial basis
of the local variable named and rounded to double; the degree is the smallest
whose rounded polynomial meets the target relative error on a dense grid:
  log_mantissa   (log(1 + f) - f + f^2/2)/f^3 in f, f in [sqrt(2)/2 - 1, sqrt(2) - 1]
  digamma_base   psi(z) z (z + 1)(z + 2)/(z - x0) in z - 0.5, z in [0.5, 2.5]
  digamma_large  x^2 (psi(x) - log x + 1/(2x)) in w = 1/x^2, x >= 5
  trigamma_large x^3 (psi'(x) - 1/x - 1/(2x^2)) in w = 1/x^2, x >= 6
Also prints the two parts of the digamma root.

    python3 scripts/fit_vmath_polynomials.py
"""
import mpmath as mp

mp.mp.dps = 60


def cheb_fit(f, a, b, n):
    pts = [mp.cos(mp.pi * (k + mp.mpf(1) / 2) / n) for k in range(n)]
    fx = [f((a + b) / 2 + (b - a) / 2 * t) for t in pts]
    return [(2 if j else 1) * mp.fsum(fx[k] * mp.cos(mp.pi * j * (k + mp.mpf(1) / 2) / n)
                                       for k in range(n)) / n for j in range(n)]


def to_monomial(c, a, b, center):
    """Coefficients of (x - center)^k for sum c_j T_j((2x - a - b)/(b - a))."""
    n = len(c)
    T = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
    for j in range(2, n):
        t = [mp.mpf(0)] * (j + 1)
        for k, v in enumerate(T[j - 1]):
            t[k + 1] += 2 * v
        for k, v in enumerate(T[j - 2]):
            t[k] -= v
        T.append(t)
    in_u = [mp.fsum(c[j] * T[j][k] for j in range(k, n) if k < len(T[j])) for k in range(n)]
    alpha, beta = 2 / (b - a), (2 * center - a - b) / (b - a)
    return [mp.fsum(in_u[k] * mp.binomial(k, i) * alpha**i * beta**(k - i) for k in range(i, n))
            for i in range(n)]


def fit(f, a, b, center, target, grid=4000):
    a, b, center = mp.mpf(a), mp.mpf(b), mp.mpf(center)
    xs = [a + (b - a) * k / grid for k in range(grid + 1)]
    fx = [f(x) for x in xs]
    for degree in range(2, 40):
        rounded = [float(v) for v in to_monomial(cheb_fit(f, a, b, degree + 1), a, b, center)]
        def value(t):
            acc = mp.mpf(rounded[-1])
            for c in reversed(rounded[:-1]):
                acc = acc * t + mp.mpf(c)
            return acc
        error = max(abs(value(x - center) / v - 1) for x, v in zip(xs, fx))
        if error < target:
            return rounded, degree, error
    raise RuntimeError("no degree meets the target")


root = mp.findroot(mp.digamma, mp.mpf("1.4616321449683623"))


def digamma_base(z):
    if abs(z - root) < mp.mpf(10) ** -40:
        return mp.psi(1, z) * z * (z + 1) * (z + 2)
    return mp.digamma(z) * z * (z + 1) * (z + 2) / (z - root)


def log_mantissa(f):
    if f == 0:
        return mp.mpf(1) / 3
    return (mp.log1p(f) - f + f * f / 2) / f**3


def digamma_large(w):
    if w == 0:
        return -mp.mpf(1) / 12
    x = 1 / mp.sqrt(w)
    return (mp.digamma(x) - mp.log(x) + 1 / (2 * x)) * x * x


def trigamma_large(w):
    if w == 0:
        return mp.mpf(1) / 6
    x = 1 / mp.sqrt(w)
    return (mp.psi(1, x) - 1 / x - 1 / (2 * x * x)) * x**3


hi = float(root)
print(f"digamma root {mp.nstr(root, 40)}: hi {hi!r}, lo {float(root - mp.mpf(hi))!r}")
for name, f, a, b, center, target in (
        ("log_mantissa", log_mantissa, mp.sqrt(2) / 2 - 1, mp.sqrt(2) - 1, 0, 2**-52),
        ("digamma_base", digamma_base, 0.5, 2.5, 0.5, 2**-54),
        ("digamma_large", digamma_large, 0, mp.mpf(1) / 25, 0, 2**-52),
        ("trigamma_large", trigamma_large, 0, mp.mpf(1) / 36, 0, 2**-52)):
    coefficients, degree, error = fit(f, a, b, center, target)
    print(f"{name}: degree {degree}, max relative error {float(error):.3g}")
    print("    " + ", ".join(repr(c) for c in coefficients))
