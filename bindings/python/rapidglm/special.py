"""Special functions of float64 arrays through the C interface
(``rapidglm_<f>_array``, include/rapidglm/vmath.hpp): RapidGLM's ``log``,
``log1p``, ``exp``, ``digamma`` and ``trigamma``, the same bits on every
processor (AVX2, AArch64 Advanced SIMD or portable; measured errors up to 0.67,
0.68 and 0.86 ulp for log, log1p and exp, 3.7 and 2.7 ulp for digamma and
trigamma), and the C library's ``lgamma`` (lgamma_r).

Each function takes array-likes of any shape and returns a new float64 array of
that shape (a NumPy float64 scalar for a scalar argument), or writes into
``out``, a C-contiguous float64 array of the same shape (it may be ``x``). A
NaN argument gives that NaN back (lgamma: quieted, payload kept). Unlike
``scipy.special.digamma`` and ``polygamma(1, x)``, digamma and trigamma are not
defined for negative arguments: they give NaN there. lgamma is log|Gamma(x)|
for every x, as ``scipy.special.gammaln``.
"""

import ctypes

import numpy as np

from .capi import _library, _division_parameter

__all__ = ["log", "log1p", "exp", "lgamma", "digamma", "trigamma", "path"]

_functions = {}
_from_buffer, _address = ctypes.c_char.from_buffer, ctypes.addressof


def _pointer(a):
    """The address of a's data (a quarter of the cost of a.ctypes.data)."""
    try:
        return _address(_from_buffer(a))
    except TypeError:  # read-only
        return a.ctypes.data


def _apply(name, x, out):
    f = _functions.get(name)
    if f is None:
        f = _functions[name] = getattr(_library(), f"rapidglm_{name}_array")
    values = np.asarray(x, dtype=np.float64)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    if out is None:
        result = np.empty(values.shape)
    elif (type(out) is np.ndarray and out.dtype == np.float64 and out.flags.c_contiguous
          and out.shape == values.shape):
        result = out
    else:
        raise ValueError("out must be a C-contiguous float64 array of the argument's shape")
    if values.size and f(_pointer(values), _pointer(result), values.size) != 0:
        raise ValueError(_library().rapidglm_last_error().decode())
    return result[()] if out is None and result.ndim == 0 else result


@_division_parameter
def log(x, out=None, *, division=None):
    """Natural logarithm (RapidGLM's)."""
    return _apply("log", x, out)


@_division_parameter
def log1p(x, out=None, *, division=None):
    """log(1 + x) (RapidGLM's)."""
    return _apply("log1p", x, out)


@_division_parameter
def exp(x, out=None, *, division=None):
    """Exponential (RapidGLM's)."""
    return _apply("exp", x, out)


@_division_parameter
def lgamma(x, out=None, *, division=None):
    """log|Gamma(x)| (the C library's lgamma_r)."""
    return _apply("lgamma", x, out)


@_division_parameter
def digamma(x, out=None, *, division=None):
    """psi(x) = Gamma'(x)/Gamma(x) for x >= 0 (NaN for x < 0; RapidGLM's)."""
    return _apply("digamma", x, out)


@_division_parameter
def trigamma(x, out=None, *, division=None):
    """psi'(x) for x >= 0 (NaN for x < 0; RapidGLM's)."""
    return _apply("trigamma", x, out)


def path():
    """The path RapidGLM's functions take on this processor: avx2, neon or portable."""
    return _library().rapidglm_special_path().decode()
