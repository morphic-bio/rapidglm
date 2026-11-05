#pragma once
// vmath.hpp: log, log1p, exp, lgamma, digamma and trigamma of double, for
// arrays and one value at a time. Part of RapidGLM (MIT licence, see LICENSE;
// Boost-derived fast polygamma sections use the Boost licence reproduced below).
// It depends only on the C++17 standard library (lgamma from the C library)
// and may be copied into other projects on its own: rename the
// namespace and the RAPIDGLM_VMATH_* macros so that copies of different versions
// cannot collide.
//
// Usage:
//   rapidglm::vmath::log1p_array(x, out, n);   // out[i] = log1p(x[i]); out may be x
//   double v = rapidglm::vmath::digamma(2.5);   // one value, the same result
//
// The default slow-division policy uses the following choices. The fast policy
// uses division-based log1p and positive lgamma, and Boost-derived digamma;
// negative lgamma still uses libm. Set RAPIDGLM_DIVISION=slow|fast before first
// use, or use DivisionScope for an explicit, thread-local selection.
// Each slow-policy function is the faster of the C library's and RapidGLM's on the
// efficiency cores (Intel Gracemont) of RapidGLM's benchmark host, whose 256-bit
// division is slow (METHODS.md, Numerical backend):
//   lgamma         the C library's (lgamma_r where the C library has it, since
//                  std::lgamma writes the global signgam). Its bits follow the
//                  platform's libm.
//   log, log1p,    RapidGLM's, the same bits on every processor and path: AVX2
//   exp, digamma,  with FMA on x86-64 (where the processor has both, unless
//   trigamma       RAPIDGLM_VMATH_FORCE_PORTABLE or RAPIDGLM_FORCE_PORTABLE is
//                  defined), Advanced SIMD on AArch64, and the portable path
//                  (scalar, std::fma) otherwise, and the one-value forms. They
//                  use only IEEE operations, fused multiply-adds and integer
//                  operations on the bits, in the same order on every path, and
//                  never let a rounded product feed an addition, so
//                  floating-point contraction cannot change them. This needs
//                  round-to-nearest, no flush-to-zero and no -ffast-math. On an
//                  x86-64 processor without FMA the portable path calls the C
//                  library's fma (correctly rounded, slow). Unused vector lanes
//                  may raise floating-point exception flags.
// Their arrays are the fast forms: one value at a time, the C library's log is
// faster than log here. For RapidGLM's functions *_array_portable and, with AVX2
// compiled in, *_array_avx2 select a path; avx2_active() and path() say which
// *_array takes.
//
// Domains, special values and errors (units in the last place of the correctly
// rounded result, the largest measured against long double and mpmath
// references; METHODS.md, Numerical backend):
//   log(x)       log(0) = -inf, x < 0 gives NaN                     measured 0.67
//   log1p(x)     x >= -1; log1p(-1) = -inf, x < -1 gives NaN        measured 0.68
//   exp(x)       overflow to +inf, underflow to 0                    measured 0.86
//   lgamma(x)    C library: log|Gamma(x)| for all x; +inf at the poles
//   digamma(x)   x >= 0; digamma(0) = -inf, x < 0 gives NaN          measured 3.7
//   trigamma(x)  x >= 0; trigamma(0) = +inf, x < 0 gives NaN         measured 2.7
// RapidGLM's functions return a NaN argument unchanged, keeping its payload (R's
// NA, for example). For digamma and trigamma -0 is treated as 0, and +inf gives
// +inf and 0.
//
// log: x = 2^k m with m in [sqrt(2)/2, sqrt(2)) from the bits, g = m - 1 (exact),
//   log x = k log 2 + g - g^2/2 + g^3 R(g), with log 2 in two parts (k log2_hi
//   exact), k log2_hi + g and the sum with -g^2/2 split exactly (Fast2Sum), the
//   rounding error of g^2 kept, and R of degree 19 (Chebyshev interpolation with
//   mpmath, scripts/fit_vmath_polynomials.py), evaluated by Estrin's scheme;
//   subnormal x are scaled by 2^54 first. No division.
// log1p: u = 1 + x rounded and c = x - (u - 1) its rounding error (exact, with
//   the larger of x and 1 first), and log1p(x) = log u + c/u as for log with
//   g + r = (m - 1) + c 2^-k split exactly (g = x when k = 0) and r/u from
//   r 2^-k (1 - g + g^2): no division. |x| < 2^-60 gives x (correctly rounded).
// exp: exp(x) = 2^k exp(r) with k = round(x / log 2) and r = x - k log 2 (log 2
//   in two parts, |r| <= 0.35), exp(r) from its Taylor polynomial of degree 13
//   (truncation below 1e-17 relative) and 2^k added to the exponent bits; for
//   |x| > 708 the scaling takes two steps (overflow, gradual underflow).
// digamma: psi(z) = (z - x0) P(v) / D(v) on z in [0.5, 2.5), v = z - 0.5, x0 the
//   positive root in two parts, P of degree 16 and D(v) = z (z + 1)(z + 2);
//   psi(x) = psi(x + 1) - 1/x below and psi(x) = psi(x - n) + sum 1/(x - j)
//   above, both free of cancellation; for x >= 5, log x - 1/(2x) + G(1/x^2)/x^2
//   with G of degree 9 fitted in 1/x^2, and RapidGLM's log x.
// trigamma: psi'(x) = sum_{j < m} 1/(x + j)^2 + psi'(x + m) with m = ceil(6 - x)
//   (0 to 6), and psi'(y) = 1/y + 1/(2y^2) + H(1/y^2)/y^3 with H of degree 8
//   fitted in 1/y^2.
// log's polynomial is evaluated by Estrin's scheme, whose dependency chain is
// shorter than Horner's, the others by Horner's (P has v up to 2, and the powers
// of 1/x^2 in Estrin's scheme would underflow).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string_view>
#if !defined(RAPIDGLM_VMATH_FORCE_PORTABLE) && !defined(RAPIDGLM_FORCE_PORTABLE) &&                \
    (defined(__GNUC__) || defined(__clang__))
#if defined(__x86_64__)
#define RAPIDGLM_VMATH_AVX2 1
#include <immintrin.h>
#elif defined(__aarch64__)
#define RAPIDGLM_VMATH_NEON 1
#include <arm_neon.h>
#endif
#endif
#ifdef __FAST_MATH__
#error "rapidglm/vmath.hpp needs IEEE arithmetic: do not compile it with -ffast-math"
#endif
// The scalar code is inlined into its FMA-compiled instances (fused multiply-adds
// in registers there).
// The one-value forms call instances that are never inlined, which keeps them
// small at their call sites.
#if defined(__GNUC__) || defined(__clang__)
#define RAPIDGLM_VMATH_KERNEL inline __attribute__((always_inline))
#define RAPIDGLM_VMATH_INSTANCE inline __attribute__((noinline))
#else
#define RAPIDGLM_VMATH_KERNEL inline
#define RAPIDGLM_VMATH_INSTANCE inline
#endif

namespace rapidglm::vmath {

// The environment is read once, at the first use. An explicit option overrides
// it for the calling thread; scopes restore the previous option on exceptions.
// Workers inherit their caller's resolved mode (sparse.hpp).
enum class DivisionMode { inherit = -1, slow = 0, fast = 1 };
inline DivisionMode parse_division(std::string_view value) {
    if (value == "slow")
        return DivisionMode::slow;
    if (value == "fast")
        return DivisionMode::fast;
    throw std::invalid_argument("division must be slow or fast");
}
inline DivisionMode default_division() {
    static const DivisionMode mode = [] {
        if (const char *value = std::getenv("RAPIDGLM_DIVISION"))
            return parse_division(value);
#ifdef RAPIDGLM_DIVISION_DEFAULT_FAST
        return DivisionMode::fast;
#else
        return DivisionMode::slow;
#endif
    }();
    return mode;
}
namespace detail {
inline thread_local DivisionMode division_override = DivisionMode::inherit;
}
inline DivisionMode division_mode() {
    return detail::division_override == DivisionMode::inherit ? default_division()
                                                              : detail::division_override;
}
inline const char *division_name() {
    return division_mode() == DivisionMode::fast ? "fast" : "slow";
}
inline DivisionMode set_division(DivisionMode mode) {
    if (mode != DivisionMode::inherit && mode != DivisionMode::slow && mode != DivisionMode::fast)
        throw std::invalid_argument("unknown division mode");
    const auto previous = detail::division_override;
    detail::division_override = mode;
    return previous;
}
class DivisionScope {
    DivisionMode previous_;

  public:
    explicit DivisionScope(DivisionMode mode = DivisionMode::inherit)
        : previous_(set_division(mode == DivisionMode::inherit ? division_mode() : mode)) {}
    ~DivisionScope() {
        detail::division_override = previous_;
    }
    DivisionScope(const DivisionScope &) = delete;
    DivisionScope &operator=(const DivisionScope &) = delete;
};
namespace detail {
// From scripts/fit_vmath_polynomials.py (mpmath): Chebyshev interpolants in the
// local variable named, converted to monomials and rounded to double.
// log1p: (log(1 + f) - f + f^2/2)/f^3 in f on [sqrt(2)/2 - 1, sqrt(2) - 1], degree 19.
inline constexpr double log_mantissa[20] = {
    0.33333333333333337, -0.2499999999999993,  0.19999999999993223, -0.16666666666697305,
    0.1428571428759174,  -0.12499999997109267, 0.11111110909841072, -0.0999999999631727,
    0.09090919875998202, -0.08333345056413204, 0.0769198812013375,  -0.0714223204915847,
    0.06671985555271291, -0.06265031869518063, 0.05837018351464999, -0.05368280388050548,
    0.0537187789066478,  -0.06129561980727906, 0.0566642324205218,  -0.025061228095282464};
inline constexpr double digamma_root_hi = 1.4616321449683622,
                        digamma_root_lo = 9.549995429965697e-17;
// digamma: psi(z) z (z + 1)(z + 2)/(z - x0) in z - 0.5 on [0.5, 2.5], degree 16.
inline constexpr double digamma_base[17] = {
    3.8284715398228424,      6.099941743958366,       2.4308890609831875,
    0.13543493659826641,     -0.014067701656003037,   0.002324694735537435,
    -0.00047262417015556785, 0.00010770509259799963,  -2.6297142126807152e-05,
    6.690566132601583e-06,   -1.7276337879698932e-06, 4.3568899042063677e-07,
    -1.0113901555173833e-07, 1.9938842400804787e-08,  -3.012043592322404e-09,
    3.007632743544492e-10,   -1.458759560021784e-11};
// digamma: x^2 (psi(x) - log x + 1/(2x)) in w = 1/x^2, x >= 5, degree 9.
inline constexpr double digamma_large[10] = {
    -0.08333333333333333,  0.008333333333329031, -0.003968253964670696, 0.004166665498748365,
    -0.007575562114231473, 0.021073708175058693, -0.08216979989741699,  0.39738019460474266,
    -1.85960480180837,     5.31505138695299};
// trigamma: x^3 (psi'(x) - 1/x - 1/(2 x^2)) in w = 1/x^2, x >= 6, degree 8.
inline constexpr double trigamma_large[9] = {
    0.16666666666666666,   -0.033333333333261904, 0.023809523740380884,
    -0.033333307366846185, 0.07575264096686857,   -0.2525785652536767,
    1.1316812833788938,    -5.686494459221039,    20.39907980934681};
// exp: the Taylor coefficients 1/j!, j = 0 to 13.
inline constexpr double exp_taylor[14] = {1.0,
                                          1.0,
                                          1.0 / 2,
                                          1.0 / 6,
                                          1.0 / 24,
                                          1.0 / 120,
                                          1.0 / 720,
                                          1.0 / 5040,
                                          1.0 / 40320,
                                          1.0 / 362880,
                                          1.0 / 3628800,
                                          1.0 / 39916800,
                                          1.0 / 479001600,
                                          1.0 / 6227020800};
// log 2 in two parts for log and log1p: log2_hi has 32 significant bits, so
// k log2_hi is exact for |k| < 2^21. exp keeps the double nearest log 2 and the
// remainder.
inline constexpr double log2_hi = 6.93147180369123816490e-01, log2_lo = 1.90821492927058770002e-10;
inline constexpr double exp_log2e = 1.4426950408889634, exp_ln2_hi = 6.93147180559945286227e-01,
                        exp_ln2_lo = 2.31904681384629955842e-17;
// 1.5 x 2^52: the integer k is in the low bits of k + exp_shift.
inline constexpr double exp_shift = 6755399441055744.0;
// t = bits(u) - bits(sqrt(2)/2) has k in its exponent field; (t ^ sign) >> 52 is
// 2048 + k, and 2^52 + 2048 + k from those bits under magic_bits.
inline constexpr std::uint64_t sqrt_half_bits = 0x3fe6a09e667f3bcdULL,
                               exponent_mask = 0xfff0000000000000ULL,
                               one_bits = 0x3ff0000000000000ULL, sign_bit = 0x8000000000000000ULL,
                               magic_bits = 0x4330000000000000ULL;
inline constexpr double magic_k = 0x1p52 + 2048;
inline constexpr double inf = std::numeric_limits<double>::infinity();
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
inline constexpr double digamma_series = 5, trigamma_series = 6;
// The main paths: log for normal positive x, log1p for x in (-1, 2^1000) with
// |x| >= 2^-60, exp for |x| <= 708.
inline constexpr double log_min = 0x1p-1022, log1p_tiny = 0x1p-60, log1p_limit = 0x1p1000,
                        exp_limit = 708;

RAPIDGLM_VMATH_KERNEL std::uint64_t bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
RAPIDGLM_VMATH_KERNEL double from_bits(std::uint64_t u) {
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
template <std::size_t N> RAPIDGLM_VMATH_KERNEL double horner(const double (&c)[N], double t) {
    double p = c[N - 1];
    for (std::size_t j = N - 1; j-- > 0;)
        p = std::fma(p, t, c[j]);
    return p;
}

// c[0] + c[1] t + ... + c[N - 1] t^(N - 1) by Estrin's scheme: pairs in t, then
// pairs of those in t^2, t^4, ... (the same order on every path).
template <std::size_t N> RAPIDGLM_VMATH_KERNEL double estrin(const double (&p)[N], double t) {
    if constexpr (N == 1) {
        return p[0];
    } else {
        double q[(N + 1) / 2];
        for (std::size_t j = 0; j < N / 2; ++j)
            q[j] = std::fma(p[2 * j + 1], t, p[2 * j]);
        if constexpr (N % 2 == 1)
            q[N / 2] = p[N - 1];
        return estrin(q, t * t);
    }
}

// The C library's function.
inline double lgamma(double x) {
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) ||     \
    defined(__OpenBSD__)
    int sign;
    return ::lgamma_r(x, &sign);
#else
    return std::lgamma(x);
#endif
}

// RapidGLM's functions one value at a time: the portable path, and the
// operations that the vector paths perform lane by lane.
// log(u) + c/u for u = 2^k m, positive and normal (Correction: u < 2^1001; without
// it c = 0 and u any such value; `magic` magic_k + j gives log(2^-j u)), as
//   k log2_hi + g - g^2/2 + [low + e + r/m + k log2_lo + (g^2 - g2)(-1/2) + g^3 R(g)]
// with g + r = f + c 2^-k (Fast2Sum: |c 2^-k| <= ulp(m)/2 <= |f| unless f = 0; g = x
// when k = 0), hi + low = k log2_hi + g and s + e = hi - g2/2 split exactly, and
// the bracket added before the last rounding.
template <bool Correction>
RAPIDGLM_VMATH_KERNEL double log_reduced(double u, double c, double magic = magic_k) {
    const std::uint64_t ub = bits(u), t = ub - sqrt_half_bits, kb = t & exponent_mask;
    const double m = from_bits(ub - kb);
    const double k = from_bits(((t ^ sign_bit) >> 52) | magic_bits) - magic;
    double g = m - 1, r = 0;
    if (Correction) {
        const double f = g, cs = c * from_bits(one_bits - kb);
        g = f + cs;
        r = (f - g) + cs;
    }
    const double g2 = g * g, half = -0.5 * g2;        // half is exact
    const double hi = std::fma(k, log2_hi, g);        // k log2_hi is exact
    const double low = g - std::fma(-k, log2_hi, hi); // |k log2_hi| >= |g| unless k = 0
    const double s = hi + half, e = half - (s - hi);  // |hi| >= |half|
    double tail = low + e;
    if (Correction)
        tail = std::fma(r, std::fma(g, g, 1 - g), tail); // r/m, 1/m to relative error g^3
    tail = std::fma(k, log2_lo, tail);
    tail = std::fma(std::fma(g, g, -g2), -0.5, tail);
    tail = std::fma(g2 * g, estrin(log_mantissa, g), tail);
    return s + tail;
}
inline double log_special(double x) {
    if (x != x || x == inf)
        return x;
    if (x == 0)
        return -inf;
    if (x < 0)
        return nan;
    return log_reduced<false>(x * 0x1p54, 0, magic_k + 54); // subnormal
}
RAPIDGLM_VMATH_KERNEL double log(double x) {
    if (!(x >= log_min && x < inf))
        return log_special(x);
    return log_reduced<false>(x, 0);
}
inline double log1p_special(double x) {
    // |x| < 2^-60: log1p(x) = x - x^2/2 + ... rounds to x.
    if (x != x || std::abs(x) < log1p_tiny || x == inf)
        return x;
    if (x >= log1p_limit) // c/u is below 2^-1000 of the result
        return log_reduced<false>(x, 0);
    return x == -1 ? -inf : nan;
}
RAPIDGLM_VMATH_KERNEL double log1p(double x) {
    if (!(x > -1 && x < log1p_limit && std::abs(x) >= log1p_tiny))
        return log1p_special(x);
    const double u = 1 + x;
    return log_reduced<true>(u, std::min(x, 1.0) - (u - std::max(x, 1.0)));
}
// exp(r) for r = x - k log 2, k = round(x / log 2).
RAPIDGLM_VMATH_KERNEL double exp_reduced(double x, double &k) {
    k = std::nearbyint(x * exp_log2e);
    double r = std::fma(-k, exp_ln2_hi, x);
    r = std::fma(-k, exp_ln2_lo, r);
    return horner(exp_taylor, r);
}
// p 2^k through the exponent bits (the result normal).
RAPIDGLM_VMATH_KERNEL double exp_scaled(double p, double k) {
    return from_bits(bits(p) + (bits(k + exp_shift) << 52));
}
inline double exp_special(double x) {
    if (x != x)
        return x;
    if (x > 710)
        return inf;
    if (x < -746)
        return 0;
    // 2^(k -+ 64) into the bits, then the power of two 2^(+-64).
    double k;
    const double p = exp_reduced(x, k);
    return x > 0 ? exp_scaled(p, k - 64) * 0x1p64 : exp_scaled(p, k + 64) * 0x1p-64;
}
RAPIDGLM_VMATH_KERNEL double exp(double x) {
    if (!(std::abs(x) <= exp_limit))
        return exp_special(x);
    double k;
    const double p = exp_reduced(x, k);
    return exp_scaled(p, k);
}
RAPIDGLM_VMATH_KERNEL double digamma(double x) {
    if (!(x > 0) || x == inf)
        return x != x ? x : x == 0 ? -inf : x > 0 ? x : nan;
    if (x >= digamma_series) {
        // log x - 1/(2x) + G(1/x^2)/x^2; G's argument with 1/x at least 2^-500,
        // which leaves G's rounded value as it is and keeps 1/x^2 out of the
        // (slow) subnormal range.
        const double r = 1 / x, rg = std::max(r, 0x1p-500), g = horner(digamma_large, rg * rg);
        return std::fma(r, std::fma(r, g, -0.5), log_reduced<false>(x, 0));
    }
    // psi(z) = (z - x0) P(v) / D(v), v = z - 0.5 and D(v) = z (z + 1)(z + 2), for z = x,
    // x + 1 or x - n in [0.5, 2.5) (z - x0 from x itself when z = x + 1).
    const bool below = x < 0.5, above = x >= 2.5;
    const double n = above ? std::floor(x - 1.5) : 0, z = x - n;
    const double th = below ? x - (digamma_root_hi - 1) : z - digamma_root_hi;
    const double v = below ? x + 0.5 : z - 0.5, p = horner(digamma_base, v);
    const double d = std::fma(std::fma(v + 4.5, v, 5.75), v, 1.875);
    const double psi = std::fma(th, p, -digamma_root_lo * p) / d;
    if (below)
        return psi - 1 / x;
    double sum = 0;
    for (double j = 1; j <= n; ++j)
        sum += 1 / (x - j);
    return psi + sum;
}
RAPIDGLM_VMATH_KERNEL double trigamma(double x) {
    if (!(x > 0) || x == inf)
        return x != x ? x : x == 0 ? inf : x > 0 ? 0 : nan;
    // psi'(x) = sum_{j < m} 1/(x + j)^2 + psi'(x + m), m = ceil(6 - x) from 0 to 6.
    const double m = std::min(std::max(0.0, std::ceil(trigamma_series - x)), trigamma_series);
    // The correction w (1/2 + r H(w)) is dropped (w = 0) for r below 2^-500, far
    // under half a unit of r there, which keeps 1/y^2 out of the (slow)
    // subnormal range.
    const double y = x + m, r = 1 / y, w = r < 0x1p-500 ? 0 : r * r;
    double sum = std::fma(w, std::fma(r, horner(trigamma_large, w), 0.5), r);
    for (double j = m; j-- > 0;) {
        const double rj = 1 / (x + j);
        sum = std::fma(rj, rj, sum);
    }
    return sum;
}
// The portable path's instances (std::fma: the C library's fma on x86-64).
RAPIDGLM_VMATH_INSTANCE double log_portable(double x) {
    return log(x);
}
RAPIDGLM_VMATH_INSTANCE double log1p_portable(double x) {
    return log1p(x);
}
RAPIDGLM_VMATH_INSTANCE double exp_portable(double x) {
    return exp(x);
}
RAPIDGLM_VMATH_INSTANCE double digamma_portable(double x) {
    return digamma(x);
}
RAPIDGLM_VMATH_INSTANCE double trigamma_portable(double x) {
    return trigamma(x);
}

#ifdef RAPIDGLM_VMATH_AVX2
#define RAPIDGLM_VMATH_TARGET __attribute__((target("avx2,fma"))) inline
#define RAPIDGLM_VMATH_TARGET_KERNEL __attribute__((target("avx2,fma"), always_inline)) inline
#define RAPIDGLM_VMATH_TARGET_INSTANCE __attribute__((target("avx2,fma"), noinline)) inline
// The scalar code compiled for AVX2 and FMA.
RAPIDGLM_VMATH_TARGET_INSTANCE double log_fma(double x) {
    return log(x);
}
RAPIDGLM_VMATH_TARGET_INSTANCE double log1p_fma(double x) {
    return log1p(x);
}
RAPIDGLM_VMATH_TARGET_INSTANCE double exp_fma(double x) {
    return exp(x);
}
RAPIDGLM_VMATH_TARGET_INSTANCE double digamma_fma(double x) {
    return digamma(x);
}
RAPIDGLM_VMATH_TARGET_INSTANCE double trigamma_fma(double x) {
    return trigamma(x);
}
// Four lanes at a time, the operations of the scalar code; lanes outside the
// main path's domain go through the scalar code.
namespace avx2 {
RAPIDGLM_VMATH_TARGET_KERNEL __m256d set(double c) {
    return _mm256_set1_pd(c);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256i set_bits(std::uint64_t c) {
    return _mm256_set1_epi64x(static_cast<long long>(c));
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d fma(__m256d a, __m256d b, __m256d c) {
    return _mm256_fmadd_pd(a, b, c);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d select(__m256d mask, __m256d yes, __m256d no) {
    return _mm256_blendv_pd(no, yes, mask);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d lt(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d ge(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d eq(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}
RAPIDGLM_VMATH_TARGET_KERNEL bool any(__m256d mask) {
    return _mm256_movemask_pd(mask) != 0;
}
RAPIDGLM_VMATH_TARGET_KERNEL double largest(__m256d v) {
    const __m128d h = _mm_max_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_max_sd(h, _mm_unpackhi_pd(h, h)));
}
template <std::size_t N>
RAPIDGLM_VMATH_TARGET_KERNEL __m256d horner(const double (&c)[N], __m256d t) {
    __m256d p = set(c[N - 1]);
    for (std::size_t j = N - 1; j-- > 0;)
        p = fma(p, t, set(c[j]));
    return p;
}
template <std::size_t N>
RAPIDGLM_VMATH_TARGET_KERNEL __m256d estrin(const __m256d (&p)[N], __m256d t) {
    if constexpr (N == 1) {
        return p[0];
    } else {
        __m256d q[(N + 1) / 2];
        for (std::size_t j = 0; j < N / 2; ++j)
            q[j] = fma(p[2 * j + 1], t, p[2 * j]);
        if constexpr (N % 2 == 1)
            q[N / 2] = p[N - 1];
        return estrin(q, _mm256_mul_pd(t, t));
    }
}
template <std::size_t N>
RAPIDGLM_VMATH_TARGET_KERNEL __m256d estrin(const double (&c)[N], __m256d t) {
    __m256d q[(N + 1) / 2];
    for (std::size_t j = 0; j < N / 2; ++j)
        q[j] = fma(set(c[2 * j + 1]), t, set(c[2 * j]));
    if constexpr (N % 2 == 1)
        q[N / 2] = set(c[N - 1]);
    return estrin(q, _mm256_mul_pd(t, t));
}
template <bool Correction> RAPIDGLM_VMATH_TARGET_KERNEL __m256d log_reduced(__m256d u, __m256d c) {
    const __m256d one = set(1), minus_half = set(-0.5);
    const __m256i ub = _mm256_castpd_si256(u), t = _mm256_sub_epi64(ub, set_bits(sqrt_half_bits)),
                  kb = _mm256_and_si256(t, set_bits(exponent_mask));
    const __m256d m = _mm256_castsi256_pd(_mm256_sub_epi64(ub, kb));
    const __m256d k = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(
            _mm256_srli_epi64(_mm256_xor_si256(t, set_bits(sign_bit)), 52), set_bits(magic_bits))),
        set(magic_k));
    __m256d g = _mm256_sub_pd(m, one), r = _mm256_setzero_pd();
    if (Correction) {
        const __m256d f = g, cs = _mm256_mul_pd(
                                 c, _mm256_castsi256_pd(_mm256_sub_epi64(set_bits(one_bits), kb)));
        g = _mm256_add_pd(f, cs);
        r = _mm256_add_pd(_mm256_sub_pd(f, g), cs);
    }
    const __m256d g2 = _mm256_mul_pd(g, g), half = _mm256_mul_pd(minus_half, g2);
    const __m256d hi = fma(k, set(log2_hi), g);
    const __m256d low = _mm256_sub_pd(g, _mm256_fnmadd_pd(k, set(log2_hi), hi));
    const __m256d s = _mm256_add_pd(hi, half), e = _mm256_sub_pd(half, _mm256_sub_pd(s, hi));
    __m256d tail = _mm256_add_pd(low, e);
    if (Correction)
        tail = fma(r, fma(g, g, _mm256_sub_pd(one, g)), tail);
    tail = fma(k, set(log2_lo), tail);
    tail = fma(_mm256_fmsub_pd(g, g, g2), minus_half, tail);
    tail = fma(_mm256_mul_pd(g2, g), estrin(log_mantissa, g), tail);
    return _mm256_add_pd(s, tail);
}
// The lanes not in `ok` from the scalar function F instead.
template <double (*F)(double)>
__attribute__((target("avx2,fma"), noinline)) __m256d patch(__m256d x, __m256d value, __m256d ok) {
    alignas(32) double in[4], out[4];
    _mm256_store_pd(in, x);
    _mm256_store_pd(out, value);
    const int good = _mm256_movemask_pd(ok);
    for (int lane = 0; lane < 4; ++lane)
        if (!(good >> lane & 1))
            out[lane] = F(in[lane]);
    return _mm256_load_pd(out);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d log(__m256d x) {
    const __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, set(log_min), _CMP_GE_OQ),
                                     _mm256_cmp_pd(x, set(inf), _CMP_LT_OQ));
    // Lanes outside the main domain are evaluated at 1 (no subnormal arithmetic)
    // and then replaced.
    const bool all = _mm256_movemask_pd(ok) == 0xf;
    const __m256d y = all ? x : select(ok, x, set(1));
    const __m256d value = log_reduced<false>(y, _mm256_setzero_pd());
    if (all)
        return value;
    return patch<log_fma>(x, value, ok);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d log1p(__m256d x) {
    const __m256d one = set(1);
    const __m256d ok =
        _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(x, set(-1), _CMP_GT_OQ),
                                    _mm256_cmp_pd(x, set(log1p_limit), _CMP_LT_OQ)),
                      _mm256_cmp_pd(_mm256_andnot_pd(set(-0.0), x), set(log1p_tiny), _CMP_GE_OQ));
    // As in log: the other lanes at 1, then replaced.
    const bool all = _mm256_movemask_pd(ok) == 0xf;
    const __m256d y = all ? x : select(ok, x, one);
    const __m256d u = _mm256_add_pd(y, one);
    const __m256d c = _mm256_sub_pd(_mm256_min_pd(y, one), _mm256_sub_pd(u, _mm256_max_pd(y, one)));
    const __m256d value = log_reduced<true>(u, c);
    if (all)
        return value;
    return patch<log1p_fma>(x, value, ok);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d exp(__m256d x) {
    const __m256d ok = _mm256_cmp_pd(_mm256_andnot_pd(set(-0.0), x), set(exp_limit), _CMP_LE_OQ);
    const __m256d k = _mm256_round_pd(_mm256_mul_pd(x, set(exp_log2e)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, set(exp_ln2_hi), x);
    r = _mm256_fnmadd_pd(k, set(exp_ln2_lo), r);
    const __m256d p = horner(exp_taylor, r);
    const __m256i power =
        _mm256_slli_epi64(_mm256_castpd_si256(_mm256_add_pd(k, set(exp_shift))), 52);
    const __m256d value = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(p), power));
    if (_mm256_movemask_pd(ok) == 0xf)
        return value;
    return patch<exp_fma>(x, value, ok);
}
// Lanes with NaN keep it; x < 0 gives NaN; x == 0 and x == +inf the given values.
RAPIDGLM_VMATH_TARGET_KERNEL __m256d specials(__m256d x, __m256d value, double at_zero,
                                              double at_inf) {
    value = select(lt(x, _mm256_setzero_pd()), set(nan), value);
    value = select(eq(x, _mm256_setzero_pd()), set(at_zero), value);
    value = select(eq(x, set(inf)), set(at_inf), value);
    return select(_mm256_cmp_pd(x, x, _CMP_UNORD_Q), x, value);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d digamma(__m256d x) {
    const __m256d large = ge(x, set(digamma_series)), below = lt(x, set(0.5)),
                  above = _mm256_andnot_pd(large, ge(x, set(2.5)));
    __m256d value = _mm256_setzero_pd();
    if (_mm256_movemask_pd(large) != 0xf) {
        const __m256d n = _mm256_and_pd(above, _mm256_floor_pd(_mm256_sub_pd(x, set(1.5))));
        const __m256d z = _mm256_sub_pd(x, n);
        const __m256d th = select(below, _mm256_sub_pd(x, set(digamma_root_hi - 1)),
                                  _mm256_sub_pd(z, set(digamma_root_hi)));
        const __m256d v = select(below, _mm256_add_pd(x, set(0.5)), _mm256_sub_pd(z, set(0.5)));
        const __m256d p = horner(digamma_base, v);
        const __m256d d = fma(fma(_mm256_add_pd(v, set(4.5)), v, set(5.75)), v, set(1.875));
        const __m256d psi = _mm256_div_pd(fma(th, p, _mm256_mul_pd(set(-digamma_root_lo), p)), d);
        value = psi;
        if (any(below))
            value = select(below, _mm256_sub_pd(psi, _mm256_div_pd(set(1), x)), value);
        if (any(above)) {
            __m256d sum = _mm256_setzero_pd();
            const double steps = largest(n); // lanes with fewer terms add zeros
            for (double j = 1; j <= steps; ++j)
                sum = _mm256_add_pd(sum,
                                    _mm256_and_pd(_mm256_cmp_pd(set(j), n, _CMP_LE_OQ),
                                                  _mm256_div_pd(set(1), _mm256_sub_pd(x, set(j)))));
            value = select(above, _mm256_add_pd(psi, sum), value);
        }
    }
    if (any(large)) {
        // log x in the lanes with x >= 5 (the others take log 5, unused).
        const __m256d r = _mm256_div_pd(set(1), x), rg = _mm256_max_pd(r, set(0x1p-500));
        const __m256d g = horner(digamma_large, _mm256_mul_pd(rg, rg));
        const __m256d logs =
            log_reduced<false>(select(large, x, set(digamma_series)), _mm256_setzero_pd());
        value = select(large, fma(r, fma(r, g, set(-0.5)), logs), value);
    }
    return specials(x, value, -inf, inf);
}
RAPIDGLM_VMATH_TARGET_KERNEL __m256d trigamma(__m256d x) {
    const __m256d limit = set(trigamma_series);
    const __m256d m = _mm256_min_pd(
        _mm256_max_pd(_mm256_ceil_pd(_mm256_sub_pd(limit, x)), _mm256_setzero_pd()), limit);
    const __m256d y = _mm256_add_pd(x, m), r = _mm256_div_pd(set(1), y);
    const __m256d rw = _mm256_max_pd(r, set(0x1p-500));
    const __m256d w = _mm256_andnot_pd(lt(r, set(0x1p-500)), _mm256_mul_pd(rw, rw));
    __m256d sum = fma(w, fma(r, horner(trigamma_large, w), set(0.5)), r);
    const double steps = largest(m); // lanes with fewer terms leave the sum unchanged
    for (double j = steps; j-- > 0;) {
        const __m256d ra = _mm256_div_pd(set(1), _mm256_add_pd(x, set(j)));
        sum = select(lt(set(j), m), fma(ra, ra, sum), sum);
    }
    return specials(x, sum, inf, 0);
}
// out[i] = f(x[i]) four at a time, the last ones masked; out may be x.
template <__m256d (*F)(__m256d)>
RAPIDGLM_VMATH_TARGET void apply(const double *x, double *out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, F(_mm256_loadu_pd(x + i)));
    if (i < n) {
        const std::size_t lanes = n - i;
        const __m256i mask = _mm256_setr_epi64x(-1, lanes > 1 ? -1 : 0, lanes > 2 ? -1 : 0, 0);
        _mm256_maskstore_pd(out + i, mask, F(_mm256_maskload_pd(x + i, mask)));
    }
}
} // namespace avx2
#undef RAPIDGLM_VMATH_TARGET_INSTANCE
#undef RAPIDGLM_VMATH_TARGET_KERNEL
#undef RAPIDGLM_VMATH_TARGET
#endif
#ifdef RAPIDGLM_VMATH_NEON
// log1p and exp two lanes at a time, the operations of the scalar code; digamma
// and trigamma one value at a time (the scalar code has the FMA instruction here).
namespace neon {
inline float64x2_t set(double c) {
    return vdupq_n_f64(c);
}
inline uint64x2_t set_bits(std::uint64_t c) {
    return vdupq_n_u64(c);
}
inline float64x2_t fma(float64x2_t a, float64x2_t b, float64x2_t c) {
    return vfmaq_f64(c, a, b);
}
template <std::size_t N> inline float64x2_t horner(const double (&c)[N], float64x2_t t) {
    float64x2_t p = set(c[N - 1]);
    for (std::size_t j = N - 1; j-- > 0;)
        p = fma(p, t, set(c[j]));
    return p;
}
template <std::size_t N>
RAPIDGLM_VMATH_KERNEL float64x2_t estrin(const float64x2_t (&p)[N], float64x2_t t) {
    if constexpr (N == 1) {
        return p[0];
    } else {
        float64x2_t q[(N + 1) / 2];
        for (std::size_t j = 0; j < N / 2; ++j)
            q[j] = fma(p[2 * j + 1], t, p[2 * j]);
        if constexpr (N % 2 == 1)
            q[N / 2] = p[N - 1];
        return estrin(q, vmulq_f64(t, t));
    }
}
template <std::size_t N>
RAPIDGLM_VMATH_KERNEL float64x2_t estrin(const double (&c)[N], float64x2_t t) {
    float64x2_t q[(N + 1) / 2];
    for (std::size_t j = 0; j < N / 2; ++j)
        q[j] = fma(set(c[2 * j + 1]), t, set(c[2 * j]));
    if constexpr (N % 2 == 1)
        q[N / 2] = set(c[N - 1]);
    return estrin(q, vmulq_f64(t, t));
}
template <bool Correction>
RAPIDGLM_VMATH_KERNEL float64x2_t log_reduced(float64x2_t u, float64x2_t c) {
    const float64x2_t one = set(1), minus_half = set(-0.5);
    const uint64x2_t ub = vreinterpretq_u64_f64(u), t = vsubq_u64(ub, set_bits(sqrt_half_bits)),
                     kb = vandq_u64(t, set_bits(exponent_mask));
    const float64x2_t m = vreinterpretq_f64_u64(vsubq_u64(ub, kb));
    const float64x2_t k =
        vsubq_f64(vreinterpretq_f64_u64(vorrq_u64(vshrq_n_u64(veorq_u64(t, set_bits(sign_bit)), 52),
                                                  set_bits(magic_bits))),
                  set(magic_k));
    float64x2_t g = vsubq_f64(m, one), r = vdupq_n_f64(0);
    if (Correction) {
        const float64x2_t f = g, cs = vmulq_f64(
                                     c, vreinterpretq_f64_u64(vsubq_u64(set_bits(one_bits), kb)));
        g = vaddq_f64(f, cs);
        r = vaddq_f64(vsubq_f64(f, g), cs);
    }
    const float64x2_t g2 = vmulq_f64(g, g), half = vmulq_f64(minus_half, g2);
    const float64x2_t hi = fma(k, set(log2_hi), g);
    const float64x2_t low = vsubq_f64(g, vfmsq_f64(hi, k, set(log2_hi)));
    const float64x2_t s = vaddq_f64(hi, half), e = vsubq_f64(half, vsubq_f64(s, hi));
    float64x2_t tail = vaddq_f64(low, e);
    if (Correction)
        tail = fma(r, fma(g, g, vsubq_f64(one, g)), tail);
    tail = fma(k, set(log2_lo), tail);
    tail = fma(vfmaq_f64(vnegq_f64(g2), g, g), minus_half, tail);
    tail = fma(vmulq_f64(g2, g), estrin(log_mantissa, g), tail);
    return vaddq_f64(s, tail);
}
// The lanes not in `ok` from the scalar function F instead.
template <double (*F)(double)>
__attribute__((noinline)) float64x2_t patch(float64x2_t x, float64x2_t value, uint64x2_t ok) {
    if (!vgetq_lane_u64(ok, 0))
        value = vsetq_lane_f64(F(vgetq_lane_f64(x, 0)), value, 0);
    if (!vgetq_lane_u64(ok, 1))
        value = vsetq_lane_f64(F(vgetq_lane_f64(x, 1)), value, 1);
    return value;
}
inline bool all(uint64x2_t ok) {
    return (vgetq_lane_u64(ok, 0) & vgetq_lane_u64(ok, 1)) != 0;
}
RAPIDGLM_VMATH_KERNEL float64x2_t log(float64x2_t x) {
    const uint64x2_t ok = vandq_u64(vcgeq_f64(x, set(log_min)), vcltq_f64(x, set(inf)));
    // Lanes outside the main domain are evaluated at 1, then replaced.
    const bool good = all(ok);
    const float64x2_t y = good ? x : vbslq_f64(ok, x, set(1));
    const float64x2_t value = log_reduced<false>(y, vdupq_n_f64(0));
    return good ? value : patch<detail::log_portable>(x, value, ok);
}
RAPIDGLM_VMATH_KERNEL float64x2_t log1p(float64x2_t x) {
    const float64x2_t one = set(1);
    const uint64x2_t ok =
        vandq_u64(vandq_u64(vcgtq_f64(x, set(-1)), vcltq_f64(x, set(log1p_limit))),
                  vcgeq_f64(vabsq_f64(x), set(log1p_tiny)));
    const bool good = all(ok);
    const float64x2_t y = good ? x : vbslq_f64(ok, x, one);
    const float64x2_t u = vaddq_f64(y, one);
    const float64x2_t c = vsubq_f64(vminq_f64(y, one), vsubq_f64(u, vmaxq_f64(y, one)));
    const float64x2_t value = log_reduced<true>(u, c);
    return good ? value : patch<detail::log1p_portable>(x, value, ok);
}
RAPIDGLM_VMATH_KERNEL float64x2_t exp(float64x2_t x) {
    const uint64x2_t ok = vcleq_f64(vabsq_f64(x), set(exp_limit));
    const float64x2_t k = vrndnq_f64(vmulq_f64(x, set(exp_log2e)));
    float64x2_t r = vfmsq_f64(x, k, set(exp_ln2_hi));
    r = vfmsq_f64(r, k, set(exp_ln2_lo));
    const float64x2_t p = horner(exp_taylor, r);
    const uint64x2_t power = vshlq_n_u64(vreinterpretq_u64_f64(vaddq_f64(k, set(exp_shift))), 52);
    const float64x2_t value = vreinterpretq_f64_u64(vaddq_u64(vreinterpretq_u64_f64(p), power));
    return all(ok) ? value : patch<detail::exp_portable>(x, value, ok);
}
template <float64x2_t (*F)(float64x2_t)>
inline void apply(const double *x, double *out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(out + i, F(vld1q_f64(x + i)));
    if (i < n)
        out[i] = vgetq_lane_f64(F(vdupq_n_f64(x[i])), 0);
}
} // namespace neon
#endif
} // namespace detail

// Whether RapidGLM's functions take the AVX2 path on this processor.
inline bool avx2_active() {
#ifdef RAPIDGLM_VMATH_AVX2
    static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return available;
#else
    return false;
#endif
}
// The path RapidGLM's functions take here: "avx2", "neon" or "portable".
inline const char *path() {
#if defined(RAPIDGLM_VMATH_NEON)
    return "neon";
#else
    return avx2_active() ? "avx2" : "portable";
#endif
}

namespace slow {
// The C library's lgamma: lgamma(x) for one value and lgamma_array(x, out, n)
// over n values (out may be x); lgamma_array_portable is lgamma_array.
inline double lgamma(double x) {
    return detail::lgamma(x);
}
inline void lgamma_array(const double *x, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = detail::lgamma(x[i]);
}
inline void lgamma_array_portable(const double *x, double *out, std::size_t n) {
    lgamma_array(x, out, n);
}

// RapidGLM's functions: f(x) for one value, f_array(x, out, n) over n values (out
// may be x), and the paths f_array_portable and, with AVX2 compiled in,
// f_array_avx2 (four lanes at a time).
#define RAPIDGLM_VMATH_PORTABLE(name)                                                              \
    inline void name##_array_portable(const double *x, double *out, std::size_t n) {               \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = detail::name(x[i]);                                                           \
    }
#if defined(RAPIDGLM_VMATH_AVX2)
#define RAPIDGLM_VMATH_FUNCTION(name, kernel)                                                      \
    RAPIDGLM_VMATH_PORTABLE(name)                                                                  \
    __attribute__((target("avx2,fma"))) inline void name##_array_avx2(                             \
        const double *x, double *out, std::size_t n) {                                             \
        kernel(x, out, n);                                                                         \
    }                                                                                              \
    inline void name##_array(const double *x, double *out, std::size_t n) {                        \
        if (avx2_active())                                                                         \
            return name##_array_avx2(x, out, n);                                                   \
        name##_array_portable(x, out, n);                                                          \
    }                                                                                              \
    inline double name(double x) {                                                                 \
        return avx2_active() ? detail::name##_fma(x) : detail::name##_portable(x);                 \
    }
RAPIDGLM_VMATH_FUNCTION(log, detail::avx2::apply<detail::avx2::log>)
RAPIDGLM_VMATH_FUNCTION(log1p, detail::avx2::apply<detail::avx2::log1p>)
RAPIDGLM_VMATH_FUNCTION(exp, detail::avx2::apply<detail::avx2::exp>)
RAPIDGLM_VMATH_FUNCTION(digamma, detail::avx2::apply<detail::avx2::digamma>)
RAPIDGLM_VMATH_FUNCTION(trigamma, detail::avx2::apply<detail::avx2::trigamma>)
#elif defined(RAPIDGLM_VMATH_NEON)
#define RAPIDGLM_VMATH_FUNCTION(name, kernel)                                                      \
    RAPIDGLM_VMATH_PORTABLE(name)                                                                  \
    inline void name##_array(const double *x, double *out, std::size_t n) {                        \
        kernel(x, out, n);                                                                         \
    }                                                                                              \
    inline double name(double x) {                                                                 \
        return detail::name##_portable(x);                                                         \
    }
RAPIDGLM_VMATH_FUNCTION(log, detail::neon::apply<detail::neon::log>)
RAPIDGLM_VMATH_FUNCTION(log1p, detail::neon::apply<detail::neon::log1p>)
RAPIDGLM_VMATH_FUNCTION(exp, detail::neon::apply<detail::neon::exp>)
RAPIDGLM_VMATH_FUNCTION(digamma, digamma_array_portable)
RAPIDGLM_VMATH_FUNCTION(trigamma, trigamma_array_portable)
#else
#define RAPIDGLM_VMATH_FUNCTION(name)                                                              \
    RAPIDGLM_VMATH_PORTABLE(name)                                                                  \
    inline void name##_array(const double *x, double *out, std::size_t n) {                        \
        name##_array_portable(x, out, n);                                                          \
    }                                                                                              \
    inline double name(double x) {                                                                 \
        return detail::name##_portable(x);                                                         \
    }
RAPIDGLM_VMATH_FUNCTION(log)
RAPIDGLM_VMATH_FUNCTION(log1p)
RAPIDGLM_VMATH_FUNCTION(exp)
RAPIDGLM_VMATH_FUNCTION(digamma)
RAPIDGLM_VMATH_FUNCTION(trigamma)
#endif
#undef RAPIDGLM_VMATH_FUNCTION
#undef RAPIDGLM_VMATH_PORTABLE
} // namespace slow

// Division-based choices measured on Zen 3. log1p and lgamma are RapidGLM's
// earlier polynomial kernels; the selected digamma rational approximation is
// derived from Boost.Math (John Maddock, Copyright 2006), distributed under
// the Boost Software License 1.0, https://www.boost.org/LICENSE_1_0.txt.
// Tests cover identical operation order on AVX2, portable and AArch64 paths.
namespace fast {
namespace own {

namespace detail {
// Generated by an mpmath script (METHODS.md): Chebyshev interpolants in the
// local variable named, converted to monomials and rounded to double.
inline constexpr double ln2_hi = 0.6931471805599453, ln2_lo = 2.3190468138462996e-17;
inline constexpr double half_log_two_pi_minus_half = 0.4189385332046727;
inline constexpr double sqrt_half_minus_one = -0.29289321881345254;
inline constexpr double lgamma_base_low = 1.504604114787798e-17; // lgamma_base[0] rounding error
// log: (2 artanh(s)/s - 2)/z in z = s^2 on [0, 0.0294373], degree 7.
inline constexpr double log_poly[8] = {
    0.6666666666666666,  0.4000000000000088,  0.28571428570803614, 0.22222222391713917,
    0.18181795640132906, 0.15386239702814658, 0.13268773138656886, 0.13086626147840102};
// lgamma: (1/Gamma(z) - 1)/((z - 1)(z - 2)) in z - 2.375 on [0.5, 3], degree 18.
inline constexpr double lgamma_base[19] = {
    -0.3526611360033088,     0.1814961848948937,     -0.021067583270863072,
    -0.014335993799597472,   0.006949682596911872,   -0.0010318493494285792,
    -0.00016785831720668182, 0.00010628820247749711, -1.9356218450761406e-05,
    1.0519815091165423e-07,  7.387250014862306e-07,  -1.7254151249225575e-07,
    1.4360459434355036e-08,  1.967784979012889e-09,  -7.886582743657398e-10,
    9.525634233459452e-11,   -9.341268892248155e-12, -3.1932744171249133e-12,
    2.9664737893958424e-13};
// lgamma: x (lgamma(x) - (x - 1/2) log x + x - log(2 pi)/2) in w = 1/x^2, x >= 6, degree 7.
inline constexpr double lgamma_large[8] = {
    0.08333333333333333,   -0.0027777777777734623, 0.0007936507903643918, -0.0005952371351851137,
    0.0008416117568838537, -0.001906405156123843,  0.005901549297067006,  -0.016450184918873966};
// digamma: psi(z) z (z + 1)(z + 2)/(z - x0) in z - 0.5 on [0.5, 2.5], degree 16.
// digamma: x^2 (psi(x) - log x + 1/(2x)) in w = 1/x^2, x >= 5, degree 9.
// trigamma: x^3 (psi'(x) - 1/x - 1/(2 x^2)) in w = 1/x^2, x >= 6, degree 8.
inline constexpr double inf = std::numeric_limits<double>::infinity();
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
// Thresholds: lgamma and trigamma switch to series at 6, digamma at 5.
inline constexpr double lgamma_series = 6;
inline constexpr std::uint64_t mantissa_mask = 0x000fffffffffffffULL;
inline constexpr std::uint64_t sqrt_half_bits = 0x3fe6a09e667f3bcdULL; // sqrt(1/2) rounded
inline constexpr std::uint64_t exponent_shift = 0x3ff0000000000000ULL - sqrt_half_bits;

inline std::uint64_t bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
RAPIDGLM_VMATH_KERNEL double from_bits(std::uint64_t u) {
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
// c[first] + c[first + 1] t + ... by Horner's rule.
template <std::size_t N>
RAPIDGLM_VMATH_KERNEL double horner(const double (&c)[N], double t, std::size_t first = 0) {
    double p = c[N - 1];
    for (std::size_t j = N - 1; j-- > first;)
        p = std::fma(p, t, c[j]);
    return p;
}
// w > 0 finite as 2^k m, m in [sqrt(1/2), sqrt(2)): f = m - 1 (exact) and k.
RAPIDGLM_VMATH_KERNEL void reduce(double w, double &f, double &k) {
    double bias = 0;
    if (w < 0x1p-1022) { // subnormal: scale into the normal range exactly
        w *= 0x1p54;
        bias = -54;
    }
    const std::uint64_t u = bits(w) + exponent_shift;
    k = static_cast<double>(static_cast<std::int64_t>(u >> 52)) - 1023 + bias;
    f = from_bits((u & mantissa_mask) + sqrt_half_bits) - 1;
}
// log(1 + f) + k log 2 + extra for f in [sqrt(1/2) - 1, sqrt(2) - 1], given
// d = 2 + f (computed by the caller, so that no product feeds the addition), an
// integer k and a correction `extra` far below the result's last place. The
// rounding error of f - f^2/2 (f - h is exact) joins the small terms.
RAPIDGLM_VMATH_KERNEL double log_kernel(double f, double d, double k, double extra) {
    const double s = f / d, z = s * s, half = 0.5 * f;
    const double p = horner(log_poly, z);
    const double low = std::fma(k, ln2_lo, extra);
    const double t = std::fma(s, std::fma(z, p, half * f), low); // s (f^2/2 + R) + low
    const double h = std::fma(-half, f, f), e = std::fma(-half, f, f - h);
    return std::fma(k, ln2_hi, h + (t + e));
}
RAPIDGLM_VMATH_KERNEL double log_reduced(double w, double extra) {
    double f, k;
    reduce(w, f, k);
    return log_kernel(f, 2 + f, k, extra);
}
} // namespace detail

RAPIDGLM_VMATH_KERNEL double log(double x) {
    if (!(x > 0) || x == detail::inf)
        return x != x ? x : x == 0 ? -detail::inf : x > 0 ? x : detail::nan;
    return detail::log_reduced(x, 0);
}
RAPIDGLM_VMATH_KERNEL double log1p(double x) {
    if (!(x > -1) || x == detail::inf || x == 0)
        return x != x || x == 0 || x == detail::inf ? x : x == -1 ? -detail::inf : detail::nan;
    // 1 + x = u + c exactly: c = x - (u - 1) for x <= 1 and 1 - (u - x) above
    // (both differences exact), and log(u + c) = log u + c/u to within a last place.
    const double u = 1 + x, c = x > 1 ? 1 - (u - x) : x - (u - 1);
    return detail::log_reduced(u, c / u);
}
RAPIDGLM_VMATH_KERNEL double lgamma(double x) {
    using namespace detail;
    if (!(x > 0) || x == inf)
        return x != x ? x : x >= 0 ? inf : nan;
    if (x >= lgamma_series) {
        // (x - 1/2)(log x - 1) + log(2 pi)/2 - 1/2 + c(x), c(x) = F(1/x^2)/x.
        const double r = 1 / x, p = horner(lgamma_large, r * r);
        return std::fma(x - 0.5, log_reduced(x, 0) - 1, std::fma(r, p, half_log_two_pi_minus_half));
    }
    // 1/Gamma(z) = 1 + u, u = a1 a2 K(t), for z = x, x + 1 or x - n in [0.5, 3):
    // a1 = z - 1, a2 = z - 2 and t = z - 2.375 (from x itself when z = x + 1).
    // K = c0 + t q(t) is carried as kh + kl and u as uh + ul (exact products).
    const bool below = x < 0.5, above = x >= 3;
    const double n = above ? std::floor(x) - 2 : 0, z = x - n;
    const double a1 = below ? x : z - 1, a2 = below ? x - 1 : z - 2;
    const double t = below ? x - 1.375 : z - 2.375;
    const double q = horner(lgamma_base, t, 1);
    const double kh = std::fma(t, q, lgamma_base[0]);
    const double kl = std::fma(t, q, lgamma_base[0] - kh) + lgamma_base_low;
    const double a = a1 * a2, al = std::fma(a1, a2, -a);
    const double uh = a * kh, ul = std::fma(al, kh, std::fma(a, kl, std::fma(a, kh, -uh)));
    if (!below && !above) {
        // lgamma(x) = -log(1 + u): f = uh, or f = 1 + 2 uh (exact) and k = -1 where
        // 1 + u < sqrt(1/2); ul (twice that) over 1 + f ~ 1 - f + f^2. 0 - l rather
        // than -l, so that lgamma(1) = lgamma(2) = +0.
        const bool low = uh < sqrt_half_minus_one;
        const double f = low ? std::fma(2, uh, 1) : uh,
                     d = low ? std::fma(2, uh, 3) : std::fma(a, kh, 2);
        const double e = low ? 2 * ul : ul, g = std::fma(f, f - 1, 1);
        return 0 - log_kernel(f, d, low ? -1 : 0, e * g);
    }
    const double recip = std::fma(a, kh, 1),
                 c = std::fma(a, kh, 1 - recip) + ul; // 1/Gamma(z) = recip + c
    double w, num, den; // log(w) + num/den: log of 1/Gamma(x) (below) or Gamma(x) (above)
    if (below) {
        w = x * recip;
        num = std::fma(x, c, std::fma(x, recip, -w));
        den = w;
    } else {
        double p = x - 1, pl = 0; // (x - 1)..(x - n) = p + pl
        for (double j = 2; j <= 3; ++j) {
            const double factor = j <= n ? x - j : 1, next = p * factor;
            pl = std::fma(pl, factor, std::fma(p, factor, -next));
            p = next;
        }
        w = p / recip;
        num = std::fma(-w, c, std::fma(-w, recip, p)) + pl;
        den = p;
    }
    const double l = log_reduced(w, num / den);
    return above ? l : 0 - l;
}
#ifdef RAPIDGLM_VMATH_AVX2
// The same operations four lanes at a time; regions no lane needs are skipped.
namespace detail::avx2 {
#define RAPIDGLM_VMATH_FAST_TARGET __attribute__((target("avx2,fma"))) inline
RAPIDGLM_VMATH_FAST_TARGET __m256d set(double c) {
    return _mm256_set1_pd(c);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d select(__m256d mask, __m256d yes, __m256d no) {
    return _mm256_blendv_pd(no, yes, mask);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d lt(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d ge(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d eq(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}
RAPIDGLM_VMATH_FAST_TARGET bool any(__m256d mask) {
    return _mm256_movemask_pd(mask) != 0;
}
template <std::size_t N>
RAPIDGLM_VMATH_FAST_TARGET __m256d horner(const double (&c)[N], __m256d t, std::size_t first = 0) {
    __m256d p = set(c[N - 1]);
    for (std::size_t j = N - 1; j-- > first;)
        p = _mm256_fmadd_pd(p, t, set(c[j]));
    return p;
}
RAPIDGLM_VMATH_FAST_TARGET void reduce(__m256d w, __m256d &f, __m256d &k) {
    const __m256d subnormal = lt(w, set(0x1p-1022));
    w = select(subnormal, _mm256_mul_pd(w, set(0x1p54)), w);
    const __m256d bias = _mm256_and_pd(subnormal, set(-54.0));
    const __m256i u = _mm256_add_epi64(_mm256_castpd_si256(w),
                                       _mm256_set1_epi64x(static_cast<long long>(exponent_shift)));
    // The biased exponent (0 to 2047) as a double: its bits under those of 2^52.
    const __m256d e =
        _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(
                          _mm256_srli_epi64(u, 52), _mm256_set1_epi64x(0x4330000000000000LL))),
                      set(0x1p52));
    k = _mm256_add_pd(_mm256_sub_pd(e, set(1023)), bias);
    const __m256i m = _mm256_add_epi64(
        _mm256_and_si256(u, _mm256_set1_epi64x(static_cast<long long>(mantissa_mask))),
        _mm256_set1_epi64x(static_cast<long long>(sqrt_half_bits)));
    f = _mm256_sub_pd(_mm256_castsi256_pd(m), set(1));
}
RAPIDGLM_VMATH_FAST_TARGET __m256d log_kernel(__m256d f, __m256d d, __m256d k, __m256d extra) {
    const __m256d s = _mm256_div_pd(f, d), z = _mm256_mul_pd(s, s),
                  half = _mm256_mul_pd(set(0.5), f);
    const __m256d p = horner(log_poly, z);
    const __m256d low = _mm256_fmadd_pd(k, set(ln2_lo), extra);
    const __m256d t = _mm256_fmadd_pd(s, _mm256_fmadd_pd(z, p, _mm256_mul_pd(half, f)), low);
    const __m256d h = _mm256_fnmadd_pd(half, f, f),
                  e = _mm256_fnmadd_pd(half, f, _mm256_sub_pd(f, h));
    return _mm256_fmadd_pd(k, set(ln2_hi), _mm256_add_pd(h, _mm256_add_pd(t, e)));
}
RAPIDGLM_VMATH_FAST_TARGET __m256d log_reduced(__m256d w, __m256d extra) {
    __m256d f, k;
    reduce(w, f, k);
    return log_kernel(f, _mm256_add_pd(set(2), f), k, extra);
}
// Lanes with NaN keep it; x < 0 gives NaN; x == 0 and x == +inf the given values.
RAPIDGLM_VMATH_FAST_TARGET __m256d specials(__m256d x, __m256d value, double at_zero, double at_inf,
                                            double lower) {
    value = select(lt(x, set(lower)), set(nan), value);
    value = select(eq(x, set(lower)), set(at_zero), value);
    value = select(eq(x, set(inf)), set(at_inf), value);
    return select(_mm256_cmp_pd(x, x, _CMP_UNORD_Q), x, value);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d log(__m256d x) {
    return specials(x, log_reduced(x, _mm256_setzero_pd()), -inf, inf, 0);
}
RAPIDGLM_VMATH_FAST_TARGET __m256d log1p(__m256d x) {
    const __m256d one = set(1), u = _mm256_add_pd(one, x);
    const __m256d c =
        select(_mm256_cmp_pd(x, one, _CMP_GT_OQ), _mm256_sub_pd(one, _mm256_sub_pd(u, x)),
               _mm256_sub_pd(x, _mm256_sub_pd(u, one)));
    const __m256d value = log_reduced(u, _mm256_div_pd(c, u));
    return select(eq(x, _mm256_setzero_pd()), x, specials(x, value, -inf, inf, -1));
}
RAPIDGLM_VMATH_FAST_TARGET __m256d lgamma(__m256d x) {
    const __m256d zero = _mm256_setzero_pd(), one = set(1);
    const __m256d large = ge(x, set(lgamma_series)), below = lt(x, set(0.5)),
                  above = _mm256_andnot_pd(large, ge(x, set(3)));
    const __m256d base =
        _mm256_andnot_pd(_mm256_or_pd(large, _mm256_or_pd(below, above)), lt(x, set(3)));
    // Per lane: the log argument w (or base's f, d, k directly), the correction
    // num/den, and the quotient 1/x (large) or Gamma(x) = p/recip (above).
    __m256d w = x, numerator = one, denominator = x, num = zero, den = one;
    __m256d fb = zero, db = set(2), kb = zero, eb = zero;
    if (_mm256_movemask_pd(large) != 0xf) {
        const __m256d n = _mm256_and_pd(above, _mm256_sub_pd(_mm256_floor_pd(x), set(2)));
        const __m256d z = _mm256_sub_pd(x, n);
        const __m256d a1 = select(below, x, _mm256_sub_pd(z, one)),
                      a2 = select(below, _mm256_sub_pd(x, one), _mm256_sub_pd(z, set(2)));
        const __m256d t = select(below, _mm256_sub_pd(x, set(1.375)), _mm256_sub_pd(z, set(2.375)));
        const __m256d q = horner(lgamma_base, t, 1);
        const __m256d c0 = set(lgamma_base[0]), kh = _mm256_fmadd_pd(t, q, c0);
        const __m256d kl =
            _mm256_add_pd(_mm256_fmadd_pd(t, q, _mm256_sub_pd(c0, kh)), set(lgamma_base_low));
        const __m256d a = _mm256_mul_pd(a1, a2), al = _mm256_fmsub_pd(a1, a2, a);
        const __m256d uh = _mm256_mul_pd(a, kh);
        const __m256d ul =
            _mm256_fmadd_pd(al, kh, _mm256_fmadd_pd(a, kl, _mm256_fmsub_pd(a, kh, uh)));
        if (any(base)) {
            const __m256d low = lt(uh, set(sqrt_half_minus_one));
            fb = select(low, _mm256_fmadd_pd(set(2), uh, one), uh);
            db = select(low, _mm256_fmadd_pd(set(2), uh, set(3)), _mm256_fmadd_pd(a, kh, set(2)));
            kb = _mm256_and_pd(low, set(-1));
            const __m256d e = select(low, _mm256_mul_pd(set(2), ul), ul),
                          g = _mm256_fmadd_pd(fb, _mm256_sub_pd(fb, one), one);
            eb = _mm256_mul_pd(e, g);
        }
        const __m256d recip = _mm256_fmadd_pd(a, kh, one);
        const __m256d c = _mm256_add_pd(_mm256_fmadd_pd(a, kh, _mm256_sub_pd(one, recip)), ul);
        if (any(below)) {
            const __m256d wb = _mm256_mul_pd(x, recip);
            w = select(below, wb, w);
            num = select(below, _mm256_fmadd_pd(x, c, _mm256_fmsub_pd(x, recip, wb)), num);
            den = select(below, wb, den);
        }
        if (any(above)) {
            __m256d p = _mm256_sub_pd(x, one), pl = zero;
            for (double j = 2; j <= 3; ++j) {
                const __m256d factor =
                    select(_mm256_cmp_pd(set(j), n, _CMP_LE_OQ), _mm256_sub_pd(x, set(j)), one);
                const __m256d next = _mm256_mul_pd(p, factor);
                pl = _mm256_fmadd_pd(pl, factor, _mm256_fmsub_pd(p, factor, next));
                p = next;
            }
            numerator = select(above, p, numerator);
            denominator = select(above, recip, denominator);
            den = select(above, p, den);
            num = select(above, pl, num);  // completed below, once the quotient is known
            fb = select(above, recip, fb); // (recip and c kept for that)
            eb = select(above, c, eb);
        }
    }
    const __m256d quotient = _mm256_div_pd(numerator, denominator); // 1/x or Gamma(x)
    w = select(above, quotient, w);
    if (any(above))
        num = select(
            above,
            _mm256_add_pd(_mm256_fnmadd_pd(quotient, eb, _mm256_fnmadd_pd(quotient, fb, numerator)),
                          num),
            num);
    __m256d f, k;
    reduce(w, f, k);
    __m256d d = _mm256_add_pd(set(2), f), extra = zero;
    const __m256d corrected = _mm256_or_pd(below, above);
    if (any(corrected))
        extra = _mm256_and_pd(corrected, _mm256_div_pd(num, den));
    if (any(base)) {
        f = select(base, fb, f);
        d = select(base, db, d);
        k = select(base, kb, k);
        extra = select(base, eb, extra);
    }
    const __m256d l = log_kernel(f, d, k, extra);
    __m256d value = select(above, l, _mm256_sub_pd(zero, l));
    if (any(large)) {
        const __m256d r = quotient, p = horner(lgamma_large, _mm256_mul_pd(r, r));
        value = select(large,
                       _mm256_fmadd_pd(_mm256_sub_pd(x, set(0.5)), _mm256_sub_pd(l, one),
                                       _mm256_fmadd_pd(r, p, set(half_log_two_pi_minus_half))),
                       value);
    }
    return specials(x, value, inf, inf, 0);
}
template <__m256d (*F)(__m256d)>
RAPIDGLM_VMATH_FAST_TARGET void apply(const double *x, double *out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, F(_mm256_loadu_pd(x + i)));
    if (i < n) {
        const std::size_t lanes = n - i;
        const __m256i mask = _mm256_setr_epi64x(-1, lanes > 1 ? -1 : 0, lanes > 2 ? -1 : 0, 0);
        _mm256_maskstore_pd(out + i, mask, F(_mm256_maskload_pd(x + i, mask)));
    }
}
#undef RAPIDGLM_VMATH_FAST_TARGET
} // namespace detail::avx2
#endif

} // namespace own
namespace boost_algorithm {
// (C) Copyright John Maddock 2006.
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// trigamma.
inline constexpr double tg_offset = 2.1093254089355469;
inline constexpr double tg_P12[6] = {-1.1093280605946045, -3.8310674472619321, -3.3703848401898283,
                                     0.28080574467981213, 1.6638069578676164,  0.64468386819102836};
inline constexpr double tg_Q12[6] = {1.0,
                                     3.4535389668541151,
                                     4.5208926987851437,
                                     2.7012734178351534,
                                     0.64468798399785611,
                                     -0.20314516859987728e-6};
inline constexpr double tg_P24[6] = {-0.13803835004508849e-7, 0.50000049158540261,
                                     1.6077979838469348,      2.5645435828098254,
                                     2.0534873203680393,      0.74566981111565923};
inline constexpr double tg_Q24[6] = {1.0,
                                     2.8822787662376169,
                                     4.1681660554090917,
                                     2.7853527819234466,
                                     0.74967671848044792,
                                     -0.00057069112416246805};
inline constexpr double tg_P4[7] = {
    0.68947581948701249e-17, 0.49999999999998975, 1.0177274392923795, 2.498208511343429,
    2.1921221359427595,      1.5897035272532764,  0.40154388356961734};
inline constexpr double tg_Q4[7] = {1.0,
                                    1.7021215452463932,
                                    4.4290431747556469,
                                    2.9745631894384922,
                                    2.3013614809773616,
                                    0.28360399799075752,
                                    0.022892987908906897};

namespace vd = rapidglm::vmath::detail;
// digamma, x >= 10: on x - 1, log + 1/(2x) - z P(z), z = 1/x^2.
inline constexpr double dg_large[8] = {0.083333333333333333333333333333333333333333333333333,
                                       -0.0083333333333333333333333333333333333333333333333333,
                                       0.003968253968253968253968253968253968253968253968254,
                                       -0.0041666666666666666666666666666666666666666666666667,
                                       0.0075757575757575757575757575757575757575757575757576,
                                       -0.021092796092796092796092796092796092796092796092796,
                                       0.083333333333333333333333333333333333333333333333333,
                                       -0.44325980392156862745098039215686274509803921568627};
// digamma on [1, 2]: (x - root)(Y + P(x - 1)/Q(x - 1)).
inline constexpr double dg_Y = 0.99558162689208984;
inline constexpr double dg_root1 = 1569415565.0 / 1073741824.0,
                        dg_root2 = (381566830.0 / 1073741824.0) / 1073741824.0,
                        dg_root3 = 0.9016312093258695918615325266959189453125e-19;
inline constexpr double dg_P[6] = {0.25479851061131551,   -0.32555031186804491,
                                   -0.65031853770896507,  -0.28919126444774784,
                                   -0.045251321448739056, -0.0020713321167745952};
inline constexpr double dg_Q[7] = {1.0,
                                   2.0767117023730469,
                                   1.4606242909763515,
                                   0.43593529692665969,
                                   0.054151797245674225,
                                   0.0021284987017821144,
                                   -0.55789841321675513e-6};
constexpr double inf = vd::inf;

// ---- scalar (the operations of the vector code, lane by lane) ----
template <bool Combined> RAPIDGLM_VMATH_KERNEL double digamma(double x) {
    if (!(x > 0) || x == inf)
        return x != x ? x : x == 0 ? -inf : x > 0 ? x : vd::nan;
    if (x >= 10) {
        const double xm = x - 1, r = 1 / xm, rg = std::max(r, 0x1p-500), z = rg * rg;
        const double s = std::fma(0.5, r, vd::log(xm));
        return std::fma(-z, vd::horner(dg_large, z), s);
    }
    // Reduce to z in [1, 2]: x < 1 one step up, x > 2 n = ceil(x - 2) steps down.
    const bool below = x < 1, above = x > 2;
    const double n = above ? std::ceil(x - 2) : 0, z = below ? x + 1 : x - n;
    const double g = ((z - dg_root1) - dg_root2) - dg_root3, t = z - 1;
    const double r = vd::horner(dg_P, t) / vd::horner(dg_Q, t);
    const double psi = std::fma(g, dg_Y, g * r);
    if (below)
        return psi - 1 / x;
    if (!above)
        return psi;
    if (Combined) {
        // sum_{k < n} 1/(z + k) = N/D: N <- N a + D, D <- D a over a = z + k.
        double num = 0, den = 1;
        for (double k = 0; k < n; ++k) {
            const double a = z + k;
            num = std::fma(num, a, den);
            den = den * a;
        }
        return psi + num / den;
    }
    double sum = 0;
    for (double k = n; k-- > 0;) // as Boost: the smallest terms first
        sum += 1 / (z + k);
    return psi + sum;
}
#ifdef RAPIDGLM_VMATH_AVX2
#define RAPIDGLM_VMATH_BOOST_TARGET                                                                \
    __attribute__((target("avx2,fma"))) inline // local to this header
namespace avx2 {
using namespace rapidglm::vmath::detail::avx2;
template <std::size_t N> RAPIDGLM_VMATH_BOOST_TARGET __m256d hor(const double (&c)[N], __m256d t) {
    __m256d p = _mm256_set1_pd(c[N - 1]);
    for (std::size_t j = N - 1; j-- > 0;)
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(c[j]));
    return p;
}
template <bool Combined> RAPIDGLM_VMATH_BOOST_TARGET __m256d digamma(__m256d x) {
    const __m256d large = ge(x, set(10)), below = lt(x, set(1)),
                  above = _mm256_andnot_pd(large, _mm256_cmp_pd(x, set(2), _CMP_GT_OQ));
    __m256d value = _mm256_setzero_pd();
    if (_mm256_movemask_pd(large) != 0xf) {
        const __m256d n = _mm256_and_pd(above, _mm256_ceil_pd(_mm256_sub_pd(x, set(2))));
        const __m256d z = select(below, _mm256_add_pd(x, set(1)), _mm256_sub_pd(x, n));
        const __m256d g = _mm256_sub_pd(
            _mm256_sub_pd(_mm256_sub_pd(z, set(dg_root1)), set(dg_root2)), set(dg_root3));
        const __m256d t = _mm256_sub_pd(z, set(1));
        const __m256d r = _mm256_div_pd(hor(dg_P, t), hor(dg_Q, t));
        const __m256d psi = _mm256_fmadd_pd(g, set(dg_Y), _mm256_mul_pd(g, r));
        value = psi;
        if (any(below))
            value = select(below, _mm256_sub_pd(psi, _mm256_div_pd(set(1), x)), value);
        if (any(above)) {
            const double steps = largest(n);
            __m256d add;
            if (Combined) {
                __m256d num = _mm256_setzero_pd(), den = set(1);
                for (double k = 0; k < steps; ++k) {
                    const __m256d a = _mm256_add_pd(z, set(k)), use = lt(set(k), n);
                    num = select(use, _mm256_fmadd_pd(num, a, den), num);
                    den = select(use, _mm256_mul_pd(den, a), den);
                }
                add = _mm256_div_pd(num, den);
            } else {
                add = _mm256_setzero_pd();
                for (double k = steps; k-- > 0;)
                    add = select(
                        lt(set(k), n),
                        _mm256_add_pd(add, _mm256_div_pd(set(1), _mm256_add_pd(z, set(k)))), add);
            }
            value = select(above, _mm256_add_pd(psi, add), value);
        }
    }
    if (any(large)) {
        const __m256d xm = _mm256_sub_pd(select(large, x, set(10)), set(1));
        const __m256d r = _mm256_div_pd(set(1), xm), rg = _mm256_max_pd(r, set(0x1p-500)),
                      z = _mm256_mul_pd(rg, rg);
        const __m256d s = _mm256_fmadd_pd(set(0.5), r, log_reduced<false>(xm, _mm256_setzero_pd()));
        value = select(large, _mm256_fnmadd_pd(z, hor(dg_large, z), s), value);
    }
    return specials(x, value, -inf, inf);
}
} // namespace avx2
#undef RAPIDGLM_VMATH_BOOST_TARGET
#endif
RAPIDGLM_VMATH_KERNEL double trigamma(double x) {
    if (!(x > 0) || x == inf)
        return x != x ? x : x == 0 ? inf : x > 0 ? 0 : vd::nan;
    const bool below = x < 1;
    const double pre = below ? 1 / (x * x) : 0, z = below ? x + 1 : x;
    double value;
    if (z <= 2) {
        const double q = vd::horner(tg_Q12, z);
        value = std::fma(tg_offset, q, vd::horner(tg_P12, z)) / (q * (z * z));
    } else {
        const double y = 1 / z;
        const bool mid = z <= 4;
        const double p = mid ? vd::horner(tg_P24, y) : vd::horner(tg_P4, y);
        const double q = mid ? vd::horner(tg_Q24, y) : vd::horner(tg_Q4, y);
        value = (q + p) / (q * z);
    }
    return pre + value;
}

} // namespace boost_algorithm

RAPIDGLM_VMATH_KERNEL double log1p_kernel(double x) {
    return own::log1p(x);
}
RAPIDGLM_VMATH_KERNEL double lgamma_kernel(double x) {
    // The standalone API also accepts negative arguments, using libm reflection.
    return x < 0 ? detail::lgamma(x) : own::lgamma(x);
}
RAPIDGLM_VMATH_KERNEL double trigamma_kernel(double x) {
    return boost_algorithm::trigamma(x);
}
RAPIDGLM_VMATH_KERNEL double digamma_kernel(double x) {
    return boost_algorithm::digamma<false>(x);
}
#ifdef RAPIDGLM_VMATH_AVX2
#define RAPIDGLM_VMATH_FAST_SCALAR(name)                                                           \
    __attribute__((target("avx2,fma"))) RAPIDGLM_VMATH_INSTANCE double name##_fma(double x) {      \
        return name##_kernel(x);                                                                   \
    }                                                                                              \
    inline double name(double x) {                                                                 \
        return avx2_active() ? name##_fma(x) : name##_kernel(x);                                   \
    }
#else
#define RAPIDGLM_VMATH_FAST_SCALAR(name)                                                           \
    inline double name(double x) {                                                                 \
        return name##_kernel(x);                                                                   \
    }
#endif
RAPIDGLM_VMATH_FAST_SCALAR(log1p)
RAPIDGLM_VMATH_FAST_SCALAR(lgamma)
RAPIDGLM_VMATH_FAST_SCALAR(digamma)
RAPIDGLM_VMATH_FAST_SCALAR(trigamma)
#undef RAPIDGLM_VMATH_FAST_SCALAR
#define RAPIDGLM_VMATH_FAST_PORTABLE(name)                                                         \
    inline void name##_array_portable(const double *x, double *out, std::size_t n) {               \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = name##_kernel(x[i]);                                                          \
    }
RAPIDGLM_VMATH_FAST_PORTABLE(log1p)
RAPIDGLM_VMATH_FAST_PORTABLE(lgamma)
RAPIDGLM_VMATH_FAST_PORTABLE(digamma)
#undef RAPIDGLM_VMATH_FAST_PORTABLE
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void log1p_array_avx2(const double *x, double *out,
                                                                 std::size_t n) {
    own::detail::avx2::apply<own::detail::avx2::log1p>(x, out, n);
}
__attribute__((target("avx2,fma"))) inline void lgamma_array_avx2(const double *x, double *out,
                                                                  std::size_t n) {
    // Save negative arguments before writing: out may alias x.
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d input = _mm256_loadu_pd(x + i);
        const __m256d value = own::detail::avx2::lgamma(input);
        const int negative =
            _mm256_movemask_pd(_mm256_cmp_pd(input, _mm256_setzero_pd(), _CMP_LT_OQ));
        if (!negative) {
            _mm256_storeu_pd(out + i, value);
        } else {
            alignas(32) double original[4], result[4];
            _mm256_store_pd(original, input);
            _mm256_store_pd(result, value);
            for (int lane = 0; lane < 4; ++lane)
                out[i + lane] =
                    negative & (1 << lane) ? detail::lgamma(original[lane]) : result[lane];
        }
    }
    for (; i < n; ++i)
        out[i] = lgamma_kernel(x[i]);
}
__attribute__((target("avx2,fma"))) inline void digamma_array_avx2(const double *x, double *out,
                                                                   std::size_t n) {
    detail::avx2::apply<boost_algorithm::avx2::digamma<false>>(x, out, n);
}
#endif
#define RAPIDGLM_VMATH_FAST_ARRAY(name)                                                            \
    inline void name##_array(const double *x, double *out, std::size_t n) {                        \
        RAPIDGLM_VMATH_FAST_DISPATCH(name) name##_array_portable(x, out, n);                       \
    }
#ifdef RAPIDGLM_VMATH_AVX2
#define RAPIDGLM_VMATH_FAST_DISPATCH(name)                                                         \
    if (avx2_active())                                                                             \
        return name##_array_avx2(x, out, n);
#else
#define RAPIDGLM_VMATH_FAST_DISPATCH(name)
#endif
RAPIDGLM_VMATH_FAST_ARRAY(log1p)
RAPIDGLM_VMATH_FAST_ARRAY(lgamma)
RAPIDGLM_VMATH_FAST_ARRAY(digamma)
#undef RAPIDGLM_VMATH_FAST_DISPATCH
#undef RAPIDGLM_VMATH_FAST_ARRAY
} // namespace fast
inline double log(double x) {
    return slow::log(x);
}
inline void log_array(const double *x, double *out, std::size_t n) {
    return slow::log_array(x, out, n);
}
inline void log_array_portable(const double *x, double *out, std::size_t n) {
    return slow::log_array_portable(x, out, n);
}
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void log_array_avx2(const double *x, double *out,
                                                               std::size_t n) {
    return slow::log_array_avx2(x, out, n);
}
#endif
inline double log1p(double x) {
    if (division_mode() == DivisionMode::fast)
        return fast::log1p(x);
    return slow::log1p(x);
}
inline void log1p_array(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::log1p_array(x, out, n);
    return slow::log1p_array(x, out, n);
}
inline void log1p_array_portable(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::log1p_array_portable(x, out, n);
    return slow::log1p_array_portable(x, out, n);
}
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void log1p_array_avx2(const double *x, double *out,
                                                                 std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::log1p_array_avx2(x, out, n);
    return slow::log1p_array_avx2(x, out, n);
}
#endif
inline double exp(double x) {
    return slow::exp(x);
}
inline void exp_array(const double *x, double *out, std::size_t n) {
    return slow::exp_array(x, out, n);
}
inline void exp_array_portable(const double *x, double *out, std::size_t n) {
    return slow::exp_array_portable(x, out, n);
}
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void exp_array_avx2(const double *x, double *out,
                                                               std::size_t n) {
    return slow::exp_array_avx2(x, out, n);
}
#endif
inline double lgamma(double x) {
    if (division_mode() == DivisionMode::fast)
        return fast::lgamma(x);
    return slow::lgamma(x);
}
inline void lgamma_array(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::lgamma_array(x, out, n);
    return slow::lgamma_array(x, out, n);
}
inline void lgamma_array_portable(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::lgamma_array_portable(x, out, n);
    return slow::lgamma_array_portable(x, out, n);
}
inline double digamma(double x) {
    if (division_mode() == DivisionMode::fast)
        return fast::digamma(x);
    return slow::digamma(x);
}
inline void digamma_array(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::digamma_array(x, out, n);
    return slow::digamma_array(x, out, n);
}
inline void digamma_array_portable(const double *x, double *out, std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::digamma_array_portable(x, out, n);
    return slow::digamma_array_portable(x, out, n);
}
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void digamma_array_avx2(const double *x, double *out,
                                                                   std::size_t n) {
    if (division_mode() == DivisionMode::fast)
        return fast::digamma_array_avx2(x, out, n);
    return slow::digamma_array_avx2(x, out, n);
}
#endif
inline double trigamma(double x) {
    return slow::trigamma(x);
}
inline void trigamma_array(const double *x, double *out, std::size_t n) {
    return slow::trigamma_array(x, out, n);
}
inline void trigamma_array_portable(const double *x, double *out, std::size_t n) {
    return slow::trigamma_array_portable(x, out, n);
}
#ifdef RAPIDGLM_VMATH_AVX2
__attribute__((target("avx2,fma"))) inline void trigamma_array_avx2(const double *x, double *out,
                                                                    std::size_t n) {
    return slow::trigamma_array_avx2(x, out, n);
}
#endif
} // namespace rapidglm::vmath
#undef RAPIDGLM_VMATH_INSTANCE
#undef RAPIDGLM_VMATH_KERNEL
