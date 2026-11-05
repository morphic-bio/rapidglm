#pragma once
// Derived from Boost.Math, boost/math/special_functions/digamma.hpp and trigamma.hpp:
//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For benchmarks/special_functions.cpp only: boost::math's double-precision (53-bit)
// digamma and trigamma algorithms (boost/math/special_functions/digamma.hpp and
// trigamma.hpp, Boost 1.74, John Maddock; Boost Software License 1.0) transcribed
// for four AVX2 lanes and, with the same operations, one value at a time: the same
// intervals, rational approximations, recurrences and asymptotic series and
// Boost's coefficients, evaluated with explicit fused multiply-adds. The digamma
// recurrence down to [1, 2] is one division per step, as in Boost, or
// (Combined) the sum as one fraction with one division. METHODS.md, Choice of
// backends, compares them with RapidGLM's functions.
#include <rapidglm/vmath.hpp>

namespace boost_polygamma {
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
constexpr double inf = vd::inf;

// ---- scalar (the operations of the vector code, lane by lane) ----
template <bool Combined> inline double digamma(double x) {
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
inline double trigamma(double x) {
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

#ifdef RAPIDGLM_VMATH_AVX2
#define T_ __attribute__((target("avx2,fma"))) inline // local to this header
namespace avx2 {
using namespace rapidglm::vmath::detail::avx2;
template <std::size_t N> T_ __m256d hor(const double (&c)[N], __m256d t) {
    __m256d p = _mm256_set1_pd(c[N - 1]);
    for (std::size_t j = N - 1; j-- > 0;)
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(c[j]));
    return p;
}
template <bool Combined> T_ __m256d digamma(__m256d x) {
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
T_ __m256d trigamma(__m256d x) {
    const __m256d below = lt(x, set(1));
    const __m256d pre = _mm256_and_pd(below, _mm256_div_pd(set(1), _mm256_mul_pd(x, x)));
    const __m256d z = select(below, _mm256_add_pd(x, set(1)), x);
    const __m256d small = _mm256_cmp_pd(z, set(2), _CMP_LE_OQ),
                  mid = _mm256_cmp_pd(z, set(4), _CMP_LE_OQ);
    __m256d value = _mm256_setzero_pd();
    if (any(small)) {
        const __m256d q = hor(tg_Q12, z);
        value = _mm256_div_pd(_mm256_fmadd_pd(set(tg_offset), q, hor(tg_P12, z)),
                              _mm256_mul_pd(q, _mm256_mul_pd(z, z)));
    }
    if (_mm256_movemask_pd(small) != 0xf) {
        const __m256d y = _mm256_div_pd(set(1), z);
        const __m256d mids = _mm256_andnot_pd(small, mid);
        __m256d p, q;
        if (!any(mids)) {
            p = hor(tg_P4, y);
            q = hor(tg_Q4, y);
        } else if (_mm256_movemask_pd(mid) == 0xf) {
            p = hor(tg_P24, y);
            q = hor(tg_Q24, y);
        } else {
            p = select(mids, hor(tg_P24, y), hor(tg_P4, y));
            q = select(mids, hor(tg_Q24, y), hor(tg_Q4, y));
        }
        value = select(small, value, _mm256_div_pd(_mm256_add_pd(q, p), _mm256_mul_pd(q, z)));
    }
    return specials(x, _mm256_add_pd(pre, value), inf, 0);
}
} // namespace avx2
#undef T_
#endif
} // namespace boost_polygamma
