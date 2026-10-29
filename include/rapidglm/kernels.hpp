#pragma once
#include "resampling.hpp"
#include "vmath.hpp"

// Dense cross products shared by the workflows: c = A'G with u = G'r and
// v = diag(G'WG), in register tiles, with AVX2 fused multiply-adds where the
// processor has them.
namespace rapidglm::detail {
// C = A'B for row-major A (n x p), B (n x m), C (p x m). Pack
// sample panels into six-by-eight register tiles: each B panel is reused by
// every row of C before advancing through B. The 128 x 128 B panel fits in
// L2; the two micro-panels (128 x 6 and 128 x 8) together fit in L1. Each
// accumulator visits the samples in order, including across panel boundaries.
inline void product_tile_generic(const double *a, const double *b, std::size_t n, double *c,
                                 std::size_t stride) {
    double acc[6][8];
    for (std::size_t r = 0; r < 6; ++r)
        std::copy_n(c + r * stride, 8, acc[r]);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t q = 0; q < 8; ++q)
                acc[r][q] += a[i * 6 + r] * b[i * 8 + q];
    for (std::size_t r = 0; r < 6; ++r)
        std::copy_n(acc[r], 8, c + r * stride);
}
#ifdef RAPIDGLM_PHILOX_AVX2
__attribute__((target("avx2,fma"))) inline void
product_tile_avx2(const double *a, const double *b, std::size_t n, double *c, std::size_t stride) {
    // Named accumulators keep GCC from materializing a vector array on the
    // stack inside the sample loop. Twelve sums fit in the sixteen YMM registers.
    __m256d c00 = _mm256_loadu_pd(c + 0 * stride + 0);
    __m256d c01 = _mm256_loadu_pd(c + 0 * stride + 4);
    __m256d c10 = _mm256_loadu_pd(c + 1 * stride + 0);
    __m256d c11 = _mm256_loadu_pd(c + 1 * stride + 4);
    __m256d c20 = _mm256_loadu_pd(c + 2 * stride + 0);
    __m256d c21 = _mm256_loadu_pd(c + 2 * stride + 4);
    __m256d c30 = _mm256_loadu_pd(c + 3 * stride + 0);
    __m256d c31 = _mm256_loadu_pd(c + 3 * stride + 4);
    __m256d c40 = _mm256_loadu_pd(c + 4 * stride + 0);
    __m256d c41 = _mm256_loadu_pd(c + 4 * stride + 4);
    __m256d c50 = _mm256_loadu_pd(c + 5 * stride + 0);
    __m256d c51 = _mm256_loadu_pd(c + 5 * stride + 4);
    for (std::size_t i = 0; i < n; ++i) {
        const __m256d b0 = _mm256_loadu_pd(b + i * 8), b1 = _mm256_loadu_pd(b + i * 8 + 4);
        const __m256d a0 = _mm256_broadcast_sd(a + i * 6 + 0);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        const __m256d a1 = _mm256_broadcast_sd(a + i * 6 + 1);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        const __m256d a2 = _mm256_broadcast_sd(a + i * 6 + 2);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        const __m256d a3 = _mm256_broadcast_sd(a + i * 6 + 3);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        const __m256d a4 = _mm256_broadcast_sd(a + i * 6 + 4);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        const __m256d a5 = _mm256_broadcast_sd(a + i * 6 + 5);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
    }
    _mm256_storeu_pd(c + 0 * stride + 0, c00);
    _mm256_storeu_pd(c + 0 * stride + 4, c01);
    _mm256_storeu_pd(c + 1 * stride + 0, c10);
    _mm256_storeu_pd(c + 1 * stride + 4, c11);
    _mm256_storeu_pd(c + 2 * stride + 0, c20);
    _mm256_storeu_pd(c + 2 * stride + 4, c21);
    _mm256_storeu_pd(c + 3 * stride + 0, c30);
    _mm256_storeu_pd(c + 3 * stride + 4, c31);
    _mm256_storeu_pd(c + 4 * stride + 0, c40);
    _mm256_storeu_pd(c + 4 * stride + 4, c41);
    _mm256_storeu_pd(c + 5 * stride + 0, c50);
    _mm256_storeu_pd(c + 5 * stride + 4, c51);
}
#endif
// Storage belongs to the call, so concurrent gene workers share no packing
// buffer or BLAS lock. Partial tiles are padded, never read past an input.
template <class Tile>
inline void blocked_product_impl(const double *a, const double *b, std::size_t n, std::size_t p,
                                 std::size_t m, double *c, Tile tile) {
    if (p == 0 || m == 0)
        return;
    std::fill_n(c, p * m, 0.0);
    constexpr std::size_t samples = 128, columns = 128;
    const std::size_t row_tiles = (p + 5) / 6;
    std::vector<double> packed_a(row_tiles * samples * 6), packed_b(samples * columns);
    for (std::size_t i0 = 0; i0 < n; i0 += samples) {
        const std::size_t count = std::min(samples, n - i0);
        for (std::size_t r0 = 0; r0 < p; r0 += 6) {
            double *ar = packed_a.data() + (r0 / 6) * count * 6;
            const std::size_t rows = std::min<std::size_t>(6, p - r0);
            for (std::size_t i = 0; i < count; ++i) {
                std::copy_n(a + (i0 + i) * p + r0, rows, ar + i * 6);
                std::fill(ar + i * 6 + rows, ar + (i + 1) * 6, 0.0);
            }
        }
        for (std::size_t q0 = 0; q0 < m; q0 += columns) {
            const std::size_t width = std::min(columns, m - q0);
            for (std::size_t q = 0; q < width; q += 8) {
                double *bq = packed_b.data() + (q / 8) * count * 8;
                const std::size_t lanes = std::min<std::size_t>(8, width - q);
                for (std::size_t i = 0; i < count; ++i) {
                    std::copy_n(b + (i0 + i) * m + q0 + q, lanes, bq + i * 8);
                    std::fill(bq + i * 8 + lanes, bq + (i + 1) * 8, 0.0);
                }
            }
            for (std::size_t r0 = 0; r0 < p; r0 += 6) {
                const double *ar = packed_a.data() + (r0 / 6) * count * 6;
                const std::size_t rows = std::min<std::size_t>(6, p - r0);
                for (std::size_t q = 0; q < width; q += 8) {
                    const double *bq = packed_b.data() + (q / 8) * count * 8;
                    const std::size_t lanes = std::min<std::size_t>(8, width - q);
                    double *out = c + r0 * m + q0 + q;
                    if (rows == 6 && lanes == 8) {
                        tile(ar, bq, count, out, m);
                    } else {
                        double edge[6 * 8] = {};
                        for (std::size_t r = 0; r < rows; ++r)
                            std::copy_n(out + r * m, lanes, edge + r * 8);
                        tile(ar, bq, count, edge, 8);
                        for (std::size_t r = 0; r < rows; ++r)
                            std::copy_n(edge + r * 8, lanes, out + r * m);
                    }
                }
            }
        }
    }
}
inline void blocked_product_generic(const double *a, const double *b, std::size_t n, std::size_t p,
                                    std::size_t m, double *c) {
    blocked_product_impl(a, b, n, p, m, c, product_tile_generic);
}
// c = A'G (p x m, row k at c + k m), u = G'r and v[q] = sum_i w_i G(i, q)^2 for A
// (n x p) and G (n x m), both row-major. Each sum runs over the samples in
// order from zero; tiles of four rows of c by eight variants are held in
// registers through all samples, and the variants of a tile are adjacent in G.
inline void cross_products_generic(const double *a, const double *w, const double *r,
                                   const double *g, std::size_t n, std::size_t p, std::size_t m,
                                   double *c, double *u, double *v) {
    std::size_t q0 = 0;
    for (; q0 + 8 <= m; q0 += 8) {
        std::size_t k0 = 0;
        for (; k0 + 4 <= p; k0 += 4) {
            double acc[4][8] = {};
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0, *ai = a + i * p + k0;
                for (int kk = 0; kk < 4; ++kk)
                    for (int qq = 0; qq < 8; ++qq)
                        acc[kk][qq] += ai[kk] * gi[qq];
            }
            for (int kk = 0; kk < 4; ++kk)
                for (int qq = 0; qq < 8; ++qq)
                    c[(k0 + kk) * m + q0 + qq] = acc[kk][qq];
        }
        for (; k0 < p; ++k0) {
            double acc[8] = {};
            for (std::size_t i = 0; i < n; ++i)
                for (int qq = 0; qq < 8; ++qq)
                    acc[qq] += a[i * p + k0] * g[i * m + q0 + qq];
            for (int qq = 0; qq < 8; ++qq)
                c[k0 * m + q0 + qq] = acc[qq];
        }
        double su[8] = {}, sv[8] = {};
        for (std::size_t i = 0; i < n; ++i) {
            const double *gi = g + i * m + q0;
            for (int qq = 0; qq < 8; ++qq) {
                su[qq] += r[i] * gi[qq];
                sv[qq] += w[i] * gi[qq] * gi[qq];
            }
        }
        for (int qq = 0; qq < 8; ++qq) {
            u[q0 + qq] = su[qq];
            v[q0 + qq] = sv[qq];
        }
    }
    for (; q0 < m; ++q0) {
        for (std::size_t k = 0; k < p; ++k) {
            double sum = 0;
            for (std::size_t i = 0; i < n; ++i)
                sum += a[i * p + k] * g[i * m + q0];
            c[k * m + q0] = sum;
        }
        double su = 0, sv = 0;
        for (std::size_t i = 0; i < n; ++i) {
            su += r[i] * g[i * m + q0];
            sv += w[i] * g[i * m + q0] * g[i * m + q0];
        }
        u[q0] = su;
        v[q0] = sv;
    }
}
#ifdef RAPIDGLM_PHILOX_AVX2
// The same with AVX2 fused multiply-adds (each term rounded once with its sum).
__attribute__((target("avx2,fma"))) inline void
cross_products_avx2(const double *a, const double *w, const double *r, const double *g,
                    std::size_t n, std::size_t p, std::size_t m, double *c, double *u, double *v) {
    const std::size_t tiled = m / 8 * 8;
    for (std::size_t q0 = 0; q0 < tiled; q0 += 8) {
        std::size_t k0 = 0;
        for (; k0 + 4 <= p; k0 += 4) {
            __m256d acc[4][2];
            for (auto &row : acc)
                row[0] = row[1] = _mm256_setzero_pd();
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0, *ai = a + i * p + k0;
                const __m256d g0 = _mm256_loadu_pd(gi), g1 = _mm256_loadu_pd(gi + 4);
                for (int kk = 0; kk < 4; ++kk) {
                    const __m256d b = _mm256_broadcast_sd(ai + kk);
                    acc[kk][0] = _mm256_fmadd_pd(b, g0, acc[kk][0]);
                    acc[kk][1] = _mm256_fmadd_pd(b, g1, acc[kk][1]);
                }
            }
            for (int kk = 0; kk < 4; ++kk) {
                _mm256_storeu_pd(c + (k0 + kk) * m + q0, acc[kk][0]);
                _mm256_storeu_pd(c + (k0 + kk) * m + q0 + 4, acc[kk][1]);
            }
        }
        for (; k0 < p; ++k0) {
            __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0;
                const __m256d b = _mm256_broadcast_sd(a + i * p + k0);
                s0 = _mm256_fmadd_pd(b, _mm256_loadu_pd(gi), s0);
                s1 = _mm256_fmadd_pd(b, _mm256_loadu_pd(gi + 4), s1);
            }
            _mm256_storeu_pd(c + k0 * m + q0, s0);
            _mm256_storeu_pd(c + k0 * m + q0 + 4, s1);
        }
        __m256d u0 = _mm256_setzero_pd(), u1 = _mm256_setzero_pd(), v0 = _mm256_setzero_pd(),
                v1 = _mm256_setzero_pd();
        for (std::size_t i = 0; i < n; ++i) {
            const double *gi = g + i * m + q0;
            const __m256d g0 = _mm256_loadu_pd(gi), g1 = _mm256_loadu_pd(gi + 4);
            const __m256d ri = _mm256_broadcast_sd(r + i), wi = _mm256_broadcast_sd(w + i);
            u0 = _mm256_fmadd_pd(ri, g0, u0);
            u1 = _mm256_fmadd_pd(ri, g1, u1);
            v0 = _mm256_fmadd_pd(_mm256_mul_pd(wi, g0), g0, v0);
            v1 = _mm256_fmadd_pd(_mm256_mul_pd(wi, g1), g1, v1);
        }
        _mm256_storeu_pd(u + q0, u0);
        _mm256_storeu_pd(u + q0 + 4, u1);
        _mm256_storeu_pd(v + q0, v0);
        _mm256_storeu_pd(v + q0 + 4, v1);
    }
    // The last columns, four at a time with masked loads and stores.
    for (std::size_t q0 = tiled; q0 < m; q0 += 4) {
        const std::size_t lanes = std::min<std::size_t>(4, m - q0);
        const __m256i mask = _mm256_setr_epi64x(lanes > 0 ? -1 : 0, lanes > 1 ? -1 : 0,
                                                lanes > 2 ? -1 : 0, lanes > 3 ? -1 : 0);
        std::size_t k0 = 0;
        for (; k0 + 4 <= p; k0 += 4) {
            __m256d acc[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(),
                              _mm256_setzero_pd()};
            for (std::size_t i = 0; i < n; ++i) {
                const __m256d gi = _mm256_maskload_pd(g + i * m + q0, mask);
                const double *ai = a + i * p + k0;
                for (int kk = 0; kk < 4; ++kk)
                    acc[kk] = _mm256_fmadd_pd(_mm256_broadcast_sd(ai + kk), gi, acc[kk]);
            }
            for (int kk = 0; kk < 4; ++kk)
                _mm256_maskstore_pd(c + (k0 + kk) * m + q0, mask, acc[kk]);
        }
        for (; k0 < p; ++k0) {
            __m256d sum = _mm256_setzero_pd();
            for (std::size_t i = 0; i < n; ++i)
                sum = _mm256_fmadd_pd(_mm256_broadcast_sd(a + i * p + k0),
                                      _mm256_maskload_pd(g + i * m + q0, mask), sum);
            _mm256_maskstore_pd(c + k0 * m + q0, mask, sum);
        }
        __m256d su = _mm256_setzero_pd(), sv = _mm256_setzero_pd();
        for (std::size_t i = 0; i < n; ++i) {
            const __m256d gi = _mm256_maskload_pd(g + i * m + q0, mask);
            su = _mm256_fmadd_pd(_mm256_broadcast_sd(r + i), gi, su);
            sv = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_broadcast_sd(w + i), gi), gi, sv);
        }
        _mm256_maskstore_pd(u + q0, mask, su);
        _mm256_maskstore_pd(v + q0, mask, sv);
    }
}
inline bool fma_available() {
    static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return available;
}
#endif
inline void blocked_product(const double *a, const double *b, std::size_t n, std::size_t p,
                            std::size_t m, double *c) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return blocked_product_impl(a, b, n, p, m, c, product_tile_avx2);
#endif
    blocked_product_generic(a, b, n, p, m, c);
}
// C = A'B as blocked_product, for float A and B and a double C, in
// six-by-sixteen register tiles of float sums. Each tile starts from zero in
// every 128-sample panel, visits that panel's samples in order and then adds
// its sums to C in double, so C holds the double sum of the panels' float sums,
// panel by panel in sample order (four float sums per element for 462
// samples). The AVX2 tile rounds each product with its sum (fused
// multiply-add) and the portable tile may not, so the two can differ in the
// last bits of each float sum.
// Not inlined: GCC 12 vectorizes it worse inside fused_maxima_impl (4.7
// rather than 6.6 GFMA/s).
__attribute__((noinline)) inline void product_tile_float_generic(const float *a, const float *b,
                                                                 std::size_t n, double *c,
                                                                 std::size_t stride) {
    // One row at a time: compilers keep its sixteen sums in vector registers
    // (given the whole six-by-sixteen tile, GCC 12 vectorizes across rows and
    // runs ten times slower). Each sum still visits the samples in order.
    for (std::size_t r = 0; r < 6; ++r) {
        float acc[16] = {};
        for (std::size_t i = 0; i < n; ++i) {
            const float ar = a[i * 6 + r];
            for (std::size_t q = 0; q < 16; ++q)
                acc[q] += ar * b[i * 16 + q];
        }
        for (std::size_t q = 0; q < 16; ++q)
            c[r * stride + q] += static_cast<double>(acc[q]);
    }
}
// The float driver packs every sample panel of A once, then completes C one
// block of 128 columns at a time, adding each panel's sums in sample order:
// the block (p x 128 doubles) stays in cache through the panels, where panels
// outermost would carry all of C through memory once per panel. Partial tiles
// are padded as in blocked_product_impl.
template <class Tile>
inline void blocked_product_float_impl(const float *a, const float *b, std::size_t n, std::size_t p,
                                       std::size_t m, double *c, Tile tile) {
    if (p == 0 || m == 0)
        return;
    std::fill_n(c, p * m, 0.0);
    constexpr std::size_t samples = 128, columns = 128, lanes = 16;
    const std::size_t row_tiles = (p + 5) / 6;
    // Panel i0 starts at packed_a + i0 row_tiles 6, its row tiles count x 6 each.
    std::vector<float> packed_a(row_tiles * 6 * n), packed_b(samples * columns);
    for (std::size_t i0 = 0; i0 < n; i0 += samples) {
        const std::size_t count = std::min(samples, n - i0);
        for (std::size_t r0 = 0; r0 < p; r0 += 6) {
            float *ar = packed_a.data() + i0 * row_tiles * 6 + (r0 / 6) * count * 6;
            const std::size_t rows = std::min<std::size_t>(6, p - r0);
            for (std::size_t i = 0; i < count; ++i) {
                std::copy_n(a + (i0 + i) * p + r0, rows, ar + i * 6);
                std::fill(ar + i * 6 + rows, ar + (i + 1) * 6, 0.0f);
            }
        }
    }
    for (std::size_t q0 = 0; q0 < m; q0 += columns) {
        const std::size_t width = std::min(columns, m - q0);
        for (std::size_t i0 = 0; i0 < n; i0 += samples) {
            const std::size_t count = std::min(samples, n - i0);
            for (std::size_t q = 0; q < width; q += lanes) {
                float *bq = packed_b.data() + (q / lanes) * count * lanes;
                const std::size_t used = std::min(lanes, width - q);
                for (std::size_t i = 0; i < count; ++i) {
                    std::copy_n(b + (i0 + i) * m + q0 + q, used, bq + i * lanes);
                    std::fill(bq + i * lanes + used, bq + (i + 1) * lanes, 0.0f);
                }
            }
            for (std::size_t r0 = 0; r0 < p; r0 += 6) {
                const float *ar = packed_a.data() + i0 * row_tiles * 6 + (r0 / 6) * count * 6;
                const std::size_t rows = std::min<std::size_t>(6, p - r0);
                for (std::size_t q = 0; q < width; q += lanes) {
                    const float *bq = packed_b.data() + (q / lanes) * count * lanes;
                    const std::size_t used = std::min(lanes, width - q);
                    double *out = c + r0 * m + q0 + q;
                    if (rows == 6 && used == lanes) {
                        tile(ar, bq, count, out, m);
                    } else {
                        double edge[6 * lanes] = {};
                        for (std::size_t r = 0; r < rows; ++r)
                            std::copy_n(out + r * m, used, edge + r * lanes);
                        tile(ar, bq, count, edge, lanes);
                        for (std::size_t r = 0; r < rows; ++r)
                            std::copy_n(edge + r * lanes, used, out + r * m);
                    }
                }
            }
        }
    }
}
inline void blocked_product_float_generic(const float *a, const float *b, std::size_t n,
                                          std::size_t p, std::size_t m, double *c) {
    blocked_product_float_impl(a, b, n, p, m, c, product_tile_float_generic);
}
#ifdef RAPIDGLM_PHILOX_AVX2
// c[0 .. 7] += the eight float sums of `sum`, converted to double.
__attribute__((target("avx2,fma"))) inline void add_float_sums(double *c, __m256 sum) {
    _mm256_storeu_pd(
        c, _mm256_add_pd(_mm256_loadu_pd(c), _mm256_cvtps_pd(_mm256_castps256_ps128(sum))));
    _mm256_storeu_pd(c + 4, _mm256_add_pd(_mm256_loadu_pd(c + 4),
                                          _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1))));
}
__attribute__((target("avx2,fma"))) inline void product_tile_float_avx2(const float *a,
                                                                        const float *b,
                                                                        std::size_t n, double *c,
                                                                        std::size_t stride) {
    // Twelve named accumulators of eight floats, as in product_tile_avx2.
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps(),
           c11 = _mm256_setzero_ps(), c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(),
           c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c40 = _mm256_setzero_ps(),
           c41 = _mm256_setzero_ps(), c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < n; ++i) {
        const __m256 b0 = _mm256_loadu_ps(b + i * 16), b1 = _mm256_loadu_ps(b + i * 16 + 8);
        const __m256 a0 = _mm256_broadcast_ss(a + i * 6 + 0);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        const __m256 a1 = _mm256_broadcast_ss(a + i * 6 + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        const __m256 a2 = _mm256_broadcast_ss(a + i * 6 + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        const __m256 a3 = _mm256_broadcast_ss(a + i * 6 + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        const __m256 a4 = _mm256_broadcast_ss(a + i * 6 + 4);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        const __m256 a5 = _mm256_broadcast_ss(a + i * 6 + 5);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }
    add_float_sums(c + 0 * stride + 0, c00);
    add_float_sums(c + 0 * stride + 8, c01);
    add_float_sums(c + 1 * stride + 0, c10);
    add_float_sums(c + 1 * stride + 8, c11);
    add_float_sums(c + 2 * stride + 0, c20);
    add_float_sums(c + 2 * stride + 8, c21);
    add_float_sums(c + 3 * stride + 0, c30);
    add_float_sums(c + 3 * stride + 8, c31);
    add_float_sums(c + 4 * stride + 0, c40);
    add_float_sums(c + 4 * stride + 8, c41);
    add_float_sums(c + 5 * stride + 0, c50);
    add_float_sums(c + 5 * stride + 8, c51);
}
inline void blocked_product_float_avx2(const float *a, const float *b, std::size_t n, std::size_t p,
                                       std::size_t m, double *c) {
    blocked_product_float_impl(a, b, n, p, m, c, product_tile_float_avx2);
}
#endif
inline void blocked_product_float(const float *a, const float *b, std::size_t n, std::size_t p,
                                  std::size_t m, double *c) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return blocked_product_float_avx2(a, b, n, p, m, c);
#endif
    blocked_product_float_generic(a, b, n, p, m, c);
}
// The row maxima of |C| D for C = A'B as blocked_product_float computes it and
// D = diag(inverse_info), without storing C. A holds the p rows of C in tiles
// of six, each tile's n samples in order (row r of tile t, sample i at
// a + t n 6 + i 6 + r); B holds the m columns in groups of sixteen, each
// group's n samples in order (column q of group g at b + g n 16 + i 16 + q).
// Rows and columns past the last are zeros, and inverse_info has an entry for
// every column of the last group (NaN past the last column). Each
// six-by-sixteen register tile sums every 128-sample panel from zero in float
// and adds the panel's sums to its double partials in sample order, with the
// tile of blocked_product_float, so the partials are C's elements bit for bit.
// After the last panel, |c| inverse_info of each column is folded into its
// row's running maximum as qtl_finite_maximum does, skipping NaN and infinity.
// maxima[j] is the largest finite value of row j, or NaN when it has none.
//
// The twelve float sums fill twelve of the sixteen YMM registers, so the
// partials (6 x 16 doubles) live in a 768-byte scratch in L1, read and written
// once per panel (tiles small enough to keep double partials in registers,
// two by sixteen or four by eight, reuse each load too little and ran four
// times slower). Variants are taken 128 at a time (eight groups): each row
// tile's six-by-128 partials (6 kB) stay in L1 while it passes over the
// block's packed columns (n x 128 floats, 236 kB at 462 samples), which stay
// in L2 for every row tile; each panel of the row tile (128 x 6 floats) is
// reused from L1 by the eight groups.
inline void pack_rows_float(const float *a, std::size_t n, std::size_t p, float *out) {
    for (std::size_t r0 = 0; r0 < p; r0 += 6) {
        float *tile = out + r0 * n;
        const std::size_t rows = std::min<std::size_t>(6, p - r0);
        for (std::size_t i = 0; i < n; ++i) {
            std::copy_n(a + i * p + r0, rows, tile + i * 6);
            std::fill(tile + i * 6 + rows, tile + (i + 1) * 6, 0.0f);
        }
    }
}
inline void pack_columns_float(const float *b, std::size_t n, std::size_t m, float *out) {
    for (std::size_t q0 = 0; q0 < m; q0 += 16) {
        float *group = out + q0 * n;
        const std::size_t used = std::min<std::size_t>(16, m - q0);
        for (std::size_t i = 0; i < n; ++i) {
            std::copy_n(b + i * m + q0, used, group + i * 16);
            std::fill(group + i * 16 + used, group + (i + 1) * 16, 0.0f);
        }
    }
}
// maxima[0 .. 5] folded with the finite |partial| inverse_info of a tile's rows.
inline void fold_maxima_generic(const double *partial, const double *inverse_info, double *maxima) {
    constexpr double largest = std::numeric_limits<double>::max();
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t q = 0; q < 16; ++q) {
            const double value = std::abs(partial[r * 16 + q]) * inverse_info[q];
            // NaN and infinity fail the second comparison.
            maxima[r] = value > maxima[r] && value <= largest ? value : maxima[r];
        }
}
template <class Tile, class Fold>
inline void fused_maxima_impl(const float *a, const float *b, std::size_t n, std::size_t p,
                              std::size_t m, const double *inverse_info, double *maxima, Tile tile,
                              Fold fold) {
    constexpr std::size_t samples = 128, block = 8;
    const std::size_t tiles = (p + 5) / 6, full = p / 6, groups = (m + 15) / 16;
    // Running maxima from -1 (below any value); the last tile's in `tail`.
    double tail[6];
    std::fill_n(maxima, p, -1.0);
    std::fill_n(tail, 6, -1.0);
    alignas(32) double partial[block][6 * 16];
    for (std::size_t g0 = 0; g0 < groups; g0 += block) {
        const std::size_t width = std::min(block, groups - g0);
        for (std::size_t t = 0; t < tiles; ++t) {
            const float *at = a + t * n * 6;
            double *running = t < full ? maxima + t * 6 : tail;
            for (std::size_t k = 0; k < width; ++k)
                std::fill_n(partial[k], 6 * 16, 0.0);
            for (std::size_t i0 = 0; i0 < n; i0 += samples)
                for (std::size_t k = 0; k < width; ++k)
                    tile(at + i0 * 6, b + (g0 + k) * n * 16 + i0 * 16, std::min(samples, n - i0),
                         partial[k], 16);
            for (std::size_t k = 0; k < width; ++k)
                fold(partial[k], inverse_info + (g0 + k) * 16, running);
        }
    }
    std::copy_n(tail, p - full * 6, maxima + full * 6);
    for (std::size_t j = 0; j < p; ++j)
        maxima[j] = maxima[j] >= 0 ? maxima[j] : nan;
}
inline void fused_maxima_generic(const float *a, const float *b, std::size_t n, std::size_t p,
                                 std::size_t m, const double *inverse_info, double *maxima) {
    fused_maxima_impl(a, b, n, p, m, inverse_info, maxima, product_tile_float_generic,
                      fold_maxima_generic);
}
#ifdef RAPIDGLM_PHILOX_AVX2
__attribute__((target("avx2,fma"))) inline void
fold_maxima_avx2(const double *partial, const double *inverse_info, double *maxima) {
    const __m256d sign = _mm256_set1_pd(-0.0), none = _mm256_set1_pd(-1),
                  largest = _mm256_set1_pd(std::numeric_limits<double>::max());
    const __m256d i0 = _mm256_loadu_pd(inverse_info), i1 = _mm256_loadu_pd(inverse_info + 4),
                  i2 = _mm256_loadu_pd(inverse_info + 8), i3 = _mm256_loadu_pd(inverse_info + 12);
    for (std::size_t r = 0; r < 6; ++r) {
        const double *c = partial + r * 16;
        __m256d v0 = _mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c)), i0);
        __m256d v1 = _mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c + 4)), i1);
        __m256d v2 = _mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c + 8)), i2);
        __m256d v3 = _mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c + 12)), i3);
        v0 = _mm256_blendv_pd(none, v0, _mm256_cmp_pd(v0, largest, _CMP_LE_OQ));
        v1 = _mm256_blendv_pd(none, v1, _mm256_cmp_pd(v1, largest, _CMP_LE_OQ));
        v2 = _mm256_blendv_pd(none, v2, _mm256_cmp_pd(v2, largest, _CMP_LE_OQ));
        v3 = _mm256_blendv_pd(none, v3, _mm256_cmp_pd(v3, largest, _CMP_LE_OQ));
        // Finite values and -1 only: the maximum does not depend on the order.
        __m256d v = _mm256_max_pd(_mm256_max_pd(v0, v1), _mm256_max_pd(v2, v3));
        v = _mm256_max_pd(v, _mm256_permute2f128_pd(v, v, 1));
        v = _mm256_max_pd(v, _mm256_permute_pd(v, 5));
        maxima[r] = std::max(maxima[r], _mm256_cvtsd_f64(v));
    }
}
inline void fused_maxima_avx2(const float *a, const float *b, std::size_t n, std::size_t p,
                              std::size_t m, const double *inverse_info, double *maxima) {
    fused_maxima_impl(a, b, n, p, m, inverse_info, maxima, product_tile_float_avx2,
                      fold_maxima_avx2);
}
#endif
inline void fused_maxima(const float *a, const float *b, std::size_t n, std::size_t p,
                         std::size_t m, const double *inverse_info, double *maxima) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return fused_maxima_avx2(a, b, n, p, m, inverse_info, maxima);
#endif
    fused_maxima_generic(a, b, n, p, m, inverse_info, maxima);
}
inline void cross_products(const double *a, const double *w, const double *r, const double *g,
                           std::size_t n, std::size_t p, std::size_t m, double *c, double *u,
                           double *v) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return cross_products_avx2(a, w, r, g, n, p, m, c, u, v);
#endif
    cross_products_generic(a, w, r, g, n, p, m, c, u, v);
}

// c = A'WA (p x p, row-major) and, when r is not null, u = A'r, for A (n x p,
// row-major) and weights w. Only the tiles of c (four rows by eight columns, as
// in cross_products) that reach the diagonal or above are summed, each term
// a_ik (w_i a_iq) in sample order; the lower triangle is copied from the upper.
inline void weighted_gram_generic(const double *a, const double *w, const double *r, std::size_t n,
                                  std::size_t p, double *c, double *u) {
    for (std::size_t q0 = 0; q0 < p; q0 += 8) {
        const std::size_t width = std::min<std::size_t>(8, p - q0);
        std::size_t k0 = 0;
        for (; k0 + 4 <= p && k0 < q0 + width; k0 += 4) {
            double acc[4][8] = {};
            for (std::size_t i = 0; i < n; ++i) {
                const double *ai = a + i * p;
                double wg[8] = {};
                for (std::size_t qq = 0; qq < width; ++qq)
                    wg[qq] = w[i] * ai[q0 + qq];
                for (int kk = 0; kk < 4; ++kk)
                    for (int qq = 0; qq < 8; ++qq)
                        acc[kk][qq] += ai[k0 + kk] * wg[qq];
            }
            for (int kk = 0; kk < 4; ++kk)
                for (std::size_t qq = 0; qq < width; ++qq)
                    c[(k0 + kk) * p + q0 + qq] = acc[kk][qq];
        }
        for (; k0 < p && k0 < q0 + width; ++k0) {
            double acc[8] = {};
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t qq = 0; qq < width; ++qq)
                    acc[qq] += a[i * p + k0] * (w[i] * a[i * p + q0 + qq]);
            for (std::size_t qq = 0; qq < width; ++qq)
                c[k0 * p + q0 + qq] = acc[qq];
        }
    }
    for (std::size_t k = 0; k < p; ++k)
        for (std::size_t q = k + 1; q < p; ++q)
            c[q * p + k] = c[k * p + q];
    if (!r)
        return;
    for (std::size_t k = 0; k < p; ++k)
        u[k] = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < p; ++k)
            u[k] += a[i * p + k] * r[i];
}
// out = A x for A (n x p, row-major).
inline void matrix_vector_generic(const double *a, const double *x, std::size_t n, std::size_t p,
                                  double *out) {
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (std::size_t k = 0; k < p; ++k)
            sum += a[i * p + k] * x[k];
        out[i] = sum;
    }
}
// u = A'r for A (n x p, row-major).
inline void transposed_product_generic(const double *a, const double *r, std::size_t n,
                                       std::size_t p, double *u) {
    for (std::size_t k = 0; k < p; ++k)
        u[k] = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < p; ++k)
            u[k] += a[i * p + k] * r[i];
}
// h_i = sum_j (sum_{k <= j} b_jk x_ik)^2 for B (p x p, lower triangular,
// row-major) and X given as X' (p x n, row-major): the squared norms of B x_i.
inline void triangular_norms_generic(const double *b, const double *xt, std::size_t n,
                                     std::size_t p, double *h) {
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; ++i)
        h[i] = 0;
    for (std::size_t j = 0; j < p; ++j) {
        std::fill(z.begin(), z.end(), 0.0);
        for (std::size_t k = 0; k <= j; ++k) {
            const double bjk = b[j * p + k], *xk = xt + k * n;
            for (std::size_t i = 0; i < n; ++i)
                z[i] += bjk * xk[i];
        }
        for (std::size_t i = 0; i < n; ++i)
            h[i] += z[i] * z[i];
    }
}
#ifdef RAPIDGLM_PHILOX_AVX2
// The first `lanes` (0 to 4) of four.
__attribute__((target("avx2"))) inline __m256i lane_mask(std::size_t lanes) {
    return _mm256_setr_epi64x(lanes > 0 ? -1 : 0, lanes > 1 ? -1 : 0, lanes > 2 ? -1 : 0,
                              lanes > 3 ? -1 : 0);
}
// Rows k0 .. k0 + 3 of A'WA in the eight columns from q0 (with their part of
// u = A'r when U): 4 x 2 registers held through all samples.
template <bool U>
__attribute__((target("avx2,fma"))) inline void
gram_tile8(const double *a, const double *w, const double *r, std::size_t n, std::size_t p,
           std::size_t k0, std::size_t q0, double *c, double *u) {
    __m256d acc[4][2], u0 = _mm256_setzero_pd(), u1 = _mm256_setzero_pd();
    for (auto &row : acc)
        row[0] = row[1] = _mm256_setzero_pd();
    for (std::size_t i = 0; i < n; ++i) {
        const double *ai = a + i * p;
        const __m256d x0 = _mm256_loadu_pd(ai + q0), x1 = _mm256_loadu_pd(ai + q0 + 4);
        const __m256d wi = _mm256_broadcast_sd(w + i);
        const __m256d g0 = _mm256_mul_pd(wi, x0), g1 = _mm256_mul_pd(wi, x1);
        for (int kk = 0; kk < 4; ++kk) {
            const __m256d b = _mm256_broadcast_sd(ai + k0 + kk);
            acc[kk][0] = _mm256_fmadd_pd(b, g0, acc[kk][0]);
            acc[kk][1] = _mm256_fmadd_pd(b, g1, acc[kk][1]);
        }
        if constexpr (U) {
            const __m256d ri = _mm256_broadcast_sd(r + i);
            u0 = _mm256_fmadd_pd(ri, x0, u0);
            u1 = _mm256_fmadd_pd(ri, x1, u1);
        }
    }
    for (int kk = 0; kk < 4; ++kk) {
        _mm256_storeu_pd(c + (k0 + kk) * p + q0, acc[kk][0]);
        _mm256_storeu_pd(c + (k0 + kk) * p + q0 + 4, acc[kk][1]);
    }
    if constexpr (U) {
        _mm256_storeu_pd(u + q0, u0);
        _mm256_storeu_pd(u + q0 + 4, u1);
    }
}
// The same for `rows` (1 to 4) rows and the `mask`ed columns of four from q0.
template <bool U>
__attribute__((target("avx2,fma"))) inline void
gram_tile4(const double *a, const double *w, const double *r, std::size_t n, std::size_t p,
           std::size_t k0, std::size_t rows, std::size_t q0, __m256i mask, double *c, double *u) {
    __m256d acc[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(),
                      _mm256_setzero_pd()},
            u0 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < n; ++i) {
        const double *ai = a + i * p;
        const __m256d x = _mm256_maskload_pd(ai + q0, mask);
        const __m256d g = _mm256_mul_pd(_mm256_broadcast_sd(w + i), x);
        for (std::size_t kk = 0; kk < 4; ++kk)
            if (kk < rows)
                acc[kk] = _mm256_fmadd_pd(_mm256_broadcast_sd(ai + k0 + kk), g, acc[kk]);
        if constexpr (U)
            u0 = _mm256_fmadd_pd(_mm256_broadcast_sd(r + i), x, u0);
    }
    for (std::size_t kk = 0; kk < rows; ++kk)
        _mm256_maskstore_pd(c + (k0 + kk) * p + q0, mask, acc[kk]);
    if constexpr (U)
        _mm256_maskstore_pd(u + q0, mask, u0);
}
// Rows k0 .. k0 + 7 of the eight columns from q0 when fewer than four rows are
// left: one row at a time.
template <bool U>
__attribute__((target("avx2,fma"))) inline void
gram_row8(const double *a, const double *w, const double *r, std::size_t n, std::size_t p,
          std::size_t k0, std::size_t q0, double *c, double *u) {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd(), u0 = _mm256_setzero_pd(),
            u1 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < n; ++i) {
        const double *ai = a + i * p;
        const __m256d x0 = _mm256_loadu_pd(ai + q0), x1 = _mm256_loadu_pd(ai + q0 + 4);
        const __m256d wi = _mm256_broadcast_sd(w + i), b = _mm256_broadcast_sd(ai + k0);
        s0 = _mm256_fmadd_pd(b, _mm256_mul_pd(wi, x0), s0);
        s1 = _mm256_fmadd_pd(b, _mm256_mul_pd(wi, x1), s1);
        if constexpr (U) {
            const __m256d ri = _mm256_broadcast_sd(r + i);
            u0 = _mm256_fmadd_pd(ri, x0, u0);
            u1 = _mm256_fmadd_pd(ri, x1, u1);
        }
    }
    _mm256_storeu_pd(c + k0 * p + q0, s0);
    _mm256_storeu_pd(c + k0 * p + q0 + 4, s1);
    if constexpr (U) {
        _mm256_storeu_pd(u + q0, u0);
        _mm256_storeu_pd(u + q0 + 4, u1);
    }
}
// The lower triangle (and diagonal) of c is summed, in tiles of four rows by
// eight columns (the last columns four at a time masked), u along with the
// first row tile of each block of columns; the upper triangle is then copied
// from the lower.
__attribute__((target("avx2,fma"))) inline void weighted_gram_avx2(const double *a, const double *w,
                                                                   const double *r, std::size_t n,
                                                                   std::size_t p, double *c,
                                                                   double *u) {
    const std::size_t tiled = p / 8 * 8;
    for (std::size_t q0 = 0; q0 < tiled; q0 += 8) {
        bool first = r != nullptr;
        std::size_t k0 = q0;
        for (; k0 + 4 <= p; k0 += 4, first = false)
            first ? gram_tile8<true>(a, w, r, n, p, k0, q0, c, u)
                  : gram_tile8<false>(a, w, r, n, p, k0, q0, c, u);
        for (; k0 < p; ++k0, first = false)
            first ? gram_row8<true>(a, w, r, n, p, k0, q0, c, u)
                  : gram_row8<false>(a, w, r, n, p, k0, q0, c, u);
    }
    for (std::size_t q0 = tiled; q0 < p; q0 += 4) {
        const __m256i mask = lane_mask(std::min<std::size_t>(4, p - q0));
        bool first = r != nullptr;
        for (std::size_t k0 = q0; k0 < p; k0 += 4, first = false) {
            const std::size_t rows = std::min<std::size_t>(4, p - k0);
            first ? gram_tile4<true>(a, w, r, n, p, k0, rows, q0, mask, c, u)
                  : gram_tile4<false>(a, w, r, n, p, k0, rows, q0, mask, c, u);
        }
    }
    for (std::size_t k = 0; k < p; ++k)
        for (std::size_t q = k + 1; q < p; ++q)
            c[k * p + q] = c[q * p + k];
}
// h for the eight observations from i0 (the `mask`ed ones when Masked), four
// rows of B x_i at once; B's zeros above the diagonal are multiplied through.
template <bool Masked>
__attribute__((target("avx2"))) inline __m256d load_lanes(const double *x, __m256i mask) {
    if constexpr (Masked)
        return _mm256_maskload_pd(x, mask);
    else
        return _mm256_loadu_pd(x);
}
template <bool Masked>
__attribute__((target("avx2,fma"))) inline void
norms_block(const double *b, const double *xt, std::size_t n, std::size_t p, std::size_t i0,
            __m256i m0, __m256i m1, double *h) {
    const auto load = load_lanes<Masked>;
    const std::size_t full_rows = p / 4 * 4;
    __m256d h0 = _mm256_setzero_pd(), h1 = _mm256_setzero_pd();
    for (std::size_t j0 = 0; j0 < full_rows; j0 += 4) {
        __m256d z[4][2];
        for (auto &row : z)
            row[0] = row[1] = _mm256_setzero_pd();
        for (std::size_t k = 0; k < j0 + 4; ++k) {
            const double *xk = xt + k * n + i0;
            const __m256d x0 = load(xk, m0), x1 = load(xk + 4, m1);
            for (int jj = 0; jj < 4; ++jj) {
                const __m256d bjk = _mm256_broadcast_sd(b + (j0 + jj) * p + k);
                z[jj][0] = _mm256_fmadd_pd(bjk, x0, z[jj][0]);
                z[jj][1] = _mm256_fmadd_pd(bjk, x1, z[jj][1]);
            }
        }
        for (int jj = 0; jj < 4; ++jj) {
            h0 = _mm256_fmadd_pd(z[jj][0], z[jj][0], h0);
            h1 = _mm256_fmadd_pd(z[jj][1], z[jj][1], h1);
        }
    }
    for (std::size_t j = full_rows; j < p; ++j) {
        __m256d z0 = _mm256_setzero_pd(), z1 = _mm256_setzero_pd();
        for (std::size_t k = 0; k <= j; ++k) {
            const double *xk = xt + k * n + i0;
            const __m256d bjk = _mm256_broadcast_sd(b + j * p + k);
            z0 = _mm256_fmadd_pd(bjk, load(xk, m0), z0);
            z1 = _mm256_fmadd_pd(bjk, load(xk + 4, m1), z1);
        }
        h0 = _mm256_fmadd_pd(z0, z0, h0);
        h1 = _mm256_fmadd_pd(z1, z1, h1);
    }
    if (Masked) {
        _mm256_maskstore_pd(h + i0, m0, h0);
        _mm256_maskstore_pd(h + i0 + 4, m1, h1);
    } else {
        _mm256_storeu_pd(h + i0, h0);
        _mm256_storeu_pd(h + i0 + 4, h1);
    }
}
__attribute__((target("avx2,fma"))) inline void
triangular_norms_avx2(const double *b, const double *xt, std::size_t n, std::size_t p, double *h) {
    const std::size_t tiled = n / 8 * 8;
    const __m256i all = lane_mask(4);
    for (std::size_t i0 = 0; i0 < tiled; i0 += 8)
        norms_block<false>(b, xt, n, p, i0, all, all, h);
    if (tiled < n) {
        const std::size_t width = n - tiled;
        norms_block<true>(b, xt, n, p, tiled, lane_mask(std::min<std::size_t>(4, width)),
                          lane_mask(width > 4 ? width - 4 : 0), h);
    }
}
// u = A'r with u in registers, sixteen columns at a time (the last four masked),
// even and odd samples summed separately and then added.
__attribute__((target("avx2,fma"))) inline void
transposed_product_avx2(const double *a, const double *r, std::size_t n, std::size_t p, double *u) {
    for (std::size_t k0 = 0; k0 < p; k0 += 16) {
        __m256i mask[4];
        for (std::size_t v = 0; v < 4; ++v)
            mask[v] = lane_mask(k0 + 4 * v < p ? std::min<std::size_t>(4, p - k0 - 4 * v) : 0);
        const std::size_t vectors = std::min<std::size_t>(4, (p - k0 + 3) / 4);
        __m256d even[4], odd[4];
        for (std::size_t v = 0; v < 4; ++v)
            even[v] = odd[v] = _mm256_setzero_pd();
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            const __m256d r0 = _mm256_broadcast_sd(r + i), r1 = _mm256_broadcast_sd(r + i + 1);
            for (std::size_t v = 0; v < 4; ++v)
                if (v < vectors) {
                    even[v] = _mm256_fmadd_pd(
                        r0, _mm256_maskload_pd(a + i * p + k0 + 4 * v, mask[v]), even[v]);
                    odd[v] = _mm256_fmadd_pd(
                        r1, _mm256_maskload_pd(a + (i + 1) * p + k0 + 4 * v, mask[v]), odd[v]);
                }
        }
        if (i < n) {
            const __m256d r0 = _mm256_broadcast_sd(r + i);
            for (std::size_t v = 0; v < vectors; ++v)
                even[v] = _mm256_fmadd_pd(r0, _mm256_maskload_pd(a + i * p + k0 + 4 * v, mask[v]),
                                          even[v]);
        }
        for (std::size_t v = 0; v < vectors; ++v)
            _mm256_maskstore_pd(u + k0 + 4 * v, mask[v], _mm256_add_pd(even[v], odd[v]));
    }
}
// out = A x four rows at a time: each row's products summed in four lanes
// (columns four at a time, the last ones masked), then the lanes added.
__attribute__((target("avx2,fma"))) inline void
matrix_vector_avx2(const double *a, const double *x, std::size_t n, std::size_t p, double *out) {
    const std::size_t full = p / 4 * 4;
    const __m256i last = lane_mask(p - full);
    std::size_t i0 = 0;
    for (; i0 + 4 <= n; i0 += 4) {
        __m256d acc[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(),
                          _mm256_setzero_pd()};
        for (std::size_t k0 = 0; k0 < full; k0 += 4) {
            const __m256d xv = _mm256_loadu_pd(x + k0);
            for (std::size_t rr = 0; rr < 4; ++rr)
                acc[rr] = _mm256_fmadd_pd(_mm256_loadu_pd(a + (i0 + rr) * p + k0), xv, acc[rr]);
        }
        if (full < p) {
            const __m256d xv = _mm256_maskload_pd(x + full, last);
            for (std::size_t rr = 0; rr < 4; ++rr)
                acc[rr] = _mm256_fmadd_pd(_mm256_maskload_pd(a + (i0 + rr) * p + full, last), xv,
                                          acc[rr]);
        }
        const __m256d t0 = _mm256_hadd_pd(acc[0], acc[1]), t1 = _mm256_hadd_pd(acc[2], acc[3]);
        _mm256_storeu_pd(out + i0, _mm256_add_pd(_mm256_permute2f128_pd(t0, t1, 0x20),
                                                 _mm256_permute2f128_pd(t0, t1, 0x31)));
    }
    for (; i0 < n; ++i0) {
        __m256d acc = _mm256_setzero_pd();
        for (std::size_t k0 = 0; k0 < full; k0 += 4)
            acc = _mm256_fmadd_pd(_mm256_loadu_pd(a + i0 * p + k0), _mm256_loadu_pd(x + k0), acc);
        if (full < p)
            acc = _mm256_fmadd_pd(_mm256_maskload_pd(a + i0 * p + full, last),
                                  _mm256_maskload_pd(x + full, last), acc);
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, acc);
        out[i0] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
}
#endif
// out_i = exp(x_i) (vmath::exp_array: the same bits on every processor).
inline void exp_array(const double *x, double *out, std::size_t n) {
    vmath::exp_array(x, out, n);
}
inline void matrix_vector(const double *a, const double *x, std::size_t n, std::size_t p,
                          double *out) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return matrix_vector_avx2(a, x, n, p, out);
#endif
    matrix_vector_generic(a, x, n, p, out);
}
inline void transposed_product(const double *a, const double *r, std::size_t n, std::size_t p,
                               double *u) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return transposed_product_avx2(a, r, n, p, u);
#endif
    transposed_product_generic(a, r, n, p, u);
}
inline void weighted_gram(const double *a, const double *w, const double *r, std::size_t n,
                          std::size_t p, double *c, double *u) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return weighted_gram_avx2(a, w, r, n, p, c, u);
#endif
    weighted_gram_generic(a, w, r, n, p, c, u);
}
inline void triangular_norms(const double *b, const double *xt, std::size_t n, std::size_t p,
                             double *h) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return triangular_norms_avx2(b, xt, n, p, h);
#endif
    triangular_norms_generic(b, xt, n, p, h);
}
} // namespace rapidglm::detail
