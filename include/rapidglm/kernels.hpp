#pragma once
#include "resampling.hpp"

// Dense cross products shared by the workflows: c = A'G with u = G'r and
// v = diag(G'WG), in register tiles, with AVX2 fused multiply-adds where the
// processor has them.
namespace rapidglm::detail {
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
// out_i = exp(x_i) for x_i in [-700, 700] (or NaN).
inline void exp_array_generic(const double *x, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::exp(x[i]);
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
// exp four at a time: exp(x) = 2^k exp(r) with k = round(x / log 2) and
// r = x - k log 2 (log 2 in two parts, |r| <= 0.35), exp(r) from its Taylor
// polynomial of degree 13 (truncation below 1e-17 relative) and 2^k added to the
// exponent bits. Within about one unit in the last place of std::exp for the
// arguments the fits pass ([-700, 700]; NaN stays NaN).
__attribute__((target("avx2,fma"))) inline __m256d exp_lanes(__m256d x) {
    const __m256d log2e = _mm256_set1_pd(1.4426950408889634),
                  ln2_hi = _mm256_set1_pd(6.93147180559945286227e-01),
                  ln2_lo = _mm256_set1_pd(2.31904681384629955842e-17),
                  shift = _mm256_set1_pd(6755399441055744.0); // 1.5 x 2^52
    const __m256d k =
        _mm256_round_pd(_mm256_mul_pd(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, ln2_hi, x);
    r = _mm256_fnmadd_pd(k, ln2_lo, r);
    static constexpr double c[14] = {1.0,
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
    __m256d poly = _mm256_set1_pd(c[13]);
    for (int j = 12; j >= 0; --j)
        poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(c[j]));
    const __m256i power = _mm256_slli_epi64(_mm256_castpd_si256(_mm256_add_pd(k, shift)), 52);
    const __m256d value = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(poly), power));
    return _mm256_blendv_pd(value, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
}
__attribute__((target("avx2,fma"))) inline void exp_array_avx2(const double *x, double *out,
                                                               std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, exp_lanes(_mm256_loadu_pd(x + i)));
    if (i < n) {
        const __m256i mask = lane_mask(n - i);
        _mm256_maskstore_pd(out + i, mask, exp_lanes(_mm256_maskload_pd(x + i, mask)));
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
inline void exp_array(const double *x, double *out, std::size_t n) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return exp_array_avx2(x, out, n);
#endif
    exp_array_generic(x, out, n);
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
