#pragma once
// Scoring many CRT draws against one response at once, as a sparse x dense
// product S = D R: D (draws x cells) holds each draw's treated cells, R (cells x
// (p + 2)) the response's score records (residual, weight, weighted design row;
// BinaryScoreWorkspace), and each row of S holds a draw's sums, from which its
// statistic follows as in BinaryScoreWorkspace::finish.
//
// The cells are visited in blocks, so that a block's records stay in cache while
// every draw adds the records of its cells in that block. Within a block a draw
// adds its records in ascending cell order and the blocks are visited in order,
// so each draw's sums are added in exactly the order of
// BinaryScoreWorkspace::statistics_of, and its statistic is bit for bit the same.
// The AVX2 kernel adds a record's columns as four-lane vectors, which are the
// scalar additions of each column.
#include "resampling.hpp"
#include <cstdint>
#include <vector>

namespace rapidglm::detail {
// One draw to score: its treated cells (ascending, unique, below the number of
// observations) and where its statistic is written.
struct ScoreDraw {
    const std::uint32_t *cells;
    std::uint32_t size;
    double *out;
};
struct BatchScoreScratch {
    std::vector<double, CacheAlignedAllocator<double>> sums;
    std::vector<std::uint32_t> cursor;
    BinaryScoreScratch score;
    Vector cross;
};
// Cells per block. 32,768 records of 64 bytes (six design columns) are 2 MB, in
// the shared cache; each draw's sums (64 bytes) are read and written once per
// block. Measured on the Gasperini screen at 8 threads, in ns per treated cell
// and draw: blocks of 8,192 / 16,384 / 32,768 / 65,536 cells / all cells 4.9 /
// 4.1 / 3.7 / 6.1 / 11.9 (validation/BATCH_SCORING.json).
inline constexpr std::size_t batch_cell_block = 32768;
// Every draw's sums over the records, for any record width.
inline void blocked_sums_generic(const double *records, std::size_t n, std::size_t width,
                                 const ScoreDraw *draws, std::size_t count, double *sums,
                                 std::uint32_t *cursor, std::size_t block) {
    for (std::size_t a = 0; a < n; a += block) {
        const std::size_t b = std::min(n, a + block);
        for (std::size_t d = 0; d < count; ++d) {
            const std::uint32_t *s = draws[d].cells;
            std::uint32_t k = cursor[d];
            const std::uint32_t size = draws[d].size;
            double *sum = sums + d * width;
            for (; k < size && s[k] < b; ++k) {
                const double *record = records + static_cast<std::size_t>(s[k]) * width;
                for (std::size_t j = 0; j < width; ++j)
                    sum[j] += record[j];
            }
            cursor[d] = k;
        }
    }
}
#ifdef RAPIDGLM_PHILOX_AVX2
// Records of eight doubles (six design columns), 64-byte aligned. Two draws are
// summed together while both have cells in the block, so that one draw's
// additions overlap the latency of the other's (one draw at a time measured 4.7
// ns per treated cell instead of 3.7, four at a time 4.0); each draw still adds
// its own records in ascending order.
__attribute__((target("avx2"))) inline void
blocked_sums_avx2_8(const double *records, std::size_t n, const ScoreDraw *draws, std::size_t count,
                    double *sums, std::uint32_t *cursor, std::size_t block) {
    for (std::size_t a = 0; a < n; a += block) {
        const auto b = static_cast<std::uint32_t>(std::min(n, a + block));
        std::size_t d = 0;
        for (; d + 1 < count; d += 2) {
            const std::uint32_t *s0 = draws[d].cells, *s1 = draws[d + 1].cells;
            const std::uint32_t z0 = draws[d].size, z1 = draws[d + 1].size;
            std::uint32_t k0 = cursor[d], k1 = cursor[d + 1];
            if (!(k0 < z0 && s0[k0] < b) && !(k1 < z1 && s1[k1] < b))
                continue;
            double *m0 = sums + d * 8, *m1 = m0 + 8;
            __m256d lo0 = _mm256_load_pd(m0), hi0 = _mm256_load_pd(m0 + 4);
            __m256d lo1 = _mm256_load_pd(m1), hi1 = _mm256_load_pd(m1 + 4);
            while (k0 < z0 && s0[k0] < b && k1 < z1 && s1[k1] < b) {
                const double *r0 = records + static_cast<std::size_t>(s0[k0]) * 8;
                const double *r1 = records + static_cast<std::size_t>(s1[k1]) * 8;
                lo0 = _mm256_add_pd(lo0, _mm256_load_pd(r0));
                hi0 = _mm256_add_pd(hi0, _mm256_load_pd(r0 + 4));
                lo1 = _mm256_add_pd(lo1, _mm256_load_pd(r1));
                hi1 = _mm256_add_pd(hi1, _mm256_load_pd(r1 + 4));
                ++k0;
                ++k1;
            }
            for (; k0 < z0 && s0[k0] < b; ++k0) {
                const double *r0 = records + static_cast<std::size_t>(s0[k0]) * 8;
                lo0 = _mm256_add_pd(lo0, _mm256_load_pd(r0));
                hi0 = _mm256_add_pd(hi0, _mm256_load_pd(r0 + 4));
            }
            for (; k1 < z1 && s1[k1] < b; ++k1) {
                const double *r1 = records + static_cast<std::size_t>(s1[k1]) * 8;
                lo1 = _mm256_add_pd(lo1, _mm256_load_pd(r1));
                hi1 = _mm256_add_pd(hi1, _mm256_load_pd(r1 + 4));
            }
            _mm256_store_pd(m0, lo0);
            _mm256_store_pd(m0 + 4, hi0);
            _mm256_store_pd(m1, lo1);
            _mm256_store_pd(m1 + 4, hi1);
            cursor[d] = k0;
            cursor[d + 1] = k1;
        }
        for (; d < count; ++d) {
            const std::uint32_t *s = draws[d].cells;
            const std::uint32_t z = draws[d].size;
            std::uint32_t k = cursor[d];
            if (!(k < z && s[k] < b))
                continue;
            double *m = sums + d * 8;
            __m256d lo = _mm256_load_pd(m), hi = _mm256_load_pd(m + 4);
            for (; k < z && s[k] < b; ++k) {
                const double *r = records + static_cast<std::size_t>(s[k]) * 8;
                lo = _mm256_add_pd(lo, _mm256_load_pd(r));
                hi = _mm256_add_pd(hi, _mm256_load_pd(r + 4));
            }
            _mm256_store_pd(m, lo);
            _mm256_store_pd(m + 4, hi);
            cursor[d] = k;
        }
    }
}
#endif
// The statistics of `count` draws against the workspace's records, written to
// each draw's `out`: bit for bit those of BinaryScoreWorkspace::statistics_of.
// `vectorized` false takes the generic kernel (for tests).
inline void score_draws(const BinaryScoreWorkspace &w, const ScoreDraw *draws, std::size_t count,
                        BatchScoreScratch &scratch, std::size_t block = batch_cell_block,
                        bool vectorized = true) {
    if (count == 0)
        return;
    require(block > 0, "invalid cell block");
    const std::size_t n = w.observations(), width = w.stride;
    scratch.sums.assign(count * width, 0);
    scratch.cursor.assign(count, 0);
    bool done = false;
#ifdef RAPIDGLM_PHILOX_AVX2
    if (vectorized && width == 8 && philox_avx2_available()) {
        blocked_sums_avx2_8(w.cells.data(), n, draws, count, scratch.sums.data(),
                            scratch.cursor.data(), block);
        done = true;
    }
#else
    (void)vectorized;
#endif
    if (!done)
        blocked_sums_generic(w.cells.data(), n, width, draws, count, scratch.sums.data(),
                             scratch.cursor.data(), block);
    auto &c = scratch.cross;
    c.resize(w.columns);
    for (std::size_t d = 0; d < count; ++d) {
        // Every cell was consumed: the indices were below n (their order and
        // uniqueness are checked when the draws are stored or generated).
        require(scratch.cursor[d] == draws[d].size,
                "sampler indices must be sorted, unique and in range");
        const double *sum = &scratch.sums[d * width];
        std::copy(sum + 2, sum + width, c.begin());
        *draws[d].out = w.statistic_from(w.finish(sum[0], sum[1], c, scratch.score));
    }
}
// Checks once that every stored draw's cells are ascending, unique and below n,
// so that the scoring passes need not check each cell.
inline void check_draws(const CachedDraws &c, std::size_t n) {
    for (std::size_t d = 0; d < c.count; ++d)
        for (std::size_t k = c.offsets[d]; k < c.offsets[d + 1]; ++k)
            require(c.cells[k] < n && (k == c.offsets[d] || c.cells[k - 1] < c.cells[k]),
                    "sampler indices must be sorted, unique and in range");
}
} // namespace rapidglm::detail
