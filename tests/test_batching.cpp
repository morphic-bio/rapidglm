// Batched CRT draws and scores must reproduce the per-draw computation of
// v0.2.0 exactly. The reference below is that version's generator, thinned
// sampler and binary score, kept verbatim apart from names.
#include <boost/test/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace rapidglm;

namespace {
// v0.2.0 CounterRng: one Philox block per refill.
class ReferenceRng {
    std::uint64_t seed_, block_ = 0;
    std::uint32_t test_, draw_;
    std::array<std::uint32_t, 4> output_{};
    unsigned cursor_ = 4;

  public:
    ReferenceRng(std::uint64_t seed, std::uint64_t test, std::uint64_t draw)
        : seed_(seed), test_(static_cast<std::uint32_t>(test)),
          draw_(static_cast<std::uint32_t>(draw)) {}
    std::uint32_t next32() {
        if (cursor_ == 4) {
            output_ = CounterRng::philox(
                {static_cast<std::uint32_t>(block_), static_cast<std::uint32_t>(block_ >> 32),
                 draw_, test_},
                {static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(seed_ >> 32)});
            ++block_;
            cursor_ = 0;
        }
        return output_[cursor_++];
    }
    std::uint64_t next64() {
        std::uint64_t a = next32();
        return (a << 32) | next32();
    }
    double uniform() {
        return static_cast<double>(next64() >> 11) * 0x1.0p-53;
    }
};

// v0.2.0 thinned CRT draw.
std::vector<std::size_t> reference_draw(const Vector &p, ReferenceRng &rng) {
    struct Level {
        double bound, log_miss;
        std::vector<std::size_t> members;
    };
    std::map<int, Level> levels;
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0) {
            int exponent;
            std::frexp(p[i], &exponent);
            const int b = std::max(0, -exponent);
            auto &level = levels[b];
            level.bound = std::ldexp(1.0, -b);
            level.members.push_back(i);
        }
    std::vector<std::size_t> out;
    for (auto &entry : levels) {
        auto &level = entry.second;
        level.log_miss = std::log1p(-level.bound);
        const auto &members = level.members;
        const auto merged = static_cast<std::ptrdiff_t>(out.size());
        if (level.bound >= 0.25) {
            for (auto i : members)
                if (rng.uniform() < p[i])
                    out.push_back(i);
        } else
            for (std::size_t next = 0;;) {
                const double gap = std::floor(std::log(1 - rng.uniform()) / level.log_miss);
                if (!(gap < static_cast<double>(members.size() - next)))
                    break;
                next += static_cast<std::size_t>(gap);
                const auto i = members[next++];
                if (rng.uniform() * level.bound < p[i])
                    out.push_back(i);
            }
        std::inplace_merge(out.begin(), out.begin() + merged, out.end());
    }
    return out;
}

// v0.2.0 binary score statistic.
double reference_statistic(const ScoreWorkspace &s, const std::vector<std::size_t> &selected) {
    Matrix weighted(s.design.rows, s.design.cols);
    for (std::size_t i = 0; i < s.design.rows; ++i)
        for (std::size_t j = 0; j < s.design.cols; ++j)
            weighted(i, j) = s.design(i, j) * s.weights[i];
    const Vector correction = s.project_nuisance
                                  ? detail::matvec(s.inverse_information, s.nuisance_score)
                                  : Vector(s.design.cols, 0);
    Vector c(s.design.cols, 0);
    double u = 0, v = 0;
    for (auto i : selected) {
        u += s.residuals[i];
        v += s.weights[i];
        for (std::size_t j = 0; j < c.size(); ++j)
            c[j] += weighted(i, j);
    }
    u -= detail::dot(c, correction);
    Vector solved(s.inverse_information.rows, 0);
    for (std::size_t i = 0; i < s.inverse_information.rows; ++i)
        for (std::size_t j = 0; j < s.inverse_information.cols; ++j)
            solved[i] += s.inverse_information(i, j) * c[j];
    double info = v - detail::dot(c, solved);
    if (!(info > 1e-12 * std::max(1.0, v)))
        info = 0;
    return info == 0 ? 0 : u / std::sqrt(info * s.scale);
}

// Probabilities over several thinning levels, including q >= 1/4, exact powers
// of two, zeros and a sparse tail, over n cells (more than one 4096 block).
Vector probabilities(std::size_t n) {
    Vector p(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = std::fmod(0.6180339887 * static_cast<double>(i), 1.0);
        p[i] = i % 97 == 0 ? 0 : i % 211 == 0 ? 0.5 : i % 307 == 0 ? 0.25 : 0.0005 + 0.03 * t * t;
    }
    p[1] = 1;
    return p;
}
} // namespace

BOOST_AUTO_TEST_CASE(counter_rng_stream_is_unchanged) {
    for (std::uint64_t draw : {0u, 1u, 77u}) {
        CounterRng a(20261002, 5, draw);
        ReferenceRng b(20261002, 5, draw);
        for (int k = 0; k < 1000; ++k)
            BOOST_REQUIRE(a.next32() == b.next32());
        for (int k = 0; k < 1000; ++k)
            BOOST_REQUIRE(a.uniform() == b.uniform());
    }
}

BOOST_AUTO_TEST_CASE(batched_crt_draws_equal_per_draw_draws) {
    for (std::size_t n : {500u, 30000u}) {
        const auto p = probabilities(n);
        const auto sampler = crt_sampler(p);
        std::vector<std::vector<std::size_t>> batch;
        SamplerScratch scratch;
        for (auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {0, 300}, {517, 41}}) {
            sampler.batch(91, 7, first, count, batch, scratch);
            for (std::size_t k = 0; k < count; ++k) {
                ReferenceRng reference(91, 7, first + k);
                const auto expected = reference_draw(p, reference);
                BOOST_REQUIRE(batch[k] == expected);
                CounterRng rng(91, 7, first + k);
                std::vector<std::size_t> single;
                sampler(rng, single, scratch);
                BOOST_REQUIRE(single == expected);
            }
        }
    }
    // The hazard line's batches equal its per-draw draws, whatever the batch.
    for (std::size_t n : {500u, 30000u, 70000u}) { // 70,000: several stretches
        const auto sampler = crt_sampler(probabilities(n), CrtDraw::hazard, true);
        std::vector<std::vector<std::size_t>> batch;
        SamplerScratch scratch;
        std::size_t selected = 0;
        for (auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {0, 300}, {517, 41}}) {
            sampler.batch(91, 7, first, count, batch, scratch);
            for (std::size_t k = 0; k < count; ++k) {
                CounterRng rng(91, 7, first + k);
                std::vector<std::size_t> single;
                sampler(rng, single, scratch);
                BOOST_REQUIRE(batch[k] == single);
                selected += single.size();
            }
        }
        BOOST_TEST(selected > 0u);
    }
    // Samplers without a batch form fall back to per-draw calls.
    Vector labels(200);
    for (std::size_t i = 0; i < labels.size(); ++i)
        labels[i] = i % 7 == 0;
    for (const auto &sampler : {permutation_sampler(labels), crt_sampler(Vector(200, 0.3))}) {
        std::vector<std::vector<std::size_t>> batch;
        SamplerScratch scratch;
        sampler.batch(3, 4, 10, 20, batch, scratch);
        for (std::size_t k = 0; k < 20; ++k) {
            CounterRng rng(3, 4, 10 + k);
            std::vector<std::size_t> single;
            sampler(rng, single, scratch);
            BOOST_REQUIRE(batch[k] == single);
        }
    }
}

BOOST_AUTO_TEST_CASE(batched_scores_and_null_draws_equal_v020) {
    const std::size_t n = 30000;
    Matrix x(n, 3, 1);
    Vector y(n), g(n);
    const auto p = probabilities(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::sin(0.013 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 5) / 4;
        y[i] = static_cast<double>((i * 7919) % 11) * (i % 3 == 0);
        g[i] = p[i] > 0.02;
    }
    const auto null = fit_sceptre_null(x, y);
    BOOST_REQUIRE(null.ok());
    const ScoreWorkspace score(x, null);
    const BinaryScoreWorkspace binary(score);
    const auto sampler = crt_sampler(p);

    // Blocked statistics equal one-at-a-time statistics and v0.2.0's.
    std::vector<std::vector<std::size_t>> draws;
    SamplerScratch scratch;
    sampler.batch(11, 2, 0, 64, draws, scratch);
    draws[3].clear(); // an empty selection
    Vector blocked(64), sums;
    std::vector<std::size_t> cursor;
    BinaryScoreScratch score_scratch;
    binary.statistics(draws, 64, blocked.data(), score_scratch, sums, cursor);
    for (std::size_t k = 0; k < 64; ++k) {
        BOOST_REQUIRE(blocked[k] == binary.statistic(draws[k]));
        BOOST_REQUIRE(blocked[k] == reference_statistic(score, draws[k]));
    }
    std::vector<std::vector<std::size_t>> unsorted{{5, 3}};
    BOOST_CHECK_THROW(binary.statistics(unsorted, 1, blocked.data(), score_scratch, sums, cursor),
                      std::invalid_argument);
    std::vector<std::vector<std::size_t>> outside{{n}};
    BOOST_CHECK_THROW(binary.statistics(outside, 1, blocked.data(), score_scratch, sums, cursor),
                      std::invalid_argument);

    // The null law of resample(), serial and with parallel draws, is v0.2.0's.
    ResampleOptions o;
    o.seed = 20261002;
    o.test_id = 9;
    o.budgets = {499, 1100};
    o.approximate = false;
    o.retain_null = true;
    std::vector<std::size_t> observed;
    for (std::size_t i = 0; i < n; ++i)
        if (g[i])
            observed.push_back(i);
    for (int threads : {1, 8}) {
        ResampleResult result;
        auto ro = o;
        ro.parallel_draws = threads > 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#pragma omp single
#endif
        result = resample(score, g, sampler, ro);
        BOOST_REQUIRE(result.valid);
        BOOST_TEST(result.statistic == reference_statistic(score, observed));
        BOOST_REQUIRE(result.null_statistics.size() >= 499u);
        for (std::size_t k = 0; k < result.null_statistics.size(); ++k) {
            ReferenceRng rng(o.seed, o.test_id, k);
            BOOST_REQUIRE(result.null_statistics[k] ==
                          reference_statistic(score, reference_draw(p, rng)));
        }
    }
}

BOOST_AUTO_TEST_CASE(hazard_null_draws_do_not_depend_on_the_schedule) {
    // Each null statistic is that of the draw's own selection, however the
    // draws are batched and scheduled: serially, as tasks, and in the pipeline
    // with a sequential screen and an early stop.
    const std::size_t n = 30000, tests = 40;
    Matrix x(n, 3, 1);
    Vector y(n);
    const auto p = probabilities(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::sin(0.013 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 5) / 4;
        y[i] = static_cast<double>((i * 7919) % 11) * (i % 3 == 0);
    }
    const auto null = fit_sceptre_null(x, y);
    BOOST_REQUIRE(null.ok());
    const ScoreWorkspace score(x, null);
    const BinaryScoreWorkspace binary(score);
    const auto sampler = crt_sampler(p, CrtDraw::hazard, true);
    std::vector<Vector> observed(tests, Vector(n, 0));
    std::vector<std::vector<std::size_t>> selected;
    SamplerScratch scratch;
    for (std::size_t r = 0; r < tests; ++r) {
        sampler.batch(777, r, 0, 1, selected, scratch);
        for (auto i : selected[0])
            observed[r][i] = 1;
    }
    ResampleOptions o;
    o.seed = 20261005;
    o.budgets = {499, 1100};
    o.approximate = false;
    o.retain_null = true;
    for (int threads : {1, 8}) {
        ResampleResult result;
        auto ro = o;
        ro.test_id = 9;
        ro.parallel_draws = threads > 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#pragma omp single
#endif
        result = resample(score, observed[0], sampler, ro);
        BOOST_REQUIRE(result.valid);
        BOOST_REQUIRE(result.null_statistics.size() == 1100u);
        for (std::size_t k = 0; k < result.null_statistics.size(); ++k) {
            CounterRng rng(ro.seed, ro.test_id, k);
            std::vector<std::size_t> single;
            sampler(rng, single, scratch);
            BOOST_REQUIRE(result.null_statistics[k] == binary.statistic(single));
        }
    }
    o.retain_null = false;
    o.budgets = {99, 999};
    o.approximate = true;
    o.screen = 0.2;
    o.sequential_screen = true;
    o.early_stop = 0.1;
    std::vector<ResampleResult> reference(tests);
    std::size_t continued = 0;
    for (std::size_t r = 0; r < tests; ++r) {
        auto ro = o;
        ro.test_id = r;
        reference[r] = resample(score, observed[r], sampler, ro);
        BOOST_REQUIRE(reference[r].valid);
        continued += reference[r].draws > 99;
    }
    BOOST_TEST(continued > 0u);
#if defined(_OPENMP) && _OPENMP >= 201511
    struct Job {
        std::size_t id;
        ResampleOptions options;
        double statistic;
    };
    std::vector<std::size_t> all(tests);
    std::iota(all.begin(), all.end(), 0);
    for (int threads : {2, 8, 16}) {
        std::vector<ResampleResult> results(tests);
        detail::pair_pipeline(
            {all}, threads, true, true, [](std::size_t) { return std::make_unique<int>(0); },
            [&](std::size_t id, int &) {
                auto job = std::make_unique<Job>();
                job->id = id;
                job->options = o;
                job->options.test_id = id;
                job->statistic = reference[id].statistic;
                return job;
            },
            [&](Job &job) -> const detail::PairDraws {
                return {&binary, &sampler, &job.options, job.statistic, {}};
            },
            [&](Job &job, const ResampleResult &result) { results[job.id] = result; });
        for (std::size_t r = 0; r < tests; ++r) {
            BOOST_TEST(results[r].p_value == reference[r].p_value);
            BOOST_TEST(results[r].draws == reference[r].draws);
            BOOST_TEST(results[r].used_approximation == reference[r].used_approximation);
        }
    }
#endif
}

BOOST_AUTO_TEST_CASE(philox_lanes_equal_single_blocks) {
    // Counters that cross a 32-bit boundary inside the eight lanes, too.
    for (std::uint64_t block :
         {std::uint64_t(0), std::uint64_t(13), (std::uint64_t(1) << 32) - 3}) {
        std::uint32_t scalar[32];
        detail::philox_blocks<8>(block, 77, 5, 0x123456789abcdefULL, scalar);
#ifdef RAPIDGLM_PHILOX_AVX2
        std::uint32_t simd[32];
        if (detail::philox_avx2_available()) {
            detail::philox8_avx2(block, 77, 5, 0x123456789abcdefULL, simd);
            BOOST_TEST(std::equal(scalar, scalar + 32, simd));
        }
#endif
        for (unsigned l = 0; l < 8; ++l) {
            const std::uint64_t b = block + l;
            const auto expected = CounterRng::philox(
                {static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(b >> 32), 77, 5},
                {0x89abcdefU, 0x01234567U});
            for (unsigned j = 0; j < 4; ++j)
                BOOST_REQUIRE(scalar[j * 8 + l] == expected[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(pair_pipeline_results_do_not_depend_on_schedule) {
    // 3,000 cells, 9 responses, 7 targets plus non-targeting guides; one target
    // has too few cells and one response is empty, so pairs also end early.
    const std::size_t n = 3000, genes = 9, targets = 8;
    Matrix x(n, 3, 1);
    SparseCSC responses{n, genes, {0}, {}, {}}, assignments{n, targets, {0}, {}, {}};
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::cos(0.7 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 4);
    }
    for (std::size_t j = 0; j < genes; ++j) {
        for (std::size_t i = 0; i < n && j != 4; ++i) {
            const auto v = static_cast<double>((i * (j + 3) * 2654435761u >> 7) % (j + 4) / 2);
            if (v > 0) {
                responses.row_index.push_back(i);
                responses.values.push_back(v);
            }
        }
        responses.column_ptr.push_back(responses.values.size());
    }
    std::vector<Guide> guides;
    for (std::size_t t = 0; t < targets; ++t) {
        for (std::size_t i = t; i < n; i += t == 6 ? 997 : 9 + t)
            if ((i * 7 + t) % 3 != 0) {
                assignments.row_index.push_back(i);
                assignments.values.push_back(1);
            }
        assignments.column_ptr.push_back(assignments.values.size());
        const bool nt = t == targets - 1;
        guides.push_back({"g" + std::to_string(t), nt ? "NT" : "T" + std::to_string(t), nt});
    }
    std::vector<Pair> pairs;
    for (std::size_t j = 0; j < genes; ++j)
        for (std::size_t t = 0; t + 1 < targets; ++t)
            pairs.push_back({j, "T" + std::to_string(t)});
    auto run = [&](bool high_moi, int threads, bool parallel_draws, std::size_t cache) {
        SceptreOptions o;
        o.high_moi = high_moi;
        o.nt_controls = !high_moi;
        o.crt = high_moi;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {99, 999};
        o.threads = threads;
        o.parallel_draws = parallel_draws;
        o.sampler_cache_bytes = cache;
        return sceptre(x, responses, assignments, guides, pairs, {}, o);
    };
    for (bool high_moi : {true, false}) {
        const auto reference = run(high_moi, 1, false, 0);
        std::size_t ok = 0, ended = 0;
        for (const auto &r : reference.tests)
            (r.status == "ok" ? ok : ended) += 1;
        BOOST_TEST(ok > 20u);
        BOOST_TEST(ended > 0u);
        for (auto [threads, draws, cache] : {std::tuple<int, bool, std::size_t>{3, true, 0},
                                             {8, true, std::size_t(1) << 30},
                                             {16, false, std::size_t(1) << 30},
                                             {32, true, 50000}}) {
            const auto other = run(high_moi, threads, draws, cache);
            BOOST_REQUIRE(other.tests.size() == reference.tests.size());
            for (std::size_t k = 0; k < reference.tests.size(); ++k) {
                const auto &a = reference.tests[k], &b = other.tests[k];
                BOOST_TEST(a.status == b.status);
                BOOST_TEST(a.draws == b.draws);
                BOOST_TEST((a.statistic == b.statistic ||
                            (std::isnan(a.statistic) && std::isnan(b.statistic))));
                BOOST_TEST(
                    (a.p_value == b.p_value || (std::isnan(a.p_value) && std::isnan(b.p_value))));
                BOOST_TEST(a.adjusted_p == b.adjusted_p);
                BOOST_TEST(
                    (a.log2fc == b.log2fc || (std::isnan(a.log2fc) && std::isnan(b.log2fc))));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(reused_draws_are_shared_by_a_targets_genes) {
    // Response 5 is a copy of response 2: with reused draws their pairs with a
    // target must agree exactly; with per-pair draws they need not.
    const std::size_t n = 3000, genes = 6, targets = 5;
    Matrix x(n, 3, 1);
    SparseCSC responses{n, genes, {0}, {}, {}}, assignments{n, targets, {0}, {}, {}};
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::cos(0.7 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 4);
    }
    for (std::size_t j = 0; j < genes; ++j) {
        const std::size_t source = j == genes - 1 ? 2 : j;
        for (std::size_t i = 0; i < n; ++i) {
            const auto v =
                static_cast<double>((i * (source + 3) * 2654435761u >> 7) % (source + 4) / 2);
            if (v > 0) {
                responses.row_index.push_back(i);
                responses.values.push_back(v);
            }
        }
        responses.column_ptr.push_back(responses.values.size());
    }
    std::vector<Guide> guides;
    for (std::size_t t = 0; t < targets; ++t) {
        for (std::size_t i = t; i < n; i += 9 + t)
            if ((i * 7 + t) % 3 != 0) {
                assignments.row_index.push_back(i);
                assignments.values.push_back(1);
            }
        assignments.column_ptr.push_back(assignments.values.size());
        guides.push_back({"g" + std::to_string(t), "T" + std::to_string(t), false});
    }
    std::vector<Pair> pairs;
    for (std::size_t j = 0; j < genes; ++j)
        for (std::size_t t = 0; t < targets; ++t)
            pairs.push_back({j, "T" + std::to_string(t)});
    auto run = [&](bool reuse, int threads, std::size_t draw_cache) {
        SceptreOptions o;
        o.high_moi = true;
        o.nt_controls = false;
        o.crt = true;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {199, 999};
        o.resampling.screen = 0.2; // send some pairs to the second stage
        o.threads = threads;
        o.reuse_draws = reuse;
        o.draw_cache_bytes = draw_cache;
        return sceptre(x, responses, assignments, guides, pairs, {}, o);
    };
    const auto shared = run(true, 1, std::size_t(1) << 30);
    const auto fresh = run(false, 1, 0);
    std::size_t ok = 0, differ = 0, second_stage = 0;
    for (std::size_t k = 0; k < shared.tests.size(); ++k) {
        if (shared.tests[k].status != "ok")
            continue;
        ++ok;
        second_stage += shared.tests[k].draws > 199;
    }
    for (std::size_t t = 0; t < targets; ++t) {
        const auto &a = shared.tests[2 * targets + t], &b = shared.tests[(genes - 1) * targets + t];
        BOOST_TEST(a.status == b.status);
        if (a.status != "ok")
            continue;
        BOOST_TEST(a.statistic == b.statistic);
        BOOST_TEST(a.p_value == b.p_value);
        BOOST_TEST(a.draws == b.draws);
        const auto &c = fresh.tests[2 * targets + t], &d = fresh.tests[(genes - 1) * targets + t];
        BOOST_TEST(c.statistic == a.statistic); // the observed statistic is unchanged
        differ += c.p_value != d.p_value;
    }
    BOOST_TEST(ok >= 10u);
    BOOST_TEST(differ > 0u);
    BOOST_TEST(second_stage > 0u);
    // Neither the schedule nor the cache changes the results.
    for (auto [threads, cache] :
         {std::pair<int, std::size_t>{1, 0}, {8, std::size_t(1) << 30}, {32, 0}, {32, 3000}}) {
        const auto other = run(true, threads, cache);
        for (std::size_t k = 0; k < shared.tests.size(); ++k) {
            BOOST_TEST(other.tests[k].status == shared.tests[k].status);
            BOOST_TEST(other.tests[k].draws == shared.tests[k].draws);
            BOOST_TEST(
                (other.tests[k].p_value == shared.tests[k].p_value ||
                 (std::isnan(other.tests[k].p_value) && std::isnan(shared.tests[k].p_value))));
        }
    }
    // Low MOI (per-pair cells) cannot share draws.
    SceptreOptions low;
    low.reuse_draws = true;
    low.crt = true;
    BOOST_CHECK_THROW(sceptre(x, responses, assignments, guides, pairs, {}, low),
                      std::invalid_argument);
}

namespace {
struct ScreenData {
    Matrix x;
    SparseCSC responses, assignments;
    std::vector<Guide> guides;
    std::vector<Pair> pairs;
};
// 2,400 cells, 8 responses, 6 targets plus non-targeting guides and negative
// controls; one target has few cells so its pairs fail QC.
ScreenData screen_data() {
    const std::size_t n = 2400, genes = 8, targets = 7;
    ScreenData d{Matrix(n, 3, 1), {n, genes, {0}, {}, {}}, {n, targets, {0}, {}, {}}, {}, {}};
    for (std::size_t i = 0; i < n; ++i) {
        d.x(i, 1) = std::cos(0.7 * static_cast<double>(i));
        d.x(i, 2) = static_cast<double>(i % 4);
    }
    std::vector<unsigned char> treated_t1(n);
    for (std::size_t t = 0; t < targets; ++t) {
        // Disjoint guide calls (cell i to target i mod 9), as low MOI requires.
        for (std::size_t i = t; i < (t == 5 ? 60 : n); i += 9)
            if ((i / 9 + t) % 3 != 0) {
                d.assignments.row_index.push_back(i);
                d.assignments.values.push_back(1);
                if (t == 1)
                    treated_t1[i] = 1;
            }
        d.assignments.column_ptr.push_back(d.assignments.values.size());
        const bool nt = t == targets - 1;
        d.guides.push_back({"g" + std::to_string(t), nt ? "NT" : "T" + std::to_string(t), nt});
    }
    // Responses 3 and 4 rise in the cells of target T1, so some pairs are discoveries.
    for (std::size_t j = 0; j < genes; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            auto v = static_cast<double>((i * (j + 3) * 2654435761u >> 7) % (j + 4) / 2);
            if ((j == 3 || j == 4) && treated_t1[i])
                v += 3;
            if (v > 0) {
                d.responses.row_index.push_back(i);
                d.responses.values.push_back(v);
            }
        }
        d.responses.column_ptr.push_back(d.responses.values.size());
    }
    for (std::size_t j = 0; j < genes; ++j) {
        for (std::size_t t = 0; t + 1 < targets; ++t)
            d.pairs.push_back({j, "T" + std::to_string(t)});
        d.pairs.push_back({j, "NT", PairKind::negative_control});
    }
    d.pairs.push_back({1, "T0", PairKind::positive_control});
    return d;
}
bool same_record(const PairResult &a, const PairResult &b) {
    const auto same = [](double u, double v) {
        return std::memcmp(&u, &v, sizeof u) == 0 || (std::isnan(u) && std::isnan(v));
    };
    return a.response == b.response && a.target == b.target && a.guide == b.guide &&
           a.status == b.status && a.kind == b.kind && a.cells == b.cells &&
           a.nonzero_treatment == b.nonzero_treatment && a.nonzero_control == b.nonzero_control &&
           a.draws == b.draws && a.approximation == b.approximation &&
           same(a.statistic, b.statistic) && same(a.p_value, b.p_value) &&
           same(a.adjusted_p, b.adjusted_p) && same(a.log2fc, b.log2fc) &&
           same(a.log2fc_se, b.log2fc_se) && same(a.sceptre_mean_ratio, b.sceptre_mean_ratio) &&
           same(a.mean_ratio_log2, b.mean_ratio_log2);
}
} // namespace

BOOST_AUTO_TEST_CASE(sharded_runs_equal_a_single_run) {
    const auto d = screen_data();
    // default, reuse_draws, shared_screen, sequential_screen with early_stop
    for (int mode : {0, 1, 2, 3}) {
        const bool reuse = mode == 1;
        for (bool high_moi : {true, false}) {
            if ((mode == 1 || mode == 2) && !high_moi)
                continue;
            SceptreOptions o;
            o.high_moi = high_moi;
            o.nt_controls = !high_moi;
            o.crt = high_moi;
            o.cell_qc = false;
            o.min_nonzero = 3;
            o.alpha = 0.5; // some deferred effect fits
            o.resampling.budgets = {99, 999};
            o.resampling.screen = 0.2;
            o.reuse_draws = reuse;
            o.shared_screen = mode == 2;
            o.shared_screen_p = 0.4;
            o.resampling.sequential_screen = mode == 3;
            o.resampling.early_stop = mode == 3 ? 0.1 : 0;
            o.threads = 4;
            const auto single = sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
            const auto layout =
                sceptre_layout(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
            BOOST_REQUIRE(layout.task_response.size() == single.tests.size());
            std::vector<FitResult> nulls;
            if (high_moi) {
                std::vector<std::size_t> which(d.responses.cols);
                std::iota(which.begin(), which.end(), 0);
                nulls =
                    sceptre_nulls(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o, which);
                std::size_t compact = 0;
                for (const auto &f : nulls)
                    compact += f.ok() && f.fitted.empty();
                BOOST_TEST(compact > 0u); // rebuilt rather than stored
            }
            // Tiles of 2 targets x 3 responses, in an arbitrary order.
            std::vector<PairResult> records(layout.task_response.size());
            for (std::size_t tb = 0; tb < layout.targets; tb += 2)
                for (std::size_t rb = 0; rb < d.responses.cols; rb += 3) {
                    std::vector<std::size_t> tasks;
                    for (std::size_t i = 0; i < records.size(); ++i)
                        if (layout.task_target[i] / 2 == tb / 2 &&
                            layout.task_response[i] / 3 == rb / 3)
                            tasks.push_back(i);
                    if (tasks.empty())
                        continue;
                    SceptreShard shard;
                    shard.tasks = &tasks;
                    shard.nulls = high_moi ? &nulls : nullptr;
                    auto part =
                        sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o, shard);
                    BOOST_REQUIRE(part.tests.size() == tasks.size());
                    for (std::size_t k = 0; k < tasks.size(); ++k)
                        records[tasks[k]] = part.tests[k];
                }
            SceptreShard finish;
            finish.results = &records;
            finish.nulls = high_moi ? &nulls : nullptr;
            const auto merged =
                sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o, finish);
            BOOST_REQUIRE(merged.tests.size() == single.tests.size());
            std::size_t effects = 0;
            for (std::size_t k = 0; k < single.tests.size(); ++k) {
                BOOST_TEST(same_record(merged.tests[k], single.tests[k]));
                effects += std::isfinite(single.tests[k].log2fc);
            }
            BOOST_TEST(effects > 0u);
            BOOST_TEST(merged.discovery_rejections == single.discovery_rejections);
            BOOST_TEST(merged.calibration_qq.size() == single.calibration_qq.size());
        }
    }
}

BOOST_AUTO_TEST_CASE(shared_screen_keeps_the_default_records_of_pairs_it_passes) {
    const auto d = screen_data();
    auto run = [&](bool shared, int threads, std::size_t draw_cache) {
        SceptreOptions o;
        o.high_moi = true;
        o.nt_controls = false;
        o.crt = true;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {99, 999};
        o.resampling.screen = 0.2; // send some pairs to the second stage
        o.shared_screen = shared;
        o.shared_screen_p = 0.4;
        o.threads = threads;
        o.draw_cache_bytes = draw_cache;
        return sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
    };
    const auto fresh = run(false, 1, 0);
    const auto shared = run(true, 1, std::size_t(1) << 30);
    BOOST_REQUIRE(shared.tests.size() == fresh.tests.size());
    std::size_t same = 0, stopped = 0, second_stage = 0;
    for (std::size_t k = 0; k < fresh.tests.size(); ++k) {
        const auto &a = fresh.tests[k], &b = shared.tests[k];
        BOOST_TEST(a.status == b.status);
        if (a.status != "ok")
            continue;
        BOOST_TEST(a.statistic == b.statistic);
        if (b.p_value == a.p_value && b.draws == a.draws && b.approximation == a.approximation) {
            ++same;
            second_stage += b.draws > 99;
        } else {
            // Only the shared screen ends a pair differently, and only above its threshold.
            BOOST_TEST(b.draws == 99u);
            BOOST_TEST(b.p_value > 0.4);
            BOOST_TEST(!b.approximation);
            ++stopped;
        }
        if (a.p_value <= 0.05) // small p-values are the default's
            BOOST_TEST(b.p_value == a.p_value);
    }
    BOOST_TEST(same > 10u);
    BOOST_TEST(stopped > 0u);
    BOOST_TEST(second_stage > 0u);
    BOOST_TEST(shared.discovery_rejections == fresh.discovery_rejections);
    BOOST_TEST(fresh.discovery_rejections > 0u);
    // Neither the schedule nor the cache changes the results.
    for (auto [threads, cache] :
         {std::pair<int, std::size_t>{1, 0}, {4, std::size_t(1) << 30}, {16, 0}, {32, 3000}}) {
        const auto other = run(true, threads, cache);
        for (std::size_t k = 0; k < shared.tests.size(); ++k)
            BOOST_TEST(same_record(other.tests[k], shared.tests[k]));
    }
    // Shared streams need shared samplers, and the two sharing modes exclude each other.
    SceptreOptions low;
    low.shared_screen = true;
    low.crt = true;
    BOOST_CHECK_THROW(sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, low),
                      std::invalid_argument);
    SceptreOptions both;
    both.high_moi = true;
    both.nt_controls = false;
    both.crt = true;
    both.reuse_draws = true;
    both.shared_screen = true;
    BOOST_CHECK_THROW(sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, both),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sequential_screen_keeps_decisions_and_stays_valid) {
    // Null replicates: each observed indicator is itself a draw from the CRT law,
    // independent of the response.
    const std::size_t n = 2000, replicates = 600;
    Matrix x(n, 3, 1);
    Vector y(n), p(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::sin(0.013 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 5) / 4;
        y[i] = static_cast<double>((i * 7919) % 11) * (i % 3 == 0);
        p[i] = 0.05 + 0.2 * static_cast<double>(i % 7) / 6;
    }
    const auto null = fit_sceptre_null(x, y);
    BOOST_REQUIRE(null.ok());
    const ScoreWorkspace score(x, null);
    const auto sampler = crt_sampler(p);
    ResampleOptions o;
    o.seed = 5;
    o.budgets = {99, 999};
    o.screen = 0.2;
    std::size_t early = 0, continued = 0;
    std::vector<double> sequential_p;
    std::vector<std::vector<std::size_t>> observed;
    SamplerScratch scratch;
    for (std::size_t r = 0; r < replicates; ++r) {
        sampler.batch(777, r, 0, 1, observed, scratch);
        Vector g(n, 0);
        for (auto i : observed[0])
            g[i] = 1;
        auto ro = o;
        ro.test_id = r;
        const auto full = resample(score, g, sampler, ro);
        ro.sequential_screen = true;
        const auto sequential = resample(score, g, sampler, ro);
        BOOST_REQUIRE(full.valid && sequential.valid);
        if (full.draws > 99) {
            // The screen continued: the same result.
            ++continued;
            BOOST_TEST(sequential.p_value == full.p_value);
            BOOST_TEST(sequential.draws == full.draws);
            BOOST_TEST(sequential.used_approximation == full.used_approximation);
        } else {
            // The screen ended: still above the screen, from fewer draws.
            BOOST_TEST(sequential.p_value > o.screen);
            BOOST_TEST(sequential.draws <= 99u);
            BOOST_TEST((sequential.draws % 32 == 0 || sequential.draws == 99));
            early += sequential.draws < 99;
        }
        sequential_p.push_back(sequential.p_value);
    }
    BOOST_TEST(continued > 0u);
    BOOST_TEST(early > replicates / 2);
    // Valid p-values: P(p <= a) <= a, within three binomial standard errors.
    for (double a : {0.3, 0.5, 0.7, 0.9}) {
        const auto below = std::count_if(sequential_p.begin(), sequential_p.end(),
                                         [&](double v) { return v <= a; });
        BOOST_TEST(static_cast<double>(below) / replicates <=
                   a + 3 * std::sqrt(a * (1 - a) / replicates));
    }
}

BOOST_AUTO_TEST_CASE(sequential_screen_in_sceptre_keeps_the_screened_in_records) {
    const auto d = screen_data();
    auto run = [&](bool sequential, bool shared, int threads, bool parallel_draws) {
        SceptreOptions o;
        o.high_moi = true;
        o.nt_controls = false;
        o.crt = true;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {99, 999};
        o.resampling.screen = 0.2;
        o.resampling.sequential_screen = sequential;
        o.shared_screen = shared;
        o.shared_screen_p = 0.4;
        o.threads = threads;
        o.parallel_draws = parallel_draws;
        return sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
    };
    const auto fresh = run(false, false, 1, false);
    for (bool shared : {false, true}) {
        const auto reference = run(true, shared, 1, false);
        std::size_t same = 0, early = 0;
        for (std::size_t k = 0; k < fresh.tests.size(); ++k) {
            const auto &a = fresh.tests[k], &b = reference.tests[k];
            BOOST_TEST(a.status == b.status);
            if (a.status != "ok")
                continue;
            if (b.p_value == a.p_value && b.draws == a.draws &&
                b.approximation == a.approximation) {
                ++same;
                continue;
            }
            // Only pairs whose screens end differ, and they stay above the screens.
            BOOST_TEST(b.p_value > 0.2);
            BOOST_TEST(b.draws <= 99u);
            BOOST_TEST(!b.approximation);
            early += b.draws < 99;
        }
        BOOST_TEST(same > 10u);
        BOOST_TEST(early > 0u);
        BOOST_TEST(reference.discovery_rejections == fresh.discovery_rejections);
        // The same records on any schedule.
        for (auto [threads, parallel_draws] :
             {std::pair<int, bool>{1, true}, {4, true}, {8, false}, {16, true}, {32, true}}) {
            const auto other = run(true, shared, threads, parallel_draws);
            for (std::size_t k = 0; k < reference.tests.size(); ++k)
                BOOST_TEST(same_record(other.tests[k], reference.tests[k]));
        }
    }
}

BOOST_AUTO_TEST_CASE(sequential_screens_draw_only_what_they_use) {
    // Every draw that is generated must be one the result uses: a screen that
    // ends early may not have draws in flight beyond its end, on any schedule.
    const std::size_t n = 2000, tests = 300;
    Matrix x(n, 3, 1);
    Vector y(n), p(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::sin(0.013 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 5) / 4;
        y[i] = static_cast<double>((i * 7919) % 11) * (i % 3 == 0);
        p[i] = 0.05 + 0.2 * static_cast<double>(i % 7) / 6;
    }
    const auto null = fit_sceptre_null(x, y);
    BOOST_REQUIRE(null.ok());
    const ScoreWorkspace score(x, null);
    const BinaryScoreWorkspace binary(score);
    const auto base = crt_sampler(p);
    std::atomic<std::size_t> generated{0};
    const auto sampler = Sampler::with_batch(
        [&](CounterRng &rng, std::vector<std::size_t> &out, SamplerScratch &scratch) {
            ++generated;
            base(rng, out, scratch);
        },
        [&](std::uint64_t seed, std::uint64_t test, std::size_t first, std::size_t count,
            std::vector<std::vector<std::size_t>> &out) {
            generated += count;
            SamplerScratch scratch;
            base.batch(seed, test, first, count, out, scratch);
        });
    std::vector<Vector> observed(tests, Vector(n, 0));
    std::vector<std::vector<std::size_t>> selected;
    SamplerScratch scratch;
    for (std::size_t r = 0; r < tests; ++r) {
        base.batch(777, r, 0, 1, selected, scratch);
        for (auto i : selected[0])
            observed[r][i] = 1;
    }
    // The screen alone, then also the early stop in the budget after it.
    for (double early_stop : {0.0, 0.1}) {
        ResampleOptions o;
        o.seed = 5;
        o.budgets = {99, 999};
        o.screen = 0.2;
        o.sequential_screen = true;
        o.early_stop = early_stop;
        // One caller.
        generated = 0;
        std::vector<ResampleResult> reference(tests);
        std::size_t used = 0, early = 0, stopped = 0;
        for (std::size_t r = 0; r < tests; ++r) {
            auto ro = o;
            ro.test_id = r;
            reference[r] = resample(score, observed[r], sampler, ro);
            BOOST_REQUIRE(reference[r].valid);
            used += reference[r].draws;
            early += reference[r].draws < 99;
            stopped += reference[r].draws > 99 && reference[r].draws < 999;
        }
        BOOST_TEST(early > tests / 2);
        BOOST_TEST((stopped > 0) == (early_stop > 0));
        BOOST_TEST(generated.load() == used);
#if defined(_OPENMP) && _OPENMP >= 201511
        // The pipeline, at several team sizes.
        struct Job {
            std::size_t id;
            ResampleOptions options;
            double statistic;
        };
        std::vector<std::size_t> all(tests);
        std::iota(all.begin(), all.end(), 0);
        for (int threads : {1, 2, 8, 16}) {
            generated = 0;
            std::vector<ResampleResult> results(tests);
            detail::pair_pipeline(
                {all}, threads, true, true, [](std::size_t) { return std::make_unique<int>(0); },
                [&](std::size_t id, int &) {
                    auto job = std::make_unique<Job>();
                    job->id = id;
                    job->options = o;
                    job->options.test_id = id;
                    job->statistic = reference[id].statistic;
                    return job;
                },
                [&](Job &job) -> const detail::PairDraws {
                    return {&binary, &sampler, &job.options, job.statistic, {}};
                },
                [&](Job &job, const ResampleResult &result) { results[job.id] = result; });
            BOOST_TEST(generated.load() == used);
            for (std::size_t r = 0; r < tests; ++r) {
                BOOST_TEST(results[r].p_value == reference[r].p_value);
                BOOST_TEST(results[r].draws == reference[r].draws);
            }
        }
#endif
    }
}

BOOST_AUTO_TEST_CASE(early_stop_keeps_results_below_it_and_stays_valid) {
    // Null replicates as above. The expected result of every replicate is worked
    // out here from the default's draws.
    const std::size_t n = 2000, replicates = 600;
    Matrix x(n, 3, 1);
    Vector y(n), p(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::sin(0.013 * static_cast<double>(i));
        x(i, 2) = static_cast<double>(i % 5) / 4;
        y[i] = static_cast<double>((i * 7919) % 11) * (i % 3 == 0);
        p[i] = 0.05 + 0.2 * static_cast<double>(i % 7) / 6;
    }
    const auto null = fit_sceptre_null(x, y);
    BOOST_REQUIRE(null.ok());
    const ScoreWorkspace score(x, null);
    const auto sampler = crt_sampler(p);
    const std::vector<std::size_t> budgets = {99, 999, 2999};
    const double screen = 0.2, early_stop = 0.1;
    // The count that decides a budget: the smallest whose two-sided empirical
    // p-value over the budget exceeds the threshold, never below the count before.
    const auto decisive = [&](std::size_t budget, double threshold, std::size_t floor) {
        std::size_t c = std::max<std::size_t>(1, floor);
        while (!(std::min(1.0, 2.0 * (1 + c) / (budget + 1)) > threshold))
            ++c;
        return c;
    };
    const std::size_t h1 = decisive(99, screen, 0);
    const std::size_t h[3] = {h1, decisive(999, early_stop, h1),
                              decisive(2999, early_stop, decisive(999, early_stop, h1))};
    BOOST_TEST(h[0] == 10u);
    BOOST_TEST(h[1] == 50u);
    BOOST_TEST(h[2] == 150u);
    std::vector<std::vector<std::size_t>> observed;
    SamplerScratch scratch;
    // With the skew-normal fit a test rarely reaches the last budget; without it
    // every test that is not stopped does, and every p-value is exactly valid.
    for (bool approximate : {true, false}) {
        std::size_t unchanged = 0, stopped[3] = {0, 0, 0}, small = 0;
        std::vector<double> stopped_p;
        for (std::size_t r = 0; r < replicates; ++r) {
            sampler.batch(777, r, 0, 1, observed, scratch);
            Vector g(n, 0);
            for (auto i : observed[0])
                g[i] = 1;
            ResampleOptions o;
            o.seed = 5;
            o.test_id = r;
            o.budgets = budgets;
            o.screen = screen;
            o.approximate = approximate;
            o.retain_null = true;
            const auto full = resample(score, g, sampler, o);
            o.retain_null = false;
            o.early_stop = early_stop;
            const auto stop = resample(score, g, sampler, o);
            o.sequential_screen = true;
            const auto both = resample(score, g, sampler, o);
            o.early_stop = 0;
            const auto sequential = resample(score, g, sampler, o);
            BOOST_REQUIRE(full.valid && stop.valid && both.valid && sequential.valid);
            stopped_p.push_back(stop.p_value);
            // The budget that the early stop ends, if any: the first budget after
            // the screen that the default drew in which the count reaches h.
            const auto &draws = full.null_statistics;
            BOOST_REQUIRE(draws.size() == full.draws);
            std::size_t stage = 0, at = 0, last_side = 0;
            for (std::size_t b = 1; b < budgets.size() && budgets[b] <= full.draws && !stage; ++b) {
                std::size_t below = 0, above = 0, at_below = 0, at_above = 0;
                for (std::size_t k = 0; k < budgets[b] && !at; ++k) {
                    if (draws[k] <= full.statistic && ++below == h[b])
                        at_below = k + 1;
                    if (draws[k] >= full.statistic && ++above == h[b])
                        at_above = k + 1;
                    if (std::min(below, above) >= h[b] &&
                        ((k + 1) % 32 == 0 || k + 1 == budgets[b]))
                        at = k + 1;
                }
                if (at) {
                    stage = b;
                    last_side = std::max(at_below, at_above);
                }
            }
            if (!stage) {
                // Never stopped: the default's result, and with the sequential
                // screen that mode's result.
                ++unchanged;
                small += full.p_value <= early_stop;
                BOOST_TEST(stop.p_value == full.p_value);
                BOOST_TEST(stop.draws == full.draws);
                BOOST_TEST(stop.used_approximation == full.used_approximation);
                BOOST_TEST(both.p_value == sequential.p_value);
                BOOST_TEST(both.draws == sequential.draws);
                continue;
            }
            ++stopped[stage];
            const double expected =
                std::max(std::min(1.0, 2.0 * h[stage] / last_side),
                         std::min(1.0, 2.0 * (1 + h[stage]) / (budgets[stage] + 1)));
            for (const auto *result : {&stop, &both}) {
                BOOST_TEST(result->p_value == expected);
                BOOST_TEST(result->p_value > early_stop);
                BOOST_TEST(result->draws == at);
                BOOST_TEST(!result->used_approximation);
            }
        }
        BOOST_TEST_MESSAGE("early stop, approximate " << approximate << ": unchanged " << unchanged
                                                      << " (at or below the threshold " << small
                                                      << "), stopped in budget 2 " << stopped[1]
                                                      << ", in budget 3 " << stopped[2]);
        BOOST_TEST(unchanged > replicates / 2); // most end at the screen
        BOOST_TEST(small > 0u);
        BOOST_TEST(stopped[1] > 0u);
        if (!approximate) {
            BOOST_TEST(stopped[2] > 0u);
            for (double a : {0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9}) {
                const auto below = std::count_if(stopped_p.begin(), stopped_p.end(),
                                                 [&](double v) { return v <= a; });
                BOOST_TEST(static_cast<double>(below) / replicates <=
                           a + 3 * std::sqrt(a * (1 - a) / replicates));
            }
        }
    }
    // A threshold that the first budget cannot resolve still gives a p-value.
    {
        sampler.batch(777, 0, 0, 1, observed, scratch);
        Vector g(n, 0);
        for (auto i : observed[0])
            g[i] = 1;
        for (Side side : {Side::both, Side::left, Side::right}) {
            ResampleOptions o;
            o.seed = 5;
            o.budgets = {49, 999};
            o.screen = 0.01;
            o.side = side;
            o.sequential_screen = true;
            o.early_stop = 0.0005;
            const auto result = resample(score, g, sampler, o);
            BOOST_REQUIRE(result.valid);
            BOOST_TEST(result.p_value > o.screen);
            BOOST_TEST(result.p_value <= 1.0);
            BOOST_TEST(result.draws <= 49u);
        }
        ResampleOptions bad;
        bad.early_stop = 1;
        BOOST_CHECK_THROW(resample(score, g, sampler, bad), std::invalid_argument);
    }
}

BOOST_AUTO_TEST_CASE(early_stop_in_sceptre_keeps_the_records_below_it) {
    const auto d = screen_data();
    const double early_stop = 0.1;
    auto run = [&](double stop, bool sequential, bool high_moi, int threads, bool parallel_draws) {
        SceptreOptions o;
        o.high_moi = high_moi;
        o.nt_controls = !high_moi;
        o.crt = high_moi;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {99, 999};
        o.resampling.screen = 0.2;
        o.resampling.sequential_screen = sequential;
        o.resampling.early_stop = stop;
        o.threads = threads;
        o.parallel_draws = parallel_draws;
        return sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
    };
    for (bool high_moi : {true, false})
        for (bool sequential : {false, true}) {
            const auto before = run(0, sequential, high_moi, 1, false);
            const auto reference = run(early_stop, sequential, high_moi, 1, false);
            std::size_t same = 0, stopped = 0, second_stage = 0;
            for (std::size_t k = 0; k < before.tests.size(); ++k) {
                const auto &a = before.tests[k], &b = reference.tests[k];
                BOOST_TEST(a.status == b.status);
                if (a.status != "ok")
                    continue;
                if (b.p_value == a.p_value && b.draws == a.draws &&
                    b.approximation == a.approximation) {
                    ++same;
                    second_stage += b.draws == 999;
                    continue;
                }
                // Only pairs past the screen differ, and they end above the threshold.
                BOOST_TEST(a.draws == 999u);
                BOOST_TEST(b.draws > 99u);
                BOOST_TEST(b.draws <= 999u);
                BOOST_TEST(b.p_value > early_stop);
                BOOST_TEST(!b.approximation);
                ++stopped;
            }
            BOOST_TEST_MESSAGE("early stop in sceptre, high MOI "
                               << high_moi << ", sequential screen " << sequential << ": same "
                               << same << " (drew the second budget " << second_stage
                               << "), stopped " << stopped);
            BOOST_TEST(same > 10u);
            BOOST_TEST(stopped > 0u);
            BOOST_TEST(second_stage > 0u); // small p-values keep their records
            // The same records on any schedule.
            for (auto [threads, parallel_draws] :
                 {std::pair<int, bool>{1, true}, {4, true}, {8, false}, {16, true}, {32, true}}) {
                const auto other = run(early_stop, sequential, high_moi, threads, parallel_draws);
                for (std::size_t k = 0; k < reference.tests.size(); ++k)
                    BOOST_TEST(same_record(other.tests[k], reference.tests[k]));
            }
        }
}

BOOST_AUTO_TEST_CASE(sceptre_gives_the_same_records_from_the_count_matrix) {
    // The compact count matrix is another container for the same counts.
    const auto d = screen_data();
    const auto counts = CountMatrix::from(d.responses);
    for (bool high_moi : {true, false})
        for (int threads : {1, 4}) {
            SceptreOptions o;
            o.high_moi = high_moi;
            o.nt_controls = !high_moi;
            o.crt = high_moi;
            o.cell_qc = false;
            o.min_nonzero = 3;
            o.alpha = 0.5;
            o.resampling.budgets = {99, 999};
            o.resampling.screen = 0.2;
            o.threads = threads;
            const auto from_csc =
                sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
            const auto from_counts = sceptre(d.x, counts, d.assignments, d.guides, d.pairs, {}, o);
            BOOST_REQUIRE(from_counts.tests.size() == from_csc.tests.size());
            for (std::size_t k = 0; k < from_csc.tests.size(); ++k)
                BOOST_TEST(same_record(from_counts.tests[k], from_csc.tests[k]));
            BOOST_TEST(from_counts.retained_cells == from_csc.retained_cells);
            BOOST_TEST(from_counts.discovery_rejections == from_csc.discovery_rejections);
        }
    // Cell QC reads every count once; it must see the same totals.
    SceptreOptions qc;
    qc.resampling.budgets = {99, 999};
    qc.min_nonzero = 3;
    const auto a = sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, qc);
    const auto b = sceptre(d.x, counts, d.assignments, d.guides, d.pairs, {}, qc);
    BOOST_TEST(a.retained_cells == b.retained_cells);
    for (std::size_t k = 0; k < a.tests.size(); ++k)
        BOOST_TEST(same_record(a.tests[k], b.tests[k]));
}

BOOST_AUTO_TEST_CASE(sampler_cache_is_counted_and_does_not_change_results) {
    const auto d = screen_data();
    auto run = [&](std::size_t cache, int threads) {
        SceptreOptions o;
        o.high_moi = true;
        o.nt_controls = false;
        o.crt = true;
        o.cell_qc = false;
        o.min_nonzero = 3;
        o.resampling.budgets = {99, 999};
        o.resampling.screen = 0.2;
        o.threads = threads;
        if (cache != std::numeric_limits<std::size_t>::max())
            o.sampler_cache_bytes = cache;
        return sceptre(d.x, d.responses, d.assignments, d.guides, d.pairs, {}, o);
    };
    // By default every target that is tested keeps its sampler.
    const auto all = run(std::numeric_limits<std::size_t>::max(), 4);
    BOOST_TEST(all.samplers_cached >= 5u);
    BOOST_TEST(all.samplers_rebuilt == 0u);
    const std::size_t one = crt_sampler_bytes_per_observation * d.x.rows;
    for (int threads : {1, 4}) {
        const auto some = run(2 * one, threads), none = run(one - 1, threads);
        BOOST_TEST(some.samplers_cached == 2u);
        BOOST_TEST(some.samplers_rebuilt == all.samplers_cached - 2);
        BOOST_TEST(none.samplers_cached == 0u);
        BOOST_TEST(none.samplers_rebuilt == all.samplers_cached);
        for (std::size_t k = 0; k < all.tests.size(); ++k) {
            BOOST_TEST(same_record(some.tests[k], all.tests[k]));
            BOOST_TEST(same_record(none.tests[k], all.tests[k]));
        }
    }
}
