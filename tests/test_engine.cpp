#define BOOST_TEST_MODULE rapidglm
#include <boost/test/included/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
BOOST_AUTO_TEST_CASE(gaussian_analytic) {
    Matrix x(6, 2);
    Vector y = {1, 2, 1, 4, 3, 7};
    for (std::size_t i = 0; i < 6; ++i) {
        x(i, 0) = 1;
        x(i, 1) = i;
    }
    FitOptions o;
    o.family = Family::gaussian;
    o.robust = true;
    auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(f.coefficients[0] == 0.42857142857142855, boost::test_tools::tolerance(1e-10));
    BOOST_TEST(f.coefficients[1] == 1.0285714285714285, boost::test_tools::tolerance(1e-10));
    BOOST_TEST(f.robust_covariance.rows == 2u);
}
BOOST_AUTO_TEST_CASE(poisson_offset_weights_score) {
    Matrix x(5, 1, 1);
    Vector y = {1, 3, 0, 2, 4};
    FitOptions o;
    o.family = Family::poisson;
    o.offset = {0, 0.2, -0.1, 0.3, 0.5};
    o.weights = {1, 2, 1, 0, 1};
    auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    double sy = 0, exposure = 0;
    for (unsigned i = 0; i < 5; ++i) {
        sy += o.weights[i] * y[i];
        exposure += o.weights[i] * std::exp(o.offset[i]);
    }
    BOOST_TEST(f.coefficients[0] == std::log(sy / exposure), boost::test_tools::tolerance(1e-9));
    Vector g = {1, 0, 1, 0, 0};
    ScoreWorkspace s(x, f);
    auto t = s.test(g);
    double u = 0, v = 0, c = 0, h = 0;
    for (unsigned i = 0; i < 5; ++i) {
        u += g[i] * o.weights[i] * (y[i] - f.fitted[i]);
        v += g[i] * o.weights[i] * f.fitted[i];
        c += g[i] * o.weights[i] * f.fitted[i];
        h += o.weights[i] * f.fitted[i];
    }
    BOOST_TEST(t.statistic == u / std::sqrt(v - c * c / h), boost::test_tools::tolerance(1e-9));
}
BOOST_AUTO_TEST_CASE(edge_cases) {
    Matrix x(10, 2, 1);
    FitOptions o;
    o.family = Family::poisson;
    BOOST_CHECK(fit(x, Vector(10, 1), o).status == FitStatus::rank_deficient);
    x = Matrix(10, 1, 1);
    BOOST_CHECK(fit(x, Vector(10), o).status == FitStatus::boundary_response);
    o.family = Family::binomial;
    Matrix sep(10, 2, 1);
    Vector y(10);
    for (unsigned i = 0; i < 10; ++i) {
        sep(i, 1) = i >= 5;
        y[i] = i >= 5;
    }
    BOOST_CHECK(fit(sep, y, o).status == FitStatus::separation);
    o.ridge = 1;
    BOOST_TEST(fit(sep, y, o).ok());
    o.weights = Vector(10, 0);
    BOOST_CHECK_THROW(fit(x, y, o), std::invalid_argument);
    o.weights.clear();
    o.family = Family::poisson;
    y[0] = -1;
    BOOST_CHECK_THROW(fit(x, y, o), std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(nb_fixed_and_ml) {
    Matrix x(12, 1, 1);
    Vector y = {0, 0, 1, 1, 2, 2, 3, 4, 4, 8, 12, 20};
    FitOptions o;
    o.theta = 2;
    o.dispersion = Dispersion::fixed;
    auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    BOOST_TEST(f.coefficients[0] == std::log(mean), boost::test_tools::tolerance(1e-9));
    o.dispersion = Dispersion::ml;
    f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(f.theta > 0);
    BOOST_TEST(f.theta < 2);
    o.dispersion = Dispersion::moment;
    auto moment = fit(x, y, o);
    BOOST_TEST(moment.ok());
}
BOOST_AUTO_TEST_CASE(philox_known_answer) {
    auto v = CounterRng::philox({0, 0, 0, 0}, {0, 0});
    BOOST_TEST(v[0] == 0x6627e8d5U);
    BOOST_TEST(v[1] == 0xe169c58dU);
    BOOST_TEST(v[2] == 0xbc57ac4cU);
    BOOST_TEST(v[3] == 0x9b00dbd8U);
    CounterRng a(1, 2, 3), b(1, 2, 3);
    for (int i = 0; i < 100; ++i)
        BOOST_TEST(a.next64() == b.next64());
    BOOST_CHECK_THROW(CounterRng(1, UINT64_MAX, 0), std::invalid_argument);
    // Compare complete raw streams under different scheduling, before fitting
    // or tail estimation can hide a random-stream discrepancy.
    std::vector<std::uint64_t> baseline(256 * 64);
    for (int threads : {1, 2, 8, 16}) {
        std::vector<std::uint64_t> values(baseline.size());
        parallel_for(256, threads, [&](std::size_t job) {
            CounterRng rng(20260926, job / 16, job % 16);
            for (std::size_t k = 0; k < 64; ++k)
                values[job * 64 + k] = rng.next64();
        });
        if (threads == 1)
            baseline = values;
        BOOST_TEST(values == baseline, boost::test_tools::per_element());
    }
}
BOOST_AUTO_TEST_CASE(sampling_laws) {
    Vector labels = {1, 0, 0, 1, 0, 0};
    auto sampler = permutation_sampler(labels, {0, 0, 0, 1, 1, 1});
    Vector count(6);
    std::vector<std::size_t> selected;
    for (unsigned d = 0; d < 6000; ++d) {
        CounterRng rng(91, 0, d);
        sampler(rng, selected);
        BOOST_REQUIRE(selected.size() == 2);
        BOOST_TEST(selected[0] < 3);
        BOOST_TEST(selected[1] >= 3);
        for (auto i : selected)
            ++count[i];
    }
    for (auto v : count)
        BOOST_CHECK(v > 1800 && v < 2200);
    auto crt = crt_sampler({0, 1, 0.25});
    double hits = 0;
    for (unsigned d = 0; d < 6000; ++d) {
        CounterRng rng(91, 1, d);
        crt(rng, selected);
        BOOST_TEST(selected[0] == 1);
        hits += selected.size() == 2;
    }
    BOOST_CHECK(hits > 1300 && hits < 1700);
}
BOOST_AUTO_TEST_CASE(thinned_crt_draws_match_bernoulli_law) {
    // Probabilities span the direct (q >= 1/4) and geometric-gap levels, exact
    // powers of two, zero and one, and a sparse tail typical of guide calls.
    Vector p = {0, 1, 0.5, 0.25, 0.3, 0.125, 0.1, 0.02, 1.0 / 1024, 0.7, 0.003, 0.05};
    for (std::size_t i = 0; i < 300; ++i)
        p.push_back(0.004 + 0.02 * (i % 17) / 17);
    const std::size_t n = p.size(), draws = 40000;
    auto thinned = crt_sampler(p), direct = crt_sampler(p, CrtDraw::direct);
    Vector hits(n), direct_hits(n), sizes;
    double joint = 0; // cells 4 and 7 (different levels)
    std::vector<std::size_t> selected;
    for (std::size_t d = 0; d < draws; ++d) {
        CounterRng rng(20260929, 3, d);
        thinned(rng, selected);
        BOOST_REQUIRE(std::is_sorted(selected.begin(), selected.end()));
        BOOST_REQUIRE(std::adjacent_find(selected.begin(), selected.end()) == selected.end());
        for (auto i : selected)
            ++hits[i];
        joint += std::binary_search(selected.begin(), selected.end(), 4) &&
                 std::binary_search(selected.begin(), selected.end(), 7);
        sizes.push_back(selected.size());
        CounterRng other(20260929, 4, d);
        direct(other, selected);
        for (auto i : selected)
            ++direct_hits[i];
    }
    BOOST_TEST(hits[0] == 0);
    BOOST_TEST(hits[1] == draws);
    double chi = 0, chi_direct = 0, expected_size = 0, size_variance = 0;
    std::size_t df = 0;
    for (std::size_t i = 0; i < n; ++i) {
        expected_size += p[i];
        size_variance += p[i] * (1 - p[i]);
        if (p[i] > 0 && p[i] < 1) {
            const double v = draws * p[i] * (1 - p[i]);
            chi += std::pow(hits[i] - draws * p[i], 2) / v;
            chi_direct += std::pow(direct_hits[i] - draws * p[i], 2) / v;
            ++df;
        }
    }
    // Upper 0.1% points of chi-square with ~310 df are about 400.
    BOOST_TEST(chi < 400);
    BOOST_TEST(chi_direct < 400);
    const double mean = std::accumulate(sizes.begin(), sizes.end(), 0.0) / draws;
    double variance = 0;
    for (double s : sizes)
        variance += (s - mean) * (s - mean) / (draws - 1);
    BOOST_TEST(std::abs(mean - expected_size) < 5 * std::sqrt(size_variance / draws));
    BOOST_TEST(variance / size_variance > 0.95);
    BOOST_TEST(variance / size_variance < 1.05);
    const double pj = p[4] * p[7];
    BOOST_TEST(std::abs(joint / draws - pj) < 5 * std::sqrt(pj * (1 - pj) / draws));
    // Draws are a pure function of the RNG key.
    std::vector<std::size_t> a, b;
    CounterRng r1(5, 6, 7), r2(5, 6, 7);
    thinned(r1, a);
    thinned(r2, b);
    BOOST_TEST(a == b);
    // Mean probability >= 1/16 uses the direct stream, where thinning has no gain.
    Vector dense(64, 0.07);
    CounterRng r3(5, 6, 8), r4(5, 6, 8);
    crt_sampler(dense)(r3, a);
    crt_sampler(dense, CrtDraw::direct)(r4, b);
    BOOST_TEST(a == b);
}
BOOST_AUTO_TEST_CASE(ars_known_answers_and_equal_implementations) {
    // The known-answer vectors for ARS with 10 rounds distributed with Random123
    // (tests/kat_vectors; D. E. Shaw Research, BSD 3-clause): counter, key and
    // result as four 32-bit words each, lowest first.
    struct Vector128 {
        std::uint32_t w[4];
        std::uint64_t low() const {
            return w[0] | static_cast<std::uint64_t>(w[1]) << 32;
        }
        std::uint64_t high() const {
            return w[2] | static_cast<std::uint64_t>(w[3]) << 32;
        }
    };
    const Vector128 known[3][3] = {
        {{{0, 0, 0, 0}}, {{0, 0, 0, 0}}, {{0x8d73ee19, 0x506401ef, 0x13c2dbe4, 0x0cbe9c0d}}},
        {{{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}},
         {{0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89}},
         {{0xa516e7d6, 0x8357ad74, 0x5b59b3ec, 0x8763fff3}}},
        {{{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}},
         {{0xffffffff, 0xffffffff, 0, 0}},
         {{0xbb3743b1, 0x9f635551, 0xecbc87fc, 0xa19478a9}}}};
    BOOST_TEST(detail::ars_rounds == 10u);
    for (bool hardware : {false, true}) {
        if (hardware && !aes_instructions())
            continue;
        for (const auto &v : known) {
            const std::uint64_t counter[2] = {v[0].low(), v[0].high()};
            std::uint64_t out[2];
            detail::ars_blocks(counter, 1, v[1].low(), v[1].high(), 10, out, hardware);
            BOOST_TEST(out[0] == v[2].low());
            BOOST_TEST(out[1] == v[2].high());
        }
    }
    // The S-box of FIPS 197 at a few entries.
    const auto &tables = detail::aes_tables();
    BOOST_TEST(tables.sbox[0x00] == 0x63);
    BOOST_TEST(tables.sbox[0x01] == 0x7c);
    BOOST_TEST(tables.sbox[0x53] == 0xed);
    BOOST_TEST(tables.sbox[0xff] == 0x16);
    // The processor's instructions and the table lookups give the same words, at
    // any number of rounds and lanes.
    CounterRng source(20261005, 4, 5);
    if (aes_instructions())
        for (unsigned rounds : {1u, 2u, 5u, 7u, 10u})
            for (unsigned lanes : {1u, 3u, 8u})
                for (int repeat = 0; repeat < 50; ++repeat) {
                    std::uint64_t counters[16], fast[16], slow[16];
                    for (auto &c : counters)
                        c = source.next64();
                    const std::uint64_t low = source.next64(), high = source.next64();
                    detail::ars_blocks(counters, lanes, low, high, rounds, fast, true);
                    detail::ars_blocks(counters, lanes, low, high, rounds, slow, false);
                    for (unsigned k = 0; k < 2 * lanes; ++k)
                        BOOST_REQUIRE(fast[k] == slow[k]);
                }
    BOOST_TEST_MESSAGE("AES instructions used: " << aes_instructions());
    // The generator returns block b = ARS(counter (b, 0)), low word first.
    AesCounterRng rng(0x0123456789abcdefULL, 0xfedcba9876543210ULL);
    for (std::uint64_t block = 0; block < 20; ++block) {
        const std::uint64_t counter[2] = {block, 0};
        std::uint64_t out[2];
        detail::ars_blocks(counter, 1, 0x0123456789abcdefULL, 0xfedcba9876543210ULL, 10, out,
                           false);
        BOOST_REQUIRE(rng.next64() == out[0]);
        BOOST_REQUIRE(rng.next64() == out[1]);
    }
    // Keyed by the next two words of a Philox stream, which it consumes.
    CounterRng a(1, 2, 3), b(1, 2, 3), c(1, 2, 3);
    auto from_a = AesCounterRng::from(a), from_b = AesCounterRng::from(b);
    const std::uint64_t low = c.next64(), high = c.next64();
    AesCounterRng direct(low, high);
    for (int i = 0; i < 40; ++i) {
        const auto word = from_a.next64();
        BOOST_REQUIRE(word == from_b.next64());
        BOOST_REQUIRE(word == direct.next64());
    }
    BOOST_TEST(a.next64() == c.next64());
    auto again = AesCounterRng::from(a);
    BOOST_TEST(again.next64() != direct.next64());
    // Uniforms: mean 1/2 and variance 1/12.
    double sum = 0, squares = 0;
    const std::size_t draws = 1000000;
    for (std::size_t i = 0; i < draws; ++i) {
        const double u = rng.uniform();
        BOOST_REQUIRE(u >= 0 && u < 1);
        sum += u;
        squares += u * u;
    }
    BOOST_TEST(std::abs(sum / draws - 0.5) < 5 * std::sqrt(1.0 / 12 / draws));
    BOOST_TEST(std::abs(squares / draws - 1.0 / 3) < 5 * std::sqrt(4.0 / 45 / draws));
}
namespace {
// Standard normals from a counter generator (Box-Muller).
double normal_draw(CounterRng &rng) {
    const double u = 1 - rng.uniform(), v = rng.uniform();
    return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v);
}
// Skew-normal draws with shape 2, optionally with a rare large jump to the
// right: a heavy right tail that three moments and a KS test do not see.
Vector skewed_sample(std::size_t n, std::uint64_t id, double jump_probability) {
    CounterRng rng(20261006, id, 0);
    const double delta = 2 / std::sqrt(5.0);
    Vector v(n);
    for (auto &x : v) {
        x = delta * std::abs(normal_draw(rng)) + std::sqrt(1 - delta * delta) * normal_draw(rng);
        if (rng.uniform() < jump_probability)
            x += 4 + std::abs(normal_draw(rng));
    }
    return v;
}
} // namespace
BOOST_AUTO_TEST_CASE(strict_fit_check_accepts_skew_normal_tails_and_rejects_heavy_ones) {
    // Poisson limits at the 1% level for means 1, 4, 16 and 64.
    BOOST_TEST(detail::poisson_limit_power_of_four(0) == 5u);
    BOOST_TEST(detail::poisson_limit_power_of_four(1) == 10u);
    BOOST_TEST(detail::poisson_limit_power_of_four(2) == 27u);
    BOOST_TEST(detail::poisson_limit_power_of_four(3) == 84u);
    const std::size_t samples = 200;
    for (std::size_t n : {4999u, 19999u}) {
        std::size_t sound = 0, sound_left = 0, heavy = 0, heavy_ks = 0, valid = 0;
        for (std::size_t k = 0; k < samples; ++k) {
            Vector sorted;
            // A true skew-normal sample: sound on both sides.
            auto fit = fit_skew_normal(skewed_sample(n, k, 0), &sorted);
            BOOST_REQUIRE(std::is_sorted(sorted.begin(), sorted.end()));
            BOOST_REQUIRE(sorted.size() == n);
            double expected = 0;
            if (fit.valid) {
                ++valid;
                sound += detail::skew_normal_tail_is_sound(fit, sorted, 3.5, Side::both, expected);
                // The fit's own expectation of draws beyond the statistic.
                BOOST_TEST(expected ==
                               static_cast<double>(n) * skew_normal_tail(fit, 3.5, Side::right),
                           boost::test_tools::tolerance(1e-12));
                sound_left +=
                    detail::skew_normal_tail_is_sound(fit, sorted, -1.5, Side::left, expected);
            }
            // One draw in 5,000 jumps far to the right.
            fit = fit_skew_normal(skewed_sample(n, k, 0.0002), &sorted);
            if (fit.valid) {
                ++heavy_ks;
                heavy += detail::skew_normal_tail_is_sound(fit, sorted, 5.0, Side::both, expected);
            }
        }
        BOOST_TEST_MESSAGE("strict check, n "
                           << n << ": sound right " << sound << " left " << sound_left << " of "
                           << valid << "; heavy tail accepted " << heavy << " of " << heavy_ks
                           << " that pass KS");
        BOOST_TEST(valid > samples * 9 / 10);
        BOOST_TEST(sound > valid * 85 / 100);
        BOOST_TEST(sound_left > valid * 85 / 100);
        BOOST_TEST(heavy_ks > samples / 2); // the KS check alone accepts most of them
        // With 19,999 draws nearly every sample holds a jump for the check to
        // find; among 4,999 about a third hold none and are skew-normal samples.
        if (n > 10000)
            BOOST_TEST(heavy < samples / 10);
    }
}
BOOST_AUTO_TEST_CASE(strict_fit_check_confirms_small_p_values_on_the_last_budget) {
    // The staging on given null draws: skew-normal (sound) and heavy-tailed.
    const auto run = [](const Vector &draws, double statistic, FitCheck check) {
        ResampleOptions o;
        o.budgets = {4999, 19999};
        o.confirm = 0; // the confirmation draws have their own test
        o.fit_check = check;
        detail::NullStages stages(o, statistic);
        do
            std::copy(draws.begin() + static_cast<std::ptrdiff_t>(stages.begin()),
                      draws.begin() + static_cast<std::ptrdiff_t>(stages.end()),
                      stages.null() + stages.begin());
        while (!stages.next());
        return stages.result();
    };
    BOOST_TEST((ResampleOptions().fit_check == FitCheck::strict));
    std::size_t same = 0, confirmed = 0, rejected = 0, ks_small = 0;
    const std::size_t samples = 40;
    for (std::size_t k = 0; k < samples; ++k) {
        const auto sound = skewed_sample(19999, 1000 + k, 0),
                   heavy = skewed_sample(19999, k, 0.0002);
        // A p-value the first budget can check: accepted there, as by the KS check.
        const auto near_ks = run(sound, 2.5, FitCheck::ks),
                   near = run(sound, 2.5, FitCheck::strict);
        if (near_ks.used_approximation && near.used_approximation && near.draws == 4999) {
            BOOST_TEST(near.p_value == near_ks.p_value);
            ++same;
        }
        // One it cannot check: the KS check accepts it at 4,999 draws, the strict
        // check draws the last budget and accepts the fit made there.
        const auto far_ks = run(sound, 5.5, FitCheck::ks), far = run(sound, 5.5, FitCheck::strict);
        if (far_ks.used_approximation)
            BOOST_TEST(far_ks.draws == 4999u);
        BOOST_TEST(far.draws == 19999u);
        if (far.used_approximation) {
            ++confirmed;
            BOOST_TEST(far.p_value < 1e-5);
        }
        // A heavy tail: the KS check reports a tiny p-value where about one draw
        // in 10,000 lies beyond the statistic; the strict check finds them among
        // the draws of the last budget.
        const auto heavy_ks = run(heavy, 5.0, FitCheck::ks),
                   strict = run(heavy, 5.0, FitCheck::strict);
        ks_small += heavy_ks.used_approximation && heavy_ks.p_value < 1e-5;
        BOOST_TEST(strict.draws == 19999u);
        if (!strict.used_approximation) {
            ++rejected;
            BOOST_TEST(strict.p_value == empirical_tail(heavy, 5.0, Side::both));
            BOOST_TEST(strict.p_value > 1e-4);
        }
    }
    BOOST_TEST_MESSAGE("staging: same at 4,999 "
                       << same << ", confirmed at 19,999 " << confirmed << ", heavy rejected "
                       << rejected << ", heavy with a small KS-checked p " << ks_small << " of "
                       << samples);
    BOOST_TEST(same > samples * 8 / 10);
    BOOST_TEST(confirmed > samples * 8 / 10);
    BOOST_TEST(rejected >= samples * 9 / 10);
    BOOST_TEST(ks_small > samples / 2);
}
BOOST_AUTO_TEST_CASE(confirmation_draws_recheck_small_p_values) {
    const std::size_t confirm = 199999;
    const auto run = [](const Vector &draws, double statistic, std::size_t confirmation) {
        ResampleOptions o;
        o.budgets = {4999, 19999};
        o.confirm = confirmation;
        detail::NullStages stages(o, statistic);
        do
            std::copy(draws.begin() + static_cast<std::ptrdiff_t>(stages.begin()),
                      draws.begin() + static_cast<std::ptrdiff_t>(stages.end()),
                      stages.null() + stages.begin());
        while (!stages.next());
        return stages.result();
    };
    const std::size_t samples = 12;
    std::size_t confirmed = 0, passed = 0, caught = 0, ended = 0;
    for (std::size_t k = 0; k < samples; ++k) {
        const auto sound = skewed_sample(confirm, 2000 + k, 0);
        // A p-value within reach of the budgets is reported as without the option.
        const auto near = run(sound, 2.5, 0), near_confirm = run(sound, 2.5, confirm);
        BOOST_TEST(near_confirm.p_value == near.p_value);
        BOOST_TEST(near_confirm.draws == near.draws);
        BOOST_TEST(near.draws <= 19999u);
        // One beyond the last budget's reach is accepted there without the
        // option, and with it only from the fit on the confirmation draws.
        const auto far = run(sound, 5.5, 0), far_confirm = run(sound, 5.5, confirm);
        BOOST_TEST(far.draws == 19999u);
        BOOST_TEST(far_confirm.draws == confirm);
        confirmed += far.used_approximation && far_confirm.used_approximation;
        // One draw in 100,000 jumps: 19,999 draws rarely hold a jump, 199,999
        // nearly always, and the fit is then rejected for the empirical p-value.
        const auto mild = skewed_sample(confirm, 3000 + k, 1e-5);
        const auto mild_last = run(mild, 5.0, 0), mild_confirm = run(mild, 5.0, confirm);
        if (mild_last.used_approximation) {
            ++passed;
            BOOST_TEST(mild_confirm.draws == confirm);
            if (!mild_confirm.used_approximation) {
                ++caught;
                BOOST_TEST(mild_confirm.p_value == empirical_tail(mild, 5.0, Side::both));
            }
        }
        // A fit rejected on the last budget ends there: no confirmation draws.
        const auto heavy = skewed_sample(confirm, k, 0.0005);
        const auto heavy_confirm = run(heavy, 5.0, confirm);
        if (!heavy_confirm.used_approximation && heavy_confirm.draws == 19999) {
            ++ended;
            BOOST_TEST(
                heavy_confirm.p_value ==
                empirical_tail(Vector(heavy.begin(), heavy.begin() + 19999), 5.0, Side::both));
        }
    }
    BOOST_TEST_MESSAGE("confirmation: sound confirmed "
                       << confirmed << ", mild tail passed 19,999 " << passed << " and caught "
                       << caught << ", heavy ended at 19,999 " << ended << " of " << samples);
    BOOST_TEST(confirmed >= samples * 3 / 4);
    BOOST_TEST(passed >= samples / 2);
    BOOST_TEST(caught * 2 >= passed);
    BOOST_TEST(ended >= samples * 3 / 4);
    // By default ten times the last budget confirms: 199,999 draws for 19,999.
    // A number not above the last budget turns the confirmation off.
    ResampleOptions defaults;
    BOOST_TEST(defaults.confirmation_draws() == confirm);
    defaults.budgets = {99, 999};
    BOOST_TEST(defaults.confirmation_draws() == 9999u);
    defaults.fit_check = FitCheck::ks;
    BOOST_TEST(defaults.confirmation_draws() == 0u);
    const auto sound = skewed_sample(confirm, 2000, 0);
    BOOST_TEST(run(sound, 5.5, 19999).draws == 19999u);
    BOOST_TEST(run(sound, 5.5, 19999).p_value == run(sound, 5.5, 0).p_value);
}
BOOST_AUTO_TEST_CASE(exponential_ziggurat_matches_the_exponential_law) {
    const auto &ziggurat = detail::exponential_ziggurat();
    const std::size_t draws = 4000000;
    const Vector cuts = {1e-3, 0.01, 0.1, 0.5, 1, 2, 4, 7, 7.69711747013104972, 9, 12};
    std::vector<std::size_t> above(cuts.size());
    double sum = 0, squares = 0;
    AesCounterRng rng(20261005, 12);
    for (std::size_t d = 0; d < draws; ++d) {
        const double x = ziggurat(rng);
        BOOST_REQUIRE(x >= 0);
        sum += x;
        squares += x * x;
        for (std::size_t c = 0; c < cuts.size(); ++c)
            above[c] += x > cuts[c];
    }
    // Mean 1 and second moment 2 (standard errors 1/sqrt(n) and sqrt(20/n)).
    BOOST_TEST(std::abs(sum / draws - 1) < 5 / std::sqrt(static_cast<double>(draws)));
    BOOST_TEST(std::abs(squares / draws - 2) < 5 * std::sqrt(20.0 / draws));
    // Survival exp(-c) at cuts in the layers, at the tail's start and beyond it.
    for (std::size_t c = 0; c < cuts.size(); ++c) {
        const double s = std::exp(-cuts[c]);
        BOOST_TEST(std::abs(static_cast<double>(above[c]) / draws - s) <
                   5 * std::sqrt(s * (1 - s) / draws) + 1e-12);
    }
    BOOST_TEST(above.back() > 0u);
}
BOOST_AUTO_TEST_CASE(hazard_crt_draws_match_bernoulli_law) {
    // The probabilities of the thinned test: zero and one, large and small
    // values, and a sparse tail; then a long sparse vector as in a screen.
    Vector p = {0, 1, 0.5, 0.25, 0.3, 0.125, 0.1, 0.02, 1.0 / 1024, 0.7, 0.003, 0.05};
    for (std::size_t i = 0; i < 300; ++i)
        p.push_back(0.004 + 0.02 * (i % 17) / 17);
    // Sparse enough for the line: mean probability below 1/16.
    p.resize(4000, 0);
    for (std::size_t i = 400; i < p.size(); ++i)
        p[i] = i % 5 == 0 ? 0 : 0.0005 + 0.03 * ((i * 7919) % 101) / 101;
    const std::size_t n = p.size(), draws = 40000;
    // `true`: the AES generator, by table lookups where the processor lacks the
    // instructions, so that the method is tested on every machine.
    const auto hazard = crt_sampler(p, CrtDraw::hazard, true);
    Vector hits(n), sizes;
    double joint = 0, neighbours = 0; // cells 4 and 7, and the adjacent cells 5 and 6
    std::vector<std::size_t> selected;
    for (std::size_t d = 0; d < draws; ++d) {
        CounterRng rng(20261005, 3, d);
        hazard(rng, selected);
        BOOST_REQUIRE(std::is_sorted(selected.begin(), selected.end()));
        BOOST_REQUIRE(std::adjacent_find(selected.begin(), selected.end()) == selected.end());
        BOOST_REQUIRE(selected.empty() || selected.back() < n);
        for (auto i : selected)
            ++hits[i];
        const auto has = [&](std::size_t i) {
            return std::binary_search(selected.begin(), selected.end(), i);
        };
        joint += has(4) && has(7);
        neighbours += has(5) && has(6);
        sizes.push_back(selected.size());
    }
    double chi = 0, expected_size = 0, size_variance = 0;
    std::size_t df = 0;
    for (std::size_t i = 0; i < n; ++i) {
        expected_size += p[i];
        size_variance += p[i] * (1 - p[i]);
        if (p[i] == 0)
            BOOST_REQUIRE(hits[i] == 0);
        if (p[i] == 1)
            BOOST_REQUIRE(hits[i] == draws);
        if (p[i] > 0 && p[i] < 1) {
            chi += std::pow(hits[i] - draws * p[i], 2) / (draws * p[i] * (1 - p[i]));
            ++df;
        }
    }
    // Chi-square with df degrees of freedom: within 4 standard deviations.
    BOOST_TEST(df > 3000u);
    BOOST_TEST(std::abs(chi - df) < 4 * std::sqrt(2.0 * df));
    const double mean = std::accumulate(sizes.begin(), sizes.end(), 0.0) / draws;
    double variance = 0;
    for (double s : sizes)
        variance += (s - mean) * (s - mean) / (draws - 1);
    BOOST_TEST(std::abs(mean - expected_size) < 5 * std::sqrt(size_variance / draws));
    BOOST_TEST(variance / size_variance > 0.95);
    BOOST_TEST(variance / size_variance < 1.05);
    for (auto [observed, pj] :
         {std::pair<double, double>{joint, p[4] * p[7]}, {neighbours, p[5] * p[6]}})
        BOOST_TEST(std::abs(observed / draws - pj) < 5 * std::sqrt(pj * (1 - pj) / draws));
    // Draws are a pure function of the RNG key.
    std::vector<std::size_t> a, b;
    CounterRng r1(5, 6, 7), r2(5, 6, 7);
    hazard(r1, a);
    hazard(r2, b);
    BOOST_TEST(a == b);
    BOOST_TEST(!a.empty());
    // Dense probabilities, and lines made long by probabilities near 1, use
    // the direct stream; all zeros select nothing.
    for (const Vector &other : {Vector(64, 0.07), [] {
                                    Vector v(64, 0.001);
                                    v[3] = v[40] = 1;
                                    return v;
                                }()}) {
        CounterRng r3(5, 6, 8), r4(5, 6, 8);
        crt_sampler(other, CrtDraw::hazard, true)(r3, a);
        crt_sampler(other, CrtDraw::direct)(r4, b);
        BOOST_TEST(a == b);
    }
    CounterRng r5(5, 6, 9);
    crt_sampler(Vector(100, 0), CrtDraw::hazard, true)(r5, a);
    BOOST_TEST(a.empty());
    // Without AES instructions hazard is the thinned sampler, stream and all;
    // by default the processor decides.
    const auto fallback = crt_sampler(p, CrtDraw::hazard, false), thinned = crt_sampler(p),
               chosen = crt_sampler(p, CrtDraw::hazard);
    std::size_t differ = 0;
    for (std::size_t d = 0; d < 20; ++d) {
        CounterRng r6(8, 9, d), r7(8, 9, d), r8(8, 9, d), r9(8, 9, d);
        std::vector<std::size_t> with_aes, by_default;
        fallback(r6, a);
        thinned(r7, b);
        hazard(r8, with_aes);
        chosen(r9, by_default);
        BOOST_REQUIRE(a == b);
        BOOST_REQUIRE(by_default == (aes_instructions() ? with_aes : b));
        differ += with_aes != b;
    }
    BOOST_TEST(differ > 0u);
}
BOOST_AUTO_TEST_CASE(tails_adjustment_and_threads) {
    BOOST_TEST(empirical_tail({1, 2, 3}, 2, Side::left) == 0.75);
    BOOST_TEST(empirical_tail({1, 2, 3}, 4, Side::right) == 0.25);
    auto p = adjust_pvalues({0.01, 0.04, 0.03, rapidglm::nan});
    BOOST_TEST(p[0] == 0.04);
    BOOST_TEST(p[1] == 0.05333333333333, boost::test_tools::tolerance(1e-10));
    BOOST_TEST(p[3] == 1);
    Matrix x(80, 1, 1);
    Vector y(80), g(80);
    for (unsigned i = 0; i < 80; ++i) {
        y[i] = i % 7;
        g[i] = i < 20;
    }
    FitOptions o;
    o.family = Family::poisson;
    auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    ScoreWorkspace score(x, f);
    std::vector<ResampleResult> baseline(20);
    for (int threads : {1, 2, 8, 16}) {
        std::vector<ResampleResult> results(20);
        parallel_for(20, threads, [&](auto i) {
            ResampleOptions r;
            r.budgets = {49, 99};
            r.approximate = false;
            r.seed = 29;
            r.test_id = i;
            r.retain_null = true;
            results[i] = resample(score, g, permutation_sampler(g), r);
        });
        if (threads == 1)
            baseline = results;
        for (unsigned i = 0; i < 20; ++i) {
            BOOST_TEST(results[i].p_value == baseline[i].p_value);
            BOOST_TEST(results[i].draws == 99u);
            BOOST_TEST(results[i].null_statistics == baseline[i].null_statistics,
                       boost::test_tools::per_element());
        }
    }
}
BOOST_AUTO_TEST_CASE(sparse_batch_validation) {
    SparseCSC s{4, 2, {0, 2, 4}, {0, 2, 1, 3}, {1, 2, 3, 4}};
    Matrix x(4, 1, 1);
    FitOptions o;
    o.family = Family::poisson;
    auto one = fit_batch(x, s, o, 1), two = fit_batch(x, s, o, 2);
    BOOST_TEST(one[0].coefficients == two[0].coefficients, boost::test_tools::per_element());
    BOOST_TEST(one[0].fitted.empty());
    s.row_index[1] = 0;
    BOOST_CHECK_THROW(fit_batch(x, s, o, 2), std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(draw_tasks_preserve_streams_and_staged_tails) {
    Matrix x(513, 2, 1);
    Vector y(x.rows), g(x.rows), probabilities(x.rows);
    std::vector<std::size_t> strata(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i) {
        x(i, 1) = std::sin(i * 0.1);
        y[i] = i % 7;
        g[i] = i % 9 < 2;
        probabilities[i] = 0.08 + 0.3 * (i % 7) / 6;
        strata[i] = i % 3;
    }
    FitOptions options;
    options.family = Family::poisson;
    auto fitted = fit(x, y, options);
    BOOST_REQUIRE(fitted.ok());
    ScoreWorkspace score(x, fitted);
    for (const auto &sampler : {crt_sampler(probabilities), permutation_sampler(g, strata)}) {
        for (bool approximate : {false, true}) {
            ResampleOptions ro;
            ro.seed = 9123;
            ro.test_id = 71;
            ro.budgets = {31, 99, 499};
            ro.approximate = approximate;
            ro.retain_null = true;
            auto baseline = resample(score, g, sampler, ro);
            BOOST_REQUIRE(baseline.valid);
            ro.parallel_draws = true;
            for (int threads : {1, 2, 8, 16}) {
                ResampleResult result;
                // One response is deliberate: draw tasks must not depend on
                // there being enough response batches to occupy the team.
                parallel_for(1, threads, [&](auto) { result = resample(score, g, sampler, ro); });
                BOOST_REQUIRE(result.valid);
                BOOST_TEST(result.null_statistics == baseline.null_statistics,
                           boost::test_tools::per_element());
                BOOST_TEST(result.p_value == baseline.p_value);
                BOOST_TEST(result.statistic == baseline.statistic);
                BOOST_TEST(result.draws == baseline.draws);
                BOOST_TEST(result.used_approximation == baseline.used_approximation);
                if (std::isfinite(baseline.fit_discrepancy))
                    BOOST_TEST(result.fit_discrepancy == baseline.fit_discrepancy);
            }
        }
    }
}
BOOST_AUTO_TEST_CASE(draw_tasks_propagate_sampler_errors) {
    Matrix x(80, 1, 1);
    Vector y(80), g(80);
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = i % 7;
        g[i] = i < 20;
    }
    FitOptions options;
    options.family = Family::poisson;
    ScoreWorkspace score(x, fit(x, y, options));
    ResampleOptions ro;
    ro.budgets = {33};
    ro.parallel_draws = true;
    Sampler invalid = [](CounterRng &, std::vector<std::size_t> &selected) { selected = {1, 1}; };
    BOOST_CHECK_THROW(parallel_for(1, 8, [&](auto) { resample(score, g, invalid, ro); }),
                      std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(workflow_qc_integration_and_aggregation) {
    const std::size_t n = 60;
    Matrix x(n, 1, 1);
    SparseCSC responses;
    responses.rows = n;
    responses.cols = 1;
    responses.column_ptr = {0, n};
    for (std::size_t i = 0; i < n; ++i) {
        responses.row_index.push_back(i);
        responses.values.push_back(i < 20 ? 1 + i % 2 : 4 + i % 3);
    }
    SparseCSC calls;
    calls.rows = n;
    calls.cols = 3;
    calls.column_ptr = {0, 10, 20, 60};
    for (std::size_t i = 0; i < n; ++i) {
        calls.row_index.push_back(i);
        calls.values.push_back(1);
    }
    std::vector<Guide> guides = {
        {"g1", "target", false}, {"g2", "target", false}, {"nt", "NT", true}};
    std::vector<Pair> pairs = {{0, "target", PairKind::discovery}};
    SceptreOptions o;
    o.cell_qc = false;
    o.model.family = Family::poisson;
    o.null_model = FitMethod::direct;
    o.resampling.budgets = {99};
    o.resampling.approximate = false;
    auto u = sceptre(x, responses, calls, guides, pairs, {}, o);
    BOOST_REQUIRE(u.tests.size() == 1);
    BOOST_TEST(u.tests[0].cells == 60u);
    BOOST_TEST(u.tests[0].nonzero_treatment == 20u);
    BOOST_TEST(u.tests[0].log2fc < 0);
    BOOST_TEST(u.tests[0].p_value <= .05);
    o.integration = Integration::singleton;
    auto s = sceptre(x, responses, calls, guides, pairs, {}, o);
    BOOST_TEST(s.tests.size() == 2u);
    BOOST_TEST(s.tests[0].cells == 50u);
    o.integration = Integration::bonferroni;
    auto b = sceptre(x, responses, calls, guides, pairs, {}, o);
    BOOST_TEST(b.tests.size() == 1u);
    BOOST_TEST(b.tests[0].p_value ==
               std::min(1.0, 2 * std::min(s.tests[0].p_value, s.tests[1].p_value)));
    o.min_nonzero = 100;
    // A pair that fails pair QC is outside the default family (the tested
    // pairs); in the family of the listed pairs it counts with p = 1.
    auto failed = sceptre(x, responses, calls, guides, pairs, {}, o);
    BOOST_TEST(std::isnan(failed.tests[0].adjusted_p));
    BOOST_TEST(std::isnan(failed.tests[0].p_value));
    o.family = AdjustmentFamily::listed;
    BOOST_TEST(sceptre(x, responses, calls, guides, pairs, {}, o).tests[0].adjusted_p == 1);
    o.family = AdjustmentFamily::tested;
    std::vector<std::size_t> units(n);
    for (std::size_t i = 0; i < n; ++i)
        units[i] = i / 20;
    auto bulk = aggregate_pseudobulk(responses, units, 3);
    auto values = bulk.column(0);
    BOOST_TEST(values[0] == 30);
    BOOST_TEST(std::accumulate(values.begin(), values.end(), 0.0) ==
               std::accumulate(responses.values.begin(), responses.values.end(), 0.0));
    BOOST_CHECK_THROW(
        sceptre(x, responses, calls, guides, {{0, "target", PairKind::negative_control}}, {}, o),
        std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(shared_trend_and_extreme_dispersion) {
    Matrix x(60, 1, 1);
    SparseCSC s;
    s.rows = 60;
    s.cols = 3;
    s.column_ptr = {0};
    for (unsigned j = 0; j < 3; ++j) {
        for (unsigned i = 0; i < 60; ++i) {
            s.row_index.push_back(i);
            s.values.push_back((i % (3 + j)) * (2 + j));
        }
        s.column_ptr.push_back(s.values.size());
    }
    FitOptions o;
    o.dispersion = Dispersion::ml;
    auto fits = fit_batch(x, s, o, 2);
    for (auto &f : fits)
        BOOST_REQUIRE(f.ok());
    auto trend = estimate_dispersion_trend(s, fits);
    BOOST_TEST(trend.size() == 3u);
    auto trended = fit_batch(x, s, o, 2, trend);
    for (auto &f : trended)
        BOOST_TEST(f.ok());
    double shared = estimate_shared_theta(x, s, o, 2);
    BOOST_TEST(shared > o.theta_min);
    BOOST_TEST(shared < o.theta_max);
    o.dispersion = Dispersion::fixed;
    for (double theta : {1e-6, 1e8}) {
        o.theta = theta;
        auto f = fit(x, s.column(0), o);
        BOOST_TEST(f.ok());
        BOOST_TEST(std::isfinite(f.log_likelihood));
    }
}
BOOST_AUTO_TEST_CASE(invalid_lr_and_parallel_exceptions) {
    Matrix x(12, 1, 1);
    Vector y(12);
    for (unsigned i = 0; i < 12; ++i)
        y[i] = i % 3;
    FitOptions o;
    o.family = Family::poisson;
    auto a = fit(x, y, o);
    auto full = append_column(x, y);
    auto b = fit(full, y, o);
    BOOST_TEST(likelihood_ratio(a, b).valid);
    y[0] += 1;
    auto c = fit(full, y, o);
    BOOST_CHECK_THROW(likelihood_ratio(a, c), std::invalid_argument);
    BOOST_CHECK_THROW(parallel_for(4, 2,
                                   [](std::size_t i) {
                                       if (i == 2)
                                           throw std::invalid_argument("test");
                                   }),
                      std::invalid_argument);
    o.max_iterations = 1;
    auto f = fit(full, y, o);
    BOOST_TEST(!f.ok());
}
BOOST_AUTO_TEST_CASE(malformed_public_storage) {
    SparseCSC malformed;
    malformed.cols = std::numeric_limits<std::size_t>::max();
    BOOST_CHECK_THROW(malformed.validate(), std::invalid_argument);
    Matrix design(4, 1, 1);
    FitOptions o;
    o.family = Family::poisson;
    auto null = fit(design, {1, 2, 3, 4}, o);
    design.values.clear();
    BOOST_CHECK_THROW(ScoreWorkspace(design, null), std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(skew_normal_tails_stay_in_unit_interval) {
    // Fitted null and observed statistic from the public S1 singleton panel, where
    // Boost's CDF rounded to -5.2e-54 and the negative p-value aborted adjustment.
    SkewFit fit;
    fit.location = -0.46555101859508935;
    fit.scale = 1.0818401226991083;
    fit.shape = 0.74890180569388087;
    for (double statistic : {-14.48310980779172, -40.0, 0.0, 40.0})
        for (auto side : {Side::left, Side::right, Side::both}) {
            double p = skew_normal_tail(fit, statistic, side);
            BOOST_TEST((p >= 0 && p <= 1));
        }
    BOOST_TEST(skew_normal_tail(fit, 0.0, Side::left) + skew_normal_tail(fit, 0.0, Side::right) ==
                   1.0,
               boost::test_tools::tolerance(1e-12));
}
BOOST_AUTO_TEST_CASE(nb_ml_starts_inside_when_the_pilot_shows_no_overdispersion) {
    // An airway gene (design ~ cell + dex, log size-factor offsets): the Poisson
    // pilot leaves no moment overdispersion, so the moment start is the upper
    // bound, where the likelihood is flat in theta. The MLE is interior.
    const double design[8][5] = {{1, 0, 0, 1, 0}, {1, 0, 0, 1, 1}, {1, 0, 0, 0, 0},
                                 {1, 0, 0, 0, 1}, {1, 0, 1, 0, 0}, {1, 0, 1, 0, 1},
                                 {1, 1, 0, 0, 0}, {1, 1, 0, 0, 1}};
    Matrix x(8, 5);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            x(i, j) = design[i][j];
    const Vector y = {60, 200, 108, 73, 1711, 4112, 74, 202};
    const Vector sf = {1.024, 0.896, 1.18, 0.67, 1.178, 1.399, 0.921, 0.945};
    FitOptions o;
    o.family = Family::negative_binomial;
    o.dispersion = Dispersion::ml;
    for (double s : sf)
        o.offset.push_back(std::log(s));
    const auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(!f.theta_boundary);
    BOOST_TEST(f.theta > 10);
    BOOST_TEST(f.theta < 100);
    // At the MLE the profile likelihood is flat in theta: nearby values are no better.
    for (double factor : {0.9, 1.1}) {
        auto fixed = o;
        fixed.dispersion = Dispersion::fixed;
        fixed.theta = f.theta * factor;
        BOOST_TEST(fit(x, y, fixed).log_likelihood <= f.log_likelihood + 1e-9);
    }
}
BOOST_AUTO_TEST_CASE(nb_ml_joint_newton_does_not_overshoot_a_profile_maximum) {
    // An airway gene whose profile likelihood in theta has a maximum near 64, a dip
    // near 1000 and a lower Poisson asymptote. From theta = 10 the joint steps used
    // to overshoot while the coefficients lagged and climb to the theta bound.
    const double design[8][5] = {{1, 0, 0, 1, 0}, {1, 0, 0, 1, 1}, {1, 0, 0, 0, 0},
                                 {1, 0, 0, 0, 1}, {1, 0, 1, 0, 0}, {1, 0, 1, 0, 1},
                                 {1, 1, 0, 0, 0}, {1, 1, 0, 0, 1}};
    Matrix x(8, 5);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            x(i, j) = design[i][j];
    const Vector y = {59, 80, 134, 42, 1607, 1597, 89, 101};
    const Vector sf = {1.024, 0.896, 1.18, 0.67, 1.178, 1.399, 0.921, 0.945};
    FitOptions o;
    o.family = Family::negative_binomial;
    o.dispersion = Dispersion::ml;
    for (double s : sf)
        o.offset.push_back(std::log(s));
    const auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(f.theta == 64.19, boost::test_tools::tolerance(0.01));
    auto poisson_limit = o;
    poisson_limit.dispersion = Dispersion::fixed;
    poisson_limit.theta = o.theta_max;
    BOOST_TEST(f.log_likelihood > fit(x, y, poisson_limit).log_likelihood + 0.5);
}
BOOST_AUTO_TEST_CASE(nb_ml_joint_newton_holds_diverging_coefficients) {
    // Airway genes with an all-zero cell line: that coefficient diverges, its weights
    // vanish and the full Newton system turns singular while theta still moves. The
    // fit holds the diverging coefficient, finishes on the others and theta, and
    // reports standard errors for the identifiable coefficients only.
    const double design[8][5] = {{1, 0, 0, 1, 0}, {1, 0, 0, 1, 1}, {1, 0, 0, 0, 0},
                                 {1, 0, 0, 0, 1}, {1, 0, 1, 0, 0}, {1, 0, 1, 0, 1},
                                 {1, 1, 0, 0, 0}, {1, 1, 0, 0, 1}};
    Matrix x(8, 5);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            x(i, j) = design[i][j];
    const Vector sf = {1.024, 0.896, 1.18, 0.67, 1.178, 1.399, 0.921, 0.945};
    FitOptions o;
    o.family = Family::negative_binomial;
    o.dispersion = Dispersion::ml;
    for (double s : sf)
        o.offset.push_back(std::log(s));
    for (const Vector &y : {Vector{0, 1, 0, 0, 11, 3, 0, 0}, Vector{2, 2, 0, 0, 42, 6, 3, 0},
                            Vector{0, 3, 0, 0, 5, 3, 0, 4}}) {
        const auto f = fit(x, y, o);
        BOOST_REQUIRE(f.status == FitStatus::converged_boundary);
        BOOST_TEST(f.has_boundary());
        std::size_t finite = 0;
        for (std::size_t j = 0; j < 5; ++j)
            finite += std::isfinite(f.standard_errors[j]) != 0;
        BOOST_TEST(finite >= 1u);
        // No fixed theta does better than the joint estimate.
        for (double theta : {0.5, 5.0, 50.0, 5e3, 1e8}) {
            auto fixed = o;
            fixed.dispersion = Dispersion::fixed;
            fixed.theta = theta;
            BOOST_TEST(fit(x, y, fixed).log_likelihood <= f.log_likelihood + 1e-6);
        }
    }
}
BOOST_AUTO_TEST_CASE(nb_ml_keeps_the_poisson_limit_when_it_beats_an_interior_maximum) {
    // A Kang pseudobulk gene (donor + stimulation design, library-size offsets):
    // the moment start shows no overdispersion, the profile likelihood has a local
    // maximum near theta = 44, and the Poisson limit is about 4.8 higher. Fits whose
    // moment start is at the bound run from both starts and keep the better.
    const double design[16][9] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 1, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 1, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 1, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 1, 0, 0}, {1, 0, 0, 0, 0, 0, 1, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 0, 0, 1, 1}};
    Matrix x(16, 9);
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 9; ++j)
            x(i, j) = design[i][j];
    const Vector y = {19, 1434, 56, 3679, 47, 1627, 5, 771, 13, 1008, 18, 1922, 40, 2060, 5, 1500};
    FitOptions o;
    o.family = Family::negative_binomial;
    o.dispersion = Dispersion::ml;
    o.offset = {13.1706257832862, 13.496010279406,  14.3375828996333, 14.335423367904,
                13.4897222444378, 13.4693345920812, 12.6317161273265, 13.0457889815113,
                13.0595577539319, 12.9863214073446, 13.7857057754721, 13.4877451070371,
                13.8025235911956, 13.804448599482,  13.1738983425871, 13.5083221343222};
    const auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    auto interior = o;
    interior.dispersion = Dispersion::fixed;
    interior.theta = 43.57;
    BOOST_TEST(f.log_likelihood > fit(x, y, interior).log_likelihood + 4);
}
