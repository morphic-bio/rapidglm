#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
SparseCSC dense_columns(const std::vector<Vector> &columns) {
    SparseCSC s;
    s.rows = columns.front().size();
    s.cols = columns.size();
    s.column_ptr = {0};
    for (const auto &c : columns) {
        for (std::size_t i = 0; i < c.size(); ++i)
            if (c[i] != 0) {
                s.row_index.push_back(i);
                s.values.push_back(c[i]);
            }
        s.column_ptr.push_back(s.values.size());
    }
    return s;
}
// Poisson responses on an intercept + depth design; the first predictor of each
// kind (binary, continuous) has an effect on the first response only.
struct Data {
    Matrix x;
    SparseCSC y, g;
};
Data data(std::uint64_t seed, std::size_t n = 300) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;
    Data d;
    d.x = Matrix(n, 2, 1);
    std::vector<Vector> g(4, Vector(n)), y(3, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        d.x(i, 1) = normal(rng);
        g[0][i] = uniform(rng) < 0.3;
        g[1][i] = uniform(rng) < 0.3;
        g[2][i] = normal(rng);
        g[3][i] = normal(rng);
        for (std::size_t r = 0; r < 3; ++r) {
            double eta = 1 + 0.4 * d.x(i, 1);
            if (r == 0)
                eta += 0.6 * g[0][i] + 0.3 * g[2][i];
            y[r][i] = std::poisson_distribution<int>(std::exp(eta))(rng);
        }
    }
    d.y = dense_columns(y);
    d.g = dense_columns(g);
    return d;
}
} // namespace
BOOST_AUTO_TEST_CASE(scan_asymptotic_matches_the_score_test) {
    auto d = data(1);
    ScanOptions o;
    o.model.family = Family::poisson;
    o.calibration = Calibration::asymptotic;
    o.effect_fits = EffectFits::all;
    auto results = score_scan(d.x, d.y, d.g, {}, o);
    BOOST_REQUIRE(results.size() == 12u);
    for (const auto &r : results) {
        BOOST_REQUIRE(r.status == "ok");
        auto null = fit(d.x, d.y.column(r.response), [&] {
            auto m = o.model;
            m.retain_observations = true;
            return m;
        }());
        ScoreWorkspace score(d.x, null);
        auto t = score.test(d.g.column(r.predictor));
        BOOST_TEST(r.statistic == t.statistic, boost::test_tools::tolerance(1e-10));
        BOOST_TEST(r.p_value == t.p_value, boost::test_tools::tolerance(1e-9));
        BOOST_TEST(r.estimate == t.estimate, boost::test_tools::tolerance(1e-10));
        BOOST_TEST(r.standard_error == t.standard_error, boost::test_tools::tolerance(1e-10));
        // The full-fit effect is the Wald estimate of the added column.
        auto alt =
            fit(append_column(d.x, d.g.column(r.predictor)), d.y.column(r.response), o.model);
        BOOST_TEST(r.effect == alt.coefficients[2], boost::test_tools::tolerance(1e-8));
    }
    // Planted effects are found; the one-step estimate is close to the full fit.
    BOOST_TEST(results[0].p_value < 1e-6);
    BOOST_TEST(results[2].p_value < 1e-6);
    BOOST_TEST(std::abs(results[0].estimate - results[0].effect) < 0.1);
}
BOOST_AUTO_TEST_CASE(scan_resampling_is_thread_invariant_and_pair_ordered) {
    auto d = data(2);
    for (auto calibration : {Calibration::permutation, Calibration::crt}) {
        ScanOptions o;
        o.model.family = Family::poisson;
        o.calibration = calibration;
        o.resampling.budgets = {199, 999};
        std::vector<ScanPair> pairs;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t j = 0; j < (calibration == Calibration::crt ? 2u : 4u); ++j)
                pairs.push_back({r, j, j < 2 ? "binary" : "continuous"});
        o.threads = 1;
        auto one = score_scan(d.x, d.y, d.g, pairs, o);
        o.threads = 4;
        auto four = score_scan(d.x, d.y, d.g, pairs, o);
        for (std::size_t k = 0; k < one.size(); ++k) {
            BOOST_REQUIRE(one[k].status == "ok");
            BOOST_TEST(one[k].p_value == four[k].p_value);
            BOOST_TEST(one[k].statistic == four[k].statistic);
            BOOST_TEST(one[k].draws >= 199u);
        }
        BOOST_TEST(one[0].p_value < 0.01);
        // Families are adjusted separately.
        Vector binary_p;
        for (const auto &r : one)
            if (r.family == "binary")
                binary_p.push_back(r.p_value);
        auto q = adjust_pvalues(binary_p);
        BOOST_TEST(one[0].adjusted_p == q[0]);
    }
}
BOOST_AUTO_TEST_CASE(continuous_permutation_respects_strata) {
    auto d = data(3);
    // A predictor constant within strata cannot be permuted: every draw equals it.
    std::vector<std::size_t> strata(d.x.rows);
    Vector g(d.x.rows);
    for (std::size_t i = 0; i < g.size(); ++i) {
        strata[i] = i % 3;
        g[i] = 0.5 * strata[i] + 0.1;
    }
    auto null = fit(d.x, d.y.column(1), [] {
        FitOptions m;
        m.family = Family::poisson;
        return m;
    }());
    ScoreWorkspace score(d.x, null);
    ResampleOptions ro;
    ro.budgets = {99};
    ro.approximate = false;
    ro.retain_null = true;
    auto fixed = resample_permutation(score, g, strata, ro);
    BOOST_REQUIRE(fixed.valid);
    for (double t : fixed.null_statistics)
        BOOST_TEST(t == fixed.statistic, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(fixed.p_value == 1.0);
    // Without strata the draws vary and are reproducible.
    auto free1 = resample_permutation(score, g, {}, ro),
         free2 = resample_permutation(score, g, {}, ro);
    BOOST_TEST(free1.null_statistics == free2.null_statistics);
    BOOST_TEST(free1.null_statistics.front() != free1.null_statistics.back());
}
BOOST_AUTO_TEST_CASE(scan_input_validation_and_pair_qc) {
    auto d = data(4);
    ScanOptions o;
    o.model.family = Family::poisson;
    o.calibration = Calibration::crt;
    BOOST_CHECK_THROW(score_scan(d.x, d.y, d.g, {}, o), std::exception); // continuous predictors
    o.calibration = Calibration::asymptotic;
    o.null_model = FitMethod::sceptre_null;
    BOOST_CHECK_THROW(score_scan(d.x, d.y, d.g, {}, o), std::exception);
    o.null_model = FitMethod::direct;
    BOOST_CHECK_THROW(score_scan(d.x, d.y, d.g, {{0, 9, "all"}}, o), std::exception);
    o.min_nonzero = 1000;
    auto r = score_scan(d.x, d.y, d.g, {{0, 0, "all"}, {0, 2, "all"}}, o);
    BOOST_TEST(r[0].status == "failed_pair_qc");
    BOOST_TEST(r[1].status == "ok"); // QC applies to binary predictors only
    // Significant-only effect fits equal the all-pairs fits for rejected pairs.
    o.min_nonzero = 0;
    o.effect_fits = EffectFits::all;
    auto all = score_scan(d.x, d.y, d.g, {}, o);
    o.effect_fits = EffectFits::significant;
    auto some = score_scan(d.x, d.y, d.g, {}, o);
    for (std::size_t k = 0; k < all.size(); ++k)
        if (some[k].adjusted_p <= o.alpha)
            BOOST_TEST(some[k].effect == all[k].effect);
        else
            BOOST_TEST(std::isnan(some[k].effect));
}
BOOST_AUTO_TEST_CASE(empirical_screen_stops_or_continues_at_the_first_budget) {
    auto d = data(5);
    auto null = fit(d.x, d.y.column(0), [] {
        FitOptions m;
        m.family = Family::poisson;
        m.retain_observations = true;
        return m;
    }());
    ScoreWorkspace score(d.x, null);
    ResampleOptions ro;
    ro.budgets = {499, 1999};
    ro.screen = 0.02;
    // A strong predictor passes the screen and is drawn to the second budget; a
    // null one stops with its empirical first-stage p-value.
    const auto strong = resample_permutation(score, d.g.column(2), {}, ro);
    BOOST_REQUIRE(strong.valid);
    BOOST_TEST(strong.draws == 1999u);
    const auto weak = resample_permutation(score, d.g.column(3), {}, ro);
    BOOST_REQUIRE(weak.valid);
    BOOST_TEST(weak.draws == 499u);
    BOOST_TEST(!weak.used_approximation);
    BOOST_TEST(weak.p_value > 0.02);
    ro.retain_null = true;
    const auto retained = resample_permutation(score, d.g.column(3), {}, ro);
    BOOST_TEST(weak.p_value ==
               empirical_tail(retained.null_statistics, weak.statistic, Side::both));
    ro.screen = 1.5;
    BOOST_CHECK_THROW(resample_permutation(score, d.g.column(3), {}, ro), std::exception);
}
