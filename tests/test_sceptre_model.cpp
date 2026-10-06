#include "calibration_data.hpp"
#include <boost/test/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;

BOOST_AUTO_TEST_CASE(sceptre_recipe_freezes_poisson_coefficients_and_uses_published_score) {
    auto data = rapidglm_test::NullGenerator(Family::negative_binomial, true, 7191).dataset(512);
    FitOptions options;
    options.offset.resize(512);
    options.weights.resize(512);
    for (std::size_t i = 0; i < 512; ++i) {
        options.offset[i] = 0.2 * std::cos(i);
        options.weights[i] = i % 13 == 0 ? 0 : 0.5 + i % 3;
    }
    auto poisson_options = options;
    poisson_options.family = Family::poisson;
    auto poisson = fit(data.design, data.response, poisson_options);
    auto compatibility = fit_sceptre_null(data.design, data.response, options);
    auto direct = fit(data.design, data.response, options);
    BOOST_REQUIRE(compatibility.ok());
    BOOST_REQUIRE(direct.ok());
    BOOST_CHECK(compatibility.method == FitMethod::sceptre_null);
    BOOST_CHECK(direct.method == FitMethod::direct);
    BOOST_TEST(compatibility.coefficients == poisson.coefficients,
               boost::test_tools::per_element());
    BOOST_TEST(compatibility.fitted == poisson.fitted, boost::test_tools::per_element());
    BOOST_TEST(std::abs(compatibility.coefficients[1] - direct.coefficients[1]) > 1e-4);
    BOOST_TEST(compatibility.covariance.rows == 0u);
    BOOST_TEST(std::isnan(compatibility.standard_errors[0]));
    BOOST_TEST(!wald_test(compatibility, 0).valid);
    auto alternative = fit(append_column(data.design, data.assignment), data.response, options);
    BOOST_REQUIRE(alternative.ok());
    BOOST_TEST(!likelihood_ratio(compatibility, alternative).valid);

    for (double multiplier : {0.99, 1.01}) {
        auto neighbor =
            detail::likelihood(data.response, poisson.fitted, options.weights,
                               Family::negative_binomial, compatibility.theta * multiplier);
        BOOST_TEST(compatibility.log_likelihood > neighbor);
    }
    // Independent two-column Schur complement, with the documented raw numerator.
    long double a = 0, b = 0, c = 0, c0 = 0, c1 = 0, u = 0, v = 0;
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < 512; ++i) {
        const long double mu = poisson.fitted[i], x = data.design(i, 1), g = data.assignment[i];
        const long double factor = 1 / (1 + mu / compatibility.theta);
        const long double weight = options.weights[i] * mu * factor;
        a += weight;
        b += weight * x;
        c += weight * x * x;
        c0 += weight * g;
        c1 += weight * x * g;
        u += options.weights[i] * g * (data.response[i] - mu) * factor;
        v += weight * g;
        if (g)
            selected.push_back(i);
    }
    const long double information =
        v - (c * c0 * c0 - 2 * b * c0 * c1 + a * c1 * c1) / (a * c - b * b);
    const double expected = static_cast<double>(u / std::sqrt(information));
    ScoreWorkspace score(data.design, compatibility);
    BOOST_TEST(!score.project_nuisance);
    BOOST_TEST(!score.test(data.assignment).valid);
    BOOST_TEST(std::abs(score.statistic(data.assignment) - expected) < 1e-12);
    BinaryScoreWorkspace binary(score);
    BOOST_TEST(std::abs(binary.statistic(selected) - expected) < 1e-12);
    score.project_nuisance = true;
    BOOST_TEST(std::abs(score.statistic(data.assignment) - expected) > 1e-4);
    score.project_nuisance = false;
    ResampleOptions ro;
    ro.budgets = {99};
    ro.approximate = false;
    ro.retain_null = true;
    auto baseline = resample(score, data.assignment, permutation_sampler(data.assignment), ro);
    BOOST_REQUIRE(baseline.valid);
    ro.parallel_draws = true;
    for (int threads : {2, 8, 16}) {
        ResampleResult result;
        parallel_for(1, threads, [&](auto) {
            result = resample(score, data.assignment, permutation_sampler(data.assignment), ro);
        });
        BOOST_TEST(result.null_statistics == baseline.null_statistics,
                   boost::test_tools::per_element());
        BOOST_TEST(result.p_value == baseline.p_value);
    }
}

BOOST_AUTO_TEST_CASE(sceptre_recipe_rejects_conflicting_options_and_preserves_boundaries) {
    Matrix x(24, 1, 1);
    Vector zero(24, 0), y(24);
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = (i % 5) * 3;
    FitOptions options;
    auto empty = fit_sceptre_null(x, zero, options);
    BOOST_CHECK(empty.status == FitStatus::boundary_response);
    BOOST_TEST(!empty.ok());
    Matrix group(24, 2, 1);
    for (std::size_t i = 0; i < y.size(); ++i) {
        group(i, 1) = i < 12;
        if (i < 12)
            y[i] = 0;
    }
    auto boundary = fit_sceptre_null(group, y, options);
    BOOST_REQUIRE(boundary.has_boundary());
    BOOST_CHECK_THROW(ScoreWorkspace(group, boundary), std::invalid_argument);
    options.retain_observations = false;
    auto compact = fit_sceptre_null(x, y, options);
    BOOST_REQUIRE(compact.ok());
    BOOST_TEST(compact.fitted.empty());
    BOOST_TEST(compact.score_residuals.empty());
    options.family = Family::poisson;
    BOOST_CHECK_THROW(fit_sceptre_null(x, y, options), std::invalid_argument);
    options.family = Family::negative_binomial;
    options.dispersion = Dispersion::fixed;
    BOOST_CHECK_THROW(fit_sceptre_null(x, y, options), std::invalid_argument);
    options.dispersion = Dispersion::ml;
    options.ridge = 0.1;
    BOOST_CHECK_THROW(fit_sceptre_null(x, y, options), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sceptre_recipe_routes_treatment_and_nt_subsets) {
    auto data = rapidglm_test::NullGenerator(Family::negative_binomial, true, 18219).dataset(192);
    SparseCSC responses{192, 1, {0, 192}, {}, data.response};
    SparseCSC assignments{192, 3, {0}, {}, {}};
    for (std::size_t i = 0; i < 192; ++i)
        responses.row_index.push_back(i);
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = j; i < 192; i += 3) {
            assignments.row_index.push_back(i);
            assignments.values.push_back(1);
        }
        assignments.column_ptr.push_back(assignments.values.size());
    }
    std::vector<Guide> guides{{"a", "A", false}, {"b", "B", false}, {"nt", "NT", true}};
    std::vector<Pair> pairs{{0, "A"}, {0, "B"}};
    SceptreOptions options;
    options.cell_qc = false;
    options.min_nonzero = 1;
    options.null_model = FitMethod::sceptre_null;
    options.effect_fits = EffectFits::all;
    options.resampling.budgets = {99};
    options.resampling.approximate = false;
    auto baseline = sceptre(data.design, responses, assignments, guides, pairs, {}, options);
    for (std::size_t target = 0; target < 2; ++target) {
        std::vector<std::size_t> cells, selected;
        for (std::size_t i = 0; i < 192; ++i)
            if (i % 3 == target || i % 3 == 2) {
                if (i % 3 == target)
                    selected.push_back(cells.size());
                cells.push_back(i);
            }
        auto x = subset_rows(data.design, cells);
        auto fit = fit_sceptre_null(x, subset(data.response, cells));
        auto score = BinaryScoreWorkspace(ScoreWorkspace(x, fit));
        BOOST_REQUIRE(baseline.tests[target].status == "ok");
        BOOST_TEST(baseline.tests[target].cells == 128u);
        BOOST_TEST(baseline.tests[target].statistic == score.statistic(selected));
    }
    options.threads = 16;
    auto parallel = sceptre(data.design, responses, assignments, guides, pairs, {}, options);
    for (std::size_t i = 0; i < 2; ++i) {
        BOOST_TEST(parallel.tests[i].statistic == baseline.tests[i].statistic);
        BOOST_TEST(parallel.tests[i].p_value == baseline.tests[i].p_value);
        BOOST_TEST(parallel.tests[i].log2fc == baseline.tests[i].log2fc);
    }
}
