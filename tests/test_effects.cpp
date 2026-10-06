#include "calibration_data.hpp"
#include <boost/test/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;

BOOST_AUTO_TEST_CASE(effect_warm_starts_preserve_full_fit_and_uncertainty) {
    for (auto family : {Family::gaussian, Family::poisson, Family::negative_binomial})
        for (auto dispersion : {Dispersion::fixed, Dispersion::moment, Dispersion::ml})
            for (unsigned seed = 0; seed < 4; ++seed) {
                auto data = rapidglm_test::NullGenerator(family, true, 8100 + seed).dataset(384);
                FitOptions options;
                options.family = family;
                options.dispersion = dispersion;
                options.robust = true;
                options.offset.resize(384);
                options.weights.resize(384);
                for (std::size_t i = 0; i < data.response.size(); ++i) {
                    options.offset[i] = 0.1 * std::sin(i);
                    options.weights[i] = i % 17 == 0 ? 0 : 0.5 + (i % 3);
                    if (seed % 2 && data.assignment[i])
                        data.response[i] *= 4;
                }
                const auto null = fit(data.design, data.response, options);
                BOOST_REQUIRE(null.ok());
                const auto cold = fit_effect_model(data.design, data.response, data.assignment,
                                                   null, options, false);
                const auto warm = fit_effect_model(data.design, data.response, data.assignment,
                                                   null, options, true);
                BOOST_TEST_CONTEXT("family=" << int(family) << " dispersion=" << int(dispersion)
                                             << " seed=" << seed) {
                    BOOST_REQUIRE(cold.ok());
                    BOOST_REQUIRE(warm.ok());
                    BOOST_CHECK(cold.status == warm.status);
                    BOOST_TEST(std::abs(cold.log_likelihood - warm.log_likelihood) < 1e-7);
                    for (std::size_t k = 0; k < cold.coefficients.size(); ++k) {
                        BOOST_TEST(std::abs(cold.coefficients[k] - warm.coefficients[k]) < 1e-6);
                        BOOST_TEST(std::abs(cold.standard_errors[k] - warm.standard_errors[k]) <
                                   1e-6);
                    }
                    for (std::size_t k = 0; k < cold.robust_covariance.values.size(); ++k)
                        BOOST_TEST(std::abs(cold.robust_covariance.values[k] -
                                            warm.robust_covariance.values[k]) < 1e-6);
                    if (family == Family::negative_binomial)
                        BOOST_TEST(std::abs(std::log(cold.theta / warm.theta)) < 1e-6);
                }
            }
}

BOOST_AUTO_TEST_CASE(effect_boundary_falls_back_to_established_fit) {
    Matrix x(80, 1, 1);
    Vector y(80), g(80);
    for (std::size_t i = 0; i < y.size(); ++i) {
        g[i] = i < 40;
        y[i] = g[i] ? 0 : (i % 4) * 5;
    }
    for (auto family : {Family::poisson, Family::negative_binomial}) {
        FitOptions options;
        options.family = family;
        const auto null = fit(x, y, options);
        BOOST_REQUIRE(null.ok());
        const auto cold = fit_effect_model(x, y, g, null, options, false);
        const auto warm = fit_effect_model(x, y, g, null, options, true);
        BOOST_REQUIRE(cold.has_boundary());
        BOOST_CHECK(warm.status == cold.status);
        BOOST_TEST(warm.coefficients == cold.coefficients, boost::test_tools::per_element());
        BOOST_TEST(warm.boundary_coefficients == cold.boundary_coefficients,
                   boost::test_tools::per_element());
        BOOST_TEST(std::isnan(warm.standard_errors.back()));
    }
}

BOOST_AUTO_TEST_CASE(failed_effect_warm_start_recovers_fresh_fit) {
    Matrix x(80, 1, 1);
    Vector y(80), g(80);
    for (std::size_t i = 0; i < y.size(); ++i) {
        g[i] = i % 3 == 0;
        y[i] = (i % 7) * 3;
    }
    FitOptions options;
    auto seed = fit(x, y, options);
    BOOST_REQUIRE(seed.ok());
    const auto cold = fit_effect_model(x, y, g, seed, options, false);
    BOOST_REQUIRE(cold.ok());
    // Force a finite but unusable initialization, without changing the model.
    seed.coefficients[0] = 1000;
    const auto recovered = fit_effect_model(x, y, g, seed, options, true);
    BOOST_REQUIRE(recovered.ok());
    BOOST_TEST(recovered.coefficients == cold.coefficients, boost::test_tools::per_element());
    BOOST_TEST(recovered.standard_errors == cold.standard_errors, boost::test_tools::per_element());
    options.nb_warm_start = true;
    BOOST_CHECK_THROW(fit(x, y, options), std::invalid_argument);
    options.start = {1};
    options.dispersion = Dispersion::fixed;
    BOOST_CHECK_THROW(fit(x, y, options), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(pair_tasks_preserve_workflow_results_and_propagate_exceptions) {
    const std::size_t n = 160;
    Matrix x(n, 1, 1);
    SparseCSC responses{n, 1, {0, n}, {}, {}}, assignments{n, 4, {0}, {}, {}};
    for (std::size_t i = 0; i < n; ++i) {
        responses.row_index.push_back(i);
        responses.values.push_back((i % 7) * 2);
    }
    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t i = j; i < n; i += 4) {
            assignments.row_index.push_back(i);
            assignments.values.push_back(1);
        }
        assignments.column_ptr.push_back(assignments.values.size());
    }
    std::vector<Guide> guides{
        {"g0", "t0", false}, {"g1", "t1", false}, {"g2", "t2", false}, {"g3", "NT", true}};
    std::vector<Pair> pairs{{0, "t0"}, {0, "t1"}, {0, "t2"}};
    for (bool nt_controls : {false, true})
        for (bool crt : {false, true}) {
            SceptreOptions options;
            options.nt_controls = nt_controls;
            options.crt = crt;
            options.cell_qc = false;
            options.parallel_pairs = false;
            options.warm_start_effects = false;
            options.effect_fits = EffectFits::all;
            options.resampling.budgets = {31, 99};
            options.resampling.approximate = false;
            auto baseline = sceptre(x, responses, assignments, guides, pairs, {}, options);
            for (int threads : {1, 2, 8, 16}) {
                options.threads = threads;
                options.parallel_pairs = true;
                for (bool warm : {false, true}) {
                    options.warm_start_effects = warm;
                    auto result = sceptre(x, responses, assignments, guides, pairs, {}, options);
                    BOOST_REQUIRE(result.tests.size() == 3u);
                    for (std::size_t i = 0; i < 3; ++i) {
                        const auto &a = baseline.tests[i], &b = result.tests[i];
                        BOOST_REQUIRE(a.status == "ok");
                        BOOST_TEST(b.status == a.status);
                        BOOST_TEST(b.statistic == a.statistic);
                        BOOST_TEST(b.p_value == a.p_value);
                        BOOST_TEST(b.adjusted_p == a.adjusted_p);
                        BOOST_TEST(b.draws == a.draws);
                        if (warm) {
                            BOOST_TEST(std::abs(b.log2fc - a.log2fc) < 1e-6);
                            BOOST_TEST(std::abs(b.log2fc_se - a.log2fc_se) < 1e-6);
                        } else {
                            BOOST_TEST(b.log2fc == a.log2fc);
                            BOOST_TEST(b.log2fc_se == a.log2fc_se);
                        }
                    }
                }
            }
        }
    BOOST_CHECK_THROW(parallel_for(1, 8,
                                   [&](auto) {
                                       task_for(9, true, [](auto i) {
                                           if (i == 4)
                                               throw std::invalid_argument("pair failure");
                                       });
                                   }),
                      std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(deferred_effect_fits_match_all_mode_for_rejected_pairs) {
    const std::size_t n = 200;
    Matrix x(n, 2, 1);
    SparseCSC responses{n, 2, {0}, {}, {}}, assignments{n, 5, {0}, {}, {}};
    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            double value = (i % 7) * 2 + 1;
            // Guides g0 and g1 (target t0) knock down the first response.
            if (j == 0 && i % 5 < 2)
                value = std::floor(value / 4);
            if (value != 0) {
                responses.row_index.push_back(i);
                responses.values.push_back(value);
            }
        }
        responses.column_ptr.push_back(responses.values.size());
    }
    for (std::size_t i = 0; i < n; ++i)
        x(i, 1) = std::log(2 + i % 3);
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = j; i < n; i += 5) {
            assignments.row_index.push_back(i);
            assignments.values.push_back(1);
        }
        assignments.column_ptr.push_back(assignments.values.size());
    }
    std::vector<Guide> guides{{"g0", "t0", false},
                              {"g1", "t0", false},
                              {"g2", "t1", false},
                              {"g3", "t2", false},
                              {"g4", "NT", true}};
    std::vector<Pair> pairs{{0, "t0"}, {0, "t1"}, {0, "t2"}, {1, "t0"}};
    std::size_t rejected = 0;
    for (auto integration :
         {Integration::union_guides, Integration::singleton, Integration::bonferroni})
        for (bool nt_controls : {false, true}) {
            SceptreOptions options;
            options.integration = integration;
            options.nt_controls = nt_controls;
            options.crt = !nt_controls;
            options.cell_qc = false;
            options.min_nonzero = 1;
            options.resampling.budgets = {99, 199};
            options.effect_fits = EffectFits::all;
            const auto all = sceptre(x, responses, assignments, guides, pairs, {}, options);
            for (auto mode : {EffectFits::significant, EffectFits::none})
                for (int threads : {1, 8}) {
                    options.effect_fits = mode;
                    options.threads = threads;
                    const auto deferred =
                        sceptre(x, responses, assignments, guides, pairs, {}, options);
                    BOOST_REQUIRE(deferred.tests.size() == all.tests.size());
                    for (std::size_t i = 0; i < all.tests.size(); ++i) {
                        const auto &a = all.tests[i], &b = deferred.tests[i];
                        BOOST_TEST(b.p_value == a.p_value);
                        BOOST_TEST(b.adjusted_p == a.adjusted_p);
                        BOOST_TEST(b.sceptre_mean_ratio == a.sceptre_mean_ratio);
                        const bool fitted = mode == EffectFits::significant &&
                                            a.status.rfind("ok", 0) == 0 &&
                                            a.adjusted_p <= options.alpha;
                        if (fitted) {
                            rejected += mode == EffectFits::significant && threads == 1;
                            BOOST_TEST(b.status == a.status);
                            BOOST_TEST(b.log2fc == a.log2fc);
                            BOOST_TEST(b.log2fc_se == a.log2fc_se);
                        } else {
                            BOOST_TEST(std::isnan(b.log2fc));
                            BOOST_TEST(std::isnan(b.log2fc_se));
                            BOOST_TEST(b.status ==
                                       (a.status.rfind("ok", 0) == 0 ? "ok" : a.status));
                        }
                    }
                }
        }
    // The design must exercise both fitted and skipped rows.
    BOOST_TEST(rejected > 0u);
    BOOST_TEST(rejected < 6u * 4u);
}
