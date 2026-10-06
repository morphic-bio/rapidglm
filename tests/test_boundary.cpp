#include <boost/test/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>

using namespace rapidglm;
namespace {
Matrix paired_design(bool zero_reference = false, double donor_scale = 1) {
    Matrix x(16, 5, 1);
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = 1; j < 4; ++j)
            x(i, j) = donor_scale * (i / 4 == (zero_reference && j == 1 ? 0 : j));
        x(i, 4) = i % 4 >= 2;
    }
    return x;
}
const Vector counts{1, 3, 3, 5, 0, 0, 0, 0, 2, 4, 4, 8, 3, 5, 6, 10};
Vector limiting_means() {
    return {2, 2, 4, 4, 0, 0, 0, 0, 3, 3, 6, 6, 4, 4, 8, 8};
}
} // namespace

BOOST_AUTO_TEST_CASE(count_boundary_preserves_identifiable_treatment) {
    for (auto family : {Family::poisson, Family::negative_binomial})
        for (auto dispersion : {Dispersion::fixed, Dispersion::moment, Dispersion::ml}) {
            FitOptions o;
            o.family = family;
            o.dispersion = dispersion;
            o.theta = 10;
            o.robust = true;
            auto f = fit(paired_design(), counts, o);
            BOOST_TEST_CONTEXT("family=" << static_cast<int>(family)
                                         << " dispersion=" << static_cast<int>(dispersion)) {
                BOOST_REQUIRE(f.ok());
                BOOST_CHECK(f.status == FitStatus::converged_boundary);
                BOOST_TEST(f.boundary_coefficients[1] == 1u);
                BOOST_TEST(std::count(f.boundary_coefficients.begin(),
                                      f.boundary_coefficients.end(), 1) == 1);
                BOOST_TEST(f.iterations < 100u);
                BOOST_TEST(std::abs(f.coefficients[4] - std::log(2.0)) < 1e-9);
                auto expected = limiting_means();
                for (std::size_t i = 0; i < counts.size(); ++i)
                    BOOST_TEST(std::abs(f.fitted[i] - expected[i]) < 1e-8);
                BOOST_TEST(std::isnan(f.standard_errors[1]));
                BOOST_TEST(!wald_test(f, 1).valid);
                BOOST_TEST(!wald_test(f, 1, Side::both, true).valid);
                BOOST_TEST(wald_test(f, 4).valid);
                BOOST_TEST(wald_test(f, 4, Side::both, true).valid);
                BOOST_TEST(log2_fold_change(f, 4).estimate == 1,
                           boost::test_tools::tolerance(1e-9));
                for (std::size_t j = 0; j < f.parameters; ++j) {
                    BOOST_TEST(std::isnan(f.covariance(1, j)));
                    BOOST_TEST(std::isnan(f.robust_covariance(j, 1)));
                }
                if (family == Family::poisson)
                    BOOST_TEST(f.standard_errors[4] == std::sqrt(1.0 / 18 + 1.0 / 36),
                               boost::test_tools::tolerance(1e-9));
                else {
                    BOOST_TEST(std::isfinite(f.theta));
                    if (dispersion == Dispersion::ml)
                        BOOST_TEST(f.dispersion_iterations > 0u);
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(count_boundary_is_invariant_to_reference_and_units) {
    FitOptions o;
    o.family = Family::poisson;
    auto baseline = fit(paired_design(), counts, o);
    BOOST_REQUIRE(baseline.ok());
    for (bool reference : {false, true})
        for (double scale : {-1000.0, -0.001, 1.0, 1000.0}) {
            auto f = fit(paired_design(reference, scale), counts, o);
            BOOST_TEST_CONTEXT("reference=" << reference << " scale=" << scale) {
                BOOST_REQUIRE(f.ok());
                BOOST_CHECK(f.status == FitStatus::converged_boundary);
                for (std::size_t j = 0; j < 4; ++j) {
                    BOOST_TEST(static_cast<bool>(f.boundary_coefficients[j]) ==
                               (reference || j == 1));
                    if (f.boundary_coefficients[j])
                        BOOST_TEST(!wald_test(f, j).valid);
                }
                BOOST_TEST(f.boundary_coefficients[4] == 0u);
                BOOST_TEST(wald_test(f, 4).valid);
                BOOST_TEST(std::abs(f.coefficients[4] - baseline.coefficients[4]) < 1e-9);
                BOOST_TEST(std::abs(f.standard_errors[4] - baseline.standard_errors[4]) < 1e-9);
                BOOST_TEST(std::abs(f.log_likelihood - baseline.log_likelihood) < 1e-8);
            }
        }
}

BOOST_AUTO_TEST_CASE(count_boundary_inference_guards_and_iteration_budget) {
    FitOptions o;
    o.family = Family::poisson;
    auto x = paired_design();
    auto full = fit(x, counts, o);
    auto reduced = fit(remove_column(x, 4), counts, o);
    BOOST_REQUIRE(full.ok());
    BOOST_REQUIRE(reduced.ok());
    BOOST_TEST(!likelihood_ratio(reduced, full).valid);
    BOOST_CHECK_THROW(ScoreWorkspace(x, full), std::invalid_argument);
    o.max_iterations = 1;
    BOOST_CHECK(fit(x, counts, o).status == FitStatus::iteration_limit);
    o.max_iterations = 1000;
    auto replay = fit(x, counts, o);
    BOOST_TEST(replay.iterations == full.iterations);
    BOOST_TEST(replay.coefficients == full.coefficients, boost::test_tools::per_element());
    full.compact();
    BOOST_TEST(full.has_boundary());
    BOOST_TEST(!wald_test(full, 1).valid);
    BOOST_TEST(wald_test(full, 4).valid);
    SparseCSC response{counts.size(), 1, {0}, {}, {}};
    SparseCSC calls{counts.size(), 1, {0}, {}, {}};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] > 0) {
            response.row_index.push_back(i);
            response.values.push_back(counts[i]);
        }
        if (i % 4 >= 2) {
            calls.row_index.push_back(i);
            calls.values.push_back(1);
        }
    }
    response.column_ptr.push_back(response.values.size());
    calls.column_ptr.push_back(calls.values.size());
    auto treatment = regression_tests(x, response, o, 4);
    BOOST_TEST(treatment[0].test.valid);
    auto donor = regression_tests(x, response, o, 1);
    BOOST_TEST(!donor[0].test.valid);
    BOOST_TEST(donor[0].adjusted_p == 1);
    SceptreOptions options;
    options.model = o;
    options.null_model = FitMethod::direct;
    options.cell_qc = options.assignment_qc = options.nt_controls = false;
    options.min_nonzero = 0;
    auto result = sceptre(x, response, calls, {{"g", "target", false}},
                          {{0, "target", PairKind::discovery}}, {}, options);
    BOOST_REQUIRE(result.tests.size() == 1u);
    BOOST_TEST(result.tests[0].status == "null_converged_boundary");
    BOOST_TEST(std::isnan(result.tests[0].p_value));
    BOOST_TEST(result.tests[0].adjusted_p == 1);
}

BOOST_AUTO_TEST_CASE(count_boundary_controls_weights_and_ridge) {
    for (auto family : {Family::poisson, Family::negative_binomial}) {
        auto x = paired_design();
        Vector y = counts;
        y[4] = 1;
        FitOptions o;
        o.family = family;
        for (const Vector &start : {Vector{}, Vector(5, 0.1)}) {
            o.start = start;
            auto f = fit(x, y, o);
            BOOST_REQUIRE(f.ok());
            BOOST_TEST(!f.has_boundary());
            BOOST_TEST(wald_test(f, 1).valid);
        }
        o.start.clear();
        o.weights = Vector(16, 1);
        o.weights[4] = 0; // An excluded positive observation cannot remove separation.
        auto f = fit(x, y, o);
        BOOST_REQUIRE(f.ok());
        BOOST_TEST(f.boundary_coefficients[1] == 1u);
        o.weights.clear();
        o.ridge = 1;
        f = fit(x, counts, o);
        BOOST_REQUIRE(f.ok());
        BOOST_TEST(!f.has_boundary());
        BOOST_TEST(std::isfinite(f.standard_errors[1]));
    }
}

BOOST_AUTO_TEST_CASE(count_boundary_requires_a_recession_direction) {
    // Rank deficiency among positive counts alone does not establish separation:
    // increasing or decreasing the slope increases a zero-count fitted mean.
    Matrix x(4, 2, 1);
    x(0, 1) = x(1, 1) = 0;
    x(2, 1) = -1;
    x(3, 1) = 1;
    FitOptions o;
    o.family = Family::poisson;
    o.start = {1, 1};
    auto f = fit(x, {3, 5, 0, 0}, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(!f.has_boundary());
    BOOST_TEST(std::abs(f.coefficients[0] - std::log(2.0)) < 1e-9);
    BOOST_TEST(std::abs(f.coefficients[1]) < 1e-9);
    // Small means caused by offsets are also insufficient to flag a boundary.
    o.start.clear();
    o.offset = {0, 0, -40, -40};
    f = fit(Matrix(4, 1, 1), {3, 5, 0, 0}, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(!f.has_boundary());
    BOOST_TEST(wald_test(f, 0).valid);
}
