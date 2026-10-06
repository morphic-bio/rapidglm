#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
// Poisson data with an intercept, two covariates and a third column x1 + delta z.
std::pair<Matrix, Vector> collinear(double delta, std::size_t n = 600) {
    std::mt19937_64 rng(4242);
    std::normal_distribution<double> normal;
    Matrix x(n, 4, 1);
    Vector y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = normal(rng);
        x(i, 2) = normal(rng);
        x(i, 3) = x(i, 1) + delta * normal(rng);
        y[i] = std::poisson_distribution<int>(std::exp(0.4 + 0.3 * x(i, 1) - 0.2 * x(i, 2)))(rng);
    }
    return {x, y};
}
} // namespace
BOOST_AUTO_TEST_CASE(crossproduct_matches_direct_sums_for_any_row_count) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> normal;
    for (std::size_t n : {1u, 3u, 4u, 5u, 11u})
        for (std::size_t p : {1u, 2u, 7u}) {
            Matrix x(n, p);
            Vector w(n);
            for (auto &v : x.values)
                v = normal(rng);
            for (auto &v : w)
                v = std::abs(normal(rng));
            auto a = detail::crossprod(x, w);
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = 0; k < p; ++k) {
                    double expected = 0;
                    for (std::size_t i = 0; i < n; ++i)
                        expected += w[i] * x(i, j) * x(i, k);
                    BOOST_TEST(a(j, k) == expected, boost::test_tools::tolerance(1e-13));
                }
        }
}
BOOST_AUTO_TEST_CASE(aliased_columns_follow_r_threshold_and_singular_ok) {
    FitOptions o;
    o.family = Family::poisson;
    // Relative residual about 1e-12 is aliased; about 3e-11 is kept (R's glm default).
    for (auto [delta, aliased] : {std::pair{1e-12, true}, std::pair{3e-11, false}}) {
        auto [x, y] = collinear(delta);
        auto strict = fit(x, y, o);
        BOOST_TEST((strict.status == FitStatus::rank_deficient) == aliased);
        o.singular_ok = true;
        auto r = fit(x, y, o);
        o.singular_ok = false;
        BOOST_REQUIRE(r.ok());
        BOOST_TEST(r.rank == (aliased ? 3u : 4u));
        BOOST_TEST(r.parameters == 4u);
        BOOST_TEST(static_cast<bool>(r.aliased[3]) == aliased);
        BOOST_TEST(std::isnan(r.coefficients[3]) == aliased);
        BOOST_TEST(std::isnan(r.standard_errors[3]) == aliased);
        if (aliased) {
            // Identical to fitting the retained columns.
            Matrix reduced(x.rows, 3);
            for (std::size_t i = 0; i < x.rows; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    reduced(i, j) = x(i, j);
            auto ref = fit(reduced, y, o);
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_TEST(r.coefficients[j] == ref.coefficients[j]);
                BOOST_TEST(r.standard_errors[j] == ref.standard_errors[j]);
            }
            BOOST_TEST(!wald_test(r, 3).valid);
            BOOST_TEST(wald_test(r, 1).valid);
        }
    }
}
BOOST_AUTO_TEST_CASE(likelihood_ratio_degrees_of_freedom_use_rank) {
    auto [x, y] = collinear(0);
    FitOptions o;
    o.family = Family::poisson;
    o.singular_ok = true;
    auto full = fit(x, y, o); // column 3 duplicates column 1
    BOOST_REQUIRE(full.rank == 3u);
    Matrix null(x.rows, 2);
    for (std::size_t i = 0; i < x.rows; ++i) {
        null(i, 0) = 1;
        null(i, 1) = x(i, 1);
    }
    auto reduced = fit(null, y, o);
    auto lr = likelihood_ratio(reduced, full);
    BOOST_REQUIRE(lr.valid);
    boost::math::chi_squared_distribution<double> one(1);
    BOOST_TEST(lr.p_value == boost::math::cdf(boost::math::complement(one, lr.statistic)),
               boost::test_tools::tolerance(1e-12));
}
BOOST_AUTO_TEST_CASE(qr_solver_agrees_and_stabilizes_ill_conditioned_covariance) {
    for (double delta : {1.0, 1e-7}) {
        auto [x, y] = collinear(delta);
        FitOptions o;
        o.family = Family::poisson;
        auto automatic = fit(x, y, o);
        o.solver = Solver::qr;
        auto qr = fit(x, y, o);
        BOOST_REQUIRE(automatic.ok());
        BOOST_REQUIRE(qr.ok());
        BOOST_TEST(qr.used_qr);
        // The ill-conditioned design uses QR for its covariance automatically.
        BOOST_TEST(automatic.used_qr == (delta < 1));
        for (std::size_t j = 0; j < 4; ++j) {
            BOOST_TEST(automatic.coefficients[j] == qr.coefficients[j],
                       boost::test_tools::tolerance(delta < 1 ? 1e-6 : 1e-10));
            BOOST_TEST(automatic.standard_errors[j] == qr.standard_errors[j],
                       boost::test_tools::tolerance(1e-9));
        }
    }
}
BOOST_AUTO_TEST_CASE(row_groups_detect_layouts_with_as_many_distinct_rows_as_columns) {
    Matrix intercept(7, 1, 1);
    BOOST_TEST(detail::RowGroups(intercept).ok);
    // Full factorial 2 x 2 with interaction: four distinct rows, four columns.
    Matrix factorial(12, 4, 1);
    for (std::size_t i = 0; i < 12; ++i) {
        factorial(i, 1) = i % 2;
        factorial(i, 2) = (i / 2) % 2;
        factorial(i, 3) = factorial(i, 1) * factorial(i, 2);
    }
    detail::RowGroups groups(factorial);
    BOOST_REQUIRE(groups.ok);
    for (std::size_t i = 0; i < 12; ++i)
        BOOST_TEST(groups.group[i] == groups.group[i % 4]);
    // U^-1 inverts the distinct rows.
    auto product = detail::matvec(groups.rows, detail::matvec(groups.inverse, {1, 2, 3, 4}));
    for (std::size_t k = 0; k < 4; ++k)
        BOOST_TEST(product[k] == k + 1.0, boost::test_tools::tolerance(1e-12));
    // Main effects only: four distinct rows but three columns.
    Matrix additive(12, 3);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            additive(i, j) = factorial(i, j);
    BOOST_TEST(!detail::RowGroups(additive).ok);
    Matrix continuous(5, 2, 1);
    for (std::size_t i = 0; i < 5; ++i)
        continuous(i, 1) = 0.1 * i;
    BOOST_TEST(!detail::RowGroups(continuous).ok);
    // Two distinct rows that do not span two columns.
    Matrix singular(6, 2, 0);
    for (std::size_t i = 0; i < 6; ++i)
        singular(i, 0) = singular(i, 1) = i % 2;
    BOOST_TEST(!detail::RowGroups(singular).ok);
}
BOOST_AUTO_TEST_CASE(grouped_fits_match_general_fits) {
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> uniform;
    const std::size_t n = 90;
    Matrix x(n, 3, 1);
    Vector offset(n), weights(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = i % 3 == 1;
        x(i, 2) = i % 3 == 2;
        offset[i] = std::log(0.5 + uniform(rng));
        weights[i] = i % 7 == 0 ? 0 : 0.5 + uniform(rng);
    }
    for (int boundary = 0; boundary < 2; ++boundary) {
        Vector y(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double mu = std::exp(1 + 0.4 * x(i, 1) - 0.3 * x(i, 2) + offset[i]);
            std::gamma_distribution<double> gamma(3, mu / 3);
            y[i] = boundary && i % 3 == 2 ? 0 : std::poisson_distribution<int>(gamma(rng))(rng);
        }
        for (auto family : {Family::poisson, Family::negative_binomial})
            for (auto dispersion : {Dispersion::ml, Dispersion::fixed}) {
                if (family == Family::poisson && dispersion == Dispersion::fixed)
                    continue;
                FitOptions o;
                o.family = family;
                o.dispersion = dispersion;
                o.theta = 3;
                o.offset = offset;
                o.weights = weights;
                auto grouped = fit(x, y, o);
                o.solver = Solver::qr;
                auto general = fit(x, y, o);
                BOOST_REQUIRE(grouped.ok());
                BOOST_REQUIRE(general.ok());
                BOOST_TEST(grouped.has_boundary() == bool(boundary));
                BOOST_TEST(grouped.boundary_coefficients == general.boundary_coefficients);
                BOOST_TEST(grouped.log_likelihood == general.log_likelihood,
                           boost::test_tools::tolerance(1e-10));
                for (std::size_t j = 0; j < 3; ++j)
                    if (!grouped.boundary_coefficients[j]) {
                        BOOST_TEST(grouped.coefficients[j] == general.coefficients[j],
                                   boost::test_tools::tolerance(1e-9));
                        BOOST_TEST(grouped.standard_errors[j] == general.standard_errors[j],
                                   boost::test_tools::tolerance(1e-8));
                    }
                if (family == Family::negative_binomial)
                    BOOST_TEST(grouped.theta == general.theta, boost::test_tools::tolerance(1e-7));
                // A grouped Poisson fit starts at its closed-form MLE.
                if (family == Family::poisson && !boundary)
                    BOOST_TEST(grouped.iterations == 1u);
            }
    }
}
BOOST_AUTO_TEST_CASE(joint_nb_newton_reaches_the_profile_maximum) {
    std::mt19937_64 rng(31);
    std::normal_distribution<double> normal;
    const std::size_t n = 200;
    Matrix x(n, 3, 1);
    Vector y(n), offset(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = normal(rng);
        x(i, 2) = i % 2;
        offset[i] = 0.3 * normal(rng);
        const double mu = std::exp(1.5 + 0.4 * x(i, 1) - 0.5 * x(i, 2) + offset[i]);
        std::gamma_distribution<double> gamma(2.5, mu / 2.5);
        y[i] = std::poisson_distribution<int>(gamma(rng))(rng);
    }
    FitOptions o;
    o.offset = offset;
    auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(!f.theta_boundary);
    // Stationary in beta and theta.
    Vector a(n, 1), s(n);
    for (std::size_t i = 0; i < n; ++i)
        s[i] = (y[i] - f.fitted[i]) / (1 + f.fitted[i] / f.theta);
    for (double v : detail::xtv(x, s))
        BOOST_TEST(std::abs(v) < 1e-6);
    const detail::CountTerms terms(y, a);
    const auto ll = [&](double t) {
        return detail::likelihood(y, f.fitted, a, Family::negative_binomial, t, terms);
    };
    const double h = 1e-5;
    BOOST_TEST(std::abs(ll(f.theta * std::exp(h)) - ll(f.theta * std::exp(-h))) / (2 * h) < 1e-6);
    // The profile likelihood over fixed-theta fits peaks at the joint estimate.
    auto fixed = o;
    fixed.dispersion = Dispersion::fixed;
    double best = -std::numeric_limits<double>::infinity();
    for (double factor : {0.9, 0.99, 1.0, 1.01, 1.1}) {
        fixed.theta = f.theta * factor;
        best = std::max(best, fit(x, y, fixed).log_likelihood);
    }
    BOOST_TEST(best <= f.log_likelihood + 1e-9);
    // Extreme starting dispersions reach the same optimum.
    for (double start : {1e-4, 1e6}) {
        auto warm = o;
        warm.nb_warm_start = true;
        warm.start = f.coefficients;
        warm.theta = start;
        auto g = fit(x, y, warm);
        BOOST_REQUIRE(g.ok());
        BOOST_TEST(g.theta == f.theta, boost::test_tools::tolerance(1e-6));
        BOOST_TEST(g.log_likelihood == f.log_likelihood, boost::test_tools::tolerance(1e-12));
    }
}
BOOST_AUTO_TEST_CASE(joint_nb_newton_converges_at_the_poisson_bound_with_large_counts) {
    // Underdispersed large counts: the moment estimate and the MLE sit at theta_max,
    // where lgamma differences for counts above 64 carry rounding noise.
    Matrix x(16, 2, 1);
    Vector y = {70, 72, 68, 71, 69, 70, 73, 67, 90, 91, 89, 92, 88, 90, 91, 89};
    for (std::size_t i = 0; i < 8; ++i)
        x(i, 1) = 0;
    auto f = fit(x, y);
    BOOST_REQUIRE(f.status == FitStatus::converged);
    BOOST_TEST(f.theta_boundary);
    FitOptions poisson;
    poisson.family = Family::poisson;
    auto p = fit(x, y, poisson);
    for (std::size_t j = 0; j < 2; ++j)
        BOOST_TEST(f.coefficients[j] == p.coefficients[j], boost::test_tools::tolerance(1e-6));
}
