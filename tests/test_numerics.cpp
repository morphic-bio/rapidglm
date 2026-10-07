#include <array>
#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>

using namespace rapidglm;
namespace {
const std::array<Family, 4> families = {Family::gaussian, Family::poisson, Family::binomial,
                                        Family::negative_binomial};

void near(double actual, double expected, double tolerance) {
    BOOST_TEST(std::isfinite(actual));
    BOOST_TEST(std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)));
}

// Independent probability-space oracle: NB masses use the P(Y=k)/P(Y=k-1)
// recurrence, without lgamma, digamma, or the production likelihood routine.
long double reference_log_mass(Family family, double y, long double mu, long double theta) {
    if (family == Family::gaussian)
        return -(y - mu) * (y - mu) / 2; // Fitting kernel, before profiling scale.
    if (family == Family::binomial)
        return std::log(std::pow(mu, y) * std::pow(1 - mu, 1 - y));
    long double p =
        family == Family::poisson ? std::exp(-mu) : std::pow(theta / (theta + mu), theta);
    for (unsigned k = 1; k <= static_cast<unsigned>(y); ++k)
        p *= family == Family::poisson ? mu / k : (theta + k - 1) / k * mu / (theta + mu);
    return std::log(p);
}
long double reference_link(Family family, long double eta) {
    return family == Family::gaussian   ? eta
           : family == Family::binomial ? 1 / (1 + std::exp(-eta))
                                        : std::exp(eta);
}
long double reference_nb(const Vector &y, const Vector &mu, const Vector &a, long double u) {
    long double ll = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (a[i] > 0)
            ll += a[i] * reference_log_mass(Family::negative_binomial, y[i], mu[i], std::exp(u));
    return ll;
}
double reference_theta(const Vector &y, const Vector &mu, const Vector &a, double lower,
                       double upper) {
    // Derivative-free bracket search is independent of the production Newton update.
    long double lo = std::log(lower), hi = std::log(upper);
    const long double ratio = (std::sqrt(5.L) - 1) / 2;
    for (unsigned iteration = 0; iteration < 100; ++iteration) {
        long double left = hi - ratio * (hi - lo), right = lo + ratio * (hi - lo);
        if (reference_nb(y, mu, a, left) < reference_nb(y, mu, a, right))
            lo = left;
        else
            hi = right;
    }
    return static_cast<double>(std::exp((lo + hi) / 2));
}
} // namespace

BOOST_AUTO_TEST_CASE(cholesky_known_solution_inverse_and_scaling) {
    const double base[3][3] = {{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
    const double adjugate[3][3] = {{14, -6, 2}, {-6, 12, -4}, {2, -4, 16}};
    for (const Vector &scale : {Vector{1, 1, 1}, Vector{1e-6, 1, 1e6}}) {
        Matrix a(3, 3);
        Vector expected{1, -2, 3}, b(3);
        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j) {
                a(i, j) = scale[i] * base[i][j] * scale[j];
                b[i] += scale[i] * base[i][j] * expected[j];
            }
        detail::Cholesky solver(a);
        BOOST_REQUIRE(solver.ok);
        auto solution = solver.solve(b);
        auto inverse = solver.inverse();
        for (unsigned i = 0; i < 3; ++i) {
            near(solution[i] * scale[i], expected[i], 2e-12);
            for (unsigned j = 0; j < 3; ++j)
                near(inverse(i, j) * scale[i] * scale[j], adjugate[i][j] / 44, 2e-12);
        }
    }
}

BOOST_AUTO_TEST_CASE(cholesky_rejects_singular_indefinite_and_invalid_systems) {
    Matrix a(2, 2, 1);
    BOOST_TEST(!detail::Cholesky(a).ok);
    a(0, 1) = a(1, 0) = 2;
    BOOST_TEST(!detail::Cholesky(a).ok);
    a(0, 1) = a(1, 0) = 1 - 1e-14;
    BOOST_TEST(!detail::Cholesky(a).ok);
    a(0, 1) = a(1, 0) = 0;
    auto solver = detail::Cholesky(a);
    BOOST_REQUIRE(solver.ok);
    BOOST_CHECK_THROW(solver.solve({1}), std::invalid_argument);
    a(0, 0) = std::numeric_limits<double>::infinity();
    BOOST_TEST(!detail::Cholesky(a).ok);
    BOOST_TEST(!detail::Cholesky(Matrix(2, 3, 1)).ok);
}

BOOST_AUTO_TEST_CASE(cholesky_large_system_residuals_and_triangle_contract) {
    // Larger than LAPACK's typical block size, with independently formed b=A*x.
    const std::size_t p = 129;
    Matrix a(p, p);
    Vector expected(p), b(p);
    for (std::size_t i = 0; i < p; ++i)
        expected[i] = std::cos(static_cast<double>(i));
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            a(i, j) = std::pow(0.8, static_cast<double>(i > j ? i - j : j - i));
            b[i] += a(i, j) * expected[j];
        }
    auto original = a;
    // The lower triangle defines the symmetric matrix; upper storage is unused.
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = i + 1; j < p; ++j)
            a(i, j) = rapidglm::nan;
    detail::Cholesky solver(a);
    BOOST_REQUIRE(solver.ok);
    auto solution = solver.solve(b);
    auto inverse = solver.inverse();
    for (std::size_t i = 0; i < p; ++i) {
        near(solution[i], expected[i], 2e-12);
        for (std::size_t j = 0; j < p; ++j) {
            double product = 0;
            for (std::size_t k = 0; k < p; ++k)
                product += original(i, k) * inverse(k, j);
            near(product, i == j ? 1 : 0, 2e-12);
        }
    }
    a(1, 0) = rapidglm::nan;
    BOOST_TEST(!detail::Cholesky(a).ok);
    a.values.clear();
    BOOST_TEST(!detail::Cholesky(a).ok);
    detail::Cholesky empty{Matrix(0, 0)};
    BOOST_REQUIRE(empty.ok);
    BOOST_TEST(empty.solve({}).empty());
    BOOST_TEST(empty.inverse().values.empty());
}

BOOST_AUTO_TEST_CASE(qr_large_system_solution_and_inverse) {
    // More columns than the stack-scratch order and LAPACK's small block sizes:
    // the triangular solve and (R'R)^-1 against an independently formed system.
    const std::size_t n = 400, q = 65;
    Matrix x(n, q);
    Vector w(n), expected(q), u(n);
    for (std::size_t k = 0; k < q; ++k)
        expected[k] = std::cos(static_cast<double>(k));
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = 0.5 + static_cast<double>(i % 7) / 4;
        double fitted = 0;
        for (std::size_t k = 0; k < q; ++k) {
            x(i, k) = k == 0 ? 1
                             : std::sin(0.37 * static_cast<double>((i + 1) * (k + 1))) +
                                   0.1 * static_cast<double>((i * k) % 5);
            fitted += x(i, k) * expected[k];
        }
        u[i] = std::sqrt(w[i]) * fitted;
    }
    detail::WeightedQR qr(x, w, 1e-11);
    BOOST_REQUIRE(qr.ok);
    BOOST_REQUIRE(qr.rank() == q);
    BOOST_TEST(qr.rcond() > 0);
    const auto solution = qr.solve(u);
    const auto inverse = qr.inverse();
    for (std::size_t j = 0; j < q; ++j) {
        near(solution[j], expected[j], 1e-9);
        for (std::size_t k = 0; k < q; ++k) {
            // (X'WX) (R'R)^-1 = I.
            double product = 0;
            for (std::size_t l = 0; l < q; ++l) {
                double information = 0;
                for (std::size_t i = 0; i < n; ++i)
                    information += w[i] * x(i, j) * x(i, l);
                product += information * inverse(l, k);
            }
            near(product, j == k ? 1 : 0, 1e-8);
        }
    }
}

BOOST_AUTO_TEST_CASE(cholesky_concurrent_independent_systems) {
    // Exercise the linked library's reentrancy, not just OpenMP scheduling.
    // Different matrices must not share LAPACK/BLAS scratch buffers.
    const std::size_t count = 1024;
    for (int threads : {1, 2, 8, 16}) {
        Vector errors(count, std::numeric_limits<double>::infinity());
        parallel_for(count, threads, [&](std::size_t job) {
            const std::size_t p = std::array<std::size_t, 3>{4, 9, 32}[job % 3];
            Matrix a(p, p);
            Vector expected(p), b(p);
            for (std::size_t i = 0; i < p; ++i)
                expected[i] = std::sin(static_cast<double>(job + i + 1));
            for (std::size_t i = 0; i < p; ++i)
                for (std::size_t j = 0; j < p; ++j) {
                    a(i, j) = i == j ? p + 2 : 0.01 * std::cos(static_cast<double>(job + i + j));
                    b[i] += a(i, j) * expected[j];
                }
            detail::Cholesky ch(a);
            if (!ch.ok)
                return;
            auto solution = ch.solve(b);
            auto inverse = ch.inverse();
            double error = 0;
            for (std::size_t i = 0; i < p; ++i) {
                if (!std::isfinite(solution[i]))
                    return;
                error = std::max(error, std::abs(solution[i] - expected[i]));
                for (std::size_t j = 0; j < p; ++j) {
                    double product = 0;
                    for (std::size_t k = 0; k < p; ++k)
                        product += a(i, k) * inverse(k, j);
                    if (!std::isfinite(product))
                        return;
                    error = std::max(error, std::abs(product - (i == j ? 1 : 0)));
                }
            }
            errors[job] = error;
        });
        BOOST_TEST_CONTEXT("threads=" << threads) {
            double worst = *std::max_element(errors.begin(), errors.end());
            BOOST_TEST_MESSAGE("Concurrent Cholesky: threads=" << threads
                                                               << " max_error=" << worst);
            BOOST_TEST(worst < 1e-11);
        }
    }
}

BOOST_AUTO_TEST_CASE(likelihood_matches_independent_probability_masses) {
    for (auto family : families)
        for (double theta : {0.2, 2.0, 20.0}) {
            Vector y =
                family == Family::binomial ? Vector{0, 0.25, 1, 0.8, 1} : Vector{0, 1, 4, 80, 3};
            Vector eta = {-1, 0.2, 1, 2, 0.5}, mu, weights = {1, 2, 0.5, 1.5, 0};
            long double expected = 0;
            for (unsigned i = 0; i < y.size(); ++i) {
                mu.push_back(static_cast<double>(reference_link(family, eta[i])));
                expected += weights[i] * reference_log_mass(family, y[i], mu[i], theta);
            }
            BOOST_TEST_CONTEXT("family=" << static_cast<int>(family) << " theta=" << theta) {
                near(detail::likelihood(y, mu, weights, family, theta),
                     static_cast<double>(expected), 3e-12);
            }
        }
}

BOOST_AUTO_TEST_CASE(count_deviance_against_saturated_likelihood) {
    Vector y = {0, 1, 3, 8}, mu = {0.3, 1.00001, 2, 6}, weights = {2, 1, 0.5, 3};
    for (auto family : {Family::poisson, Family::negative_binomial})
        for (double theta : {0.2, 10.0, 1e8}) {
            long double expected = 0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                long double saturated =
                    y[i] == 0 ? 0 : reference_log_mass(family, y[i], y[i], theta);
                expected +=
                    2 * weights[i] * (saturated - reference_log_mass(family, y[i], mu[i], theta));
            }
            near(detail::count_deviance(y, mu, weights, family, theta),
                 static_cast<double>(expected), 1e-9);
            BOOST_TEST(detail::count_deviance({1, 2}, {1, 2}, {1, 1}, family, theta) == 0);
            double d = detail::count_deviance({3}, {3 + 1e-7}, {1}, family, theta);
            double curvature = family == Family::poisson ? 1.0 / 3 : 1.0 / (3 + 9 / theta);
            near(d / 1e-14, curvature, 1e-7);
        }
    near(detail::count_deviance({1}, {1e-300}, {1}, Family::poisson, 10), 2 * (std::log(1e300) - 1),
         1e-12);
}

BOOST_AUTO_TEST_CASE(score_and_information_against_numerical_derivatives) {
    const long double step = 0.0002L;
    for (auto family : families)
        for (double theta : {0.3, 2.0, 20.0}) {
            Vector y = family == Family::binomial ? Vector{0, 0.3, 1, 0.75} : Vector{0, 1, 4, 80};
            Vector eta = {-1, 0.2, 1, 2}, mu, a = {1, 2, 0, 0.5}, w, s;
            for (double e : eta)
                mu.push_back(static_cast<double>(reference_link(family, e)));
            detail::working(y, mu, a, family, theta, w, s);
            for (unsigned i = 0; i < y.size(); ++i) {
                auto f = [&](long double e) {
                    return a[i] *
                           reference_log_mass(family, y[i], reference_link(family, e), theta);
                };
                long double left = f(eta[i] - step), center = f(eta[i]), right = f(eta[i] + step);
                double numerical_score = static_cast<double>((right - left) / (2 * step));
                double observed_information =
                    static_cast<double>((2 * center - left - right) / (step * step));
                // NB's log link is noncanonical: observed curvature differs from
                // Fisher information by (theta+y)/(theta+mu). Others are canonical.
                double observed_from_fisher =
                    w[i] *
                    (family == Family::negative_binomial ? (theta + y[i]) / (theta + mu[i]) : 1);
                BOOST_TEST_CONTEXT("family=" << static_cast<int>(family) << " row=" << i
                                             << " theta=" << theta) {
                    near(s[i], numerical_score, 2e-8);
                    near(observed_from_fisher, observed_information, 2e-7);
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(nb_information_is_expected_score_variance) {
    for (double theta : {0.4, 2.0, 15.0})
        for (double mu : {0.5, 2.0, 10.0}) {
            long double mass = 0, mean_score = 0, second_score = 0;
            for (unsigned y = 0; y < 1024; ++y) {
                double p = static_cast<double>(
                    std::exp(reference_log_mass(Family::negative_binomial, y, mu, theta)));
                Vector w, s;
                detail::working({static_cast<double>(y)}, {mu}, {1}, Family::negative_binomial,
                                theta, w, s);
                mass += p;
                mean_score += p * s[0];
                second_score += p * s[0] * s[0];
            }
            Vector w, s;
            detail::working({0}, {mu}, {1}, Family::negative_binomial, theta, w, s);
            near(static_cast<double>(mass), 1, 1e-12);
            near(static_cast<double>(mean_score), 0, 1e-12);
            near(w[0], static_cast<double>(second_score), 1e-11);
        }
}

BOOST_AUTO_TEST_CASE(binomial_grouped_trials_analytic_coefficients_and_covariance) {
    Matrix x(5, 2, 1);
    for (unsigned i = 0; i < 5; ++i)
        x(i, 1) = i >= 2;
    Vector y = {1.0 / 5, 3.0 / 7, 2.0 / 4, 5.0 / 6, 1};
    FitOptions o;
    o.family = Family::binomial;
    o.weights = {5, 7, 4, 6, 0};
    o.offset = {0.2, 0.2, -0.3, -0.3, 20};
    for (const Vector &start : {Vector{}, Vector{3, -4}}) {
        o.start = start;
        auto f = fit(x, y, o);
        BOOST_REQUIRE(f.ok());
        double b0 = std::log(0.5) - 0.2, b1 = std::log(7.0 / 3) + 0.3 - b0;
        double v0 = 1 / (12 * (1.0 / 3) * (2.0 / 3)), v1 = 1 / (10 * 0.7 * 0.3);
        near(f.coefficients[0], b0, 1e-9);
        near(f.coefficients[1], b1, 1e-9);
        near(f.covariance(0, 0), v0, 1e-9);
        near(f.covariance(0, 1), -v0, 1e-9);
        near(f.covariance(1, 1), v0 + v1, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(nb_theta_optimizer_against_derivative_free_reference) {
    Vector y = {0, 0, 1, 1, 2, 2, 3, 4, 4, 8, 12, 80};
    Vector a = {1, 2, 1, 0, 0.5, 1, 2, 1, 1, 0.5, 2, 1};
    Vector mu = {1, 3, 2, 4, 1, 2, 3, 6, 4, 8, 9, 15};
    FitOptions o;
    o.theta_min = 0.01;
    o.theta_max = 1000;
    double expected = reference_theta(y, mu, a, o.theta_min, o.theta_max);
    BOOST_REQUIRE(expected > o.theta_min * 2 && expected < o.theta_max / 2);
    for (double start : {0.05, 2.0, 100.0}) {
        double theta = detail::estimate_theta(y, mu, a, start, o);
        near(std::log(theta), std::log(expected), 2e-6);
        const long double h = 1e-4L, u = std::log(theta);
        double score = static_cast<double>(
            (reference_nb(y, mu, a, u + h) - reference_nb(y, mu, a, u - h)) / (2 * h));
        near(score, 0, 2e-6);
    }
}

BOOST_AUTO_TEST_CASE(nb_ml_fit_against_independent_profile_optimum) {
    Vector y = {0, 0, 1, 1, 2, 2, 3, 4, 4, 8, 12, 20};
    FitOptions o;
    o.weights = {1, 2, 1, 0, 0.5, 1, 2, 1, 1, 0.5, 2, 1};
    o.offset = Vector(y.size(), 0.35);
    double total = 0, count = 0;
    for (unsigned i = 0; i < y.size(); ++i) {
        total += o.weights[i] * y[i];
        count += o.weights[i];
    }
    double mean = total / count;
    double expected_theta = reference_theta(y, Vector(y.size(), mean), o.weights, 0.01, 1000);
    auto f = fit(Matrix(y.size(), 1, 1), y, o);
    BOOST_REQUIRE(f.ok());
    near(f.coefficients[0], std::log(mean) - 0.35, 1e-9);
    near(std::log(f.theta), std::log(expected_theta), 2e-6);
    near(f.standard_errors[0], std::sqrt((1 / mean + 1 / expected_theta) / count), 2e-6);
}

BOOST_AUTO_TEST_CASE(nb_moment_estimate_and_dispersion_bounds) {
    FitOptions o;
    o.theta_min = 0.1;
    o.theta_max = 100;
    Vector y = {0, 1, 2, 3, 9, 0}, mu(6, 2), a = {1, 2, 0, 1, 1, 0};
    near(detail::moment_theta(y, mu, a, o), 10.0 / 23, 1e-12);
    near(detail::moment_theta(Vector(6, 2), mu, a, o), o.theta_max, 1e-12);
    o.theta_min = 1;
    near(detail::moment_theta(y, mu, a, o), o.theta_min, 1e-12);
    o.theta_min = 0.01;
    o.theta_max = 0.1;
    near(detail::moment_theta(y, mu, a, o), o.theta_max, 1e-12);
    o.theta_max = 1000;
    near(detail::estimate_theta(Vector(6, 2), mu, a, 1, o), o.theta_max, 1e-8);
}

BOOST_AUTO_TEST_CASE(hc0_covariance_matches_group_mean_influence_variance) {
    Matrix x(8, 2, 1);
    for (unsigned i = 0; i < 8; ++i)
        x(i, 1) = i >= 4;
    for (auto family : families) {
        Vector y = family == Family::binomial ? Vector{0, 0.25, 1, 0.75, 0.2, 1, 0.5, 0}
                                              : Vector{0, 1, 5, 2, 2, 8, 3, 0};
        FitOptions o;
        o.family = family;
        o.dispersion = Dispersion::fixed;
        o.theta = 2.5;
        o.robust = true;
        o.weights = {1, 2, 0.5, 1, 2, 1, 1, 0};
        o.offset = {0.2, 0.2, 0.2, 0.2, -0.3, -0.3, -0.3, -0.3};
        auto f = fit(x, y, o);
        BOOST_REQUIRE(f.ok());
        double variance[2] = {};
        for (unsigned group = 0; group < 2; ++group) {
            double total = 0, weight = 0, residual_sum = 0;
            for (unsigned i = group * 4; i < group * 4 + 4; ++i) {
                total += o.weights[i] * y[i];
                weight += o.weights[i];
            }
            double mean = total / weight;
            for (unsigned i = group * 4; i < group * 4 + 4; ++i)
                residual_sum += std::pow(o.weights[i] * (y[i] - mean), 2);
            double derivative = family == Family::gaussian   ? 1
                                : family == Family::binomial ? mean * (1 - mean)
                                                             : mean;
            variance[group] = residual_sum / std::pow(weight * derivative, 2);
        }
        near(f.robust_covariance(0, 0), variance[0], 1e-9);
        near(f.robust_covariance(0, 1), -variance[0], 1e-9);
        near(f.robust_covariance(1, 0), -variance[0], 1e-9);
        near(f.robust_covariance(1, 1), variance[0] + variance[1], 1e-9);
        auto test = wald_test(f, 1, Side::both, true);
        BOOST_REQUIRE(test.valid);
        near(test.standard_error, std::sqrt(variance[0] + variance[1]), 1e-9);
    }
}
BOOST_AUTO_TEST_CASE(response_only_terms_match_direct_gamma_functions) {
    // Counts on both sides of 64 (histogram and Stirling's series) and theta on
    // both sides of the counts and of 65 (the forms of nb_constant and polygamma).
    std::mt19937_64 rng(31);
    for (double scale : {40.0, 130.0, 3000.0, 2e5})
        for (double theta : {1e-6, 0.3, 7.0, 64.5, 65.0, 800.0, 1e5, 1e8}) {
            Vector y(200), a(200);
            for (std::size_t i = 0; i < y.size(); ++i) {
                y[i] = std::floor(scale * std::generate_canonical<double, 53>(rng));
                a[i] = i % 7 ? 1 : 0.5;
            }
            const detail::CountTerms terms(y, a);
            const long double t = theta;
            long double constant = 0, factorial = 0, first = 0, second = 0, counts = 0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                const long double v = y[i];
                constant += a[i] * (std::lgamma(t + v) - std::lgamma(t) - std::lgamma(v + 1));
                factorial += a[i] * std::lgamma(v + 1);
                first += a[i] * (boost::math::digamma(t + v) - boost::math::digamma(t));
                second += a[i] * (boost::math::trigamma(t + v) - boost::math::trigamma(t));
                counts += a[i] * v;
            }
            // The direct sums lose digits where their terms cancel (theta far
            // above the counts): compare within the rounding of those terms.
            const long double rounding =
                64 * std::numeric_limits<long double>::epsilon() * counts * std::log(t + scale + 2);
            BOOST_TEST(std::abs(terms.nb_constant(theta) - constant) <=
                       1e-13L * std::abs(constant) + rounding + 1e-12L);
            BOOST_TEST(std::abs(terms.log_factorial - factorial) <= 1e-14L * factorial + 1e-12L);
            const auto [digamma, trigamma] = terms.polygamma(theta);
            BOOST_TEST(std::abs(digamma - first) <= 1e-9L * std::abs(first));
            BOOST_TEST(std::abs(trigamma - second) <= 1e-9L * std::abs(second));
        }
    for (double r :
         {0.0, 1e-300, 1e-17, 3e-16, 1e-9, 1e-3, 0.5, 1.0, 40.0, 1e8, 1e300, -0.5, -1e-9})
        BOOST_TEST(std::abs(detail::log1p_by_log(r) - std::log1p(r)) <=
                   4 * std::numeric_limits<double>::epsilon() * std::abs(std::log1p(r)));
}
BOOST_AUTO_TEST_CASE(sparse_rows_give_the_dense_products) {
    // A design of factors is used through its nonzero entries; a dense one is not.
    std::mt19937_64 rng(5);
    const std::size_t n = 203, p = 9;
    Matrix factors(n, p), dense(n, p);
    Vector w(n), v(n), b(p);
    for (std::size_t i = 0; i < n; ++i) {
        factors(i, 0) = 1;
        factors(i, 1 + rng() % 4) = 1;
        factors(i, 5 + rng() % 4) = 0.5 + double(rng() % 3);
        for (std::size_t j = 0; j < p; ++j)
            dense(i, j) = std::generate_canonical<double, 53>(rng) - 0.5;
        w[i] = i % 11 ? std::generate_canonical<double, 53>(rng) : 0;
        v[i] = std::generate_canonical<double, 53>(rng) - 0.3;
    }
    for (double &e : b)
        e = std::generate_canonical<double, 53>(rng) - 0.5;
    const detail::SparseRows rows(factors), none(dense);
    BOOST_TEST(rows.sparse);
    BOOST_TEST(!none.sparse);
    const auto a = detail::crossprod(factors, w), s = detail::crossprod(factors, w, &rows);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < p; ++k) {
            BOOST_TEST(std::abs(a(j, k) - s(j, k)) <= 1e-13 * (1 + std::abs(a(j, k))));
            BOOST_TEST(s(j, k) == s(k, j));
        }
    const auto u = detail::xtv(factors, v), us = detail::xtv(factors, v, &rows);
    for (std::size_t j = 0; j < p; ++j)
        BOOST_TEST(std::abs(u[j] - us[j]) <= 1e-13 * (1 + std::abs(u[j])));
    const auto e = detail::matvec(factors, b), es = detail::matvec(factors, b, &rows);
    for (std::size_t i = 0; i < n; ++i)
        BOOST_TEST(std::abs(e[i] - es[i]) <= 1e-13 * (1 + std::abs(e[i])));
    // Without sparse rows the dense products are returned as they are.
    BOOST_TEST(detail::crossprod(dense, w, &none).values == detail::crossprod(dense, w).values);
}
BOOST_AUTO_TEST_CASE(a_shared_design_summary_changes_no_result) {
    std::mt19937_64 rng(9);
    const std::size_t n = 120, p = 4;
    for (bool sparse : {false, true}) {
        Matrix x(n, p);
        Vector offset(n);
        for (std::size_t i = 0; i < n; ++i) {
            x(i, 0) = 1;
            if (sparse) {
                // One factor of four levels, the last the reference.
                const unsigned level = rng() % 4;
                if (level < 3)
                    x(i, 1 + level) = 1;
            } else
                for (std::size_t j = 1; j < p; ++j)
                    x(i, j) = std::generate_canonical<double, 53>(rng) - 0.5;
            offset[i] = 0.2 * (std::generate_canonical<double, 53>(rng) - 0.5);
        }
        const auto summary = prepare_design(x);
        BOOST_TEST(summary->nonzero.sparse == sparse);
        for (Family family : {Family::poisson, Family::negative_binomial})
            for (unsigned gene = 0; gene < 6; ++gene) {
                Vector y(n);
                for (std::size_t i = 0; i < n; ++i)
                    y[i] = double(std::poisson_distribution<long>(
                        gene == 0 ? 0.4 : 30.0 * gene * std::exp(x(i, 1)))(rng));
                FitOptions o;
                o.family = family;
                o.dispersion = gene % 2 ? Dispersion::fixed : Dispersion::ml;
                o.theta = 5;
                o.offset = offset;
                const auto plain = fit(x, y, o);
                o.design = summary;
                const auto shared = fit(x, y, o);
                BOOST_REQUIRE(plain.ok());
                BOOST_TEST(shared.coefficients == plain.coefficients);
                BOOST_TEST(shared.standard_errors == plain.standard_errors);
                BOOST_TEST(shared.log_likelihood == plain.log_likelihood);
                BOOST_TEST(shared.iterations == plain.iterations);
                // A summary of another design, or prior weights, are not used.
                Matrix other = x;
                other(0, 1) += 0.25;
                o.weights.assign(n, 1);
                o.weights[3] = 2;
                BOOST_TEST(fit(other, y, o).ok());
            }
    }
}
BOOST_AUTO_TEST_CASE(convergence_by_the_chord_step_matches_further_iterations) {
    // The estimate at the default tolerance agrees with a fit iterated far beyond
    // it, to well within that tolerance.
    std::mt19937_64 rng(17);
    const std::size_t n = 300, p = 5;
    Matrix x(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 0) = 1;
        for (std::size_t j = 1; j < p; ++j)
            x(i, j) = std::generate_canonical<double, 53>(rng) - 0.5;
    }
    for (Family family : {Family::poisson, Family::negative_binomial, Family::binomial})
        for (unsigned gene = 0; gene < 20; ++gene) {
            Vector y(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double eta = 0.2 * gene + x(i, 1) - 0.5 * x(i, 2);
                y[i] = family == Family::binomial
                           ? double(std::bernoulli_distribution(1 / (1 + std::exp(-eta + 2)))(rng))
                           : double(std::poisson_distribution<long>(std::exp(eta))(rng));
            }
            FitOptions o;
            o.family = family;
            o.dispersion = Dispersion::fixed;
            o.theta = 3;
            const auto f = fit(x, y, o);
            o.tolerance = 1e-14;
            const auto exact = fit(x, y, o);
            BOOST_REQUIRE(f.ok());
            BOOST_REQUIRE(exact.ok());
            BOOST_TEST(f.iterations <= exact.iterations);
            for (std::size_t j = 0; j < p; ++j) {
                BOOST_TEST(std::abs(f.coefficients[j] - exact.coefficients[j]) <=
                           1e-10 * (1 + std::abs(exact.coefficients[j])));
                BOOST_TEST(std::abs(f.standard_errors[j] - exact.standard_errors[j]) <=
                           1e-9 * exact.standard_errors[j]);
            }
            BOOST_TEST(std::abs(f.log_likelihood - exact.log_likelihood) <=
                       1e-10 * (1 + std::abs(exact.log_likelihood)));
        }
}
