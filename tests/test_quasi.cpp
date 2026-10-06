#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
double nb_draw(std::mt19937_64 &rng, double mu, double theta) {
    std::gamma_distribution<double> gamma(theta, mu / theta);
    return static_cast<double>(std::poisson_distribution<long>(gamma(rng))(rng));
}
SparseCSC columns(const std::vector<Vector> &ys) {
    SparseCSC s;
    s.rows = ys.front().size();
    s.cols = ys.size();
    s.column_ptr = {0};
    for (const auto &y : ys) {
        for (std::size_t i = 0; i < y.size(); ++i)
            if (y[i] != 0) {
                s.row_index.push_back(i);
                s.values.push_back(y[i]);
            }
        s.column_ptr.push_back(s.values.size());
    }
    return s;
}
// Four-versus-four pseudobulk panel with depth offsets and a mean-dispersion
// trend; the first `de` genes have a group effect of log(4).
struct Panel {
    Matrix x;
    Vector offset;
    SparseCSC y;
};
Panel panel(std::size_t genes, std::size_t de, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform;
    Panel p;
    p.x = Matrix(8, 2, 1);
    for (std::size_t i = 0; i < 8; ++i) {
        p.x(i, 1) = i % 2;
        p.offset.push_back(std::log(0.6 + 0.8 * uniform(rng)));
    }
    std::vector<Vector> ys;
    for (std::size_t g = 0; g < genes; ++g) {
        const double base = std::exp(std::log(5.0) + std::log(400.0) * uniform(rng));
        const double dispersion = (0.02 + 1 / base) * std::exp(0.5 * (uniform(rng) - 0.5));
        Vector y(8);
        for (std::size_t i = 0; i < 8; ++i) {
            const double mu =
                base * std::exp(p.offset[i] + (g < de ? std::log(4.0) * p.x(i, 1) : 0));
            y[i] = nb_draw(rng, mu, 1 / dispersion);
        }
        ys.push_back(y);
    }
    p.y = columns(ys);
    return p;
}
double brute_weighted_median(std::vector<std::pair<double, double>> v) {
    std::stable_sort(v.begin(), v.end(), [](auto a, auto b) { return a.first < b.first; });
    double total = 0, cumulative = 0;
    for (auto &e : v)
        total += e.second;
    for (auto &e : v) {
        cumulative += e.second;
        if (cumulative >= 0.5 * total)
            return e.first;
    }
    return v.back().first;
}
} // namespace
BOOST_AUTO_TEST_CASE(weighted_median_matches_sorted_definition) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> uniform;
    for (std::size_t n : {1u, 2u, 3u, 8u, 101u, 1000u})
        for (int ties = 0; ties < 2; ++ties) {
            std::vector<std::pair<double, double>> v;
            for (std::size_t i = 0; i < n; ++i)
                v.emplace_back(ties ? std::floor(5 * uniform(rng)) : uniform(rng),
                               0.01 + uniform(rng));
            auto copy = v;
            BOOST_TEST(detail::weighted_median(copy) == brute_weighted_median(v));
        }
}
BOOST_AUTO_TEST_CASE(running_median_trend_properties) {
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> uniform;
    const std::size_t n = 500;
    Vector x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = uniform(rng);
        y[i] = (x[i] > 0.5 ? 2.0 : 1.0) + 0.01 * uniform(rng);
    }
    auto trend = running_median(x, y, 51);
    for (std::size_t i = 0; i < n; ++i)
        if (std::abs(x[i] - 0.5) > 0.1)
            BOOST_TEST(std::abs(trend[i] - (x[i] > 0.5 ? 2.0 : 1.0)) < 0.011);
    // The trend follows points, not their input order.
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    auto shuffled = running_median(subset(x, perm), subset(y, perm), 51);
    for (std::size_t k = 0; k < n; ++k)
        BOOST_TEST(shuffled[k] == trend[perm[k]]);
    // Windows wider than the data give one weighted median.
    auto wide = running_median(x, y, 10 * n);
    BOOST_TEST(std::all_of(wide.begin(), wide.end(), [&](double v) { return v == wide[0]; }));
    BOOST_TEST(running_median(Vector{3}, Vector{7}, 100)[0] == 7);
}
BOOST_AUTO_TEST_CASE(natural_spline_basis_spans_lines_and_is_linear_at_the_ends) {
    Vector x;
    for (int i = 0; i <= 60; ++i)
        x.push_back(-2 + 5 * i / 60.0);
    auto b = natural_spline_basis(x, -2, 3);
    Vector line(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        line[i] = 0.7 - 1.3 * x[i];
    detail::Cholesky ch(detail::crossprod(b, Vector(x.size(), 1)));
    BOOST_REQUIRE(ch.ok);
    auto beta = ch.solve(detail::xtv(b, line));
    auto fitted = detail::matvec(b, beta);
    for (std::size_t i = 0; i < x.size(); ++i)
        BOOST_TEST(fitted[i] == line[i], boost::test_tools::tolerance(1e-9));
    // Second derivatives vanish at both boundary knots.
    const double h = 1e-4;
    auto at = [](double t) { return natural_spline_basis(Vector{t}, 0, 1); };
    for (double edge : {0.0, 1.0}) {
        const double s = edge == 0 ? 1 : -1;
        auto b0 = at(edge), b1 = at(edge + s * h), b2 = at(edge + 2 * s * h);
        for (std::size_t j = 0; j < 4; ++j)
            BOOST_TEST(std::abs(b2(0, j) - 2 * b1(0, j) + b0(0, j)) / (h * h) < 1e-2);
    }
}
BOOST_AUTO_TEST_CASE(variance_prior_recovers_scaled_f_parameters) {
    std::mt19937_64 rng(13);
    const std::size_t n = 20000;
    const double d = 7, d0 = 10, s0 = 1.5;
    std::chi_squared_distribution<double> residual(d), prior_draw(d0);
    Vector s2(n), df(n, d), chi(n);
    for (std::size_t i = 0; i < n; ++i) {
        chi[i] = residual(rng) / d;
        s2[i] = s0 * chi[i] / (prior_draw(rng) / d0);
    }
    auto prior = fit_variance_prior(s2, df);
    BOOST_TEST(prior.df > 8.5);
    BOOST_TEST(prior.df < 11.8);
    BOOST_TEST(prior.scale[0] == s0, boost::test_tools::tolerance(0.04));
    BOOST_TEST(prior.posterior[3] == (prior.df * prior.scale[3] + d * s2[3]) / (prior.df + d),
               boost::test_tools::tolerance(1e-12));
    // No heterogeneity: d0 goes to the upper limit and posteriors approach the scale.
    for (std::size_t i = 0; i < n; ++i)
        s2[i] = s0 * chi[i];
    auto flat = fit_variance_prior(s2, df);
    BOOST_TEST(flat.df > 100 * d);
    BOOST_TEST(flat.scale[0] == s0, boost::test_tools::tolerance(0.02));
    for (std::size_t i = 0; i < 100; ++i)
        BOOST_TEST(flat.posterior[i] == flat.scale[i], boost::test_tools::tolerance(0.05));
    // Abundance trend: log s0^2 = 0.3 log c.
    std::uniform_real_distribution<double> uniform;
    Vector c(n);
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = std::exp(std::log(0.1) + std::log(1e4) * uniform(rng));
        s2[i] = std::exp(0.3 * std::log(c[i])) * chi[i] / (prior_draw(rng) / d0);
    }
    auto trended = fit_variance_prior(s2, df, c, true);
    BOOST_TEST(trended.trend);
    for (std::size_t i = 0; i < 200; ++i)
        BOOST_TEST(trended.scale[i] == std::exp(0.3 * std::log(c[i])),
                   boost::test_tools::tolerance(0.12));
    BOOST_CHECK_THROW(fit_variance_prior(Vector{1}, Vector{3}), std::exception);
}
BOOST_AUTO_TEST_CASE(cox_reid_term_derivatives) {
    std::mt19937_64 rng(14);
    std::normal_distribution<double> normal;
    Matrix x(10, 3, 1);
    Vector mu(10), a(10, 1);
    for (std::size_t i = 0; i < 10; ++i) {
        x(i, 1) = normal(rng);
        x(i, 2) = i % 2;
        mu[i] = std::exp(1 + 0.5 * normal(rng));
        a[i] = 0.5 + (i % 3);
    }
    const double theta = 3.7, h = 1e-5;
    auto c = detail::cox_reid(x, mu, a, theta, true);
    BOOST_REQUIRE(c.ok);
    auto value = [&](double t) { return detail::cox_reid(x, mu, a, t, false).value; };
    auto first = [&](double t) { return detail::cox_reid(x, mu, a, t, true).first; };
    BOOST_TEST(c.first == (value(theta + h) - value(theta - h)) / (2 * h),
               boost::test_tools::tolerance(1e-6));
    BOOST_TEST(c.second == (first(theta + h) - first(theta - h)) / (2 * h),
               boost::test_tools::tolerance(1e-5));
    // Value against a direct 3 x 3 determinant.
    Matrix info(3, 3);
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                info(j, k) += a[i] * mu[i] * theta / (theta + mu[i]) * x(i, j) * x(i, k);
    const double det = info(0, 0) * (info(1, 1) * info(2, 2) - info(1, 2) * info(2, 1)) -
                       info(0, 1) * (info(1, 0) * info(2, 2) - info(1, 2) * info(2, 0)) +
                       info(0, 2) * (info(1, 0) * info(2, 1) - info(1, 1) * info(2, 0));
    BOOST_TEST(c.value == -0.5 * std::log(det), boost::test_tools::tolerance(1e-12));
}
BOOST_AUTO_TEST_CASE(cox_reid_fit_is_a_fixed_point) {
    std::mt19937_64 rng(15);
    Matrix x(12, 3, 1);
    Vector y(12), offset(12);
    for (std::size_t i = 0; i < 12; ++i) {
        x(i, 1) = i % 2;
        x(i, 2) = (i / 2) % 3 == 0;
        offset[i] = 0.1 * (i % 4);
        y[i] = nb_draw(rng, std::exp(2 + 0.5 * x(i, 1) + offset[i]), 4);
    }
    FitOptions o;
    o.offset = offset;
    auto ml = fit(x, y, o);
    o.cox_reid = true;
    auto cr = fit(x, y, o);
    BOOST_REQUIRE(ml.ok());
    BOOST_REQUIRE(cr.ok());
    BOOST_TEST(cr.cox_reid);
    BOOST_TEST(!ml.cox_reid);
    // The adjustment raises the dispersion estimate (lowers theta).
    BOOST_TEST(cr.theta < ml.theta);
    // Coefficients maximize the likelihood at theta; theta maximizes the adjusted
    // likelihood at the fitted means.
    const Vector a(12, 1);
    auto score = detail::xtv(x, [&] {
        Vector s(12);
        for (std::size_t i = 0; i < 12; ++i)
            s[i] = (y[i] - cr.fitted[i]) / (1 + cr.fitted[i] / cr.theta);
        return s;
    }());
    for (double s : score)
        BOOST_TEST(std::abs(s) < 1e-6);
    const detail::CountTerms terms(y, a);
    auto adjusted = [&](double u) {
        const double t = std::exp(u);
        return detail::likelihood(y, cr.fitted, a, Family::negative_binomial, t, terms) +
               detail::cox_reid(x, cr.fitted, a, t, false).value;
    };
    const double u = std::log(cr.theta), h = 1e-4;
    BOOST_TEST(std::abs(adjusted(u + h) - adjusted(u - h)) / (2 * h) < 1e-6);
    // Only NB ML dispersion fits take the adjustment.
    for (auto dispersion : {Dispersion::fixed, Dispersion::moment}) {
        auto bad = o;
        bad.dispersion = dispersion;
        BOOST_CHECK_THROW(fit(x, y, bad), std::exception);
    }
    auto poisson = o;
    poisson.family = Family::poisson;
    BOOST_CHECK_THROW(fit(x, y, poisson), std::exception);
    BOOST_CHECK_THROW(fit_sceptre_null(x, y, o), std::exception);
}
BOOST_AUTO_TEST_CASE(quasi_likelihood_calibration_power_and_threads) {
    auto p = panel(1200, 120, 16);
    FitOptions o;
    o.offset = p.offset;
    auto one = quasi_likelihood_tests(p.x, p.y, o, 1, 1);
    auto four = quasi_likelihood_tests(p.x, p.y, o, 1, 4);
    BOOST_TEST(one.dispersion_features == 1200u);
    BOOST_TEST(one.abundance_trend);
    BOOST_TEST(one.df_prior > 1);
    std::size_t null_05 = 0, null_01 = 0, nulls = 0, found = 0;
    for (std::size_t j = 0; j < 1200; ++j) {
        const auto &t = one.tests[j];
        BOOST_TEST(t.test.p_value == four.tests[j].test.p_value);
        BOOST_TEST(t.fit.coefficients[1] == four.tests[j].fit.coefficients[1]);
        if (!t.test.valid)
            continue;
        if (j < 120)
            found += t.adjusted_p <= 0.05;
        else {
            ++nulls;
            null_05 += t.test.p_value <= 0.05;
            null_01 += t.test.p_value <= 0.01;
        }
    }
    BOOST_TEST(nulls > 1000u);
    BOOST_TEST(null_05 < 0.08 * nulls);
    BOOST_TEST(null_05 > 0.02 * nulls);
    BOOST_TEST(null_01 < 0.02 * nulls);
    BOOST_TEST(found > 60u);
    // Reported standard errors carry the shrunken QL dispersion.
    const auto &t = one.tests[500];
    BOOST_REQUIRE(t.fit.ok());
    BOOST_TEST(t.fit.scale == one.features[500].posterior);
    BOOST_TEST(t.fit.theta == 1 / one.features[500].trend, boost::test_tools::tolerance(1e-12));
}
BOOST_AUTO_TEST_CASE(quasi_likelihood_input_validation) {
    auto p = panel(20, 0, 17);
    FitOptions o;
    o.offset = p.offset;
    auto poisson = o;
    poisson.family = Family::poisson;
    BOOST_CHECK_THROW(quasi_likelihood_tests(p.x, p.y, poisson, 1), std::exception);
    auto weighted = o;
    weighted.weights = Vector(8, 1);
    BOOST_CHECK_THROW(quasi_likelihood_tests(p.x, p.y, weighted, 1), std::exception);
    auto single = columns({p.y.column(0)});
    BOOST_CHECK_THROW(quasi_likelihood_tests(p.x, single, o, 1), std::exception);
    BOOST_CHECK_NO_THROW(quasi_likelihood_tests(p.x, p.y, o, 1));
}
