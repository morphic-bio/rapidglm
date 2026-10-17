#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
double nb_draw(std::mt19937_64 &rng, double mu, double phi) {
    std::gamma_distribution<double> gamma(1 / phi, mu * phi);
    return static_cast<double>(std::poisson_distribution<long>(gamma(rng))(rng));
}
// Mean and variance of the NB unit deviance by summing the pmf over every count.
std::pair<double, double> summed_moments(double m, double phi) {
    const double k = 1 / phi;
    const boost::math::negative_binomial_distribution<long double> nb(k, k / (k + m));
    const double sd = std::sqrt(m + phi * m * m);
    const auto last = static_cast<long>(m + 60 * sd + 60 * m / k + 100);
    long double s0 = 0, s1 = 0, s2 = 0;
    for (long z = 0; z <= last; ++z) {
        const long double p = boost::math::pdf(nb, static_cast<long double>(z));
        const long double d = detail::nb_unit_deviance(static_cast<double>(z), m, phi);
        s0 += p;
        s1 += p * d;
        s2 += p * d * d;
    }
    const double mean = static_cast<double>(s1 / s0);
    return {mean, static_cast<double>(s2 / s0) - mean * mean};
}
// Two groups of `n` samples with unequal depths; genes with a mean from 2 to
// 2000 counts and dispersion `phi`; the first `de` genes change fourfold.
struct Bulk {
    Matrix x;
    SparseCSC counts;
};
Bulk bulk(std::size_t n, std::size_t genes, std::size_t de, double phi, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform;
    Bulk b{Matrix(2 * n, 2, 1), SparseCSC{2 * n, genes, {0}, {}, {}}};
    Vector depth(2 * n);
    for (std::size_t i = 0; i < 2 * n; ++i) {
        b.x(i, 1) = i >= n;
        depth[i] = 0.5 + uniform(rng);
    }
    for (std::size_t g = 0; g < genes; ++g) {
        const double base = std::exp(std::log(2.0) + uniform(rng) * std::log(1000.0));
        for (std::size_t i = 0; i < 2 * n; ++i) {
            const double mu = base * depth[i] * (g < de && i >= n ? 4 : 1);
            const double y = nb_draw(rng, mu, phi);
            if (y > 0) {
                b.counts.row_index.push_back(i);
                b.counts.values.push_back(y);
            }
        }
        b.counts.column_ptr.push_back(b.counts.values.size());
    }
    return b;
}
} // namespace

BOOST_AUTO_TEST_CASE(unit_deviance_is_stable_far_from_the_count) {
    // Reference in long double from the definition.
    for (double phi : {0.01, 0.2, 2.0})
        for (double y : {1.0, 7.0, 300.0})
            for (double m : {1e-8, 0.3, 50.0, 1e6, 1e12}) {
                const long double k = 1 / static_cast<long double>(phi), yl = y, ml = m;
                const long double reference =
                    2 * (yl * std::log(yl / ml) - (yl + k) * std::log((yl + k) / (ml + k)));
                BOOST_TEST(detail::nb_unit_deviance(y, m, phi) == static_cast<double>(reference),
                           boost::test_tools::tolerance(1e-11));
            }
    BOOST_TEST(detail::nb_unit_deviance(0, 2.5, 0) == 5.0, boost::test_tools::tolerance(1e-15));
    BOOST_TEST(detail::nb_unit_deviance(4, 4, 0.3) == 0.0);
}

BOOST_AUTO_TEST_CASE(deviance_moments_equal_sums_over_the_pmf) {
    for (double phi : {0.005, 0.1, 0.6, 3.0})
        for (double m : {1e-4, 0.05, 1.0, 12.0, 400.0, 3000.0}) {
            const auto [e, v] = detail::nb_deviance_moments(m, phi);
            const auto [se, sv] = summed_moments(m, phi);
            BOOST_TEST(e == se, boost::test_tools::tolerance(1e-9));
            BOOST_TEST(v == sv, boost::test_tools::tolerance(1e-8));
        }
    // Large means approach the gamma limit, E = 2k (log k - digamma(k)).
    const double phi = 0.1, k = 10;
    const auto [e, v] = detail::nb_deviance_moments(1e7, phi);
    BOOST_TEST(e == 2 * k * (std::log(k) - boost::math::digamma(k)),
               boost::test_tools::tolerance(1e-4));
    BOOST_TEST(v == 4 * k * k * (boost::math::trigamma(k) - 1 / k),
               boost::test_tools::tolerance(1e-4));
}

BOOST_AUTO_TEST_CASE(deviance_moment_table_interpolates) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> exponent(-7, 5.5);
    for (double phi : {0.02, 0.4}) {
        const detail::DevianceMoments table(phi, 1e5);
        for (int t = 0; t < 200; ++t) {
            const double m = std::pow(10.0, exponent(rng));
            const auto [e, v] = table(m);
            const auto [de, dv] = detail::nb_deviance_moments(m, phi);
            BOOST_TEST(e == de, boost::test_tools::tolerance(1e-7));
            BOOST_TEST(v == dv, boost::test_tools::tolerance(1e-6));
        }
    }
}

BOOST_AUTO_TEST_CASE(brent_minimizes) {
    const double x =
        detail::brent_minimum([](double t) { return (t - 0.3) * (t - 0.3) + 1; }, 0, 2, 1e-8);
    BOOST_TEST(x == 0.3, boost::test_tools::tolerance(1e-6));
    const double edge = detail::brent_minimum([](double t) { return std::exp(t); }, -1, 1, 1e-6);
    BOOST_TEST(edge < -0.999);
}

BOOST_AUTO_TEST_CASE(leverages_and_fixed_dispersion_fits) {
    std::mt19937_64 rng(11);
    std::normal_distribution<double> normal;
    const std::size_t n = 20;
    Matrix x(n, 3, 1);
    Vector offset(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = normal(rng);
        x(i, 2) = i % 2;
        offset[i] = 0.2 * normal(rng);
        y[i] = nb_draw(rng, std::exp(2 + 0.5 * x(i, 1) + offset[i]), 0.1);
    }
    const detail::Cholesky ls(detail::crossprod(x, Vector(n, 1)), 1e-12, false);
    const auto fast = detail::nb_fixed_fit(x, y, offset, 0.1, {}, ls);
    BOOST_REQUIRE(fast.ok);
    const auto general = fit(x, y, detail::fixed_dispersion_options(0.1, offset, nullptr));
    for (std::size_t j = 0; j < 3; ++j)
        BOOST_TEST(std::abs(fast.beta[j] - general.coefficients[j]) < 1e-6);
    // Leverages: Cholesky and QR routes, and their sum is the rank.
    Vector w(n);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = fast.mu[i] / (1 + 0.1 * fast.mu[i]);
    const auto a = detail::fit_leverage(x, w), b = detail::weighted_leverage(x, w);
    double total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        BOOST_TEST(a[i] == b[i], boost::test_tools::tolerance(1e-10));
        total += a[i];
    }
    BOOST_TEST(total == 3.0, boost::test_tools::tolerance(1e-10));
    // A group of zeros: the fit approaches the boundary, the other group's
    // observations take leverage 1 and the zeros' fitted means vanish.
    Matrix g(8, 2, 1);
    Vector z = {0, 0, 0, 0, 5, 9, 7, 12};
    for (std::size_t i = 0; i < 8; ++i)
        g(i, 1) = i >= 4;
    const detail::Cholesky gs(detail::crossprod(g, Vector(8, 1)), 1e-12, false);
    const auto boundary = detail::nb_fixed_fit(g, z, Vector(8, 0), 0.05, {}, gs);
    BOOST_REQUIRE(boundary.ok);
    for (std::size_t i = 0; i < 4; ++i)
        BOOST_TEST(boundary.mu[i] < 1e-8);
    Vector gw(8);
    for (std::size_t i = 0; i < 8; ++i)
        gw[i] = boundary.mu[i] / (1 + 0.05 * boundary.mu[i]);
    const auto h = detail::fit_leverage(g, gw);
    double rest = 0;
    for (std::size_t i = 0; i < 8; ++i)
        rest += h[i];
    BOOST_TEST(rest == 2.0, boost::test_tools::tolerance(1e-6));
}

BOOST_AUTO_TEST_CASE(r_lowess_reproduces_lines) {
    Vector x, y;
    for (int i = 0; i < 300; ++i) {
        x.push_back(std::sin(i) * 5);
        y.push_back(2 - 0.5 * x.back());
    }
    for (double delta : {0.0, 0.1}) {
        const auto fit = detail::r_lowess(x, y, 0.5, 3, delta);
        for (std::size_t i = 0; i < x.size(); ++i)
            BOOST_TEST(fit[i] == y[i], boost::test_tools::tolerance(1e-9));
        const auto weighted = detail::weighted_lowess(x, y, Vector(x.size(), 1.5), 0.4, delta);
        for (std::size_t i = 0; i < x.size(); ++i)
            BOOST_TEST(weighted[i] == y[i], boost::test_tools::tolerance(1e-9));
    }
}

BOOST_AUTO_TEST_CASE(edger_ql_finds_changed_genes_and_is_reproducible) {
    auto b = bulk(4, 1200, 120, 0.08, 21);
    // A gene of zeros and a gene with reads in one sample only are filtered.
    b.counts.column_ptr.push_back(b.counts.values.size());
    b.counts.row_index.push_back(0);
    b.counts.values.push_back(40);
    b.counts.column_ptr.push_back(b.counts.values.size());
    b.counts.cols += 2;
    QlOptions o;
    o.threads = 1;
    const auto r = edger_ql(b.x, b.counts, 1, o);
    BOOST_TEST(!r.genes[1200].kept);
    BOOST_TEST(!r.genes[1201].kept);
    BOOST_TEST(r.minimum_group == 4.0, boost::test_tools::tolerance(1e-12));
    double log_factors = 0;
    for (double f : r.norm_factors)
        log_factors += std::log(f);
    BOOST_TEST(log_factors == 0.0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(r.dispersion > 0.04);
    BOOST_TEST(r.dispersion < 0.16);
    BOOST_TEST(r.df_prior > 1);
    std::size_t true_hits = 0, false_hits = 0;
    for (std::size_t g = 0; g < 1200; ++g)
        if (r.genes[g].fdr <= 0.1)
            (g < 120 ? true_hits : false_hits) += 1;
    BOOST_TEST(true_hits > 80u);
    BOOST_TEST(false_hits <= 0.1 * (true_hits + false_hits) + 5);
    for (std::size_t g = 0; g < 120; ++g)
        if (r.genes[g].kept)
            BOOST_TEST(r.genes[g].log_fc > 0.5);
    // Same results with several threads.
    o.threads = 4;
    const auto again = edger_ql(b.x, b.counts, 1, o);
    BOOST_TEST(again.dispersion == r.dispersion);
    BOOST_TEST(again.df_prior == r.df_prior);
    for (std::size_t g = 0; g < r.genes.size(); ++g) {
        BOOST_TEST((again.genes[g].p_value == r.genes[g].p_value ||
                    (std::isnan(again.genes[g].p_value) && std::isnan(r.genes[g].p_value))));
        BOOST_TEST((again.genes[g].log_fc == r.genes[g].log_fc ||
                    (std::isnan(again.genes[g].log_fc) && std::isnan(r.genes[g].log_fc))));
    }
    // A contrast equal to the coefficient gives the same test.
    const auto contrast = edger_ql(b.x, b.counts, Hypothesis::contrast({0, 1}), o);
    for (std::size_t g = 0; g < 50; ++g)
        if (r.genes[g].kept)
            BOOST_TEST(contrast.genes[g].p_value == r.genes[g].p_value,
                       boost::test_tools::tolerance(1e-9));
}
