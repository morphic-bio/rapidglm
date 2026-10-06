#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
bool close(double a, double b) {
    return std::abs(a - b) <= 1e-6 * (1 + std::abs(b));
}
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
// Three groups of four samples: intercept, a (group 1) and b (group 2) indicators.
// The same model with columns (1, a, a + b) has beta_a - beta_b as its column 1.
struct Groups {
    Matrix x, reparametrized;
    SparseCSC y;
};
Groups groups(std::uint64_t seed, std::size_t features = 60) {
    std::mt19937_64 rng(seed);
    const std::size_t n = 12;
    Groups d;
    d.x = Matrix(n, 3);
    d.reparametrized = Matrix(n, 3);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = i / 4 == 1, b = i / 4 == 2;
        d.x(i, 0) = d.reparametrized(i, 0) = 1;
        d.x(i, 1) = d.reparametrized(i, 1) = a;
        d.x(i, 2) = b;
        d.reparametrized(i, 2) = a + b;
    }
    std::vector<Vector> y;
    for (std::size_t j = 0; j < features; ++j) {
        Vector v(n);
        const double base = 20 + 10 * (j % 7), shift = j % 3 == 0 ? 2.0 : 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double mu = base * (i / 4 == 1 ? shift : 1);
            std::gamma_distribution<double> gamma(1 / 0.1, mu * 0.1);
            v[i] = std::poisson_distribution<int>(gamma(rng))(rng);
        }
        y.push_back(v);
    }
    d.y = dense_columns(y);
    return d;
}
} // namespace
BOOST_AUTO_TEST_CASE(hypothesis_null_space_and_validation) {
    const auto h = Hypothesis::contrast({0, 1, -1, 0});
    const auto basis = h.null_space();
    BOOST_REQUIRE(basis.rows == 4u && basis.cols == 3u);
    for (std::size_t c = 0; c < 3; ++c) {
        double dot = 0;
        for (std::size_t j = 0; j < 4; ++j)
            dot += h.contrasts(0, j) * basis(j, c);
        BOOST_TEST(std::abs(dot) < 1e-12);
        for (std::size_t d = 0; d < 3; ++d) {
            double inner = 0;
            for (std::size_t j = 0; j < 4; ++j)
                inner += basis(j, c) * basis(j, d);
            BOOST_TEST(inner == (c == d ? 1.0 : 0.0), boost::test_tools::tolerance(1e-12));
        }
    }
    BOOST_TEST(Hypothesis::coefficients(4, {3, 1}).unit_columns() ==
               std::vector<std::size_t>({3, 1}));
    BOOST_TEST(h.unit_columns().empty());
    // Removing named columns keeps the others in order.
    Matrix x(2, 4);
    for (std::size_t j = 0; j < 4; ++j)
        x(0, j) = x(1, j) = j;
    const auto reduced = reduced_design(x, Hypothesis::coefficients(4, {3, 1}));
    BOOST_TEST(reduced.values == Vector({0, 2, 0, 2}));
    BOOST_CHECK_THROW(Hypothesis::coefficients(2, {0, 1}).null_space(), std::exception);
    BOOST_CHECK_THROW(reduced_design(x, Hypothesis::coefficients(4, {1, 1})), std::exception);
    Hypothesis dependent;
    dependent.contrasts = Matrix(2, 4);
    dependent.contrasts.values = {0, 1, -1, 0, 0, 2, -2, 0};
    BOOST_CHECK_THROW(dependent.null_space(), std::exception);
    BOOST_CHECK_THROW(Hypothesis::coefficient(3, 3), std::exception);
}
BOOST_AUTO_TEST_CASE(contrasts_equal_reparametrized_coefficients) {
    const auto d = groups(1);
    const auto contrast = Hypothesis::contrast({0, 1, -1});
    FitOptions o;
    // Wald and LR regression tests, then the quasi-likelihood F test.
    for (bool lr : {false, true}) {
        const auto a = regression_tests(d.x, d.y, o, contrast, lr);
        const auto b = regression_tests(d.reparametrized, d.y, o, 1, lr);
        for (std::size_t j = 0; j < a.size(); ++j) {
            BOOST_REQUIRE(a[j].test.valid);
            BOOST_TEST(a[j].test.p_value == b[j].test.p_value, boost::test_tools::tolerance(1e-6));
            BOOST_TEST(close(a[j].log2fc, b[j].log2fc));
            BOOST_TEST(close(a[j].log2fc_se, b[j].log2fc_se));
        }
    }
    const auto qa = quasi_likelihood_tests(d.x, d.y, o, contrast);
    const auto qb = quasi_likelihood_tests(d.reparametrized, d.y, o, 1);
    for (std::size_t j = 0; j < qa.tests.size(); ++j) {
        BOOST_TEST(qa.tests[j].test.p_value == qb.tests[j].test.p_value,
                   boost::test_tools::tolerance(1e-6));
        BOOST_TEST(close(qa.tests[j].log2fc, qb.tests[j].log2fc));
    }
    // A one-coefficient hypothesis reproduces the coefficient interface exactly.
    const auto u = regression_tests(d.x, d.y, o, Hypothesis::coefficient(3, 1), true);
    const auto v = regression_tests(d.x, d.y, o, 1, true);
    for (std::size_t j = 0; j < u.size(); ++j)
        BOOST_TEST(u[j].test.p_value == v[j].test.p_value);
}
BOOST_AUTO_TEST_CASE(joint_tests_have_several_degrees_of_freedom) {
    const auto d = groups(2);
    FitOptions o;
    const auto joint = Hypothesis::coefficients(3, {1, 2});
    const auto tests = regression_tests(d.x, d.y, o, joint, true);
    Matrix intercept(d.x.rows, 1, 1);
    for (std::size_t j = 0; j < tests.size(); ++j) {
        const auto y = d.y.column(j);
        const auto full = fit(d.x, y, o);
        auto null_options = o;
        const auto manual = likelihood_ratio(fit(intercept, y, null_options), full);
        BOOST_TEST(tests[j].test.statistic == manual.statistic, boost::test_tools::tolerance(1e-8));
        BOOST_TEST(tests[j].test.p_value == manual.p_value, boost::test_tools::tolerance(1e-8));
        BOOST_TEST(std::isnan(tests[j].log2fc));
    }
    BOOST_CHECK_THROW(regression_tests(d.x, d.y, o, joint, false), std::exception);
    // The QL test runs on two numerator df and is thread invariant.
    const auto one = quasi_likelihood_tests(d.x, d.y, o, joint, 1);
    const auto four = quasi_likelihood_tests(d.x, d.y, o, joint, 4);
    for (std::size_t j = 0; j < one.tests.size(); ++j) {
        BOOST_TEST(one.tests[j].test.p_value == four.tests[j].test.p_value);
        BOOST_TEST(std::isnan(one.tests[j].log2fc));
    }
    // Features with a doubled group-1 mean are found.
    std::size_t found = 0, planted = 0;
    for (std::size_t j = 0; j < one.tests.size(); ++j)
        if (j % 3 == 0) {
            ++planted;
            found += one.tests[j].test.p_value < 0.01;
        }
    BOOST_TEST(found > planted / 2);
}
BOOST_AUTO_TEST_CASE(bulk_de_contrasts_and_joint_tests) {
    const auto d = groups(3, 300);
    BulkOptions o;
    o.hypothesis = Hypothesis::contrast({0, 1, -1});
    const auto a = bulk_de(d.x, d.y, o);
    o.hypothesis = {};
    o.coefficient = 1;
    const auto b = bulk_de(d.reparametrized, d.y, o);
    for (std::size_t j = 0; j < a.genes.size(); ++j) {
        const auto &ga = a.genes[j], &gb = b.genes[j];
        BOOST_TEST(ga.status == gb.status);
        if (std::isfinite(gb.p_value)) {
            BOOST_TEST(ga.p_value == gb.p_value, boost::test_tools::tolerance(1e-6));
            BOOST_TEST(close(ga.log2_fold_change, gb.log2_fold_change));
            BOOST_TEST(close(ga.lfc_se, gb.lfc_se));
        }
    }
    // A joint LR test reports no single fold change; Wald and shrinkage refuse it.
    o.hypothesis = Hypothesis::coefficients(3, {1, 2});
    o.lrt = true;
    const auto joint = bulk_de(d.x, d.y, o);
    std::size_t tested = 0;
    for (const auto &g : joint.genes) {
        tested += std::isfinite(g.p_value);
        BOOST_TEST(std::isnan(g.log2_fold_change));
    }
    BOOST_TEST(tested > 250u);
    o.lrt = false;
    BOOST_CHECK_THROW(bulk_de(d.x, d.y, o), std::exception);
    o.hypothesis = Hypothesis::contrast({0, 1, -1});
    o.lfc_shrinkage = LfcShrinkage::normal;
    BOOST_CHECK_THROW(bulk_de(d.x, d.y, o), std::exception);
}
