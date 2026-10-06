#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <random>
#include <rapidglm/rapidglm.hpp>
#include <rapidglm/rapidglm_c.h>
using namespace rapidglm;
namespace {
struct Csc {
    SparseCSC s;
    rapidglm_sparse view() const {
        return {s.rows, s.cols, s.column_ptr.data(), s.row_index.data(), s.values.data()};
    }
};
Csc columns(const std::vector<Vector> &columns) {
    Csc c;
    c.s.rows = columns.front().size();
    c.s.cols = columns.size();
    c.s.column_ptr = {0};
    for (const auto &v : columns) {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (v[i] != 0) {
                c.s.row_index.push_back(i);
                c.s.values.push_back(v[i]);
            }
        c.s.column_ptr.push_back(c.s.values.size());
    }
    return c;
}
rapidglm_dense view(const Matrix &x) {
    return {x.rows, x.cols, x.values.data()};
}
// Numbers compare bitwise; NaN matches NaN.
bool same(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || a == b;
}
struct Groups {
    Matrix x;
    std::vector<Vector> y;
};
Groups groups(std::uint64_t seed, std::size_t genes, std::size_t reps) {
    std::mt19937_64 rng(seed);
    Groups g;
    g.x = Matrix(2 * reps, 3, 0);
    for (std::size_t i = 0; i < 2 * reps; ++i) {
        g.x(i, 0) = 1;
        g.x(i, 1) = i >= reps;
        g.x(i, 2) = std::normal_distribution<double>()(rng);
    }
    for (std::size_t j = 0; j < genes; ++j) {
        Vector v(2 * reps);
        const double base = 5 + 20 * (j % 11), effect = j % 5 == 0 ? 2.0 : 1.0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            std::gamma_distribution<double> gamma(10, base * (i >= reps ? effect : 1.0) / 10);
            v[i] = std::poisson_distribution<int>(gamma(rng))(rng);
        }
        g.y.push_back(v);
    }
    return g;
}
} // namespace
BOOST_AUTO_TEST_CASE(c_api_fit_batch_matches_cpp) {
    auto g = groups(1, 40, 6);
    auto y = columns(g.y);
    rapidglm_fit_options o;
    rapidglm_fit_options_init(&o);
    rapidglm_table *t = nullptr;
    const auto x = view(g.x);
    const auto r = y.view();
    BOOST_REQUIRE_EQUAL(rapidglm_fit_batch(&x, &r, &o, 2, &t), RAPIDGLM_OK);
    const auto fits = fit_batch(g.x, y.s, FitOptions{}, 2);
    BOOST_REQUIRE_EQUAL(rapidglm_table_rows(t), fits.size());
    const double *theta = rapidglm_table_numeric(t, "theta"),
                 *beta1 = rapidglm_table_numeric(t, "beta1"),
                 *se2 = rapidglm_table_numeric(t, "se2");
    BOOST_REQUIRE(theta && beta1 && se2);
    for (std::size_t i = 0; i < fits.size(); ++i) {
        BOOST_TEST(same(theta[i], fits[i].theta));
        BOOST_TEST(same(beta1[i], fits[i].coefficients[1]));
        BOOST_TEST(same(se2[i], fits[i].standard_errors[2]));
        BOOST_TEST(std::string(rapidglm_table_text(t, "status", i)) == status_name(fits[i].status));
    }
    BOOST_TEST(rapidglm_table_numeric(t, "no_such_column") == nullptr);
    rapidglm_table_free(t);
}
BOOST_AUTO_TEST_CASE(c_api_bulk_de_matches_cpp) {
    auto g = groups(2, 300, 4);
    auto y = columns(g.y);
    const auto x = view(g.x);
    const auto r = y.view();
    // Default (last column) with shrinkage, then a contrast.
    rapidglm_bulk_options o;
    rapidglm_bulk_options_init(&o);
    o.coefficient = 1;
    o.lfc_shrinkage = RAPIDGLM_SHRINK_NORMAL;
    rapidglm_table *t = nullptr;
    BOOST_REQUIRE_EQUAL(rapidglm_bulk_de(&x, &r, &o, &t), RAPIDGLM_OK);
    BulkOptions b;
    b.coefficient = 1;
    b.lfc_shrinkage = LfcShrinkage::normal;
    const auto expected = bulk_de(g.x, y.s, b);
    const double *p = rapidglm_table_numeric(t, "pvalue"),
                 *shrunk = rapidglm_table_numeric(t, "log2FoldChangeShrunk"),
                 *disp = rapidglm_table_numeric(t, "dispersion");
    for (std::size_t j = 0; j < expected.genes.size(); ++j) {
        BOOST_TEST(same(p[j], expected.genes[j].p_value));
        BOOST_TEST(same(shrunk[j], expected.genes[j].shrunken_lfc));
        BOOST_TEST(same(disp[j], expected.genes[j].dispersion));
        BOOST_TEST(std::string(rapidglm_table_text(t, "status", j)) == expected.genes[j].status);
    }
    BOOST_TEST(same(rapidglm_table_scalar(t, "prior_variance"), expected.prior_variance));
    std::size_t n = 0;
    const double *sf = rapidglm_table_vector(t, "size_factors", &n);
    BOOST_REQUIRE_EQUAL(n, expected.size_factors.size());
    for (std::size_t i = 0; i < n; ++i)
        BOOST_TEST(same(sf[i], expected.size_factors[i]));
    // Re-adjusting the table at its own alpha reproduces padj.
    {
        const auto rows = rapidglm_table_rows(t);
        std::vector<double> padj(rows);
        double threshold = 0;
        BOOST_REQUIRE_EQUAL(rapidglm_bulk_adjust(p, rapidglm_table_numeric(t, "baseMean"), rows,
                                                 0.1, 1, padj.data(), &threshold),
                            RAPIDGLM_OK);
        const double *q = rapidglm_table_numeric(t, "padj");
        for (std::size_t j = 0; j < rows; ++j)
            BOOST_TEST(same(padj[j], q[j]));
        BOOST_TEST(same(threshold, rapidglm_table_scalar(t, "filter_threshold")));
    }
    rapidglm_table_free(t);
    const double contrast[] = {0, 1, -1};
    rapidglm_bulk_options_init(&o);
    o.contrasts = contrast;
    o.contrast_rows = 1;
    BOOST_REQUIRE_EQUAL(rapidglm_bulk_de(&x, &r, &o, &t), RAPIDGLM_OK);
    BulkOptions c;
    c.hypothesis = Hypothesis::contrast({0, 1, -1});
    const auto with_contrast = bulk_de(g.x, y.s, c);
    p = rapidglm_table_numeric(t, "pvalue");
    for (std::size_t j = 0; j < with_contrast.genes.size(); ++j)
        BOOST_TEST(same(p[j], with_contrast.genes[j].p_value));
    rapidglm_table_free(t);
}
BOOST_AUTO_TEST_CASE(c_api_scan_and_sceptre_match_cpp) {
    std::mt19937_64 rng(3);
    const std::size_t n = 300;
    Matrix x(n, 2, 1);
    std::vector<Vector> ys(3, Vector(n)), guides(3, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 1) = std::normal_distribution<double>()(rng);
        for (auto &gv : guides)
            gv[i] = std::uniform_real_distribution<double>()(rng) < 0.2;
        for (std::size_t r = 0; r < 3; ++r) {
            const double mu = std::exp(1 + 0.3 * x(i, 1) + (r == 0 ? 0.7 * guides[0][i] : 0));
            ys[r][i] = std::poisson_distribution<int>(mu)(rng);
        }
    }
    auto y = columns(ys), g = columns(guides);
    const auto xv = view(x);
    const auto yv = y.view(), gv = g.view();
    rapidglm_scan_options so;
    rapidglm_scan_options_init(&so);
    so.model.family = RAPIDGLM_POISSON;
    const std::size_t budgets[] = {199};
    so.budgets = budgets;
    so.n_budgets = 1;
    rapidglm_table *t = nullptr;
    BOOST_REQUIRE_EQUAL(rapidglm_scan(&xv, &yv, &gv, 0, nullptr, nullptr, nullptr, &so, &t),
                        RAPIDGLM_OK);
    ScanOptions o;
    o.model.family = Family::poisson;
    o.resampling.budgets = {199};
    const auto expected = score_scan(x, y.s, g.s, {}, o);
    BOOST_REQUIRE_EQUAL(rapidglm_table_rows(t), expected.size());
    const double *p = rapidglm_table_numeric(t, "p_value");
    for (std::size_t k = 0; k < expected.size(); ++k)
        BOOST_TEST(same(p[k], expected[k].p_value));
    rapidglm_table_free(t);
    // The SCEPTRE workflow on the same data: three guides, one per target.
    const char *ids[] = {"g1", "g2", "g3"}, *targets[] = {"A", "B", "C"};
    const int nt[] = {0, 0, 0};
    const std::size_t pair_responses[] = {0, 1, 2};
    const char *pair_targets[] = {"A", "B", "C"};
    const int kinds[] = {RAPIDGLM_PAIR_DISCOVERY, RAPIDGLM_PAIR_DISCOVERY, RAPIDGLM_PAIR_DISCOVERY};
    rapidglm_sceptre_options sc;
    rapidglm_sceptre_options_init(&sc, 1);
    sc.cell_qc = 0;
    sc.min_nonzero = 1;
    sc.budgets = budgets;
    sc.n_budgets = 1;
    BOOST_REQUIRE_EQUAL(rapidglm_sceptre(&xv, &yv, &gv, 3, ids, targets, nt, 3, pair_responses,
                                         pair_targets, kinds, &sc, &t),
                        RAPIDGLM_OK);
    SceptreOptions s;
    s.high_moi = true;
    s.nt_controls = false;
    s.crt = true;
    s.cell_qc = false;
    s.min_nonzero = 1;
    s.resampling.budgets = {199};
    const auto sceptre_expected =
        sceptre(x, y.s, g.s, {{"g1", "A", false}, {"g2", "B", false}, {"g3", "C", false}},
                {{0, "A"}, {1, "B"}, {2, "C"}}, {}, s);
    BOOST_REQUIRE_EQUAL(rapidglm_table_rows(t), 3u);
    BOOST_REQUIRE_EQUAL(sceptre_expected.tests.size(), 3u);
    p = rapidglm_table_numeric(t, "p_value");
    for (std::size_t k = 0; k < 3; ++k)
        BOOST_TEST(same(p[k], sceptre_expected.tests[k].p_value));
    rapidglm_table_free(t);
}
BOOST_AUTO_TEST_CASE(c_api_reports_errors) {
    rapidglm_table *t = nullptr;
    rapidglm_bulk_options o;
    rapidglm_bulk_options_init(&o);
    const double values[] = {1, 0, 1, 1};
    rapidglm_dense x{2, 2, values};
    BOOST_TEST(rapidglm_bulk_de(&x, nullptr, &o, &t) != RAPIDGLM_OK);
    BOOST_TEST(std::string(rapidglm_last_error()).size() > 0);
    BOOST_TEST(t == nullptr);
    o.struct_size = 0;
    const std::size_t ptr[] = {0, 0};
    rapidglm_sparse counts{2, 1, ptr, nullptr, nullptr};
    BOOST_TEST(rapidglm_bulk_de(&x, &counts, &o, &t) == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("_init") != std::string::npos);
    BOOST_TEST(rapidglm_c_api_version() == RAPIDGLM_C_API_VERSION);
    BOOST_TEST(std::string(rapidglm_version()) == rapidglm::version);
}
BOOST_AUTO_TEST_CASE(c_api_older_option_structs_take_defaults) {
    // A caller built against a header without the trailing fields passes a shorter
    // struct_size; whatever follows in memory must not be read.
    auto g = groups(4, 60, 4);
    auto y = columns(g.y);
    const auto x = view(g.x);
    const auto r = y.view();
    rapidglm_bulk_options current, older;
    rapidglm_bulk_options_init(&current);
    rapidglm_bulk_options_init(&older);
    older.struct_size = offsetof(rapidglm_bulk_options, min_dispersion);
    older.min_dispersion = -1; // garbage beyond the older struct
    older.max_dispersion = -1;
    rapidglm_table *a = nullptr, *b = nullptr;
    BOOST_REQUIRE_EQUAL(rapidglm_bulk_de(&x, &r, &current, &a), RAPIDGLM_OK);
    BOOST_REQUIRE_EQUAL(rapidglm_bulk_de(&x, &r, &older, &b), RAPIDGLM_OK);
    const double *pa = rapidglm_table_numeric(a, "pvalue"),
                 *pb = rapidglm_table_numeric(b, "pvalue");
    for (std::size_t j = 0; j < rapidglm_table_rows(a); ++j)
        BOOST_TEST(same(pa[j], pb[j]));
    rapidglm_table_free(a);
    rapidglm_table_free(b);
    older.struct_size = sizeof(rapidglm_bulk_options) + 8;
    BOOST_TEST(rapidglm_bulk_de(&x, &r, &older, &b) == RAPIDGLM_ERROR_INVALID);
}
BOOST_AUTO_TEST_CASE(c_api_nb_building_blocks_match_the_library) {
    auto g = groups(5, 40, 5);
    const std::size_t n = g.x.rows, genes = g.y.size(), p = g.x.cols;
    Matrix counts(n, genes);
    for (std::size_t j = 0; j < genes; ++j)
        for (std::size_t i = 0; i < n; ++i)
            counts(i, j) = g.y[j][i];
    Vector sf(n), disp(genes);
    for (std::size_t i = 0; i < n; ++i)
        sf[i] = 0.8 + 0.05 * static_cast<double>(i);
    for (std::size_t j = 0; j < genes; ++j)
        disp[j] = 0.05 + 0.01 * static_cast<double>(j % 7);
    const auto x = view(g.x), y = view(counts);
    std::vector<double> beta(genes * p), mu(n * genes), hat(n * genes);
    std::vector<int> converged(genes);
    BOOST_REQUIRE_EQUAL(rapidglm_nb_fixed_fits(&x, &y, sf.data(), disp.data(), 1e-8, 1000, -30, 30,
                                               2, beta.data(), mu.data(), hat.data(),
                                               converged.data()),
                        RAPIDGLM_OK);
    const auto expected = nb_fixed_fits(g.x, counts, sf, disp, 1e-8, 1000, -30, 30, 1);
    for (std::size_t k = 0; k < beta.size(); ++k)
        BOOST_TEST(same(beta[k], expected.coefficients.values[k]));
    // Hat values of a full-rank fit sum to the number of columns.
    for (std::size_t j = 0; j < genes; ++j) {
        BOOST_REQUIRE(converged[j]);
        double trace = 0;
        for (std::size_t i = 0; i < n; ++i)
            trace += hat[i * genes + j];
        BOOST_TEST(trace == static_cast<double>(p), boost::test_tools::tolerance(1e-9));
    }
    // Each coefficient vector is the fixed-dispersion MLE fitted directly.
    FitOptions fo;
    fo.dispersion = Dispersion::fixed;
    fo.theta = 1 / disp[3];
    fo.offset = Vector(n);
    for (std::size_t i = 0; i < n; ++i)
        fo.offset[i] = std::log(sf[i]);
    const auto direct = fit(g.x, g.y[3], fo);
    for (std::size_t k = 0; k < p; ++k)
        BOOST_TEST(beta[3 * p + k] == direct.coefficients[k], boost::test_tools::tolerance(1e-6));
    // Dispersions: C matches C++; the prior pulls MAP estimates towards the start.
    const rapidglm_dense m{n, genes, mu.data()};
    std::vector<double> ml(genes), map(genes);
    std::vector<int> ok(genes);
    BOOST_REQUIRE_EQUAL(rapidglm_nb_dispersions(&x, &y, &m, disp.data(), 1e-8, 10, 0, 1, 1, 2,
                                                ml.data(), ok.data()),
                        RAPIDGLM_OK);
    BOOST_REQUIRE_EQUAL(rapidglm_nb_dispersions(&x, &y, &m, disp.data(), 1e-8, 10, 0.25, 1, 1, 2,
                                                map.data(), ok.data()),
                        RAPIDGLM_OK);
    Matrix fitted_means(n, genes);
    fitted_means.values = mu;
    const auto [expected_ml, converged_ml] =
        nb_dispersions(g.x, counts, fitted_means, disp, 1e-8, 10, 0, true, 1);
    for (std::size_t j = 0; j < genes; ++j) {
        BOOST_TEST(same(ml[j], expected_ml[j]));
        BOOST_TEST(std::abs(std::log(map[j] / disp[j])) <=
                   std::abs(std::log(ml[j] / disp[j])) + 1e-12);
    }
}
