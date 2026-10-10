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
namespace {
// A screen at high MOI: targets A (two guides), B and C, one non-targeting
// guide, and five responses. Response 0 falls in the cells of A; response 4
// has too few nonzero cells for any pair to pass pair QC.
struct Screen {
    Matrix x;
    Csc y, g;
    std::vector<Guide> guides;
    std::vector<Pair> pairs; // every target x response, and a positive control
    std::vector<const char *> ids, targets, pair_targets;
    std::vector<int> non_targeting, kinds;
    std::vector<std::size_t> pair_responses;
};
Screen screen() {
    std::mt19937_64 rng(8);
    const std::size_t n = 400, genes = 5;
    Screen s;
    s.x = Matrix(n, 2, 1);
    std::vector<Vector> ys(genes, Vector(n)), calls(5, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        s.x(i, 1) = std::normal_distribution<double>()(rng);
        for (auto &call : calls)
            call[i] = std::uniform_real_distribution<double>()(rng) < 0.2;
        for (std::size_t r = 0; r + 1 < genes; ++r) {
            const bool a = calls[0][i] || calls[1][i];
            const double mu = std::exp(0.8 + 0.3 * s.x(i, 1) - (r == 0 && a ? 0.9 : 0));
            ys[r][i] = std::poisson_distribution<int>(mu)(rng);
        }
        ys[genes - 1][i] = i % 40 == 0 ? 2 : 0;
    }
    s.y = columns(ys);
    s.g = columns(calls);
    s.guides = {{"a1", "A", false},
                {"a2", "A", false},
                {"b1", "B", false},
                {"c1", "C", false},
                {"nt1", "NT", true}};
    for (std::size_t r = 0; r < genes; ++r) {
        for (const char *target : {"A", "B", "C"})
            s.pairs.push_back({r, target, PairKind::discovery});
        s.pairs.push_back({r, "NT", PairKind::negative_control});
    }
    s.pairs.push_back({0, "A", PairKind::positive_control});
    return s;
}
// The C arrays of the guides and pairs; they point into `s`.
void c_arrays(Screen &s) {
    s.ids.clear();
    s.targets.clear();
    s.non_targeting.clear();
    for (const auto &guide : s.guides) {
        s.ids.push_back(guide.id.c_str());
        s.targets.push_back(guide.target.c_str());
        s.non_targeting.push_back(guide.non_targeting);
    }
    s.pair_responses.clear();
    s.pair_targets.clear();
    s.kinds.clear();
    for (const auto &pair : s.pairs) {
        s.pair_responses.push_back(pair.response);
        s.pair_targets.push_back(pair.target.c_str());
        s.kinds.push_back(static_cast<int>(pair.kind));
    }
}
void screen_options(rapidglm_sceptre_options &c, SceptreOptions &o) {
    static const std::size_t budgets[] = {199, 999};
    rapidglm_sceptre_options_init(&c, 1);
    c.cell_qc = 0;
    c.budgets = budgets;
    c.n_budgets = 2;
    c.screen = 0.2;
    c.alpha = 0.3;
    c.seed = 5;
    c.threads = 2;
    o = SceptreOptions();
    o.high_moi = true;
    o.nt_controls = false;
    o.crt = true;
    o.cell_qc = false;
    o.resampling.budgets = {199, 999};
    o.resampling.screen = 0.2;
    o.alpha = 0.3;
    o.resampling.seed = 5;
    o.threads = 2;
}
// Every column and scalar of a sceptre table against the C++ result.
void check_sceptre_table(const rapidglm_table *t, const SceptreResult &r) {
    BOOST_REQUIRE_EQUAL(rapidglm_table_rows(t), r.tests.size());
    BOOST_REQUIRE_EQUAL(rapidglm_table_columns(t), 17u);
    const auto number = [&](const char *name, std::size_t i) {
        const double *v = rapidglm_table_numeric(t, name);
        BOOST_REQUIRE(v != nullptr);
        return v[i];
    };
    const auto text = [&](const char *name, std::size_t i) {
        const char *v = rapidglm_table_text(t, name, i);
        BOOST_REQUIRE(v != nullptr);
        return std::string(v);
    };
    for (std::size_t i = 0; i < r.tests.size(); ++i) {
        const auto &e = r.tests[i];
        BOOST_TEST_CONTEXT("row " << i) {
            BOOST_TEST(number("response", i) == static_cast<double>(e.response));
            BOOST_TEST(text("target", i) == e.target);
            BOOST_TEST(text("guide", i) == e.guide);
            BOOST_TEST(number("kind", i) == static_cast<int>(e.kind));
            BOOST_TEST(text("status", i) == e.status);
            BOOST_TEST(number("cells", i) == static_cast<double>(e.cells));
            BOOST_TEST(number("nonzero_treatment", i) == static_cast<double>(e.nonzero_treatment));
            BOOST_TEST(number("nonzero_control", i) == static_cast<double>(e.nonzero_control));
            BOOST_TEST(number("draws", i) == static_cast<double>(e.draws));
            BOOST_TEST(number("approximation", i) == e.approximation);
            BOOST_TEST(same(number("statistic", i), e.statistic));
            BOOST_TEST(same(number("p_value", i), e.p_value));
            BOOST_TEST(same(number("adjusted_p", i), e.adjusted_p));
            BOOST_TEST(same(number("log2fc", i), e.log2fc));
            BOOST_TEST(same(number("log2fc_se", i), e.log2fc_se));
            BOOST_TEST(same(number("sceptre_mean_ratio", i), e.sceptre_mean_ratio));
            BOOST_TEST(same(number("mean_ratio_log2", i), e.mean_ratio_log2));
        }
    }
    const auto scalar = [&](const std::string &name) {
        return rapidglm_table_scalar(t, name.c_str());
    };
    BOOST_TEST(scalar("retained_cells") == static_cast<double>(r.retained_cells));
    BOOST_TEST(scalar("calibration_rejections") == static_cast<double>(r.calibration_rejections));
    BOOST_TEST(scalar("power_rejections") == static_cast<double>(r.power_rejections));
    BOOST_TEST(scalar("discovery_rejections") == static_cast<double>(r.discovery_rejections));
    BOOST_TEST(scalar("null_fits") == static_cast<double>(r.null_fits));
    const char *const kinds[] = {"discovery", "negative", "positive"};
    for (std::size_t k = 0; k < 3; ++k) {
        BOOST_TEST(scalar(std::string("pairs_listed_") + kinds[k]) ==
                   static_cast<double>(r.pairs_listed[k]));
        BOOST_TEST(scalar(std::string("pairs_passing_qc_") + kinds[k]) ==
                   static_cast<double>(r.pairs_passing_qc[k]));
        BOOST_TEST(scalar(std::string("family_size_") + kinds[k]) ==
                   static_cast<double>(r.family_size[k]));
    }
    BOOST_TEST(rapidglm_table_scalars(t) == 14u);
}
} // namespace
BOOST_AUTO_TEST_CASE(c_api_sceptre_stream_keys_and_families_match_cpp) {
    auto s = screen();
    c_arrays(s);
    const auto xv = view(s.x);
    const auto yv = s.y.view(), gv = s.g.view();
    const char *response_ids[] = {"gene0", "gene1", "gene2", "gene3", "gene4"};
    const auto run = [&](const rapidglm_sceptre_options &c) {
        rapidglm_table *t = nullptr;
        BOOST_REQUIRE_EQUAL(rapidglm_sceptre(&xv, &yv, &gv, s.guides.size(), s.ids.data(),
                                             s.targets.data(), s.non_targeting.data(),
                                             s.pairs.size(), s.pair_responses.data(),
                                             s.pair_targets.data(), s.kinds.data(), &c, &t),
                            RAPIDGLM_OK);
        return t;
    };
    rapidglm_sceptre_options c;
    SceptreOptions o;
    // The defaults: identity keys, the tested family.
    screen_options(c, o);
    BOOST_TEST(c.stream_keys == RAPIDGLM_STREAMS_IDENTITY);
    BOOST_TEST(c.response_ids == nullptr);
    BOOST_TEST(c.family == RAPIDGLM_FAMILY_TESTED);
    BOOST_TEST((c.family_size[0] == 0 && c.family_size[1] == 0 && c.family_size[2] == 0));
    auto *defaults = run(c);
    const auto expected = sceptre(s.x, s.y.s, s.g.s, s.guides, s.pairs, {}, o);
    check_sceptre_table(defaults, expected);
    // Pairs that fail pair QC are outside the tested family; others are resampled.
    const double *q = rapidglm_table_numeric(defaults, "adjusted_p");
    std::size_t failed = 0, resampled = 0;
    for (std::size_t i = 0; i < expected.tests.size(); ++i) {
        const bool fails =
            std::string(rapidglm_table_text(defaults, "status", i)) == "failed_pair_qc";
        BOOST_TEST(std::isnan(q[i]) == fails);
        failed += fails;
        resampled += expected.tests[i].draws > 199;
    }
    BOOST_TEST(failed == 4u);
    BOOST_TEST(resampled > 0u);
    BOOST_TEST(rapidglm_table_scalar(defaults, "pairs_listed_discovery") == 15);
    BOOST_TEST(rapidglm_table_scalar(defaults, "family_size_discovery") == 12);
    // Each new option, and all of them together.
    for (int mode = 0; mode < 5; ++mode) {
        BOOST_TEST_CONTEXT("mode " << mode) {
            screen_options(c, o);
            if (mode == 0 || mode == 4) {
                c.stream_keys = RAPIDGLM_STREAMS_POSITION;
                o.stream_keys = StreamKeys::position;
            }
            if (mode == 1) {
                c.response_ids = response_ids;
                o.response_ids.assign(response_ids, response_ids + 5);
            }
            if (mode == 2 || mode == 4) {
                c.family = RAPIDGLM_FAMILY_LISTED;
                o.family = AdjustmentFamily::listed;
            }
            if (mode == 3 || mode == 4) {
                c.family_size[0] = 40;
                c.family_size[2] = 3;
                o.family_size = {40, 0, 3};
            }
            auto *t = run(c);
            const auto r = sceptre(s.x, s.y.s, s.g.s, s.guides, s.pairs, {}, o);
            check_sceptre_table(t, r);
            // The option changes the table.
            const double *p = rapidglm_table_numeric(t, "p_value"),
                         *p0 = rapidglm_table_numeric(defaults, "p_value"),
                         *q0 = rapidglm_table_numeric(defaults, "adjusted_p");
            q = rapidglm_table_numeric(t, "adjusted_p");
            std::size_t p_differs = 0, q_differs = 0;
            for (std::size_t i = 0; i < r.tests.size(); ++i) {
                p_differs += !same(p[i], p0[i]);
                q_differs += !same(q[i], q0[i]);
            }
            BOOST_TEST((p_differs > 0) == (mode <= 1 || mode == 4));
            BOOST_TEST(q_differs > 0u);
            rapidglm_table_free(t);
        }
    }
    // A caller built against the header of 1.2.0 passes a struct that ends
    // before stream_keys: the fields beyond it take the defaults, whatever
    // follows in memory.
    const char *repeated_ids[] = {"x", "x", "x", "x", "x"};
    screen_options(c, o);
    c.struct_size = offsetof(rapidglm_sceptre_options, stream_keys);
    c.stream_keys = 7;
    c.response_ids = repeated_ids;
    c.family = RAPIDGLM_FAMILY_LISTED;
    c.family_size[0] = c.family_size[1] = c.family_size[2] = 1;
    auto *older = run(c);
    check_sceptre_table(older, expected);
    rapidglm_table_free(older);
    // The same fields are read, and checked, from a struct of this version.
    c.struct_size = sizeof c;
    rapidglm_table *t = nullptr;
    const auto call = [&] {
        return rapidglm_sceptre(&xv, &yv, &gv, s.guides.size(), s.ids.data(), s.targets.data(),
                                s.non_targeting.data(), s.pairs.size(), s.pair_responses.data(),
                                s.pair_targets.data(), s.kinds.data(), &c, &t);
    };
    BOOST_TEST(call() == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()) == "unknown stream_keys");
    c.stream_keys = RAPIDGLM_STREAMS_IDENTITY;
    BOOST_TEST(call() == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("response IDs must be distinct") !=
               std::string::npos);
    c.response_ids = nullptr;
    BOOST_TEST(call() == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("family_size is smaller") !=
               std::string::npos);
    c.family_size[0] = c.family_size[1] = c.family_size[2] = 0;
    c.family = 2;
    BOOST_TEST(call() == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()) == "unknown adjustment family");
    // With identity keys a pair is listed once within a kind.
    c.family = RAPIDGLM_FAMILY_TESTED;
    s.pairs.push_back(s.pairs[1]);
    c_arrays(s);
    BOOST_TEST(call() == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("listed twice") != std::string::npos);
    c.stream_keys = RAPIDGLM_STREAMS_POSITION;
    BOOST_TEST(call() == RAPIDGLM_OK);
    BOOST_TEST(rapidglm_table_rows(t) == s.pairs.size());
    rapidglm_table_free(t);
    rapidglm_table_free(defaults);
}
BOOST_AUTO_TEST_CASE(c_api_sceptre_grid_matches_cpp) {
    auto s = screen();
    // The grid lists A, B and C against the responses; the positive control is listed.
    s.pairs = {{0, "A", PairKind::positive_control}};
    c_arrays(s);
    const auto xv = view(s.x);
    const auto yv = s.y.view(), gv = s.g.view();
    const char *discovery[] = {"A", "B", "C"}, *negative[] = {"NT"};
    const std::size_t some[] = {3, 0, 4};
    rapidglm_sceptre_options c;
    SceptreOptions o;
    rapidglm_table *t = nullptr, *tally = nullptr;
    const auto call = [&](std::size_t n_responses, const std::size_t *grid_responses,
                          rapidglm_table **counts) {
        return rapidglm_sceptre_grid(&xv, &yv, &gv, s.guides.size(), s.ids.data(), s.targets.data(),
                                     s.non_targeting.data(), 3, discovery, 1, negative, n_responses,
                                     grid_responses, s.pairs.size(), s.pair_responses.data(),
                                     s.pair_targets.data(), s.kinds.data(), &c, &t, counts);
    };
    for (bool all_responses : {true, false})
        for (int family : {RAPIDGLM_FAMILY_TESTED, RAPIDGLM_FAMILY_LISTED})
            for (int integration : {RAPIDGLM_INTEGRATION_UNION, RAPIDGLM_INTEGRATION_BONFERRONI}) {
                BOOST_TEST_CONTEXT("all responses " << all_responses << " family " << family
                                                    << " integration " << integration) {
                    screen_options(c, o);
                    c.family = family;
                    o.family = family == RAPIDGLM_FAMILY_LISTED ? AdjustmentFamily::listed
                                                                : AdjustmentFamily::tested;
                    c.integration = integration;
                    o.integration = integration == RAPIDGLM_INTEGRATION_BONFERRONI
                                        ? Integration::bonferroni
                                        : Integration::union_guides;
                    SceptreGrid grid;
                    grid.discovery_targets = {"A", "B", "C"};
                    grid.negative_targets = {"NT"};
                    if (!all_responses)
                        grid.responses = {3, 0, 4};
                    grid.pairs = s.pairs;
                    const auto expected = sceptre_grid(s.x, s.y.s, s.g.s, s.guides, grid, {}, o);
                    BOOST_REQUIRE_EQUAL(
                        call(all_responses ? 0 : 3, all_responses ? nullptr : some, &tally),
                        RAPIDGLM_OK);
                    check_sceptre_table(t, expected.result);
                    // Only the pairs that pass pair QC have a row; every pair is counted.
                    const std::size_t responses = all_responses ? 5 : 3;
                    BOOST_TEST(rapidglm_table_rows(t) == 4 * (responses - 1) + 1);
                    BOOST_TEST(rapidglm_table_scalar(t, "pairs_listed_discovery") ==
                               static_cast<double>(3 * responses));
                    BOOST_TEST(rapidglm_table_scalar(t, "family_size_discovery") ==
                               static_cast<double>(3 * (responses - (family == 0))));
                    BOOST_REQUIRE_EQUAL(rapidglm_table_rows(tally), expected.tally.size());
                    BOOST_REQUIRE_EQUAL(rapidglm_table_columns(tally), 6u);
                    const double *kind = rapidglm_table_numeric(tally, "kind"),
                                 *requested = rapidglm_table_numeric(tally, "requested"),
                                 *passing = rapidglm_table_numeric(tally, "passing"),
                                 *few_treatment = rapidglm_table_numeric(tally, "few_treatment"),
                                 *few_control = rapidglm_table_numeric(tally, "few_control");
                    BOOST_REQUIRE(kind && requested && passing && few_treatment && few_control);
                    for (std::size_t i = 0; i < expected.tally.size(); ++i) {
                        const auto &e = expected.tally[i];
                        BOOST_TEST(std::string(rapidglm_table_text(tally, "target", i)) ==
                                   e.target);
                        BOOST_TEST(kind[i] == static_cast<int>(e.kind));
                        BOOST_TEST(requested[i] == static_cast<double>(e.requested));
                        BOOST_TEST(passing[i] == static_cast<double>(e.passing));
                        BOOST_TEST(few_treatment[i] == static_cast<double>(e.few_treatment));
                        BOOST_TEST(few_control[i] == static_cast<double>(e.few_control));
                        BOOST_TEST(requested[i] == static_cast<double>(responses));
                        BOOST_TEST(passing[i] == static_cast<double>(responses - 1));
                    }
                    rapidglm_table_free(t);
                    rapidglm_table_free(tally);
                    t = tally = nullptr;
                }
            }
    // The tally is optional.
    screen_options(c, o);
    BOOST_REQUIRE_EQUAL(call(0, nullptr, nullptr), RAPIDGLM_OK);
    rapidglm_table_free(t);
    t = nullptr;
    // Invalid requests leave the tables alone.
    c.stream_keys = RAPIDGLM_STREAMS_POSITION;
    BOOST_TEST(call(0, nullptr, &tally) == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("identity stream keys") !=
               std::string::npos);
    c.stream_keys = RAPIDGLM_STREAMS_IDENTITY;
    BOOST_TEST(call(3, nullptr, &tally) == RAPIDGLM_ERROR_INVALID);
    const std::size_t repeated[] = {1, 1};
    BOOST_TEST(call(2, repeated, &tally) == RAPIDGLM_ERROR_INVALID);
    s.pairs = {{2, "B", PairKind::discovery}}; // a pair of the grid, listed again
    c_arrays(s);
    BOOST_TEST(call(0, nullptr, &tally) == RAPIDGLM_ERROR_INVALID);
    BOOST_TEST(std::string(rapidglm_last_error()).find("repeats a pair of the grid") !=
               std::string::npos);
    BOOST_TEST((t == nullptr && tally == nullptr));
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
