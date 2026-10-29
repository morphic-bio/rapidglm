#include <boost/math/distributions/beta.hpp>
#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;

namespace {
// NB counts on a design with an intercept and two covariates, optionally raised
// by a variant's dosage; variants independent with allele frequency 0.1 to 0.5.
struct QtlData {
    Matrix x;
    SparseCSC responses;
    Genotypes genotypes;
    std::vector<CisPhenotype> phenotypes;
    Vector offset;
};
QtlData qtl_data(std::size_t n, std::size_t genes, std::size_t per_gene, std::uint32_t seed,
                 const std::vector<std::pair<std::size_t, std::size_t>> &causal = {}) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform(0.1, 0.5);
    QtlData d{Matrix(n, 3, 1), SparseCSC{n, genes, {0}, {}, {}}, {}, {}, Vector(n)};
    for (std::size_t i = 0; i < n; ++i) {
        d.x(i, 1) = normal(rng);
        d.x(i, 2) = normal(rng);
        d.offset[i] = std::log(1e4) + 0.3 * normal(rng);
    }
    auto &g = d.genotypes;
    g.samples = n;
    for (std::size_t f = 0; f < genes; ++f) {
        const long long anchor = 2'000'000LL * static_cast<long long>(f + 1);
        d.phenotypes.push_back({f, "gene" + std::to_string(f), "chr1", anchor, anchor});
        for (std::size_t k = 0; k < per_gene; ++k) {
            const double maf = uniform(rng);
            std::binomial_distribution<int> allele(2, maf);
            for (std::size_t i = 0; i < n; ++i)
                g.dosage.push_back(allele(rng));
            g.chrom.push_back("chr1");
            g.position.push_back(anchor - 400'000 + 20'000LL * static_cast<long long>(k));
            g.id.push_back("v" + std::to_string(g.variants));
            ++g.variants;
        }
    }
    for (std::size_t f = 0; f < genes; ++f) {
        double effect = 0;
        std::size_t variant = 0;
        for (const auto &[gene, v] : causal)
            if (gene == f) {
                effect = 0.5;
                variant = v;
            }
        const double theta = 5;
        for (std::size_t i = 0; i < n; ++i) {
            double eta = d.offset[i] - 6 + 0.3 * d.x(i, 1) - 0.2 * d.x(i, 2);
            if (effect != 0)
                eta += effect * g.dosage[variant * n + i];
            std::gamma_distribution<double> gamma(theta, std::exp(eta) / theta);
            std::poisson_distribution<int> poisson(gamma(rng));
            const int y = poisson(rng);
            if (y) {
                d.responses.row_index.push_back(i);
                d.responses.values.push_back(y);
            }
        }
        d.responses.column_ptr.push_back(d.responses.values.size());
    }
    return d;
}
CisQtlOptions qtl_options(const QtlData &d) {
    CisQtlOptions o;
    o.model.family = Family::negative_binomial;
    o.model.dispersion = Dispersion::ml;
    o.model.offset = d.offset;
    o.window = 500'000;
    o.permutations = 200;
    o.seed = 7;
    o.threads = 4;
    return o;
}
} // namespace

BOOST_AUTO_TEST_CASE(qtl_scores_equal_the_score_workspace) {
    const auto d = qtl_data(200, 1, 30, 11);
    const std::size_t n = 200, m = 30;
    std::vector<double> g(n * m);
    for (std::size_t q = 0; q < m; ++q)
        for (std::size_t i = 0; i < n; ++i)
            g[i * m + q] = d.genotypes.dosage[q * n + i];
    for (auto family : {Family::negative_binomial, Family::poisson, Family::gaussian}) {
        FitOptions options;
        options.family = family;
        options.dispersion = Dispersion::ml;
        if (family != Family::gaussian)
            options.offset = d.offset;
        options.retain_observations = true;
        Vector y = d.responses.column(0);
        if (family == Family::gaussian)
            for (auto &v : y)
                v = std::log1p(v);
        const auto null = fit(d.x, y, options);
        BOOST_REQUIRE(null.ok());
        const ScoreWorkspace workspace(d.x, null);
        std::vector<double> z, estimate, se;
        detail::qtl_scores(d.x, null, g, m, z, &estimate, &se);
        for (std::size_t q = 0; q < m; ++q) {
            Vector gq(n);
            for (std::size_t i = 0; i < n; ++i)
                gq[i] = g[i * m + q];
            const auto expected = workspace.test(gq);
            BOOST_REQUIRE(expected.valid);
            BOOST_TEST(z[q] == expected.statistic, boost::test_tools::tolerance(1e-9));
            BOOST_TEST(estimate[q] == expected.estimate, boost::test_tools::tolerance(1e-9));
            BOOST_TEST(se[q] == expected.standard_error, boost::test_tools::tolerance(1e-9));
        }
    }
}

BOOST_AUTO_TEST_CASE(beta_maximum_likelihood_recovers_its_parameters) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> uniform(0, 1);
    for (auto [k, n] : {std::pair{0.8, 40.0}, std::pair{1.0, 5.0}, std::pair{2.5, 300.0}}) {
        const boost::math::beta_distribution<> beta(k, n);
        std::vector<double> p;
        for (int b = 0; b < 40'000; ++b)
            p.push_back(boost::math::quantile(beta, uniform(rng)));
        double fk = 0, fn = 0;
        BOOST_REQUIRE(detail::beta_mle(p, fk, fn));
        BOOST_TEST(fk == k, boost::test_tools::tolerance(0.03));
        BOOST_TEST(fn == n, boost::test_tools::tolerance(0.05));
    }
}

BOOST_AUTO_TEST_CASE(cis_qtl_finds_a_planted_variant_and_is_calibrated_under_the_null) {
    const std::size_t genes = 60;
    const auto d = qtl_data(150, genes, 40, 3, {{5, 5 * 40 + 17}});
    const auto o = qtl_options(d);
    const auto result = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_REQUIRE(result.genes.size() == genes);
    // The planted gene: its lead is the causal variant and its gene-level p-value is small.
    const auto &planted = result.genes[5];
    BOOST_TEST(planted.status == "ok");
    BOOST_TEST(planted.lead == 5u * 40 + 17);
    BOOST_TEST(planted.beta_p < 1e-4);
    BOOST_TEST(planted.permutation_p == 1.0 / 201);
    // The other genes: gene-level p-values spread over (0, 1), and the Beta
    // approximation tracks the permutation p-value.
    std::vector<double> beta_p;
    for (std::size_t f = 0; f < genes; ++f) {
        if (f == 5)
            continue;
        const auto &gene = result.genes[f];
        BOOST_REQUIRE(gene.status == "ok");
        BOOST_TEST(gene.variants == 40u);
        BOOST_TEST(gene.permutations == 200u);
        BOOST_TEST(std::abs(gene.beta_p - gene.permutation_p) < 0.06);
        beta_p.push_back(gene.beta_p);
    }
    std::sort(beta_p.begin(), beta_p.end());
    double ks = 0;
    for (std::size_t k = 0; k < beta_p.size(); ++k) {
        const double e1 = static_cast<double>(k) / beta_p.size(),
                     e2 = static_cast<double>(k + 1) / beta_p.size();
        ks = std::max({ks, std::abs(beta_p[k] - e1), std::abs(beta_p[k] - e2)});
    }
    BOOST_TEST(ks < 0.25); // critical value about 0.21 at 1% for 59 genes
    // Identical results with one thread and the phenotypes in another order.
    auto reversed = d.phenotypes;
    std::reverse(reversed.begin(), reversed.end());
    auto single = o;
    single.threads = 1;
    const auto again = cis_qtl(d.x, d.responses, d.genotypes, reversed, single);
    for (std::size_t f = 0; f < genes; ++f) {
        const auto &a = result.genes[f], &b = again.genes[genes - 1 - f];
        BOOST_TEST(a.lead == b.lead);
        BOOST_TEST(a.statistic == b.statistic);
        BOOST_TEST(a.permutation_p == b.permutation_p);
        BOOST_TEST((a.beta_p == b.beta_p || (std::isnan(a.beta_p) && std::isnan(b.beta_p))));
    }
}

BOOST_AUTO_TEST_CASE(cis_windows_and_nominal_pairs) {
    auto d = qtl_data(120, 3, 25, 9);
    auto o = qtl_options(d);
    o.permutations = 0;
    o.nominal = true;
    d.phenotypes.push_back({1, "elsewhere", "chr2", 1000, 1000});
    o.window = 100'000; // variants at anchor - 400 kb + 20 kb k: none within 100 kb but k = 15..
    auto result = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_TEST(result.genes[3].status == "no_cis_variants");
    // Each gene's window holds the variants within 100 kb of its anchor.
    for (std::size_t f = 0; f < 3; ++f) {
        std::size_t expected = 0;
        for (std::size_t v = 0; v < d.genotypes.variants; ++v)
            expected += std::llabs(d.genotypes.position[v] - d.phenotypes[f].start) <= 100'000;
        BOOST_TEST(result.genes[f].variants == expected);
    }
    std::size_t pairs = 0;
    for (std::size_t f = 0; f < 3; ++f)
        pairs += result.genes[f].variants;
    BOOST_TEST(pairs > 0u);
    BOOST_TEST(result.nominal.size() == pairs);
    for (const auto &pair : result.nominal)
        BOOST_TEST(pair.p_value == normal_tail(pair.statistic),
                   boost::test_tools::tolerance(1e-12));
    BOOST_CHECK_THROW(cis_qtl(d.x, d.responses, d.genotypes, {{9, "bad", "chr1", 0, 0}}, o),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(qtl_kernels_and_reused_bases_agree) {
    // The tiled cross products, with AVX2 where the processor has it, against
    // direct sums, for variant counts that leave a partial tile.
    std::mt19937_64 rng(13);
    std::normal_distribution<double> normal;
    const std::size_t n = 37, p = 6;
    for (std::size_t m : {1u, 8u, 21u}) {
        std::vector<double> a(n * p), w(n), r(n), g(n * m);
        for (auto &v : a)
            v = normal(rng);
        for (auto &v : w)
            v = std::abs(normal(rng));
        for (auto &v : r)
            v = normal(rng);
        for (auto &v : g)
            v = std::floor(std::abs(normal(rng)) * 1.5);
        std::vector<double> c(p * m), u(m), v(m);
        detail::cross_products(a.data(), w.data(), r.data(), g.data(), n, p, m, c.data(), u.data(),
                               v.data());
        for (std::size_t q = 0; q < m; ++q) {
            double su = 0, sv = 0;
            for (std::size_t i = 0; i < n; ++i) {
                su += r[i] * g[i * m + q];
                sv += w[i] * g[i * m + q] * g[i * m + q];
            }
            BOOST_TEST(u[q] == su, boost::test_tools::tolerance(1e-12));
            BOOST_TEST(v[q] == sv, boost::test_tools::tolerance(1e-12));
            for (std::size_t k = 0; k < p; ++k) {
                double sc = 0;
                for (std::size_t i = 0; i < n; ++i)
                    sc += a[i * p + k] * g[i * m + q];
                BOOST_TEST(std::abs(c[k * m + q] - sc) < 1e-12 * (1 + std::abs(sc)));
            }
        }
    }
    // A Gaussian null's basis serves a refit to other responses: the same scores.
    const auto d = qtl_data(80, 1, 13, 17);
    const std::size_t n2 = 80, m2 = 13;
    std::vector<double> g2(n2 * m2);
    for (std::size_t q = 0; q < m2; ++q)
        for (std::size_t i = 0; i < n2; ++i)
            g2[i * m2 + q] = d.genotypes.dosage[q * n2 + i];
    FitOptions options;
    options.family = Family::gaussian;
    options.retain_observations = true;
    Vector y = d.responses.column(0), shuffled = y;
    std::reverse(shuffled.begin(), shuffled.end());
    detail::QtlBasis basis;
    std::vector<double> z1, z2, z3;
    detail::qtl_scores(d.x, fit(d.x, y, options), g2, m2, z1, nullptr, nullptr, &basis);
    BOOST_REQUIRE(basis.ready);
    const auto refit = fit(d.x, shuffled, options);
    detail::qtl_scores(d.x, refit, g2, m2, z2, nullptr, nullptr, &basis);
    detail::qtl_scores(d.x, refit, g2, m2, z3, nullptr, nullptr);
    for (std::size_t q = 0; q < m2; ++q)
        BOOST_TEST(z2[q] == z3[q], boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(qtl_singular_information_is_a_gene_status) {
    auto d = qtl_data(80, 2, 5, 741);
    for (std::size_t i = 0; i < d.x.rows; ++i)
        d.x(i, 2) = d.x(i, 1) + 1e-7 * std::sin(double(i));
    auto o = qtl_options(d);
    o.permutations = 10;
    auto r = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_TEST(r.genes.size() == 2u);
    for (const auto &g : r.genes)
        BOOST_TEST(g.status.rfind("null_", 0) == 0u);
    FitResult f;
    f.working_weights.assign(80, 1);
    f.score_residuals.assign(80, 0);
    Matrix x(80, 2, 1);
    std::vector<double> z;
    BOOST_TEST(!detail::qtl_scores(x, f, Vector(80), 1, z, nullptr, nullptr));
}

BOOST_AUTO_TEST_CASE(qtl_degenerate_beta_reports_missing_parameters) {
    for (double delta : {0., 1e-12, 1e-8}) {
        Vector maxima(100, 3);
        maxima[0] += delta;
        auto b = detail::beta_calibration(maxima);
        BOOST_TEST(!b.ok);
        BOOST_TEST(std::isnan(b.k));
        BOOST_TEST(std::isnan(b.n));
        BOOST_TEST(std::isnan(b.ncp));
        BOOST_TEST(std::isnan(detail::beta_adjust(b, 3)));
    }
    // cis_qtl's threshold for maxima from single-precision scores, a relative
    // variance of 1e-8: constant far above float rounding...
    for (double delta : {1e-6, 1e-4, 1e-3}) {
        Vector maxima(100, 3);
        maxima[0] += delta;
        const auto b = detail::beta_calibration(maxima, 1e-8);
        BOOST_TEST(!b.ok);
        BOOST_TEST(std::isnan(b.k));
    }
    // ... yet tight maxima that vary still get a fit (a relative SD of 1e-3,
    // which 64 float epsilons would have taken as constant), and those that vary
    // by a relative SD of 1e-5 do not.
    std::mt19937_64 rng(3);
    std::normal_distribution<double> normal;
    for (double sd : {1e-3, 1e-5}) {
        Vector maxima(1000);
        for (auto &v : maxima)
            v = 3 * (1 + sd * normal(rng));
        BOOST_TEST(detail::beta_calibration(maxima, 1e-8).ok == (sd > 1e-4));
        BOOST_TEST(detail::beta_calibration(maxima).ok);
    }
}

BOOST_AUTO_TEST_CASE(gaussian_qtl_batched_and_unbatched_permutations_agree) {
    auto d = qtl_data(81, 3, 25, 880);
    auto o = qtl_options(d);
    o.model.family = Family::gaussian;
    o.model.offset.clear();
    o.permutations = 137;
    o.nominal = true;
    o.permutation_precision = PermutationPrecision::double_precision;
    auto batch = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.permutation_precision = PermutationPrecision::single_precision;
    auto single = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.model.offset.assign(81, 0); // forces ordinary null refits on the same draws
    auto ordinary = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    for (std::size_t j = 0; j < 3; ++j) {
        const auto &a = batch.genes[j], &b = ordinary.genes[j], &s = single.genes[j];
        BOOST_TEST(a.status == "ok");
        BOOST_TEST(a.status == b.status);
        BOOST_TEST(a.lead == b.lead);
        BOOST_TEST(a.permutations == 137u);
        BOOST_TEST(a.permutations_failed == 0u);
        BOOST_TEST(a.permutation_p == b.permutation_p);
        BOOST_TEST(a.beta_p == b.beta_p, boost::test_tools::tolerance(1e-9));
        // Single-precision permutation scores leave the observed results alone.
        BOOST_TEST(s.status == a.status);
        BOOST_TEST(s.lead == a.lead);
        BOOST_TEST(s.statistic == a.statistic);
        BOOST_TEST(s.estimate == a.estimate);
        BOOST_TEST(s.permutations == a.permutations);
        BOOST_TEST(s.permutation_p == a.permutation_p);
        BOOST_TEST(s.beta_p == a.beta_p, boost::test_tools::tolerance(1e-5));
    }
    // Ties in position and score choose the first input variant.
    d.genotypes.position[1] = d.genotypes.position[0];
    for (std::size_t i = 0; i < 81; ++i)
        d.genotypes.dosage[81 + i] = d.genotypes.dosage[i];
    o.permutations = 0;
    auto tied = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_TEST(tied.genes[0].lead != 1u);
}

BOOST_AUTO_TEST_CASE(qtl_failure_status_paths) {
    auto d = qtl_data(8, 1, 5, 105);
    d.x = Matrix(8, 2, 1);
    for (std::size_t i = 0; i < 8; ++i)
        d.x(i, 1) = i >= 4;
    d.responses = SparseCSC{8, 1, {0, 2}, {0, 4}, {10, 20}};
    CisQtlOptions o;
    o.model.family = Family::poisson;
    o.permutations = 10;
    o.seed = 38;
    auto r = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_TEST(r.genes[0].status == "permutations_failed");
    BOOST_TEST(r.genes[0].permutations_failed > 0u);
    BOOST_TEST(r.genes[0].permutations + r.genes[0].permutations_failed == 10u);
    BOOST_TEST(std::isfinite(r.genes[0].statistic));
    d.responses = SparseCSC{8, 1, {0, 0}, {}, {}};
    r = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    BOOST_TEST(r.genes[0].status.rfind("null_", 0) == 0u);
    // Every permuted response sees the same set of sample-indicator variants;
    // the maximum is constant, so the gene retains its nominal result only.
    Matrix x(6, 1, 1);
    SparseCSC y{6, 1, {0, 6}, {0, 1, 2, 3, 4, 5}, {1, 3, 7, 2, 8, 5}};
    Genotypes g;
    g.samples = g.variants = 6;
    g.dosage.assign(36, 0);
    for (std::size_t i = 0; i < 6; ++i) {
        g.dosage[i * 6 + i] = 1;
        g.chrom.push_back("1");
        g.position.push_back(1);
    }
    o.model.family = Family::gaussian;
    o.permutations = 30;
    for (auto precision :
         {PermutationPrecision::double_precision, PermutationPrecision::single_precision}) {
        o.permutation_precision = precision;
        r = cis_qtl(x, y, g, {{0, "constant_max", "1", 1, 1}}, o);
        BOOST_TEST(r.genes[0].status == "ok_beta_failed");
        BOOST_TEST(std::isnan(r.genes[0].beta_k));
        BOOST_TEST(std::isnan(r.genes[0].reference_ncp));
    }
}

BOOST_AUTO_TEST_CASE(blocked_product_kernels_agree_with_direct_sums) {
    // Ragged register tiles, both cache-panel boundaries, empty products and
    // the Geuvadis sample/covariate dimensions. Sentinels catch output overruns.
    for (const auto shape : {std::array<std::size_t, 3>{0, 7, 9},
                             {1, 0, 8},
                             {1, 6, 0},
                             {1, 1, 1},
                             {17, 5, 7},
                             {127, 6, 8},
                             {128, 7, 9},
                             {129, 13, 127},
                             {257, 19, 128},
                             {131, 17, 129},
                             {462, 43, 137},
                             {19, 257, 31}}) {
        const auto n = shape[0], p = shape[1], m = shape[2];
        std::vector<double> a(n * p), b(n * m), expected(p * m), scale(p * m);
        std::mt19937_64 rng(8171);
        std::uniform_real_distribution<double> uniform(-2, 2);
        for (auto &v : a)
            v = uniform(rng);
        for (auto &v : b)
            v = uniform(rng);
        for (std::size_t r = 0; r < p; ++r)
            for (std::size_t q = 0; q < m; ++q) {
                long double sum = 0, magnitude = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const long double term = static_cast<long double>(a[i * p + r]) * b[i * m + q];
                    sum += term;
                    magnitude += std::abs(term);
                }
                expected[r * m + q] = static_cast<double>(sum);
                scale[r * m + q] = static_cast<double>(magnitude);
            }
        for (auto kernel : {detail::blocked_product_generic, detail::blocked_product}) {
            std::vector<double> c(p * m + 2, 817.0);
            kernel(a.data(), b.data(), n, p, m, c.data() + 1);
            BOOST_TEST(c.front() == 817.0);
            BOOST_TEST(c.back() == 817.0);
            for (std::size_t j = 0; j < expected.size(); ++j)
                BOOST_TEST(std::abs(c[j + 1] - expected[j]) <= 2e-14 * std::max(1.0, scale[j]));
        }
    }
}

BOOST_AUTO_TEST_CASE(gaussian_qtl_projection_spans_batches_and_covariate_units) {
    auto d = qtl_data(83, 2, 137, 319);
    auto o = qtl_options(d);
    o.model.family = Family::gaussian;
    o.model.offset.clear();
    o.permutations = 1033; // two complete batches and a partial batch
    o.nominal = true;
    // Exercise equilibration and a non-leading intercept.
    for (std::size_t i = 0; i < d.x.rows; ++i) {
        std::swap(d.x(i, 0), d.x(i, 2));
        d.x(i, 0) *= 1e-3;
        d.x(i, 1) *= 1e3;
    }
    o.permutation_precision = PermutationPrecision::double_precision;
    const auto batch = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.permutation_precision = PermutationPrecision::single_precision;
    const auto single = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.model.offset.assign(d.x.rows, 0);
    const auto ordinary = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    for (std::size_t f = 0; f < batch.genes.size(); ++f) {
        const auto &a = batch.genes[f], &b = ordinary.genes[f], &s = single.genes[f];
        BOOST_TEST(a.status == "ok");
        BOOST_TEST(a.status == b.status);
        BOOST_TEST(a.lead == b.lead);
        BOOST_TEST(a.permutations == b.permutations);
        BOOST_TEST(a.permutations_failed == b.permutations_failed);
        BOOST_TEST(a.permutation_p == b.permutation_p);
        BOOST_TEST(a.beta_p == b.beta_p, boost::test_tools::tolerance(1e-8));
        BOOST_TEST(a.p_value == b.p_value, boost::test_tools::tolerance(1e-12));
        BOOST_TEST(a.estimate == b.estimate, boost::test_tools::tolerance(1e-12));
        BOOST_TEST(a.standard_error == b.standard_error, boost::test_tools::tolerance(1e-12));
        BOOST_TEST(s.status == a.status);
        BOOST_TEST(s.permutations == a.permutations);
        BOOST_TEST(s.permutations_failed == a.permutations_failed);
        BOOST_TEST(s.permutation_p == a.permutation_p);
        BOOST_TEST(s.beta_p == a.beta_p, boost::test_tools::tolerance(1e-5));
    }
}

BOOST_AUTO_TEST_CASE(gaussian_qtl_projection_handles_small_residual_scale) {
    auto d = qtl_data(83, 1, 15, 512);
    // A large intercept makes |y|^2 - |Q'y|^2 ill-conditioned; the scale and
    // scores must use explicit residuals when that subtraction loses precision.
    SparseCSC shifted{83, 1, {0, 83}, {}, {}};
    const auto y = d.responses.column(0);
    for (std::size_t i = 0; i < 83; ++i) {
        shifted.row_index.push_back(i);
        shifted.values.push_back(1e6 + y[i]);
    }
    auto o = qtl_options(d);
    o.model.family = Family::gaussian;
    o.model.offset.clear();
    o.permutations = 53;
    o.permutation_precision = PermutationPrecision::double_precision;
    const auto batch = cis_qtl(d.x, shifted, d.genotypes, d.phenotypes, o);
    o.permutation_precision = PermutationPrecision::single_precision;
    const auto single = cis_qtl(d.x, shifted, d.genotypes, d.phenotypes, o);
    o.model.offset.assign(83, 0);
    const auto ordinary = cis_qtl(d.x, shifted, d.genotypes, d.phenotypes, o);
    const auto &a = batch.genes[0], &b = ordinary.genes[0], &s = single.genes[0];
    BOOST_TEST(a.status == "ok");
    BOOST_TEST(a.status == b.status);
    BOOST_TEST(a.lead == b.lead);
    BOOST_TEST(a.permutations == b.permutations);
    BOOST_TEST(a.permutations_failed == b.permutations_failed);
    BOOST_TEST(a.permutation_p == b.permutation_p);
    BOOST_TEST(a.beta_p == b.beta_p, boost::test_tools::tolerance(1e-8));
    BOOST_TEST(s.status == a.status);
    BOOST_TEST(s.permutations == a.permutations);
    BOOST_TEST(s.permutation_p == a.permutation_p);
    BOOST_TEST(s.beta_p == a.beta_p, boost::test_tools::tolerance(1e-5));
}

namespace {
// One Gaussian phenotype: an intercept and p - 1 normal covariates, m variants
// with allele frequencies 0.05 to 0.5 (sample by sample in g), and
// y = 2 + 0.3 x_1 + 0.2 g_0 + standard normal noise.
struct GaussianGene {
    Matrix x;
    Vector y;
    std::vector<double> g;
    std::size_t m = 0;
};
GaussianGene gaussian_gene(std::size_t n, std::size_t p, std::size_t m, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> frequency(0.05, 0.5);
    GaussianGene d{Matrix(n, p, 1), Vector(n), std::vector<double>(n * m), m};
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 1; k < p; ++k)
            d.x(i, k) = normal(rng);
    for (std::size_t q = 0; q < m; ++q) {
        std::binomial_distribution<int> allele(2, frequency(rng));
        for (std::size_t i = 0; i < n; ++i)
            d.g[i * m + q] = allele(rng);
    }
    for (std::size_t i = 0; i < n; ++i)
        d.y[i] = 2 + (p > 1 ? 0.3 * d.x(i, 1) : 0.0) + 0.2 * d.g[i * m] + normal(rng);
    return d;
}
// The permutation maxima of detail::qtl_projected_maxima, from the observed
// basis as cis_qtl builds it.
std::vector<double> projected_maxima(const GaussianGene &d, std::size_t permutations,
                                     PermutationPrecision precision, std::size_t &failed) {
    FitOptions options;
    options.family = Family::gaussian;
    options.retain_observations = true;
    const auto null = fit(d.x, d.y, options);
    BOOST_REQUIRE(null.ok());
    detail::QtlBasis basis;
    std::vector<double> z, g = d.g;
    BOOST_REQUIRE(detail::qtl_scores(d.x, null, g, d.m, z, nullptr, nullptr, &basis));
    std::vector<double> maxima;
    failed = 0;
    detail::qtl_projected_maxima(d.x, basis, g, d.m, d.y, keyed_stream(3, 7, 0, name_key("gene")),
                                 permutations, precision, maxima, failed);
    return maxima;
}
// The largest |single - double| / max(1, double) over the permutation maxima,
// which must be equal in number.
double maxima_difference(const GaussianGene &d, std::size_t permutations) {
    std::size_t failed_double = 0, failed_single = 0;
    const auto a =
        projected_maxima(d, permutations, PermutationPrecision::double_precision, failed_double);
    const auto b =
        projected_maxima(d, permutations, PermutationPrecision::single_precision, failed_single);
    BOOST_TEST(failed_double == failed_single);
    BOOST_REQUIRE(a.size() == b.size());
    BOOST_TEST(a.size() + failed_double == permutations);
    double worst = 0;
    for (std::size_t j = 0; j < a.size(); ++j)
        worst = std::max(worst, std::abs(b[j] - a[j]) / std::max(1.0, a[j]));
    return worst;
}
} // namespace

BOOST_AUTO_TEST_CASE(single_precision_permutation_maxima_follow_double) {
    // Sample counts on both sides of the 128-sample panels, variant counts on
    // both sides of the sixteen-lane tiles, few and many covariates, and a
    // partial batch of permutations (523 = 512 + 11).
    double worst = 0;
    std::uint32_t seed = 100;
    for (std::size_t n : {37u, 128u, 129u, 462u})
        for (std::size_t m : {1u, 7u, 16u, 17u, 100u})
            for (std::size_t p : {std::size_t{1}, std::size_t{4}, n > 100 ? 43 : std::size_t{12}}) {
                const double difference = maxima_difference(gaussian_gene(n, p, m, ++seed), 523);
                BOOST_TEST(difference <= 1e-5, "n " << n << ", m " << m << ", p " << p);
                worst = std::max(worst, difference);
            }
    // The Geuvadis dimensions: 462 samples, 43 covariates and 4,500 cis variants.
    const double geuvadis = maxima_difference(gaussian_gene(462, 43, 4500, 7), 523);
    BOOST_TEST(geuvadis <= 1e-5);
    BOOST_TEST_MESSAGE("largest difference of single- from double-precision maxima, relative to "
                       "max(1, |z|): "
                       << std::max(worst, geuvadis));
}

BOOST_AUTO_TEST_CASE(single_precision_permutations_resist_cancellation) {
    const auto base = gaussian_gene(150, 5, 40, 61);
    // A huge mean (every permutation takes the explicit-residual fallback), and
    // a large one that does not (the residualization before the float product
    // removes it).
    for (double shift : {1e6, 300.0}) {
        auto d = base;
        for (auto &v : d.y)
            v += shift;
        BOOST_TEST(maxima_difference(d, 600) <= 1e-5, "shift " << shift);
    }
    // Responses at the edges of the float range, which the residuals leave
    // through their scaling by the residual standard deviation.
    for (double scale : {1e38, 1e-42}) {
        auto d = base;
        for (auto &v : d.y)
            v *= scale;
        BOOST_TEST(maxima_difference(d, 600) <= 1e-5, "scale " << scale);
    }
    // Nearly collinear covariates.
    auto collinear = base;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> normal;
    for (std::size_t i = 0; i < 150; ++i)
        collinear.x(i, 2) = collinear.x(i, 1) + 1e-4 * normal(rng);
    BOOST_TEST(maxima_difference(collinear, 600) <= 1e-5);
    // A variant with one carrier beside a constant one (no information).
    auto rare = base;
    for (std::size_t i = 0; i < 150; ++i) {
        rare.g[i * 40 + 3] = i == 17;
        rare.g[i * 40 + 4] = 1;
    }
    BOOST_TEST(maxima_difference(rare, 600) <= 1e-5);
}

BOOST_AUTO_TEST_CASE(single_precision_gene_results_follow_double_for_any_thread_count) {
    const std::size_t genes = 20;
    auto d = qtl_data(140, genes, 40, 23, {{4, 4 * 40 + 9}});
    for (auto &v : d.responses.values)
        v = std::log1p(v);
    auto o = qtl_options(d);
    o.model.family = Family::gaussian;
    o.model.offset.clear();
    o.permutations = 1000;
    o.permutation_precision = PermutationPrecision::double_precision;
    const auto full = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.permutation_precision = PermutationPrecision::single_precision;
    const auto single = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    o.threads = 1;
    const auto serial = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, o);
    double worst = 0;
    for (std::size_t f = 0; f < genes; ++f) {
        const auto &a = full.genes[f], &s = single.genes[f], &t = serial.genes[f];
        BOOST_TEST(a.status == "ok");
        BOOST_TEST(s.status == a.status);
        BOOST_TEST(s.lead == a.lead);
        BOOST_TEST(s.statistic == a.statistic);
        BOOST_TEST(s.p_value == a.p_value);
        BOOST_TEST(s.estimate == a.estimate);
        BOOST_TEST(s.standard_error == a.standard_error);
        BOOST_TEST(s.permutations == a.permutations);
        BOOST_TEST(s.permutation_p == a.permutation_p);
        BOOST_TEST(s.beta_p == a.beta_p, boost::test_tools::tolerance(1e-5));
        BOOST_TEST(s.beta_k == a.beta_k, boost::test_tools::tolerance(1e-5));
        BOOST_TEST(s.beta_n == a.beta_n, boost::test_tools::tolerance(1e-5));
        worst = std::max(worst, std::abs(s.beta_p - a.beta_p) / a.beta_p);
        // The same bits with one thread as with four.
        BOOST_TEST(t.permutation_p == s.permutation_p);
        BOOST_TEST(t.beta_p == s.beta_p);
        BOOST_TEST(t.beta_k == s.beta_k);
        BOOST_TEST(t.beta_n == s.beta_n);
        BOOST_TEST(t.reference_ncp == s.reference_ncp);
    }
    BOOST_TEST(full.genes[4].lead == 4u * 40 + 9);
    BOOST_TEST(single.genes[4].beta_p < 1e-4);
    BOOST_TEST_MESSAGE("largest relative beta_p difference, single from double: " << worst);
}

BOOST_AUTO_TEST_CASE(float_product_kernels_agree_with_direct_sums) {
    // Float sums of at most 128 products per panel: each panel sum is within
    // gamma_129 = 129 u / (1 - 129 u) (u = 2^-24) of the sum of |a_i b_i| over
    // the panel, and the double sum of four panels adds next to nothing.
    using Kernel =
        void (*)(const float *, const float *, std::size_t, std::size_t, std::size_t, double *);
    std::vector<std::pair<const char *, Kernel>> kernels{
        {"generic", detail::blocked_product_float_generic},
        {"dispatched", detail::blocked_product_float}};
#ifdef RAPIDGLM_PHILOX_AVX2
    if (detail::fma_available())
        kernels.push_back({"avx2", detail::blocked_product_float_avx2});
#endif
    std::vector<std::array<std::size_t, 3>> shapes{
        {0, 7, 9},      {1, 0, 16},    {1, 6, 0},      {1, 1, 1},      {17, 5, 7},
        {127, 6, 16},   {128, 7, 17},  {129, 13, 127}, {257, 19, 128}, {131, 17, 129},
        {462, 43, 137}, {19, 257, 31}, {462, 13, 4500}};
    for (std::size_t n : {37u, 128u, 129u, 462u})
        for (std::size_t m : {1u, 7u, 16u, 17u, 100u})
            shapes.push_back({n, 11, m});
    const double u = std::ldexp(1.0, -24), gamma = 129 * u / (1 - 129 * u);
    double worst = 0;
    for (const auto shape : shapes) {
        const auto n = shape[0], p = shape[1], m = shape[2];
        std::vector<float> a(n * p), b(n * m);
        std::vector<double> expected(p * m), scale(p * m);
        std::mt19937_64 rng(8171);
        std::uniform_real_distribution<float> uniform(-2, 2);
        for (auto &v : a)
            v = uniform(rng);
        for (auto &v : b)
            v = uniform(rng);
        for (std::size_t r = 0; r < p; ++r)
            for (std::size_t q = 0; q < m; ++q) {
                long double sum = 0, magnitude = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const long double term = static_cast<long double>(a[i * p + r]) * b[i * m + q];
                    sum += term;
                    magnitude += std::abs(term);
                }
                expected[r * m + q] = static_cast<double>(sum);
                scale[r * m + q] = static_cast<double>(magnitude);
            }
        std::vector<std::vector<double>> outputs;
        for (const auto &[name, kernel] : kernels) {
            std::vector<double> c(p * m + 2, 817.0);
            kernel(a.data(), b.data(), n, p, m, c.data() + 1);
            BOOST_TEST(c.front() == 817.0);
            BOOST_TEST(c.back() == 817.0);
            for (std::size_t j = 0; j < expected.size(); ++j) {
                const double error = std::abs(c[j + 1] - expected[j]);
                BOOST_TEST(error <= gamma * scale[j] + 1e-300,
                           name << " " << n << " x " << p << " x " << m);
                if (scale[j] > 0)
                    worst = std::max(worst, error / scale[j]);
            }
            outputs.emplace_back(c.begin() + 1, c.end() - 1);
        }
        // Generic and AVX2 tiles round differently, within twice the bound.
        for (std::size_t k = 1; k < outputs.size(); ++k)
            for (std::size_t j = 0; j < expected.size(); ++j)
                BOOST_TEST(std::abs(outputs[k][j] - outputs[0][j]) <=
                           2 * gamma * scale[j] + 1e-300);
    }
    BOOST_TEST_MESSAGE("largest float-product error relative to the sum of |terms|: " << worst);
}

BOOST_AUTO_TEST_CASE(qtl_maxima_tied_with_the_observed_statistic_count_as_extreme) {
    // An intercept only and one private variant per sample: the largest |z| of
    // those variants is the same in every permutation in exact arithmetic, and is
    // the observed statistic when one of them leads. Common variants can only
    // raise a maximum. Every maximum must then count as extreme (permutation_p 1)
    // in both precisions, wherever rounding puts it, also at a phenotype mean of
    // 300 residual SDs, where the double scales round worst. Without common
    // variants the maxima are constant: float rounding spreads the single-
    // precision ones (a relative variance of about 3e-14 at 462 samples), and the
    // gene must still report ok_beta_failed.
    const std::size_t n = 462;
    for (double mean : {3.0, 300.0})
        for (const auto &[common, outlier] :
             std::vector<std::pair<std::size_t, double>>{{0, 0.0}, {0, 10.0}, {200, 10.0}}) {
            // The outlier keeps the lead on a private variant beside common ones.
            std::mt19937_64 rng(31 + common);
            std::normal_distribution<double> normal;
            std::binomial_distribution<int> allele(2, 0.3);
            const std::size_t m = n + common;
            SparseCSC y{n, 1, {0, n}, {}, {}};
            Genotypes g;
            g.samples = n;
            g.variants = m;
            g.dosage.assign(m * n, 0);
            g.chrom.assign(m, "1");
            g.position.assign(m, 1);
            for (std::size_t i = 0; i < n; ++i) {
                y.row_index.push_back(i);
                y.values.push_back(mean + normal(rng) + (i == 7 ? outlier : 0.0));
                g.dosage[i * n + i] = 1;
            }
            for (std::size_t v = n; v < m; ++v)
                for (std::size_t i = 0; i < n; ++i)
                    g.dosage[v * n + i] = allele(rng);
            CisQtlOptions o;
            o.model.family = Family::gaussian;
            o.permutations = 1000;
            for (auto precision :
                 {PermutationPrecision::double_precision, PermutationPrecision::single_precision}) {
                o.permutation_precision = precision;
                const auto r = cis_qtl(Matrix(n, 1, 1), y, g, {{0, "tied", "1", 1, 1}}, o);
                const auto &gene = r.genes[0];
                const auto label = "mean " + std::to_string(mean) + ", common " +
                                   std::to_string(common) + ", outlier " + std::to_string(outlier);
                BOOST_TEST(gene.lead < n, label);
                BOOST_TEST(gene.permutations == 1000u, label);
                BOOST_TEST(gene.permutation_p == 1.0, label);
                if (common == 0) {
                    BOOST_TEST(gene.status == "ok_beta_failed", label);
                    BOOST_TEST(std::isnan(gene.beta_p), label);
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(qtl_finite_maxima_match_the_direct_loop) {
    // The lane and AVX2 maxima against the loop they replace, bit for bit, with
    // variants without information (NaN) and infinite or NaN scales.
    const auto direct = [](const std::vector<double> &c, const std::vector<double> &inverse_info,
                           double inverse_scale) {
        double maximum = rapidglm::nan;
        for (std::size_t q = 0; q < c.size(); ++q) {
            const double value = std::abs(c[q]) * inverse_info[q] * inverse_scale;
            if (std::isfinite(value))
                maximum = std::isfinite(maximum) ? std::max(maximum, value) : value;
        }
        return maximum;
    };
    using Maximum = double (*)(const double *, const double *, std::size_t, double);
    std::vector<Maximum> kernels{detail::qtl_finite_maximum_generic, detail::qtl_finite_maximum};
#ifdef RAPIDGLM_PHILOX_AVX2
    if (detail::fma_available())
        kernels.push_back(detail::qtl_finite_maximum_avx2);
#endif
    std::mt19937_64 rng(77);
    std::normal_distribution<double> normal;
    for (std::size_t m : {0u, 1u, 7u, 8u, 9u, 17u, 4500u})
        for (double inverse_scale :
             {0.7, 0.0, std::numeric_limits<double>::infinity(), rapidglm::nan}) {
            std::vector<double> c(m), inverse_info(m);
            for (std::size_t q = 0; q < m; ++q) {
                c[q] = normal(rng);
                inverse_info[q] = q % 5 == 3 ? rapidglm::nan : std::abs(normal(rng));
            }
            const double expected = direct(c, inverse_info, inverse_scale);
            for (auto kernel : kernels) {
                const double got = kernel(c.data(), inverse_info.data(), m, inverse_scale);
                BOOST_TEST((got == expected || (std::isnan(got) && std::isnan(expected))),
                           "m " << m << ", scale " << inverse_scale);
            }
        }
}

BOOST_AUTO_TEST_CASE(fused_maxima_equal_those_of_the_stored_product) {
    // The fused kernels against the product and maxima they replace
    // (blocked_product_float, then qtl_finite_maximum of each row at unit scale),
    // bit for bit: sample counts on both sides of the 128-sample panels, variant
    // counts on both sides of the sixteen-variant groups and 128-variant blocks,
    // row counts on both sides of the six-row tiles, variants without
    // information (NaN; every variant when m is 7, so that no row has a maximum)
    // and float sums that overflow (an infinite score, skipped).
    using Fused = void (*)(const float *, const float *, std::size_t, std::size_t, std::size_t,
                           const double *, double *);
    using Product =
        void (*)(const float *, const float *, std::size_t, std::size_t, std::size_t, double *);
    using Maximum = double (*)(const double *, const double *, std::size_t, double);
    struct Kernels {
        const char *name;
        Fused fused;
        Product product;
        Maximum maximum;
    };
    std::vector<Kernels> kernels{{"generic", detail::fused_maxima_generic,
                                  detail::blocked_product_float_generic,
                                  detail::qtl_finite_maximum_generic},
                                 {"dispatched", detail::fused_maxima, detail::blocked_product_float,
                                  detail::qtl_finite_maximum}};
#ifdef RAPIDGLM_PHILOX_AVX2
    if (detail::fma_available())
        kernels.push_back({"avx2", detail::fused_maxima_avx2, detail::blocked_product_float_avx2,
                           detail::qtl_finite_maximum_avx2});
#endif
    const auto same = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
    std::mt19937_64 rng(4141);
    std::normal_distribution<float> normal;
    std::size_t without = 0, overflowed = 0;
    for (std::size_t n : {37u, 128u, 129u, 462u})
        for (std::size_t m : {1u, 7u, 16u, 17u, 100u, 129u, 4500u})
            for (std::size_t rows : {1u, 5u, 6u, 11u, 512u}) {
                if (m == 4500 && (n != 462 || rows != 11))
                    continue;
                std::vector<float> a(n * rows), b(n * m);
                for (auto &v : a)
                    v = normal(rng);
                for (auto &v : b)
                    v = normal(rng);
                if (m > 5)
                    for (std::size_t i = 0; i < n; ++i)
                        b[i * m + 5] = 3e38f;
                const std::size_t groups = (m + 15) / 16;
                std::vector<double> inverse_info(16 * groups, rapidglm::nan);
                for (std::size_t q = 0; q < m; ++q)
                    inverse_info[q] =
                        q % 5 == 3 || m == 7 ? rapidglm::nan : 0.1 + std::abs(normal(rng));
                std::vector<float> packed_a(6 * ((rows + 5) / 6) * n), packed_b(16 * groups * n);
                detail::pack_rows_float(a.data(), n, rows, packed_a.data());
                detail::pack_columns_float(b.data(), n, m, packed_b.data());
                for (const auto &kernel : kernels) {
                    std::vector<double> c(rows * m), got(rows + 2, 817.0);
                    kernel.product(a.data(), b.data(), n, rows, m, c.data());
                    kernel.fused(packed_a.data(), packed_b.data(), n, rows, m, inverse_info.data(),
                                 got.data() + 1);
                    BOOST_TEST(got.front() == 817.0);
                    BOOST_TEST(got.back() == 817.0);
                    for (std::size_t j = 0; j < rows; ++j) {
                        const double expected =
                            kernel.maximum(&c[j * m], inverse_info.data(), m, 1.0);
                        BOOST_TEST(same(got[j + 1], expected), kernel.name << " n " << n << ", m "
                                                                           << m << ", rows " << rows
                                                                           << ", row " << j);
                        without += std::isnan(expected);
                        overflowed += m > 5 && !std::isfinite(c[j * m + 5]);
                    }
                }
            }
    // Both kinds of score to skip occurred.
    BOOST_TEST(without > 0u);
    BOOST_TEST(overflowed > 0u);
}

namespace {
// The single-precision permutation maxima as computed before the fused kernel:
// the batch's residuals in a row-major float matrix, the whole score matrix by
// blocked_product_float and then qtl_finite_maximum of each row, with each
// shuffle drawn by qtl_shuffle. Permutations whose centred sum of squares is at
// most `ratio` times their residual sum of squares are scored from their
// centred response (ratio detail::qtl_centred_ratio, as qtl_projected_maxima
// does; 0, never), the others from y_b - Q t_b with Q T' for every permutation;
// `centred` (may be null) counts the former.
std::vector<double> stored_product_maxima(const GaussianGene &d, std::size_t permutations,
                                          std::size_t &failed, double ratio,
                                          std::size_t *centred = nullptr) {
    FitOptions options;
    options.family = Family::gaussian;
    options.retain_observations = true;
    const auto null = fit(d.x, d.y, options);
    BOOST_REQUIRE(null.ok());
    detail::QtlBasis basis;
    std::vector<double> z, g = d.g;
    BOOST_REQUIRE(detail::qtl_scores(d.x, null, g, d.m, z, nullptr, nullptr, &basis));
    const auto &factor = *basis.factor;
    const auto &x = d.x;
    const std::size_t n = x.rows, p = x.cols, m = d.m, batch = 512, df = n - p;
    std::vector<double> q(n * p), qt(p * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < p; ++k) {
            double value = x(i, k) / factor.scale[k];
            for (std::size_t l = 0; l < k; ++l)
                value -= factor.l(k, l) * q[i * p + l];
            q[i * p + k] = value / factor.l(k, k);
            qt[k * n + i] = q[i * p + k];
        }
    std::vector<double> projection(n * m), inverse_info(m);
    detail::blocked_product(qt.data(), basis.s.data(), p, n, m, projection.data());
    std::vector<float> g_float(n * m);
    for (std::size_t e = 0; e < n * m; ++e)
        g_float[e] = static_cast<float>(g[e] - projection[e]);
    for (std::size_t k = 0; k < m; ++k)
        inverse_info[k] =
            basis.information[k] > 0 ? 1 / std::sqrt(basis.information[k]) : rapidglm::nan;
    double norm = 0, centre = 0, spread = 0;
    for (double v : d.y) {
        norm += v * v;
        centre += v;
    }
    centre /= static_cast<double>(n);
    for (double v : d.y)
        spread += (v - centre) * (v - centre);
    if (!detail::qtl_has_intercept(x)) {
        centre = 0;
        spread = norm;
    }
    if (centred)
        *centred = 0;
    const Stream stream = keyed_stream(3, 7, 0, name_key("gene"));
    std::vector<std::size_t> index(n);
    std::vector<double> maxima;
    failed = 0;
    for (std::size_t first = 0; first < permutations; first += batch) {
        const std::size_t count = std::min(batch, permutations - first);
        std::vector<double> shuffled(n * count), t(count * p), t_t(p * count), fitted(n * count),
            inverse_scales(count), cross(count * m);
        std::vector<bool> use_centred(count);
        for (std::size_t j = 0; j < count; ++j) {
            detail::qtl_shuffle(stream, first + j, index);
            for (std::size_t i = 0; i < n; ++i)
                shuffled[i * count + j] = d.y[index[i]];
        }
        detail::blocked_product(shuffled.data(), q.data(), n, count, p, t.data());
        for (std::size_t j = 0; j < count; ++j) {
            double rss = norm;
            for (std::size_t k = 0; k < p; ++k)
                rss -= t[j * p + k] * t[j * p + k];
            use_centred[j] = rss > 1e-6 * norm && spread <= ratio * rss;
            if (centred)
                *centred += use_centred[j];
            if (rss <= 1e-6 * norm) {
                Vector yb(n);
                for (std::size_t i = 0; i < n; ++i)
                    yb[i] = shuffled[i * count + j];
                const auto beta = factor.solve(detail::xtv(x, yb));
                rss = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    double fitted_value = 0;
                    for (std::size_t k = 0; k < p; ++k)
                        fitted_value += x(i, k) * beta[k];
                    shuffled[i * count + j] = yb[i] - fitted_value;
                    rss += shuffled[i * count + j] * shuffled[i * count + j];
                }
                std::fill_n(&t[j * p], p, 0.0);
            }
            const double scale = rss / static_cast<double>(df);
            inverse_scales[j] = 1 / std::sqrt(scale);
        }
        for (std::size_t j = 0; j < count; ++j)
            for (std::size_t k = 0; k < p; ++k)
                t_t[k * count + j] = t[j * p + k];
        detail::blocked_product(qt.data(), t_t.data(), p, n, count, fitted.data());
        std::vector<float> residuals(n * count);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < count; ++j) {
                const std::size_t e = i * count + j;
                const double residual =
                    use_centred[j] ? shuffled[e] - centre : shuffled[e] - fitted[e];
                residuals[e] = static_cast<float>(residual * inverse_scales[j]);
            }
        detail::blocked_product_float(residuals.data(), g_float.data(), n, count, m, cross.data());
        for (std::size_t j = 0; j < count; ++j) {
            const double maximum =
                detail::qtl_finite_maximum(&cross[j * m], inverse_info.data(), m, 1.0);
            if (std::isfinite(maximum))
                maxima.push_back(maximum);
            else
                ++failed;
        }
    }
    return maxima;
}
} // namespace

BOOST_AUTO_TEST_CASE(fused_permutation_maxima_equal_the_stored_product_path) {
    // qtl_projected_maxima (packed operands, fused maxima, batched shuffles,
    // centred responses, Q T' for the other permutations alone, a reused
    // workspace) against the stored-product path, bit for bit, on the shapes of
    // single_precision_permutation_maxima_follow_double with 523 permutations (a
    // complete and a partial batch), on genes with variants without
    // information, and on a large mean that takes the least-squares fallback.
    // One workspace serves every gene, largest first.
    detail::QtlWorkspace workspace;
    std::vector<GaussianGene> genes{gaussian_gene(462, 43, 4500, 7)};
    std::uint32_t seed = 100;
    for (std::size_t n : {37u, 128u, 129u, 462u})
        for (std::size_t m : {1u, 7u, 16u, 17u, 100u})
            for (std::size_t p : {std::size_t{1}, std::size_t{4}, n > 100 ? 43 : std::size_t{12}})
                genes.push_back(gaussian_gene(n, p, m, ++seed));
    auto rare = gaussian_gene(150, 5, 40, 61);
    for (std::size_t i = 0; i < 150; ++i) {
        rare.g[i * 40 + 3] = i == 17;
        rare.g[i * 40 + 4] = 1;
    }
    genes.push_back(rare);
    auto shifted = gaussian_gene(150, 5, 40, 61);
    for (auto &v : shifted.y)
        v += 1e6;
    genes.push_back(shifted);
    for (const auto &d : genes) {
        FitOptions options;
        options.family = Family::gaussian;
        options.retain_observations = true;
        const auto null = fit(d.x, d.y, options);
        BOOST_REQUIRE(null.ok());
        detail::QtlBasis basis;
        std::vector<double> z, g = d.g, maxima;
        BOOST_REQUIRE(detail::qtl_scores(d.x, null, g, d.m, z, nullptr, nullptr, &basis));
        std::size_t failed = 0, failed_stored = 0;
        detail::qtl_projected_maxima(
            d.x, basis, g, d.m, d.y, keyed_stream(3, 7, 0, name_key("gene")), 523,
            PermutationPrecision::single_precision, maxima, failed, &workspace);
        const auto stored = stored_product_maxima(d, 523, failed_stored, detail::qtl_centred_ratio);
        const auto label = "n " + std::to_string(d.x.rows) + ", m " + std::to_string(d.m) + ", p " +
                           std::to_string(d.x.cols);
        BOOST_TEST(failed == failed_stored, label);
        BOOST_REQUIRE(maxima.size() == stored.size());
        std::size_t differing = 0;
        for (std::size_t j = 0; j < maxima.size(); ++j)
            differing += maxima[j] != stored[j];
        BOOST_TEST(differing == 0u, label);
    }
}

BOOST_AUTO_TEST_CASE(single_precision_centred_permutations_follow_double) {
    // Permutations that the covariates fit poorly are scored from their centred
    // response rather than their residual. At the Geuvadis dimensions every one
    // is (a permutation's R^2 against 42 covariates is about 0.09); with 34
    // covariates for 37 samples none is (R^2 about 0.9), and those maxima are the
    // residual path's bit for bit; 40 covariates for 60 samples mix the two.
    // The maxima stay within 1e-5 of max(1, |z|) of double precision.
    struct Case {
        GaussianGene gene;
        const char *name;
    };
    std::vector<Case> cases{{gaussian_gene(462, 43, 4500, 7), "Geuvadis"},
                            {gaussian_gene(37, 34, 50, 9), "37 x 34"},
                            {gaussian_gene(60, 40, 50, 10), "60 x 40"}};
    for (const auto &[d, name] : cases) {
        std::size_t failed = 0, centred = 0, failed_residual = 0;
        const auto stored =
            stored_product_maxima(d, 523, failed, detail::qtl_centred_ratio, &centred);
        const auto residual = stored_product_maxima(d, 523, failed_residual, 0);
        BOOST_REQUIRE(stored.size() == residual.size());
        std::size_t differing = 0;
        for (std::size_t j = 0; j < stored.size(); ++j)
            differing += stored[j] != residual[j];
        if (std::string(name) == "Geuvadis") {
            BOOST_TEST(centred == 523u);
            BOOST_TEST(differing > 0u);
        } else if (std::string(name) == "37 x 34") {
            BOOST_TEST(centred == 0u);
            BOOST_TEST(differing == 0u);
        } else {
            BOOST_TEST(centred > 0u);
            BOOST_TEST(centred < 523u);
        }
        const double difference = maxima_difference(d, 523);
        BOOST_TEST(difference <= 1e-5, name);
        BOOST_TEST_MESSAGE(name << ": " << centred << " of 523 permutations centred; largest "
                                << "difference from double, relative to max(1, |z|): "
                                << difference);
    }
    // The centre is the mean only when a column is a nonzero constant.
    Matrix x(5, 2, 0);
    BOOST_TEST(!detail::qtl_has_intercept(x));
    for (std::size_t i = 0; i < 5; ++i) {
        x(i, 0) = static_cast<double>(i);
        x(i, 1) = 2.5;
    }
    BOOST_TEST(detail::qtl_has_intercept(x));
    x(3, 1) = 2.4;
    BOOST_TEST(!detail::qtl_has_intercept(x));
}

BOOST_AUTO_TEST_CASE(batched_shuffles_equal_the_scalar_ones) {
    // QtlShuffler against qtl_shuffle, with the values computed before the swaps
    // (by default) and with one group of eight blocks, so that nearly every
    // shuffle continues with further blocks, as one that rejects a value does.
    std::vector<std::size_t> expected, got;
    for (std::size_t n : {1u, 2u, 3u, 16u, 17u, 33u, 37u, 128u, 129u, 462u, 1000u})
        for (std::size_t groups : {0u, 1u}) {
            detail::QtlShuffler shuffler;
            shuffler.prepare(n, groups);
            expected.resize(n);
            got.resize(n);
            for (const auto &stream :
                 {keyed_stream(3, 7, 0, name_key("gene")),
                  keyed_stream(20261016, 7, 0, name_key("ENSG00000100181.15"))})
                for (std::size_t b :
                     {0u, 1u, 2u, 7u, 8u, 9u, 511u, 512u, 4096u, 2147483648u, 4294967295u}) {
                    detail::qtl_shuffle(stream, b, expected);
                    shuffler.shuffle(stream, b, got.data());
                    BOOST_TEST(got == expected, "n " << n << ", groups " << groups << ", b " << b);
                }
            BOOST_CHECK_THROW(shuffler.shuffle(keyed_stream(3, 7, 0, 1), 4294967296u, got.data()),
                              std::invalid_argument);
        }
    // The remainders without division equal those of %.
    detail::QtlShuffler shuffler;
    shuffler.prepare(2000);
    std::mt19937_64 rng(99);
    for (std::uint64_t r = 2; r <= 2000; ++r) {
        std::vector<std::uint64_t> values{0,
                                          1,
                                          r - 1,
                                          r,
                                          r + 1,
                                          ~0ULL,
                                          ~0ULL - r,
                                          ~0ULL / r * r,
                                          ~0ULL / r * r - 1,
                                          (1ULL << 63) + r};
        for (int k = 0; k < 64; ++k)
            values.push_back(rng());
        for (const auto value : values)
            BOOST_TEST(shuffler.remainder(value, r) == value % r, "r " << r << ", value " << value);
    }
}

BOOST_AUTO_TEST_CASE(qtl_workspaces_carry_nothing_between_genes) {
    // Gene workers keep their buffers from gene to gene. Genes of very different
    // window sizes, the largest first, must give the same bits with any number of
    // threads as each gene run alone, in both precisions and with NB refits.
    const std::size_t n = 97;
    const std::vector<std::size_t> sizes{300, 3, 41, 17, 1, 130, 5, 260, 16, 2, 129};
    std::mt19937_64 rng(808);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> frequency(0.05, 0.5);
    Matrix x(n, 3, 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 1; k < 3; ++k)
            x(i, k) = normal(rng);
    Genotypes genotypes;
    genotypes.samples = n;
    std::vector<CisPhenotype> phenotypes;
    SparseCSC gaussian{n, sizes.size(), {0}, {}, {}}, counts{n, sizes.size(), {0}, {}, {}};
    for (std::size_t f = 0; f < sizes.size(); ++f) {
        const long long anchor = 3'000'000LL * static_cast<long long>(f + 1);
        phenotypes.push_back({f, "gene" + std::to_string(f), "chr1", anchor, anchor});
        const std::size_t first = genotypes.variants;
        for (std::size_t k = 0; k < sizes[f]; ++k) {
            std::binomial_distribution<int> allele(2, frequency(rng));
            for (std::size_t i = 0; i < n; ++i)
                genotypes.dosage.push_back(allele(rng));
            genotypes.chrom.push_back("chr1");
            genotypes.position.push_back(anchor - 500'000 + 1000LL * static_cast<long long>(k));
            ++genotypes.variants;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double eta = 1 + 0.4 * x(i, 1) + 0.5 * genotypes.dosage[first * n + i];
            gaussian.row_index.push_back(i);
            gaussian.values.push_back(eta + normal(rng));
            std::poisson_distribution<int> poisson(std::exp(eta));
            counts.row_index.push_back(i);
            counts.values.push_back(poisson(rng));
        }
        gaussian.column_ptr.push_back(gaussian.values.size());
        counts.column_ptr.push_back(counts.values.size());
    }
    const auto same = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
    for (int mode = 0; mode < 3; ++mode) {
        CisQtlOptions o;
        o.model.family = mode == 2 ? Family::negative_binomial : Family::gaussian;
        o.permutations = mode == 2 ? 40 : 600;
        o.permutation_precision = mode == 1 ? PermutationPrecision::double_precision
                                            : PermutationPrecision::single_precision;
        o.nominal = true;
        const auto &y = mode == 2 ? counts : gaussian;
        std::vector<CisQtlResult> alone;
        o.threads = 1;
        for (const auto &ph : phenotypes)
            alone.push_back(cis_qtl(x, y, genotypes, {ph}, o));
        for (int threads : {1, 2, 3, 5}) {
            o.threads = threads;
            const auto all = cis_qtl(x, y, genotypes, phenotypes, o);
            std::size_t k = 0;
            for (std::size_t f = 0; f < phenotypes.size(); ++f) {
                const auto &a = alone[f].genes[0], &b = all.genes[f];
                const auto label = "mode " + std::to_string(mode) + ", threads " +
                                   std::to_string(threads) + ", gene " + std::to_string(f);
                BOOST_TEST(a.status.rfind("ok", 0) == 0u, label);
                BOOST_TEST(a.status == b.status, label);
                BOOST_TEST(a.variants == b.variants, label);
                BOOST_TEST(a.lead == b.lead, label);
                BOOST_TEST(a.permutations == b.permutations, label);
                BOOST_TEST(a.permutations_failed == b.permutations_failed, label);
                for (auto [u, v] : {std::pair{a.statistic, b.statistic},
                                    {a.p_value, b.p_value},
                                    {a.estimate, b.estimate},
                                    {a.standard_error, b.standard_error},
                                    {a.theta, b.theta},
                                    {a.permutation_p, b.permutation_p},
                                    {a.beta_p, b.beta_p},
                                    {a.beta_k, b.beta_k},
                                    {a.beta_n, b.beta_n},
                                    {a.reference_ncp, b.reference_ncp}})
                    BOOST_TEST(same(u, v), label);
                for (const auto &pair : alone[f].nominal) {
                    BOOST_REQUIRE(k < all.nominal.size());
                    const auto &other = all.nominal[k++];
                    BOOST_TEST((pair.variant == other.variant &&
                                same(pair.statistic, other.statistic) &&
                                same(pair.estimate, other.estimate) &&
                                same(pair.standard_error, other.standard_error)),
                               label);
                }
            }
            BOOST_TEST(k == all.nominal.size());
        }
    }
}

namespace {
// The hard-call form of whole dosages (NaN missing), with every bit past the
// last sample of a variant set to the missing code, which readers must ignore.
Genotypes hard_call_form(Genotypes g) {
    const std::size_t n = g.samples, stride = g.call_bytes();
    g.hard_calls.assign(stride * g.variants, 0);
    for (std::size_t v = 0; v < g.variants; ++v)
        for (std::size_t i = 0; i < 4 * stride; ++i) {
            const double d = i < n ? g.dosage[v * n + i] : rapidglm::nan;
            const unsigned code = std::isnan(d) ? 1 : d == 2 ? 0 : d == 1 ? 2 : 3;
            g.hard_calls[v * stride + i / 4] |= static_cast<std::uint8_t>(code << (2 * (i % 4)));
        }
    g.dosage.clear();
    return g;
}
bool same(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
}
} // namespace

BOOST_AUTO_TEST_CASE(qtl_hard_calls_give_the_results_of_dosages) {
    // 83 samples (a partial last byte), missing calls in a third of the
    // variants, one variant missing in every sample and one called only once.
    auto d = qtl_data(83, 3, 25, 4242);
    const std::size_t n = 83;
    auto &dosage = d.genotypes.dosage;
    std::mt19937_64 rng(9);
    for (std::size_t v = 0; v < d.genotypes.variants; ++v)
        for (std::size_t i = 0; i < n; ++i)
            if (v % 3 == 1 && rng() % 8 == 0)
                dosage[v * n + i] = rapidglm::nan;
    for (std::size_t i = 0; i < n; ++i) {
        dosage[4 * n + i] = rapidglm::nan;
        dosage[30 * n + i] = i == 5 ? 1.0 : rapidglm::nan;
    }
    const auto hard = hard_call_form(d.genotypes);
    hard.validate();
    BOOST_TEST(hard.hard_calls.size() == 21u * 75);
    for (std::size_t v = 0; v < hard.variants; ++v)
        BOOST_TEST(hard.mean_dosage(v) == d.genotypes.mean_dosage(v), "variant " << v);
    BOOST_TEST(hard.mean_dosage(4) == 0.0);
    BOOST_TEST(hard.mean_dosage(30) == 1.0);
    // NB refits, Gaussian projected permutations in both precisions, and
    // Gaussian refits (forced by a zero offset), each with nominal results.
    auto nb = qtl_options(d);
    nb.permutations = 60;
    nb.nominal = true;
    auto gaussian = nb;
    gaussian.model.family = Family::gaussian;
    gaussian.model.offset.clear();
    std::vector<CisQtlOptions> runs{nb, gaussian, gaussian, gaussian};
    runs[1].permutation_precision = PermutationPrecision::double_precision;
    runs[3].model.offset.assign(n, 0);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto a = cis_qtl(d.x, d.responses, d.genotypes, d.phenotypes, runs[r]);
        const auto b = cis_qtl(d.x, d.responses, hard, d.phenotypes, runs[r]);
        BOOST_REQUIRE(a.genes.size() == b.genes.size());
        for (std::size_t f = 0; f < a.genes.size(); ++f) {
            const auto &x = a.genes[f], &y = b.genes[f];
            BOOST_TEST(x.status.rfind("ok", 0) == 0u, "run " << r);
            BOOST_TEST(x.status == y.status);
            BOOST_TEST(x.variants == y.variants);
            BOOST_TEST(x.lead == y.lead);
            BOOST_TEST(x.permutations == y.permutations);
            BOOST_TEST(x.permutations_failed == y.permutations_failed);
            for (auto [u, w] : {std::pair{x.statistic, y.statistic},
                                {x.p_value, y.p_value},
                                {x.estimate, y.estimate},
                                {x.standard_error, y.standard_error},
                                {x.theta, y.theta},
                                {x.permutation_p, y.permutation_p},
                                {x.beta_p, y.beta_p},
                                {x.beta_k, y.beta_k},
                                {x.beta_n, y.beta_n},
                                {x.reference_ncp, y.reference_ncp}})
                BOOST_TEST(same(u, w), "run " << r << ", gene " << f);
        }
        BOOST_REQUIRE(a.nominal.size() == b.nominal.size());
        BOOST_TEST(a.nominal.size() == 75u);
        std::size_t missing_scores = 0;
        for (std::size_t k = 0; k < a.nominal.size(); ++k) {
            const auto &x = a.nominal[k], &y = b.nominal[k];
            BOOST_TEST(x.variant == y.variant);
            BOOST_TEST((same(x.statistic, y.statistic) && same(x.p_value, y.p_value) &&
                        same(x.estimate, y.estimate) && same(x.standard_error, y.standard_error)),
                       "run " << r << ", pair " << k);
            missing_scores += std::isnan(x.statistic);
        }
        // The variants without a call or with a single one carry no information.
        BOOST_TEST(missing_scores == 2u);
    }
    // One form at a time, each of its declared size.
    auto both = hard;
    both.dosage = d.genotypes.dosage;
    BOOST_CHECK_THROW(both.validate(), std::invalid_argument);
    auto truncated = hard;
    truncated.hard_calls.pop_back();
    BOOST_CHECK_THROW(truncated.validate(), std::invalid_argument);
    BOOST_CHECK_THROW(cis_qtl(d.x, d.responses, truncated, d.phenotypes, nb),
                      std::invalid_argument);
}
