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
    auto d = qtl_data(120, 3, 10, 9);
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
