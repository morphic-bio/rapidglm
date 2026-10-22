#include <boost/test/unit_test.hpp>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
namespace {
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
// Two groups of `reps` samples with size factors around 1 and NB counts with
// dispersion 0.05 + 1 / mean; every tenth gene changes by 2^(+-1.5).
struct Bulk {
    Matrix x;
    std::vector<Vector> y;
    std::vector<unsigned char> de;
};
Bulk simulate(std::uint64_t seed, std::size_t reps, std::size_t genes = 600) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform;
    const std::size_t m = 2 * reps;
    Bulk b;
    b.x = Matrix(m, 2, 1);
    Vector sf(m);
    for (std::size_t i = 0; i < m; ++i) {
        b.x(i, 1) = i >= reps;
        sf[i] = std::exp(0.2 * (uniform(rng) - 0.5));
    }
    for (std::size_t j = 0; j < genes; ++j) {
        const double base = std::exp(std::log(5.0) + uniform(rng) * std::log(400.0)),
                     disp = 0.05 + 1 / base;
        const bool de = j % 10 == 0;
        const double lfc = de ? (j % 20 == 0 ? 1.5 : -1.5) * std::log(2.0) : 0;
        Vector y(m);
        for (std::size_t i = 0; i < m; ++i) {
            const double mu = base * sf[i] * std::exp(lfc * b.x(i, 1));
            std::gamma_distribution<double> gamma(1 / disp, mu * disp);
            y[i] = std::poisson_distribution<int>(gamma(rng))(rng);
        }
        b.y.push_back(y);
        b.de.push_back(de);
    }
    return b;
}
} // namespace
BOOST_AUTO_TEST_CASE(bulk_helpers_follow_r_conventions) {
    BOOST_TEST(detail::median({3, 1, 2, 4}) == 2.5);
    BOOST_TEST(detail::median({3, 1, 2}) == 2.0);
    BOOST_TEST(detail::mad({1, 2, 3, 4, 100}) == 1.4826, boost::test_tools::tolerance(1e-14));
    BOOST_TEST(detail::quantile7({4, 1, 3, 2}, 0.3) == 1.9, boost::test_tools::tolerance(1e-14));
    // LOWESS and the gamma identity-link fit reproduce exact lines.
    Vector x, y;
    for (int i = 0; i < 40; ++i) {
        x.push_back(0.1 * i);
        y.push_back(0.1 + 2 * x.back());
    }
    const auto fit = detail::lowess(x, y, 0.2, 3);
    for (std::size_t i = 0; i < x.size(); ++i)
        BOOST_TEST(fit[i] == y[i], boost::test_tools::tolerance(1e-10));
    double a0 = 1, a1 = 1;
    BOOST_REQUIRE(detail::gamma_identity_fit(y, x, a0, a1));
    BOOST_TEST(a0 == 0.1, boost::test_tools::tolerance(1e-9));
    BOOST_TEST(a1 == 2.0, boost::test_tools::tolerance(1e-9));
}
BOOST_AUTO_TEST_CASE(simulated_prior_variance_recovers_its_law) {
    // Residuals drawn from log(chi2_3 / 3) + N(0, 0.5) give back about 0.5, at
    // any thread count.
    std::mt19937_64 rng(7);
    std::chi_squared_distribution<double> chi(3);
    std::normal_distribution<double> normal(0, std::sqrt(0.5));
    Vector residuals(20000);
    for (auto &r : residuals)
        r = std::log(chi(rng) / 3) + normal(rng);
    const double one = detail::simulated_prior_variance(residuals, 3, 1, 1);
    BOOST_TEST(one == detail::simulated_prior_variance(residuals, 3, 1, 4));
    BOOST_TEST(std::abs(one - 0.5) < 0.1);
}
BOOST_AUTO_TEST_CASE(size_factors_are_medians_of_ratios) {
    // Genes proportional to (0.5, 1, 2) across samples give exactly those factors;
    // the gene with a zero is left out by `ratio` and used by `poscounts`.
    const Vector c{0.5, 1, 2};
    std::vector<Vector> genes;
    for (double b : {10.0, 20.0, 40.0, 6.0})
        genes.push_back({c[0] * b, c[1] * b, c[2] * b});
    genes.push_back({0, 7, 3});
    const auto counts = dense_columns(genes);
    const auto ratio = estimate_size_factors(counts, SizeFactorMethod::ratio);
    for (std::size_t i = 0; i < 3; ++i)
        BOOST_TEST(ratio[i] == c[i], boost::test_tools::tolerance(1e-12));
    const auto pos = estimate_size_factors(counts, SizeFactorMethod::poscounts);
    BOOST_TEST(std::log(pos[0]) + std::log(pos[1]) + std::log(pos[2]) == 0.0,
               boost::test_tools::tolerance(1e-12));
    BOOST_CHECK_THROW(
        estimate_size_factors(dense_columns({{0, 1, 2}, {3, 0, 1}}), SizeFactorMethod::ratio),
        std::exception);
}
BOOST_AUTO_TEST_CASE(small_cholesky_matches_the_lapack_path) {
    // A 5 x 5 SPD block takes the in-house path; embedded in a 17 x 17 block
    // diagonal matrix with the identity it takes LAPACK. The solutions, inverses
    // and condition numbers agree.
    const std::size_t p = 5, q = detail::Cholesky::small + 1;
    Matrix a(p, p), big(q, q);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            a(i, j) = 1.0 / (1 + i + j) + (i == j ? 1e-3 * (i + 1) : 0);
    for (std::size_t i = 0; i < q; ++i)
        big(i, i) = 1;
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            big(i, j) = a(i, j);
    const detail::Cholesky small(a), lapack(big);
    BOOST_REQUIRE(small.ok);
    BOOST_REQUIRE(lapack.ok);
    Vector b{1, -2, 3, 0.5, 4}, b_big(q);
    std::copy(b.begin(), b.end(), b_big.begin());
    const auto x = small.solve(b), x_big = lapack.solve(b_big);
    const auto inv = small.inverse(), inv_big = lapack.inverse();
    for (std::size_t i = 0; i < p; ++i) {
        BOOST_TEST(x[i] == x_big[i], boost::test_tools::tolerance(1e-9));
        for (std::size_t j = 0; j < p; ++j) {
            BOOST_TEST(inv(i, j) == inv_big(i, j), boost::test_tools::tolerance(1e-9));
            BOOST_TEST(inv(i, j) == inv(j, i));
        }
    }
    // The in-house value is exact; LAPACK's estimate is not below it.
    BOOST_TEST(small.rcond <= lapack.rcond * (1 + 1e-9));
    BOOST_TEST(small.rcond >= lapack.rcond / 3);
    // Rank deficiency is detected.
    Matrix singular(3, 3, 1);
    BOOST_TEST(!detail::Cholesky(singular).ok);
}
BOOST_AUTO_TEST_CASE(log_theta_prior_moves_between_likelihood_and_prior) {
    auto b = simulate(1, 4, 1);
    const auto &y = b.y[0];
    FitOptions o;
    o.theta_min = 1e-8;
    o.theta_max = 1e8;
    Vector mu(y.size()), a(y.size(), 1);
    double mean = 0;
    for (double v : y)
        mean += v / y.size();
    std::fill(mu.begin(), mu.end(), std::max(mean, 0.5));
    const detail::CountTerms terms(y, a);
    const double ml = detail::estimate_theta(y, mu, a, 1, o, terms);
    const detail::LogThetaPrior wide{std::log(ml) + 3, 1e8}, tight{std::log(ml) + 3, 1e-10};
    BOOST_TEST(detail::estimate_theta(y, mu, a, 1, o, terms, nullptr, &wide) == ml,
               boost::test_tools::tolerance(1e-4));
    BOOST_TEST(detail::estimate_theta(y, mu, a, 1, o, terms, nullptr, &tight) ==
                   std::exp(tight.center),
               boost::test_tools::tolerance(1e-6));
    const detail::LogThetaPrior middle{std::log(ml) + 3, 0.5};
    const double map = detail::estimate_theta(y, mu, a, 1, o, terms, nullptr, &middle);
    BOOST_TEST(map > ml);
    BOOST_TEST(map < std::exp(middle.center));
}
BOOST_AUTO_TEST_CASE(bulk_de_finds_changes_and_is_thread_invariant) {
    auto b = simulate(2, 4);
    const auto counts = dense_columns(b.y);
    BulkOptions o;
    o.coefficient = 1;
    o.threads = 1;
    const auto one = bulk_de(b.x, counts, o);
    o.threads = 4;
    const auto four = bulk_de(b.x, counts, o);
    std::size_t calls = 0, true_calls = 0, de = 0, null = 0, null_05 = 0;
    for (std::size_t j = 0; j < one.genes.size(); ++j) {
        const auto &g = one.genes[j];
        BOOST_TEST(g.p_value == four.genes[j].p_value);
        BOOST_TEST(g.dispersion == four.genes[j].dispersion);
        if (std::isfinite(g.adjusted_p) && g.adjusted_p < 0.1) {
            ++calls;
            true_calls += b.de[j];
        }
        de += b.de[j];
        if (!b.de[j] && std::isfinite(g.p_value)) {
            ++null;
            null_05 += g.p_value < 0.05;
        }
        // A MAP estimate lies between the gene-wise estimate and the trend.
        if (g.status == "ok" && g.dispersion_gene >= 1e-7) {
            const double lo = std::min(g.dispersion_gene, g.dispersion_trend),
                         hi = std::max(g.dispersion_gene, g.dispersion_trend);
            BOOST_TEST(g.dispersion_map >= lo * (1 - 1e-6));
            BOOST_TEST(g.dispersion_map <= hi * (1 + 1e-6));
        }
    }
    BOOST_TEST(one.trend == "parametric");
    BOOST_TEST(one.asymptotic_dispersion > 0.02);
    BOOST_TEST(one.asymptotic_dispersion < 0.1);
    BOOST_TEST(one.prior_variance >= 0.25);
    BOOST_TEST(true_calls > de / 2);
    BOOST_TEST(calls - true_calls <= calls / 5);
    BOOST_TEST(null_05 <= null * 8 / 100);
    // The likelihood ratio test makes nearly the same calls.
    o.lrt = true;
    const auto lrt = bulk_de(b.x, counts, o);
    std::size_t both = 0, either = 0;
    for (std::size_t j = 0; j < one.genes.size(); ++j) {
        const bool w = one.genes[j].adjusted_p < 0.1, l = lrt.genes[j].adjusted_p < 0.1;
        both += w && l;
        either += w || l;
    }
    BOOST_TEST(both >= either * 9 / 10);
}
BOOST_AUTO_TEST_CASE(bulk_de_reports_zero_genes_and_infinite_changes) {
    auto b = simulate(3, 4, 200);
    b.y.push_back(Vector(8, 0));
    b.y.push_back({120, 90, 150, 110, 0, 0, 0, 0});
    BulkOptions o;
    o.coefficient = 1;
    const auto r = bulk_de(b.x, dense_columns(b.y), o);
    const auto &zero = r.genes[200], &gone = r.genes[201];
    BOOST_TEST(zero.status == "all_zero");
    BOOST_TEST(zero.base_mean == 0.0);
    BOOST_TEST(std::isnan(zero.p_value));
    BOOST_TEST(std::isnan(zero.adjusted_p));
    // A group of zeros: fitted with counts floored at 0.5 as DESeq2 fits them, so
    // the fold change is finite and the Wald test applies (DESeq2 gives -9.35,
    // SE 1.10 for such a gene).
    BOOST_TEST(gone.status == "ok");
    BOOST_TEST(gone.log2_fold_change < -5);
    BOOST_TEST(std::isfinite(gone.lfc_se));
    BOOST_TEST(gone.p_value < 1e-6);
    BOOST_TEST(gone.adjusted_p < 0.1);
    BOOST_CHECK_THROW(bulk_de(b.x, dense_columns({{1, 2, 3, 4, 5, 6, 7, 8.5}}), o), std::exception);
}
BOOST_AUTO_TEST_CASE(cooks_outliers_are_flagged_or_replaced) {
    // One extreme count in a steady gene. With three replicates per group it is
    // flagged (no p-value); with seven it is replaced and the gene is tested.
    // Both robust dispersions do so: DESeq2's as implemented in PyDESeq2, the
    // default, and RapidGLM's own.
    BOOST_TEST(BulkOptions().legacy_cooks);
    for (bool legacy : {true, false})
        for (std::size_t reps : {3u, 7u}) {
            auto b = simulate(4, reps, 300);
            auto &y = b.y[1]; // not a changed gene
            y.assign(2 * reps, 0);
            for (std::size_t i = 0; i < y.size(); ++i)
                y[i] = 200 + 10 * (i % 3);
            y[reps + 1] = 5000;
            BulkOptions o;
            o.coefficient = 1;
            o.legacy_cooks = legacy;
            auto r = bulk_de(b.x, dense_columns(b.y), o);
            const auto &g = r.genes[1];
            if (reps == 3) {
                BOOST_TEST(g.max_cooks > r.cooks_cutoff);
                BOOST_TEST(g.status == "cooks_outlier");
                BOOST_TEST(std::isnan(g.p_value));
                BOOST_TEST(!g.replaced);
            } else {
                BOOST_TEST(g.replaced);
                BOOST_TEST(g.status == "ok");
                BOOST_TEST(g.p_value > 0.05);
                BOOST_TEST(r.replaced_genes >= 1u);
                // Without replacement the same sample is flagged instead.
                o.min_replicates_for_replace = 0;
                r = bulk_de(b.x, dense_columns(b.y), o);
                BOOST_TEST(r.genes[1].status == "cooks_outlier");
                BOOST_TEST(r.replaced_genes == 0u);
            }
        }
}
BOOST_AUTO_TEST_CASE(weighted_quantile_scales_weights_to_the_sample_size) {
    // Equal weights give R's type-7 quantile.
    const Vector x{4, 1, 3, 2, 7};
    for (double p : {0.1, 0.5, 0.95})
        BOOST_TEST(detail::weighted_quantile(x, Vector(5, 2.5), p) == detail::quantile7(x, p),
                   boost::test_tools::tolerance(1e-14));
    // Weights (1, 2, 1) scale to (0.75, 1.5, 0.75): position 2.9 lies between the
    // values at cumulative weights 2 and 3.
    BOOST_TEST(detail::weighted_quantile({1, 2, 3}, {1, 2, 1}, 0.5) == 2.0);
    BOOST_TEST(detail::weighted_quantile({1, 2, 3}, {1, 2, 1}, 0.95) == 2.9,
               boost::test_tools::tolerance(1e-14));
}
BOOST_AUTO_TEST_CASE(map_fit_limits_match_the_mle_and_the_prior) {
    auto b = simulate(5, 4, 1);
    const auto &y = b.y[0];
    const std::size_t m = y.size();
    Vector offset(m, 0.1);
    FitOptions o;
    o.dispersion = Dispersion::fixed;
    o.theta = 1 / 0.2;
    o.offset = offset;
    const auto mle = fit(b.x, y, o);
    BOOST_REQUIRE(mle.ok());
    // A vanishing penalty gives the maximum likelihood fit and its covariance.
    const auto flat = detail::nb_map_fit(b.x, y, offset, 0.2, {1e-12, 1e-12}, Vector(2, 0));
    BOOST_REQUIRE(flat.ok);
    for (std::size_t j = 0; j < 2; ++j) {
        BOOST_TEST(flat.beta[j] == mle.coefficients[j], boost::test_tools::tolerance(1e-7));
        BOOST_TEST(std::sqrt(flat.covariance(j, j)) == mle.standard_errors[j],
                   boost::test_tools::tolerance(1e-6));
    }
    // A tight prior on the slope pulls it to zero; a moderate one shrinks it.
    const auto tight = detail::nb_map_fit(b.x, y, offset, 0.2, {1e-6, 1e8}, mle.coefficients);
    BOOST_REQUIRE(tight.ok);
    BOOST_TEST(std::abs(tight.beta[1]) < 1e-6);
    const auto moderate = detail::nb_map_fit(b.x, y, offset, 0.2, {1e-6, 2}, mle.coefficients);
    BOOST_REQUIRE(moderate.ok);
    BOOST_TEST(std::abs(moderate.beta[1]) < std::abs(mle.coefficients[1]));
    BOOST_TEST(moderate.beta[1] * mle.coefficients[1] > 0);
}
BOOST_AUTO_TEST_CASE(bulk_de_normal_shrinkage) {
    auto b = simulate(6, 4, 400);
    b.y.push_back({120, 90, 150, 110, 0, 0, 0, 0}); // no finite maximum likelihood change
    const auto counts = dense_columns(b.y);
    BulkOptions o;
    o.coefficient = 1;
    const auto plain = bulk_de(b.x, counts, o);
    o.lfc_shrinkage = LfcShrinkage::normal;
    o.threads = 3;
    const auto shrunk = bulk_de(b.x, counts, o);
    BOOST_REQUIRE(shrunk.lfc_prior_variance.size() == 2u);
    BOOST_TEST(shrunk.lfc_prior_variance[0] == 1e6); // constant intercept column
    BOOST_TEST(shrunk.lfc_prior_variance[1] > 0.05);
    BOOST_TEST(shrunk.lfc_prior_variance[1] < 5);
    std::size_t smaller = 0, finite = 0;
    for (std::size_t j = 0; j < plain.genes.size(); ++j) {
        const auto &p = plain.genes[j], &s = shrunk.genes[j];
        // Shrinkage leaves the tests alone.
        BOOST_TEST((p.p_value == s.p_value || (std::isnan(p.p_value) && std::isnan(s.p_value))));
        BOOST_TEST(std::isnan(p.shrunken_lfc));
        BOOST_REQUIRE(std::isfinite(s.shrunken_lfc));
        BOOST_TEST(s.shrunken_lfc_se > 0);
        if (std::isfinite(p.log2_fold_change)) {
            ++finite;
            smaller += std::abs(s.shrunken_lfc) <= std::abs(p.log2_fold_change) + 1e-9;
        }
    }
    BOOST_TEST(smaller == finite);
    BOOST_TEST(shrunk.genes.back().status == "ok");
    BOOST_TEST(shrunk.genes.back().shrunken_lfc < -1); // finite under the prior
    // A supplied, very wide prior returns the maximum likelihood change.
    o.lfc_prior_variance = {1e6, 1e8};
    const auto wide = bulk_de(b.x, counts, o);
    BOOST_TEST(wide.genes[0].shrunken_lfc == plain.genes[0].log2_fold_change,
               boost::test_tools::tolerance(1e-6));
}
BOOST_AUTO_TEST_CASE(nb_fixed_theta_fits_take_newton_steps) {
    // A high-dispersion gene on a five-column design: Newton steps converge in a
    // few iterations to a point where the NB score X's vanishes.
    Matrix x(8, 5);
    for (std::size_t i = 0; i < 8; ++i) {
        x(i, 0) = 1;
        if (i / 2 > 0)
            x(i, i / 2) = 1;
        x(i, 4) = i % 2;
    }
    const Vector y{0, 2, 0, 3, 0, 7, 3, 0};
    FitOptions o;
    o.dispersion = Dispersion::fixed;
    o.theta = 1 / 5.0;
    o.max_iterations = 1000;
    const auto f = fit(x, y, o);
    BOOST_REQUIRE(f.ok());
    BOOST_TEST(f.iterations <= 30u);
    const auto g = fit(x, Vector{12, 30, 8, 25, 14, 40, 9, 20}, o);
    BOOST_REQUIRE(g.status == FitStatus::converged);
    BOOST_TEST(g.iterations <= 8u);
    Vector score(5);
    const Vector yy{12, 30, 8, 25, 14, 40, 9, 20};
    for (std::size_t i = 0; i < 8; ++i) {
        double eta = 0;
        for (std::size_t j = 0; j < 5; ++j)
            eta += x(i, j) * g.coefficients[j];
        const double mu = std::exp(eta), r = (yy[i] - mu) / (1 + mu / o.theta);
        for (std::size_t j = 0; j < 5; ++j)
            score[j] += x(i, j) * r;
    }
    for (double s : score)
        BOOST_TEST(std::abs(s) < 1e-6);
}
BOOST_AUTO_TEST_CASE(legacy_cooks_dispersion_follows_pydeseq2) {
    // Reference values from PyDESeq2's robust_method_of_moments_disp (MIT),
    // recomputed independently in NumPy: cells of 4, 5 and 1 samples (the last
    // ignored), and a design without any cell of three or more samples.
    const Vector q = {10, 50, 5, 30, 100, 20, 60, 80, 200, 7};
    Matrix x(10, 3, 0.0);
    for (std::size_t i = 0; i < 10; ++i) {
        x(i, 0) = 1;
        x(i, 1) = i >= 4 && i < 9;
        x(i, 2) = i == 9;
    }
    const double mean = std::accumulate(q.begin(), q.end(), 0.0) / q.size();
    BOOST_TEST(detail::legacy_cooks_dispersion(q, x, mean) == 0.8459239371335217,
               boost::test_tools::tolerance(1e-13));
    const Vector q2 = {3, 90, 15, 60, 8, 120, 40, 25};
    Matrix x2(8, 2, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        x2(i, 0) = 1;
        x2(i, 1) = static_cast<double>(i);
    }
    const double mean2 = std::accumulate(q2.begin(), q2.end(), 0.0) / q2.size();
    BOOST_TEST(detail::legacy_cooks_dispersion(q2, x2, mean2) == 0.7339355727609347,
               boost::test_tools::tolerance(1e-13));
    // The 0.04 floor.
    const Vector flat = {100, 101, 99, 100, 100, 102};
    BOOST_TEST(detail::legacy_cooks_dispersion(flat, Matrix(6, 1, 1.0), 100.333333) == 0.04);
}

BOOST_AUTO_TEST_CASE(floored_fits_optimize_bounds_and_covariate_units) {
    Matrix x(20, 2, 1);
    Vector y(20), offset(20);
    for (std::size_t i = 0; i < 20; ++i) {
        x(i, 1) = i >= 10;
        y[i] = i < 10 ? 0 : 10 + i % 4;
    }
    const auto a = detail::floored_nb_fit(x, y, offset, 0.2, 0.5, {}, 250, 1e-8, -30, 30, true);
    for (std::size_t i = 0; i < 20; ++i)
        x(i, 1) *= 0.01;
    const auto b = detail::floored_nb_fit(x, y, offset, 0.2, 0.5, {}, 250, 1e-8, -30, 30, true);
    BOOST_REQUIRE(a.converged);
    BOOST_REQUIRE(b.converged);
    BOOST_TEST(a.deviance == b.deviance, boost::test_tools::tolerance(1e-10));
    BOOST_TEST(a.beta[1] == b.beta[1] * 0.01, boost::test_tools::tolerance(1e-9));
    auto bound = detail::floored_nb_fit(x, y, offset, 0.2, 0.5, {}, 1, 1e-8, -1, 1);
    BOOST_TEST(bound.used_optimizer);
    BOOST_TEST(bound.beta[1] <= 1);
    // Whole-procedure p-values must also survive a change of units.
    auto d = simulate(992, 6, 150);
    d.y[0] = y = Vector(12, 10);
    for (std::size_t i = 0; i < 6; ++i)
        d.y[0][i] = 0;
    BulkOptions o;
    o.coefficient = 1;
    o.cooks_filter = false;
    const auto first = bulk_de(d.x, dense_columns(d.y), o);
    for (std::size_t i = 0; i < d.x.rows; ++i)
        d.x(i, 1) *= 0.01;
    const auto second = bulk_de(d.x, dense_columns(d.y), o);
    BOOST_TEST(first.genes[0].p_value == second.genes[0].p_value,
               boost::test_tools::tolerance(1e-5));
}

BOOST_AUTO_TEST_CASE(independent_filter_few_rejections_and_positive_residual_rms) {
    Vector p(200, .9), mean(200);
    for (std::size_t i = 0; i < 200; ++i)
        mean[i] = i + 1;
    for (std::size_t i = 195; i < 200; ++i)
        p[i] = 1e-8;
    auto few = independent_filter(p, mean, .1, true);
    BOOST_TEST(few.theta == 0);
    for (std::size_t i = 155; i < 200; ++i)
        p[i] = .008 + .0001 * (i - 155);
    auto many = independent_filter(p, mean, .1, true);
    double sq = 0, peak = *std::max_element(many.fit.begin(), many.fit.end());
    std::size_t positive = 0, chosen = 0;
    for (std::size_t k = 0; k < many.fit.size(); ++k)
        if (many.rejections[k].second > 0) {
            ++positive;
            sq += std::pow(many.rejections[k].second - many.fit[k], 2);
        }
    const double threshold = peak - std::sqrt(sq / positive);
    for (std::size_t k = 0; k < many.fit.size(); ++k)
        if (many.rejections[k].second > threshold) {
            chosen = k;
            break;
        }
    BOOST_TEST(many.theta == many.rejections[chosen].first);
}

BOOST_AUTO_TEST_CASE(cooks_replacement_all_zero_discards_first_test) {
    auto d = simulate(101, 7, 250);
    d.y[0].assign(14, 0);
    d.y[0][13] = 50;
    BulkOptions o;
    o.coefficient = 1;
    o.independent_filtering = false;
    o.size_factors.assign(14, 1);
    const auto r = bulk_de(d.x, dense_columns(d.y), o);
    const auto &g = r.genes[0];
    BOOST_REQUIRE(g.replaced);
    BOOST_TEST(g.base_mean == 0);
    BOOST_TEST(g.p_value == 1);
    BOOST_TEST(g.adjusted_p == 1);
    BOOST_TEST(g.statistic == 0);
    BOOST_TEST(g.status == "replacement_all_zero");
}

BOOST_AUTO_TEST_CASE(floored_lrt_and_replacement_have_explicit_status) {
    auto d = simulate(91, 7, 200);
    d.y[0].assign(14, 0);
    for (std::size_t i = 7; i < 14; ++i)
        d.y[0][i] = 20 + i % 3;
    BulkOptions o;
    o.coefficient = 1;
    o.lrt = true;
    o.cooks_filter = false;
    auto r = bulk_de(d.x, dense_columns(d.y), o);
    BOOST_TEST(r.genes[0].status == "ok");
    BOOST_TEST(r.genes[0].p_value < 1e-6);
    // A failed floor optimization is explicitly unsuccessful, even if the
    // bounded iterates and means happen to be finite.
    Matrix x(4, 2, 1);
    Vector y{0, 0, 10, 10};
    auto f = detail::floored_nb_fit(x, y, Vector(4), .2, .5);
    BOOST_TEST(!f.converged);
}

BOOST_AUTO_TEST_CASE(floored_fit_pins_deseq2_black_box_reference) {
    // DESeq2 1.48.2, nbinomWaldTest(minmu=.5), supplied factors=1 and
    // dispersions=.2; scripts/reference_review_regressions.R reproduces it.
    Matrix x(20, 2, 1);
    Vector y(20), offset(20);
    for (std::size_t i = 0; i < 20; ++i) {
        x(i, 1) = i >= 10;
        y[i] = i < 10 ? 0 : 10 + i % 4;
    }
    auto f = detail::floored_nb_fit(x, y, offset, .2, .5);
    BOOST_REQUIRE(f.converged);
    BOOST_TEST(f.beta[1] / std::log(2.) == 5.991127447802053, boost::test_tools::tolerance(1e-6));
    Vector w(20);
    for (std::size_t i = 0; i < 20; ++i)
        w[i] = f.floored[i] / (1 + .2 * f.floored[i]);
    auto h = detail::crossprod(x, w), a = h;
    for (std::size_t j = 0; j < 2; ++j)
        a(j, j) += 1e-6;
    auto inverse = detail::Cholesky(a).inverse();
    double variance = 0;
    for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t k = 0; k < 2; ++k)
            variance += inverse(1, j) * h(j, k) * inverse(k, 1);
    BOOST_TEST(std::sqrt(variance) / std::log(2.) == .7192475816565518,
               boost::test_tools::tolerance(1e-6));
    const auto null = detail::floored_nb_fit(Matrix(20, 1, 1), y, offset, .2, .5);
    BOOST_REQUIRE(null.converged);
    const double lr = null.deviance - f.deviance;
    // DESeq2 1.48.2 nbinomLRT on the same supplied factors and dispersion.
    BOOST_TEST(lr == 92.02103007895477, boost::test_tools::tolerance(1e-7));
    BOOST_TEST(boost::math::cdf(boost::math::complement(
                   boost::math::chi_squared_distribution<double>(1), lr)) == 8.576025085782601e-22,
               boost::test_tools::tolerance(1e-5));
}

BOOST_AUTO_TEST_CASE(successful_cooks_replacement_refits_a_floored_gene) {
    auto d = simulate(712, 7, 200);
    d.y[0] = {0, 0, 0, 0, 0, 0, 0, 20, 21, 19, 20, 22, 18, 5000};
    BulkOptions o;
    o.coefficient = 1;
    o.size_factors.assign(14, 1);
    const auto r = bulk_de(d.x, dense_columns(d.y), o);
    const auto &g = r.genes[0];
    BOOST_REQUIRE(g.replaced);
    BOOST_TEST(g.status == "ok");
    BOOST_TEST(g.base_mean == 129.0 / 14, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(std::isfinite(g.log2_fold_change));
    BOOST_TEST(std::isfinite(g.lfc_se));
    BOOST_TEST(g.p_value < 1e-6);
}

BOOST_AUTO_TEST_CASE(moments_start_prefers_the_smaller_pydeseq2_estimate) {
    // PyDESeq2 0.5.2 public helpers on this column: rough=0.66873563,
    // moments=-0.37142857. The minimum starts at the dispersion floor.
    auto d = simulate(711, 4, 200);
    for (std::size_t i = 0; i < 8; ++i)
        d.x(i, 1) = i;
    d.y[0] = {0, 0, 5, 20, 10, 40, 5, 20};
    BulkOptions o;
    o.coefficient = 1;
    o.cooks_filter = false;
    o.size_factors = {.01, .02, 1, 1, 1, 1, 1, 1};
    const auto kept = bulk_de(d.x, dense_columns(d.y), o);
    BOOST_TEST(kept.genes[0].dispersion_gene == o.min_dispersion);
    o.keep_minimum_start = false;
    const auto optimized = bulk_de(d.x, dense_columns(d.y), o);
    BOOST_TEST(optimized.genes[0].dispersion_gene > .001);
}
