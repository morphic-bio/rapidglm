#pragma once
#include "batch_scoring.hpp"
#include "hypothesis.hpp"
#include "mergeable_null.hpp"
#include "resampling.hpp"
#include "shrinkage.hpp"
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <iterator>
#include <map>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
namespace rapidglm {
inline Matrix subset_rows(const Matrix &x, const std::vector<std::size_t> &rows) {
    Matrix out(rows.size(), x.cols);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < x.cols; ++j)
            out(i, j) = x(rows[i], j);
    return out;
}
inline Vector subset(const Vector &x, const std::vector<std::size_t> &rows) {
    if (x.empty())
        return {};
    Vector out;
    out.reserve(rows.size());
    for (auto i : rows)
        out.push_back(x.at(i));
    return out;
}
inline Matrix append_column(const Matrix &x, const Vector &g) {
    require(x.rows == g.size(), "column size mismatch");
    Matrix out(x.rows, x.cols + 1);
    for (std::size_t i = 0; i < x.rows; ++i) {
        for (std::size_t j = 0; j < x.cols; ++j)
            out(i, j) = x(i, j);
        out(i, x.cols) = g[i];
    }
    return out;
}
inline Matrix remove_column(const Matrix &x, std::size_t column) {
    require(column < x.cols && x.cols > 1, "invalid reduced design column");
    Matrix out(x.rows, x.cols - 1);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t j = 0, k = 0; j < x.cols; ++j)
            if (j != column)
                out(i, k++) = x(i, j);
    return out;
}
struct FeatureTest {
    FitResult fit;
    TestResult test;
    double adjusted_p = nan, log2fc = nan, log2fc_se = nan;
    // The tested coefficient or contrast and its standard error (one-row
    // hypotheses), and whether a tested coefficient has no finite estimate.
    double estimate = nan, standard_error = nan;
    bool boundary = false;
};
namespace detail {
inline void record_estimate(FeatureTest &r, const Hypothesis &h) {
    r.boundary = hypothesis_boundary(r.fit, h);
    if (h.df() == 1 && r.fit.parameters == h.contrasts.cols &&
        r.fit.coefficients.size() == h.contrasts.cols)
        std::tie(r.estimate, r.standard_error) = hypothesis_estimate(r.fit, h);
}
} // namespace detail
// Tests the hypothesis for every response: a Wald test (one-row hypotheses) or a
// likelihood ratio test against the reduced design. For a one-row hypothesis the
// full model is fitted in tested_design(design, hypothesis), so `fit` holds that
// parametrization, whose coefficient `column` is the tested contrast.
inline std::vector<FeatureTest> regression_tests(const Matrix &design, const SparseCSC &responses,
                                                 FitOptions options, const Hypothesis &hypothesis,
                                                 bool lr = false, int threads = 1) {
    responses.validate();
    require(design.rows == responses.rows && hypothesis.contrasts.cols == design.cols,
            "regression dimensions mismatch");
    require(lr || hypothesis.df() == 1,
            "the Wald test takes one coefficient or contrast; use the likelihood ratio test");
    options.retain_observations = false;
    std::vector<FeatureTest> results(responses.cols);
    Vector p(responses.cols, nan);
    Matrix reduced;
    if (lr)
        reduced = reduced_design(design, hypothesis);
    const auto tested =
        hypothesis.df() == 1 ? tested_design(design, hypothesis) : TestedDesign{design, 0};
    const auto target =
        hypothesis.df() == 1 ? Hypothesis::coefficient(design.cols, tested.column) : hypothesis;
    parallel_for(responses.cols, threads, [&](auto j) {
        auto y = responses.column(j);
        auto &r = results[j];
        r.fit = fit(tested.x, y, options);
        if (lr) {
            auto null_options = options;
            null_options.start.clear();
            r.test = likelihood_ratio(fit(reduced, y, null_options), r.fit);
        } else
            r.test = hypothesis_wald(r.fit, target, Side::both, options.robust);
        detail::record_estimate(r, target);
        if (r.fit.ok() &&
            (options.family == Family::poisson || options.family == Family::negative_binomial)) {
            auto effect = log2_fold_change(r.fit, target, options.robust);
            r.log2fc = effect.estimate;
            r.log2fc_se = effect.standard_error;
        }
        p[j] = r.test.p_value;
    });
    auto q = adjust_pvalues(p);
    for (std::size_t j = 0; j < q.size(); ++j)
        results[j].adjusted_p = q[j];
    return results;
}
inline std::vector<FeatureTest> regression_tests(const Matrix &design, const SparseCSC &responses,
                                                 FitOptions options, std::size_t coefficient,
                                                 bool lr = false, int threads = 1) {
    require(coefficient < design.cols, "regression dimensions mismatch");
    return regression_tests(design, responses, std::move(options),
                            Hypothesis::coefficient(design.cols, coefficient), lr, threads);
}
struct QuasiLikelihoodFeature {
    double dispersion = nan; // Cox-Reid adjusted ML NB dispersion, 1/theta
    double trend = nan;      // trended dispersion used by the tested fits
    double mean = nan;       // mean fitted count of the dispersion fit
    double ql = nan;         // (1 + mean dispersion) / (1 + mean trend)
    double prior = nan, posterior = nan, df_residual = nan;
};
struct QuasiLikelihoodResult {
    std::vector<FeatureTest> tests;
    std::vector<QuasiLikelihoodFeature> features;
    double df_prior = nan;
    bool abundance_trend = false;
    std::size_t trend_window = 0, dispersion_features = 0;
};
// Quasi-likelihood F test with empirical Bayes shrinkage (Lund et al. 2012;
// recipe of glmGamPoi, Ahlmann-Eltze and Huber 2020):
// 1. per-feature NB fit with Cox-Reid adjusted ML dispersion;
// 2. weighted running median of the dispersions against mean fitted count
//    (window max(10% of features, 100));
// 3. full and reduced fits at the trended dispersion; deviance difference D;
// 4. QL dispersion s2 = (1 + m phi) / (1 + m phi_trend) on the residual df;
// 5. Smyth's scaled F prior for s2 (spline abundance trend with >= 100 features)
//    and the shrunken s2_post;
// 6. F = D / (df_test s2_post) on (df_test, d0 + df_residual) degrees of freedom.
// Standard errors and covariances of the reported trended fit are scaled by s2_post.
inline QuasiLikelihoodResult quasi_likelihood_tests(const Matrix &design,
                                                    const SparseCSC &responses, FitOptions options,
                                                    const Hypothesis &hypothesis, int threads = 1) {
    responses.validate();
    require(design.rows == responses.rows && hypothesis.contrasts.cols == design.cols,
            "regression dimensions mismatch");
    require(options.family == Family::negative_binomial && options.dispersion == Dispersion::ml &&
                options.ridge == 0 && !options.robust && options.weights.empty(),
            "the quasi-likelihood test requires unweighted, unpenalized NB fits with ML "
            "dispersion");
    const std::size_t m = responses.cols;
    QuasiLikelihoodResult out;
    out.tests.resize(m);
    out.features.resize(m);
    Vector deviance(m, nan), tested_df(m, nan);
    std::vector<unsigned char> dispersion_ok(m, 0);
    auto dispersion_options = options;
    dispersion_options.cox_reid = true;
    dispersion_options.retain_observations = true;
    if (dispersion_options.weights.empty())
        dispersion_options.design = prepare_design(design, dispersion_options.rank_tolerance);
    parallel_for(m, threads, [&](auto j) {
        auto f = fit(design, responses.column(j), dispersion_options);
        auto &q = out.features[j];
        if (f.ok() && std::isfinite(f.theta) && f.theta > 0) {
            q.dispersion = 1 / f.theta;
            q.mean = std::accumulate(f.fitted.begin(), f.fitted.end(), 0.0) / f.fitted.size();
            q.df_residual = static_cast<double>(f.observations) -
                            static_cast<double>(f.rank ? f.rank : f.parameters);
            dispersion_ok[j] = std::isfinite(q.mean) && q.df_residual > 0;
        }
        f.compact();
        out.tests[j].fit = std::move(f);
    });
    std::vector<std::size_t> usable;
    for (std::size_t j = 0; j < m; ++j)
        if (dispersion_ok[j])
            usable.push_back(j);
    require(usable.size() >= 2, "the quasi-likelihood test needs at least two features with "
                                "valid dispersion estimates");
    out.dispersion_features = usable.size();
    out.trend_window = std::max<std::size_t>(usable.size() / 10, 100);
    {
        Vector x, y;
        for (auto j : usable) {
            x.push_back(out.features[j].mean);
            y.push_back(out.features[j].dispersion);
        }
        const auto trend = running_median(x, y, out.trend_window);
        for (std::size_t k = 0; k < usable.size(); ++k)
            out.features[usable[k]].trend = trend[k];
    }
    auto trended = options;
    trended.dispersion = Dispersion::fixed;
    trended.cox_reid = false;
    trended.retain_observations = false;
    const auto reduced = reduced_design(design, hypothesis);
    // One-row hypotheses report the tested contrast as a coefficient.
    const auto tested =
        hypothesis.df() == 1 ? tested_design(design, hypothesis) : TestedDesign{design, 0};
    const auto target =
        hypothesis.df() == 1 ? Hypothesis::coefficient(design.cols, tested.column) : hypothesis;
    std::vector<unsigned char> tested_fit(m,
                                          0); // fit of the tested design (else the dispersion fit)
    parallel_for(usable.size(), threads, [&](auto k) {
        const auto j = usable[k];
        auto &q = out.features[j];
        auto o = trended;
        o.theta = q.trend > 0 ? std::clamp(1 / q.trend, o.theta_min, o.theta_max) : o.theta_max;
        const auto y = responses.column(j);
        auto full = fit(tested.x, y, o);
        auto null_options = o;
        null_options.start.clear();
        const auto null = fit(reduced, y, null_options);
        const auto lr = likelihood_ratio(null, full);
        if (lr.valid) {
            // At a common fixed dispersion the LR statistic is the deviance difference.
            const auto rank = [](const FitResult &f) { return f.rank ? f.rank : f.parameters; };
            deviance[j] = lr.statistic;
            tested_df[j] = static_cast<double>(rank(full) - rank(null));
        }
        q.ql = (1 + q.mean * q.dispersion) / (1 + q.mean * q.trend);
        out.tests[j].fit = std::move(full);
        tested_fit[j] = 1;
    });
    Vector s2(m, nan), df(m, nan), means(m, nan);
    for (auto j : usable) {
        s2[j] = out.features[j].ql;
        df[j] = out.features[j].df_residual;
        means[j] = out.features[j].mean;
    }
    const auto prior = fit_variance_prior(s2, df, means, usable.size() >= 100);
    out.df_prior = prior.df;
    out.abundance_trend = prior.trend;
    Vector p(m, nan);
    for (std::size_t j = 0; j < m; ++j) {
        auto &q = out.features[j];
        auto &r = out.tests[j];
        q.prior = prior.scale[j];
        q.posterior = prior.posterior[j];
        if (!(q.posterior > 0) || !std::isfinite(deviance[j]) || !(tested_df[j] > 0))
            continue;
        const double f = deviance[j] / (tested_df[j] * q.posterior);
        boost::math::fisher_f_distribution<double> dist(tested_df[j], prior.df + q.df_residual);
        r.test = {f, boost::math::cdf(boost::math::complement(dist, f)), nan, nan, true};
        p[j] = r.test.p_value;
    }
    for (std::size_t j = 0; j < m; ++j) {
        auto &r = out.tests[j];
        const double scale = out.features[j].posterior;
        if (r.fit.ok() && std::isfinite(scale) && scale > 0) {
            r.fit.scale = scale;
            for (double &v : r.fit.covariance.values)
                v *= scale;
            for (double &se : r.fit.standard_errors)
                se *= std::sqrt(scale);
            auto effect = log2_fold_change(r.fit, target);
            r.log2fc = effect.estimate;
            r.log2fc_se = effect.standard_error;
        }
        detail::record_estimate(r, tested_fit[j] ? target : hypothesis);
    }
    auto q = adjust_pvalues(p);
    for (std::size_t j = 0; j < m; ++j)
        out.tests[j].adjusted_p = q[j];
    return out;
}
inline QuasiLikelihoodResult quasi_likelihood_tests(const Matrix &design,
                                                    const SparseCSC &responses, FitOptions options,
                                                    std::size_t coefficient, int threads = 1) {
    require(coefficient < design.cols, "regression dimensions mismatch");
    return quasi_likelihood_tests(design, responses, std::move(options),
                                  Hypothesis::coefficient(design.cols, coefficient), threads);
}
inline SparseCSC aggregate_pseudobulk(const SparseCSC &counts, const std::vector<std::size_t> &unit,
                                      std::size_t units) {
    counts.validate();
    require(unit.size() == counts.rows && units > 0, "invalid pseudobulk units");
    std::vector<bool> present(units, false);
    for (auto u : unit) {
        require(u < units, "pseudobulk unit out of range");
        present[u] = true;
    }
    require(std::all_of(present.begin(), present.end(), [](bool p) { return p; }),
            "empty pseudobulk unit");
    SparseCSC out;
    out.rows = units;
    out.cols = counts.cols;
    out.column_ptr = {0};
    for (std::size_t j = 0; j < counts.cols; ++j) {
        Vector y(units);
        for (auto k = counts.column_ptr[j]; k < counts.column_ptr[j + 1]; ++k) {
            require(counts.values[k] >= 0 && std::floor(counts.values[k]) == counts.values[k],
                    "pseudobulk requires integer counts");
            y[unit[counts.row_index[k]]] += counts.values[k];
        }
        for (std::size_t u = 0; u < units; ++u)
            if (y[u] != 0) {
                out.row_index.push_back(u);
                out.values.push_back(y[u]);
            }
        out.column_ptr.push_back(out.values.size());
    }
    return out;
}
inline std::vector<FeatureTest>
differential_accessibility(const Matrix &covariates, const SparseCSC &peaks, const Vector &group,
                           FitOptions options = {}, int threads = 1) {
    peaks.validate();
    require(covariates.rows == peaks.rows && group.size() == peaks.rows, "DA dimensions mismatch");
    options.family = Family::binomial;
    options.retain_observations = false;
    auto null = fit(covariates, group, options);
    std::vector<FeatureTest> out(peaks.cols);
    Vector p(peaks.cols, nan);
    parallel_for(peaks.cols, threads, [&](auto j) {
        auto x = append_column(covariates, peaks.column(j));
        auto &r = out[j];
        r.fit = fit(x, group, options);
        r.test = likelihood_ratio(null, r.fit);
        detail::record_estimate(r, Hypothesis::coefficient(x.cols, covariates.cols));
        p[j] = r.test.p_value;
    });
    auto q = adjust_pvalues(p);
    for (std::size_t j = 0; j < q.size(); ++j)
        out[j].adjusted_p = q[j];
    return out;
}
enum class Integration { union_guides, singleton, bonferroni };
// Which tested pairs also get a full GLM effect fit (adjusted log2FC and SE).
// The resampling p-value and mean-ratio effect never depend on this choice.
enum class EffectFits { all, significant, none };
enum class PairKind { discovery, negative_control, positive_control };
// What a pair's random stream is keyed by.
// identity: the seed, the response and the target alone (the guide for a
// per-guide test), so a pair's record is the same wherever it is listed, with
// whatever other pairs, and however the run is split. The response is
// identified by its ID (SceptreOptions::response_ids) or else by its column,
// the target by its name: FNV-1a keys of the strings (name_key), turned into a
// stream by keyed_stream(). A pair may be listed once within a kind.
// position: the pair's place in the list (releases before 1.3.0).
enum class StreamKeys { identity, position };
// The family within which p-values of a kind are adjusted.
// tested: the pairs that pass pair QC (a pair that passes and whose test then
// fails counts with p = 1); the other pairs get no adjusted p-value.
// listed: every listed pair, those that fail pair QC counting with p = 1
// (releases before 1.3.0).
enum class AdjustmentFamily { tested, listed };
struct Guide {
    std::string id, target;
    bool non_targeting = false;
};
struct Pair {
    std::size_t response;
    std::string target;
    PairKind kind = PairKind::discovery;
};
struct CellQC {
    Vector total_counts, detected_features, mitochondrial_fraction;
    std::vector<bool> keep;
};
// How a pair's null is started with non-targeting controls and the SCEPTRE
// null model (SceptreOptions::null_start).
enum class NullStart { refine, cold };
struct SceptreOptions {
    bool high_moi = false, nt_controls = true, crt = false, cell_qc = true, assignment_qc = true,
         bonferroni_adjustment = false, parallel_draws = true, parallel_pairs = true,
         warm_start_effects = true;
    // SCEPTRE's documented response model: Poisson coefficients, conditional NB theta.
    FitMethod null_model = FitMethod::sceptre_null;
    // With non-targeting controls and null_model=sceptre_null, each pair's null
    // (fitted on its target's and the NT cells) starts by default from the
    // response's reference: the same null fitted once on all retained cells. The
    // pair takes one Fisher-scoring step from it using the NT cells' totals and
    // its target cells' totals at the reference, then exact Newton passes over
    // its cells to the convergence criteria of the cold fit, with step halving
    // (mergeable_null.hpp). A pair that does not converge is fitted cold. cold
    // fits every pair from scratch, as releases before 1.6.0 did. The two agree
    // to the convergence tolerance; empirical p-values are those of the cold fit
    // unless a draw lies within rounding of the statistic.
    NullStart null_start = NullStart::refine;
    Integration integration = Integration::union_guides;
    // significant: pairs with adjusted p <= alpha in their testing family.
    EffectFits effect_fits = EffectFits::significant;
    // hazard is thinned on a processor without AES instructions
    // (aes_instructions()), so CRT p-values depend on the kind of processor.
    CrtDraw crt_draw = CrtDraw::hazard;
    // Memory for CRT samplers shared by all pairs of a target (complement
    // controls; 12 bytes per cell per target). By default every target keeps
    // its sampler. Targets beyond a smaller budget rebuild their sampler for
    // each pair, which is slow: SceptreResult::samplers_rebuilt counts them.
    // Results do not depend on it.
    std::size_t sampler_cache_bytes = std::numeric_limits<std::size_t>::max();
    // CRT with complement controls only: draw each target's resamples once and
    // score every gene against them (random streams keyed by target instead of
    // by pair). P-values then differ from the default, which draws afresh for
    // every pair, and tests that share a target share their resamples. The
    // first stage's draws are kept in memory (about 4 bytes per treated cell
    // per draw per target) within draw_cache_bytes; results do not depend on it.
    bool reuse_draws = false;
    // CRT with complement controls only: screen each pair first with its
    // target's shared first-stage draws (ResampleOptions::prescreen). A pair
    // whose shared p-value is above shared_screen_p ends there with it; every
    // other pair is then tested from its own draws exactly as in the default, so
    // its p-value is the default's. A pair the default would have continued can
    // end at the shared screen; a threshold above the default's 0.02 makes that
    // rare and leaves small p-values unaffected. The shared draws are kept in
    // memory within draw_cache_bytes; results do not depend on it.
    bool shared_screen = false;
    double shared_screen_p = 0.05;
    std::size_t draw_cache_bytes = std::size_t(8) << 30;
    // CRT with complement controls: score each response's pairs together, stage
    // by stage, all their draws of a stage in one pass over the response's score
    // records in blocks of cells (a sparse x dense product, detail::score_draws),
    // one response per thread. The records are bit for bit those of scoring the
    // pairs one by one in the pair pipeline (false), which is also used with
    // shared_screen, sequential_screen or early_stop.
    bool batch_scoring = true;
    StreamKeys stream_keys = StreamKeys::identity;
    // With identity keys: one ID per response column (any distinct strings that
    // stay with the response, such as gene or peak names). Empty: the column
    // index, which identifies a response only while the matrix keeps its columns.
    std::vector<std::string> response_ids;
    AdjustmentFamily family = AdjustmentFamily::tested;
    // The number of hypotheses in each kind's family (discovery, negative
    // control, positive control) when it is larger than the pairs of the family
    // at hand, for a caller that lists only some of them; 0: those at hand. The
    // pairs not listed count with p = 1.
    std::array<std::size_t, 3> family_size{};
    std::size_t min_nonzero = 7;
    double lower_quantile = 0.01, upper_quantile = 0.99, max_mito = 0.2, alpha = 0.1;
    FitOptions model;
    // SCEPTRE's staging: an empirical first stage, continued only at p <= 0.02.
    ResampleOptions resampling = [] {
        ResampleOptions r;
        r.screen = 0.02;
        return r;
    }();
    int threads = 1;
};
struct PairResult {
    std::size_t response = 0;
    std::string target, guide, status = "not_run";
    PairKind kind = PairKind::discovery;
    std::size_t cells = 0, nonzero_treatment = 0, nonzero_control = 0, draws = 0;
    // adjusted_p is missing for a pair outside the adjustment family.
    double statistic = nan, p_value = nan, adjusted_p = 1, log2fc = nan, log2fc_se = nan,
           sceptre_mean_ratio = nan, mean_ratio_log2 = nan;
    bool approximation = false;
};
struct QQPoint {
    double expected, observed, lower, upper;
};
struct SceptreResult {
    std::vector<PairResult> tests;
    std::vector<QQPoint> calibration_qq;
    std::size_t retained_cells = 0, calibration_rejections = 0, power_rejections = 0,
                discovery_rejections = 0;
    // By kind (discovery, negative control, positive control): the pairs
    // listed, those that pass pair QC, and the size of the adjustment family.
    std::array<std::size_t, 3> pairs_listed{}, pairs_passing_qc{}, family_size{};
    // Null models fitted: one per response with a pair that passes pair QC
    // (complement controls), or one per such pair (non-targeting controls).
    std::size_t null_fits = 0;
    // The null start used (refine only with NT controls and the SCEPTRE null),
    // the pairs whose null was refined from the reference, and those fitted cold
    // because refining did not converge.
    NullStart null_start_used = NullStart::cold;
    std::size_t null_refined = 0, null_cold_refits = 0;
    // Targets whose CRT sampler was kept for all their pairs, and targets that
    // did not fit SceptreOptions::sampler_cache_bytes and rebuilt it per pair.
    std::size_t samplers_cached = 0, samplers_rebuilt = 0;
};
// Splitting a sceptre() run into shards (complement controls only), for
// checkpoints and for running shards on separate workers. Tasks are numbered as
// in sceptre_layout(); draws are keyed by task or target, never by schedule, so
// the shards' records are exactly those of a single run.
struct SceptreShard {
    // Compute only these tasks (ascending). The result then holds their raw
    // records in this order, without adjustment, calibration summary or deferred
    // effect fits.
    const std::vector<std::size_t> *tasks = nullptr;
    // The raw records of every task, from earlier shards: only adjust, summarize
    // and fit the deferred effects.
    const std::vector<PairResult> *results = nullptr;
    // Response null fits from sceptre_nulls(), by response index. Each response's
    // score state is rebuilt from its fit instead of refitting it.
    const std::vector<FitResult> *nulls = nullptr;
};
inline double quantile(Vector values, double p) {
    require(!values.empty() && p >= 0 && p <= 1, "invalid quantile");
    std::sort(values.begin(), values.end());
    double h = p * (values.size() - 1);
    auto lo = static_cast<std::size_t>(h);
    return values[lo] + (h - lo) * (values[std::min(lo + 1, values.size() - 1)] - values[lo]);
}
// Reuse a regular nested null fit only as an initial point. Estimate the full
// effect model, including NB ML dispersion and coefficient covariance.
inline FitResult fit_effect_model(const Matrix &x, const Vector &y, const Vector &g,
                                  const FitResult &null, FitOptions options,
                                  bool warm_start = true) {
    const auto augmented = append_column(x, g);
    options.start.clear();
    options.nb_warm_start = false;
    options.retain_observations = false;
    const bool nb = options.family == Family::negative_binomial;
    if (warm_start && null.ok() && !null.has_boundary() && null.coefficients.size() == x.cols &&
        (!nb || (options.dispersion == Dispersion::fixed ||
                 (options.dispersion == Dispersion::ml && !null.theta_boundary)))) {
        auto initial = options;
        initial.start = null.coefficients;
        initial.start.push_back(0);
        if (nb && options.dispersion == Dispersion::ml) {
            initial.theta = null.theta;
            initial.nb_warm_start = true;
        }
        auto candidate = fit(augmented, y, initial);
        if (candidate.ok() && !candidate.has_boundary() && !candidate.theta_boundary)
            return candidate;
        // Retain the established boundary and failure behavior of fresh fits.
    }
    return fit(augmented, y, options);
}
namespace detail {
// The O(p) part of a null fit that fit_effect_model uses as a warm start.
inline FitResult effect_start(const FitResult &null) {
    FitResult start;
    start.family = null.family;
    start.method = null.method;
    start.status = null.status;
    start.parameters = null.parameters;
    start.coefficients = null.coefficients;
    start.boundary_coefficients = null.boundary_coefficients;
    start.theta = null.theta;
    start.theta_boundary = null.theta_boundary;
    return start;
}
// Records the adjusted GLM log2FC of the tested indicator (last design column).
inline void record_effect(PairResult &r, const FitResult &alternative, const FitOptions &options,
                          std::size_t column) {
    if (alternative.ok() &&
        (options.family == Family::poisson || options.family == Family::negative_binomial)) {
        auto effect = log2_fold_change(alternative, column, options.robust);
        r.log2fc = effect.estimate;
        r.log2fc_se = effect.standard_error;
        if (!effect.valid)
            r.status = "ok_effect_boundary";
    } else if (!alternative.ok())
        r.status = "ok_effect_fit_failed";
}
} // namespace detail
#if defined(_OPENMP) && _OPENMP >= 201511
namespace detail {
// What a pipelined pair needs for its null draws.
struct PairDraws {
    const BinaryScoreWorkspace *workspace;
    const Sampler *sampler;
    const ResampleOptions *options;
    double statistic;
    std::function<const CachedDraws *(std::size_t)> cache;
};
// Runs pairs response by response with every thread on a few pairs at a time.
// A small ring of active pairs is kept; threads take chunks of null draws from
// the active pairs, so the response's score records and the pairs' sampler
// arrays are shared in cache rather than each thread streaming its own. The
// thread that completes a stage applies the stage rule (NullStages) and, at the
// end, records the pair; idle threads prepare the next responses (null fits)
// and the next pairs (samplers). Draw k of a pair always uses its own key and
// is written to slot k, so results are those of any other schedule.
// A stage that may end early (ResampleOptions::sequential_screen, early_stop)
// starts one chunk at a time, up to its next check, since draws beyond its end
// would be wasted. Once a stage has gone 1,024 draws without ending, one more
// chunk may be in flight for each further 1,024 draws: a stage that then ends
// has generated at most about 3% more draws than it used, and a long stage (a
// test with a small p-value, or one drawing confirmation draws) is drawn by
// several threads. With such stages (`sequential_stages`) about one pair per
// thread is kept active, and threads without such a chunk of their own help
// the pairs in the other stages.
template <class PrepareResponse, class PreparePair, class Draws, class Finish>
inline void pair_pipeline(const std::vector<std::vector<std::size_t>> &batches, int threads,
                          bool parallel_draws, bool sequential_stages,
                          PrepareResponse prepare_response, PreparePair prepare_pair, Draws draws,
                          Finish finish) {
    using ContextPtr = decltype(prepare_response(std::size_t{}));
    using Context = typename ContextPtr::element_type;
    using JobPtr = decltype(prepare_pair(std::size_t{}, std::declval<Context &>()));
    std::vector<std::size_t> order, order_batch;
    for (std::size_t b = 0; b < batches.size(); ++b)
        for (auto t : batches[b]) {
            order.push_back(t);
            order_batch.push_back(b);
        }
    const std::size_t total = order.size();
    // Chunks of 32 draws; 8 active pairs at 32 threads (measured on the
    // Gasperini panel: fewer slots leave threads waiting for pairs to start,
    // more spread the threads over more samplers).
    const std::size_t chunk = NullStages::checkpoint;
    const auto team = static_cast<std::size_t>(threads);
    const std::size_t slot_count =
        std::max<std::size_t>(4, sequential_stages ? team + team / 4 : team / 4);
    const std::size_t lookahead = std::max<std::size_t>(2, threads);
    struct ContextState {
        std::atomic<int> state{0}; // 0 waiting, 1 preparing, 2 ready
        ContextPtr context;
        std::atomic<std::size_t> remaining{0};
    };
    std::vector<ContextState> contexts(batches.size());
    for (std::size_t b = 0; b < batches.size(); ++b)
        contexts[b].remaining = batches[b].size();
    struct Slot {
        std::mutex mutex;
        bool reserved = false, running = false;
        JobPtr job;
        std::unique_ptr<NullStages> stages;
        PairDraws draws{};
        std::size_t batch = 0, next = 0, end = 0, inflight = 0;
        // Sequential stages: draws written without gaps from the stage's start,
        // and chunks finished beyond them (first -> last).
        std::size_t written = 0;
        std::map<std::size_t, std::size_t> ahead;
    };
    std::vector<Slot> slots(slot_count);
    std::mutex activation;
    std::size_t next_pair = 0;
    std::atomic<std::size_t> frontier{0}, next_prepare{0}, done{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    // A response's state (its vectors and score records, tens of megabytes) is
    // allocated by the thread that prepares it and freed by the thread that ends
    // its last pair. glibc keeps freed blocks in the allocating thread's arena,
    // where other threads do not reuse them, so resident memory would grow by a
    // few megabytes per response for the whole run. Returning the free pages now
    // and then bounds it; doing so often would cost more than it saves, since
    // each return interrupts every thread.
    std::atomic<std::size_t> released{0};
    const std::size_t trim_every = 256;
    auto release = [&](std::size_t b) {
        if (--contexts[b].remaining == 0) {
            contexts[b].context.reset();
#ifdef __GLIBC__
            if (++released % trim_every == 0)
                malloc_trim(0);
#else
            (void)released;
            (void)trim_every;
#endif
        }
        ++done;
    };
    auto prefetch = [&](std::size_t limit) {
        std::size_t b = next_prepare.load();
        if (b >= batches.size() || b >= limit || !next_prepare.compare_exchange_strong(b, b + 1))
            return false;
        contexts[b].state = 1;
        contexts[b].context = prepare_response(b);
        contexts[b].state.store(2, std::memory_order_release);
        return true;
    };
    const auto division = vmath::division_mode();
#pragma omp parallel num_threads(threads)
    {
        const vmath::DivisionScope scope(division);
        DrawScratch scratch;
        const std::size_t offset = static_cast<std::size_t>(omp_get_thread_num()) % slot_count;
        try {
            while (!failed && done.load() < total) {
                // Keep the next responses' null fits ahead of the pairs.
                if (next_prepare.load() < frontier.load() + 2 && prefetch(frontier.load() + 2))
                    continue;
                // Draws of an active pair.
                bool worked = false;
                for (std::size_t k = 0; k < slot_count && !worked; ++k) {
                    auto &slot = slots[(offset + k) % slot_count];
                    std::size_t first, last;
                    Stream stream;
                    {
                        std::lock_guard<std::mutex> lock(slot.mutex);
                        if (!slot.running || slot.next >= slot.end)
                            continue;
                        const bool sequential = slot.stages->sequential();
                        if (sequential &&
                            slot.inflight > (slot.written - slot.stages->begin()) / 1024)
                            continue;
                        stream = slot.stages->stream();
                        first = slot.next;
                        last = sequential       ? NullStages::next_checkpoint(first)
                               : parallel_draws ? first + chunk
                                                : slot.end;
                        last = std::min(slot.end, last);
                        slot.next = last;
                        ++slot.inflight;
                    }
                    worked = true;
                    const auto &d = slot.draws;
                    draw_statistics(*d.workspace, *d.sampler, stream.seed, stream.test,
                                    d.cache ? d.cache(first) : nullptr, first, last - first,
                                    slot.stages->null() + first, scratch);
                    JobPtr finished;
                    ResampleResult result;
                    {
                        std::lock_guard<std::mutex> lock(slot.mutex);
                        --slot.inflight;
                        if (slot.stages->sequential()) {
                            // Chunks finish out of order; the stage is checked on the
                            // draws written without gaps, and once decided no more
                            // chunks are handed out.
                            if (first == slot.written) {
                                slot.written = last;
                                for (auto it = slot.ahead.find(slot.written);
                                     it != slot.ahead.end(); it = slot.ahead.find(slot.written)) {
                                    slot.written = it->second;
                                    slot.ahead.erase(it);
                                }
                            } else
                                slot.ahead[first] = last;
                            if (slot.stages->advance(slot.written))
                                slot.end = slot.next;
                        }
                        if (slot.next == slot.end && slot.inflight == 0) {
                            if (slot.stages->next()) {
                                result = std::move(slot.stages->result());
                                finished = std::move(slot.job);
                                slot.stages.reset();
                                slot.running = false;
                            } else {
                                slot.next = slot.written = slot.stages->begin();
                                slot.end = slot.stages->end();
                                slot.ahead.clear();
                            }
                        }
                    }
                    if (finished) {
                        finish(*finished, result);
                        const auto b = slot.batch;
                        finished.reset();
                        {
                            std::lock_guard<std::mutex> lock(slot.mutex);
                            slot.reserved = false;
                        }
                        release(b);
                    }
                }
                if (worked)
                    continue;
                // Start the next pair in a free slot.
                Slot *free = nullptr;
                for (std::size_t k = 0; k < slot_count && !free; ++k) {
                    auto &slot = slots[(offset + k) % slot_count];
                    std::lock_guard<std::mutex> lock(slot.mutex);
                    if (!slot.reserved) {
                        slot.reserved = true;
                        free = &slot;
                    }
                }
                if (free) {
                    std::size_t index = total;
                    {
                        std::lock_guard<std::mutex> lock(activation);
                        if (next_pair < total && contexts[order_batch[next_pair]].state.load(
                                                     std::memory_order_acquire) == 2) {
                            index = next_pair++;
                            frontier = order_batch[std::min(next_pair, total - 1)];
                        }
                    }
                    if (index < total) {
                        const auto b = order_batch[index];
                        auto job = prepare_pair(order[index], *contexts[b].context);
                        if (!job) {
                            {
                                std::lock_guard<std::mutex> lock(free->mutex);
                                free->reserved = false;
                            }
                            release(b);
                        } else {
                            std::lock_guard<std::mutex> lock(free->mutex);
                            free->draws = draws(*job);
                            free->job = std::move(job);
                            free->stages = std::make_unique<NullStages>(*free->draws.options,
                                                                        free->draws.statistic);
                            free->batch = b;
                            free->next = free->written = free->stages->begin();
                            free->end = free->stages->end();
                            free->ahead.clear();
                            free->inflight = 0;
                            free->running = true;
                        }
                        continue;
                    }
                    std::lock_guard<std::mutex> lock(free->mutex);
                    free->reserved = false;
                }
                // Prepare responses further ahead, or wait.
                if (!prefetch(frontier.load() + lookahead))
                    std::this_thread::yield();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure)
                failure = std::current_exception();
            failed = true;
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}
} // namespace detail
#endif
namespace detail {
struct SceptreTask {
    Pair pair;
    std::vector<std::size_t> guides;
    std::size_t pair_id;
    std::string label;
    std::size_t target_context = 0;
    // With identity stream keys: the key of the target, or of the guide.
    std::uint64_t target_key = 0;
};
// The cell universe and the task list of a sceptre() call: one task per pair,
// or per guide with singleton or Bonferroni integration. Tasks sharing a guide
// set share a target context, numbered in order of first appearance.
struct SceptreSetup {
    std::vector<std::size_t> cells;
    boost::dynamic_bitset<std::uint64_t> nt;
    std::vector<SceptreTask> tasks;
    std::size_t target_contexts = 0;
    // With identity stream keys: the key of each response column.
    std::vector<std::uint64_t> response_keys;
};
template <class Responses>
inline SceptreSetup sceptre_setup(const Matrix &design, const Responses &responses,
                                  const SparseCSC &assignments, const std::vector<Guide> &guides,
                                  const std::vector<Pair> &pairs, const CellQC &qc,
                                  const SceptreOptions &o) {
    SceptreSetup setup;
    design.validate();
    responses.validate();
    assignments.validate();
    require(design.rows == responses.rows && design.rows == assignments.rows &&
                assignments.cols == guides.size(),
            "perturbation input dimensions mismatch");
    require(!o.high_moi || !o.nt_controls, "high MOI requires complement controls");
    require(!o.reuse_draws || (o.crt && !o.nt_controls),
            "reuse_draws requires CRT resampling with complement controls");
    require(!o.shared_screen || (o.crt && !o.nt_controls),
            "shared_screen requires CRT resampling with complement controls");
    require(!(o.reuse_draws && o.shared_screen), "use reuse_draws or shared_screen, not both");
    require(!o.shared_screen || (o.shared_screen_p > 0 && o.shared_screen_p < 1),
            "shared_screen_p must be in (0, 1)");
    require(o.alpha > 0 && o.alpha < 1 && o.lower_quantile >= 0 && o.upper_quantile <= 1 &&
                o.lower_quantile < o.upper_quantile,
            "invalid workflow thresholds");
    require(o.model.family == Family::negative_binomial || o.model.family == Family::poisson ||
                o.model.family == Family::gaussian,
            "unsupported perturbation response family");
    require(o.null_model == FitMethod::direct || (o.null_model == FitMethod::sceptre_null &&
                                                  o.model.family == Family::negative_binomial &&
                                                  o.model.dispersion == Dispersion::ml &&
                                                  o.model.ridge == 0 && !o.model.nb_warm_start),
            "the SCEPTRE null model (default) requires unpenalized NB with ML dispersion and "
            "no NB warm start; use FitMethod::direct for other response models");
    const auto n = responses.rows;
    std::map<std::string, std::vector<std::size_t>> targets;
    std::set<std::string> guide_ids;
    const bool identity = o.stream_keys == StreamKeys::identity;
    require(o.response_ids.empty() || o.response_ids.size() == responses.cols,
            "response IDs must give one ID per response column");
    Vector guide_count(n), total(n), detected(n);
    // NT membership is built before parallel work and then only read.
    auto &nt = setup.nt;
    nt.resize(n);
    for (std::size_t j = 0; j < guides.size(); ++j) {
        require(!guides[j].id.empty() && !guides[j].target.empty() &&
                    guide_ids.insert(guides[j].id).second,
                "guide IDs must be unique and targets nonempty");
        targets[guides[j].target].push_back(j);
        for (auto k = assignments.column_ptr[j]; k < assignments.column_ptr[j + 1]; ++k) {
            require(assignments.values[k] == 0 || assignments.values[k] == 1,
                    "guide assignments must be binary calls");
            if (assignments.values[k]) {
                ++guide_count[assignments.row_index[k]];
                if (guides[j].non_targeting)
                    nt[assignments.row_index[k]] = 1;
            }
        }
    }
    for (std::size_t j = 0; j < responses.cols; ++j)
        responses.for_each(j, [&](std::size_t row, double value) {
            total[row] += value;
            detected[row] += value != 0;
        });
    if (!qc.total_counts.empty())
        total = qc.total_counts;
    if (!qc.detected_features.empty())
        detected = qc.detected_features;
    require(total.size() == n && detected.size() == n &&
                (qc.mitochondrial_fraction.empty() || qc.mitochondrial_fraction.size() == n) &&
                (qc.keep.empty() || qc.keep.size() == n),
            "QC dimensions mismatch");
    for (double x : total)
        require(std::isfinite(x) && x >= 0, "invalid QC total");
    for (double x : detected)
        require(std::isfinite(x) && x >= 0, "invalid QC detection count");
    for (double x : qc.mitochondrial_fraction)
        require(std::isfinite(x) && x >= 0 && x <= 1, "invalid mitochondrial fraction");
    double tl = quantile(total, o.lower_quantile), tu = quantile(total, o.upper_quantile),
           dl = quantile(detected, o.lower_quantile), du = quantile(detected, o.upper_quantile);
    auto &cells = setup.cells;
    for (std::size_t i = 0; i < n; ++i)
        if ((qc.keep.empty() || qc.keep[i]) &&
            (!o.assignment_qc || o.high_moi || guide_count[i] == 1) &&
            (!o.cell_qc ||
             (total[i] >= tl && total[i] <= tu && detected[i] >= dl && detected[i] <= du &&
              (qc.mitochondrial_fraction.empty() || qc.mitochondrial_fraction[i] <= o.max_mito))))
            cells.push_back(i);
    require(cells.size() > design.cols, "insufficient cells after QC");
    // Keys of the names, which must tell the names apart.
    auto distinct_keys = [](const auto &names, const char *message) {
        std::vector<std::uint64_t> keys;
        for (const auto &name : names)
            keys.push_back(name_key(name));
        auto sorted = keys;
        std::sort(sorted.begin(), sorted.end());
        require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), message);
        return keys;
    };
    std::map<std::string, std::size_t> target_number;
    std::vector<std::uint64_t> name_keys, guide_keys;
    if (identity) {
        setup.response_keys.resize(responses.cols);
        std::iota(setup.response_keys.begin(), setup.response_keys.end(), 0);
        if (!o.response_ids.empty())
            setup.response_keys = distinct_keys(
                o.response_ids, "response IDs must be distinct (and have distinct keys)");
        std::vector<std::string> names;
        for (const auto &entry : targets) {
            target_number.emplace(entry.first, names.size());
            names.push_back(entry.first);
        }
        name_keys = distinct_keys(names, "two target names have the same stream key");
        names.clear();
        for (const auto &guide : guides)
            names.push_back(guide.id);
        guide_keys = distinct_keys(names, "two guide IDs have the same stream key");
    }
    auto &tasks = setup.tasks;
    // With identity keys a pair is listed once within a kind: listed twice it
    // would be the same test, with the same draws, counted twice in its family.
    std::vector<std::uint64_t> listed;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const auto &pair = pairs[k];
        require(pair.response < responses.cols && targets.count(pair.target),
                "unknown pair response or target");
        const auto &indices = targets.at(pair.target);
        if (pair.kind == PairKind::negative_control)
            for (auto j : indices)
                require(
                    guides[j].non_targeting,
                    "negative-control target must contain only explicitly non-targeting guides");
        const std::size_t number = identity ? target_number.at(pair.target) : 0;
        if (identity) {
            require(pair.response <= UINT32_MAX && number < (std::size_t(1) << 30),
                    "too many responses or targets for identity stream keys");
            listed.push_back((static_cast<std::uint64_t>(pair.kind) << 62) |
                             (static_cast<std::uint64_t>(number) << 32) | pair.response);
        }
        if (o.integration == Integration::union_guides) {
            tasks.push_back({pair, indices, k, "union"});
            if (identity)
                tasks.back().target_key = name_keys[number];
        } else
            for (auto j : indices) {
                tasks.push_back({pair, {j}, k, guides[j].id});
                if (identity)
                    tasks.back().target_key = guide_keys[j];
            }
    }
    if (identity) {
        auto sorted = listed;
        std::sort(sorted.begin(), sorted.end());
        const auto twice = std::adjacent_find(sorted.begin(), sorted.end());
        if (twice != sorted.end()) {
            const auto k = static_cast<std::size_t>(
                std::find(listed.begin(), listed.end(), *twice) - listed.begin());
            throw std::invalid_argument("a pair is listed twice within a kind: response " +
                                        std::to_string(pairs[k].response) + ", target " +
                                        pairs[k].target +
                                        " (with identity stream keys a pair is listed once)");
        }
    }
    std::map<std::vector<std::size_t>, std::size_t> target_keys;
    for (auto &task : tasks)
        task.target_context = target_keys.emplace(task.guides, target_keys.size()).first->second;
    setup.target_contexts = target_keys.size();
    return setup;
}
inline FitResult sceptre_fit_null(const SceptreOptions &o, const Matrix &x, const Vector &y,
                                  const FitOptions &options) {
    return o.null_model == FitMethod::sceptre_null ? fit_sceptre_null(x, y, options)
                                                   : fit(x, y, options);
}
// The per-observation state of a regular null fit (fitted means, working weights
// and score residuals) recomputed from its coefficients and dispersion with the
// fit's own formulas. A fit that still holds these vectors, or is not regular, is
// returned as it is.
inline FitResult rebuild_null(const FitResult &fit, const Matrix &x, const Vector &y,
                              const FitOptions &options) {
    FitResult r = fit;
    if (!r.ok() || r.has_boundary() || !r.fitted.empty())
        return r;
    const Vector offset = options.offset.empty() ? Vector(x.rows, 0) : options.offset;
    const Vector a = options.weights.empty() ? Vector(y.size(), 1) : options.weights;
    // The SCEPTRE null keeps the Poisson pilot's means (same log link).
    const Family link = r.method == FitMethod::sceptre_null ? Family::poisson : r.family;
    r.fitted = means(x, r.coefficients, offset, link);
    working(y, r.fitted, a, r.family, r.theta, r.working_weights, r.score_residuals);
    return r;
}
inline bool same_bits(const Vector &a, const Vector &b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
} // namespace detail
// Which tasks a sceptre() call creates, for splitting it into shards: each
// task's response and target context (numbered in order of first appearance).
struct SceptreLayout {
    std::vector<std::size_t> task_response, task_target;
    std::size_t targets = 0, retained_cells = 0;
};
template <class Responses>
inline SceptreLayout sceptre_layout(const Matrix &design, const Responses &responses,
                                    const SparseCSC &assignments, const std::vector<Guide> &guides,
                                    const std::vector<Pair> &pairs, const CellQC &qc = {},
                                    const SceptreOptions &o = {}) {
    const auto setup = detail::sceptre_setup(design, responses, assignments, guides, pairs, qc, o);
    SceptreLayout layout;
    for (const auto &task : setup.tasks) {
        layout.task_response.push_back(task.pair.response);
        layout.task_target.push_back(task.target_context);
    }
    layout.targets = setup.target_contexts;
    layout.retained_cells = setup.cells.size();
    return layout;
}
// Response null fits for sceptre() shards (complement controls), by response
// index; entries are left empty for responses not listed and for responses
// none of whose pairs passes pair QC, which need no null. A fit is kept compact
// (without per-observation vectors) when rebuilding it gives exactly the fitted
// state, which is checked here; otherwise it is kept whole.
template <class Responses>
inline std::vector<FitResult>
sceptre_nulls(const Matrix &design, const Responses &responses, const SparseCSC &assignments,
              const std::vector<Guide> &guides, const std::vector<Pair> &pairs, const CellQC &qc,
              const SceptreOptions &o, const std::vector<std::size_t> &which) {
    require(!o.nt_controls, "shared response nulls need complement controls");
    const auto setup = detail::sceptre_setup(design, responses, assignments, guides, pairs, qc, o);
    const Matrix x = subset_rows(design, setup.cells);
    auto options = o.model;
    options.offset = subset(o.model.offset, setup.cells);
    options.weights = subset(o.model.weights, setup.cells);
    options.retain_observations = true;
    if (options.weights.empty())
        options.design = prepare_design(x, options.rank_tolerance);
    std::vector<FitResult> nulls(responses.cols);
    for (auto j : which)
        require(j < responses.cols, "unknown response");
    // The pairs of each response, and the treated cells of their targets.
    const std::set<std::size_t> wanted(which.begin(), which.end());
    std::map<std::size_t, std::vector<std::size_t>> response_tasks;
    std::vector<std::vector<std::size_t>> positions(setup.target_contexts);
    std::vector<bool> built(setup.target_contexts);
    for (std::size_t i = 0; i < setup.tasks.size(); ++i) {
        const auto &task = setup.tasks[i];
        if (!wanted.count(task.pair.response))
            continue;
        response_tasks[task.pair.response].push_back(i);
        if (built[task.target_context])
            continue;
        built[task.target_context] = true;
        boost::dynamic_bitset<std::uint64_t> treatment(responses.rows);
        for (auto j : task.guides)
            for (auto k = assignments.column_ptr[j]; k < assignments.column_ptr[j + 1]; ++k)
                if (assignments.values[k])
                    treatment[assignments.row_index[k]] = 1;
        for (std::size_t c = 0; c < setup.cells.size(); ++c)
            if (treatment[setup.cells[c]])
                positions[task.target_context].push_back(c);
    }
    parallel_for(which.size(), o.threads, [&](std::size_t k) {
        const auto y = subset(responses.column(which[k]), setup.cells);
        std::size_t nonzero = 0;
        for (double v : y)
            nonzero += v != 0;
        bool testable = false;
        const auto listed = response_tasks.find(which[k]);
        if (listed != response_tasks.end())
            for (auto i : listed->second) {
                std::size_t treated = 0;
                for (auto c : positions[setup.tasks[i].target_context])
                    treated += y[c] != 0;
                if (treated >= o.min_nonzero && nonzero - treated >= o.min_nonzero) {
                    testable = true;
                    break;
                }
            }
        if (!testable)
            return;
        auto fit = detail::sceptre_fit_null(o, x, y, options);
        auto compact = fit;
        compact.compact();
        const auto rebuilt = detail::rebuild_null(compact, x, y, options);
        const bool exact = !fit.ok() || fit.has_boundary() ||
                           (detail::same_bits(rebuilt.fitted, fit.fitted) &&
                            detail::same_bits(rebuilt.working_weights, fit.working_weights) &&
                            detail::same_bits(rebuilt.score_residuals, fit.score_residuals));
        nulls[which[k]] = exact ? std::move(compact) : std::move(fit);
    });
    return nulls;
}
template <class Responses>
inline SceptreResult sceptre(const Matrix &design, const Responses &responses,
                             const SparseCSC &assignments, const std::vector<Guide> &guides,
                             const std::vector<Pair> &pairs, const CellQC &qc = {},
                             const SceptreOptions &o = {}, const SceptreShard &shard = {}) {
    const vmath::DivisionScope division_scope(o.model.division);
    auto setup = detail::sceptre_setup(design, responses, assignments, guides, pairs, qc, o);
    const auto n = responses.rows;
    const auto &cells = setup.cells;
    const auto &nt = setup.nt;
    const auto &tasks = setup.tasks;
    auto fit_null = [&](const Matrix &x, const Vector &y, const FitOptions &options) {
        return detail::sceptre_fit_null(o, x, y, options);
    };
    require(!shard.results || shard.results->size() == tasks.size(),
            "shard results must cover every task");
    const auto shared_null = [&](std::size_t response) -> const FitResult * {
        return shard.nulls && response < shard.nulls->size() && (*shard.nulls)[response].parameters
                   ? &(*shard.nulls)[response]
                   : nullptr;
    };
    // Keys include the actual guide union, distinguishing singleton and pooled
    // tests. All contexts are scoped to this call's cell universe and options.
    struct TargetContext {
        std::once_flag membership_ready, propensity_ready;
        boost::dynamic_bitset<std::uint64_t> treatment;
        // Positions in `cells` of the treated cells, ascending.
        std::vector<std::size_t> positions;
        FitStatus propensity_status = FitStatus::numerical_failure;
        Vector propensity_coefficients;
        // With complement controls every pair of a target draws from the same
        // propensities, so its CRT sampler is built once and shared (within
        // SceptreOptions::sampler_cache_bytes).
        std::once_flag sampler_ready;
        std::shared_ptr<const Sampler> sampler;
        // With reuse_draws: the target's draws by stage, stored on first use
        // (state 0 none, 1 being stored, 2 stored, 3 over the budget).
        std::unique_ptr<std::atomic<int>[]> stage_state;
        std::vector<std::shared_ptr<const CachedDraws>> stage_draws;
    };
    std::atomic<std::size_t> sampler_cache_used{0}, draw_cache_used{0}, samplers_cached{0},
        samplers_rebuilt{0}, null_fits{0};
    const bool reuse_draws = o.reuse_draws && o.crt && !o.nt_controls;
    const bool shared_screen = o.shared_screen && o.crt && !o.nt_controls;
    const bool batch_scoring = o.batch_scoring && o.crt && !o.nt_controls && !shared_screen &&
                               !o.resampling.sequential_screen && o.resampling.early_stop == 0;
    // Shared screen streams are numbered apart from the pairs' own streams.
    constexpr std::uint64_t screen_streams = std::uint64_t(1) << 31;
    require(!shared_screen || o.stream_keys != StreamKeys::position ||
                (tasks.size() <= screen_streams && setup.target_contexts <= screen_streams),
            "too many tests for shared_screen streams");
    std::vector<std::unique_ptr<TargetContext>> target_contexts;
    for (std::size_t k = 0; k < setup.target_contexts; ++k) {
        target_contexts.push_back(std::make_unique<TargetContext>());
        auto &target = *target_contexts.back();
        target.stage_state = std::make_unique<std::atomic<int>[]>(o.resampling.budgets.size());
        target.stage_draws.resize(o.resampling.budgets.size());
    }
    std::map<std::size_t, std::vector<std::size_t>> response_tasks;
    if (shard.tasks) {
        for (std::size_t k = 0; k < shard.tasks->size(); ++k) {
            const auto i = (*shard.tasks)[k];
            require(i < tasks.size() && (k == 0 || (*shard.tasks)[k - 1] < i),
                    "shard tasks must be ascending task indices");
            response_tasks[tasks[i].pair.response].push_back(i);
        }
    } else if (!shard.results)
        for (std::size_t i = 0; i < tasks.size(); ++i)
            response_tasks[tasks[i].pair.response].push_back(i);
    std::vector<std::vector<std::size_t>> batches;
    for (auto &entry : response_tasks)
        batches.push_back(std::move(entry.second));
    std::vector<PairResult> raw =
        shard.results ? *shard.results : std::vector<PairResult>(tasks.size());
    // Deferred effect fits start from the O(p) null state saved by each task.
    // Only tested pairs save one: a pointer per task, not a fit.
    std::vector<std::unique_ptr<FitResult>> effect_starts(
        o.effect_fits == EffectFits::all ? 0 : tasks.size());
    Matrix common_design = subset_rows(design, cells);
    auto common_options = o.model;
    common_options.offset = subset(o.model.offset, cells);
    common_options.weights = subset(o.model.weights, cells);
    common_options.retain_observations = true;
    if (common_options.weights.empty())
        common_options.design = prepare_design(common_design, common_options.rank_tolerance);
    // The refine start (SceptreOptions::null_start): the NT cells' positions and
    // the design's nonzero entries by row.
    const bool refine = o.nt_controls && o.null_model == FitMethod::sceptre_null &&
                        o.null_start == NullStart::refine;
    std::vector<std::size_t> nt_positions;
    detail::mergeable::Rows design_rows;
    detail::mergeable::Cells refine_cells;
    if (refine) {
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (nt[cells[i]])
                nt_positions.push_back(i);
        design_rows = detail::mergeable::Rows(common_design);
        refine_cells = {&design_rows, &common_options.offset, &common_options.weights,
                        common_design.cols};
    }
    std::atomic<std::size_t> null_refined{0}, null_cold_refits{0};
    // The target's treated cells, built on first use.
    auto membership = [&](const detail::SceptreTask &task) -> TargetContext & {
        auto &target = *target_contexts[task.target_context];
        std::call_once(target.membership_ready, [&] {
            target.treatment.resize(n);
            for (auto j : task.guides)
                for (auto k = assignments.column_ptr[j]; k < assignments.column_ptr[j + 1]; ++k)
                    if (assignments.values[k])
                        target.treatment[assignments.row_index[k]] = 1;
            for (std::size_t i = 0; i < cells.size(); ++i)
                if (target.treatment[cells[i]])
                    target.positions.push_back(i);
        });
        return target;
    };
    // Per-response state: the response column and, with complement controls,
    // the shared null fit and score workspace.
    struct ResponseContext {
        Vector full_y, common_y;
        // Nonzero entries of common_y, and those in non-targeting cells.
        std::size_t nonzero = 0, nt_nonzero = 0;
        FitResult common_fit;
        std::once_flag common_score_ready;
        std::unique_ptr<ResamplingWorkspace> common_score;
        // The refine start's reference, with NT controls.
        std::unique_ptr<detail::mergeable::Reference> reference;
    };
    // The cells and the nonzero counts of a pair's two arms, from the target's
    // treated cells alone, so that pair QC is decided before anything is
    // gathered or fitted for the pair. With non-targeting controls the pair's
    // cells are the treated cells and the non-targeting ones, and a treated
    // cell counts as treated.
    struct PairCounts {
        std::size_t cells = 0, treatment = 0, control = 0;
    };
    std::size_t nt_cells = 0;
    for (auto i : cells)
        nt_cells += nt[i];
    auto pair_counts = [&](const detail::SceptreTask &task, const ResponseContext &context) {
        const auto &target = membership(task);
        PairCounts c;
        if (o.nt_controls) {
            std::size_t both = 0, both_nonzero = 0;
            for (auto position : target.positions) {
                const bool nonzero = context.common_y[position] != 0;
                c.treatment += nonzero;
                if (nt[cells[position]]) {
                    ++both;
                    both_nonzero += nonzero;
                }
            }
            c.cells = target.positions.size() + nt_cells - both;
            c.control = context.nt_nonzero - both_nonzero;
        } else {
            for (auto position : target.positions)
                c.treatment += context.common_y[position] != 0;
            c.cells = cells.size();
            c.control = context.nonzero - c.treatment;
        }
        return c;
    };
    auto passes_pair_qc = [&](const PairCounts &c) {
        return c.treatment >= o.min_nonzero && c.control >= o.min_nonzero;
    };
    auto prepare_response = [&](std::size_t batch) {
        auto context = std::make_unique<ResponseContext>();
        context->full_y = responses.column(tasks[batches[batch][0]].pair.response);
        context->common_y = subset(context->full_y, cells);
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (context->common_y[i] != 0) {
                ++context->nonzero;
                context->nt_nonzero += nt[cells[i]];
            }
        if (refine) {
            // The reference is fitted only when one of the response's pairs
            // passes pair QC.
            for (auto task_id : batches[batch])
                if (passes_pair_qc(pair_counts(tasks[task_id], *context))) {
                    context->reference =
                        std::make_unique<detail::mergeable::Reference>(detail::mergeable::reference(
                            fit_null(common_design, context->common_y, common_options),
                            refine_cells, nt_positions, context->common_y));
                    break;
                }
        }
        if (!o.nt_controls) {
            // The shared null is fitted only when one of the response's pairs
            // passes pair QC.
            bool testable = false;
            for (auto task_id : batches[batch])
                if (passes_pair_qc(pair_counts(tasks[task_id], *context))) {
                    testable = true;
                    break;
                }
            if (testable) {
                const auto *null = shared_null(tasks[batches[batch][0]].pair.response);
                context->common_fit =
                    null ? detail::rebuild_null(*null, common_design, context->common_y,
                                                common_options)
                         : fit_null(common_design, context->common_y, common_options);
                null_fits += !null;
            }
        }
        return context;
    };
    // A pair prepared up to its null draws: everything the draws and the final
    // record need. Pointers refer to the response context, which outlives it.
    struct PairJob {
        std::size_t task_id = 0;
        const Vector *y = nullptr;
        Vector pair_y, g; // g only where a dense indicator is needed
        // Treated positions in the pair's cells, ascending.
        const std::vector<std::size_t> *treated = nullptr;
        std::vector<std::size_t> pair_treated;
        const Matrix *x = nullptr;
        Matrix pair_design;
        FitOptions options;
        const FitResult *null = nullptr;
        FitResult pair_fit;
        std::shared_ptr<const Sampler> sampler;
        TargetContext *target = nullptr;
        std::unique_ptr<ResamplingWorkspace> pair_score;
        const ResamplingWorkspace *score = nullptr;
        // The score records the draws use: score->binary, or a refined pair's own.
        std::unique_ptr<BinaryScoreWorkspace> pair_binary;
        const BinaryScoreWorkspace *binary = nullptr;
        ResampleOptions resampling;
        double statistic = nan;
    };
    // A pair's null refined from the response's reference: its cells, labels,
    // null and score records into `job`. False when the pair is to be fitted cold.
    auto refine_null = [&](PairJob &job, const TargetContext &target,
                           const ResponseContext &context) {
        namespace m = detail::mergeable;
        const auto &reference = *context.reference;
        std::vector<std::size_t> block, pos;
        for (auto position : target.positions)
            if (!nt[cells[position]])
                block.push_back(position);
        const auto start =
            m::one_step(reference, m::pair_totals(reference, refine_cells, block, context.common_y),
                        common_options);
        if (!start.ok)
            return false;
        pos.reserve(nt_positions.size() + block.size());
        std::merge(nt_positions.begin(), nt_positions.end(), block.begin(), block.end(),
                   std::back_inserter(pos));
        const std::size_t n = pos.size();
        Vector a(n);
        job.pair_y.resize(n);
        job.g.assign(n, 0);
        for (std::size_t k = 0; k < n; ++k) {
            job.pair_y[k] = context.common_y[pos[k]];
            a[k] = refine_cells.a(pos[k]);
        }
        for (std::size_t k = 0, t = 0; t < target.positions.size(); ++k)
            if (pos[k] == target.positions[t]) {
                job.g[k] = 1;
                job.pair_treated.push_back(k);
                ++t;
            }
        auto refined = m::refine(refine_cells, pos, job.pair_y, a, start, common_options);
        if (!refined.ok)
            return false;
        if (refined.fit.ok()) {
            job.pair_binary = m::score_records(refine_cells, pos, job.pair_y, refined.fit.fitted,
                                               refined.fit.theta);
            if (!job.pair_binary)
                return false;
        }
        job.pair_fit = std::move(refined.fit);
        job.null = &job.pair_fit;
        job.y = &job.pair_y;
        job.treated = &job.pair_treated;
        return true;
    };
    // A target's propensity model, fitted once from the first pair's design and
    // labels; false when the fit failed.
    auto fit_propensity = [&](TargetContext &target, const Matrix &x, const FitOptions &options,
                              const std::function<Vector()> &labels) {
        std::call_once(target.propensity_ready, [&] {
            auto po = options;
            po.family = Family::binomial;
            po.offset.clear();
            po.weights.clear();
            po.start.clear();
            po.retain_observations = false;
            auto propensity = fit(x, labels(), po);
            target.propensity_status = propensity.status;
            target.propensity_coefficients = std::move(propensity.coefficients);
        });
        return target.propensity_status == FitStatus::converged ||
               target.propensity_status == FitStatus::converged_boundary;
    };
    // Retain only O(p) propensity parameters per target, not O(n) probabilities;
    // reconstruct the same fitted values when a sampler is built.
    auto build_sampler = [&](const TargetContext &target, const Matrix &x) {
        return std::make_shared<const Sampler>(crt_sampler(
            detail::means(x, target.propensity_coefficients, Vector(x.rows, 0), Family::binomial),
            o.crt_draw));
    };
    // With complement controls the sampler is shared by the target's pairs while
    // the cache budget allows (12 bytes per cell each); null beyond the budget.
    auto shared_sampler = [&](TargetContext &target, const Matrix &x) {
        std::call_once(target.sampler_ready, [&] {
            const std::size_t bytes = crt_sampler_bytes_per_observation * x.rows;
            if (bytes <= o.sampler_cache_bytes &&
                sampler_cache_used.fetch_add(bytes) + bytes <= o.sampler_cache_bytes) {
                target.sampler = build_sampler(target, x);
                ++samplers_cached;
            } else {
                if (bytes <= o.sampler_cache_bytes)
                    sampler_cache_used -= bytes;
                ++samplers_rebuilt;
            }
        });
        return target.sampler;
    };
    // Everything before the null draws; nullptr when the pair ends early (its
    // status is then recorded).
    auto prepare_pair = [&](std::size_t task_id,
                            ResponseContext &context) -> std::unique_ptr<PairJob> {
        const auto &task = tasks[task_id];
        auto &r = raw[task_id];
        r.response = task.pair.response;
        r.target = task.pair.target;
        r.kind = task.pair.kind;
        r.guide = task.label;
        auto &target = membership(task);
        const auto &treatment = target.treatment;
        // Pair QC first: nothing is gathered for a pair that fails it.
        const auto counts = pair_counts(task, context);
        r.cells = counts.cells;
        r.nonzero_treatment = counts.treatment;
        r.nonzero_control = counts.control;
        if (!passes_pair_qc(counts)) {
            r.status = "failed_pair_qc";
            return nullptr;
        }
        auto job = std::make_unique<PairJob>();
        job->task_id = task_id;
        // With NT controls the pair's null is refined from the response's
        // reference unless null_start is cold; a pair that cannot be refined is
        // fitted cold below.
        bool refined = false;
        if (refine && context.reference && context.reference->ok) {
            refined = refine_null(*job, target, context);
            if (refined)
                ++null_refined;
            else {
                ++null_cold_refits;
                job = std::make_unique<PairJob>();
                job->task_id = task_id;
            }
        }
        // A refined pair needs its design rows only for CRT propensities and
        // effect fits made with the draws.
        const bool gather = !refined || o.crt || o.effect_fits == EffectFits::all;
        std::vector<std::size_t> pair_cells;
        if (o.nt_controls && gather)
            for (auto i : cells)
                if (treatment[i] || nt[i])
                    pair_cells.push_back(i);
        const auto &use = o.nt_controls ? pair_cells : cells;
        if (!refined)
            job->y =
                o.nt_controls ? &(job->pair_y = subset(context.full_y, use)) : &context.common_y;
        const auto &y = *job->y;
        // The numerical APIs still consume exact double-valued 0/1 labels; with
        // complement controls the treated positions are the target's and the
        // dense indicator is built only where it is needed.
        auto indicator = [&] {
            Vector g(use.size());
            for (std::size_t i = 0; i < use.size(); ++i)
                g[i] = treatment[use[i]];
            return g;
        };
        if (refined) {
            // The labels and treated positions are the refined null's.
        } else if (o.nt_controls) {
            job->g = indicator();
            for (std::size_t i = 0; i < use.size(); ++i)
                if (job->g[i])
                    job->pair_treated.push_back(i);
            job->treated = &job->pair_treated;
        } else
            job->treated = &target.positions;
        // A refined pair without CRT or effect fits made with the draws needs no
        // design rows of its own; x is then the common design, unused.
        job->x = o.nt_controls && gather ? &(job->pair_design = subset_rows(design, use))
                                         : &common_design;
        const auto &x = *job->x;
        job->options = common_options;
        auto &options = job->options;
        if (o.nt_controls && gather) {
            options.offset = subset(o.model.offset, use);
            options.weights = subset(o.model.weights, use);
        }
        if (!refined)
            job->null =
                o.nt_controls ? &(job->pair_fit = fit_null(x, y, options)) : &context.common_fit;
        null_fits += o.nt_controls;
        const auto &null = *job->null;
        if (!null.ok() || null.has_boundary()) {
            r.status = std::string("null_") + status_name(null.status);
            return nullptr;
        }
        if (o.crt) {
            if (!fit_propensity(target, x, options,
                                [&] { return job->g.empty() ? indicator() : job->g; })) {
                r.status = std::string("propensity_") + status_name(target.propensity_status);
                return nullptr;
            }
            if (!o.nt_controls)
                job->sampler = shared_sampler(target, x);
            if (!job->sampler)
                job->sampler = build_sampler(target, x);
        } else {
            if (job->g.empty())
                job->g = indicator();
            job->sampler = std::make_shared<const Sampler>(permutation_sampler(job->g));
        }
        job->resampling = o.resampling;
        if (o.stream_keys == StreamKeys::position) {
            // Streams are keyed by the pair's place in the list, or by the
            // target's (in order of first appearance) when its draws are reused.
            job->resampling.test_id = reuse_draws ? task.target_context : task_id;
            if (shared_screen) {
                job->resampling.prescreen = o.shared_screen_p;
                job->resampling.prescreen_test_id = screen_streams + task.target_context;
            }
        } else {
            // Streams are keyed by the pair's response and target (or guide), or
            // by the target alone when its draws are reused; the shared screen's
            // by the target, in a domain of its own.
            const std::uint32_t domain = o.integration == Integration::union_guides ? 0 : 3;
            const Stream own =
                reuse_draws
                    ? keyed_stream(o.resampling.seed, domain + 2, 0, task.target_key)
                    : keyed_stream(o.resampling.seed, domain + 1,
                                   setup.response_keys[task.pair.response], task.target_key);
            job->resampling.seed = own.seed;
            job->resampling.test_id = own.test;
            if (shared_screen) {
                const Stream screen =
                    keyed_stream(o.resampling.seed, domain + 3, 0, task.target_key);
                job->resampling.prescreen = o.shared_screen_p;
                job->resampling.prescreen_seeded = true;
                job->resampling.prescreen_seed = screen.seed;
                job->resampling.prescreen_test_id = screen.test;
            }
        }
        job->resampling.retain_null = false;
        job->resampling.parallel_draws = o.parallel_draws;
        detail::check_budgets(job->resampling);
        job->target = &target;
        if (refined)
            ; // the refined null's own records
        else if (o.nt_controls)
            job->pair_score = std::make_unique<ResamplingWorkspace>(x, null);
        else
            std::call_once(context.common_score_ready, [&] {
                context.common_score = std::make_unique<ResamplingWorkspace>(x, null);
            });
        job->score = o.nt_controls ? job->pair_score.get() : context.common_score.get();
        job->binary = refined ? job->pair_binary.get() : &job->score->binary;
        job->statistic = detail::observed_statistic_of(*job->binary, *job->treated);
        if (!std::isfinite(job->statistic)) {
            r.status = "invalid_score";
            return nullptr;
        }
        return job;
    };
    // With reuse_draws (every stage) or shared_screen (the screen only): the
    // stored draws of the pair's target covering draw `first`, storing that
    // stage's draws on first use within the budget. A stage being stored by
    // another thread, or over the budget, is drawn afresh from the same stream,
    // so results do not depend on the cache. Stored draws are used only for
    // their own stream (draw_statistics checks it).
    // The stored draws of a target's stage, stored on first use within the
    // budget. With `wait` (batch scoring), a stage being stored by another thread
    // is waited for instead of being drawn afresh.
    auto stage_draws = [&](TargetContext &target, const Stream &stream,
                           const std::vector<std::size_t> &budgets, std::size_t stage,
                           const Sampler &sampler, std::size_t rows,
                           bool wait) -> const CachedDraws * {
        auto &state = target.stage_state[stage];
        if (state.load(std::memory_order_acquire) == 2)
            return target.stage_draws[stage].get();
        int idle = 0;
        if (!state.compare_exchange_strong(idle, 1)) {
            if (!wait)
                return nullptr;
            int now;
            while ((now = state.load(std::memory_order_acquire)) == 1)
                std::this_thread::yield();
            return now == 2 ? target.stage_draws[stage].get() : nullptr;
        }
        const std::size_t begin = stage == 0 ? 0 : budgets[stage - 1];
        auto draws = std::make_shared<const CachedDraws>(sampler, stream.seed, stream.test, begin,
                                                         budgets[stage] - begin, rows);
        // Batch scoring does not check each cell as it adds it: check once here.
        if (batch_scoring)
            detail::check_draws(*draws, rows);
        const auto bytes = draws->bytes();
        if (draw_cache_used.fetch_add(bytes) + bytes > o.draw_cache_bytes) {
            draw_cache_used -= bytes;
            state.store(3, std::memory_order_release);
            return nullptr;
        }
        target.stage_draws[stage] = std::move(draws);
        state.store(2, std::memory_order_release);
        return target.stage_draws[stage].get();
    };
    auto stored_draws = [&](const PairJob &job, std::size_t first,
                            bool wait = false) -> const CachedDraws * {
        const auto &budgets = job.resampling.budgets;
        std::size_t stage = 0;
        while (stage + 1 < budgets.size() && first >= budgets[stage])
            ++stage;
        if (shared_screen && stage > 0)
            return nullptr;
        const Stream stream =
            shared_screen ? Stream{job.resampling.prescreen_seeded ? job.resampling.prescreen_seed
                                                                   : job.resampling.seed,
                                   job.resampling.prescreen_test_id}
                          : Stream{job.resampling.seed, job.resampling.test_id};
        return stage_draws(*job.target, stream, budgets, stage, *job.sampler, job.x->rows, wait);
    };
    auto cache_of = [&](const PairJob &job) -> std::function<const CachedDraws *(std::size_t)> {
        if (!reuse_draws && !shared_screen)
            return {};
        return [&stored_draws, &job](std::size_t first) { return stored_draws(job, first); };
    };
    // Everything after the null draws.
    auto finish_pair = [&](PairJob &job, const ResampleResult &result) {
        auto &r = raw[job.task_id];
        if (!result.valid) {
            r.status = "invalid_score";
            return;
        }
        const auto &y = *job.y;
        const auto &null = *job.null;
        r.status = "ok";
        r.statistic = result.statistic;
        r.p_value = result.p_value;
        r.draws = result.draws;
        r.approximation = result.used_approximation;
        double observed = 0, expected = 0;
        for (auto i : *job.treated) {
            observed += y[i];
            expected += null.fitted[i];
        }
        r.sceptre_mean_ratio = expected > 0 ? observed / expected : nan;
        r.mean_ratio_log2 = r.sceptre_mean_ratio > 0 ? std::log2(r.sceptre_mean_ratio) : nan;
        if (o.effect_fits == EffectFits::all) {
            if (job.g.empty()) {
                job.g.assign(y.size(), 0);
                for (auto i : *job.treated)
                    job.g[i] = 1;
            }
            detail::record_effect(
                r, fit_effect_model(*job.x, y, job.g, null, job.options, o.warm_start_effects),
                job.options, job.x->cols);
        } else
            effect_starts[job.task_id] = std::make_unique<FitResult>(detail::effect_start(null));
    };
    // The null draws of one pair in a single caller, in batches.
    auto run_pair = [&](PairJob &job) {
        ResampleResult result;
        result.statistic = job.statistic;
        detail::staged_null(
            result, job.resampling,
            [&] { return detail::BinaryDrawBatch(*job.binary, *job.sampler, cache_of(job)); }, 256);
        finish_pair(job, result);
    };
    // Batch scoring: one response per thread, its pairs scored stage by stage
    // together (SceptreOptions::batch_scoring).
    const bool batched = batch_scoring && !batches.empty();
    if (batched) {
        // Every target's propensity model and sampler, and with reuse_draws its
        // first-stage draws, once and in parallel before the responses, so that
        // the responses' threads do not wait for them one at a time.
        std::vector<std::size_t> first_task(setup.target_contexts, tasks.size());
        for (const auto &batch : batches)
            for (auto i : batch)
                if (first_task[tasks[i].target_context] == tasks.size())
                    first_task[tasks[i].target_context] = i;
        std::vector<std::size_t> used;
        for (std::size_t k = 0; k < first_task.size(); ++k)
            if (first_task[k] < tasks.size())
                used.push_back(k);
        parallel_for(used.size(), o.threads, [&](std::size_t q) {
            const auto &task = tasks[first_task[used[q]]];
            auto &target = membership(task);
            const bool fitted = fit_propensity(target, common_design, common_options, [&] {
                Vector g(cells.size());
                for (std::size_t i = 0; i < cells.size(); ++i)
                    g[i] = target.treatment[cells[i]];
                return g;
            });
            if (!fitted)
                return;
            const auto sampler = shared_sampler(target, common_design);
            if (!reuse_draws || !sampler)
                return;
            // The target's stream, as prepare_pair keys it with reuse_draws.
            const std::uint32_t domain = o.integration == Integration::union_guides ? 0 : 3;
            const Stream stream =
                o.stream_keys == StreamKeys::position
                    ? Stream{o.resampling.seed, task.target_context}
                    : keyed_stream(o.resampling.seed, domain + 2, 0, task.target_key);
            stage_draws(target, stream, o.resampling.budgets, 0, *sampler, cells.size(), true);
        });
        std::atomic<std::size_t> released{0};
        parallel_for(batches.size(), o.threads, [&](std::size_t b) {
            thread_local detail::BatchScoreScratch scratch;
            thread_local detail::DrawScratch draw_scratch;
            thread_local std::vector<detail::ScoreDraw> list;
            // A round's draws: a stored stage's (whole), or up to `chunk` drawn
            // for the pair into `pool`.
            struct Piece {
                std::size_t job, first, count, offset;
                const CachedDraws *stored;
            };
            thread_local std::vector<Piece> pieces;
            thread_local std::vector<std::uint32_t> pool;
            thread_local std::vector<std::size_t> pool_offsets;
            constexpr std::size_t chunk = 8192;
            const std::size_t rows = cells.size();
            auto context = prepare_response(b);
            std::vector<std::unique_ptr<PairJob>> jobs;
            std::vector<std::unique_ptr<detail::NullStages>> stages;
            for (auto task_id : batches[b]) {
                auto job = prepare_pair(task_id, *context);
                if (!job)
                    continue;
                stages.push_back(
                    std::make_unique<detail::NullStages>(job->resampling, job->statistic));
                jobs.push_back(std::move(job));
            }
            std::vector<std::size_t> active(jobs.size()), next, next_draw(jobs.size()),
                stage_end(jobs.size());
            std::iota(active.begin(), active.end(), 0);
            for (std::size_t k = 0; k < jobs.size(); ++k) {
                next_draw[k] = stages[k]->begin();
                stage_end[k] = stages[k]->end();
            }
            // Each round takes every active pair's next draws and scores them all
            // in one pass; a pair whose stage is complete then applies its stage
            // rule. Draw k of a pair is its stream's draw k, written to slot k.
            while (!active.empty()) {
                pieces.clear();
                pool.clear();
                pool_offsets.assign(1, 0);
                for (auto k : active) {
                    const Stream stream = stages[k]->stream();
                    const std::size_t first = next_draw[k];
                    const CachedDraws *c =
                        reuse_draws ? stored_draws(*jobs[k], first, true) : nullptr;
                    if (c && c->seed == stream.seed && c->test == stream.test &&
                        first >= c->first && first < c->first + c->count) {
                        const std::size_t count =
                            std::min(stage_end[k], c->first + c->count) - first;
                        pieces.push_back({k, first, count, 0, c});
                        next_draw[k] += count;
                        continue;
                    }
                    // Drawn for the pair (its own stream, or a stored stage beyond
                    // the draw budget), in batches of 256 as CachedDraws draws them.
                    const std::size_t count = std::min(chunk, stage_end[k] - first);
                    pieces.push_back({k, first, count, pool_offsets.size() - 1, nullptr});
                    for (std::size_t done = 0; done < count; done += 256) {
                        const std::size_t size = std::min<std::size_t>(256, count - done);
                        jobs[k]->sampler->batch(stream.seed, stream.test, first + done, size,
                                                draw_scratch.selected, draw_scratch.sampler);
                        for (std::size_t d = 0; d < size; ++d) {
                            const auto &s = draw_scratch.selected[d];
                            for (std::size_t j = 0; j < s.size(); ++j) {
                                require(s[j] < rows && (j == 0 || s[j - 1] < s[j]),
                                        "sampler indices must be sorted, unique and in range");
                                pool.push_back(static_cast<std::uint32_t>(s[j]));
                            }
                            pool_offsets.push_back(pool.size());
                        }
                    }
                    next_draw[k] += count;
                }
                list.clear();
                for (const auto &piece : pieces) {
                    double *null = stages[piece.job]->null() + piece.first;
                    for (std::size_t d = 0; d < piece.count; ++d) {
                        std::size_t from, to;
                        const std::uint32_t *base;
                        if (piece.stored) {
                            const auto &c = *piece.stored;
                            from = c.offsets[piece.first - c.first + d];
                            to = c.offsets[piece.first - c.first + d + 1];
                            base = c.cells.data();
                        } else {
                            from = pool_offsets[piece.offset + d];
                            to = pool_offsets[piece.offset + d + 1];
                            base = pool.data();
                        }
                        list.push_back(
                            {base + from, static_cast<std::uint32_t>(to - from), null + d});
                    }
                }
                if (!list.empty())
                    detail::score_draws(*jobs[active.front()]->binary, list.data(), list.size(),
                                        scratch);
                next.clear();
                for (auto k : active) {
                    if (next_draw[k] < stage_end[k]) {
                        next.push_back(k);
                        continue;
                    }
                    if (stages[k]->next())
                        finish_pair(*jobs[k], stages[k]->result());
                    else {
                        next_draw[k] = stages[k]->begin();
                        stage_end[k] = stages[k]->end();
                        next.push_back(k);
                    }
                }
                active.swap(next);
            }
            // As in the pair pipeline, return freed pages now and then (glibc keeps
            // a thread's freed blocks in its arena).
            jobs.clear();
            stages.clear();
            context.reset();
#ifdef __GLIBC__
            if (++released % 256 == 0)
                malloc_trim(0);
#else
            (void)released;
#endif
        });
    }
    bool pipelined = false;
#if defined(_OPENMP) && _OPENMP >= 201511
    pipelined = !batched && o.parallel_pairs && o.threads > 1 && !batches.empty();
    if (pipelined)
        detail::pair_pipeline(
            batches, o.threads, o.parallel_draws,
            o.resampling.sequential_screen || o.resampling.early_stop > 0,
            [&](std::size_t batch) { return prepare_response(batch); },
            [&](std::size_t task_id, ResponseContext &context) {
                return prepare_pair(task_id, context);
            },
            [&](PairJob &job) -> const detail::PairDraws {
                return {job.binary, job.sampler.get(), &job.resampling, job.statistic,
                        cache_of(job)};
            },
            [&](PairJob &job, const ResampleResult &result) { finish_pair(job, result); });
#endif
    if (!pipelined && !batched)
        parallel_for(batches.size(), o.threads, [&](auto batch) {
            auto context = prepare_response(batch);
            task_for(batches[batch].size(), o.parallel_pairs, [&](std::size_t pair_index) {
                auto job = prepare_pair(batches[batch][pair_index], *context);
                if (job)
                    run_pair(*job);
            });
        });
    SceptreResult result;
    result.retained_cells = cells.size();
    result.samplers_cached = samplers_cached;
    result.samplers_rebuilt = samplers_rebuilt;
    result.null_fits = null_fits;
    result.null_start_used = refine ? NullStart::refine : NullStart::cold;
    result.null_refined = null_refined;
    result.null_cold_refits = null_cold_refits;
    if (shard.tasks) {
        for (auto i : *shard.tasks)
            result.tests.push_back(std::move(raw[i]));
        return result;
    }
    std::vector<std::size_t> row_task; // raw task behind each reported row
    if (o.integration == Integration::bonferroni) {
        // Tasks were created pair by pair, so each pair's guides are contiguous.
        for (std::size_t begin = 0, end = 0; begin < tasks.size(); begin = end) {
            end = begin;
            while (end < tasks.size() && tasks[end].pair_id == tasks[begin].pair_id)
                ++end;
            auto best = begin;
            for (auto i = begin; i < end; ++i)
                if (std::isfinite(raw[i].p_value) &&
                    (!std::isfinite(raw[best].p_value) || raw[i].p_value < raw[best].p_value))
                    best = i;
            auto r = raw[best];
            r.guide = "bonferroni:" + r.guide;
            if (std::isfinite(r.p_value))
                r.p_value = std::min(1.0, r.p_value * (end - begin));
            result.tests.push_back(r);
            row_task.push_back(best);
        }
    } else {
        result.tests = std::move(raw);
        row_task.resize(tasks.size());
        std::iota(row_task.begin(), row_task.end(), 0);
    }
    for (auto kind :
         {PairKind::discovery, PairKind::negative_control, PairKind::positive_control}) {
        const auto family = static_cast<std::size_t>(kind);
        Vector p;
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i < result.tests.size(); ++i)
            if (result.tests[i].kind == kind) {
                auto &r = result.tests[i];
                const bool passes = r.status != "failed_pair_qc" && r.status != "not_run";
                ++result.pairs_listed[family];
                result.pairs_passing_qc[family] += passes;
                if (o.family == AdjustmentFamily::tested && !passes) {
                    r.adjusted_p = nan;
                    continue;
                }
                indices.push_back(i);
                p.push_back(r.p_value);
            }
        require(o.family_size[family] == 0 || o.family_size[family] >= p.size(),
                "family_size is smaller than the pairs of its family");
        result.family_size[family] = o.family_size[family] ? o.family_size[family] : p.size();
        auto q = adjust_pvalues(p, o.bonferroni_adjustment, result.family_size[family]);
        for (std::size_t j = 0; j < q.size(); ++j) {
            result.tests[indices[j]].adjusted_p = q[j];
            if (q[j] <= o.alpha) {
                if (kind == PairKind::discovery)
                    ++result.discovery_rejections;
                else if (kind == PairKind::negative_control)
                    ++result.calibration_rejections;
                else
                    ++result.power_rejections;
            }
        }
        if (kind == PairKind::negative_control) {
            p.erase(std::remove_if(p.begin(), p.end(), [](double v) { return !std::isfinite(v); }),
                    p.end());
            std::sort(p.begin(), p.end());
            for (std::size_t j = 0; j < p.size(); ++j) {
                boost::math::beta_distribution<double> b(j + 1, p.size() - j);
                result.calibration_qq.push_back({static_cast<double>(j + 1) / (p.size() + 1), p[j],
                                                 boost::math::quantile(b, 0.025),
                                                 boost::math::quantile(b, 0.975)});
            }
        }
    }
    if (o.effect_fits == EffectFits::significant) {
        // Fit effects for rejected pairs only, from the same inputs and warm start as
        // EffectFits::all would use, so their estimates are identical.
        std::vector<std::size_t> selected;
        for (std::size_t k = 0; k < result.tests.size(); ++k)
            if (result.tests[k].status == "ok" && result.tests[k].adjusted_p <= o.alpha)
                selected.push_back(k);
        parallel_for(selected.size(), o.threads, [&](auto s) {
            auto &r = result.tests[selected[s]];
            const auto task_id = row_task[selected[s]];
            const auto &task = tasks[task_id];
            const auto &treatment = membership(task).treatment;
            const auto full_y = responses.column(task.pair.response);
            std::vector<std::size_t> pair_cells;
            if (o.nt_controls)
                for (auto i : cells)
                    if (treatment[i] || nt[i])
                        pair_cells.push_back(i);
            const auto &use = o.nt_controls ? pair_cells : cells;
            const auto y = subset(full_y, use);
            Vector g(use.size());
            for (std::size_t i = 0; i < use.size(); ++i)
                g[i] = treatment[use[i]];
            Matrix pair_design;
            const auto &x =
                o.nt_controls ? (pair_design = subset_rows(design, use)) : common_design;
            auto options = common_options;
            if (o.nt_controls) {
                options.offset = subset(o.model.offset, use);
                options.weights = subset(o.model.weights, use);
            }
            // Shards keep no per-task null state: take it from the response's
            // null (or refit the pair's), which gives the same O(p) start.
            FitResult start = effect_starts[task_id] ? *effect_starts[task_id] : FitResult();
            if (shard.results) {
                const auto *null = o.nt_controls ? nullptr : shared_null(task.pair.response);
                start = detail::effect_start(null ? *null : fit_null(x, y, options));
            }
            detail::record_effect(r,
                                  fit_effect_model(x, y, g, start, options, o.warm_start_effects),
                                  options, x.cols);
        });
    }
    return result;
}
// A request for every pair of some targets and some responses, without a list
// of the pairs.
struct SceptreGrid {
    // Targets tested against every response as discovery pairs, and targets of
    // non-targeting guides tested against every response as negative controls.
    std::vector<std::string> discovery_targets, negative_targets;
    // Response columns (distinct); empty: every response.
    std::vector<std::size_t> responses;
    // Further pairs, listed as for sceptre(): positive controls, say. A pair of
    // the grid may not be listed again in its kind.
    std::vector<Pair> pairs;
};
// The pairs of a target and kind in a grid request, by the outcome of pair QC:
// requested = passing + few_treatment + few_control. few_treatment: too few
// treated cells with a nonzero count; few_control: enough of those, too few
// control cells with one. With per-guide tests a pair passes when one of its
// guides does, and otherwise counts under few_treatment when no guide has
// enough treated cells.
struct PairTally {
    std::string target;
    PairKind kind = PairKind::discovery;
    std::size_t requested = 0, passing = 0, few_treatment = 0, few_control = 0;
};
struct SceptrePairQC {
    // The grid's pairs that pass pair QC: by response in the order requested,
    // then discovery targets and negative-control targets in the order given.
    std::vector<Pair> passing;
    // One entry per target and kind, in that order.
    std::vector<PairTally> tally;
    std::size_t retained_cells = 0;
};
// Pair QC of a grid request, by the rule sceptre() applies to a listed pair (the
// same cells, control group and min_nonzero), without a record or any other
// work per pair: one pass over each response's nonzero cells counts them for
// every target at once.
template <class Responses>
inline SceptrePairQC sceptre_pair_qc(const Matrix &design, const Responses &responses,
                                     const SparseCSC &assignments, const std::vector<Guide> &guides,
                                     const SceptreGrid &grid, const CellQC &qc = {},
                                     const SceptreOptions &o = {}) {
    const auto setup = detail::sceptre_setup(design, responses, assignments, guides, {}, qc, o);
    const auto &cells = setup.cells;
    const auto &nt = setup.nt;
    const bool per_guide = o.integration != Integration::union_guides;
    SceptrePairQC out;
    out.retained_cells = cells.size();
    std::vector<std::size_t> columns = grid.responses;
    if (columns.empty()) {
        columns.resize(responses.cols);
        std::iota(columns.begin(), columns.end(), 0);
    }
    {
        auto sorted = columns;
        std::sort(sorted.begin(), sorted.end());
        require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end() &&
                    (sorted.empty() || sorted.back() < responses.cols),
                "grid responses must be distinct response columns");
    }
    std::map<std::string, std::vector<std::size_t>> target_guides;
    for (std::size_t j = 0; j < guides.size(); ++j)
        target_guides[guides[j].target].push_back(j);
    // A unit is what one test treats: a target's guides together, or one guide.
    struct Requested {
        PairKind kind;
        std::vector<std::size_t> units;
    };
    std::vector<Requested> requested;
    std::vector<std::vector<std::size_t>> unit_guides;
    for (auto kind : {PairKind::discovery, PairKind::negative_control}) {
        const auto &names =
            kind == PairKind::discovery ? grid.discovery_targets : grid.negative_targets;
        std::set<std::string> seen;
        for (const auto &name : names) {
            require(target_guides.count(name) && seen.insert(name).second,
                    "grid targets must be distinct known targets");
            const auto &indices = target_guides.at(name);
            if (kind == PairKind::negative_control)
                for (auto j : indices)
                    require(guides[j].non_targeting, "negative-control target must contain only "
                                                     "explicitly non-targeting guides");
            Requested r{kind, {}};
            if (per_guide)
                for (auto j : indices) {
                    r.units.push_back(unit_guides.size());
                    unit_guides.push_back({j});
                }
            else {
                r.units.push_back(unit_guides.size());
                unit_guides.push_back(indices);
            }
            requested.push_back(std::move(r));
            out.tally.push_back({name, kind, 0, 0, 0, 0});
        }
    }
    // The units that treat each retained cell (by row), as offsets into `treats`.
    const auto none = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> position(responses.rows, none);
    for (std::size_t i = 0; i < cells.size(); ++i)
        position[cells[i]] = i;
    std::vector<std::vector<std::uint32_t>> cell_units(cells.size());
    require(unit_guides.size() <= UINT32_MAX, "too many targets in the grid");
    for (std::size_t u = 0; u < unit_guides.size(); ++u) {
        boost::dynamic_bitset<std::uint64_t> treated(cells.size());
        for (auto j : unit_guides[u])
            for (auto k = assignments.column_ptr[j]; k < assignments.column_ptr[j + 1]; ++k)
                if (assignments.values[k] && position[assignments.row_index[k]] != none)
                    treated[position[assignments.row_index[k]]] = 1;
        for (auto i = treated.find_first(); i != treated.npos; i = treated.find_next(i))
            cell_units[i].push_back(static_cast<std::uint32_t>(u));
    }
    std::vector<std::vector<Pair>> passing(columns.size());
    std::vector<std::vector<std::array<std::size_t, 3>>> tallies(
        columns.size()); // passing, few_treatment, few_control by requested target
    parallel_for(columns.size(), o.threads, [&](std::size_t c) {
        const auto column = columns[c];
        std::size_t nonzero = 0, nt_nonzero = 0;
        std::vector<std::uint32_t> treatment(unit_guides.size()), both(unit_guides.size());
        responses.for_each(column, [&](std::size_t row, double value) {
            const auto i = position[row];
            if (value == 0 || i == none)
                return;
            const bool control = nt[row];
            ++nonzero;
            nt_nonzero += control;
            for (auto u : cell_units[i]) {
                ++treatment[u];
                both[u] += control;
            }
        });
        auto &tally = tallies[c];
        tally.resize(requested.size());
        for (std::size_t r = 0; r < requested.size(); ++r) {
            bool pass = false, enough_treated = false;
            for (auto u : requested[r].units) {
                const std::size_t treated = treatment[u], control = o.nt_controls
                                                                        ? nt_nonzero - both[u]
                                                                        : nonzero - treated;
                enough_treated |= treated >= o.min_nonzero;
                pass |= treated >= o.min_nonzero && control >= o.min_nonzero;
            }
            ++tally[r][pass ? 0 : enough_treated ? 2 : 1];
            if (pass)
                passing[c].push_back({column, out.tally[r].target, requested[r].kind});
        }
    });
    for (std::size_t c = 0; c < columns.size(); ++c) {
        for (auto &pair : passing[c])
            out.passing.push_back(std::move(pair));
        for (std::size_t r = 0; r < requested.size(); ++r) {
            ++out.tally[r].requested;
            out.tally[r].passing += tallies[c][r][0];
            out.tally[r].few_treatment += tallies[c][r][1];
            out.tally[r].few_control += tallies[c][r][2];
        }
    }
    return out;
}
struct SceptreGridResult {
    // Records of the grid's pairs that pass pair QC, in the order of
    // SceptrePairQC::passing, then of the pairs listed in the request.
    SceptreResult result;
    std::vector<PairTally> tally;
};
namespace detail {
// The rows of each kind that listing every pair of a grid request would give:
// one per pair, or one per guide of its target with singleton integration.
// Checks that no listed pair repeats a pair of the grid in its kind.
inline std::array<std::size_t, 3> grid_rows(const std::vector<Guide> &guides,
                                            const SceptreGrid &grid,
                                            const std::vector<PairTally> &tally,
                                            const SceptreOptions &o) {
    std::map<std::string, std::size_t> target_guides;
    for (const auto &guide : guides)
        ++target_guides[guide.target];
    auto rows_of = [&](const std::string &target) {
        return o.integration == Integration::singleton ? target_guides.at(target) : 1;
    };
    const std::set<std::size_t> columns(grid.responses.begin(), grid.responses.end());
    const std::set<std::string> discovery(grid.discovery_targets.begin(),
                                          grid.discovery_targets.end()),
        negative(grid.negative_targets.begin(), grid.negative_targets.end());
    std::array<std::size_t, 3> rows{};
    for (const auto &pair : grid.pairs) {
        require(!((grid.responses.empty() || columns.count(pair.response)) &&
                  ((pair.kind == PairKind::discovery && discovery.count(pair.target)) ||
                   (pair.kind == PairKind::negative_control && negative.count(pair.target)))),
                "a listed pair repeats a pair of the grid");
        require(target_guides.count(pair.target), "unknown pair response or target");
        rows[static_cast<std::size_t>(pair.kind)] += rows_of(pair.target);
    }
    for (const auto &entry : tally)
        rows[static_cast<std::size_t>(entry.kind)] += entry.requested * rows_of(entry.target);
    return rows;
}
} // namespace detail
// sceptre() for a grid request. The records are those sceptre() gives for a list
// of the same pairs (streams are keyed by identity), and the adjustment families
// are those of listing every pair of the grid: with AdjustmentFamily::listed the
// pairs that fail pair QC are counted without being listed.
template <class Responses>
inline SceptreGridResult sceptre_grid(const Matrix &design, const Responses &responses,
                                      const SparseCSC &assignments,
                                      const std::vector<Guide> &guides, const SceptreGrid &grid,
                                      const CellQC &qc = {}, const SceptreOptions &o = {}) {
    const vmath::DivisionScope division_scope(o.model.division);
    require(o.stream_keys == StreamKeys::identity,
            "a grid request needs identity stream keys (a list has the positions)");
    auto checked = sceptre_pair_qc(design, responses, assignments, guides, grid, qc, o);
    const auto rows = detail::grid_rows(guides, grid, checked.tally, o);
    auto options = o;
    if (o.family == AdjustmentFamily::listed)
        for (std::size_t k = 0; k < 3; ++k)
            if (!options.family_size[k])
                options.family_size[k] = rows[k];
    auto pairs = std::move(checked.passing);
    pairs.insert(pairs.end(), grid.pairs.begin(), grid.pairs.end());
    SceptreGridResult out;
    out.result = sceptre(design, responses, assignments, guides, pairs, qc, options);
    out.result.pairs_listed = rows;
    out.tally = std::move(checked.tally);
    return out;
}
} // namespace rapidglm
