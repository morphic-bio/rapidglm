#pragma once
#include "hypothesis.hpp"
#include "resampling.hpp"
#include "shrinkage.hpp"
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/fisher_f.hpp>
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
struct SceptreOptions {
    bool high_moi = false, nt_controls = true, crt = false, cell_qc = true, assignment_qc = true,
         bonferroni_adjustment = false, parallel_draws = true, parallel_pairs = true,
         warm_start_effects = true;
    // SCEPTRE's documented response model: Poisson coefficients, conditional NB theta.
    FitMethod null_model = FitMethod::sceptre_null;
    Integration integration = Integration::union_guides;
    // significant: pairs with adjusted p <= alpha in their testing family.
    EffectFits effect_fits = EffectFits::significant;
    CrtDraw crt_draw = CrtDraw::thinned;
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
#pragma omp parallel num_threads(threads)
    {
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
                    std::uint64_t test;
                    {
                        std::lock_guard<std::mutex> lock(slot.mutex);
                        if (!slot.running || slot.next >= slot.end)
                            continue;
                        const bool sequential = slot.stages->sequential();
                        if (sequential &&
                            slot.inflight > (slot.written - slot.stages->begin()) / 1024)
                            continue;
                        test = slot.stages->test_id();
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
                    draw_statistics(*d.workspace, *d.sampler, d.options->seed, test,
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
};
// The cell universe and the task list of a sceptre() call: one task per pair,
// or per guide with singleton or Bonferroni integration. Tasks sharing a guide
// set share a target context, numbered in order of first appearance.
struct SceptreSetup {
    std::vector<std::size_t> cells;
    boost::dynamic_bitset<std::uint64_t> nt;
    std::vector<SceptreTask> tasks;
    std::size_t target_contexts = 0;
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
    auto &tasks = setup.tasks;
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
        if (o.integration == Integration::union_guides)
            tasks.push_back({pair, indices, k, "union"});
        else
            for (auto j : indices)
                tasks.push_back({pair, {j}, k, guides[j].id});
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
// index; entries for responses not listed are left empty. A fit is kept compact
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
    std::vector<FitResult> nulls(responses.cols);
    for (auto j : which)
        require(j < responses.cols, "unknown response");
    parallel_for(which.size(), o.threads, [&](std::size_t k) {
        const auto y = subset(responses.column(which[k]), setup.cells);
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
        samplers_rebuilt{0};
    const bool reuse_draws = o.reuse_draws && o.crt && !o.nt_controls;
    const bool shared_screen = o.shared_screen && o.crt && !o.nt_controls;
    // Shared screen streams are numbered apart from the pairs' own streams.
    constexpr std::uint64_t screen_streams = std::uint64_t(1) << 31;
    require(!shared_screen ||
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
    std::vector<FitResult> effect_starts(o.effect_fits == EffectFits::all ? 0 : tasks.size());
    Matrix common_design = subset_rows(design, cells);
    auto common_options = o.model;
    common_options.offset = subset(o.model.offset, cells);
    common_options.weights = subset(o.model.weights, cells);
    common_options.retain_observations = true;
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
        std::size_t nonzero = 0; // nonzero entries of common_y
        FitResult common_fit;
        std::once_flag common_score_ready;
        std::unique_ptr<ResamplingWorkspace> common_score;
    };
    auto prepare_response = [&](std::size_t batch) {
        auto context = std::make_unique<ResponseContext>();
        context->full_y = responses.column(tasks[batches[batch][0]].pair.response);
        context->common_y = subset(context->full_y, cells);
        for (double v : context->common_y)
            context->nonzero += v != 0;
        if (!o.nt_controls) {
            const auto *null = shared_null(tasks[batches[batch][0]].pair.response);
            context->common_fit =
                null ? detail::rebuild_null(*null, common_design, context->common_y, common_options)
                     : fit_null(common_design, context->common_y, common_options);
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
        ResampleOptions resampling;
        double statistic = nan;
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
        auto job = std::make_unique<PairJob>();
        job->task_id = task_id;
        std::vector<std::size_t> pair_cells;
        if (o.nt_controls)
            for (auto i : cells)
                if (treatment[i] || nt[i])
                    pair_cells.push_back(i);
        const auto &use = o.nt_controls ? pair_cells : cells;
        job->y = o.nt_controls ? &(job->pair_y = subset(context.full_y, use)) : &context.common_y;
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
        r.cells = use.size();
        if (o.nt_controls) {
            job->g = indicator();
            for (std::size_t i = 0; i < use.size(); ++i)
                if (job->g[i])
                    job->pair_treated.push_back(i);
            job->treated = &job->pair_treated;
            for (std::size_t i = 0; i < y.size(); ++i)
                if (y[i] != 0) {
                    if (job->g[i])
                        ++r.nonzero_treatment;
                    else
                        ++r.nonzero_control;
                }
        } else {
            job->treated = &target.positions;
            for (auto i : target.positions)
                r.nonzero_treatment += y[i] != 0;
            r.nonzero_control = context.nonzero - r.nonzero_treatment;
        }
        if (r.nonzero_treatment < o.min_nonzero || r.nonzero_control < o.min_nonzero) {
            r.status = "failed_pair_qc";
            return nullptr;
        }
        job->x = o.nt_controls ? &(job->pair_design = subset_rows(design, use)) : &common_design;
        const auto &x = *job->x;
        job->options = common_options;
        auto &options = job->options;
        if (o.nt_controls) {
            options.offset = subset(o.model.offset, use);
            options.weights = subset(o.model.weights, use);
        }
        job->null =
            o.nt_controls ? &(job->pair_fit = fit_null(x, y, options)) : &context.common_fit;
        const auto &null = *job->null;
        if (!null.ok() || null.has_boundary()) {
            r.status = std::string("null_") + status_name(null.status);
            return nullptr;
        }
        if (o.crt) {
            std::call_once(target.propensity_ready, [&] {
                auto po = options;
                po.family = Family::binomial;
                po.offset.clear();
                po.weights.clear();
                po.start.clear();
                po.retain_observations = false;
                auto propensity = fit(x, job->g.empty() ? indicator() : job->g, po);
                target.propensity_status = propensity.status;
                target.propensity_coefficients = std::move(propensity.coefficients);
            });
            if (target.propensity_status != FitStatus::converged &&
                target.propensity_status != FitStatus::converged_boundary) {
                r.status = std::string("propensity_") + status_name(target.propensity_status);
                return nullptr;
            }
            // Retain only O(p) propensity parameters per target, not O(n)
            // probabilities; reconstruct the same fitted values when a sampler is
            // built. With complement controls the sampler is shared by the target's
            // pairs while the cache budget allows (12 bytes per cell each).
            auto build = [&] {
                return std::make_shared<const Sampler>(
                    crt_sampler(detail::means(x, target.propensity_coefficients, Vector(x.rows, 0),
                                              Family::binomial),
                                o.crt_draw));
            };
            if (!o.nt_controls) {
                std::call_once(target.sampler_ready, [&] {
                    const std::size_t bytes = crt_sampler_bytes_per_observation * x.rows;
                    if (bytes <= o.sampler_cache_bytes &&
                        sampler_cache_used.fetch_add(bytes) + bytes <= o.sampler_cache_bytes) {
                        target.sampler = build();
                        ++samplers_cached;
                    } else {
                        if (bytes <= o.sampler_cache_bytes)
                            sampler_cache_used -= bytes;
                        ++samplers_rebuilt;
                    }
                });
                job->sampler = target.sampler;
            }
            if (!job->sampler)
                job->sampler = build();
        } else {
            if (job->g.empty())
                job->g = indicator();
            job->sampler = std::make_shared<const Sampler>(permutation_sampler(job->g));
        }
        job->resampling = o.resampling;
        // Streams are keyed by pair, or by target when its draws are reused.
        job->resampling.test_id = reuse_draws ? task.target_context : task_id;
        if (shared_screen) {
            job->resampling.prescreen = o.shared_screen_p;
            job->resampling.prescreen_test_id = screen_streams + task.target_context;
        }
        job->resampling.retain_null = false;
        job->resampling.parallel_draws = o.parallel_draws;
        detail::check_budgets(job->resampling);
        job->target = &target;
        if (o.nt_controls)
            job->pair_score = std::make_unique<ResamplingWorkspace>(x, null);
        else
            std::call_once(context.common_score_ready, [&] {
                context.common_score = std::make_unique<ResamplingWorkspace>(x, null);
            });
        job->score = o.nt_controls ? job->pair_score.get() : context.common_score.get();
        job->statistic = detail::observed_statistic_of(job->score->binary, *job->treated);
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
    auto stored_draws = [&](const PairJob &job, std::size_t first) -> const CachedDraws * {
        auto &target = *job.target;
        const auto &budgets = job.resampling.budgets;
        std::size_t stage = 0;
        while (stage + 1 < budgets.size() && first >= budgets[stage])
            ++stage;
        if (shared_screen && stage > 0)
            return nullptr;
        const auto stream =
            shared_screen ? job.resampling.prescreen_test_id : job.resampling.test_id;
        auto &state = target.stage_state[stage];
        if (state.load(std::memory_order_acquire) == 2)
            return target.stage_draws[stage].get();
        int idle = 0;
        if (!state.compare_exchange_strong(idle, 1))
            return nullptr;
        const std::size_t begin = stage == 0 ? 0 : budgets[stage - 1];
        auto draws = std::make_shared<const CachedDraws>(
            *job.sampler, job.resampling.seed, stream, begin, budgets[stage] - begin, job.x->rows);
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
            effect_starts[job.task_id] = detail::effect_start(null);
    };
    // The null draws of one pair in a single caller, in batches.
    auto run_pair = [&](PairJob &job) {
        ResampleResult result;
        result.statistic = job.statistic;
        detail::staged_null(
            result, job.resampling,
            [&] {
                return detail::BinaryDrawBatch(job.score->binary, *job.sampler, job.resampling.seed,
                                               cache_of(job));
            },
            256);
        finish_pair(job, result);
    };
    bool pipelined = false;
#if defined(_OPENMP) && _OPENMP >= 201511
    pipelined = o.parallel_pairs && o.threads > 1 && !batches.empty();
    if (pipelined)
        detail::pair_pipeline(
            batches, o.threads, o.parallel_draws,
            o.resampling.sequential_screen || o.resampling.early_stop > 0,
            [&](std::size_t batch) { return prepare_response(batch); },
            [&](std::size_t task_id, ResponseContext &context) {
                return prepare_pair(task_id, context);
            },
            [&](PairJob &job) -> const detail::PairDraws {
                return {&job.score->binary, job.sampler.get(), &job.resampling, job.statistic,
                        cache_of(job)};
            },
            [&](PairJob &job, const ResampleResult &result) { finish_pair(job, result); });
#endif
    if (!pipelined)
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
        Vector p;
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i < result.tests.size(); ++i)
            if (result.tests[i].kind == kind) {
                indices.push_back(i);
                p.push_back(result.tests[i].p_value);
            }
        auto q = adjust_pvalues(p, o.bonferroni_adjustment);
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
            FitResult start = effect_starts[task_id];
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
} // namespace rapidglm
