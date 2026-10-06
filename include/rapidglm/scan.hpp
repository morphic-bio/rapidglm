#pragma once
#include "workflows.hpp"

// Score scans: fit each response's null model once, then test many predictors
// against it with the efficient score statistic, calibrated asymptotically or by
// resampling (permutation, or a conditional randomization test for binary
// predictors), with multiple-testing adjustment per family of tests.
namespace rapidglm {
enum class Calibration { asymptotic, permutation, crt };
struct ScanPair {
    std::size_t response = 0, predictor = 0;
    std::string family = "all"; // adjusted together (BH or Bonferroni)
};
struct ScanOptions {
    FitOptions model;
    // direct: the configured GLM; sceptre_null: Poisson coefficients with a
    // conditional NB theta (resampling only; no ordinary score test).
    FitMethod null_model = FitMethod::direct;
    Calibration calibration = Calibration::permutation;
    ResampleOptions resampling;
    CrtDraw crt_draw = CrtDraw::thinned;
    // Permutation strata, one integer per observation (empty: one stratum).
    std::vector<std::size_t> strata;
    EffectFits effect_fits = EffectFits::significant;
    double alpha = 0.1;
    // Binary predictors: nonzero responses required among predictor 1s and 0s.
    std::size_t min_nonzero = 0;
    bool bonferroni = false, parallel_pairs = true, parallel_draws = true,
         warm_start_effects = true;
    int threads = 1;
};
struct ScanResult {
    std::size_t response = 0, predictor = 0;
    std::string family, status = "not_run";
    // statistic: efficient score z. estimate, standard_error: one-step score
    // estimate of the predictor's coefficient and its standard error (direct null
    // model). effect, effect_se: the coefficient and Wald SE from a full fit with
    // the predictor added (see effect_fits).
    double statistic = nan, p_value = nan, adjusted_p = 1, estimate = nan, standard_error = nan,
           effect = nan, effect_se = nan;
    std::size_t draws = 0;
    bool approximation = false;
};
namespace detail {
inline void record_scan_effect(ScanResult &r, const FitResult &alternative,
                               const FitOptions &options, std::size_t column) {
    const auto w = wald_test(alternative, column, Side::both, options.robust);
    if (w.valid) {
        r.effect = w.estimate;
        r.effect_se = w.standard_error;
    } else
        r.status = alternative.ok() ? "ok_effect_boundary" : "ok_effect_fit_failed";
}
} // namespace detail
// Tests every pair (all response x predictor pairs when `pairs` is empty). Each
// response's null model is fitted once and shared by its pairs. Resampling draw k
// of pair i uses CounterRng(seed, i, k), so results do not depend on threads.
inline std::vector<ScanResult> score_scan(const Matrix &design, const SparseCSC &responses,
                                          const SparseCSC &predictors,
                                          std::vector<ScanPair> pairs = {},
                                          const ScanOptions &o = {}) {
    design.validate();
    responses.validate();
    predictors.validate();
    const std::size_t n = design.rows;
    require(responses.rows == n && predictors.rows == n, "scan dimensions mismatch");
    require(o.alpha > 0 && o.alpha < 1, "alpha must be in (0, 1)");
    require(o.strata.empty() || o.strata.size() == n, "strata length mismatch");
    require(o.strata.empty() || o.calibration == Calibration::permutation,
            "strata apply to permutation calibration");
    require(o.calibration != Calibration::asymptotic || o.null_model == FitMethod::direct,
            "asymptotic calibration needs the direct null model");
    require(o.null_model == FitMethod::direct ||
                (o.model.family == Family::negative_binomial &&
                 o.model.dispersion == Dispersion::ml && o.model.ridge == 0 &&
                 !o.model.nb_warm_start && !o.model.cox_reid),
            "the SCEPTRE null model requires unpenalized NB with ML dispersion");
    if (pairs.empty())
        for (std::size_t r = 0; r < responses.cols; ++r)
            for (std::size_t j = 0; j < predictors.cols; ++j)
                pairs.push_back({r, j, "all"});
    require(pairs.size() <= UINT32_MAX, "too many pairs for the RNG test counter");
    for (const auto &pair : pairs)
        require(pair.response < responses.cols && pair.predictor < predictors.cols,
                "pair index out of range");
    // Binary predictors (all stored values 0 or 1) take the sparse score path.
    std::vector<unsigned char> binary(predictors.cols, 1);
    for (std::size_t j = 0; j < predictors.cols; ++j)
        for (auto k = predictors.column_ptr[j]; k < predictors.column_ptr[j + 1]; ++k)
            if (predictors.values[k] != 0 && predictors.values[k] != 1)
                binary[j] = 0;
    require(o.calibration != Calibration::crt ||
                std::all_of(pairs.begin(), pairs.end(),
                            [&](const ScanPair &pair) { return binary[pair.predictor] != 0; }),
            "CRT calibration requires binary predictors");
    // CRT propensities are fitted once per predictor; only O(p) coefficients are kept.
    struct Propensity {
        std::once_flag ready;
        FitStatus status = FitStatus::numerical_failure;
        Vector coefficients;
    };
    std::vector<std::unique_ptr<Propensity>> propensity(
        o.calibration == Calibration::crt ? predictors.cols : 0);
    for (auto &p : propensity)
        p = std::make_unique<Propensity>();
    std::map<std::size_t, std::vector<std::size_t>> by_response;
    for (std::size_t k = 0; k < pairs.size(); ++k)
        by_response[pairs[k].response].push_back(k);
    std::vector<std::vector<std::size_t>> batches;
    for (auto &entry : by_response)
        batches.push_back(std::move(entry.second));
    std::vector<ScanResult> out(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        out[k].response = pairs[k].response;
        out[k].predictor = pairs[k].predictor;
        out[k].family = pairs[k].family;
    }
    std::vector<FitResult> starts(o.effect_fits == EffectFits::significant ? pairs.size() : 0);
    auto options = o.model;
    options.retain_observations = true;
    const bool direct = o.null_model == FitMethod::direct;
    parallel_for(batches.size(), o.threads, [&](auto b) {
        const auto &batch = batches[b];
        const auto y = responses.column(pairs[batch.front()].response);
        const auto null = direct ? fit(design, y, options) : fit_sceptre_null(design, y, options);
        if (!null.ok() || null.has_boundary()) {
            for (auto k : batch)
                out[k].status = std::string("null_") + status_name(null.status);
            return;
        }
        const ResamplingWorkspace workspace(design, null);
        task_for(batch.size(), o.parallel_pairs, [&](std::size_t t) {
            const auto k = batch[t];
            auto &r = out[k];
            const auto j = pairs[k].predictor;
            const auto g = predictors.column(j);
            std::vector<std::size_t> selected;
            if (binary[j]) {
                for (auto q = predictors.column_ptr[j]; q < predictors.column_ptr[j + 1]; ++q)
                    if (predictors.values[q] == 1)
                        selected.push_back(predictors.row_index[q]);
                if (o.min_nonzero) {
                    std::size_t in = 0, total = 0;
                    for (auto i : selected)
                        in += y[i] != 0;
                    for (double v : y)
                        total += v != 0;
                    if (in < o.min_nonzero || total - in < o.min_nonzero) {
                        r.status = "failed_pair_qc";
                        return;
                    }
                }
            }
            // One-step score estimate u / I and its standard error sqrt(scale / I).
            TestResult score;
            if (binary[j]) {
                BinaryScoreScratch scratch;
                const auto [u, info] = workspace.binary.score(selected, scratch);
                if (info > 0) {
                    const double z = u / std::sqrt(info * workspace.score.scale);
                    score = {z, normal_tail(z, o.resampling.side), u / info,
                             std::sqrt(workspace.score.scale / info), true};
                }
            } else
                score = workspace.score.test(g, o.resampling.side);
            if (direct && score.valid) {
                r.estimate = score.estimate;
                r.standard_error = score.standard_error;
            }
            if (o.calibration == Calibration::asymptotic) {
                if (!score.valid) {
                    r.status = "invalid_score";
                    return;
                }
                r.statistic = score.statistic;
                r.p_value = score.p_value;
            } else {
                auto ro = o.resampling;
                ro.test_id = k;
                ro.retain_null = false;
                ro.parallel_draws = o.parallel_draws;
                ResampleResult result;
                if (o.calibration == Calibration::crt) {
                    auto &pr = *propensity[j];
                    std::call_once(pr.ready, [&] {
                        auto po = o.model;
                        po.family = Family::binomial;
                        po.offset.clear();
                        po.weights.clear();
                        po.start.clear();
                        po.cox_reid = false;
                        po.retain_observations = false;
                        auto fitted = fit(design, g, po);
                        pr.status = fitted.status;
                        pr.coefficients = std::move(fitted.coefficients);
                    });
                    if (pr.status != FitStatus::converged &&
                        pr.status != FitStatus::converged_boundary) {
                        r.status = std::string("propensity_") + status_name(pr.status);
                        return;
                    }
                    result = resample(workspace, g,
                                      crt_sampler(detail::means(design, pr.coefficients,
                                                                Vector(n, 0), Family::binomial),
                                                  o.crt_draw),
                                      ro);
                } else if (binary[j])
                    result = resample(workspace, g, permutation_sampler(g, o.strata), ro);
                else
                    result = resample_permutation(workspace.score, g, o.strata, ro);
                if (!result.valid) {
                    r.status = "invalid_score";
                    return;
                }
                r.statistic = result.statistic;
                r.p_value = result.p_value;
                r.draws = result.draws;
                r.approximation = result.used_approximation;
            }
            r.status = "ok";
            if (o.effect_fits == EffectFits::all)
                detail::record_scan_effect(
                    r, fit_effect_model(design, y, g, null, o.model, o.warm_start_effects), o.model,
                    design.cols);
            else if (o.effect_fits == EffectFits::significant)
                starts[k] = detail::effect_start(null);
        });
    });
    std::map<std::string, std::vector<std::size_t>> families;
    for (std::size_t k = 0; k < out.size(); ++k)
        families[out[k].family].push_back(k);
    for (const auto &entry : families) {
        Vector p;
        for (auto k : entry.second)
            p.push_back(out[k].p_value);
        const auto q = adjust_pvalues(p, o.bonferroni);
        for (std::size_t i = 0; i < q.size(); ++i)
            out[entry.second[i]].adjusted_p = q[i];
    }
    if (o.effect_fits == EffectFits::significant) {
        std::vector<std::size_t> selected;
        for (std::size_t k = 0; k < out.size(); ++k)
            if (out[k].status == "ok" && out[k].adjusted_p <= o.alpha)
                selected.push_back(k);
        parallel_for(selected.size(), o.threads, [&](auto s) {
            const auto k = selected[s];
            detail::record_scan_effect(out[k],
                                       fit_effect_model(design, responses.column(pairs[k].response),
                                                        predictors.column(pairs[k].predictor),
                                                        starts[k], o.model, o.warm_start_effects),
                                       o.model, design.cols);
        });
    }
    return out;
}
} // namespace rapidglm
