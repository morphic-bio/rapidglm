#pragma once
#include "model.hpp"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

namespace rapidglm {
enum class Side { left, right, both };
struct TestResult {
    double statistic = nan, p_value = nan, estimate = nan, standard_error = nan;
    bool valid = false;
};
inline double normal_tail(double z, Side side = Side::both) {
    boost::math::normal_distribution<double> n;
    double left = boost::math::cdf(n, z), right = boost::math::cdf(boost::math::complement(n, z));
    return side == Side::left    ? left
           : side == Side::right ? right
                                 : std::min(1.0, 2 * std::min(left, right));
}
inline TestResult wald_test(const FitResult &f, std::size_t coefficient, Side side = Side::both,
                            bool robust = false) {
    require(coefficient < f.parameters, "coefficient index out of range");
    if (!f.ok() || f.method != FitMethod::direct ||
        (!f.boundary_coefficients.empty() && f.boundary_coefficients[coefficient]))
        return {};
    if (robust)
        require(f.robust_covariance.rows == f.parameters,
                "robust covariance not requested during fit");
    double se = robust ? std::sqrt(f.robust_covariance(coefficient, coefficient))
                       : f.standard_errors[coefficient];
    if (!(se > 0) || !std::isfinite(se))
        return {};
    double z = f.coefficients[coefficient] / se, p = normal_tail(z, side);
    if (f.family == Family::gaussian && !robust) {
        boost::math::students_t_distribution<double> t(f.observations - f.parameters);
        double l = boost::math::cdf(t, z), r = boost::math::cdf(boost::math::complement(t, z));
        p = side == Side::left ? l : side == Side::right ? r : std::min(1.0, 2 * std::min(l, r));
    }
    return {z, p, f.coefficients[coefficient], se, true};
}
inline TestResult likelihood_ratio(const FitResult &null, const FitResult &alternative) {
    require(null.family == alternative.family &&
                null.response_signature == alternative.response_signature &&
                null.observations == alternative.observations,
            "LR models must have identical response, offset, weights and family");
    // Degrees of freedom count non-aliased columns, as R's anova does.
    const auto rank = [](const FitResult &f) { return f.rank ? f.rank : f.parameters; };
    require(rank(alternative) > rank(null), "LR requires a larger nested alternative");
    require(null.ridge == 0 && alternative.ridge == 0,
            "ordinary LR inference is undefined for ridge fits");
    // Ordinary chi-square LR calibration is not asserted for boundary models.
    if (!null.ok() || !alternative.ok() || null.has_boundary() || alternative.has_boundary() ||
        null.method != FitMethod::direct || alternative.method != FitMethod::direct)
        return {};
    double stat = 2 * (alternative.log_likelihood - null.log_likelihood);
    if (!std::isfinite(stat) || stat < -1e-7 * (1 + std::abs(null.log_likelihood)))
        return {};
    stat = std::max(0.0, stat);
    boost::math::chi_squared_distribution<double> chi(rank(alternative) - rank(null));
    return {stat, boost::math::cdf(boost::math::complement(chi, stat)), nan, nan, true};
}
struct ScoreWorkspace {
    Matrix design, inverse_information;
    Vector weights, residuals, nuisance_score;
    double scale = 1;
    bool project_nuisance = true;
    ScoreWorkspace(const Matrix &x, const FitResult &null)
        : design(x), weights(null.working_weights), residuals(null.score_residuals),
          scale(null.scale), project_nuisance(null.method == FitMethod::direct) {
        x.validate();
        require(null.ok() && !null.has_boundary() && null.ridge == 0,
                "score test requires a regular unpenalized null fit");
        require(x.rows == weights.size() && x.cols == null.parameters,
                "score workspace requires retained null observations and matching design");
        auto ch = detail::Cholesky(detail::crossprod(x, weights));
        require(ch.ok && scale > 0, "singular score information");
        inverse_information = ch.inverse();
        nuisance_score = detail::xtv(x, residuals);
    }

  private:
    TestResult evaluate(const Vector &g) const {
        require(g.size() == design.rows, "score covariate size mismatch");
        Vector c(design.cols);
        double u = 0, v = 0;
        for (std::size_t i = 0; i < g.size(); ++i) {
            require(std::isfinite(g[i]), "nonfinite score covariate");
            u += g[i] * residuals[i];
            v += g[i] * g[i] * weights[i];
            for (std::size_t j = 0; j < design.cols; ++j)
                c[j] += design(i, j) * weights[i] * g[i];
        }
        auto solved = detail::matvec(inverse_information, c);
        if (project_nuisance)
            u -= detail::dot(solved, nuisance_score);
        double info = v - detail::dot(c, solved);
        if (!(info > 1e-12 * std::max(1.0, v)))
            return {};
        double z = u / std::sqrt(info * scale);
        return {z, nan, u / info, std::sqrt(scale / info), true};
    }

  public:
    double statistic(const Vector &g) const {
        return evaluate(g).statistic;
    }
    TestResult test(const Vector &g, Side side = Side::both) const {
        // The two-stage working model supplies a resampling statistic, not an
        // ordinary Gaussian-tail score test at a joint NB maximum.
        if (!project_nuisance)
            return {};
        auto result = evaluate(g);
        if (result.valid)
            result.p_value = normal_tail(result.statistic, side);
        return result;
    }
};
// BH (or Bonferroni) adjustment within a family of `family_size` hypotheses, of
// which p holds the p-values at hand (0: the family is p itself). The others
// count as p = 1, as a missing p-value does, so the result is that of listing
// them.
inline Vector adjust_pvalues(const Vector &p, bool bonferroni = false,
                             std::size_t family_size = 0) {
    const auto n = p.size();
    require(family_size == 0 || family_size >= n, "the family is smaller than its p-values");
    const double m = static_cast<double>(family_size ? family_size : n);
    Vector adjusted(n, 1);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (double v : p)
        require(std::isnan(v) || (v >= 0 && v <= 1), "invalid p-value");
    auto safe = [&](std::size_t i) { return std::isnan(p[i]) ? 1.0 : p[i]; };
    if (bonferroni) {
        for (std::size_t i = 0; i < n; ++i)
            adjusted[i] = std::min(1.0, m * safe(i));
        return adjusted;
    }
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return safe(a) < safe(b); });
    double previous = 1;
    for (std::size_t k = n; k > 0; --k) {
        previous = std::min(previous, safe(order[k - 1]) * m / k);
        adjusted[order[k - 1]] = previous;
    }
    return adjusted;
}
inline TestResult log2_fold_change(const FitResult &fit, std::size_t column, bool robust = false) {
    require(fit.family == Family::poisson || fit.family == Family::negative_binomial,
            "fold change requires a log link");
    auto t = wald_test(fit, column, Side::both, robust);
    t.estimate /= std::log(2.0);
    t.standard_error /= std::log(2.0);
    return t;
}
} // namespace rapidglm
