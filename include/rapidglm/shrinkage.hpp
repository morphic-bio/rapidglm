#pragma once
#include "model.hpp"
#include <boost/math/special_functions/gamma.hpp>

// Empirical Bayes building blocks for the quasi-likelihood test: a weighted
// running-median dispersion trend and the scaled F prior of Smyth (2004), with
// the abundance trend of Lund, Nettleton, McCarthy and Smyth (2012). The recipe
// follows glmGamPoi (Ahlmann-Eltze and Huber 2020, MIT licence); see
// DEPENDENCIES.md for provenance.
namespace rapidglm {
namespace detail {
// Lower weighted median: the smallest value whose cumulative weight reaches half
// of the total weight. Reorders the pairs (value, weight); expected linear time.
inline double weighted_median(std::vector<std::pair<double, double>> &values) {
    require(!values.empty(), "weighted median of no values");
    double total = 0;
    for (const auto &v : values)
        total += v.second;
    const double half = 0.5 * total;
    std::size_t lo = 0, hi = values.size();
    double below = 0;
    while (hi - lo > 1) {
        const std::size_t mid = lo + (hi - lo) / 2;
        std::nth_element(values.begin() + lo, values.begin() + mid, values.begin() + hi,
                         [](const auto &a, const auto &b) { return a.first < b.first; });
        double left = 0;
        for (std::size_t k = lo; k < mid; ++k)
            left += values[k].second;
        if (below + left >= half)
            hi = mid;
        else {
            below += left;
            lo = mid;
        }
    }
    return values[lo].first;
}
} // namespace detail
// Running weighted median of y against x. Points are ordered by x; the trend at
// each point is the weighted median of the `window` points nearest in rank, with
// Gaussian weights over +-3 standard deviations of the rank offset. Points in the
// first and last half-window take the value of the nearest complete window.
inline Vector running_median(const Vector &x, const Vector &y, std::size_t window) {
    require(x.size() == y.size(), "trend size mismatch");
    const std::size_t n = x.size();
    if (n == 0)
        return {};
    window = std::clamp<std::size_t>(window, 1, n);
    const std::size_t half = window / 2, span = 2 * half + 1;
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return x[a] < x[b]; });
    const auto gaussian = [](std::size_t points) {
        Vector w(points, 1);
        for (std::size_t k = 0; points > 1 && k < points; ++k) {
            const double t = -3 + 6.0 * k / (points - 1);
            w[k] = std::exp(-0.5 * t * t);
        }
        return w;
    };
    std::vector<std::pair<double, double>> buffer;
    if (n < span) {
        const auto w = gaussian(n);
        for (std::size_t k = 0; k < n; ++k)
            buffer.emplace_back(y[order[k]], w[k]);
        return Vector(n, detail::weighted_median(buffer));
    }
    const auto w = gaussian(span);
    Vector sorted(n);
    for (std::size_t c = half; c + half < n; ++c) {
        buffer.clear();
        for (std::size_t k = 0; k < span; ++k)
            buffer.emplace_back(y[order[c - half + k]], w[k]);
        sorted[c] = detail::weighted_median(buffer);
    }
    for (std::size_t c = 0; c < half; ++c) {
        sorted[c] = sorted[half];
        sorted[n - 1 - c] = sorted[n - 1 - half];
    }
    Vector out(n);
    for (std::size_t c = 0; c < n; ++c)
        out[order[c]] = sorted[c];
    return out;
}
// Natural cubic spline basis (intercept included, 4 functions) with boundary
// knots at lo and hi and interior knots at the thirds, in the truncated power
// form of Hastie, Tibshirani and Friedman (2009, eqs. 5.4-5.5) on [0, 1].
inline Matrix natural_spline_basis(const Vector &x, double lo, double hi) {
    require(hi > lo, "spline range must be positive");
    const double knots[4] = {0, 1.0 / 3, 2.0 / 3, 1};
    const auto cube = [](double v) { return v > 0 ? v * v * v : 0.0; };
    const auto d = [&](double t, int k) {
        return (cube(t - knots[k]) - cube(t - knots[3])) / (knots[3] - knots[k]);
    };
    Matrix b(x.size(), 4);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double t = (x[i] - lo) / (hi - lo);
        b(i, 0) = 1;
        b(i, 1) = t;
        b(i, 2) = d(t, 0) - d(t, 2);
        b(i, 3) = d(t, 1) - d(t, 2);
    }
    return b;
}
struct VariancePrior {
    Vector scale;     // prior scale s0^2 per feature; NaN where s2 was unusable
    Vector posterior; // (d0 s0^2 + d s2) / (d0 + d)
    double df = nan;  // prior degrees of freedom d0
    bool trend = false;
};
namespace detail {
struct ScaledF {
    Vector beta;
    double df = nan, loglik = -std::numeric_limits<double>::infinity();
};
// Maximum likelihood fit of s2_i ~ exp(B_i beta) F(d_i, d0). For fixed d0 the
// log likelihood is concave in beta and is maximized by damped Newton; d0 is
// profiled on a log grid over [1e-3, 1e7] and refined by golden section. The
// upper limit stands in for d0 = infinity (no dispersion heterogeneity).
inline ScaledF fit_scaled_f(const Vector &s2, const Vector &df, const Matrix &basis) {
    constexpr double min_df = 1e-3, max_df = 1e7;
    const std::size_t n = s2.size(), q = basis.cols;
    // lgamma((d+d0)/2) - lgamma(d0/2) - (d/2) log d0 depends on d only through its
    // few distinct values; terms free of beta and d0 are omitted.
    Vector distinct(df.begin(), df.end());
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    std::vector<std::size_t> which(n);
    for (std::size_t i = 0; i < n; ++i)
        which[i] = std::lower_bound(distinct.begin(), distinct.end(), df[i]) - distinct.begin();
    const auto loglik = [&](const Vector &beta, double d0) {
        Vector constant(distinct.size());
        for (std::size_t k = 0; k < distinct.size(); ++k)
            constant[k] = -std::log(boost::math::tgamma_delta_ratio(0.5 * d0, 0.5 * distinct[k])) -
                          0.5 * distinct[k] * std::log(d0);
        double total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double eta = 0;
            for (std::size_t j = 0; j < q; ++j)
                eta += basis(i, j) * beta[j];
            const double d = df[i], dz = d * s2[i] * std::exp(-eta);
            total +=
                constant[which[i]] + 0.5 * d * std::log(dz) - 0.5 * (d + d0) * std::log1p(dz / d0);
        }
        return total;
    };
    const auto maximize = [&](Vector beta, double d0) {
        double current = loglik(beta, d0);
        for (unsigned it = 0; it < 100 && std::isfinite(current); ++it) {
            Vector g(q), eta(n);
            Matrix h(q, q);
            Vector hw(n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < q; ++j)
                    eta[i] += basis(i, j) * beta[j];
                const double d = df[i], u = d * s2[i] * std::exp(-eta[i]) / d0;
                const double grad = -0.5 * d + 0.5 * (d + d0) * u / (1 + u);
                hw[i] = 0.5 * (d + d0) * u / ((1 + u) * (1 + u));
                for (std::size_t j = 0; j < q; ++j)
                    g[j] += basis(i, j) * grad;
            }
            Cholesky ch(crossprod(basis, hw));
            if (!ch.ok)
                break;
            const auto delta = ch.solve(g);
            double step = 1, next = current;
            Vector trial(q);
            bool accepted = false;
            for (unsigned halve = 0; halve < 40 && !accepted; ++halve) {
                for (std::size_t j = 0; j < q; ++j)
                    trial[j] = beta[j] + step * delta[j];
                next = loglik(trial, d0);
                accepted = std::isfinite(next) && next >= current - 1e-12 * (1 + std::abs(current));
                step *= 0.5;
            }
            if (!accepted)
                break;
            double change = 0;
            for (std::size_t j = 0; j < q; ++j)
                change = std::max(change, std::abs(trial[j] - beta[j]));
            beta = trial;
            current = next;
            if (change < 1e-10)
                break;
        }
        return std::make_pair(beta, current);
    };
    // Least-squares start on log s2.
    Vector start(q), logs(n);
    for (std::size_t i = 0; i < n; ++i)
        logs[i] = std::log(s2[i]);
    Cholesky ls(crossprod(basis, Vector(n, 1)));
    require(ls.ok, "singular variance prior basis");
    start = ls.solve(xtv(basis, logs));
    ScaledF best;
    const double lo = std::log(min_df), hi = std::log(max_df);
    constexpr std::size_t grid = 41;
    Vector values(grid);
    std::vector<Vector> betas(grid);
    Vector beta = start;
    std::size_t arg = 0;
    for (std::size_t k = 0; k < grid; ++k) {
        const double d0 = std::exp(lo + (hi - lo) * k / (grid - 1));
        auto [b, v] = maximize(beta, d0);
        values[k] = v;
        betas[k] = b;
        beta = b;
        if (v > values[arg])
            arg = k;
    }
    best.beta = betas[arg];
    best.df = std::exp(lo + (hi - lo) * arg / (grid - 1));
    best.loglik = values[arg];
    if (arg + 1 < grid) {
        // Golden-section refinement of log d0 between the neighbouring grid points.
        double a = lo + (hi - lo) * (arg ? arg - 1 : 0) / (grid - 1),
               b = lo + (hi - lo) * (arg + 1) / (grid - 1);
        const double r = 0.6180339887498948482;
        auto evaluate = [&](double logd0) { return maximize(best.beta, std::exp(logd0)); };
        double c = b - r * (b - a), d = a + r * (b - a);
        auto fc = evaluate(c), fd = evaluate(d);
        while (b - a > 1e-8) {
            if (fc.second >= fd.second) {
                b = d;
                d = c;
                fd = fc;
                c = b - r * (b - a);
                fc = evaluate(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + r * (b - a);
                fd = evaluate(d);
            }
        }
        const auto &f = fc.second >= fd.second ? fc : fd;
        if (f.second > best.loglik) {
            best.beta = f.first;
            best.df = std::exp(fc.second >= fd.second ? c : d);
            best.loglik = f.second;
        }
    }
    return best;
}
} // namespace detail
// Smyth's (2004) empirical Bayes prior for variances s2 with residual degrees of
// freedom df: s2 / s0^2 ~ F(df, d0), fitted by maximum likelihood. With a
// covariate (mean abundance) and at least 10 usable features with covariate
// above 1e-8, log s0^2 is a natural cubic spline in log covariate (Lund et al.
// 2012); features at or below 1e-8 keep the intercept-only scale. Unusable s2
// (missing, nonpositive or with nonpositive df) get NaN.
inline VariancePrior fit_variance_prior(const Vector &s2, const Vector &df,
                                        const Vector &covariate = {},
                                        bool abundance_trend = false) {
    require(df.size() == s2.size() && (covariate.empty() || covariate.size() == s2.size()),
            "variance prior size mismatch");
    require(!abundance_trend || !covariate.empty(), "abundance trend requires a covariate");
    VariancePrior prior;
    prior.scale.assign(s2.size(), nan);
    prior.posterior.assign(s2.size(), nan);
    std::vector<std::size_t> usable;
    for (std::size_t i = 0; i < s2.size(); ++i)
        if (std::isfinite(s2[i]) && s2[i] > 0 && std::isfinite(df[i]) && df[i] > 0)
            usable.push_back(i);
    require(usable.size() >= 2, "variance prior needs at least two usable variances");
    const auto fit_on = [&](const std::vector<std::size_t> &rows, const Matrix &basis) {
        Vector v, d;
        for (auto i : rows) {
            v.push_back(s2[i]);
            d.push_back(df[i]);
        }
        return detail::fit_scaled_f(v, d, basis);
    };
    auto flat = fit_on(usable, Matrix(usable.size(), 1, 1));
    prior.df = flat.df;
    for (auto i : usable)
        prior.scale[i] = std::exp(flat.beta[0]);
    if (abundance_trend) {
        std::vector<std::size_t> positive;
        for (auto i : usable)
            if (covariate[i] > 1e-8)
                positive.push_back(i);
        Vector logc;
        for (auto i : positive)
            logc.push_back(std::log(covariate[i]));
        if (positive.size() >= 10) {
            const auto [lo, hi] = std::minmax_element(logc.begin(), logc.end());
            if (*hi > *lo) {
                const auto basis = natural_spline_basis(logc, *lo, *hi);
                auto trended = fit_on(positive, basis);
                prior.df = trended.df;
                prior.trend = true;
                for (std::size_t k = 0; k < positive.size(); ++k) {
                    double eta = 0;
                    for (std::size_t j = 0; j < basis.cols; ++j)
                        eta += basis(k, j) * trended.beta[j];
                    prior.scale[positive[k]] = std::exp(eta);
                }
            }
        }
    }
    for (auto i : usable)
        prior.posterior[i] = (prior.df * prior.scale[i] + df[i] * s2[i]) / (prior.df + df[i]);
    return prior;
}
} // namespace rapidglm
