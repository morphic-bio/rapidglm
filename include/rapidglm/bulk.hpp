#pragma once
#include "hypothesis.hpp"
#include "shrinkage.hpp"
#include "sparse.hpp"
#include "workflows.hpp"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/trigamma.hpp>

// Bulk RNA-seq differential expression following the procedure published for
// DESeq2 (Love, Huber and Anders 2014, Genome Biology 15:550) and its
// documentation: median-of-ratios size factors, Cox-Reid gene-wise dispersions,
// a parametric dispersion trend, a log-normal dispersion prior with MAP
// estimates, Wald or likelihood-ratio tests, Cook's distance outlier handling
// and independent filtering. Written from the paper and manual pages; details the
// documentation leaves open are RapidGLM choices (see METHODS.md).
namespace rapidglm {
enum class SizeFactorMethod { ratio, poscounts };
enum class DispersionTrend { parametric, local, mean };
// none: maximum likelihood fold changes only. normal: also the posterior mode under
// the zero-centred normal prior of the DESeq2 paper, as DESeq2's
// lfcShrink(type = "normal") computes it for a user-supplied (standard) model matrix.
enum class LfcShrinkage { none, normal };
struct BulkOptions {
    std::size_t coefficient = 0; // tested design column
    // Tested hypothesis; when empty, the coefficient above. Wald tests take one row
    // (a coefficient or contrast); LR tests take any, against the reduced design.
    Hypothesis hypothesis;
    bool lrt = false; // likelihood ratio test against the reduced design
    SizeFactorMethod size_factor_method = SizeFactorMethod::ratio;
    Vector size_factors; // supplied size factors (no estimation)
    DispersionTrend trend = DispersionTrend::parametric;
    double min_dispersion = 1e-8, max_dispersion = 0; // 0: max(10, samples)
    double min_mean = 0.5; // floor on fitted counts in dispersion estimation
    // DESeq2 optimizes each gene-wise dispersion by a line search in log dispersion
    // from a rough moment estimate and stops when the gain is below 1e-6. From a
    // start at min_dispersion the adjusted likelihood is flat and the search does
    // not move, so such genes keep min_dispersion (and leave the trend fit). With
    // false, every gene gets the maximizer of the adjusted profile likelihood.
    bool keep_minimum_start = true;
    double outlier_sd = 2; // dispersion outliers keep their gene-wise estimate
    bool cooks_filter = true;
    // Cook's distances with the robust dispersion of DESeq2 as implemented in
    // PyDESeq2 (MIT; Muzellec et al. 2023): trimmed within-cell variances with
    // size-dependent trims and scales, their maximum over cells, a 0.04 floor, and
    // no flag when three or more samples have larger counts than the sample with
    // the largest distance. Off: RapidGLM's pooled-MAD estimator (see METHODS.md).
    bool legacy_cooks = false;
    double cooks_cutoff = 0; // 0: the 0.99 quantile of F(p, m - p)
    // Samples whose design row has at least this many replicates get outlier
    // counts replaced by the trimmed mean and refitted (0 disables).
    std::size_t min_replicates_for_replace = 7;
    bool independent_filtering = true;
    double alpha = 0.1;
    LfcShrinkage lfc_shrinkage = LfcShrinkage::none;
    // Prior variances of the coefficients on the log2 scale, one per design column
    // (empty: estimated from the maximum likelihood fold changes).
    Vector lfc_prior_variance;
    std::uint64_t seed = 1; // prior-variance simulation (three or fewer residual df)
    int threads = 1;
};
struct BulkGene {
    std::string status = "not_run";
    double base_mean = nan, dispersion_gene = nan, dispersion_trend = nan, dispersion_map = nan,
           dispersion = nan, log2_fold_change = nan, lfc_se = nan, statistic = nan, p_value = nan,
           adjusted_p = nan, max_cooks = nan;
    // log2_fold_change and lfc_se are missing for hypotheses with several rows.
    // Posterior mode of the tested log2 fold change and its standard deviation
    // (LfcShrinkage::normal only).
    double shrunken_lfc = nan, shrunken_lfc_se = nan;
    bool dispersion_outlier = false, replaced = false;
};
struct BulkResult {
    std::vector<BulkGene> genes;
    Vector size_factors;
    std::string trend = "parametric";
    double asymptotic_dispersion = nan, extra_poisson = nan; // parametric trend
    double log_residual_variance = nan, prior_variance = nan, cooks_cutoff = nan;
    double filter_threshold = nan, filter_theta = nan;
    std::vector<std::pair<double, std::size_t>> filter_rejections;
    Vector filter_fit;
    Vector lfc_prior_variance; // log2 scale, per design column (LfcShrinkage::normal)
    std::size_t replaced_genes = 0;
};
namespace detail {
// Median as R computes it: the mean of the two middle values for even counts.
inline double median(Vector v) {
    require(!v.empty(), "median of no values");
    const auto mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double upper = v[mid];
    if (v.size() % 2)
        return upper;
    return 0.5 * (upper + *std::max_element(v.begin(), v.begin() + mid));
}
inline double mad(const Vector &v) {
    const double m = median(v);
    Vector d(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        d[i] = std::abs(v[i] - m);
    return 1.4826 * median(d);
}
// R's default (type 7) quantile.
inline double quantile7(Vector v, double p) {
    std::sort(v.begin(), v.end());
    const double h = p * (v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const auto hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - lo) * (v[hi] - v[lo]);
}
// Cleveland's (1979) LOWESS: local linear fits with tricube weights over the
// nearest ceil(f n) points, then `iterations` robustness passes with bisquare
// weights of residuals scaled by six times their median absolute value (none
// when that scale is below 1e-10 of the largest |y|).
inline Vector lowess(const Vector &x, const Vector &y, double f = 2.0 / 3, int iterations = 3) {
    const std::size_t n = x.size();
    Vector fit(n), robustness(n, 1.0);
    if (n < 2)
        return y;
    const auto span = std::clamp<std::size_t>(static_cast<std::size_t>(std::ceil(f * n)), 2, n);
    for (int pass = 0; pass <= iterations; ++pass) {
        for (std::size_t i = 0; i < n; ++i) {
            // Nearest `span` points to x[i] (x is sorted).
            std::size_t lo = 0, hi = span - 1;
            while (hi + 1 < n && x[i] - x[lo] > x[hi + 1] - x[i]) {
                ++lo;
                ++hi;
            }
            const double h = std::max(x[i] - x[lo], x[hi] - x[i]);
            double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (std::size_t k = lo; k <= hi; ++k) {
                double w = robustness[k];
                if (h > 0) {
                    const double r = std::abs(x[k] - x[i]) / h;
                    w *= r < 1 ? std::pow(1 - r * r * r, 3) : 0;
                }
                sw += w;
                sx += w * x[k];
                sy += w * y[k];
                sxx += w * x[k] * x[k];
                sxy += w * x[k] * y[k];
            }
            if (!(sw > 0)) {
                fit[i] = y[i];
                continue;
            }
            const double mx = sx / sw, my = sy / sw, vx = sxx / sw - mx * mx;
            fit[i] = vx > 1e-12 * (1 + mx * mx) ? my + (sxy / sw - mx * my) / vx * (x[i] - mx) : my;
        }
        if (pass == iterations)
            break;
        Vector residual(n);
        for (std::size_t i = 0; i < n; ++i)
            residual[i] = std::abs(y[i] - fit[i]);
        // Residuals at rounding level leave nothing to downweight.
        double scale = 0;
        for (double v : y)
            scale = std::max(scale, std::abs(v));
        const double s = 6 * median(residual);
        if (!(s > 1e-10 * scale))
            break;
        for (std::size_t i = 0; i < n; ++i) {
            const double u = residual[i] / s;
            robustness[i] = u < 1 ? (1 - u * u) * (1 - u * u) : 0;
        }
    }
    return fit;
}
// Gamma-family GLM with identity link, y ~ a0 + a1 x, by IRLS (weights 1/mean^2).
inline bool gamma_identity_fit(const Vector &y, const Vector &x, double &a0, double &a1) {
    const auto fitted = [&](double b0, double b1, std::size_t i) { return b0 + b1 * x[i]; };
    for (int it = 0; it < 200; ++it) {
        double s00 = 0, s01 = 0, s11 = 0, t0 = 0, t1 = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double m = fitted(a0, a1, i);
            if (!(m > 0))
                return false;
            const double w = 1 / (m * m);
            s00 += w;
            s01 += w * x[i];
            s11 += w * x[i] * x[i];
            t0 += w * y[i];
            t1 += w * x[i] * y[i];
        }
        const double det = s00 * s11 - s01 * s01;
        if (!(std::abs(det) > 0))
            return false;
        double b0 = (s11 * t0 - s01 * t1) / det, b1 = (s00 * t1 - s01 * t0) / det;
        // Halve the step until every fitted mean stays positive.
        for (int h = 0; h < 50; ++h) {
            bool positive = true;
            for (std::size_t i = 0; i < y.size() && positive; ++i)
                positive = fitted(b0, b1, i) > 0;
            if (positive)
                break;
            b0 = 0.5 * (b0 + a0);
            b1 = 0.5 * (b1 + a1);
        }
        const double change =
            std::abs(b0 - a0) / (std::abs(a0) + 1e-12) + std::abs(b1 - a1) / (std::abs(a1) + 1e-12);
        a0 = b0;
        a1 = b1;
        if (change < 1e-12)
            return true;
    }
    return true;
}
// Kullback-Leibler divergence of binned densities (common bins, small floor).
inline double binned_kl(const std::vector<double> &p, const std::vector<double> &q) {
    double kl = 0;
    for (std::size_t k = 0; k < p.size(); ++k)
        if (p[k] > 0)
            kl += p[k] * std::log(p[k] / std::max(q[k], 1e-12));
    return kl;
}
// Prior variance for three or fewer residual df (DESeq2 paper): simulate
// log(chi2_df / df) + N(0, s2) over a grid of s2 in [0, 1] and choose the value
// minimizing the KL divergence from the observed log-residual density. Densities
// are histograms on 100 common bins over the observed range (a RapidGLM choice).
inline double simulated_prior_variance(const Vector &residuals, double df, std::uint64_t seed,
                                       int threads = 1) {
    constexpr std::size_t draws = 20000, bins = 100, grid = 1000;
    const auto [lo_it, hi_it] = std::minmax_element(residuals.begin(), residuals.end());
    const double lo = *lo_it, hi = *hi_it, width = (hi - lo) / bins;
    if (!(width > 0))
        return 0;
    // Bin shares over the observed range; values at the upper edge join the last bin.
    const auto histogram = [&](std::size_t count, auto value, std::vector<double> &h) {
        h.assign(bins, 0);
        for (std::size_t k = 0; k < count; ++k) {
            // Truncation equals the floor for the nonnegative positions kept.
            const double t = (value(k) - lo) / width;
            if (t >= 0 && t < bins + 1)
                ++h[std::min<std::size_t>(static_cast<std::size_t>(t), bins - 1)];
        }
        for (auto &x : h)
            x /= std::max<std::size_t>(count, 1);
    };
    std::vector<double> observed;
    histogram(residuals.size(), [&](std::size_t k) { return residuals[k]; }, observed);
    // Common random numbers: the same chi-square and normal draws for every s2.
    CounterRng rng(seed, 0, 0);
    boost::math::gamma_distribution<double> chi(df / 2, 2);
    Vector logchi(draws), z(draws);
    for (std::size_t k = 0; k < draws; ++k) {
        const double u = std::clamp(rng.uniform(), 1e-300, 1 - 1e-16);
        logchi[k] = std::log(boost::math::quantile(chi, u) / df);
        const double u1 = std::max(rng.uniform(), 1e-300), u2 = rng.uniform();
        z[k] = std::sqrt(-2 * std::log(u1)) * std::cos(2 * 3.14159265358979323846 * u2);
    }
    // Grid points are independent; the first minimum in grid order is chosen.
    Vector kl(grid);
    parallel_for(grid, threads, [&](std::size_t g) {
        const double s = std::sqrt(static_cast<double>(g) / (grid - 1));
        std::vector<double> h;
        histogram(draws, [&](std::size_t k) { return logchi[k] + s * z[k]; }, h);
        kl[g] = binned_kl(observed, h);
    });
    double best = 0, best_kl = std::numeric_limits<double>::infinity();
    for (std::size_t g = 0; g < grid; ++g)
        if (kl[g] < best_kl) {
            best_kl = kl[g];
            best = static_cast<double>(g) / (grid - 1);
        }
    return best;
}
// Quantile p of x with weights w: the type-7 quantile of a sample in which each
// value is repeated by its weight, after scaling the weights to sum to the number
// of values. DESeq2 documents a weighted quantile (Hmisc's wtd.quantile); this rule
// reproduces its prior variances from its own estimates to rounding.
inline double weighted_quantile(const Vector &x, const Vector &w, double p) {
    require(!x.empty() && x.size() == w.size(), "weighted quantile needs matching values");
    std::vector<std::size_t> order(x.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return x[a] < x[b]; });
    double total = 0;
    for (double v : w)
        total += v;
    const double n = static_cast<double>(x.size());
    Vector cumulative(x.size());
    double sum = 0;
    for (std::size_t k = 0; k < order.size(); ++k)
        cumulative[k] = sum += w[order[k]] * n / total;
    // The value at position t: the first whose cumulative weight reaches t.
    const auto at = [&](double t) {
        const auto k =
            std::lower_bound(cumulative.begin(), cumulative.end(), t) - cumulative.begin();
        return x[order[std::min<std::size_t>(k, order.size() - 1)]];
    };
    const double position = 1 + (n - 1) * p, low = std::floor(position), fraction = position - low;
    return fraction == 0 ? at(low) : (1 - fraction) * at(low) + fraction * at(low + 1);
}
struct MapFit {
    Vector beta;       // natural-log scale
    Matrix covariance; // (H + P)^-1 H (H + P)^-1
    bool ok = false;
};
// Posterior mode of NB GLM coefficients at a fixed dispersion under independent
// priors beta_j ~ N(0, 1 / penalty_j), by Fisher scoring with step halving on the
// log posterior. The covariance is the sandwich (H + P)^-1 H (H + P)^-1 of the DESeq2
// paper, with H = X'WX from fitted counts floored at 0.5 (DESeq2's documented minmu).
inline MapFit nb_map_fit(const Matrix &x, const Vector &y, const Vector &offset, double dispersion,
                         const Vector &penalty, Vector beta, std::size_t max_iterations = 1000) {
    const std::size_t n = x.rows, p = x.cols;
    MapFit r;
    const auto means = [&](const Vector &b) {
        Vector mu(n);
        for (std::size_t i = 0; i < n; ++i) {
            double eta = offset[i];
            for (std::size_t j = 0; j < p; ++j)
                eta += x(i, j) * b[j];
            mu[i] = std::exp(eta);
        }
        return mu;
    };
    const auto objective = [&](const Vector &b, const Vector &mu) {
        double v = 0;
        for (std::size_t i = 0; i < n; ++i)
            v += y[i] * std::log(mu[i]) - (y[i] + 1 / dispersion) * std::log1p(dispersion * mu[i]);
        for (std::size_t j = 0; j < p; ++j)
            v -= 0.5 * penalty[j] * b[j] * b[j];
        return v;
    };
    Vector mu = means(beta), w(n), z(n);
    double value = objective(beta, mu);
    if (!std::isfinite(value))
        return r;
    for (std::size_t it = 0; it < max_iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = mu[i] / (1 + dispersion * mu[i]);
            z[i] = std::log(mu[i]) - offset[i] + (y[i] - mu[i]) / mu[i];
        }
        auto a = crossprod(x, w);
        for (std::size_t j = 0; j < p; ++j)
            a(j, j) += penalty[j];
        const Cholesky ch(a, 1e-12, false);
        if (!ch.ok)
            return r;
        Vector wz(n);
        for (std::size_t i = 0; i < n; ++i)
            wz[i] = w[i] * z[i];
        const auto target = ch.solve(xtv(x, wz));
        // Halve the step until the log posterior does not decrease.
        Vector next(p), next_mu;
        double next_value = -std::numeric_limits<double>::infinity(), step = 1;
        for (int h = 0; h < 40; ++h, step *= 0.5) {
            for (std::size_t j = 0; j < p; ++j)
                next[j] = beta[j] + step * (target[j] - beta[j]);
            next_mu = means(next);
            next_value = objective(next, next_mu);
            if (std::isfinite(next_value) && next_value >= value - 1e-12 * (1 + std::abs(value)))
                break;
        }
        if (!std::isfinite(next_value))
            return r;
        double change = 0;
        for (std::size_t j = 0; j < p; ++j)
            change = std::max(change, std::abs(next[j] - beta[j]) / (1 + std::abs(beta[j])));
        beta = std::move(next);
        mu = std::move(next_mu);
        value = next_value;
        if (change < 1e-10) {
            r.ok = true;
            break;
        }
    }
    if (!r.ok)
        return r;
    for (std::size_t i = 0; i < n; ++i) {
        const double m = std::max(mu[i], 0.5);
        w[i] = m / (1 + dispersion * m);
    }
    const auto h = crossprod(x, w);
    auto a = h;
    for (std::size_t j = 0; j < p; ++j)
        a(j, j) += penalty[j];
    const Cholesky ch(a);
    if (!ch.ok) {
        r.ok = false;
        return r;
    }
    const auto inverse = ch.inverse();
    Matrix left(p, p);
    r.covariance = Matrix(p, p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < p; ++k)
            for (std::size_t l = 0; l < p; ++l)
                left(j, k) += inverse(j, l) * h(l, k);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < p; ++k)
            for (std::size_t l = 0; l < p; ++l)
                r.covariance(j, k) += left(j, l) * inverse(l, k);
    r.beta = std::move(beta);
    return r;
}
} // namespace detail
// Median-of-ratios size factors (Anders and Huber 2010). `ratio` uses genes with
// no zero count; `poscounts` takes each gene's geometric mean as the n-th root of
// the product of its positive counts and uses the positive ratios, then scales the
// factors to geometric mean 1.
inline Vector estimate_size_factors(const SparseCSC &counts, SizeFactorMethod method) {
    counts.validate();
    const std::size_t m = counts.rows;
    std::vector<Vector> ratios(m);
    for (std::size_t j = 0; j < counts.cols; ++j) {
        const auto nonzero = counts.column_ptr[j + 1] - counts.column_ptr[j];
        if (nonzero == 0 || (method == SizeFactorMethod::ratio && nonzero < m))
            continue;
        double lg = 0;
        for (auto k = counts.column_ptr[j]; k < counts.column_ptr[j + 1]; ++k)
            lg += std::log(counts.values[k]);
        lg /= static_cast<double>(m);
        for (auto k = counts.column_ptr[j]; k < counts.column_ptr[j + 1]; ++k)
            ratios[counts.row_index[k]].push_back(std::log(counts.values[k]) - lg);
    }
    Vector sf(m);
    for (std::size_t i = 0; i < m; ++i) {
        require(!ratios[i].empty(), method == SizeFactorMethod::ratio
                                        ? "every gene has a zero count; use size_factors=poscounts"
                                        : "a sample has no positive counts");
        sf[i] = std::exp(detail::median(ratios[i]));
    }
    if (method == SizeFactorMethod::poscounts) {
        double lg = 0;
        for (double s : sf)
            lg += std::log(s);
        lg /= m;
        for (double &s : sf)
            s /= std::exp(lg);
    }
    return sf;
}
struct FilterResult {
    Vector adjusted_p; // missing for filtered genes and missing p-values
    double threshold = nan, theta = nan;
    std::vector<std::pair<double, std::size_t>> rejections;
    Vector fit;
};
// Benjamini-Hochberg adjustment after independent filtering on the mean of
// normalized counts (base_mean), as bulk_de applies it; `alpha` is the target
// false discovery rate the filter maximizes discoveries at.
inline FilterResult independent_filter(const Vector &p_all, const Vector &filter, double alpha,
                                       bool enabled) {
    require(p_all.size() == filter.size(), "p-values and base means differ in length");
    const std::size_t genes = p_all.size();
    FilterResult out;
    out.adjusted_p.assign(genes, nan);
    // Independent filtering on the mean of normalized counts: 50 quantiles from
    // the fraction of zero means to 0.95; LOWESS of rejections over quantiles; the
    // lowest quantile whose rejections exceed the curve's peak minus one residual
    // standard deviation.
    std::vector<unsigned char> pass(genes, 1);
    if (enabled && genes > 0) {
        double zero = 0;
        for (double f : filter)
            zero += f == 0;
        const double lower = zero / genes, upper = lower < 0.95 ? 0.95 : 1;
        Vector theta, rejections;
        for (std::size_t k = 0; k < 50; ++k) {
            const double t = lower + (upper - lower) * k / 49;
            const double cut = detail::quantile7(filter, t);
            Vector sub;
            std::vector<std::size_t> idx;
            for (std::size_t j = 0; j < genes; ++j)
                if (filter[j] >= cut && std::isfinite(p_all[j])) {
                    sub.push_back(p_all[j]);
                    idx.push_back(j);
                }
            const auto q = adjust_pvalues(sub);
            std::size_t rej = 0;
            for (double v : q)
                rej += v < alpha;
            theta.push_back(t);
            rejections.push_back(static_cast<double>(rej));
            out.rejections.push_back({t, rej});
        }
        // Span 0.2 with three robustness passes reproduces the curve DESeq2 records.
        out.fit = detail::lowess(theta, rejections, 0.2, 3);
        double peak = *std::max_element(out.fit.begin(), out.fit.end()), sq = 0;
        for (std::size_t k = 0; k < theta.size(); ++k)
            sq += std::pow(rejections[k] - out.fit[k], 2);
        const double sd = std::sqrt(sq / (theta.size() - 1));
        std::size_t chosen = 0;
        for (std::size_t k = 0; k < theta.size(); ++k)
            if (rejections[k] > peak - sd) {
                chosen = k;
                break;
            }
        out.theta = theta[chosen];
        out.threshold = detail::quantile7(filter, theta[chosen]);
        for (std::size_t j = 0; j < genes; ++j)
            pass[j] = filter[j] >= out.threshold;
    }
    Vector sub;
    std::vector<std::size_t> idx;
    for (std::size_t j = 0; j < genes; ++j)
        if (pass[j] && std::isfinite(p_all[j])) {
            sub.push_back(p_all[j]);
            idx.push_back(j);
        }
    const auto q = adjust_pvalues(sub);
    for (std::size_t k = 0; k < idx.size(); ++k)
        out.adjusted_p[idx[k]] = q[k];
    return out;
}
namespace detail {
// Robust dispersion for Cook's distances as DESeq2 computes it, following its
// implementation in PyDESeq2 (MIT; Muzellec et al. 2023;
// pydeseq2.utils.robust_method_of_moments_disp and trimmed_cell_variance). Cells
// are samples with identical design rows; per cell of at least three samples, the
// trimmed mean of the normalized counts and the scaled trimmed mean of their
// squared deviations from it, with trim 1/3, 1/4 or 1/8 and scale 2.04, 1.86 or
// 1.51 for cells of fewer than 3.5, fewer than 23.5, or more samples. The variance
// is the maximum over cells; without such cells, 1.51 times the 1/8-trimmed
// variance of all samples. The dispersion (v - mean) / mean^2 is floored at 0.04.
inline double trimmed_mean(Vector v, double trim) {
    std::sort(v.begin(), v.end());
    const auto k = static_cast<std::size_t>(std::floor(static_cast<double>(v.size()) * trim));
    double sum = 0;
    for (std::size_t i = k; i < v.size() - k; ++i)
        sum += v[i];
    return sum / static_cast<double>(v.size() - 2 * k);
}
inline double legacy_cooks_dispersion(const Vector &normalized, const Matrix &design, double mean) {
    const std::size_t m = design.rows, p = design.cols;
    std::map<std::vector<double>, Vector> cells;
    for (std::size_t i = 0; i < m; ++i)
        cells[std::vector<double>(&design.values[i * p], &design.values[i * p] + p)].push_back(
            normalized[i]);
    const auto scaled_trimmed_variance = [](const Vector &v, double trim, double scale) {
        const double centre = trimmed_mean(v, trim);
        Vector squares(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            squares[i] = (v[i] - centre) * (v[i] - centre);
        return scale * trimmed_mean(squares, trim);
    };
    double var = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (const auto &entry : cells) {
        const double n = static_cast<double>(entry.second.size());
        if (n < 3)
            continue;
        const int bin = n >= 23.5 ? 2 : n >= 3.5 ? 1 : 0;
        const double trims[] = {1.0 / 3, 1.0 / 4, 1.0 / 8}, scales[] = {2.04, 1.86, 1.51};
        var = std::max(var, scaled_trimmed_variance(entry.second, trims[bin], scales[bin]));
        any = true;
    }
    if (!any)
        var = scaled_trimmed_variance(normalized, 0.125, 1.51);
    return std::max((var - mean) / (mean * mean), 0.04);
}
} // namespace detail
// Batch building blocks of the procedure, for callers that run their own pipeline
// around RapidGLM's fitters (e.g. a PyDESeq2 inference back end). Counts are
// observations x features (row-major), as are fitted means and hat values.
struct NbFixedFits {
    Matrix coefficients; // features x design columns, natural log scale
    Matrix fitted, hat;  // observations x features
    std::vector<unsigned char> converged;
};
// NB GLM fits at a fixed dispersion per feature with log size-factor offsets.
// Coefficients are clamped to [min_beta, max_beta]; fitted means are
// s_i exp(x_i' beta) at the returned coefficients and hat values the diagonal of
// W^1/2 X (X'WX)^-1 X'W^1/2 with W = mu / (1 + dispersion mu) there. Features
// whose fit fails are marked not converged (their values are missing).
inline NbFixedFits nb_fixed_fits(const Matrix &design, const Matrix &counts,
                                 const Vector &size_factors, const Vector &dispersions,
                                 double tolerance = 1e-8, std::size_t max_iterations = 1000,
                                 double min_beta = -30, double max_beta = 30, int threads = 1) {
    design.validate();
    const std::size_t m = design.rows, p = design.cols, genes = counts.cols;
    require(counts.rows == m && size_factors.size() == m && dispersions.size() == genes,
            "nb_fixed_fits dimensions mismatch");
    Vector offset(m);
    for (std::size_t i = 0; i < m; ++i) {
        require(size_factors[i] > 0 && std::isfinite(size_factors[i]),
                "size factors must be positive");
        offset[i] = std::log(size_factors[i]);
    }
    NbFixedFits out;
    out.coefficients = Matrix(genes, p, nan);
    out.fitted = Matrix(m, genes, nan);
    out.hat = Matrix(m, genes, nan);
    out.converged.assign(genes, 0);
    parallel_for(genes, threads, [&](auto j) {
        const double alpha = dispersions[j];
        if (!(alpha > 0) || !std::isfinite(alpha))
            return;
        Vector y(m);
        for (std::size_t i = 0; i < m; ++i)
            y[i] = counts(i, j);
        FitOptions fo;
        fo.family = Family::negative_binomial;
        fo.dispersion = Dispersion::fixed;
        fo.theta = 1 / alpha;
        fo.theta_min = std::min(fo.theta_min, fo.theta);
        fo.theta_max = std::max(fo.theta_max, fo.theta);
        fo.offset = offset;
        fo.tolerance = tolerance;
        fo.max_iterations = max_iterations;
        const auto f = fit(design, y, fo);
        if (!f.ok() || f.coefficients.size() != p)
            return;
        Vector beta(p), mu(m), w(m);
        for (std::size_t k = 0; k < p; ++k)
            beta[k] = std::clamp(f.coefficients[k], min_beta, max_beta);
        const auto eta = detail::matvec(design, beta);
        for (std::size_t i = 0; i < m; ++i) {
            mu[i] = size_factors[i] * std::exp(eta[i]);
            w[i] = mu[i] / (1 + alpha * mu[i]);
        }
        const detail::Cholesky information(detail::crossprod(design, w), 1e-12, false);
        if (!information.ok)
            return;
        const auto covariance = information.inverse();
        for (std::size_t i = 0; i < m; ++i) {
            double h = 0;
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    h += design(i, a) * covariance(a, b) * design(i, b);
            out.hat(i, j) = w[i] * h;
            out.fitted(i, j) = mu[i];
        }
        for (std::size_t k = 0; k < p; ++k)
            out.coefficients(j, k) = beta[k];
        out.converged[j] = 1;
    });
    return out;
}
// Dispersion estimates (1 / theta) per feature at fixed means: the NB
// likelihood, Cox-Reid adjusted when `cox_reid` (and the adjustment is defined
// at the start), maximized by Newton steps in log dispersion from `start`. With
// prior_variance > 0 the objective adds a normal prior on log dispersion centred
// at log(start), giving the MAP estimates of the procedure. Results are clamped
// to [min_dispersion, max_dispersion]; features with no counts are not converged.
// With keep_minimum_start (and no prior), features starting at min_dispersion keep
// it, as the procedure's line search does from that flat start (see
// BulkOptions::keep_minimum_start).
inline std::pair<Vector, std::vector<unsigned char>>
nb_dispersions(const Matrix &design, const Matrix &counts, const Matrix &mu, const Vector &start,
               double min_dispersion = 1e-8, double max_dispersion = 10, double prior_variance = 0,
               bool cox_reid = true, int threads = 1, bool keep_minimum_start = true) {
    design.validate();
    const std::size_t m = design.rows, genes = counts.cols;
    require(counts.rows == m && mu.rows == m && mu.cols == genes && start.size() == genes,
            "nb_dispersions dimensions mismatch");
    require(min_dispersion > 0 && max_dispersion > min_dispersion, "invalid dispersion bounds");
    Vector dispersion(genes, nan);
    std::vector<unsigned char> converged(genes, 0);
    FitOptions options;
    options.theta_min = 1 / max_dispersion;
    options.theta_max = 1 / min_dispersion;
    options.tolerance = 1e-8;
    parallel_for(genes, threads, [&](auto j) {
        Vector y(m), means(m);
        double total = 0;
        for (std::size_t i = 0; i < m; ++i) {
            y[i] = counts(i, j);
            means[i] = mu(i, j);
            total += y[i];
        }
        const double s = std::clamp(start[j], min_dispersion, max_dispersion);
        if (total == 0 || !std::isfinite(s)) {
            dispersion[j] = s;
            return;
        }
        if (keep_minimum_start && prior_variance <= 0 && start[j] <= min_dispersion) {
            dispersion[j] = min_dispersion;
            converged[j] = 1;
            return;
        }
        const Vector a(m, 1);
        const detail::CountTerms terms(y, a);
        const bool cr = cox_reid && detail::cox_reid(design, means, a, 1 / s, false).ok;
        detail::LogThetaPrior prior{-std::log(s), prior_variance};
        const double theta =
            detail::estimate_theta(y, means, a, 1 / s, options, terms, cr ? &design : nullptr,
                                   prior_variance > 0 ? &prior : nullptr);
        if (std::isfinite(theta) && theta > 0) {
            dispersion[j] = std::clamp(1 / theta, min_dispersion, max_dispersion);
            converged[j] = 1;
        }
    });
    return {dispersion, converged};
}
inline BulkResult bulk_de(const Matrix &design, const SparseCSC &counts,
                          const BulkOptions &o = {}) {
    design.validate();
    counts.validate();
    const std::size_t m = counts.rows, genes = counts.cols, p = design.cols;
    require(design.rows == m && o.coefficient < p, "bulk design dimensions mismatch");
    const Hypothesis hypothesis =
        o.hypothesis.contrasts.rows ? o.hypothesis : Hypothesis::coefficient(p, o.coefficient);
    require(hypothesis.contrasts.cols == p, "hypothesis and design columns differ");
    require(o.lrt || hypothesis.df() == 1,
            "the Wald test takes one coefficient or contrast; use the likelihood ratio test");
    const auto tested_columns = hypothesis.unit_columns();
    require(o.lfc_shrinkage == LfcShrinkage::none || tested_columns.size() == 1,
            "fold-change shrinkage applies to a single tested coefficient");
    require(m > p, "bulk DE needs more samples than design columns");
    for (double v : counts.values)
        require(v >= 0 && std::floor(v) == v, "bulk DE needs nonnegative integer counts");
    require(o.alpha > 0 && o.alpha < 1, "alpha must be in (0, 1)");
    BulkResult out;
    out.genes.resize(genes);
    out.size_factors = o.size_factors.empty() ? estimate_size_factors(counts, o.size_factor_method)
                                              : o.size_factors;
    require(out.size_factors.size() == m, "size factor count mismatch");
    for (double s : out.size_factors)
        require(std::isfinite(s) && s > 0, "size factors must be positive");
    const auto &sf = out.size_factors;
    const double min_disp = o.min_dispersion,
                 max_disp = o.max_dispersion > 0 ? o.max_dispersion : std::max(10.0, double(m));
    Vector offset(m);
    for (std::size_t i = 0; i < m; ++i)
        offset[i] = std::log(sf[i]);
    const detail::RowGroups groups(design);
    // Rows repeated at least k times (replicates of a design cell).
    std::vector<std::size_t> replicates(m, 0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < m; ++k)
            replicates[i] +=
                std::equal(&design.values[i * p], &design.values[i * p] + p, &design.values[k * p]);
    FitOptions dispersion_options;
    dispersion_options.theta_min = 1 / max_disp;
    dispersion_options.theta_max = 1 / min_disp;
    dispersion_options.tolerance = 1e-8;
    // Initial means: the group-wise mean of normalized counts for designs with as
    // many distinct rows as columns (a linear model), otherwise an NB fit at a
    // rough moment dispersion; floored at min_mean.
    std::vector<Vector> mu0(genes), start0(genes);
    std::vector<unsigned char> usable(genes, 0);
    // Rough moment dispersion from a least-squares fit of normalized counts on the
    // design: the mean over samples of (r^2 - f) / f^2 on the residual df, with
    // fitted values f (floored at 1e-3) and residuals r.
    detail::Cholesky least_squares(detail::crossprod(design, Vector(m, 1)));
    require(least_squares.ok, "singular bulk design");
    const auto moment_dispersion = [&](const Vector &y) {
        Vector nc(m);
        for (std::size_t i = 0; i < m; ++i)
            nc[i] = y[i] / sf[i];
        const auto fitted = detail::matvec(design, least_squares.solve(detail::xtv(design, nc)));
        double sum = 0;
        for (std::size_t i = 0; i < m; ++i) {
            const double f = std::max(fitted[i], 1e-3), r = nc[i] - fitted[i];
            sum += (r * r - f) / (f * f);
        }
        return std::clamp(sum / (m - p), min_disp, max_disp);
    };
    // Initial means for designs that are not group designs: an NB fit at the rough
    // dispersion, else a Poisson fit, else least-squares means of normalized counts.
    // Sparse genes with large dispersions approach coefficient boundaries about one
    // unit per iteration, so the fits allow 1000 iterations.
    constexpr std::size_t fit_iterations = 1000;
    // `start` receives the NB coefficients of fits without a boundary, which
    // warm-start the final fit.
    const auto initial_means = [&](const Vector &y, double rough, Vector &start) {
        FitOptions fo;
        fo.max_iterations = fit_iterations;
        fo.dispersion = Dispersion::fixed;
        fo.theta = 1 / rough;
        fo.offset = offset;
        auto f = fit(design, y, fo);
        if (f.ok() && !f.has_boundary())
            start = f.coefficients; // diverging starts can derail the final fit
        if (!f.ok()) {
            fo.family = Family::poisson;
            f = fit(design, y, fo);
        }
        if (f.ok())
            return f.fitted;
        Vector nc(m), mu(m);
        for (std::size_t i = 0; i < m; ++i)
            nc[i] = y[i] / sf[i];
        const auto fitted = detail::matvec(design, least_squares.solve(detail::xtv(design, nc)));
        for (std::size_t i = 0; i < m; ++i)
            mu[i] = sf[i] * std::max(fitted[i], 0.0);
        return mu;
    };
    // Gene-wise dispersion: the Cox-Reid adjusted profile likelihood maximized at
    // the initial means (written to `mu`), starting from the rough moment estimate.
    const auto gene_dispersion = [&](const Vector &y, BulkGene &g, Vector &mu, Vector &start) {
        const double rough = moment_dispersion(y);
        start.clear();
        if (groups.ok) {
            Vector sum(p), size(p);
            for (std::size_t i = 0; i < m; ++i) {
                sum[groups.group[i]] += y[i] / sf[i];
                ++size[groups.group[i]];
            }
            mu.assign(m, 0);
            for (std::size_t i = 0; i < m; ++i)
                mu[i] = sf[i] * sum[groups.group[i]] / size[groups.group[i]];
        } else
            mu = initial_means(y, rough, start);
        for (double &v : mu)
            v = std::max(v, o.min_mean);
        if (o.keep_minimum_start && rough <= min_disp) {
            g.dispersion_gene = min_disp;
            return;
        }
        const Vector a(m, 1);
        const detail::CountTerms terms(y, a);
        const bool cr = detail::cox_reid(design, mu, a, 1 / rough, false).ok;
        const double theta = detail::estimate_theta(y, mu, a, 1 / rough, dispersion_options, terms,
                                                    cr ? &design : nullptr);
        g.dispersion_gene = std::clamp(1 / theta, min_disp, max_disp);
    };
    parallel_for(genes, o.threads, [&](auto j) {
        auto &g = out.genes[j];
        const auto y = counts.column(j);
        double total = 0;
        for (std::size_t i = 0; i < m; ++i) {
            total += y[i] / sf[i];
        }
        g.base_mean = total / m;
        if (total == 0) {
            g.status = "all_zero";
            return;
        }
        gene_dispersion(y, g, mu0[j], start0[j]);
        usable[j] = 1;
    });
    // Dispersion trend over genes whose estimate is at least 10 min_disp.
    std::vector<std::size_t> fitted_genes;
    for (std::size_t j = 0; j < genes; ++j)
        if (usable[j] && out.genes[j].dispersion_gene >= 10 * min_disp)
            fitted_genes.push_back(j);
    require(!fitted_genes.empty(), "no genes with dispersion estimates for the trend");
    auto trend = o.trend;
    if (trend == DispersionTrend::parametric) {
        // Iteratively refit, leaving out genes whose ratio to the fit is outside
        // [1e-4, 15], until the squared log changes of both coefficients sum below 1e-6.
        double a0 = 0.1, a1 = 1;
        std::vector<std::size_t> selected = fitted_genes;
        bool ok = false;
        for (int it = 0; it < 30; ++it) {
            Vector y, x;
            for (auto j : selected) {
                y.push_back(out.genes[j].dispersion_gene);
                x.push_back(1 / out.genes[j].base_mean);
            }
            double b0 = a0, b1 = a1;
            if (y.size() < 3 || !detail::gamma_identity_fit(y, x, b0, b1) || !(b0 > 0) || !(b1 > 0))
                break;
            const double change =
                it == 0 ? 1 : std::pow(std::log(b0 / a0), 2) + std::pow(std::log(b1 / a1), 2);
            a0 = b0;
            a1 = b1;
            if (change < 1e-6) {
                ok = true;
                break;
            }
            selected.clear();
            for (auto j : fitted_genes) {
                const double ratio =
                    out.genes[j].dispersion_gene / (a0 + a1 / out.genes[j].base_mean);
                if (ratio >= 1e-4 && ratio <= 15)
                    selected.push_back(j);
            }
        }
        if (ok) {
            out.asymptotic_dispersion = a0;
            out.extra_poisson = a1;
            for (std::size_t j = 0; j < genes; ++j)
                if (usable[j])
                    out.genes[j].dispersion_trend = a0 + a1 / out.genes[j].base_mean;
        } else
            trend = DispersionTrend::local;
    }
    if (trend == DispersionTrend::local) {
        // Weighted running median of log dispersion over log mean (RapidGLM's
        // local fit; DESeq2's local fit uses locfit).
        Vector x, y;
        for (auto j : fitted_genes) {
            x.push_back(std::log(out.genes[j].base_mean));
            y.push_back(std::log(out.genes[j].dispersion_gene));
        }
        const auto smooth = running_median(x, y, std::max<std::size_t>(x.size() / 10, 100));
        // Genes outside the fitted set take the value of the nearest fitted mean.
        std::vector<std::size_t> order(x.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) { return x[a] < x[b]; });
        for (std::size_t j = 0; j < genes; ++j)
            if (usable[j]) {
                const double lx = std::log(out.genes[j].base_mean);
                auto pos = std::lower_bound(order.begin(), order.end(), lx,
                                            [&](std::size_t k, double v) { return x[k] < v; });
                if (pos == order.end())
                    --pos;
                out.genes[j].dispersion_trend = std::exp(smooth[*pos]);
            }
    }
    if (trend == DispersionTrend::mean) {
        double mean = 0;
        for (auto j : fitted_genes)
            mean += out.genes[j].dispersion_gene;
        mean /= fitted_genes.size();
        for (std::size_t j = 0; j < genes; ++j)
            if (usable[j])
                out.genes[j].dispersion_trend = mean;
    }
    out.trend = trend == DispersionTrend::parametric ? "parametric"
                : trend == DispersionTrend::local    ? "local"
                                                     : "mean";
    // Prior on log dispersion: robust residual variance minus the expected
    // sampling variance trigamma((m - p) / 2), at least 0.25.
    Vector residuals;
    for (auto j : fitted_genes)
        residuals.push_back(std::log(out.genes[j].dispersion_gene) -
                            std::log(out.genes[j].dispersion_trend));
    const double s_lr = detail::mad(residuals), df = static_cast<double>(m - p);
    out.log_residual_variance = s_lr * s_lr;
    out.prior_variance =
        df > 3 ? std::max(s_lr * s_lr - boost::math::trigamma(df / 2), 0.25)
               : std::max(detail::simulated_prior_variance(residuals, df, o.seed, o.threads), 0.25);
    const detail::LogThetaPrior base_prior{0, out.prior_variance};
    // MAP dispersions; dispersion outliers keep the gene-wise estimate. Then the
    // final fit and test.
    const double cutoff =
        o.cooks_cutoff > 0
            ? o.cooks_cutoff
            : boost::math::quantile(boost::math::fisher_f_distribution<double>(p, m - p), 0.99);
    out.cooks_cutoff = cutoff;
    const auto reduced = reduced_design(design, hypothesis);
    // One-row hypotheses are fitted in a design where the contrast is a coefficient.
    const auto tested =
        hypothesis.df() == 1 ? tested_design(design, hypothesis) : TestedDesign{design, 0};
    const auto target =
        hypothesis.df() == 1 ? Hypothesis::coefficient(p, tested.column) : hypothesis;
    // Designs fitted as given start the final fit from the initial NB fit.
    const bool warm_start = tested.x.values == design.values;
    const auto finish = [&](std::size_t j, const Vector &y, const Vector &mu, const Vector &start) {
        auto &g = out.genes[j];
        const Vector a(m, 1);
        const detail::CountTerms terms(y, a);
        auto prior = base_prior;
        prior.center = -std::log(g.dispersion_trend);
        const bool cr = detail::cox_reid(design, mu, a, 1 / g.dispersion_gene, false).ok;
        const double theta =
            detail::estimate_theta(y, mu, a, 1 / g.dispersion_trend, dispersion_options, terms,
                                   cr ? &design : nullptr, &prior);
        g.dispersion_map = std::clamp(1 / theta, min_disp, max_disp);
        g.dispersion_outlier =
            std::log(g.dispersion_gene) > std::log(g.dispersion_trend) + o.outlier_sd * s_lr;
        g.dispersion = g.dispersion_outlier ? g.dispersion_gene : g.dispersion_map;
        FitOptions fo;
        fo.max_iterations = fit_iterations;
        fo.dispersion = Dispersion::fixed;
        fo.theta = 1 / g.dispersion;
        fo.offset = offset;
        fo.theta_max = 1 / min_disp;
        fo.theta_min = 1 / max_disp;
        auto full_options = fo;
        if (warm_start && start.size() == p)
            full_options.start = start;
        auto full = fit(tested.x, y, full_options);
        if (!full.ok()) {
            g.status = std::string("fit_") + status_name(full.status);
            return full;
        }
        const double ln2 = std::log(2.0);
        const bool boundary = hypothesis_boundary(full, target);
        if (hypothesis.df() == 1 && !boundary) {
            const auto [estimate, se] = hypothesis_estimate(full, target);
            g.log2_fold_change = estimate / ln2;
            g.lfc_se = se / ln2;
        }
        if (o.lrt) {
            auto null = fit(reduced, y, fo);
            if (!null.ok()) {
                g.status = std::string("reduced_") + status_name(null.status);
                return full;
            }
            g.statistic = std::max(0.0, 2 * (full.log_likelihood - null.log_likelihood));
            boost::math::chi_squared_distribution<double> chi(static_cast<double>(hypothesis.df()));
            g.p_value = boost::math::cdf(boost::math::complement(chi, g.statistic));
            g.status = full.has_boundary() ? "ok_boundary" : "ok";
        } else {
            const auto w = hypothesis_wald(full, target);
            if (!w.valid && boundary) {
                // The fold change is infinite (e.g. an all-zero group) and has no Wald
                // test; the likelihood ratio at the final dispersion still does.
                auto null = fit(reduced, y, fo);
                g.log2_fold_change = nan;
                if (!null.ok()) {
                    g.status = "coefficient_boundary";
                    return full;
                }
                g.statistic = std::max(0.0, 2 * (full.log_likelihood - null.log_likelihood));
                boost::math::chi_squared_distribution<double> chi(1);
                g.p_value = boost::math::cdf(boost::math::complement(chi, g.statistic));
                g.status = "boundary_lrt";
                return full;
            }
            if (!w.valid) {
                g.status = "wald_" + std::string(status_name(full.status));
                return full;
            }
            g.statistic = w.statistic;
            g.p_value = w.p_value;
            g.status = "ok";
        }
        return full;
    };
    // Cook's distances D = R^2 / p h / (1 - h)^2 with Pearson residuals under a
    // robust moment dispersion (see robust_dispersion) and hat values from the
    // final fit; the maximum over samples with at least three replicates.
    const auto legacy_dispersion = [&](const Vector &y, double base_mean) {
        Vector normalized(m);
        for (std::size_t i = 0; i < m; ++i)
            normalized[i] = y[i] / sf[i];
        return detail::legacy_cooks_dispersion(normalized, design, base_mean);
    };
    const auto robust_dispersion = [&](const Vector &y, double base_mean, double trend) {
        // Pooled within-cell variance of normalized counts, each cell's variance
        // estimated by the squared scaled MAD, as a moment dispersion floored at a
        // quarter of the trended dispersion (RapidGLM choice; the paper leaves the
        // robust estimator unspecified). Cells with fewer than 3 samples do not
        // contribute; without such cells the ordinary variance is used.
        std::map<std::vector<double>, Vector> cells;
        for (std::size_t i = 0; i < m; ++i)
            cells[std::vector<double>(&design.values[i * p], &design.values[i * p] + p)].push_back(
                y[i] / sf[i]);
        double pooled = 0, weight = 0;
        for (auto &entry : cells)
            if (entry.second.size() >= 3) {
                const double s = detail::mad(entry.second);
                pooled += s * s * (entry.second.size() - 1);
                weight += entry.second.size() - 1;
            }
        double var;
        if (weight > 0)
            var = pooled / weight;
        else {
            double sum = 0, sq = 0;
            for (std::size_t i = 0; i < m; ++i) {
                sum += y[i] / sf[i];
                sq += (y[i] / sf[i]) * (y[i] / sf[i]);
            }
            var = (sq - sum * sum / m) / (m - 1);
        }
        return std::max((var - base_mean) / (base_mean * base_mean), 0.25 * trend);
    };
    std::vector<Vector> cooks(genes);
    const auto cook_distances = [&](std::size_t j, const Vector &y, const FitResult &f) {
        Vector d(m, nan);
        if (!f.ok() || f.fitted.size() != m || f.covariance.rows != p)
            return d;
        const double arob = o.legacy_cooks ? legacy_dispersion(y, out.genes[j].base_mean)
                                           : robust_dispersion(y, out.genes[j].base_mean,
                                                               out.genes[j].dispersion_trend);
        for (std::size_t i = 0; i < m; ++i) {
            const double mu = f.fitted[i], w = f.working_weights[i];
            double h = 0;
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    h += tested.x(i, a) * f.covariance(a, b) * tested.x(i, b);
            h *= w;
            const double r = (y[i] - mu) / std::sqrt(mu + arob * mu * mu);
            d[i] = h < 1 ? r * r / p * h / ((1 - h) * (1 - h)) : nan;
        }
        return d;
    };
    // Samples eligible for replacement are handled by replacement, not flagging.
    const std::size_t flag_limit = o.min_replicates_for_replace > 0
                                       ? o.min_replicates_for_replace
                                       : std::numeric_limits<std::size_t>::max();
    const auto max_cooks = [&](const Vector &d, std::size_t min_rep, std::size_t below) {
        double best = nan;
        for (std::size_t i = 0; i < m; ++i)
            if (replicates[i] >= min_rep && replicates[i] < below && std::isfinite(d[i]))
                best = std::isnan(best) ? d[i] : std::max(best, d[i]);
        return best;
    };
    // Final maximum likelihood coefficients, with coefficients that have no finite
    // estimate stored as infinite (for the fold-change prior).
    std::vector<Vector> mle(genes);
    const auto keep_mle = [&](std::size_t j, const FitResult &full) {
        mle[j].clear();
        if (!full.ok())
            return;
        mle[j] = full.coefficients;
        for (std::size_t c = 0; c < p; ++c)
            if (!full.boundary_coefficients.empty() && full.boundary_coefficients[c])
                mle[j][c] = std::numeric_limits<double>::infinity();
    };
    parallel_for(genes, o.threads, [&](auto j) {
        if (!usable[j])
            return;
        const auto y = counts.column(j);
        auto full = finish(j, y, mu0[j], start0[j]);
        cooks[j] = cook_distances(j, y, full);
        keep_mle(j, full);
    });
    // Outlier replacement: counts of samples in cells with enough replicates whose
    // Cook's distance exceeds the cutoff become the 20% trimmed mean of normalized
    // counts times the size factor; those genes are re-estimated.
    std::vector<Vector> replaced(genes);
    if (o.min_replicates_for_replace > 0) {
        std::vector<std::size_t> refit;
        for (std::size_t j = 0; j < genes; ++j) {
            if (!usable[j] || cooks[j].empty())
                continue;
            auto y = counts.column(j);
            Vector nc(m);
            for (std::size_t i = 0; i < m; ++i)
                nc[i] = y[i] / sf[i];
            std::sort(nc.begin(), nc.end());
            const auto trim = static_cast<std::size_t>(std::floor(0.2 * m));
            double trimmed = 0;
            for (std::size_t i = trim; i < m - trim; ++i)
                trimmed += nc[i];
            trimmed /= m - 2 * trim;
            bool any = false;
            for (std::size_t i = 0; i < m; ++i)
                if (replicates[i] >= o.min_replicates_for_replace && cooks[j][i] > cutoff) {
                    y[i] = std::floor(trimmed * sf[i]);
                    any = true;
                }
            if (any) {
                replaced[j] = std::move(y);
                refit.push_back(j);
            }
        }
        out.replaced_genes = refit.size();
        parallel_for(refit.size(), o.threads, [&](auto k) {
            const auto j = refit[k];
            auto &g = out.genes[j];
            const auto &y = replaced[j];
            g.replaced = true;
            Vector mu;
            Vector start;
            gene_dispersion(y, g, mu, start);
            auto full = finish(j, y, mu, start);
            cooks[j] = cook_distances(j, y, full);
            keep_mle(j, full);
        });
    }
    // Normal-prior fold-change shrinkage (DESeq2 paper, "empirical prior"). Each
    // non-constant design column's prior variance on the log2 scale matches the 0.05
    // upper quantile of the weighted absolute maximum likelihood coefficients to a
    // zero-centred normal, with weights 1 / (1 / base mean + trended dispersion);
    // constant columns (the intercept) get variance 1e6. Counts are the replaced
    // ones where outliers were replaced. p-values are unchanged.
    if (o.lfc_shrinkage == LfcShrinkage::normal) {
        const double ln2 = std::log(2.0);
        out.lfc_prior_variance = o.lfc_prior_variance;
        if (out.lfc_prior_variance.empty()) {
            const double z =
                boost::math::quantile(boost::math::normal_distribution<double>(), 1 - 0.05 / 2);
            out.lfc_prior_variance.assign(p, 1e6);
            for (std::size_t c = 0; c < p; ++c) {
                bool constant = true;
                for (std::size_t i = 1; i < m && constant; ++i)
                    constant = design(i, c) == design(0, c);
                if (constant)
                    continue;
                Vector magnitude, weight;
                for (std::size_t j = 0; j < genes; ++j)
                    if (!mle[j].empty()) {
                        const auto &g = out.genes[j];
                        magnitude.push_back(std::abs(mle[j][c]) / ln2);
                        weight.push_back(1 / (1 / g.base_mean + g.dispersion_trend));
                    }
                require(!magnitude.empty(), "no fitted genes for the fold-change prior");
                const double q = detail::weighted_quantile(magnitude, weight, 1 - 0.05);
                out.lfc_prior_variance[c] = std::max(q / z * (q / z), 1e-10);
            }
        }
        require(out.lfc_prior_variance.size() == p,
                "lfc_prior_variance needs one value per column");
        Vector penalty(p);
        for (std::size_t c = 0; c < p; ++c) {
            require(out.lfc_prior_variance[c] > 0, "lfc prior variances must be positive");
            penalty[c] = 1 / (out.lfc_prior_variance[c] * ln2 * ln2);
        }
        parallel_for(genes, o.threads, [&](auto j) {
            auto &g = out.genes[j];
            if (!usable[j] || !std::isfinite(g.dispersion))
                return;
            const auto y = replaced[j].empty() ? counts.column(j) : replaced[j];
            // Start from the maximum likelihood coefficients when all are finite.
            Vector start(p, 0);
            if (!mle[j].empty() && std::all_of(mle[j].begin(), mle[j].end(),
                                               [](double b) { return std::isfinite(b); }))
                start = mle[j];
            const auto map =
                detail::nb_map_fit(design, y, offset, g.dispersion, penalty, start, fit_iterations);
            if (!map.ok)
                return;
            const auto c = tested_columns[0];
            g.shrunken_lfc = map.beta[c] / ln2;
            g.shrunken_lfc_se = std::sqrt(map.covariance(c, c)) / ln2;
        });
    }
    // Cook's filtering over samples with at least three replicates (replaced
    // samples are no longer outliers after replacement).
    for (std::size_t j = 0; j < genes; ++j) {
        auto &g = out.genes[j];
        if (cooks[j].empty())
            continue;
        g.max_cooks = max_cooks(cooks[j], 3, flag_limit);
        bool outlier = std::isfinite(g.max_cooks) && g.max_cooks > cutoff;
        if (outlier && o.legacy_cooks) {
            // Legacy rule (PyDESeq2): not an outlier when three or more samples have
            // larger counts than the sample with the largest distance over all samples.
            const auto y = counts.column(j); // original counts
            std::size_t top = 0;
            for (std::size_t i = 1; i < m; ++i)
                if (std::isfinite(cooks[j][i]) &&
                    (!std::isfinite(cooks[j][top]) || cooks[j][i] > cooks[j][top]))
                    top = i;
            std::size_t larger = 0;
            for (std::size_t i = 0; i < m; ++i)
                larger += y[i] > y[top];
            outlier = larger < 3;
        }
        if (o.cooks_filter && outlier && std::isfinite(g.p_value)) {
            g.p_value = nan;
            g.status = "cooks_outlier";
        }
    }
    Vector p_all(genes), filter(genes);
    for (std::size_t j = 0; j < genes; ++j) {
        p_all[j] = out.genes[j].p_value;
        filter[j] = out.genes[j].base_mean;
    }
    auto adjusted = independent_filter(p_all, filter, o.alpha, o.independent_filtering);
    for (std::size_t j = 0; j < genes; ++j)
        out.genes[j].adjusted_p = adjusted.adjusted_p[j];
    out.filter_threshold = adjusted.threshold;
    out.filter_theta = adjusted.theta;
    out.filter_rejections = std::move(adjusted.rejections);
    out.filter_fit = std::move(adjusted.fit);
    return out;
}
} // namespace rapidglm
