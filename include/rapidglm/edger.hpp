#pragma once
#include "bulk.hpp"
#include "hypothesis.hpp"
#include "kernels.hpp"
#include "sparse.hpp"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <functional>
#include <string>

// Quasi-likelihood differential expression following the default pipeline of
// edgeR 4 (Chen, Chen, Lun, Baldoni and Smyth 2025, NAR 53:gkaf018; Lund et al.
// 2012; Chen, Lun and Smyth 2016): filtering by expression, TMM normalization
// (Robinson and Oshlack 2010), one NB dispersion, the common Cox-Reid dispersion
// of the most abundant genes, quasi-dispersions from unit deviances matched to
// scaled chi-square variables on fractional df, empirical Bayes moderation of
// them against average log CPM (Smyth 2004, with weights for unequal df), and
// quasi-likelihood F tests.
//
// Written from the papers, the documentation of edgeR and limma and black-box
// runs of edgeR 4.6.2 (docs/EDGER_QL_SPEC.md); no edgeR or limma
// source was read. Where the documentation leaves a rule open and black-box runs
// settled it, the rule is the one the runs showed; the remaining differences are
// listed in METHODS.md. The unit-deviance moments here are exact; edgeR's own
// approximation of them differs from the exact values by up to about 1e-3 for
// dispersions below 0.3.
namespace rapidglm {
struct QlOptions {
    // filterByExpr: keep a gene with at least min_count reads per median library
    // (as CPM) in at least the minimum group size of samples, and min_total_count
    // reads in all. The minimum group size is the inverse of the largest leverage
    // of the design; above large_n it grows only by min_prop per sample.
    bool filter = true;
    double min_count = 10, min_total_count = 15, large_n = 10, min_prop = 0.7;
    // TMM normalization factors; otherwise factors of 1 (or norm_factors).
    bool tmm = true;
    double logratio_trim = 0.3, sum_trim = 0.05;
    Vector norm_factors;
    // The NB dispersion; NaN: the common Cox-Reid dispersion of the most abundant
    // genes, estimated.
    double dispersion = nan;
    // Prior count (per average library) of the reported log fold changes.
    double prior_count = 0.125;
    int threads = 1;
};
struct QlGene {
    bool kept = false;
    // filtered, ok, fit_failed
    std::string status = "filtered";
    double total = 0, ave_log_cpm = nan, log_fc = nan, coefficient = nan,
           unshrunk_coefficient = nan, f = nan, p_value = nan, fdr = nan, deviance = nan,
           deviance_adj = nan, df_residual_adj = nan, s2 = nan, s2_prior = nan, s2_post = nan,
           df_total = nan;
};
struct QlResult {
    std::vector<QlGene> genes;
    // Library sizes of all genes, of the kept genes, normalization factors and
    // effective library sizes (kept-gene sizes times factors).
    Vector library_sizes, filtered_library_sizes, norm_factors, effective_library_sizes;
    // The NB dispersion, the share of genes it was estimated from, the scale a by
    // which it is divided for the fits (the average quasi-dispersion), and the
    // prior df of the quasi-dispersions.
    double dispersion = nan, top_proportion = nan, scale = 1, df_prior = nan;
    double minimum_group = nan;
    std::size_t kept = 0, top_genes = 0, tested_df = 0;
};

namespace detail {
// Ranks with ties given their average rank.
inline Vector average_ranks(const Vector &v) {
    const std::size_t n = v.size();
    std::vector<std::pair<double, std::size_t>> sorted(n);
    for (std::size_t i = 0; i < n; ++i)
        sorted[i] = {v[i], i};
    std::sort(sorted.begin(), sorted.end());
    Vector rank(n);
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i;
        while (j + 1 < n && sorted[j + 1].first == sorted[i].first)
            ++j;
        for (std::size_t k = i; k <= j; ++k)
            rank[sorted[k].second] = 0.5 * static_cast<double>(i + j) + 1;
        i = j + 1;
    }
    return rank;
}
// Whether each value's average rank (ties sharing their mean rank) lies in
// [lo, hi], without sorting: the average rank grows with the value, so the
// values kept are those between two thresholds, found by selection.
inline std::vector<unsigned char> rank_window(const Vector &v, double lo, double hi) {
    const std::size_t n = v.size();
    std::vector<unsigned char> keep(n, 0);
    if (n == 0 || lo > hi)
        return keep;
    const auto average_rank = [&](double x) {
        std::size_t less = 0, equal = 0;
        for (double e : v) {
            less += e < x;
            equal += e == x;
        }
        return static_cast<double>(less) + (static_cast<double>(equal) + 1) / 2;
    };
    Vector work = v;
    std::size_t split = 0; // after a selection, work is partitioned at split
    bool selected = false;
    const auto at = [&](double rank) { // the value at 1-based sorted position `rank`
        const auto k = static_cast<std::size_t>(std::clamp(rank, 1.0, static_cast<double>(n))) - 1;
        auto first = work.begin(), last = work.end();
        if (selected)
            (k >= split ? first : last) = work.begin() + static_cast<std::ptrdiff_t>(split);
        std::nth_element(first, work.begin() + static_cast<std::ptrdiff_t>(k), last);
        split = k;
        selected = true;
        return work[k];
    };
    // The smallest value with average rank >= lo, the largest with <= hi.
    double low = at(std::ceil(lo));
    if (average_rank(low) < lo) {
        double next = std::numeric_limits<double>::infinity();
        for (double e : v)
            if (e > low)
                next = std::min(next, e);
        low = next;
    }
    double high = at(std::floor(hi));
    if (average_rank(high) > hi) {
        double previous = -std::numeric_limits<double>::infinity();
        for (double e : v)
            if (e < high)
                previous = std::max(previous, e);
        high = previous;
    }
    for (std::size_t i = 0; i < n; ++i)
        keep[i] = v[i] >= low && v[i] <= high;
    return keep;
}
// Diagonal of X (X'X)^-1 X'.
inline Vector design_leverage(const Matrix &x) {
    const Cholesky ch(crossprod(x, Vector(x.rows, 1)), 1e-12, false);
    require(ch.ok, "the design must have full column rank");
    const auto c = ch.inverse();
    Vector h(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t a = 0; a < x.cols; ++a)
            for (std::size_t b = 0; b < x.cols; ++b)
                h[i] += x(i, a) * c(a, b) * x(i, b);
    return h;
}
// Diagonal of W^1/2 X (X'WX)^-1 X'W^1/2: the squared row norms of Q in the
// Householder QR of W^1/2 X, which stay accurate when fitted means at the
// boundary make X'WX nearly singular (a column whose remainder falls below
// 1e-10 of its norm is taken as aliased and skipped).
inline Vector weighted_leverage(const Matrix &x, const Vector &w) {
    const std::size_t n = x.rows, p = x.cols;
    Vector a(n * p), h(n);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < n; ++i)
            a[j * n + i] = std::sqrt(w[i]) * x(i, j);
    std::vector<Vector> reflections;
    std::vector<std::size_t> rows;
    std::size_t row = 0;
    for (std::size_t j = 0; j < p && row < n; ++j) {
        double *column = &a[j * n];
        double original = 0, norm = 0;
        for (std::size_t i = 0; i < n; ++i)
            original += column[i] * column[i];
        for (std::size_t i = row; i < n; ++i)
            norm += column[i] * column[i];
        norm = std::sqrt(norm);
        if (!(norm > 1e-10 * std::sqrt(original)))
            continue;
        Vector v(column + row, column + n);
        v[0] += v[0] >= 0 ? norm : -norm;
        double vv = 0;
        for (double e : v)
            vv += e * e;
        for (std::size_t k = j + 1; k < p; ++k) {
            double *other = &a[k * n + row];
            double dot = 0;
            for (std::size_t i = 0; i < v.size(); ++i)
                dot += v[i] * other[i];
            const double f = 2 * dot / vv;
            for (std::size_t i = 0; i < v.size(); ++i)
                other[i] -= f * v[i];
        }
        reflections.push_back(std::move(v));
        rows.push_back(row++);
    }
    // Columns of Q: the reflections applied in reverse order to unit vectors.
    for (std::size_t c = 0; c < reflections.size(); ++c) {
        Vector q(n);
        q[rows[c]] = 1;
        for (std::size_t r = reflections.size(); r-- > 0;) {
            const auto &v = reflections[r];
            const std::size_t start = rows[r];
            double dot = 0, vv = 0;
            for (std::size_t i = 0; i < v.size(); ++i) {
                dot += v[i] * q[start + i];
                vv += v[i] * v[i];
            }
            const double f = 2 * dot / vv;
            for (std::size_t i = 0; i < v.size(); ++i)
                q[start + i] -= f * v[i];
        }
        for (std::size_t i = 0; i < n; ++i)
            h[i] += q[i] * q[i];
    }
    return h;
}
// Brent's (1973) minimization of f on [a, b] by golden sections and parabolic
// steps, to within tol (relative tolerance sqrt(machine epsilon)).
inline double brent_minimum(const std::function<double(double)> &f, double a, double b,
                            double tol) {
    const double golden = (3 - std::sqrt(5.0)) / 2,
                 eps = std::sqrt(std::numeric_limits<double>::epsilon());
    double x = a + golden * (b - a), w = x, v = x, fx = f(x), fw = fx, fv = fx, d = 0, e = 0;
    for (;;) {
        const double xm = (a + b) / 2, tol1 = eps * std::abs(x) + tol / 3, tol2 = 2 * tol1;
        if (std::abs(x - xm) <= tol2 - (b - a) / 2)
            break;
        double p = 0, q = 0, r = 0;
        if (std::abs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2 * (q - r);
            if (q > 0)
                p = -p;
            else
                q = -q;
            r = e;
            e = d;
        }
        if (std::abs(p) >= std::abs(q * r / 2) || p <= q * (a - x) || p >= q * (b - x)) {
            e = x < xm ? b - x : a - x;
            d = golden * e;
        } else {
            d = p / q;
            const double u = x + d;
            if (u - a < tol2 || b - u < tol2)
                d = x < xm ? tol1 : -tol1;
        }
        const double u = std::abs(d) >= tol1 ? x + d : (d > 0 ? x + tol1 : x - tol1), fu = f(u);
        if (fu <= fx) {
            (u < x ? b : a) = x;
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            (u < x ? a : b) = u;
            if (fu <= fw || w == x) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    return x;
}
// NB unit deviance d(y, m, phi) = 2 [y log(y/m) - (y + 1/phi) log((1 + phi y)/(1 + phi m))],
// Poisson at phi = 0. Near y = m in count_deviance's form, which loses no digits
// to cancellation; elsewhere from the logarithms directly, since that form
// cancels terms of the size of m when m is far above y + 1/phi.
inline double nb_unit_deviance(double y, double m, double phi) {
    double t;
    if (y == 0)
        t = phi > 0 ? std::log1p(phi * m) / phi : m;
    else {
        const double difference = m - y;
        if (std::abs(difference) < 0.125 * y) {
            t = y * log1pmx(difference / y);
            if (phi > 0)
                t -= (y + 1 / phi) * log1pmx(difference / (y + 1 / phi));
        } else if (phi > 0) {
            const double k = 1 / phi;
            t = y * std::log(y / m) + (y + k) * std::log1p((m - y) / (y + k));
        } else
            t = y * std::log(y / m) + difference;
    }
    return std::isnan(t) ? t : 2 * std::max(0.0, t);
}
// The factor of a symmetric matrix scaled to unit diagonal, as detail::Cholesky
// computes it (without a condition estimate), refactored in place for each
// Newton step: reciprocals of the scales and pivots are kept so that factoring
// and solving multiply where Cholesky divides.
class SmallCholesky {
  public:
    explicit SmallCholesky(std::size_t p)
        : p_(p), l_(p * p), inverse_scale_(p), inverse_pivot_(p) {}
    // False when a is not numerically positive definite (a squared pivot of the
    // scaled matrix at most tol).
    bool factor(const Matrix &a, double tol) {
        const std::size_t p = p_;
        for (std::size_t j = 0; j < p; ++j) {
            if (!(a(j, j) > 0) || !std::isfinite(a(j, j)))
                return false;
            inverse_scale_[j] = 1 / std::sqrt(a(j, j));
        }
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k) {
                l_[j * p + k] = a(j, k) * inverse_scale_[j] * inverse_scale_[k];
                if (!std::isfinite(l_[j * p + k]))
                    return false;
            }
        for (std::size_t j = 0; j < p; ++j) {
            double d = l_[j * p + j];
            for (std::size_t k = 0; k < j; ++k)
                d -= l_[j * p + k] * l_[j * p + k];
            if (!(d > 0))
                return false;
            const double pivot = std::sqrt(d);
            if (!std::isfinite(pivot) || !(pivot * pivot > tol))
                return false;
            l_[j * p + j] = pivot;
            inverse_pivot_[j] = 1 / pivot;
            for (std::size_t i = j + 1; i < p; ++i) {
                double v = l_[i * p + j];
                for (std::size_t k = 0; k < j; ++k)
                    v -= l_[i * p + k] * l_[j * p + k];
                l_[i * p + j] = v * inverse_pivot_[j];
            }
        }
        return true;
    }
    // x = A^-1 b (x and b of length p, may coincide).
    void solve(const double *b, double *x) const {
        const std::size_t p = p_;
        for (std::size_t i = 0; i < p; ++i) {
            double v = b[i] * inverse_scale_[i];
            for (std::size_t k = 0; k < i; ++k)
                v -= l_[i * p + k] * x[k];
            x[i] = v * inverse_pivot_[i];
        }
        for (std::size_t i = p; i-- > 0;) {
            double v = x[i];
            for (std::size_t k = i + 1; k < p; ++k)
                v -= l_[k * p + i] * x[k];
            x[i] = v * inverse_pivot_[i];
        }
        for (std::size_t i = 0; i < p; ++i)
            x[i] *= inverse_scale_[i];
    }

  private:
    std::size_t p_;
    Vector l_, inverse_scale_, inverse_pivot_;
};
// Fits at a fixed NB dispersion phi (Poisson at 0) with a log link by Newton's
// method from `start` (or the least-squares fit of log(y + 1/2) - offset).
// Steps are taken whole while the Newton decrement b'A^-1 b (score b, observed
// information A; about twice the deviance still to gain) shrinks; once it does
// not, steps are halved until the deviance does not increase. Converged when the
// decrement is below 1e-14 of 100 plus the total count (after a whole step, when
// a bound on it from the previous information is). Not ok when that takes
// more than max_iterations steps or the information is singular; callers then
// use fit(). The deviance (computed when want_deviance) is that of the returned
// fit, log_mu the logarithms of its means (the linear predictors).
struct FixedFit {
    Vector beta, mu, log_mu;
    double deviance = nan;
    bool ok = false;
    std::size_t iterations = 0, halvings = 0;
};
inline FixedFit nb_fixed_fit(const Matrix &x, const Vector &y, const Vector &offset, double phi,
                             const Vector &start, const Cholesky &least_squares,
                             std::size_t max_iterations = 60, bool want_deviance = true,
                             bool convergence_shortcut = true) {
    const std::size_t n = x.rows, p = x.cols;
    FixedFit f;
    if (start.size() == p)
        f.beta = start;
    else {
        if (!least_squares.ok)
            return f;
        Vector z(n);
        for (std::size_t i = 0; i < n; ++i)
            z[i] = std::log(y[i] + 0.5) - offset[i];
        f.beta = least_squares.solve(xtv(x, z));
    }
    const auto means = [&](const Vector &beta, Vector &mu, Vector &log_mu) {
        matrix_vector(x.values.data(), beta.data(), n, p, log_mu.data());
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i) {
            const double eta = offset[i] + log_mu[i];
            finite &= std::isfinite(eta);
            log_mu[i] = std::clamp(eta, -700.0, 700.0);
        }
        exp_array(log_mu.data(), mu.data(), n);
        return finite;
    };
    const auto deviance = [&](const Vector &mu) {
        double total = 0;
        for (std::size_t i = 0; i < n; ++i)
            total += nb_unit_deviance(y[i], mu[i], phi);
        return total;
    };
    f.mu.resize(n);
    f.log_mu.resize(n);
    if (!means(f.beta, f.mu, f.log_mu))
        return f;
    // Newton steps: the observed information X'WX, W = mu (1 + phi y)/(1 + phi mu)^2
    // (positive, the log likelihood being concave in the coefficients), and the
    // score X'(y - mu)/(1 + phi mu) (detail::weighted_gram). The decrement is
    // compared with 1e-14 of 100 plus the total count, the scale of the deviance.
    // Scratch reused across calls on a thread (these fits run millions of times).
    thread_local Vector weight, residual, next_mu, next_log_mu;
    weight.resize(n);
    residual.resize(n);
    next_mu.resize(n);
    next_log_mu.resize(n);
    Matrix a(p, p);
    SmallCholesky ch(p);
    Vector b(p), next(p), step(p), direction(p);
    double previous = std::numeric_limits<double>::infinity(), current = nan, total = 0;
    for (double v : y)
        total += v;
    const double tolerance = 1e-14 * (100 + total);
    bool halving = false;
    for (std::size_t it = 0; it < max_iterations; ++it) {
        f.iterations = it + 1;
        for (std::size_t i = 0; i < n; ++i) {
            const double inverse = 1 / (1 + phi * f.mu[i]);
            weight[i] = f.mu[i] * (1 + phi * y[i]) * inverse * inverse;
            residual[i] = (y[i] - f.mu[i]) * inverse;
        }
        weighted_gram(x.values.data(), weight.data(), residual.data(), n, p, a.values.data(),
                      b.data());
        if (!ch.factor(a, 1e-14))
            return f;
        ch.solve(b.data(), step.data());
        double decrement = 0;
        for (std::size_t j = 0; j < p; ++j)
            decrement += b[j] * step[j];
        if (!std::isfinite(decrement))
            return f;
        if (decrement < tolerance) {
            f.deviance = !std::isnan(current) ? current : want_deviance ? deviance(f.mu) : nan;
            f.ok = !want_deviance || std::isfinite(f.deviance);
            return f;
        }
        halving = halving || decrement >= previous;
        previous = decrement;
        if (!halving) {
            for (std::size_t j = 0; j < p; ++j)
                next[j] = f.beta[j] + step[j];
            if (!means(next, next_mu, next_log_mu))
                return f;
            f.beta.swap(next);
            f.mu.swap(next_mu);
            f.log_mu.swap(next_log_mu);
            current = nan;
            // Convergence at the new point from its score b alone: each weight has
            // changed by at most a factor exp(delta), delta the largest change in a
            // linear predictor, so the decrement there is at most exp(delta) b'A^-1 b
            // with the information A here. When that bound does not settle it, the
            // next step computes the decrement.
            if (convergence_shortcut && it + 1 < max_iterations) {
                double delta = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    delta = std::max(delta, std::abs(f.log_mu[i] - next_log_mu[i]));
                    residual[i] = (y[i] - f.mu[i]) / (1 + phi * f.mu[i]);
                }
                transposed_product(x.values.data(), residual.data(), n, p, b.data());
                ch.solve(b.data(), direction.data());
                double bound = 0;
                for (std::size_t j = 0; j < p; ++j)
                    bound += b[j] * direction[j];
                if (bound * std::exp(delta) * (1 + 1e-9) < tolerance) {
                    f.deviance = want_deviance ? deviance(f.mu) : nan;
                    f.ok = !want_deviance || std::isfinite(f.deviance);
                    return f;
                }
            }
            continue;
        }
        if (std::isnan(current))
            current = deviance(f.mu);
        const double allowed = current + 1e-12 * (current + 0.1);
        double t = 1, value = nan;
        for (int h = 0; h < 40; ++h, t /= 2) {
            f.halvings += h > 0;
            for (std::size_t j = 0; j < p; ++j)
                next[j] = f.beta[j] + t * step[j];
            value = means(next, next_mu, next_log_mu) ? deviance(next_mu) : nan;
            if (value <= allowed)
                break;
        }
        if (!(value <= allowed)) {
            // No decrease along the step: converged if the step is at rounding level.
            bool small = true;
            for (std::size_t j = 0; j < p; ++j)
                small &= std::abs(step[j]) <= 1e-7 * (1 + std::abs(f.beta[j]));
            f.deviance = current;
            f.ok = small;
            return f;
        }
        f.beta.swap(next);
        f.mu.swap(next_mu);
        f.log_mu.swap(next_log_mu);
        current = value;
    }
    return f;
}
// Leverages diag(W^1/2 X (X'WX)^-1 X'W^1/2) = w_i |B x_i|^2 with B = L^-1 D^-1 from
// the factor of X'WX (D its root diagonal; detail::weighted_gram and
// detail::triangular_norms, `xt` being X' row-major, p x n); from
// weighted_leverage when the factor is ill-conditioned.
inline Vector fit_leverage(const Matrix &x, const Vector &xt, const Vector &w) {
    const std::size_t n = x.rows, p = x.cols;
    Matrix a(p, p);
    weighted_gram(x.values.data(), w.data(), nullptr, n, p, a.values.data(), nullptr);
    const Cholesky ch(a, 1e-12, false);
    bool conditioned = ch.ok;
    for (std::size_t j = 0; conditioned && j < p; ++j)
        conditioned = ch.l(j, j) * ch.l(j, j) > 1e-8;
    if (!conditioned)
        return weighted_leverage(x, w);
    // B row-major; column c of B solves L u = e_c / d_c.
    Vector b(p * p, 0.0), u(p);
    for (std::size_t c = 0; c < p; ++c) {
        for (std::size_t j = c; j < p; ++j) {
            double value = j == c ? 1 / ch.scale[c] : 0.0;
            for (std::size_t k = c; k < j; ++k)
                value -= ch.l(j, k) * u[k];
            u[j] = value / ch.l(j, j);
            b[j * p + c] = u[j];
        }
    }
    Vector h(n);
    triangular_norms(b.data(), xt.data(), n, p, h.data());
    for (std::size_t i = 0; i < n; ++i)
        h[i] *= w[i];
    return h;
}
// Mean and variance of the NB unit deviance d(z, m, phi) for z ~ NB(mean m,
// dispersion phi) (Poisson below phi = 1e-10). Counts up to 1000 are summed
// exactly; above, the pmf and deviance are smooth on the scale of a count and
// their sum is the integral from 1000.5 with the first Euler-Maclaurin
// correction, by 8-point Gauss-Legendre panels no wider than half a standard
// deviation or a quarter of their position (relative error about 1e-10).
inline std::pair<double, double> nb_deviance_moments(double m, double phi) {
    if (!(m > 0))
        return {0, 0};
    const bool poisson = !(phi >= 1e-10);
    const double k = poisson ? 0 : 1 / phi, sd = std::sqrt(m + (poisson ? 0 : phi * m * m));
    const double log_q = poisson ? std::log(m) : std::log(m) - std::log(k + m),
                 log_p0 = poisson ? -m : -k * std::log1p(m / k);
    const double tail = poisson ? 0 : 1 / std::log1p(k / m);
    const double upper = m + 40 * sd + 40 * tail + 50;
    const std::size_t discrete = std::min<std::size_t>(1000, static_cast<std::size_t>(upper));
    long double s0 = 0, s1 = 0, s2 = 0;
    double lp = log_p0;
    for (std::size_t z = 0; z <= discrete; ++z) {
        if (z > 0)
            lp += std::log((poisson ? 1.0 : (z - 1 + k)) / static_cast<double>(z)) + log_q;
        const double p = std::exp(lp);
        if (p > 0) {
            const double d = nb_unit_deviance(static_cast<double>(z), m, poisson ? 0 : phi);
            s0 += p;
            s1 += p * d;
            s2 += p * d * d;
        }
    }
    if (upper > discrete + 0.5) {
        // log Gamma(x + k) - log Gamma(k) - log Gamma(x + 1) for x >= 1000 by
        // Stirling's series; for large k in a form free of cancellation.
        const double lgamma_k = poisson || k >= 65 ? 0 : boost::math::lgamma(k);
        const auto log_gamma = [](double z) {
            return (z - 0.5) * std::log(z) - z + half_log_two_pi + stirling_correction(z);
        };
        const auto log_pmf = [&](double x) {
            const double factorial = log_gamma(x + 1);
            if (poisson)
                return x * log_q - m - factorial;
            const double rising = k >= 65
                                      ? x * std::log(x + k) + (k - 0.5) * std::log1p(x / k) - x +
                                            stirling_correction(x + k) - stirling_correction(k)
                                      : log_gamma(x + k) - lgamma_k;
            return rising - factorial + log_p0 + x * log_q;
        };
        static constexpr double nodes[4] = {0.1834346424956498, 0.5255324099163290,
                                            0.7966664774136267, 0.9602898564975363},
                                weights[4] = {0.3626837833783620, 0.3137066458778873,
                                              0.2223810344533745, 0.1012285362903763};
        // Euler-Maclaurin: the sum over z > Z is the integral from Z + 1/2 plus
        // f'(Z + 1/2) / 24 (and terms in f''' that are far smaller here).
        const double start = discrete + 0.5, h = 0.25;
        if (start >= m - 40 * sd) {
            const auto terms = [&](double x, long double &t0, long double &t1, long double &t2) {
                const double p = std::exp(log_pmf(x));
                const double d = p > 0 ? nb_unit_deviance(x, m, poisson ? 0 : phi) : 0;
                t0 = p;
                t1 = p * d;
                t2 = p * d * d;
            };
            long double a0, a1, a2, b0, b1, b2;
            terms(start + h, a0, a1, a2);
            terms(start - h, b0, b1, b2);
            s0 += (a0 - b0) / (2 * h) / 24;
            s1 += (a1 - b1) / (2 * h) / 24;
            s2 += (a2 - b2) / (2 * h) / 24;
        }
        double a = std::max(start, m - 40 * sd);
        while (a < upper) {
            const double b = std::min(upper, a + std::max(1.0, std::min(sd / 2, a / 4)));
            const double centre = (a + b) / 2, half = (b - a) / 2;
            for (int j = 0; j < 4; ++j)
                for (double sign : {-1.0, 1.0}) {
                    const double x = centre + sign * half * nodes[j], p = std::exp(log_pmf(x));
                    if (!(p > 0))
                        continue;
                    const double d = nb_unit_deviance(x, m, poisson ? 0 : phi),
                                 wp = weights[j] * half * p;
                    s0 += wp;
                    s1 += wp * d;
                    s2 += wp * d * d;
                }
            a = b;
        }
    }
    if (!(s0 > 0))
        return {0, 0};
    const double mean = static_cast<double>(s1 / s0);
    return {mean, std::max(0.0, static_cast<double>(s2 / s0) - mean * mean)};
}
// nb_deviance_moments for one dispersion over a range of means: tabulated on a
// grid of 64 points per decade of the mean (log moments interpolated by cubics
// through four points), computed directly outside it.
class DevianceMoments {
  public:
    DevianceMoments(double phi, double largest) : phi_(phi) {
        lo_ = std::log(1e-6);
        const double hi = std::log(std::max(largest, 1.0) * 1.01);
        const double step = std::log(10.0) / 64;
        const auto points = static_cast<std::size_t>(std::ceil((hi - lo_) / step)) + 4;
        step_ = step;
        log_mean_.resize(points);
        log_variance_.resize(points);
        inverse_scale_.resize(points);
        df_.resize(points);
        for (std::size_t g = 0; g < points; ++g) {
            const auto [e, v] = nb_deviance_moments(std::exp(lo_ + g * step), phi);
            log_mean_[g] = std::log(e);
            log_variance_[g] = std::log(v);
            inverse_scale_[g] = 2 * e / v;
            df_[g] = 2 * e * e / v;
        }
        hi_ = lo_ + (points - 3) * step;
    }
    // 1 / s = 2E / V and k = 2E^2 / V at mean m, from log m (cubics through four
    // points of the tables of those values, no exponentials); (0, 0) where the
    // moments vanish.
    std::pair<double, double> factors(double m, double log_m) const {
        if (!(log_m >= lo_ + step_) || !(log_m <= hi_)) {
            const auto [e, v] = nb_deviance_moments(m, phi_);
            return e > 0 && v > 0 ? std::pair{2 * e / v, 2 * e * e / v} : std::pair{0.0, 0.0};
        }
        const double t = (log_m - lo_) / step_;
        const auto i = static_cast<std::size_t>(t) - 1;
        const double u = t - static_cast<double>(i);
        const double w0 = -(u - 1) * (u - 2) * (u - 3) / 6, w1 = u * (u - 2) * (u - 3) / 2,
                     w2 = -u * (u - 1) * (u - 3) / 2, w3 = u * (u - 1) * (u - 2) / 6;
        const auto at = [&](const Vector &v) {
            return w0 * v[i] + w1 * v[i + 1] + w2 * v[i + 2] + w3 * v[i + 3];
        };
        return {at(inverse_scale_), at(df_)};
    }
    std::pair<double, double> operator()(double m) const {
        const double x = std::log(m);
        if (!(x >= lo_ + step_) || !(x <= hi_))
            return nb_deviance_moments(m, phi_);
        const double t = (x - lo_) / step_;
        const auto i = static_cast<std::size_t>(t) - 1; // nodes i .. i + 3 around t
        const double u = t - static_cast<double>(i);
        // Lagrange weights for nodes at 0, 1, 2, 3.
        const double w0 = -(u - 1) * (u - 2) * (u - 3) / 6, w1 = u * (u - 2) * (u - 3) / 2,
                     w2 = -u * (u - 1) * (u - 3) / 2, w3 = u * (u - 1) * (u - 2) / 6;
        const auto at = [&](const Vector &v) {
            return std::exp(w0 * v[i] + w1 * v[i + 1] + w2 * v[i + 2] + w3 * v[i + 3]);
        };
        return {at(log_mean_), at(log_variance_)};
    }

  private:
    double phi_, lo_, hi_, step_;
    Vector log_mean_, log_variance_, inverse_scale_, df_;
};
// R's lowess (Cleveland 1979) as its documentation describes it: local linear
// fits with tricube weights within the distance to the floor(f n)-th nearest
// neighbour, `iterations` robustness passes with bisquare weights cut at six
// times the median absolute residual (stopping when that is effectively zero),
// fits computed only at points more than `delta` beyond the last computed one
// and interpolated linearly between them. Fitted values in input order.
inline Vector r_lowess(const Vector &x_in, const Vector &y_in, double f, int iterations,
                       double delta) {
    const std::size_t n = x_in.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return x_in[a] < x_in[b]; });
    Vector x(n), y(n), fit(n), robustness(n, 1.0), residual(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = x_in[order[i]];
        y[i] = y_in[order[i]];
    }
    if (n < 2)
        return y_in;
    const auto ns = std::clamp<std::size_t>(static_cast<std::size_t>(f * n + 1e-7), 2, n);
    const double range = x[n - 1] - x[0];
    Vector w(n);
    // The local fit at x[i] over the window [left, right]; false when no weight.
    const auto local = [&](std::size_t i, std::size_t left, std::size_t right, double &value) {
        const double h = std::max(x[i] - x[left], x[right] - x[i]), h9 = 0.999 * h, h1 = 0.001 * h;
        double total = 0;
        std::size_t j = left, end = left;
        for (; j < n; ++j) {
            const double r = std::abs(x[j] - x[i]);
            w[j] = 0;
            if (r <= h9) {
                const double u = r / h, t = 1 - u * u * u;
                w[j] = (r <= h1 ? 1 : t * t * t) * robustness[j];
                total += w[j];
                end = j;
            } else if (x[j] > x[i])
                break;
        }
        if (!(total > 0))
            return false;
        for (std::size_t k = left; k <= end; ++k)
            w[k] /= total;
        if (h > 0) {
            double a = 0;
            for (std::size_t k = left; k <= end; ++k)
                a += w[k] * x[k];
            double b = x[i] - a, c = 0;
            for (std::size_t k = left; k <= end; ++k)
                c += w[k] * (x[k] - a) * (x[k] - a);
            if (std::sqrt(c) > 0.001 * range) {
                b /= c;
                for (std::size_t k = left; k <= end; ++k)
                    w[k] *= b * (x[k] - a) + 1;
            }
        }
        value = 0;
        for (std::size_t k = left; k <= end; ++k)
            value += w[k] * y[k];
        return true;
    };
    for (int pass = 0; pass <= iterations; ++pass) {
        std::size_t left = 0, right = ns - 1, i = 0, last = 0;
        bool started = false;
        for (;;) {
            while (right < n - 1 && x[i] - x[left] > x[right + 1] - x[i]) {
                ++left;
                ++right;
            }
            double value = y[i];
            if (!local(i, left, right, value))
                value = y[i];
            fit[i] = value;
            if (started && last + 1 < i) {
                const double denominator = x[i] - x[last];
                for (std::size_t j = last + 1; j < i; ++j) {
                    const double alpha = (x[j] - x[last]) / denominator;
                    fit[j] = alpha * fit[i] + (1 - alpha) * fit[last];
                }
            }
            started = true;
            last = i;
            const double cut = x[last] + delta;
            for (i = last + 1; i < n; ++i) {
                if (x[i] > cut)
                    break;
                if (x[i] == x[last]) {
                    fit[i] = fit[last];
                    last = i;
                }
            }
            i = std::max(last + 1, i - 1);
            if (last >= n - 1)
                break;
        }
        if (pass == iterations)
            break;
        for (std::size_t k = 0; k < n; ++k)
            residual[k] = std::abs(y[k] - fit[k]);
        const double cmad = 6 * median(residual);
        double mean_abs = 0;
        for (double v : y)
            mean_abs += std::abs(v);
        if (cmad < 1e-7 * mean_abs / n)
            break;
        const double c9 = 0.999 * cmad, c1 = 0.001 * cmad;
        for (std::size_t k = 0; k < n; ++k) {
            const double r = residual[k];
            const double u = 1 - (r / cmad) * (r / cmad);
            robustness[k] = r <= c1 ? 1 : r > c9 ? 0 : u * u;
        }
    }
    Vector out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[order[i]] = fit[i];
    return out;
}
// limma's weightedLowess with one iteration (no robustness weights), as its
// documentation describes it: at each anchor, a local linear regression over the
// smallest set of nearest points whose prior weights sum to at least `span` of
// the total, weighted by prior weights times tricube weights of distance relative
// to the farthest point of that set. Anchors: the smallest x, then each next x
// more than `delta` beyond the previous anchor, and the largest x; values between
// anchors are interpolated linearly. Fitted values in input order.
inline Vector weighted_lowess(const Vector &x_in, const Vector &y_in, const Vector &w_in,
                              double span, double delta) {
    const std::size_t n = x_in.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return x_in[a] < x_in[b]; });
    Vector x(n), y(n), w(n), fit(n);
    double total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = x_in[order[i]];
        y[i] = y_in[order[i]];
        w[i] = w_in[order[i]];
        total += w[i];
    }
    if (n == 0)
        return {};
    const double needed = span * total;
    const auto local = [&](std::size_t i) {
        // Grow the window from x[i] by the nearer side until the weight suffices.
        std::size_t left = i, right = i;
        double weight = w[i];
        while (weight < needed && (left > 0 || right + 1 < n)) {
            const bool go_left =
                right + 1 >= n || (left > 0 && x[i] - x[left - 1] <= x[right + 1] - x[i]);
            if (go_left)
                weight += w[--left];
            else
                weight += w[++right];
        }
        // Points tied with the window's ends belong to it.
        while (left > 0 && x[left - 1] == x[left])
            --left;
        while (right + 1 < n && x[right + 1] == x[right])
            ++right;
        const double h = std::max(x[i] - x[left], x[right] - x[i]);
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (std::size_t k = left; k <= right; ++k) {
            double v = w[k];
            if (h > 0) {
                const double r = std::abs(x[k] - x[i]) / h, t = 1 - r * r * r;
                v *= r < 1 ? t * t * t : 0;
            }
            sw += v;
            sx += v * x[k];
            sy += v * y[k];
            sxx += v * x[k] * x[k];
            sxy += v * x[k] * y[k];
        }
        if (!(sw > 0))
            return y[i];
        const double mx = sx / sw, my = sy / sw, vx = sxx / sw - mx * mx;
        return vx > 1e-12 * (1 + mx * mx) ? my + (sxy / sw - mx * my) / vx * (x[i] - mx) : my;
    };
    std::size_t last = 0;
    fit[0] = local(0);
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] == x[last]) {
            fit[i] = fit[last];
            last = i;
            continue;
        }
        if (x[i] - x[last] <= delta && i + 1 < n)
            continue;
        // An anchor at i (and every point between it and the last interpolated).
        fit[i] = local(i);
        for (std::size_t j = last + 1; j < i; ++j) {
            const double alpha = (x[j] - x[last]) / (x[i] - x[last]);
            fit[j] = alpha * fit[i] + (1 - alpha) * fit[last];
        }
        last = i;
    }
    Vector out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[order[i]] = fit[i];
    return out;
}
// log Gamma, reentrant.
inline double log_gamma(double x) {
    int sign = 0;
    return ::lgamma_r(x, &sign);
}
// log density of the F distribution with df d1 and d2 at x, given log Gamma(d1/2)
// and log Gamma(d2/2).
inline double log_f_density(double x, double d1, double d2, double lgamma_half_d1,
                            double lgamma_half_d2) {
    const double lbeta = lgamma_half_d1 + lgamma_half_d2 - log_gamma((d1 + d2) / 2);
    return d1 / 2 * std::log(d1 / d2) + (d1 / 2 - 1) * std::log(x) -
           (d1 + d2) / 2 * std::log1p(d1 * x / d2) - lbeta;
}
struct SqueezedVariances {
    Vector prior, posterior;
    double df = nan;
};
// The largest prior df edgeR 4.6.2 reports: its value when the likelihood of the
// prior df keeps increasing (black-box runs; EDGER_QL_SPEC.md 6.5).
inline constexpr double largest_prior_df = 8134.84478;
// Empirical Bayes moderation of variances s2 on unequal df d with an abundance
// covariate, as limma's squeezeVar documents it for unequal df
// (fitFDistUnequalDF1): log-moment terms weighted by inverse trigamma, their
// weighted LOWESS trend in the covariate, the prior scale tied to the trend and
// the prior df d0, and d0 by profile maximum likelihood of the scaled F model.
inline SqueezedVariances squeeze_variances(const Vector &s2, const Vector &d,
                                           const Vector &covariate, double delta) {
    const std::size_t g = s2.size();
    SqueezedVariances out;
    out.prior.assign(g, nan);
    out.posterior.assign(g, nan);
    if (g == 0)
        return out;
    const double floor = 1e-12 * median(s2);
    Vector x(g), e(g), w(g);
    // A gene without residual df has no weight.
    for (std::size_t i = 0; i < g; ++i) {
        x[i] = std::max(s2[i], floor);
        const double half = d[i] / 2;
        if (half > 0) {
            e[i] = std::log(x[i]) - boost::math::digamma(half) + std::log(half);
            w[i] = 1 / boost::math::trigamma(half);
        }
    }
    const double span =
        g <= 500 ? 1 : 0.3 + 0.7 * std::pow(500.0 / static_cast<double>(g), 1.0 / 3);
    const auto trend = weighted_lowess(covariate, e, w, span, delta);
    const auto log_scale = [&](double d0, std::size_t i) {
        return trend[i] + boost::math::digamma(d0 / 2) - std::log(d0 / 2);
    };
    Vector lgamma_half(g);
    for (std::size_t i = 0; i < g; ++i)
        if (d[i] > 0)
            lgamma_half[i] = log_gamma(d[i] / 2);
    const auto objective = [&](double log_d0) {
        const double d0 = std::exp(log_d0), shift = boost::math::digamma(d0 / 2) - std::log(d0 / 2),
                     lgamma_d0 = log_gamma(d0 / 2);
        long double total = 0;
        for (std::size_t i = 0; i < g; ++i)
            if (d[i] > 0) {
                const double ls = trend[i] + shift;
                total +=
                    log_f_density(x[i] / std::exp(ls), d[i], d0, lgamma_half[i], lgamma_d0) - ls;
            }
        return -static_cast<double>(total);
    };
    out.df = std::exp(brent_minimum(objective, std::log(1e-3), std::log(largest_prior_df), 1e-10));
    for (std::size_t i = 0; i < g; ++i) {
        out.prior[i] = std::exp(log_scale(out.df, i));
        out.posterior[i] =
            (out.df * out.prior[i] + std::max(0.0, d[i]) * s2[i]) / (out.df + std::max(0.0, d[i]));
    }
    return out;
}
// One-group NB fit at dispersion phi (Fisher scoring): log of the rate per unit
// of exposure (the exponential of the offset), means exp(beta) x exposure.
inline double one_group_rate(const Vector &y, const Vector &exposure, double phi) {
    double sy = 0, so = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        sy += y[i];
        so += exposure[i];
    }
    double beta = std::log(sy / so);
    for (int it = 0; it < 100; ++it) {
        double score = 0, information = 0;
        const double rate = std::exp(beta);
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double mu = rate * exposure[i], inverse = 1 / (1 + phi * mu);
            score += (y[i] - mu) * inverse;
            information += mu * inverse;
        }
        const double step = score / information;
        beta += step;
        if (std::abs(step) < 1e-11)
            break;
    }
    return beta;
}
// Fits at a fixed NB dispersion (Poisson at 0).
inline FitOptions fixed_dispersion_options(double phi, const Vector &offset,
                                           const std::shared_ptr<const DesignSummary> &design) {
    FitOptions o;
    o.offset = offset;
    o.design = design;
    o.retain_observations = true;
    if (phi > 0) {
        o.family = Family::negative_binomial;
        o.dispersion = Dispersion::fixed;
        o.theta = 1 / phi;
        o.theta_min = std::min(o.theta_min, o.theta);
        o.theta_max = std::max(o.theta_max, o.theta * 2);
    } else
        o.family = Family::poisson;
    return o;
}
} // namespace detail

// The pipeline on counts (samples x genes) and a design of full column rank,
// testing `hypothesis` (one or more coefficients, or contrasts):
// 1. filterByExpr (QlOptions), library sizes then recomputed over kept genes;
// 2. TMM factors; offsets log(library size x factor);
// 3. average log CPM: one-group NB fits at dispersion 0.05 of the counts plus
//    2 x library / mean library, with offsets log(library + 4 x library / mean);
// 4. the NB dispersion phi: Cox-Reid adjusted profile likelihood summed over the
//    ceil(t G) genes of highest average log CPM, t = 0.02 + 0.98 (20 / (G sqrt(n - p)))^(1/3)
//    (1 when G sqrt(n - p) <= 20), maximized by Brent's method in phi^(1/4) on
//    [0, 4^(1/4)];
// 5. fits at phi / a; per observation the unit deviance over s and df (1 - h) k,
//    with k = 2 E^2 / V and s = V / (2 E) the moments of the NB unit deviance at
//    mean mu / a and dispersion phi, h the leverage; s2 = sum / sum. The scale a
//    is max(1, the 0.9 quantile of lowess(average log CPM, s2^(1/4), f = 0.5)^4),
//    found at a = 1 and again at that a;
// 6. s2 moderated against average log CPM (detail::squeeze_variances);
// 7. F = (deviance of the reduced fit - deviance of the full fit) / (q s2_post)
//    on q and df_prior + df_residual_adj df; BH over the kept genes;
// 8. log fold changes from fits of the counts plus prior_count x library / mean
//    library, offsets log(library + 2 x that), at phi / a (one-row hypotheses).
inline QlResult edger_ql(const Matrix &design, const SparseCSC &counts,
                         const Hypothesis &hypothesis, const QlOptions &o = {}) {
    design.validate();
    counts.validate();
    const std::size_t n = counts.rows, genes = counts.cols, p = design.cols;
    require(design.rows == n && hypothesis.contrasts.cols == p, "edger_ql dimensions mismatch");
    require(n > p, "edger_ql needs more samples than design columns");
    require(o.threads > 0, "threads must be positive");
    require(o.prior_count >= 0 && std::isfinite(o.prior_count),
            "prior_count must be finite and nonnegative");
    for (double v : counts.values)
        require(v >= 0 && std::isfinite(v) && std::floor(v) == v,
                "edger_ql requires nonnegative integer counts");
    QlResult out;
    out.genes.resize(genes);
    out.tested_df = hypothesis.df();
    // Library sizes and filtering.
    out.library_sizes.assign(n, 0);
    for (std::size_t j = 0; j < genes; ++j)
        counts.for_each(j, [&](std::size_t i, double v) {
            out.library_sizes[i] += v;
            out.genes[j].total += v;
        });
    for (double v : out.library_sizes)
        require(v > 0, "every sample needs a positive library size");
    std::vector<std::size_t> kept;
    if (o.filter) {
        const auto h = detail::design_leverage(design);
        double minimum = 1 / *std::max_element(h.begin(), h.end());
        if (minimum > o.large_n)
            minimum = o.large_n + (minimum - o.large_n) * o.min_prop;
        out.minimum_group = minimum;
        const double cutoff = o.min_count / detail::median(out.library_sizes) * 1e6;
        for (std::size_t j = 0; j < genes; ++j) {
            std::size_t above = 0, stored = 0;
            counts.for_each(j, [&](std::size_t i, double v) {
                ++stored;
                above += v * 1e6 / out.library_sizes[i] >= cutoff;
            });
            if (cutoff <= 0)
                above += n - stored;
            if (static_cast<double>(above) >= minimum - 1e-14 &&
                out.genes[j].total >= o.min_total_count)
                kept.push_back(j);
        }
    } else
        for (std::size_t j = 0; j < genes; ++j)
            kept.push_back(j);
    const std::size_t g = kept.size();
    out.kept = g;
    require(g >= 2, "fewer than two genes pass the expression filter");
    for (auto j : kept) {
        out.genes[j].kept = true;
        out.genes[j].status = "ok";
    }
    // Dense counts of the kept genes, sample-major per gene.
    Matrix y(g, n);
    out.filtered_library_sizes.assign(n, 0);
    for (std::size_t r = 0; r < g; ++r)
        counts.for_each(kept[r], [&](std::size_t i, double v) {
            y(r, i) = v;
            out.filtered_library_sizes[i] += v;
        });
    for (double v : out.filtered_library_sizes)
        require(v > 0, "a sample has no reads in the kept genes");
    // TMM.
    out.norm_factors.assign(n, 1);
    if (!o.norm_factors.empty()) {
        require(o.norm_factors.size() == n, "norm_factors needs one value per sample");
        out.norm_factors = o.norm_factors;
    } else if (o.tmm) {
        const auto &lib = out.filtered_library_sizes;
        Matrix t(n, g); // sample-major
        for (std::size_t r = 0; r < g; ++r)
            for (std::size_t i = 0; i < n; ++i)
                t(i, r) = y(r, i);
        Vector upper(n);
        for (std::size_t i = 0; i < n; ++i) {
            Vector share(g);
            for (std::size_t r = 0; r < g; ++r)
                share[r] = t(i, r) / lib[i];
            upper[i] = detail::quantile7(std::move(share), 0.75);
        }
        const double mean_upper = std::accumulate(upper.begin(), upper.end(), 0.0) / n;
        std::size_t reference = 0;
        for (std::size_t i = 1; i < n; ++i)
            if (std::abs(upper[i] - mean_upper) < std::abs(upper[reference] - mean_upper))
                reference = i;
        // log2 of each share, as A takes it: (log2(y_k / N_k) + log2(y_r / N_r)) / 2.
        Vector reference_share(g);
        for (std::size_t r = 0; r < g; ++r)
            reference_share[r] =
                t(reference, r) > 0 ? std::log2(t(reference, r) / lib[reference]) : 0;
        double log_mean = 0;
        for (std::size_t k = 0; k < n; ++k) {
            Vector m, a, v;
            m.reserve(g);
            a.reserve(g);
            v.reserve(g);
            const double nk = lib[k], nr = lib[reference];
            for (std::size_t r = 0; r < g; ++r) {
                const double ok = t(k, r), rr = t(reference, r);
                if (ok > 0 && rr > 0) {
                    m.push_back(std::log2((ok / nk) / (rr / nr)));
                    a.push_back((std::log2(ok / nk) + reference_share[r]) / 2);
                    v.push_back((nk - ok) / (nk * ok) + (nr - rr) / (nr * rr));
                }
            }
            const double count = static_cast<double>(m.size());
            const double lo_m = std::floor(count * o.logratio_trim) + 1, hi_m = count + 1 - lo_m,
                         lo_a = std::floor(count * o.sum_trim) + 1, hi_a = count + 1 - lo_a;
            const auto keep_m = detail::rank_window(m, lo_m, hi_m),
                       keep_a = detail::rank_window(a, lo_a, hi_a);
            double numerator = 0, denominator = 0;
            for (std::size_t r = 0; r < m.size(); ++r)
                if (keep_m[r] && keep_a[r]) {
                    numerator += m[r] / v[r];
                    denominator += 1 / v[r];
                }
            out.norm_factors[k] = denominator > 0 ? std::exp2(numerator / denominator) : 1;
            log_mean += std::log(out.norm_factors[k]);
        }
        log_mean /= n;
        for (auto &f : out.norm_factors)
            f /= std::exp(log_mean);
    }
    for (double f : out.norm_factors)
        require(f > 0 && std::isfinite(f), "normalization factors must be positive");
    out.effective_library_sizes.resize(n);
    Vector offset(n);
    double mean_library = 0;
    for (std::size_t i = 0; i < n; ++i) {
        out.effective_library_sizes[i] = out.filtered_library_sizes[i] * out.norm_factors[i];
        offset[i] = std::log(out.effective_library_sizes[i]);
        mean_library += out.effective_library_sizes[i] / n;
    }
    // Counts plus prior counts c x library / mean library, offsets log(library + 2 x prior).
    const auto with_prior = [&](double c, Vector &prior, Vector &prior_offset) {
        prior.resize(n);
        prior_offset.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            prior[i] = c * out.effective_library_sizes[i] / mean_library;
            prior_offset[i] = std::log(out.effective_library_sizes[i] + 2 * prior[i]);
        }
    };
    // Average log CPM.
    Vector ave(g);
    {
        Vector prior, prior_offset, exposure(n);
        with_prior(2, prior, prior_offset);
        for (std::size_t i = 0; i < n; ++i)
            exposure[i] = std::exp(prior_offset[i]);
        parallel_for(g, o.threads, [&](auto r) {
            Vector v(n);
            for (std::size_t i = 0; i < n; ++i)
                v[i] = y(r, i) + prior[i];
            ave[r] = (detail::one_group_rate(v, exposure, 0.05) + std::log(1e6)) / std::log(2.0);
            out.genes[kept[r]].ave_log_cpm = ave[r];
        });
    }
    const auto row = [&](std::size_t r) { return Vector(&y.values[r * n], &y.values[r * n] + n); };
    const auto summary = prepare_design(design);
    const double residual_df = static_cast<double>(n - p);
    // The NB dispersion.
    if (std::isnan(o.dispersion)) {
        const double size = static_cast<double>(g) * std::sqrt(residual_df);
        out.top_proportion = size <= 20 ? 1 : 0.02 + 0.98 * std::cbrt(20 / size);
        out.top_genes = std::min<std::size_t>(
            g, static_cast<std::size_t>(std::ceil(out.top_proportion * static_cast<double>(g))));
        std::vector<std::size_t> order(g);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](auto a, auto b) { return ave[a] > ave[b]; });
        std::vector<std::size_t> top;
        for (std::size_t k = 0; k < out.top_genes; ++k) {
            const auto r = order[k];
            double total = 0;
            for (std::size_t i = 0; i < n; ++i)
                total += y(r, i);
            if (total >= 5)
                top.push_back(r);
        }
        require(!top.empty(), "no genes to estimate the dispersion from");
        Vector apl(top.size());
        std::vector<Vector> starts(top.size());
        // The count-only terms of each gene's likelihood, once (they cache their
        // dispersion term, so each gene has its own).
        std::vector<detail::CountTerms> terms;
        terms.reserve(top.size());
        for (auto r : top)
            terms.emplace_back(row(r), Vector(n, 1));
        const Vector ones(n, 1);
        const detail::Cholesky least_squares(detail::crossprod(design, ones), 1e-12, false);
        require(least_squares.ok, "singular design for dispersion estimation");
        const auto objective = [&](double s) {
            const double phi = s * s * s * s;
            parallel_for(top.size(), o.threads, [&](auto k) {
                const auto v = row(top[k]);
                auto fast = detail::nb_fixed_fit(design, v, offset, phi, starts[k], least_squares,
                                                 60, false);
                if (!fast.ok) {
                    auto options = detail::fixed_dispersion_options(phi, offset, summary);
                    const auto f = fit(design, v, options);
                    if (!f.ok() || f.fitted.size() != n) {
                        apl[k] = nan;
                        return;
                    }
                    fast.beta = f.coefficients;
                    fast.mu = f.fitted;
                }
                starts[k] = fast.beta;
                const auto &fitted = fast.mu;
                const double theta = phi > 0 ? 1 / phi : std::numeric_limits<double>::infinity();
                const auto family = phi > 0 ? Family::negative_binomial : Family::poisson;
                double value = detail::likelihood(v, fitted, ones, family, theta, terms[k]);
                // The Cox-Reid term -log det(X'WX)/2, W = mu / (1 + phi mu).
                Vector w(n);
                for (std::size_t i = 0; i < n; ++i)
                    w[i] = fitted[i] / (1 + phi * fitted[i]);
                Matrix information(p, p);
                detail::weighted_gram(design.values.data(), w.data(), nullptr, n, p,
                                      information.values.data(), nullptr);
                const detail::Cholesky ch(information, 1e-12, false);
                if (!ch.ok) {
                    apl[k] = nan;
                    return;
                }
                double log_det = 0;
                for (std::size_t j = 0; j < p; ++j)
                    log_det += 2 * (std::log(ch.scale[j]) + std::log(ch.l(j, j)));
                apl[k] = value - 0.5 * log_det;
            });
            long double total = 0;
            for (double v : apl) {
                // Every candidate must use exactly the same dispersion genes.
                // An incomplete objective is invalid, never a smaller gene set.
                if (!std::isfinite(v))
                    return std::numeric_limits<double>::infinity();
                total += v;
            }
            return -static_cast<double>(total);
        };
        const double s = detail::brent_minimum(objective, 0, std::pow(4.0, 0.25), 1e-5);
        require(std::isfinite(objective(s)), "dispersion objective failed for the fixed gene set");
        out.dispersion = s * s * s * s;
    } else {
        require(o.dispersion >= 0 && std::isfinite(o.dispersion), "invalid dispersion");
        out.dispersion = o.dispersion;
    }
    const double phi = out.dispersion;
    // Quasi-dispersions at scale a.
    const auto tested =
        hypothesis.df() == 1 ? tested_design(design, hypothesis) : TestedDesign{design, 0};
    const auto tested_summary = prepare_design(tested.x);
    double largest = 0;
    for (double v : y.values)
        largest = std::max(largest, v);
    Vector deviance(g, nan), deviance_adj(g, nan), df_adj(g, nan), s2(g, nan), unshrunk(g, nan);
    std::vector<Vector> coefficients(g);
    std::vector<unsigned char> failed(g, 0);
    const detail::Cholesky tested_least_squares(detail::crossprod(tested.x, Vector(n, 1)), 1e-12,
                                                false);
    Vector tested_transpose(p * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            tested_transpose[j * n + i] = tested.x(i, j);
    // The moments at dispersion phi for means up to the largest a = 1 needs (a >= 1).
    const detail::DevianceMoments moments(phi, 4 * (largest + 10));
    const auto pass = [&](double a) {
        const double working = phi / a;
        const auto options = detail::fixed_dispersion_options(working, offset, tested_summary);
        parallel_for(g, o.threads, [&](auto r) {
            const auto v = row(r);
            auto f = detail::nb_fixed_fit(tested.x, v, offset, working, coefficients[r],
                                          tested_least_squares, 60, false);
            if (!f.ok) {
                const auto general = fit(tested.x, v, options);
                f.ok = general.ok() && general.fitted.size() == n;
                f.beta = general.coefficients;
                f.mu = general.fitted;
                f.log_mu.resize(f.mu.size());
                for (std::size_t i = 0; i < f.mu.size(); ++i)
                    f.log_mu[i] = std::log(f.mu[i]);
            }
            failed[r] = !f.ok;
            if (failed[r]) {
                deviance[r] = deviance_adj[r] = df_adj[r] = s2[r] = unshrunk[r] = nan;
                return;
            }
            unshrunk[r] = hypothesis.df() == 1 ? f.beta[tested.column] : nan;
            coefficients[r] = f.beta;
            Vector w(n);
            for (std::size_t i = 0; i < n; ++i)
                w[i] = f.mu[i] / (1 + working * f.mu[i]);
            const auto leverage = detail::fit_leverage(tested.x, tested_transpose, w);
            const double log_a = std::log(a);
            double dev = 0, adj = 0, df = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double h = std::clamp(leverage[i], 0.0, 1.0);
                const double d = detail::nb_unit_deviance(v[i], f.mu[i], working);
                dev += d;
                const auto [inverse_scale, k] = moments.factors(f.mu[i] / a, f.log_mu[i] - log_a);
                adj += d * inverse_scale;
                df += (1 - h) * k;
            }
            deviance[r] = dev;
            deviance_adj[r] = adj;
            df_adj[r] = df;
            s2[r] = df > 0 ? adj / df : 0;
        });
    };
    const auto average = [&]() {
        Vector x, v;
        for (std::size_t r = 0; r < g; ++r)
            if (!failed[r]) {
                x.push_back(ave[r]);
                v.push_back(std::pow(s2[r], 0.25));
            }
        require(!x.empty(), "no genes could be fitted for quasi-dispersion estimation");
        const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
        auto smooth = detail::r_lowess(x, v, 0.5, 3, 0.01 * (*hi - *lo));
        for (auto &s : smooth)
            s = s * s * s * s;
        return std::max(1.0, detail::quantile7(smooth, 0.9));
    };
    pass(1);
    double a = average();
    if (a > 1) {
        pass(a);
        a = average();
        pass(a);
    }
    out.scale = a;
    const double working = phi / a;
    // Moderation.
    std::vector<std::size_t> usable;
    for (std::size_t r = 0; r < g; ++r)
        if (!failed[r])
            usable.push_back(r);
    require(usable.size() >= 2, "fewer than two genes could be fitted");
    Vector us2, udf, uave;
    for (auto r : usable) {
        us2.push_back(s2[r]);
        udf.push_back(df_adj[r]);
        uave.push_back(ave[r]);
    }
    const auto [lo, hi] = std::minmax_element(uave.begin(), uave.end());
    const auto squeezed = detail::squeeze_variances(us2, udf, uave, (*hi - *lo) / 200);
    out.df_prior = squeezed.df;
    // Tests and fold changes.
    const auto reduced = reduced_design(design, hypothesis);
    const auto reduced_summary = prepare_design(reduced);
    const auto reduced_options = detail::fixed_dispersion_options(working, offset, reduced_summary);
    Vector prior, prior_offset;
    with_prior(o.prior_count, prior, prior_offset);
    auto unit_columns = hypothesis.unit_columns();
    std::sort(unit_columns.begin(), unit_columns.end());
    const detail::Cholesky reduced_least_squares(detail::crossprod(reduced, Vector(n, 1)), 1e-12,
                                                 false);
    const double q = static_cast<double>(hypothesis.df());
    Vector pvalues(g, nan);
    parallel_for(usable.size(), o.threads, [&](auto k) {
        const auto r = usable[k];
        auto &gene = out.genes[kept[r]];
        gene.deviance = deviance[r];
        gene.deviance_adj = deviance_adj[r];
        gene.df_residual_adj = df_adj[r];
        gene.s2 = s2[r];
        gene.s2_prior = squeezed.prior[k];
        gene.s2_post = squeezed.posterior[k];
        gene.df_total = squeezed.df + df_adj[r];
        gene.unshrunk_coefficient = unshrunk[r];
        const auto v = row(r);
        // From the full fit without the tested coefficients.
        Vector start;
        if (!unit_columns.empty())
            for (std::size_t j = 0, next = 0; j < p; ++j) {
                if (next < unit_columns.size() && unit_columns[next] == j) {
                    ++next;
                    continue;
                }
                start.push_back(coefficients[r][j]);
            }
        auto null = detail::nb_fixed_fit(reduced, v, offset, working, start, reduced_least_squares);
        if (!null.ok && !start.empty())
            null = detail::nb_fixed_fit(reduced, v, offset, working, {}, reduced_least_squares);
        if (!null.ok) {
            const auto general = fit(reduced, v, reduced_options);
            null.ok = general.ok() && general.fitted.size() == n;
            null.mu = general.fitted;
            null.deviance = 0;
            if (null.ok)
                for (std::size_t i = 0; i < n; ++i)
                    null.deviance += detail::nb_unit_deviance(v[i], null.mu[i], working);
        }
        if (!null.ok) {
            gene.status = "fit_failed";
            return;
        }
        const double null_deviance = null.deviance;
        const double lr = std::max(0.0, null_deviance - deviance[r]);
        gene.f = lr / (q * gene.s2_post);
        if (gene.s2_post > 0 && gene.df_total > 0 && std::isfinite(gene.f)) {
            boost::math::fisher_f_distribution<double> dist(q, gene.df_total);
            gene.p_value = boost::math::cdf(boost::math::complement(dist, gene.f));
            pvalues[r] = gene.p_value;
        }
        if (hypothesis.df() == 1) {
            Vector shifted(n);
            for (std::size_t i = 0; i < n; ++i)
                shifted[i] = v[i] + prior[i];
            auto shrunk = detail::nb_fixed_fit(tested.x, shifted, prior_offset, working,
                                               coefficients[r], tested_least_squares, 60, false);
            if (!shrunk.ok) {
                // The counts plus prior counts are not integers, which fit() needs.
                const auto map =
                    detail::nb_map_fit(tested.x, shifted, prior_offset, std::max(working, 1e-10),
                                       Vector(p, 0), coefficients[r]);
                shrunk.ok = map.ok;
                shrunk.beta = map.beta;
            }
            if (shrunk.ok) {
                gene.coefficient = shrunk.beta[tested.column];
                gene.log_fc = gene.coefficient / std::log(2.0);
            }
        }
    });
    for (std::size_t r = 0; r < g; ++r)
        if (failed[r])
            out.genes[kept[r]].status = "fit_failed";
    const auto fdr = adjust_pvalues(pvalues);
    for (std::size_t r = 0; r < g; ++r)
        if (std::isfinite(pvalues[r]))
            out.genes[kept[r]].fdr = fdr[r];
    return out;
}
inline QlResult edger_ql(const Matrix &design, const SparseCSC &counts, std::size_t coefficient,
                         const QlOptions &o = {}) {
    require(coefficient < design.cols, "coefficient out of range");
    return edger_ql(design, counts, Hypothesis::coefficient(design.cols, coefficient), o);
}
} // namespace rapidglm
