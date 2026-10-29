#pragma once
// Pair nulls with non-targeting controls started from additive totals
// (SceptreOptions::null_start = NullStart::refine; METHODS.md, "Mergeable null
// start").
//
// A pair's cells are the NT cells and the target's treated cells. Its null is
// SCEPTRE's response model fitted on those cells: Poisson coefficients, then
// theta by conditional ML at the Poisson means. Today's cold fit starts that
// fit from scratch for every pair. Here each response first gets a reference,
// the same null fitted once on all retained cells, and the NT cells' sums of
// the Poisson score and information and of the theta score terms at it. A
// pair adds the same sums over its target's cells and takes one Fisher-scoring
// step (and one linearized Newton step in log theta) from the reference: the
// one-step estimator, a fixed function of two sets of totals, and of the
// pair's cells only, whichever of them carry the target's label. Exact Newton
// passes over the pair's cells then refine it to today's convergence criteria,
// with step halving; a pair that does not converge is refitted cold by the
// caller.
#include "model.hpp"
#include "resampling.hpp"
#include "vmath.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace rapidglm::detail::mergeable {
// The nonzero entries of every design row, by increasing column.
struct Rows {
    std::vector<std::size_t> start;
    std::vector<std::uint32_t> column;
    Vector value;
    Rows() = default;
    explicit Rows(const Matrix &x) {
        start.reserve(x.rows + 1);
        for (std::size_t i = 0; i < x.rows; ++i) {
            start.push_back(column.size());
            for (std::size_t j = 0; j < x.cols; ++j)
                if (x(i, j) != 0) {
                    column.push_back(static_cast<std::uint32_t>(j));
                    value.push_back(x(i, j));
                }
        }
        start.push_back(column.size());
    }
    double dot(std::size_t i, const Vector &beta) const {
        double e = 0;
        for (auto s = start[i]; s < start[i + 1]; ++s)
            e += value[s] * beta[column[s]];
        return e;
    }
    void add(std::size_t i, double v, Vector &out) const {
        for (auto s = start[i]; s < start[i + 1]; ++s)
            out[column[s]] += v * value[s];
    }
    // The lower triangle of w x_i x_i' (row >= column), as detail::crossprod.
    void add_outer(std::size_t i, double w, Matrix &a) const {
        if (w == 0)
            return;
        const std::size_t first = start[i], last = start[i + 1], p = a.cols;
        for (std::size_t s = first; s < last; ++s) {
            const double v = w * value[s];
            double *row = &a.values[column[s] * p];
            for (std::size_t t = first; t <= s; ++t)
                row[column[t]] += v * value[t];
        }
    }
};
inline void mirror(Matrix &a) {
    for (std::size_t j = 0; j < a.rows; ++j)
        for (std::size_t k = 0; k < j; ++k)
            a(k, j) = a(j, k);
}
// The retained cells of a sceptre() call, by position.
struct Cells {
    const Rows *rows = nullptr;
    const Vector *offset = nullptr, *weight = nullptr; // empty: zeros, ones
    std::size_t p = 0;
    double off(std::size_t i) const {
        return offset->empty() ? 0.0 : (*offset)[i];
    }
    double a(std::size_t i) const {
        return weight->empty() ? 1.0 : (*weight)[i];
    }
};
// The terms of one cell (count v, mean m, prior weight a > 0) at theta t,
// added to the totals: the Poisson score a (v - m) x and information a m x x',
// the conditional theta score and curvature of estimate_theta, and the
// derivative of the log-theta score in beta, a t m (v - m)/(t + m)^2 x.
struct Totals {
    Vector score, cross;
    Matrix information; // lower triangle until mirrored
    double theta_score = 0, theta_curvature = 0, ylogy = 0;
    explicit Totals(std::size_t p) : score(p), cross(p), information(p, p) {}
};
// The response's reference and the NT cells' totals at it.
struct Reference {
    bool ok = false;
    FitResult fit; // today's null on all retained cells (retained means)
    Totals nt{0};
    // digamma(y + t) - digamma(t) and trigamma(y + t) - trigamma(t) for y <= 64.
    std::array<double, 65> d1{}, d2{};
    void add(const Cells &c, std::size_t i, double v, Totals &t) const {
        const double ai = c.a(i);
        if (!(ai > 0))
            return;
        const double m = fit.fitted[i], theta = fit.theta;
        c.rows->add(i, ai * (v - m), t.score);
        c.rows->add_outer(i, ai * m, t.information);
        double g1, g2;
        if (v <= 64) {
            g1 = d1[static_cast<std::size_t>(v)];
            g2 = d2[static_cast<std::size_t>(v)];
        } else {
            g1 = detail::digamma(v + theta) - detail::digamma(theta);
            g2 = detail::trigamma(v + theta) - detail::trigamma(theta);
        }
        const double inv = 1 / (theta + m);
        t.theta_score += ai * (g1 + (m - v) * inv - log1p_by_log(m / theta));
        t.theta_curvature += ai * (g2 + 1 / theta + (v - m) * inv * inv - inv);
        c.rows->add(i, ai * theta * m * (v - m) * inv * inv, t.cross);
        if (v > 0)
            t.ylogy += ai * v * std::log(v);
    }
};
// The reference from today's null fitted on all retained cells (`fit`, with
// retained means): usable when that fit converged without a boundary.
inline Reference reference(FitResult fit, const Cells &c, const std::vector<std::size_t> &nt,
                           const Vector &y) {
    Reference r;
    r.fit = std::move(fit);
    r.ok = r.fit.ok() && !r.fit.has_boundary() && r.fit.status == FitStatus::converged &&
           !r.fit.fitted.empty() && std::isfinite(r.fit.theta);
    if (!r.ok)
        return r;
    const double t = r.fit.theta;
    for (std::size_t k = 0; k < 64; ++k) {
        const double inv = 1 / (t + static_cast<double>(k));
        r.d1[k + 1] = r.d1[k] + inv;
        r.d2[k + 1] = r.d2[k] - inv * inv;
    }
    r.nt = Totals(c.p);
    for (auto i : nt)
        r.add(c, i, y[i], r.nt);
    mirror(r.nt.information);
    return r;
}
// The pair's totals: the NT totals, then the block's (its cells outside the NT
// cells, ascending), in this fixed order.
inline Totals pair_totals(const Reference &r, const Cells &c, const std::vector<std::size_t> &block,
                          const Vector &y) {
    Totals b(c.p);
    for (auto i : block)
        r.add(c, i, y[i], b);
    mirror(b.information);
    Totals t = r.nt;
    for (std::size_t j = 0; j < c.p; ++j) {
        t.score[j] += b.score[j];
        t.cross[j] += b.cross[j];
    }
    for (std::size_t k = 0; k < c.p * c.p; ++k)
        t.information.values[k] += b.information.values[k];
    t.theta_score += b.theta_score;
    t.theta_curvature += b.theta_curvature;
    t.ylogy += b.ylogy;
    return t;
}
// One-step null of a pair from its totals.
struct Start {
    bool ok = false;
    Vector beta;
    double theta = nan, ylogy = 0;
};
inline Start one_step(const Reference &r, const Totals &t, const FitOptions &o) {
    Start s;
    Cholesky ch(t.information, o.rank_tolerance);
    if (!ch.ok)
        return s;
    const Vector step = ch.solve(t.score);
    s.beta = r.fit.coefficients;
    for (std::size_t j = 0; j < step.size(); ++j)
        s.beta[j] += step[j];
    const double theta = r.fit.theta;
    const double gu = t.theta_score * theta,
                 hu = t.theta_curvature * theta * theta + t.theta_score * theta;
    const double slope = gu + dot(t.cross, step);
    double u = std::log(theta);
    if (hu < 0 && std::isfinite(slope) && std::isfinite(hu))
        u = std::clamp(u - slope / hu, std::log(o.theta_min), std::log(o.theta_max));
    s.theta = std::exp(u);
    s.ylogy = t.ylogy;
    s.ok = std::isfinite(s.theta);
    for (double v : s.beta)
        s.ok = s.ok && std::isfinite(v);
    return s;
}
// A refined pair null and the work it took.
struct Refined {
    bool ok = false;
    FitResult fit;
    std::size_t passes = 0, halvings = 0, theta_passes = 0;
    bool theta_fallback = false;
};
// Today's theta at fixed means (as fit_sceptre_null).
inline std::pair<double, bool> conditional_theta(const Vector &y, const Vector &mu, const Vector &a,
                                                 const FitOptions &o, const CountTerms &terms) {
    double theta = moment_theta(y, mu, a, o);
    for (std::size_t iteration = 0; iteration < o.max_dispersion_iterations; ++iteration) {
        const double next = estimate_theta(y, mu, a, theta, o, terms);
        const double change = std::abs(std::log(next / theta));
        theta = next;
        if (change < o.tolerance)
            return {theta, true};
    }
    return {theta, false};
}
// Refines the start over the pair's cells `pos` (positions in the retained
// cells, ascending; y_pair and a_pair over them) to today's criteria: Newton
// passes, each the exact gradient and Poisson deviance (and, while steps are
// large, the information) over the pair's cells, halving a step that raised
// the deviance; converged when the step's predicted gain relative to the
// deviance and its relative coefficient change are within the tolerance (the
// chord criterion of fit()), and the step is then taken. Theta: Newton in log
// theta at the converged means from the start's theta; today's theta procedure
// takes over for a non-concave or large step.
inline Refined refine(const Cells &c, const std::vector<std::size_t> &pos, const Vector &y_pair,
                      const Vector &a_pair, const Start &start, const FitOptions &o,
                      std::size_t max_passes = 30) {
    Refined out;
    const std::size_t p = c.p, n = pos.size();
    const auto &rows = *c.rows;
    Vector beta = start.beta, previous, last_step;
    double previous_deviance = std::numeric_limits<double>::infinity(),
           last_change = std::numeric_limits<double>::infinity();
    std::optional<Cholesky> factor;
    Vector eta(n), mu(n), u(p);
    bool converged = false;
    for (std::size_t pass = 0; pass < max_passes && !converged; ++pass) {
        ++out.passes;
        for (std::size_t k = 0; k < n; ++k) {
            const double e = rows.dot(pos[k], beta) + c.off(pos[k]);
            eta[k] = e > 700 ? 710 : std::max(-700.0, e);
        }
        vmath::exp_array(eta.data(), mu.data(), n);
        std::fill(u.begin(), u.end(), 0.0);
        double yeta = 0, sum_y = 0, sum_mu = 0;
        bool finite = true;
        for (std::size_t k = 0; k < n; ++k) {
            const double ai = a_pair[k];
            if (!(ai > 0))
                continue;
            finite = finite && std::isfinite(mu[k]);
            rows.add(pos[k], ai * (y_pair[k] - mu[k]), u);
            yeta += ai * y_pair[k] * eta[k];
            sum_y += ai * y_pair[k];
            sum_mu += ai * mu[k];
        }
        const double deviance = 2 * (start.ylogy - yeta - sum_y + sum_mu);
        // A step is kept unless it raised the deviance by more than the cold
        // fit's step acceptance allows (1e-12 of the log likelihood, whose
        // terms are of the size of these sums); the deviance, a difference of
        // such sums, carries their rounding.
        const double slack = 2e-12 * (1 + std::abs(yeta) + std::abs(sum_mu));
        if (!finite || !std::isfinite(deviance) || deviance > previous_deviance + slack) {
            // The last step raised the deviance: halve it.
            if (previous.empty() || ++out.halvings > 30)
                return out;
            for (std::size_t j = 0; j < p; ++j) {
                last_step[j] *= 0.5;
                beta[j] = previous[j] + last_step[j];
            }
            continue;
        }
        if (!factor || last_change > 1e-6) {
            Matrix h(p, p);
            for (std::size_t k = 0; k < n; ++k)
                if (a_pair[k] > 0)
                    rows.add_outer(pos[k], a_pair[k] * mu[k], h);
            mirror(h);
            Cholesky exact(h, o.rank_tolerance);
            if (exact.ok)
                factor.emplace(std::move(exact));
            else if (!factor)
                return out;
        }
        const Vector d = factor->solve(u);
        const double gain = dot(u, d);
        double change = 0;
        for (std::size_t j = 0; j < p; ++j)
            change = std::max(change, std::abs(d[j]) / (1 + std::abs(beta[j])));
        if (!std::isfinite(gain) || !std::isfinite(change))
            return out;
        const double resolution =
            100 * std::numeric_limits<double>::epsilon() / std::sqrt(factor->rcond);
        converged = gain / (0.1 + std::abs(deviance)) < o.tolerance &&
                    change < std::max(o.tolerance, resolution);
        previous = beta;
        previous_deviance = deviance;
        last_step = d;
        last_change = change;
        for (std::size_t j = 0; j < p; ++j)
            beta[j] += d[j];
    }
    if (!converged)
        return out;
    for (std::size_t k = 0; k < n; ++k)
        eta[k] = rows.dot(pos[k], beta) + c.off(pos[k]);
    exp_means(eta);
    mu = std::move(eta);
    const CountTerms terms(y_pair, a_pair);
    double theta = start.theta, lu = std::log(theta);
    bool theta_ok = theta > o.theta_min * 1.00001 && theta < o.theta_max / 1.00001;
    if (theta_ok) {
        theta_ok = false;
        Vector ratio(n), logs(n);
        for (unsigned it = 0; it < 20; ++it) {
            ++out.theta_passes;
            const double t = std::exp(lu), inverse_t = 1 / t;
            auto [pg, ph] = terms.polygamma(t);
            ph += terms.weight / t;
            for (std::size_t k = 0; k < n; ++k) {
                ratio[k] = mu[k] * inverse_t;
                logs[k] = 1 + ratio[k];
            }
            vmath::log_array(logs.data(), logs.data(), n);
            double g = 0, h = 0;
            for (std::size_t k = 0; k < n; ++k) {
                const double ai = a_pair[k];
                if (ai > 0) {
                    const double v = y_pair[k], m = mu[k], inv = 1 / (t + m);
                    g += ai * ((m - v) * inv - log1p_from_log(ratio[k], logs[k]));
                    h += ai * ((v - m) * inv * inv - inv);
                }
            }
            const long double gl = pg + g, hl = ph + h;
            const double gu = static_cast<double>(gl * t),
                         hu = static_cast<double>(hl * t * t + gl * t);
            if (std::abs(gu) < 1e-10) {
                theta_ok = true;
                break;
            }
            if (!(hu < 0))
                break;
            const double step = -gu / hu;
            if (!(std::abs(step) <= 1))
                break;
            const double next = std::clamp(lu + step, std::log(o.theta_min), std::log(o.theta_max));
            const double change = std::abs(next - lu);
            lu = next;
            if (change < o.tolerance * 0.1) {
                theta_ok = true;
                break;
            }
        }
    }
    bool theta_converged = true;
    theta = std::exp(lu);
    if (!theta_ok) {
        out.theta_fallback = true;
        std::tie(theta, theta_converged) = conditional_theta(y_pair, mu, a_pair, o, terms);
    }
    auto &r = out.fit;
    r.family = Family::negative_binomial;
    r.method = FitMethod::sceptre_null;
    r.status = theta_converged ? FitStatus::converged : FitStatus::iteration_limit;
    r.coefficients = std::move(beta);
    r.standard_errors.assign(p, nan);
    r.boundary_coefficients.assign(p, 0);
    r.aliased.assign(p, 0);
    r.parameters = r.rank = p;
    for (double ai : a_pair)
        r.observations += ai > 0;
    r.theta = theta;
    r.theta_boundary = theta <= o.theta_min * 1.00001 || theta >= o.theta_max / 1.00001;
    r.fitted = std::move(mu);
    r.iterations = out.passes;
    r.dispersion_iterations = out.theta_passes;
    out.ok = true;
    return out;
}
// The score records and information of an NB null with means mu and theta over
// the pair's cells (the layout and formulas of ScoreWorkspace and
// BinaryScoreWorkspace, with the information summed over the rows' nonzero
// entries); nullptr when the information is singular.
inline std::unique_ptr<BinaryScoreWorkspace> score_records(const Cells &c,
                                                           const std::vector<std::size_t> &pos,
                                                           const Vector &y_pair, const Vector &mu,
                                                           double theta) {
    const std::size_t p = c.p, n = pos.size();
    auto ws = std::make_unique<BinaryScoreWorkspace>(p, n);
    Matrix info(p, p);
    const auto &rows = *c.rows;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = pos[k];
        const double m = mu[k], factor = 1 / (1 + m / theta), ai = c.a(i);
        const double w = ai * (m * factor), s = ai * (y_pair[k] - m) * factor;
        double *record = &ws->cells[k * ws->stride];
        record[0] = s;
        record[1] = w;
        for (auto e = rows.start[i]; e < rows.start[i + 1]; ++e)
            record[2 + rows.column[e]] = rows.value[e] * w;
        rows.add_outer(i, w, info);
    }
    mirror(info);
    Cholesky ch(info);
    if (!ch.ok)
        return nullptr;
    ws->inverse = ch.inverse();
    return ws;
}
} // namespace rapidglm::detail::mergeable
