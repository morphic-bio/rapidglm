#pragma once
// Lockstep null fits of a gene's permuted count phenotypes (cis_qtl with a
// Poisson or NB model; CisQtlOptions::permutation_fits). Every permutation of a
// gene is fitted against the same design X, so the sums over samples of a block
// of fits are taken together as dense products (detail::blocked_product, in
// sample order from zero): X'WX of every fit as Q'W, with Q the samples'
// products x_ik x_il (k >= l) computed once, X's as X'S, and the linear
// predictors as B'X'. Each fit keeps its own path: the steps, step halving,
// convergence tests and theta updates of fit() (the Poisson IRLS with its chord
// check and refinement, then for NB the joint Newton steps in coefficients and
// log theta), on that fit's values. A fit that leaves the regular path (an
// ill-conditioned or singular step, a possible boundary, exhausted step
// halving or iterations, a moment theta at its bound) is marked for the caller
// to refit with fit(). The sums are those of fit() in another order, so the
// fits agree with fit()'s to rounding; they do not depend on the processor's
// vector path or on threads.
#include "kernels.hpp"
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace rapidglm::detail {
// An operand of blocked products packed once and reused: the rows of A (n x p)
// in tiles of six, or the columns of B (n x m) in groups of eight, panel by
// panel of 128 samples, padded with zeros, as blocked_product packs them. A
// product of packed operands sums each element over the samples in order from
// zero with the same tiles, so it gives blocked_product's bits.
struct PackedOperand {
    std::size_t n = 0, width = 0, lanes = 0, groups = 0;
    Vector data;
    const double *group(std::size_t i0, std::size_t g) const {
        const std::size_t count = std::min<std::size_t>(128, n - i0);
        return data.data() + i0 * lanes * groups + g * count * lanes;
    }
};
// Packs a (n x width, row-major) in groups of `lanes` (6 for A, 8 for B).
inline void pack_operand(const double *a, std::size_t n, std::size_t width, std::size_t lanes,
                         PackedOperand &out) {
    out.n = n;
    out.width = width;
    out.lanes = lanes;
    out.groups = (width + lanes - 1) / lanes;
    out.data.assign(n * lanes * out.groups, 0.0);
    for (std::size_t i0 = 0; i0 < n; i0 += 128) {
        const std::size_t count = std::min<std::size_t>(128, n - i0);
        for (std::size_t g = 0; g < out.groups; ++g) {
            double *d = out.data.data() + i0 * lanes * out.groups + g * count * lanes;
            const std::size_t used = std::min(lanes, width - g * lanes);
            for (std::size_t i = 0; i < count; ++i)
                std::copy_n(a + (i0 + i) * width + g * lanes, used, d + i * lanes);
        }
    }
}
// C = A'B for packed A (6-row tiles) and B (8-column groups), row k of C at
// out[k] (m = B's width values each), zeroed here.
inline void packed_product(const PackedOperand &a, const PackedOperand &b, double *const *out) {
    const std::size_t p = a.width, m = b.width, n = a.n;
    for (std::size_t k = 0; k < p; ++k)
        std::fill_n(out[k], m, 0.0);
#ifdef RAPIDGLM_PHILOX_AVX2
    const bool avx2 = fma_available();
#endif
    double edge[6 * 8];
    for (std::size_t i0 = 0; i0 < n; i0 += 128) {
        const std::size_t count = std::min<std::size_t>(128, n - i0);
        for (std::size_t q0 = 0; q0 < b.groups; q0 += 16) {
            const std::size_t q1 = std::min(b.groups, q0 + 16);
            for (std::size_t r = 0; r < a.groups; ++r) {
                const double *ar = a.group(i0, r);
                const std::size_t rows = std::min<std::size_t>(6, p - r * 6);
                for (std::size_t q = q0; q < q1; ++q) {
                    const double *bq = b.group(i0, q);
                    const std::size_t lanes = std::min<std::size_t>(8, m - q * 8);
                    for (std::size_t k = 0; k < rows; ++k)
                        std::copy_n(out[r * 6 + k] + q * 8, lanes, edge + k * 8);
                    for (std::size_t k = rows; k < 6; ++k)
                        std::fill_n(edge + k * 8, 8, 0.0);
                    for (std::size_t k = 0; k < rows; ++k)
                        std::fill(edge + k * 8 + lanes, edge + k * 8 + 8, 0.0);
#ifdef RAPIDGLM_PHILOX_AVX2
                    if (avx2)
                        product_tile_avx2(ar, bq, count, edge, 8);
                    else
#endif
                        product_tile_generic(ar, bq, count, edge, 8);
                    for (std::size_t k = 0; k < rows; ++k)
                        std::copy_n(edge + k * 8, lanes, out[r * 6 + k] + q * 8);
                }
            }
        }
    }
}
// The design of a gene's fits: X and X' row-major, and the products Q.
struct QtlLockstepDesign {
    std::size_t n = 0, p = 0, kk = 0;
    Vector x, xt, q;
    // Packed once: Q and X as the A operands of X'WX and X'v, X' as the B
    // operand of the linear predictors B'X'.
    PackedOperand q_rows, x_rows, xt_columns;
    bool ok = false;
    QtlLockstepDesign() = default;
    explicit QtlLockstepDesign(const Matrix &design) : n(design.rows), p(design.cols) {
        kk = p * (p + 1) / 2;
        x = design.values;
        xt.resize(p * n);
        q.resize(n * kk);
        for (std::size_t i = 0; i < n; ++i) {
            const double *xi = &design.values[i * p];
            for (std::size_t k = 0; k < p; ++k)
                xt[k * n + i] = xi[k];
            double *qi = &q[i * kk];
            for (std::size_t j = 0, slot = 0; j < p; ++j)
                for (std::size_t k = 0; k <= j; ++k, ++slot)
                    qi[slot] = xi[j] * xi[k];
        }
        pack_operand(q.data(), n, kk, 6, q_rows);
        pack_operand(x.data(), n, p, 6, x_rows);
        pack_operand(xt.data(), p, n, 8, xt_columns);
        ok = true;
    }
};
// The fits of one block: responses and offsets by column, values by column
// (row t of a K x n matrix holds column t).
struct QtlLockstepFit {
    Vector coefficients, fitted, working_weights, score_residuals;
    double theta = nan, log_likelihood = nan;
    bool ok = false;
    // The factor of X'WX at the working weights (the scores' basis factor).
    std::optional<Cholesky> factor;
};
namespace lockstep {
// Unpacks column t of C (kk x K) into the symmetric p x p matrix h.
inline void unpack_gram(const Vector &c, std::size_t K, std::size_t t, std::size_t p, Matrix &h) {
    h = Matrix(p, p);
    for (std::size_t j = 0, slot = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k, ++slot)
            h(j, k) = h(k, j) = c[slot * K + t];
}
} // namespace lockstep
// Fits the K responses y[t] (offsets off[t], may be empty) of one gene in
// lockstep, family Poisson or NB with ML theta; out[t].ok false marks a column
// for the caller's fit(). `terms` holds CountTerms of the responses' common
// multiset of counts. `design_check` is the factor of X'X.
inline void qtl_lockstep_fits(const QtlLockstepDesign &d, const std::vector<const Vector *> &y,
                              const std::vector<const Vector *> &off, const FitOptions &o,
                              const Cholesky &design_check, const CountTerms &terms,
                              std::vector<QtlLockstepFit> &out) {
    namespace L = lockstep;
    const std::size_t K = y.size(), n = d.n, p = d.p, kk = d.kk;
    const bool nb = o.family == Family::negative_binomial;
    const double tol = o.tolerance;
    out.assign(K, {});
    if (K == 0)
        return;
    struct Col {
        Vector beta, delta, coupling, trial, u, mu, eta, mu2, eta2, w, s, mixed;
        std::optional<Cholesky> factor;
        double ll = 0, dev = 0, resolution = 0, step = 1, theta = 0, next_theta = 0, scale = 0,
               gu = 0, du = 0, ln = 0;
        int positive = -1;
        bool done = false, failed = false, pending = false, have_u = false, chord = false,
             previous_stable = false, held = false;
        std::size_t iterations = 0;
    };
    std::vector<Col> col(K);
    // Column-wise products over samples.
    Vector bt(p * K), etak(K * n), wk(n * K), gram(kk * K), xs(p * K);
    PackedOperand packed;
    std::vector<double *> rows_out(K), gram_rows(kk), xs_rows(p);
    // The linear predictors of the coefficients in bt (p x K): eta[t] = X b_t + off_t.
    auto linear = [&](const std::vector<char> &want, auto coefficients, auto target) {
        std::fill(bt.begin(), bt.end(), 0.0);
        for (std::size_t t = 0; t < K; ++t)
            if (want[t]) {
                const Vector &b = coefficients(t);
                for (std::size_t j = 0; j < p; ++j)
                    bt[j * K + t] = b[j];
            }
        pack_operand(bt.data(), p, K, 6, packed);
        for (std::size_t t = 0; t < K; ++t)
            rows_out[t] = &etak[t * n];
        packed_product(packed, d.xt_columns, rows_out.data());
        for (std::size_t t = 0; t < K; ++t) {
            if (!want[t])
                continue;
            auto [eta, mu] = target(t);
            eta.resize(n);
            mu.resize(n);
            const double *e = &etak[t * n];
            for (std::size_t i = 0; i < n; ++i) {
                const double v = e[i] + (off[t]->empty() ? 0.0 : (*off[t])[i]);
                eta[i] = v > 700 ? 710 : std::max(-700.0, v);
            }
            vmath::exp_array(eta.data(), mu.data(), n);
        }
    };
    // Gram X'WX (into gram, kk x K) of the columns' weight vectors.
    auto grams = [&](const std::vector<char> &want, auto weights) {
        std::fill(wk.begin(), wk.end(), 0.0);
        for (std::size_t t = 0; t < K; ++t)
            if (want[t]) {
                const Vector &w = weights(t);
                for (std::size_t i = 0; i < n; ++i)
                    wk[i * K + t] = w[i];
            }
        pack_operand(wk.data(), n, K, 8, packed);
        for (std::size_t k = 0; k < kk; ++k)
            gram_rows[k] = &gram[k * K];
        packed_product(d.q_rows, packed, gram_rows.data());
    };
    // X'v (into xs, p x K) of the columns' vectors.
    auto xtvs = [&](const std::vector<char> &want, auto vectors) {
        std::fill(wk.begin(), wk.end(), 0.0);
        for (std::size_t t = 0; t < K; ++t)
            if (want[t]) {
                const Vector &v = vectors(t);
                for (std::size_t i = 0; i < n; ++i)
                    wk[i * K + t] = v[i];
            }
        pack_operand(wk.data(), n, K, 8, packed);
        for (std::size_t k = 0; k < p; ++k)
            xs_rows[k] = &xs[k * K];
        packed_product(d.x_rows, packed, xs_rows.data());
    };
    auto column_of = [&](const Vector &m, std::size_t t) {
        Vector v(p);
        for (std::size_t j = 0; j < p; ++j)
            v[j] = m[j * K + t];
        return v;
    };
    // Log likelihoods (Poisson or NB at theta) of column t at means mu (log means eta).
    auto loglik = [&](std::size_t t, const Vector &eta, const Vector &mu, double theta,
                      bool poisson) -> double {
        const Vector &v = *y[t];
        long double ll = 0, ltw = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double m = mu[i], yi = v[i];
            if (!std::isfinite(m))
                return -std::numeric_limits<double>::infinity();
            if (poisson)
                ll += (yi > 0 ? yi * eta[i] : 0) - m;
            else {
                const double ratio = log1p_by_log(m / theta), shape = theta * ratio;
                if (yi == 0)
                    ll -= shape;
                else if (theta < m)
                    ll -= yi * log1p_by_log(theta / m) + shape;
                else {
                    ll += yi * (eta[i] - ratio) - shape;
                    ltw += yi;
                }
            }
        }
        if (poisson)
            ll -= terms.log_factorial;
        else
            ll += terms.nb_constant(theta) - ltw * std::log(static_cast<long double>(theta));
        return static_cast<double>(ll);
    };
    // Count deviance of column t (count_deviance).
    auto deviance = [&](std::size_t t, const Vector &mu, double theta, bool poisson) {
        const Vector &v = *y[t];
        double total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double term;
            if (v[i] == 0)
                term = poisson ? mu[i] : theta * log1p_by_log(mu[i] / theta);
            else {
                const double vi = v[i], m = mu[i], difference = m - vi;
                term = std::abs(difference) < 0.125 * vi ? vi * log1pmx(difference / vi)
                                                         : vi * std::log(vi / m) + difference;
                if (!poisson)
                    term -= (vi + theta) * log1pmx(difference / (vi + theta));
            }
            if (!std::isfinite(term))
                return std::numeric_limits<double>::infinity();
            total += 2 * std::max(0.0, term);
        }
        return total;
    };
    // Full rank of the design among positive counts (positive_counts_full_rank).
    auto positive_full_rank = [&](std::size_t t) {
        Vector ind(n);
        bool zero = false;
        for (std::size_t i = 0; i < n; ++i) {
            ind[i] = (*y[t])[i] > 0;
            zero |= (*y[t])[i] == 0;
        }
        if (!zero)
            return design_check.ok;
        Matrix h(p, p);
        for (std::size_t i = 0; i < n; ++i)
            if (ind[i]) {
                const double *xi = &d.x[i * p];
                for (std::size_t j = 0; j < p; ++j)
                    for (std::size_t k = 0; k <= j; ++k)
                        h(j, k) += xi[j] * xi[k];
            }
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < j; ++k)
                h(k, j) = h(j, k);
        return Cholesky(h, o.rank_tolerance).ok;
    };
    auto all_done = [&] {
        for (const auto &c : col)
            if (!c.done)
                return false;
        return true;
    };
    std::vector<char> want(K, 1);
    // ---- Start: least squares on the link of the shrunken counts (as fit_from).
    {
        long double sumy = 0, suma = 0;
        for (double v : *y[0]) {
            sumy += v;
            suma += 1;
        }
        const double shift = 0.1 * static_cast<double>(sumy) / static_cast<double>(suma);
        std::vector<Vector> z(K, Vector(n));
        for (std::size_t t = 0; t < K; ++t)
            for (std::size_t i = 0; i < n; ++i) {
                const double m = ((*y[t])[i] + shift + 0.1) / 1.1;
                z[t][i] = std::log(m) - (off[t]->empty() ? 0.0 : (*off[t])[i]);
            }
        xtvs(want, [&](std::size_t t) -> const Vector & { return z[t]; });
        for (std::size_t t = 0; t < K; ++t)
            col[t].beta = design_check.solve(column_of(xs, t));
    }
    // ---- Poisson IRLS (the pilot of an NB fit).
    linear(
        want, [&](std::size_t t) -> const Vector & { return col[t].beta; },
        [&](std::size_t t) { return std::pair<Vector &, Vector &>(col[t].eta, col[t].mu); });
    for (std::size_t t = 0; t < K; ++t) {
        auto &c = col[t];
        c.ll = loglik(t, c.eta, c.mu, 0, true);
        c.dev = deviance(t, c.mu, 0, true);
    }
    while (!all_done()) {
        std::vector<char> active(K), grad(K);
        for (std::size_t t = 0; t < K; ++t) {
            active[t] = !col[t].done;
            grad[t] = active[t] && !col[t].have_u;
        }
        for (std::size_t t = 0; t < K; ++t)
            if (active[t]) {
                auto &c = col[t];
                c.w = c.mu; // Poisson working weights a mu (a = 1)
                c.s.resize(n);
                for (std::size_t i = 0; i < n; ++i)
                    c.s[i] = (*y[t])[i] - c.mu[i];
            }
        grams(active, [&](std::size_t t) -> const Vector & { return col[t].w; });
        std::vector<Matrix> h(K);
        for (std::size_t t = 0; t < K; ++t)
            if (active[t])
                L::unpack_gram(gram, K, t, p, h[t]);
        xtvs(grad, [&](std::size_t t) -> const Vector & { return col[t].s; });
        for (std::size_t t = 0; t < K; ++t) {
            auto &c = col[t];
            if (!active[t])
                continue;
            if (grad[t])
                c.u = column_of(xs, t);
            c.have_u = false;
            if (++c.iterations > o.max_iterations) {
                c.failed = c.done = true;
                continue;
            }
            c.factor.emplace(h[t], o.rank_tolerance);
            if (!c.factor->ok || c.factor->rcond < 1e-8) {
                c.failed = c.done = true; // fit() would take a QR step
                continue;
            }
            c.delta = c.factor->solve(c.u);
            c.resolution =
                100 * std::numeric_limits<double>::epsilon() / std::sqrt(c.factor->rcond);
            c.step = 1;
            c.pending = true;
            c.chord = false;
        }
        for (unsigned halve = 0; halve < 40; ++halve) {
            std::vector<char> pend(K);
            bool any = false;
            for (std::size_t t = 0; t < K; ++t) {
                pend[t] = col[t].pending;
                any |= pend[t] != 0;
                if (pend[t]) {
                    auto &c = col[t];
                    c.trial.resize(p);
                    for (std::size_t j = 0; j < p; ++j)
                        c.trial[j] = c.beta[j] + c.step * c.delta[j];
                }
            }
            if (!any)
                break;
            linear(
                pend, [&](std::size_t t) -> const Vector & { return col[t].trial; },
                [&](std::size_t t) {
                    return std::pair<Vector &, Vector &>(col[t].eta2, col[t].mu2);
                });
            for (std::size_t t = 0; t < K; ++t) {
                if (!pend[t])
                    continue;
                auto &c = col[t];
                const double ln = loglik(t, c.eta2, c.mu2, 0, true);
                if (!(std::isfinite(ln) && ln >= c.ll - 1e-12 * (1 + std::abs(c.ll)))) {
                    c.step *= 0.5;
                    continue;
                }
                c.pending = false;
                const double next_dev = c.dev - 2 * (ln - c.ll);
                const bool stable = std::isfinite(next_dev) &&
                                    std::abs(next_dev - c.dev) / (0.1 + std::abs(next_dev)) < tol;
                if (stable) {
                    if (c.positive < 0)
                        c.positive = positive_full_rank(t) ? 1 : 0;
                    if (!c.positive) {
                        c.failed = c.done = true; // a boundary is possible
                        continue;
                    }
                }
                double change = 0;
                for (std::size_t j = 0; j < p; ++j)
                    change = std::max(change,
                                      std::abs(c.trial[j] - c.beta[j]) / (1 + std::abs(c.beta[j])));
                bool close = true;
                for (std::size_t i = 0; i < n && close; ++i)
                    close = std::abs(c.mu2[i] - c.mu[i]) <= 0.25 * c.mu[i];
                c.beta = c.trial;
                std::swap(c.mu, c.mu2);
                std::swap(c.eta, c.eta2);
                c.ll = ln;
                c.dev = next_dev;
                if (stable && change < std::max(tol, c.resolution)) {
                    c.done = true;
                    continue;
                }
                c.chord = close;
            }
        }
        for (auto &c : col)
            if (c.pending) {
                c.pending = false;
                c.failed = c.done = true; // step halving exhausted
            }
        // Chord checks of the columns whose weights stayed close.
        std::vector<char> chord(K);
        bool any_chord = false;
        for (std::size_t t = 0; t < K; ++t) {
            chord[t] = !col[t].done && col[t].chord;
            any_chord |= chord[t] != 0;
            if (chord[t]) {
                auto &c = col[t];
                c.s.resize(n);
                for (std::size_t i = 0; i < n; ++i)
                    c.s[i] = (*y[t])[i] - c.mu[i];
            }
        }
        if (!any_chord)
            continue;
        xtvs(chord, [&](std::size_t t) -> const Vector & { return col[t].s; });
        std::vector<char> refine(K);
        bool any_refine = false;
        std::vector<double> gains(K);
        for (std::size_t t = 0; t < K; ++t) {
            if (!chord[t])
                continue;
            auto &c = col[t];
            c.u = column_of(xs, t);
            c.have_u = true;
            const Vector chord_step = c.factor->solve(c.u);
            const double gain = dot(c.u, chord_step);
            const bool stable = std::isfinite(gain) && gain / (0.1 + std::abs(c.dev)) < tol;
            if (stable) {
                if (c.positive < 0)
                    c.positive = positive_full_rank(t) ? 1 : 0;
                if (!c.positive) {
                    c.failed = c.done = true;
                    continue;
                }
            }
            double chord_change = 0;
            for (std::size_t j = 0; j < p; ++j)
                chord_change =
                    std::max(chord_change, std::abs(chord_step[j]) / (1 + std::abs(c.beta[j])));
            if (stable && std::isfinite(chord_change) &&
                chord_change < std::max(tol, c.resolution)) {
                c.trial.resize(p);
                for (std::size_t j = 0; j < p; ++j)
                    c.trial[j] = c.beta[j] + chord_step[j];
                gains[t] = gain;
                refine[t] = 1;
                any_refine = true;
                c.done = true;
            }
        }
        if (!any_refine)
            continue;
        linear(
            refine, [&](std::size_t t) -> const Vector & { return col[t].trial; },
            [&](std::size_t t) { return std::pair<Vector &, Vector &>(col[t].eta2, col[t].mu2); });
        for (std::size_t t = 0; t < K; ++t) {
            if (!refine[t])
                continue;
            auto &c = col[t];
            bool finite = true;
            for (double m : c.mu2)
                finite = finite && std::isfinite(m);
            const double rise = 0.5 * gains[t];
            const bool known = rise <= 1e-13 * (1 + std::abs(c.ll)) && finite;
            const double lr = known ? c.ll + rise : loglik(t, c.eta2, c.mu2, 0, true);
            if (std::isfinite(lr) && lr >= c.ll - 1e-12 * (1 + std::abs(c.ll))) {
                c.beta = c.trial;
                std::swap(c.mu, c.mu2);
                std::swap(c.eta, c.eta2);
                c.ll = lr;
            }
        }
    }
    // The pilot's final information (fit()'s covariance step) must factor well.
    {
        std::vector<char> ok(K);
        for (std::size_t t = 0; t < K; ++t)
            ok[t] = !col[t].failed;
        grams(ok, [&](std::size_t t) -> const Vector & { return col[t].mu; });
        for (std::size_t t = 0; t < K; ++t) {
            if (!ok[t])
                continue;
            Matrix h;
            L::unpack_gram(gram, K, t, p, h);
            Cholesky ch(h, o.rank_tolerance);
            if (!ch.ok || ch.rcond < 1e-8)
                col[t].failed = true;
            else if (!nb)
                out[t].factor.emplace(std::move(ch));
        }
    }
    if (!nb) {
        for (std::size_t t = 0; t < K; ++t) {
            auto &c = col[t];
            if (c.failed)
                continue;
            auto &r = out[t];
            r.coefficients = c.beta;
            r.fitted = c.mu;
            r.working_weights = c.mu;
            r.score_residuals.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                r.score_residuals[i] = (*y[t])[i] - c.mu[i];
            r.log_likelihood = c.ll;
            r.ok = true;
        }
        return;
    }
    // ---- NB: joint Newton in (coefficients, log theta) from the Poisson pilot
    // and the moment theta (fit_from's joint loop).
    const double lower = std::log(o.theta_min), upper = std::log(o.theta_max);
    for (std::size_t t = 0; t < K; ++t) {
        auto &c = col[t];
        c.done = c.failed;
        c.iterations = 0;
        c.previous_stable = false;
        if (c.failed)
            continue;
        // Moment theta at the pilot's means; at the bound fit() would also fit
        // from o.theta and keep the better.
        double num = 0, den = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double m = c.mu[i], v = (*y[t])[i];
            num += m * m;
            den += (v - m) * (v - m) - m;
        }
        const double moment =
            den > 0 ? std::clamp(num / den, o.theta_min, o.theta_max) : o.theta_max;
        if (moment >= o.theta_max) {
            c.failed = c.done = true;
            continue;
        }
        c.theta = moment;
        // mean_values(beta) as fit_from recomputes it.
        c.ll = loglik(t, c.eta, c.mu, c.theta, false);
        c.scale = 0.1 + std::abs(deviance(t, c.mu, c.theta, false));
    }
    while (!all_done()) {
        std::vector<char> active(K);
        for (std::size_t t = 0; t < K; ++t)
            active[t] = !col[t].done;
        std::vector<long double> g(K), hsum(K);
        for (std::size_t t = 0; t < K; ++t) {
            if (!active[t])
                continue;
            auto &c = col[t];
            const double th = c.theta;
            auto [pg, ph] = terms.polygamma(th);
            ph += terms.weight / th;
            c.w.resize(n);
            c.s.resize(n);
            c.mixed.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double m = c.mu[i], v = (*y[t])[i], dd = th + m;
                c.w[i] = th * m * (th + v) / (dd * dd);
                c.s[i] = th * (v - m) / dd;
                c.mixed[i] = th * m * (v - m) / (dd * dd);
                pg += (m - v) / dd - log1p_by_log(m / th);
                ph += (v - m) / (dd * dd) - 1 / dd;
            }
            g[t] = pg;
            hsum[t] = ph;
        }
        grams(active, [&](std::size_t t) -> const Vector & { return col[t].w; });
        std::vector<Matrix> h(K);
        for (std::size_t t = 0; t < K; ++t)
            if (active[t])
                L::unpack_gram(gram, K, t, p, h[t]);
        xtvs(active, [&](std::size_t t) -> const Vector & { return col[t].s; });
        std::vector<Vector> xs_s(K);
        for (std::size_t t = 0; t < K; ++t)
            if (active[t])
                xs_s[t] = column_of(xs, t);
        xtvs(active, [&](std::size_t t) -> const Vector & { return col[t].mixed; });
        for (std::size_t t = 0; t < K; ++t) {
            auto &c = col[t];
            if (!active[t])
                continue;
            if (++c.iterations > o.max_iterations) {
                c.failed = c.done = true;
                continue;
            }
            const Vector cc = column_of(xs, t);
            c.factor.emplace(h[t], o.rank_tolerance);
            if (!c.factor->ok || c.factor->rcond < 1e-8) {
                c.failed = c.done = true; // fit() would take a QR step
                continue;
            }
            c.delta = c.factor->solve(xs_s[t]);
            c.coupling = c.factor->solve(cc);
            c.resolution =
                100 * std::numeric_limits<double>::epsilon() / std::sqrt(c.factor->rcond);
            const double th = c.theta, u = std::log(th);
            const double gu = static_cast<double>(g[t] * th),
                         hu = static_cast<double>(hsum[t] * th * th + g[t] * th);
            const double slope = gu + dot(cc, c.delta), schur = hu + dot(cc, c.coupling);
            double coefficient_change = 0;
            for (std::size_t j = 0; j < p; ++j)
                coefficient_change =
                    std::max(coefficient_change, std::abs(c.delta[j]) / (1 + std::abs(c.beta[j])));
            double du = std::abs(slope) < 1e-10               ? 0
                        : schur < 0                           ? -slope / schur
                        : coefficient_change > std::sqrt(tol) ? 0
                                                              : (slope > 0 ? 1 : -1);
            const bool pinned = (slope > 0 && u >= upper) || (slope < 0 && u <= lower);
            c.held = schur >= 0 && du == 0 && std::abs(slope) >= 1e-10 && !pinned;
            du = std::clamp(u + std::clamp(du, -2.0, 2.0), lower, upper) - u;
            c.du = du;
            c.gu = gu;
            for (std::size_t j = 0; j < p; ++j)
                c.delta[j] = c.delta[j] + c.coupling[j] * du; // the direction
            c.u = xs_s[t];
            c.step = 1;
            c.pending = true;
        }
        for (unsigned halve = 0; halve < 40; ++halve) {
            std::vector<char> pend(K);
            bool any = false;
            for (std::size_t t = 0; t < K; ++t) {
                pend[t] = col[t].pending;
                any |= pend[t] != 0;
                if (pend[t]) {
                    auto &c = col[t];
                    c.trial.resize(p);
                    for (std::size_t j = 0; j < p; ++j)
                        c.trial[j] = c.beta[j] + c.step * c.delta[j];
                    const double u = std::log(c.theta);
                    c.next_theta = c.du == 0 ? c.theta
                                             : std::clamp(std::exp(u + c.step * c.du), o.theta_min,
                                                          o.theta_max);
                }
            }
            if (!any)
                break;
            linear(
                pend, [&](std::size_t t) -> const Vector & { return col[t].trial; },
                [&](std::size_t t) {
                    return std::pair<Vector &, Vector &>(col[t].eta2, col[t].mu2);
                });
            for (std::size_t t = 0; t < K; ++t) {
                if (!pend[t])
                    continue;
                auto &c = col[t];
                const double ln = loglik(t, c.eta2, c.mu2, c.next_theta, false);
                if (!(std::isfinite(ln) && ln >= c.ll - 1e-12 * (1 + std::abs(c.ll)))) {
                    c.step *= 0.5;
                    continue;
                }
                c.pending = false;
                const bool stable = 2 * std::abs(ln - c.ll) < tol * c.scale;
                c.previous_stable = stable;
                if (stable) {
                    if (c.positive < 0)
                        c.positive = positive_full_rank(t) ? 1 : 0;
                    if (!c.positive) {
                        c.failed = c.done = true;
                        continue;
                    }
                }
                double change = 0;
                for (std::size_t j = 0; j < p; ++j)
                    change = std::max(change,
                                      std::abs(c.trial[j] - c.beta[j]) / (1 + std::abs(c.beta[j])));
                const double u_change = std::abs(std::log(c.next_theta / c.theta));
                c.beta = c.trial;
                std::swap(c.mu, c.mu2);
                std::swap(c.eta, c.eta2);
                c.theta = c.next_theta;
                c.ll = ln;
                if (stable && !c.held && change < std::max(tol, c.resolution) && u_change < tol)
                    c.done = true;
            }
        }
        for (std::size_t t = 0; t < K; ++t) {
            auto &c = col[t];
            if (!c.pending)
                continue;
            c.pending = false;
            // No step improves: converged if the objective had settled or the
            // step promises less than the tolerance (fit_from), else fit() fails.
            const double predicted = std::abs(dot(c.u, c.delta) + c.gu * c.du);
            if (c.previous_stable || predicted < tol * c.scale)
                c.done = true;
            else
                c.failed = c.done = true;
        }
    }
    // The final information (fit()'s covariance step) at the NB working weights.
    std::vector<char> ok(K);
    for (std::size_t t = 0; t < K; ++t) {
        auto &c = col[t];
        ok[t] = !c.failed;
        if (c.failed)
            continue;
        c.w.resize(n);
        c.s.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double factor = 1 / (1 + c.mu[i] / c.theta);
            c.w[i] = c.mu[i] * factor;
            c.s[i] = ((*y[t])[i] - c.mu[i]) * factor;
        }
    }
    grams(ok, [&](std::size_t t) -> const Vector & { return col[t].w; });
    for (std::size_t t = 0; t < K; ++t) {
        auto &c = col[t];
        if (!ok[t])
            continue;
        Matrix h;
        L::unpack_gram(gram, K, t, p, h);
        Cholesky ch(h, o.rank_tolerance);
        if (!ch.ok || ch.rcond < 1e-8 || c.theta <= o.theta_min * 1.00001 ||
            c.theta >= o.theta_max / 1.00001) {
            c.failed = true;
            continue;
        }
        auto &r = out[t];
        r.factor.emplace(std::move(ch));
        r.coefficients = c.beta;
        r.theta = c.theta;
        r.fitted = c.mu;
        r.working_weights = c.w;
        r.score_residuals = c.s;
        r.log_likelihood = c.ll;
        r.ok = true;
    }
}
} // namespace rapidglm::detail
