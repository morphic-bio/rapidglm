#pragma once
#include "inference.hpp"
#include "lapack.hpp"

// Linear hypotheses C beta = 0 on a design's coefficients: one coefficient, several
// coefficients jointly, or contrasts such as beta_a - beta_b.
namespace rapidglm {
struct Hypothesis {
    Matrix contrasts; // k x p, full row rank, k < p
    static Hypothesis coefficient(std::size_t p, std::size_t j) {
        return coefficients(p, {j});
    }
    static Hypothesis coefficients(std::size_t p, const std::vector<std::size_t> &columns) {
        require(!columns.empty(), "a hypothesis needs at least one coefficient");
        Hypothesis h;
        h.contrasts = Matrix(columns.size(), p);
        for (std::size_t r = 0; r < columns.size(); ++r) {
            require(columns[r] < p, "hypothesis coefficient out of range");
            h.contrasts(r, columns[r]) = 1;
        }
        return h;
    }
    static Hypothesis contrast(const Vector &c) {
        Hypothesis h;
        h.contrasts = Matrix(1, c.size());
        h.contrasts.values = c;
        return h;
    }
    std::size_t df() const {
        return contrasts.rows;
    }
    // The tested columns when every row is a distinct unit vector, else empty.
    std::vector<std::size_t> unit_columns() const {
        std::vector<std::size_t> columns;
        for (std::size_t r = 0; r < contrasts.rows; ++r) {
            std::size_t ones = 0, column = 0;
            for (std::size_t j = 0; j < contrasts.cols; ++j) {
                const double v = contrasts(r, j);
                if (v == 1) {
                    ++ones;
                    column = j;
                } else if (v != 0)
                    return {};
            }
            if (ones != 1 || std::count(columns.begin(), columns.end(), column))
                return {};
            columns.push_back(column);
        }
        return columns;
    }
    // Orthonormal basis of the coefficient vectors the hypothesis leaves free
    // (p x (p - k)); fails unless C has full row rank k < p.
    Matrix null_space() const {
        const auto k = contrasts.rows, p = contrasts.cols;
        require(k > 0 && k < p, "a hypothesis needs between 1 and p - 1 rows");
        for (double v : contrasts.values)
            require(std::isfinite(v), "nonfinite contrast");
        // SVD of the column-major k x p array, which is C itself.
        Vector a(k * p), singular(std::min(k, p)), vt(p * p);
        for (std::size_t r = 0; r < k; ++r)
            for (std::size_t j = 0; j < p; ++j)
                a[j * k + r] = contrasts(r, j);
        const auto m = static_cast<lapack::integer>(k), n = static_cast<lapack::integer>(p);
        double query = 0, unused = 0;
        const auto svd = [&](double *work, lapack::integer length) {
            return lapack::gesvd('N', 'A', m, n, a.data(), m, singular.data(), &unused, 1,
                                 vt.data(), n, work, length);
        };
        require(svd(&query, -1) == 0 && query > 0, "hypothesis SVD workspace query failed");
        Vector work(static_cast<std::size_t>(query));
        require(svd(work.data(), static_cast<lapack::integer>(work.size())) == 0,
                "hypothesis SVD failed");
        require(singular.back() > 1e-10 * singular.front(),
                "hypothesis rows must be linearly independent");
        // Rows k..p-1 of V' span the null space of C.
        Matrix basis(p, p - k);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t c = k; c < p; ++c)
                basis(j, c - k) = vt[c + j * p];
        return basis;
    }
};
// The design under the hypothesis: the tested columns removed when the hypothesis
// names coefficients, otherwise X N for a basis N of the free coefficients (any
// basis gives the same fitted model).
inline Matrix reduced_design(const Matrix &x, const Hypothesis &h) {
    require(h.contrasts.cols == x.cols, "hypothesis and design columns differ");
    auto columns = h.unit_columns();
    if (!columns.empty()) {
        require(columns.size() < x.cols, "a hypothesis needs between 1 and p - 1 rows");
        std::sort(columns.begin(), columns.end());
        Matrix out(x.rows, x.cols - columns.size());
        for (std::size_t i = 0; i < x.rows; ++i)
            for (std::size_t j = 0, k = 0, next = 0; j < x.cols; ++j) {
                if (next < columns.size() && columns[next] == j) {
                    ++next;
                    continue;
                }
                out(i, k++) = x(i, j);
            }
        return out;
    }
    const auto basis = h.null_space();
    Matrix out(x.rows, basis.cols);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t c = 0; c < basis.cols; ++c)
            for (std::size_t j = 0; j < x.cols; ++j)
                out(i, c) += x(i, j) * basis(j, c);
    return out;
}
// For a one-row hypothesis c, a design whose coefficient `column` is c'beta:
// with k the entry of c largest in magnitude, column k becomes x_k / c_k and each
// other column j becomes x_j - (c_j / c_k) x_k. It spans the same model, so fitted
// values and likelihoods are unchanged, and a contrast of diverging coefficients
// that is itself finite keeps a Wald test. For one coefficient it is x itself.
struct TestedDesign {
    Matrix x;
    std::size_t column = 0;
};
inline TestedDesign tested_design(const Matrix &x, const Hypothesis &h) {
    require(h.df() == 1 && h.contrasts.cols == x.cols, "one-row hypothesis required");
    const auto unit = h.unit_columns();
    if (!unit.empty())
        return {x, unit[0]};
    h.null_space(); // validates the weights
    std::size_t k = 0;
    for (std::size_t j = 1; j < x.cols; ++j)
        if (std::abs(h.contrasts(0, j)) > std::abs(h.contrasts(0, k)))
            k = j;
    const double ck = h.contrasts(0, k);
    TestedDesign t{Matrix(x.rows, x.cols), k};
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t j = 0; j < x.cols; ++j)
            t.x(i, j) = j == k ? x(i, k) / ck : x(i, j) - h.contrasts(0, j) / ck * x(i, k);
    return t;
}
// Whether any coefficient the hypothesis involves has no finite estimate.
inline bool hypothesis_boundary(const FitResult &f, const Hypothesis &h) {
    if (f.boundary_coefficients.empty())
        return false;
    for (std::size_t r = 0; r < h.contrasts.rows; ++r)
        for (std::size_t j = 0; j < h.contrasts.cols; ++j)
            if (h.contrasts(r, j) != 0 && f.boundary_coefficients[j])
                return true;
    return false;
}
// Estimate c'beta and its standard error for a one-row hypothesis (the coefficient
// and its reported standard error when the row names one coefficient).
inline std::pair<double, double> hypothesis_estimate(const FitResult &f, const Hypothesis &h,
                                                     bool robust = false) {
    require(h.df() == 1 && h.contrasts.cols == f.parameters, "one-row hypothesis required");
    const auto unit = h.unit_columns();
    if (!unit.empty() && !robust)
        return {f.coefficients[unit[0]], f.standard_errors[unit[0]]};
    const Matrix &cov = robust ? f.robust_covariance : f.covariance;
    if (cov.rows != f.parameters || f.coefficients.size() != f.parameters)
        return {nan, nan};
    double estimate = 0, variance = 0;
    for (std::size_t j = 0; j < f.parameters; ++j) {
        estimate += h.contrasts(0, j) * f.coefficients[j];
        for (std::size_t k = 0; k < f.parameters; ++k)
            variance += h.contrasts(0, j) * cov(j, k) * h.contrasts(0, k);
    }
    return {estimate, variance > 0 ? std::sqrt(variance) : nan};
}
// Wald test of a one-row hypothesis; identical to wald_test for one coefficient.
inline TestResult hypothesis_wald(const FitResult &f, const Hypothesis &h, Side side = Side::both,
                                  bool robust = false) {
    require(h.df() == 1, "the Wald test takes one-row hypotheses; use a likelihood ratio test");
    const auto unit = h.unit_columns();
    if (!unit.empty())
        return wald_test(f, unit[0], side, robust);
    if (!f.ok() || f.method != FitMethod::direct || hypothesis_boundary(f, h))
        return {};
    if (robust)
        require(f.robust_covariance.rows == f.parameters,
                "robust covariance not requested during fit");
    const auto [estimate, se] = hypothesis_estimate(f, h, robust);
    if (!(se > 0) || !std::isfinite(se) || !std::isfinite(estimate))
        return {};
    double z = estimate / se, p = normal_tail(z, side);
    if (f.family == Family::gaussian && !robust) {
        boost::math::students_t_distribution<double> t(f.observations - f.parameters);
        double l = boost::math::cdf(t, z), r = boost::math::cdf(boost::math::complement(t, z));
        p = side == Side::left ? l : side == Side::right ? r : std::min(1.0, 2 * std::min(l, r));
    }
    return {z, p, estimate, se, true};
}
// log2 fold change of a one-row hypothesis for log-link fits (missing otherwise).
inline TestResult log2_fold_change(const FitResult &fit, const Hypothesis &h, bool robust = false) {
    require(fit.family == Family::poisson || fit.family == Family::negative_binomial,
            "fold change requires a log link");
    if (h.df() != 1)
        return {};
    auto t = hypothesis_wald(fit, h, Side::both, robust);
    t.estimate /= std::log(2.0);
    t.standard_error /= std::log(2.0);
    return t;
}
} // namespace rapidglm
