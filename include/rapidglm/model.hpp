#pragma once
#include "lapack.hpp"
#include <algorithm>
#include <array>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rapidglm {
using Vector = std::vector<double>;
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
inline void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
struct Matrix {
    std::size_t rows = 0, cols = 0;
    Vector values;
    Matrix() = default;
    Matrix(std::size_t n, std::size_t p, double value = 0) : rows(n), cols(p) {
        require(p == 0 || n <= std::numeric_limits<std::size_t>::max() / p, "matrix size overflow");
        values.assign(n * p, value);
    }
    double &operator()(std::size_t i, std::size_t j) {
        return values[i * cols + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return values[i * cols + j];
    }
    void validate() const {
        require(cols == 0 || rows <= std::numeric_limits<std::size_t>::max() / cols,
                "matrix size overflow");
        require(values.size() == rows * cols, "matrix storage size mismatch");
        for (double x : values)
            require(std::isfinite(x), "nonfinite design");
    }
};
enum class Family { gaussian, poisson, binomial, negative_binomial };
enum class Dispersion { ml, moment, fixed };
enum class FitMethod { direct, sceptre_null };
// automatic: Cholesky of X'WX for well-conditioned steps, Householder QR of
// sqrt(W)X for rank detection, ill-conditioned steps and final covariance.
// qr: QR for every step.
enum class Solver { automatic, qr };
enum class FitStatus {
    converged,
    converged_boundary,
    rank_deficient,
    boundary_response,
    separation,
    iteration_limit,
    numerical_failure
};
inline const char *status_name(FitStatus s) {
    switch (s) {
    case FitStatus::converged:
        return "converged";
    case FitStatus::converged_boundary:
        return "converged_boundary";
    case FitStatus::rank_deficient:
        return "rank_deficient";
    case FitStatus::boundary_response:
        return "boundary_response";
    case FitStatus::separation:
        return "separation";
    case FitStatus::iteration_limit:
        return "iteration_limit";
    default:
        return "numerical_failure";
    }
}
namespace detail {
struct DesignSummary;
}
struct FitOptions {
    Family family = Family::negative_binomial;
    Dispersion dispersion = Dispersion::ml;
    double theta = 10, theta_min = 1e-6, theta_max = 1e8;
    double tolerance = 1e-9, ridge = 0, rank_tolerance = 1e-12;
    std::size_t max_iterations = 100, max_dispersion_iterations = 60;
    Vector offset, weights, start;
    bool robust = false, retain_observations = true;
    // For NB ML only: use supplied coefficients and theta instead of a new
    // Poisson/moment pilot. The usual joint optimization still runs.
    bool nb_warm_start = false;
    Solver solver = Solver::automatic;
    // A column is aliased when its weighted norm after projection on the retained
    // preceding columns is below alias_tolerance times its own norm; 1e-11 is the
    // threshold R's glm shows with its default epsilon. singular_ok fits the
    // remaining columns and reports aliased coefficients as NaN (R's NA);
    // otherwise an aliased design is a rank_deficient failure.
    double alias_tolerance = 1e-11;
    bool singular_ok = false;
    // NB ML dispersion only: maximize the Cox-Reid adjusted profile likelihood,
    // log L - log det(X'WX)/2 at the current means (McCarthy, Chen and Smyth 2012),
    // which reduces the small-sample downward bias of the dispersion.
    bool cox_reid = false;
    // What a fit derives from the design alone, from prepare_design(x): fits of
    // many responses to one design then compute it once. Used only for that
    // design, held unchanged, and only without prior weights; results are the
    // same with or without it.
    std::shared_ptr<const detail::DesignSummary> design;
};
struct FitResult {
    Family family = Family::gaussian;
    FitMethod method = FitMethod::direct;
    FitStatus status = FitStatus::numerical_failure;
    Vector coefficients, standard_errors, fitted, score_residuals, working_weights;
    // Finite iteration values are retained for diagnostics, not reported as
    // finite MLEs when the corresponding boundary flag is set.
    std::vector<unsigned char> boundary_coefficients;
    // Columns dropped as linear combinations of preceding columns (singular_ok).
    std::vector<unsigned char> aliased;
    Matrix covariance, robust_covariance;
    double theta = nan, scale = 1, log_likelihood = nan, ridge = 0;
    std::size_t iterations = 0, dispersion_iterations = 0, observations = 0, parameters = 0;
    std::size_t rank = 0; // number of non-aliased columns
    bool theta_boundary = false, rank_deficient = false, used_qr = false;
    // The final theta maximized the Cox-Reid adjusted likelihood (false when the
    // adjustment was not requested or its information matrix was singular).
    bool cox_reid = false;
    std::uint64_t response_signature = 0;
    bool ok() const {
        return status == FitStatus::converged || status == FitStatus::converged_boundary;
    }
    bool has_boundary() const {
        return std::any_of(boundary_coefficients.begin(), boundary_coefficients.end(),
                           [](unsigned char v) { return v != 0; });
    }
    void compact() {
        Vector().swap(fitted);
        Vector().swap(score_residuals);
        Vector().swap(working_weights);
    }
};
namespace detail {
// Triangular work (Cholesky factors, their solves and inverses, the QR back
// substitution) is done in-house at every order. Defining RAPIDGLM_LAPACK_SOLVES
// (CMake option of the same name) hands orders above Cholesky::small, and the
// QR triangular solves, to LAPACK instead, which is faster for designs with
// hundreds of columns but makes those results depend on the linked BLAS: its
// level-3 kernels are chosen per processor and are not all equally reliable.
struct Cholesky {
    static constexpr std::size_t small = 16; // scratch stays on the stack up to this order
    static constexpr bool in_house(std::size_t order) {
#ifdef RAPIDGLM_LAPACK_SOLVES
        return order <= small;
#else
        (void)order;
        return true;
#endif
    }
    Matrix l;
    Vector scale;
    bool ok = false;
    // Reciprocal 1-norm condition number of the scaled matrix (LAPACK's estimate
    // above `small` with RAPIDGLM_LAPACK_SOLVES); left at 0 when the constructor
    // is asked not to compute it.
    double rcond = 0;
    // Solves (L L') x = v in place for the scaled matrix (in-house path).
    void substitute(Vector &v) const {
        substitute(v.data(), v.size());
    }
    void substitute(double *v, std::size_t p) const {
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t k = 0; k < i; ++k)
                v[i] -= l(i, k) * v[k];
            v[i] /= l(i, i);
        }
        for (std::size_t i = p; i-- > 0;) {
            for (std::size_t k = i + 1; k < p; ++k)
                v[i] -= l(k, i) * v[k];
            v[i] /= l(i, i);
        }
    }
    explicit Cholesky(const Matrix &a, double tol = 1e-12, bool condition = true)
        : l(a.rows, a.cols), scale(a.rows) {
        const auto p = a.rows;
        if (p != a.cols ||
            p > static_cast<std::size_t>(std::numeric_limits<lapack::integer>::max()) ||
            a.values.size() != p * p || !std::isfinite(tol) || tol < 0)
            return;
        if (p == 0) {
            ok = true;
            return;
        }
        for (std::size_t j = 0; j < p; ++j) {
            if (!(a(j, j) > 0) || !std::isfinite(a(j, j)))
                return;
            scale[j] = std::sqrt(a(j, j));
        }
        // Small scratch stays on the stack: these factorizations run millions of times.
        double sums_small[small] = {};
        Vector sums_large;
        double *column_sums =
            p <= small ? sums_small : (sums_large.assign(p, 0.0), sums_large.data());
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k) {
                l(j, k) = a(j, k) / scale[j] / scale[k];
                if (!std::isfinite(l(j, k)))
                    return;
                column_sums[j] += std::abs(l(j, k));
                if (k < j)
                    column_sums[k] += std::abs(l(j, k));
            }
        const double norm = *std::max_element(column_sums, column_sums + p);
        if (in_house(p)) {
            // Factored in place without LAPACK: per-call LAPACK overhead and
            // OpenBLAS's shared buffer lock dominate small problems and serialize
            // concurrent callers. The reciprocal condition number is exact, from
            // the explicit inverse.
            for (std::size_t j = 0; j < p; ++j) {
                double d = l(j, j);
                for (std::size_t k = 0; k < j; ++k)
                    d -= l(j, k) * l(j, k);
                if (!(d > 0))
                    return;
                l(j, j) = std::sqrt(d);
                for (std::size_t i = j + 1; i < p; ++i) {
                    double v = l(i, j);
                    for (std::size_t k = 0; k < j; ++k)
                        v -= l(i, k) * l(j, k);
                    l(i, j) = v / l(j, j);
                }
            }
            for (std::size_t j = 0; j < p; ++j)
                if (!std::isfinite(l(j, j)) || !(l(j, j) * l(j, j) > tol))
                    return;
            ok = true;
            if (!condition)
                return;
            double inverse_norm = 0, e_small[small];
            Vector e_large;
            double *e = p <= small ? e_small : (e_large.resize(p), e_large.data());
            for (std::size_t c = 0; c < p; ++c) {
                std::fill(e, e + p, 0.0);
                e[c] = 1;
                substitute(e, p);
                double sum = 0;
                for (std::size_t k = 0; k < p; ++k)
                    sum += std::abs(e[k]);
                inverse_norm = std::max(inverse_norm, sum);
            }
            rcond = 1 / (norm * inverse_norm);
            return;
        }
        // Row-major lower storage is column-major upper storage. This avoids
        // a row-major copy without changing our Matrix representation.
        const auto n = static_cast<lapack::integer>(p);
        if (lapack::potrf('U', n, l.values.data(), n) != 0)
            return;
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isfinite(l(j, j)) || !(l(j, j) * l(j, j) > tol))
                return;
        ok = true;
        if (!condition)
            return;
        Vector work(3 * p);
        std::vector<lapack::integer> iwork(p);
        if (lapack::pocon('U', n, l.values.data(), n, norm, &rcond, work.data(), iwork.data()) != 0)
            rcond = 0;
    }
    Vector solve(const Vector &b) const {
        require(ok && b.size() == scale.size(), "invalid linear solve");
        auto v = b;
        const auto p = v.size();
        for (std::size_t j = 0; j < p; ++j)
            v[j] /= scale[j];
        const auto n = static_cast<lapack::integer>(p);
        if (in_house(p))
            substitute(v);
        else if (p && lapack::potrs('U', n, 1, l.values.data(), n, v.data(), n) != 0)
            throw std::runtime_error("LAPACK Cholesky solve failed");
        for (std::size_t j = 0; j < p; ++j)
            v[j] /= scale[j];
        return v;
    }
    Matrix inverse() const {
        require(ok, "invalid matrix inverse");
        const auto p = scale.size();
        if (in_house(p)) {
            Matrix a(p, p);
            double e_small[small];
            Vector e_large;
            double *e = p <= small ? e_small : (e_large.resize(p), e_large.data());
            for (std::size_t c = 0; c < p; ++c) {
                std::fill(e, e + p, 0.0);
                e[c] = 1;
                substitute(e, p);
                for (std::size_t j = 0; j < p; ++j)
                    a(j, c) = e[j] / scale[j] / scale[c];
            }
            // Symmetrize the explicitly solved columns.
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = 0; k < j; ++k)
                    a(j, k) = a(k, j) = 0.5 * (a(j, k) + a(k, j));
            return a;
        }
        Matrix a = l;
        const auto n = static_cast<lapack::integer>(p);
        if (p && lapack::potri('U', n, a.values.data(), n) != 0)
            throw std::runtime_error("LAPACK Cholesky inverse failed");
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                a(k, j) = a(j, k) = a(j, k) / scale[j] / scale[k];
        return a;
    }
};
// Householder QR (LAPACK dgeqrf) of sqrt(w) X, optionally stacked on sqrt(ridge) I.
// Columns are kept in order; a column whose diagonal R entry falls below tol
// times its weighted norm is aliased and the factorization is repeated
// without it. This is the limited pivoting R documents for base::qr.
struct WeightedQR {
    std::size_t rows = 0;
    std::vector<std::size_t> columns; // retained design columns, in order
    std::vector<unsigned char> aliased;
    Vector factor, tau; // column-major rows x columns.size()
    bool ok = false;
    WeightedQR(const Matrix &x, const Vector &w, double tol, double ridge = 0)
        : aliased(x.cols, 0) {
        const std::size_t n = x.rows, extra = ridge > 0 ? x.cols : 0;
        rows = n + extra;
        if (rows > static_cast<std::size_t>(std::numeric_limits<lapack::integer>::max()))
            return;
        std::vector<std::size_t> candidates(x.cols);
        std::iota(candidates.begin(), candidates.end(), 0);
        while (!candidates.empty()) {
            const std::size_t q = candidates.size();
            factor.assign(rows * q, 0);
            Vector norms(q);
            for (std::size_t k = 0; k < q; ++k) {
                double *column = factor.data() + k * rows;
                for (std::size_t i = 0; i < n; ++i) {
                    column[i] = std::sqrt(std::max(w[i], 0.0)) * x(i, candidates[k]);
                    norms[k] += column[i] * column[i];
                }
                if (extra) {
                    column[n + candidates[k]] = std::sqrt(ridge);
                    norms[k] += ridge;
                }
                norms[k] = std::sqrt(norms[k]);
            }
            tau.assign(q, 0);
            const auto m = static_cast<lapack::integer>(rows), c = static_cast<lapack::integer>(q);
            double query = 0;
            if (lapack::geqrf(m, c, factor.data(), m, tau.data(), &query, -1) != 0)
                return;
            Vector work(std::max<std::size_t>(1, static_cast<std::size_t>(query)));
            if (lapack::geqrf(m, c, factor.data(), m, tau.data(), work.data(),
                              static_cast<lapack::integer>(work.size())) != 0)
                return;
            std::size_t drop = q;
            for (std::size_t k = 0; k < q && drop == q; ++k)
                if (k >= rows || !(std::abs(factor[k * rows + k]) > tol * norms[k]))
                    drop = k;
            if (drop == q) {
                columns = std::move(candidates);
                ok = true;
                return;
            }
            aliased[candidates[drop]] = 1;
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(drop));
        }
    }
    std::size_t rank() const {
        return columns.size();
    }
    // Reciprocal 1-norm condition estimate of R, i.e. of the weighted design.
    double rcond() const {
        const std::size_t q = rank();
        if (!ok || q == 0)
            return 0;
        double value = 0;
        Vector work(3 * q);
        std::vector<lapack::integer> iwork(q);
        if (lapack::trcon('1', 'U', 'N', static_cast<lapack::integer>(q), factor.data(),
                          static_cast<lapack::integer>(rows), &value, work.data(),
                          iwork.data()) != 0)
            return 0;
        return value;
    }
    // Least-squares coefficients for the retained columns given the stacked
    // right-hand side u (length rows).
    Vector solve(Vector u) const {
        require(ok && u.size() == rows, "invalid QR solve");
        const auto m = static_cast<lapack::integer>(rows), q = static_cast<lapack::integer>(rank());
        double query = 0;
        if (lapack::ormqr('L', 'T', m, 1, q, factor.data(), m, tau.data(), u.data(), m, &query,
                          -1) != 0)
            throw std::runtime_error("LAPACK QR apply failed");
        Vector work(std::max<std::size_t>(1, static_cast<std::size_t>(query)));
        if (lapack::ormqr('L', 'T', m, 1, q, factor.data(), m, tau.data(), u.data(), m, work.data(),
                          static_cast<lapack::integer>(work.size())) != 0)
            throw std::runtime_error("LAPACK QR apply failed");
#ifdef RAPIDGLM_LAPACK_SOLVES
        if (lapack::trtrs('U', 'N', 'N', q, 1, factor.data(), m, u.data(), m) != 0)
            throw std::runtime_error("LAPACK QR solve failed");
#else
        // Back substitution with the column-major upper factor R.
        for (std::size_t i = rank(); i-- > 0;) {
            double value = u[i];
            for (std::size_t k = i + 1; k < rank(); ++k)
                value -= factor[k * rows + i] * u[k];
            const double diagonal = factor[i * rows + i];
            if (diagonal == 0)
                throw std::runtime_error("QR solve failed: singular factor");
            u[i] = value / diagonal;
        }
#endif
        u.resize(rank());
        return u;
    }
    // (R'R)^-1 for the retained columns, row-major.
    Matrix inverse() const {
        require(ok, "invalid QR inverse");
        const std::size_t q = rank();
        Matrix r(q, q);
        // Column-major upper R is row-major lower storage of R'.
        for (std::size_t j = 0; j < q; ++j)
            for (std::size_t i = 0; i <= j; ++i)
                r(j, i) = factor[j * rows + i];
#ifdef RAPIDGLM_LAPACK_SOLVES
        const auto n = static_cast<lapack::integer>(q);
        if (q && lapack::potri('U', n, r.values.data(), n) != 0)
            throw std::runtime_error("LAPACK QR inverse failed");
        for (std::size_t j = 0; j < q; ++j)
            for (std::size_t k = 0; k < j; ++k)
                r(k, j) = r(j, k);
        return r;
#else
        // Column c of (R'R)^-1: solve R' z = e_c, then R x = z, with L = R' in r.
        Matrix a(q, q);
        Vector e(q);
        for (std::size_t c = 0; c < q; ++c) {
            std::fill(e.begin(), e.end(), 0.0);
            e[c] = 1;
            for (std::size_t i = 0; i < q; ++i) {
                for (std::size_t k = 0; k < i; ++k)
                    e[i] -= r(i, k) * e[k];
                if (r(i, i) == 0)
                    throw std::runtime_error("QR inverse failed: singular factor");
                e[i] /= r(i, i);
            }
            for (std::size_t i = q; i-- > 0;) {
                for (std::size_t k = i + 1; k < q; ++k)
                    e[i] -= r(k, i) * e[k];
                e[i] /= r(i, i);
            }
            for (std::size_t j = 0; j < q; ++j)
                a(j, c) = e[j];
        }
        for (std::size_t j = 0; j < q; ++j)
            for (std::size_t k = 0; k < j; ++k)
                a(j, k) = a(k, j) = 0.5 * (a(j, k) + a(k, j));
        return a;
#endif
    }
};
// X' diag(w) X, lower triangle mirrored. Four rows per pass combine four rank-one
// updates per accumulator element, which roughly halves the time for p >= 10
// without a BLAS dependency.
inline Matrix crossprod(const Matrix &x, const Vector &w) {
    const std::size_t n = x.rows, p = x.cols;
    Matrix a(p, p);
    if (p == 0)
        return a;
    double v_small[64];
    Vector v_large;
    double *v = 4 * p <= 64 ? v_small : (v_large.resize(4 * p), v_large.data());
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const double *r0 = &x.values[i * p], *r1 = r0 + p, *r2 = r1 + p, *r3 = r2 + p;
        for (std::size_t j = 0; j < p; ++j) {
            v[j] = w[i] * r0[j];
            v[p + j] = w[i + 1] * r1[j];
            v[2 * p + j] = w[i + 2] * r2[j];
            v[3 * p + j] = w[i + 3] * r3[j];
        }
        for (std::size_t j = 0; j < p; ++j) {
            const double a0 = v[j], a1 = v[p + j], a2 = v[2 * p + j], a3 = v[3 * p + j];
            double *row = &a.values[j * p];
            for (std::size_t k = 0; k <= j; ++k)
                row[k] += a0 * r0[k] + a1 * r1[k] + a2 * r2[k] + a3 * r3[k];
        }
    }
    for (; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            const double a0 = x(i, j) * w[i];
            for (std::size_t k = 0; k <= j; ++k)
                a(j, k) += a0 * x(i, k);
        }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < j; ++k)
            a(k, j) = a(j, k);
    return a;
}
// The nonzero entries of each design row, by increasing column. A design of
// factors is mostly zeros (one indicator per factor in each row), and X'WX then
// needs only the products among each row's nonzero entries. `sparse` says
// whether that is worth it: at most half of the entries are nonzero.
struct SparseRows {
    std::vector<std::size_t> start; // per row, into the two below; rows + 1 entries
    std::vector<std::uint32_t> column;
    Vector value;
    bool sparse = false;
    SparseRows() = default;
    explicit SparseRows(const Matrix &x) {
        const std::size_t n = x.rows, p = x.cols;
        std::size_t nonzero = 0;
        for (double v : x.values)
            nonzero += v != 0;
        if (p == 0 || 2 * nonzero > n * p)
            return;
        start.reserve(n + 1);
        column.reserve(nonzero);
        value.reserve(nonzero);
        for (std::size_t i = 0; i < n; ++i) {
            start.push_back(column.size());
            for (std::size_t j = 0; j < p; ++j)
                if (x(i, j) != 0) {
                    column.push_back(static_cast<std::uint32_t>(j));
                    value.push_back(x(i, j));
                }
        }
        start.push_back(column.size());
        sparse = true;
    }
};
// X' diag(w) X with the rows' nonzero entries when `rows` (of x) says the design
// is sparse; the dense product otherwise.
inline Matrix crossprod(const Matrix &x, const Vector &w, const SparseRows *rows) {
    if (!rows || !rows->sparse)
        return crossprod(x, w);
    const std::size_t n = x.rows, p = x.cols;
    Matrix a(p, p);
    for (std::size_t i = 0; i < n; ++i) {
        const double wi = w[i];
        if (wi == 0)
            continue;
        const std::size_t first = rows->start[i], last = rows->start[i + 1];
        for (std::size_t s = first; s < last; ++s) {
            const double v = wi * rows->value[s];
            double *row = &a.values[rows->column[s] * p];
            for (std::size_t t = first; t <= s; ++t)
                row[rows->column[t]] += v * rows->value[t];
        }
    }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < j; ++k)
            a(k, j) = a(j, k);
    return a;
}
inline Vector xtv(const Matrix &x, const Vector &v) {
    Vector r(x.cols);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t j = 0; j < x.cols; ++j)
            r[j] += x(i, j) * v[i];
    return r;
}
inline double dot(const Vector &a, const Vector &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
inline Vector matvec(const Matrix &a, const Vector &b) {
    Vector v(a.rows);
    for (std::size_t i = 0; i < a.rows; ++i)
        for (std::size_t j = 0; j < a.cols; ++j)
            v[i] += a(i, j) * b[j];
    return v;
}
// X'v and X b over the rows' nonzero entries when `rows` (of x) says the design
// is sparse (see SparseRows); the dense products otherwise.
inline Vector xtv(const Matrix &x, const Vector &v, const SparseRows *rows) {
    if (!rows || !rows->sparse)
        return xtv(x, v);
    Vector r(x.cols);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t s = rows->start[i]; s < rows->start[i + 1]; ++s)
            r[rows->column[s]] += rows->value[s] * v[i];
    return r;
}
inline Vector matvec(const Matrix &x, const Vector &b, const SparseRows *rows) {
    if (!rows || !rows->sparse)
        return matvec(x, b);
    Vector v(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t s = rows->start[i]; s < rows->start[i + 1]; ++s)
            v[i] += rows->value[s] * b[rows->column[s]];
    return v;
}
// Designs whose rows take exactly p = cols distinct values (intercept-only,
// one-way and full factorial layouts) have eta constant within each group of
// equal rows. With U the p x p matrix of distinct rows, X'WX = U' diag(W_g) U and
// X's = U' S_g, so the Fisher step is U^-1 (S_g / W_g): per-group sums replace
// the cross-product and its factorization.
struct RowGroups {
    std::vector<std::size_t> group; // group of each row
    Matrix rows, inverse;           // U and U^-1
    double resolution = 0;          // relative rounding level of U^-1 b
    bool ok = false;
    RowGroups() = default;
    explicit RowGroups(const Matrix &x) {
        const std::size_t n = x.rows, p = x.cols;
        if (p == 0 || n < p)
            return;
        std::vector<std::size_t> first; // first row of each group
        group.resize(n);
        std::size_t last = 0;
        const auto same = [&](std::size_t i, std::size_t k) {
            return std::equal(&x.values[i * p], &x.values[i * p] + p, &x.values[first[k] * p]);
        };
        for (std::size_t i = 0; i < n; ++i) {
            // Rows usually arrive grouped or alternating, so the last match is tried first.
            std::size_t id = first.size();
            if (!first.empty() && same(i, last))
                id = last;
            else
                for (std::size_t k = 0; k < first.size(); ++k)
                    if (k != last && same(i, k)) {
                        id = k;
                        break;
                    }
            if (id == first.size()) {
                if (first.size() == p)
                    return;
                first.push_back(i);
            }
            group[i] = last = id;
        }
        if (first.size() != p)
            return;
        rows = Matrix(p, p);
        for (std::size_t k = 0; k < p; ++k)
            for (std::size_t j = 0; j < p; ++j)
                rows(k, j) = x(first[k], j);
        WeightedQR qr(rows, Vector(p, 1), 1e-11);
        const double rcond = qr.ok && qr.rank() == p ? qr.rcond() : 0;
        if (!(rcond > 1e-10))
            return;
        inverse = Matrix(p, p);
        for (std::size_t j = 0; j < p; ++j) {
            Vector e(p);
            e[j] = 1;
            const auto column = qr.solve(e);
            for (std::size_t k = 0; k < p; ++k)
                inverse(k, j) = column[k];
        }
        resolution = 100 * std::numeric_limits<double>::epsilon() / rcond;
        ok = true;
    }
};
// The parts of a fit that depend on the design alone: its validation, its
// nonzero entries by row, X'X with the factor that checks its rank and gives
// the least-squares start, and its row groups.
struct DesignSummary {
    const double *values; // the storage of the design described
    std::size_t rows, cols;
    double rank_tolerance;
    SparseRows nonzero;
    Matrix information;
    Cholesky check;
    RowGroups groups;
    DesignSummary(const Matrix &x, double tolerance)
        : values(x.values.data()), rows(x.rows), cols(x.cols), rank_tolerance(tolerance),
          nonzero(x), information(crossprod(x, Vector(x.rows, 1), &nonzero)),
          check(information, tolerance), groups(x) {}
    bool describes(const Matrix &x, double tolerance) const {
        return values == x.values.data() && rows == x.rows && cols == x.cols &&
               rank_tolerance == tolerance;
    }
};
inline double mean_link(double eta, Family family) {
    if (family == Family::gaussian)
        return eta;
    if (family == Family::binomial) {
        double z = std::exp(-std::abs(eta));
        return std::clamp(eta >= 0 ? 1 / (1 + z) : z / (1 + z), 1e-15, 1 - 1e-15);
    }
    if (eta > 700)
        return std::numeric_limits<double>::infinity();
    return std::exp(std::max(-700.0, eta));
}
inline Vector means(const Matrix &x, const Vector &beta, const Vector &offset, Family family,
                    const SparseRows *rows = nullptr) {
    auto mu = matvec(x, beta, rows);
    for (std::size_t i = 0; i < mu.size(); ++i)
        mu[i] = mean_link(mu[i] + offset[i], family);
    return mu;
}
// Same values as means(x, ...): each group's eta is the same row product.
inline Vector means(const RowGroups &g, const Vector &beta, const Vector &offset, Family family) {
    const auto eta = matvec(g.rows, beta);
    Vector mu(g.group.size());
    for (std::size_t i = 0; i < mu.size(); ++i)
        mu[i] = mean_link(eta[g.group[i]] + offset[i], family);
    return mu;
}
// log(1 + r) for r > -1 from one logarithm: log(u) r / (u - 1) with u = 1 + r
// rounded cancels the rounding of u (Kahan), and is cheaper than log1p.
inline double log1p_by_log(double r) {
    const double u = 1 + r;
    return u == 1 ? r : std::log(u) * (r / (u - 1));
}
// Stirling's series, log Gamma(z) = (z - 1/2) log z - z + log(2 pi)/2 + c(z), for
// z >= 65 in double: c(z) = 1/(12 z) - 1/(360 z^3) + 1/(1260 z^5); the next term
// is below 2e-16.
inline constexpr double half_log_two_pi = 0.91893853320467274178;
inline double stirling_correction(double z) {
    const double r = 1 / z, r2 = r * r;
    return r * (1.0 / 12 - r2 * (1.0 / 360 - r2 / 1260));
}
// The same expansion for digamma(z) - log z and trigamma(z) - 1/z, z >= 65:
// -1/(2z) - 1/(12 z^2) + 1/(120 z^4) - 1/(252 z^6) and
// 1/(2 z^2) + 1/(6 z^3) - 1/(30 z^5) + 1/(42 z^7); the next terms are below 2e-17.
inline double digamma_correction(double z) {
    const double r = 1 / z, r2 = r * r;
    return -r * (0.5 + r * (1.0 / 12 - r2 * (1.0 / 120 - r2 / 252)));
}
inline double trigamma_correction(double z) {
    const double r = 1 / z, r2 = r * r;
    return r2 * (0.5 + r * (1.0 / 6 - r2 * (1.0 / 30 - r2 / 42)));
}
// log(v!) for v <= 64.
inline const std::array<long double, 65> &small_log_factorials() {
    static const std::array<long double, 65> table = [] {
        std::array<long double, 65> t{};
        for (std::size_t v = 2; v <= 64; ++v)
            t[v] = std::lgamma(v + 1.L);
        return t;
    }();
    return table;
}
// Response-only terms of the count likelihoods, computed once per response so
// each evaluation needs only the mean-dependent terms. Counts up to 64 are
// handled through their histogram, exactly; larger ones by Stirling's series,
// one logarithm each. Sums use long double.
struct CountTerms {
    // above[k]: total weight of observations with k < y <= 64.
    std::array<long double, 64> above{};
    // A count above 64: y, its weight, log(y + 1) and the Stirling correction at y + 1.
    struct Large {
        double count, weight, log_next, correction;
        double log_factorial() const { // log(y!)
            return (count + 0.5) * log_next - (count + 1) + half_log_two_pi + correction;
        }
    };
    std::vector<Large> large;
    long double log_factorial = 0;       // sum a lgamma(y + 1)
    long double small_log_factorial = 0; // its part from the counts up to 64
    long double weight = 0;              // sum a
    mutable double cached_theta = std::numeric_limits<double>::quiet_NaN();
    mutable long double cached_constant = 0;
    CountTerms(const Vector &y, const Vector &a) {
        std::array<long double, 65> small{};
        long double large_log_factorial = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            if (a[i] > 0) {
                weight += a[i];
                if (y[i] > 64) {
                    const double next = y[i] + 1;
                    large.push_back({y[i], a[i], std::log(next), stirling_correction(next)});
                    large_log_factorial += a[i] * large.back().log_factorial();
                } else
                    small[static_cast<std::size_t>(y[i])] += a[i];
            }
        const auto &factorials = small_log_factorials();
        for (std::size_t v = 2; v <= 64; ++v)
            if (small[v] != 0)
                small_log_factorial += small[v] * factorials[v];
        log_factorial = small_log_factorial + large_log_factorial;
        for (std::size_t k = 64; k-- > 0;)
            above[k] = small[k + 1] + (k + 1 < 64 ? above[k + 1] : 0.L);
    }
    // sum a [lgamma(y + t) - lgamma(t) - lgamma(y + 1)], the response-only part of
    // the NB log likelihood. Fits at a fixed theta need it at every step, so the
    // last value is kept (CountTerms objects are never shared).
    //
    // For a count above 64 the three values are large and nearly cancel, so each
    // observation uses the form whose terms are of the size of the result, as the
    // mean-dependent terms do (see likelihood below). With c the Stirling
    // correction:
    // - t <= y + 1:  lgamma(y + t) - lgamma(y + 1)
    //       = (t - 1) log(y + 1) + (y + t - 1/2) log1p((t - 1)/(y + 1)) - (t - 1)
    //         + c(y + t) - c(y + 1),  and lgamma(t) is subtracted once for all;
    // - t > y + 1:   lgamma(y + t) - lgamma(t)
    //       = y log t + (y + t - 1/2) log1p(y/t) - y + c(y + t) - c(t),
    //   where y log t is summed apart: it cancels against the mean-dependent
    //   terms in long double (see likelihood), and log(y!) does not depend on t.
    long double nb_constant(double theta) const {
        if (theta == cached_theta)
            return cached_constant;
        const long double t = theta;
        long double total = -small_log_factorial;
        for (unsigned k = 0; k < 64; ++k)
            if (above[k] != 0)
                total += above[k] * std::log(t + k);
        if (!large.empty()) {
            const double correction = theta >= 65 ? stirling_correction(theta) : 0;
            long double below = 0, below_weight = 0, beyond = 0, beyond_counts = 0, factorials = 0;
            for (const auto &e : large) {
                const double z = theta + e.count;
                if (theta <= e.count + 1) {
                    const double shift = theta - 1;
                    below += e.weight *
                             (shift * e.log_next + (z - 0.5) * log1p_by_log(shift / (e.count + 1)) -
                              shift + (stirling_correction(z) - e.correction));
                    below_weight += e.weight;
                } else { // theta > y + 1 >= 66
                    beyond += e.weight * ((z - 0.5) * log1p_by_log(e.count / theta) - e.count +
                                          (stirling_correction(z) - correction));
                    beyond_counts += e.weight * e.count;
                    factorials += e.weight * e.log_factorial();
                }
            }
            if (beyond_counts != 0)
                total += beyond_counts * std::log(t) - factorials;
            if (below_weight != 0)
                total -= below_weight * std::lgamma(t);
            total += below + beyond;
        }
        cached_theta = theta;
        cached_constant = total;
        return total;
    }
    // sum a [digamma(y + t) - digamma(t)] and the trigamma difference.
    std::pair<long double, long double> polygamma(double theta) const {
        const long double t = theta;
        long double first = 0, second = 0;
        for (unsigned k = 0; k < 64; ++k)
            if (above[k] != 0) {
                const long double inverse = 1 / (t + k);
                first += above[k] * inverse;
                second -= above[k] * inverse * inverse;
            }
        if (!large.empty() && theta >= 65) {
            // Both arguments are in the range of the expansions; the leading terms
            // are differenced exactly: log(t + y) - log t = log1p(y/t) and
            // 1/(t + y) - 1/t = -y / (t (t + y)).
            const double digamma = digamma_correction(theta), trigamma = trigamma_correction(theta);
            for (const auto &e : large) {
                const double z = theta + e.count;
                first +=
                    e.weight * (log1p_by_log(e.count / theta) + (digamma_correction(z) - digamma));
                second += e.weight * (-e.count / (theta * z) + (trigamma_correction(z) - trigamma));
            }
        } else if (!large.empty()) {
            // y > 64 here, so t + y >= 65; the values at t are needed once.
            long double weights = 0;
            for (const auto &e : large) {
                const double z = theta + e.count;
                first += e.weight * (std::log(z) + digamma_correction(z));
                second += e.weight * (1 / z + trigamma_correction(z));
                weights += e.weight;
            }
            first -= weights * boost::math::digamma(t);
            second -= weights * boost::math::trigamma(t);
        }
        return {first, second};
    }
};
// NB log mass: lgamma(y+t) - lgamma(t) - lgamma(y+1) + y log(m/(t+m)) + t log(t/(t+m)).
// Mean-dependent terms are evaluated in double; each observation uses the grouping
// whose double-precision terms are small, and y log(t) cancels in long double.
inline double likelihood(const Vector &y, const Vector &mu, const Vector &a, Family f, double theta,
                         const CountTerms &terms) {
    long double ll = 0, log_theta_weight = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (a[i] > 0) {
            const double m = mu[i], v = y[i];
            if (!std::isfinite(m))
                return -std::numeric_limits<double>::infinity();
            if (f == Family::gaussian)
                ll -= 0.5L * a[i] * (v - m) * (v - m);
            else if (f == Family::binomial)
                ll += a[i] * (v * std::log(m) + (1 - v) * std::log1p(-m));
            else if (f == Family::poisson)
                ll += a[i] * ((v > 0 ? v * std::log(m) : 0) - m);
            else {
                const double ratio = log1p_by_log(m / theta), shape = theta * ratio;
                if (v == 0)
                    ll -= a[i] * shape;
                else if (theta < m)
                    ll -= a[i] * (v * log1p_by_log(theta / m) + shape);
                else {
                    // y log(m/(t+m)) = y log(m) - y log(t) - y log1p(m/t)
                    ll += a[i] * (v * (std::log(m) - ratio) - shape);
                    log_theta_weight += a[i] * v;
                }
            }
        }
    if (f == Family::poisson)
        ll -= terms.log_factorial;
    else if (f == Family::negative_binomial)
        ll +=
            terms.nb_constant(theta) - log_theta_weight * std::log(static_cast<long double>(theta));
    return static_cast<double>(ll);
}
inline double likelihood(const Vector &y, const Vector &mu, const Vector &a, Family f,
                         double theta) {
    if (f == Family::gaussian || f == Family::binomial)
        return likelihood(y, mu, a, f, theta, CountTerms(Vector(), Vector()));
    return likelihood(y, mu, a, f, theta, CountTerms(y, a));
}
// Stable r - log(1 + r), including nearly saturated fits.
inline double log1pmx(double r) {
    if (std::abs(r) >= 1e-3)
        return r - log1p_by_log(r);
    return r * r * (0.5 + r * (-1.0 / 3 + r * (0.25 + r * (-0.2 + r / 6))));
}
// The deviance of a count fit, the scale of the relative convergence criterion.
// The unit deviances are v log(v/m) - (v - m) for Poisson, less
// (v + t) (d - log(1 + d)) with d = (m - v)/(v + t) for NB; both are written so
// that nearly saturated fits lose no digits. Double precision throughout: a
// scale needs few digits, and long double logarithms cost several times more.
inline double count_deviance(const Vector &y, const Vector &mu, const Vector &a, Family family,
                             double theta) {
    double total = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (a[i] == 0)
            continue;
        double term;
        if (y[i] == 0)
            term = family == Family::poisson ? mu[i] : theta * log1p_by_log(mu[i] / theta);
        else {
            const double v = y[i], m = mu[i], difference = m - v;
            term = std::abs(difference) < 0.125 * v ? v * log1pmx(difference / v)
                                                    : v * std::log(v / m) + difference;
            if (family == Family::negative_binomial)
                term -= (v + theta) * log1pmx(difference / (v + theta));
        }
        if (!std::isfinite(term))
            return std::numeric_limits<double>::infinity();
        total += 2 * a[i] * std::max(0.0, term);
    }
    return total;
}

// LAPACK supplies the null space; no custom decomposition. Column scaling is
// based on the complete positive-weight design, including its zero-count rows.
struct NullSpace {
    Vector scale;
    Matrix basis; // columns are null vectors in scaled coefficient coordinates
    NullSpace(const Matrix &x, const Vector &a, const std::vector<unsigned char> &keep) {
        std::size_t n = std::count(keep.begin(), keep.end(), 1), p = x.cols;
        require(n > 0 && n <= static_cast<std::size_t>(std::numeric_limits<lapack::integer>::max()),
                "invalid boundary diagnostic dimensions");
        scale.assign(p, 0);
        for (std::size_t i = 0; i < x.rows; ++i)
            if (a[i] > 0)
                for (std::size_t j = 0; j < p; ++j)
                    scale[j] = std::max(scale[j], std::abs(x(i, j)));
        Vector design(n * p), singular(std::min(n, p)), vt(p * p);
        for (std::size_t i = 0, row = 0; i < x.rows; ++i)
            if (keep[i]) {
                for (std::size_t j = 0; j < p; ++j)
                    design[j * n + row] = x(i, j) / scale[j];
                ++row;
            }
        const auto m = static_cast<lapack::integer>(n), columns = static_cast<lapack::integer>(p);
        double query = 0, unused = 0;
        auto svd = [&](double *work, lapack::integer length) {
            return lapack::gesvd('N', 'A', m, columns, design.data(), m, singular.data(), &unused,
                                 1, vt.data(), columns, work, length);
        };
        require(svd(&query, -1) == 0 && query > 0 &&
                    query <= std::numeric_limits<lapack::integer>::max(),
                "boundary SVD workspace query failed");
        Vector work(static_cast<std::size_t>(query));
        require(svd(work.data(), static_cast<lapack::integer>(work.size())) == 0,
                "boundary SVD failed");
        const double threshold =
            8 * std::numeric_limits<double>::epsilon() * std::max(n, p) * singular.front();
        std::size_t rank = 0;
        while (rank < singular.size() && singular[rank] > threshold)
            ++rank;
        basis = Matrix(p, p - rank);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = rank; k < p; ++k)
                basis(j, k - rank) = vt[k + j * p];
    }
};

// Whether the design restricted to positive counts has full rank; if so no
// coefficient can diverge. Depends only on the data, so fits compute it once.
inline bool positive_counts_full_rank(const Matrix &x, const Vector &y, const Vector &a,
                                      double rank_tolerance, const SparseRows *rows = nullptr) {
    Vector positive(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i)
        positive[i] = a[i] > 0 && y[i] > 0;
    return Cholesky(crossprod(x, positive, rows), rank_tolerance).ok;
}
inline std::vector<unsigned char> count_boundary(const Matrix &x, const Vector &y, const Vector &a,
                                                 const Vector &direction, double /*rank_tolerance*/,
                                                 bool positive_full_rank) {
    std::vector<unsigned char> flags(x.cols, 0), positive(x.rows, 0);
    for (std::size_t i = 0; i < x.rows; ++i)
        positive[i] = a[i] > 0 && y[i] > 0;
    // Most fits have full rank among positive counts: a boundary is impossible.
    if (positive_full_rank)
        return flags;
    NullSpace support(x, a, positive);
    Vector projected(x.cols);
    for (std::size_t k = 0; k < support.basis.cols; ++k) {
        double component = 0;
        for (std::size_t j = 0; j < x.cols; ++j)
            component += support.basis(j, k) * direction[j] * support.scale[j];
        for (std::size_t j = 0; j < x.cols; ++j)
            projected[j] += support.basis(j, k) * component;
    }
    double norm = 0;
    for (double v : projected)
        norm = std::max(norm, std::abs(v));
    if (!(norm > 0) || !std::isfinite(norm))
        return flags;
    for (double &v : projected)
        v /= norm;
    // Certify a recession direction: no change on positive counts, no increase
    // on zero counts, and a strict decrease on at least one zero-count row.
    const double roundoff = 128 * std::numeric_limits<double>::epsilon() * x.cols;
    std::vector<unsigned char> retained(x.rows, 0);
    bool separated = false;
    for (std::size_t i = 0; i < x.rows; ++i) {
        if (a[i] == 0)
            continue;
        double change = 0, row_norm = 0;
        for (std::size_t j = 0; j < x.cols; ++j) {
            double value = x(i, j) / support.scale[j];
            change += value * projected[j];
            row_norm += std::abs(value);
        }
        const double tolerance = roundoff * row_norm;
        if ((y[i] > 0 && std::abs(change) > tolerance) || change > tolerance)
            return flags;
        retained[i] = change >= -tolerance;
        separated |= !retained[i];
    }
    if (!separated)
        return flags;
    NullSpace face(x, a, retained);
    for (std::size_t j = 0; j < x.cols; ++j) {
        double component = 0;
        for (std::size_t k = 0; k < face.basis.cols; ++k)
            component = std::hypot(component, face.basis(j, k));
        flags[j] = component > roundoff;
    }
    return flags;
}
inline std::vector<unsigned char> count_boundary(const Matrix &x, const Vector &y, const Vector &a,
                                                 const Vector &direction, double rank_tolerance) {
    return count_boundary(x, y, a, direction, rank_tolerance,
                          positive_counts_full_rank(x, y, a, rank_tolerance));
}
inline void working(const Vector &y, const Vector &mu, const Vector &a, Family f, double theta,
                    Vector &w, Vector &s) {
    w.resize(y.size());
    s.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        double factor = f == Family::negative_binomial ? 1 / (1 + mu[i] / theta) : 1;
        w[i] = a[i] * (f == Family::gaussian   ? 1
                       : f == Family::binomial ? mu[i] * (1 - mu[i])
                                               : mu[i] * factor);
        s[i] = a[i] * (y[i] - mu[i]) * factor;
    }
}
inline double moment_theta(const Vector &y, const Vector &mu, const Vector &a,
                           const FitOptions &o) {
    double numerator = 0, denominator = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        numerator += a[i] * mu[i] * mu[i];
        denominator += a[i] * ((y[i] - mu[i]) * (y[i] - mu[i]) - mu[i]);
    }
    return denominator > 0 ? std::clamp(numerator / denominator, o.theta_min, o.theta_max)
                           : o.theta_max;
}
// Cox-Reid term -log det(X'WX)/2 of the NB coefficients' expected information at
// fixed means, W = diag(a mu theta / (theta + mu)), and its first two derivatives
// in theta. dW/dtheta = a mu^2/(theta+mu)^2 and d2W/dtheta2 = -2 a mu^2/(theta+mu)^3;
// d log det A = tr(A^-1 A') and d2 log det A = tr(A^-1 A'') - tr((A^-1 A')^2).
struct CoxReid {
    double value = nan, first = 0, second = 0;
    bool ok = false;
};
// Factor of the Cox-Reid information X'WX at theta (W the NB working weights).
// `rows` (may be null) holds the nonzero entries of x, as for crossprod.
inline Cholesky cox_reid_factor(const Matrix &x, const Vector &mu, const Vector &a, double theta,
                                const SparseRows *rows = nullptr) {
    Vector w(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double m = mu[i], d = theta + m;
        w[i] = a[i] * m * (theta / d);
    }
    return Cholesky(crossprod(x, w, rows), 1e-12, false);
}
// With `factor`, the caller supplies cox_reid_factor at the same theta.
inline CoxReid cox_reid(const Matrix &x, const Vector &mu, const Vector &a, double theta,
                        bool derivatives, const Cholesky *factor = nullptr,
                        const SparseRows *rows = nullptr) {
    CoxReid c;
    const std::size_t p = x.cols;
    if (p == 0) {
        c.value = 0;
        c.ok = true;
        return c;
    }
    Vector w1(derivatives ? x.rows : 0), w2(derivatives ? x.rows : 0);
    if (derivatives)
        for (std::size_t i = 0; i < x.rows; ++i) {
            const double m = mu[i], d = theta + m;
            w1[i] = a[i] * (m / d) * (m / d);
            w2[i] = -2 * w1[i] / d;
        }
    std::optional<Cholesky> own;
    if (!factor)
        factor = &own.emplace(cox_reid_factor(x, mu, a, theta, rows));
    const Cholesky &ch = *factor;
    if (!ch.ok)
        return c;
    // The factor is of the unit-diagonal scaling S^-1 A S^-1.
    double log_det = 0;
    for (std::size_t j = 0; j < p; ++j)
        log_det += 2 * (std::log(ch.scale[j]) + std::log(ch.l(j, j)));
    c.value = -0.5 * log_det;
    c.ok = std::isfinite(c.value);
    if (!c.ok || !derivatives)
        return c;
    const auto inverse = ch.inverse();
    const auto a1 = crossprod(x, w1, rows), a2 = crossprod(x, w2, rows);
    Matrix m(p, p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < p; ++k)
            for (std::size_t l = 0; l < p; ++l)
                m(j, k) += inverse(j, l) * a1(l, k);
    double first = 0, second = 0;
    for (std::size_t j = 0; j < p; ++j) {
        first += m(j, j);
        for (std::size_t k = 0; k < p; ++k)
            second += inverse(j, k) * a2(k, j) - m(j, k) * m(k, j);
    }
    c.first = -0.5 * first;
    c.second = -0.5 * second;
    return c;
}
// Normal prior on log theta, adding -(log theta - center)^2 / (2 variance).
struct LogThetaPrior {
    double center = 0, variance = 1;
};
// Newton in log theta at fixed means. With `adjust` (the design columns that are
// not at a boundary), the objective is the Cox-Reid adjusted likelihood; the
// caller checks that the adjustment is defined at the starting theta. With
// `prior`, the objective is the log posterior (a MAP estimate).
inline double estimate_theta(const Vector &y, const Vector &mu, const Vector &a, double theta,
                             const FitOptions &o, const CountTerms &terms,
                             const Matrix *adjust = nullptr, const LogThetaPrior *prior = nullptr,
                             const SparseRows *rows = nullptr) {
    // The factor of the last objective evaluation is reused by the next Newton
    // step, which starts at that theta.
    std::optional<Cholesky> factor;
    double factor_theta = std::numeric_limits<double>::quiet_NaN();
    const auto objective = [&](double t) {
        double v = likelihood(y, mu, a, Family::negative_binomial, t, terms);
        if (adjust) {
            factor.emplace(cox_reid_factor(*adjust, mu, a, t, rows));
            factor_theta = t;
            const auto c = cox_reid(*adjust, mu, a, t, false, &*factor, rows);
            v = c.ok ? v + c.value : -std::numeric_limits<double>::infinity();
        }
        if (prior) {
            const double d = std::log(t) - prior->center;
            v -= d * d / (2 * prior->variance);
        }
        return v;
    };
    double u = std::log(theta), ll = objective(theta);
    for (unsigned it = 0; it < 50; ++it) {
        const double t = std::exp(u);
        // Response-only polygamma sums come from the count histogram.
        auto [g, h] = terms.polygamma(t);
        h += terms.weight / t;
        for (std::size_t i = 0; i < y.size(); ++i)
            if (a[i] > 0) {
                const double v = y[i], m = mu[i], tm = t + m;
                g += a[i] * ((m - v) / tm - log1p_by_log(m / t));
                h += a[i] * ((v - m) / (tm * tm) - 1 / tm);
            }
        if (adjust) {
            const auto c = cox_reid(*adjust, mu, a, t, true,
                                    factor && factor_theta == t ? &*factor : nullptr, rows);
            if (!c.ok)
                break;
            g += c.first;
            h += c.second;
        }
        double gu = static_cast<double>(g * t), hu = static_cast<double>(h * t * t + g * t);
        if (prior) {
            gu -= (u - prior->center) / prior->variance;
            hu -= 1 / prior->variance;
        }
        if (std::abs(gu) < 1e-10)
            break;
        double step = hu < 0 ? -gu / hu : (gu > 0 ? 1 : -1);
        step = std::clamp(step, -2.0, 2.0);
        double unew = u, ln = ll;
        bool accepted = false;
        for (unsigned halve = 0; halve < 35 && !accepted; ++halve) {
            unew = std::clamp(u + step, std::log(o.theta_min), std::log(o.theta_max));
            ln = objective(std::exp(unew));
            accepted = ln >= ll - 1e-12 * (1 + std::abs(ll));
            step *= 0.5;
        }
        // Never move to a lower likelihood when step halving is exhausted.
        if (!accepted)
            break;
        double change = std::abs(unew - u);
        u = unew;
        ll = ln;
        if (change < o.tolerance * 0.1)
            break;
    }
    return std::exp(u);
}
inline double estimate_theta(const Vector &y, const Vector &mu, const Vector &a, double theta,
                             const FitOptions &o) {
    return estimate_theta(y, mu, a, theta, o, CountTerms(y, a));
}
// Identifies the response, weights and offset of a fit, so that fits can be
// checked to share them (not a cryptographic hash). One multiply per value.
inline std::uint64_t signature(const Vector &y, const Vector &a, const Vector &offset) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto *v : {&y, &a, &offset}) {
        for (double d : *v) {
            std::uint64_t u;
            std::memcpy(&u, &d, sizeof u);
            h = (h ^ u) * 0x9E3779B97F4A7C15ULL;
            h ^= h >> 29;
        }
        h = (h ^ v->size()) * 0xBF58476D1CE4E5B9ULL;
    }
    return h;
}
inline Matrix select_columns(const Matrix &x, const std::vector<std::size_t> &columns) {
    Matrix out(x.rows, columns.size());
    for (std::size_t i = 0; i < x.rows; ++i)
        for (std::size_t k = 0; k < columns.size(); ++k)
            out(i, k) = x(i, columns[k]);
    return out;
}
// Places a fit of the retained columns back in the full column space; aliased
// coefficients, standard errors and covariance entries are NaN.
inline FitResult expand_aliased(FitResult r, const std::vector<std::size_t> &columns,
                                const std::vector<unsigned char> &aliased) {
    const std::size_t p = aliased.size(), q = columns.size();
    auto spread = [&](const Vector &v) {
        Vector out(p, nan);
        if (v.size() == q)
            for (std::size_t k = 0; k < q; ++k)
                out[columns[k]] = v[k];
        return out;
    };
    auto spread_matrix = [&](const Matrix &m) {
        if (m.rows != q)
            return Matrix();
        Matrix out(p, p, nan);
        for (std::size_t j = 0; j < q; ++j)
            for (std::size_t k = 0; k < q; ++k)
                out(columns[j], columns[k]) = m(j, k);
        return out;
    };
    r.coefficients = spread(r.coefficients);
    r.standard_errors = spread(r.standard_errors);
    r.covariance = spread_matrix(r.covariance);
    r.robust_covariance = spread_matrix(r.robust_covariance);
    std::vector<unsigned char> boundary(p, 0);
    for (std::size_t k = 0; k < q && k < r.boundary_coefficients.size(); ++k)
        boundary[columns[k]] = r.boundary_coefficients[k];
    r.boundary_coefficients = std::move(boundary);
    r.aliased = aliased;
    r.parameters = p;
    r.rank = q;
    r.rank_deficient = true;
    return r;
}
} // namespace detail

inline FitResult fit(const Matrix &x, const Vector &y, const FitOptions &o = {});
// For FitOptions::design: the design-only part of fits of many responses to x.
// The design is validated here once; it must stay unchanged, and alive, while
// the summary is in use.
inline std::shared_ptr<const detail::DesignSummary> prepare_design(const Matrix &x,
                                                                   double rank_tolerance = 1e-12) {
    x.validate();
    return std::make_shared<const detail::DesignSummary>(x, rank_tolerance);
}
namespace detail {
// The fit itself. For NB ML from a moment start, `start_at_bound` starts theta at
// the upper bound instead of o.theta when the moment estimate shows no
// overdispersion; `moment_at_bound` (may be null) reports whether it did.
inline FitResult fit_from(const Matrix &x, const Vector &y, const FitOptions &o,
                          bool start_at_bound, bool *moment_at_bound) {
    const detail::DesignSummary *const shared =
        o.design && o.weights.empty() && o.design->describes(x, o.rank_tolerance) ? o.design.get()
                                                                                  : nullptr;
    if (!shared)
        x.validate();
    require(x.rows == y.size() && x.rows > 0 && x.cols > 0, "empty or mismatched design/response");
    require(o.max_iterations > 0 && o.max_dispersion_iterations > 0,
            "iteration limits must be positive");
    require(!o.nb_warm_start || (o.family == Family::negative_binomial &&
                                 o.dispersion == Dispersion::ml && !o.start.empty()),
            "NB warm start requires ML dispersion and starting coefficients");
    require(std::isfinite(o.tolerance) && o.tolerance > 0 && std::isfinite(o.rank_tolerance) &&
                o.rank_tolerance > 0,
            "invalid tolerance");
    require(std::isfinite(o.ridge) && o.ridge >= 0, "invalid ridge");
    require(o.theta_min > 0 && o.theta_max > o.theta_min && std::isfinite(o.theta_max),
            "invalid theta bounds");
    require(o.theta > 0 && std::isfinite(o.theta), "invalid theta");
    require(!o.cox_reid ||
                (o.family == Family::negative_binomial && o.dispersion == Dispersion::ml),
            "the Cox-Reid adjustment applies to NB fits with ML dispersion");
    Vector offset = o.offset.empty() ? Vector(x.rows, 0) : o.offset;
    Vector a = o.weights.empty() ? Vector(x.rows, 1) : o.weights;
    require(offset.size() == x.rows && a.size() == x.rows, "offset/weights size mismatch");
    FitResult r;
    r.family = o.family;
    r.parameters = x.cols;
    r.ridge = o.ridge;
    double sumy = 0, suma = 0;
    bool all_zero = true, all_one = true, zero_count = false;
    for (std::size_t i = 0; i < x.rows; ++i) {
        require(std::isfinite(y[i]) && std::isfinite(offset[i]), "nonfinite response or offset");
        require(std::isfinite(a[i]) && a[i] >= 0, "invalid weight");
        if (o.family == Family::binomial)
            require(y[i] >= 0 && y[i] <= 1, "binomial response outside [0,1]");
        else if (o.family != Family::gaussian)
            require(y[i] >= 0 && std::floor(y[i]) == y[i],
                    "count response must be nonnegative integer");
        if (a[i] > 0) {
            ++r.observations;
            sumy += a[i] * y[i];
            suma += a[i];
            all_zero &= y[i] == 0;
            all_one &= y[i] == 1;
            zero_count |= y[i] == 0;
        }
    }
    require(suma > 0, "all weights zero");
    r.response_signature = detail::signature(y, a, offset);
    r.coefficients = Vector(x.cols, nan);
    r.standard_errors = Vector(x.cols, nan);
    r.boundary_coefficients.assign(x.cols, 0);
    r.aliased.assign(x.cols, 0);
    r.rank = x.cols;
    // X'AX and its factor: from the shared summary, or computed here (always so
    // with prior weights).
    Matrix own_information;
    std::optional<detail::Cholesky> own_check;
    const detail::SparseRows own_rows = shared ? detail::SparseRows() : detail::SparseRows(x);
    const detail::SparseRows *const rows = shared ? &shared->nonzero : &own_rows;
    if (!shared) {
        own_information = detail::crossprod(x, a, rows);
        own_check.emplace(own_information, o.rank_tolerance);
    }
    const Matrix &prior_information = shared ? shared->information : own_information;
    const detail::Cholesky &design_check = shared ? shared->check : *own_check;
    r.rank_deficient = !design_check.ok;
    // Poorly conditioned or singular designs are checked for aliased columns by QR.
    if (o.ridge == 0 && (o.solver == Solver::qr || !design_check.ok || design_check.rcond < 1e-8)) {
        detail::WeightedQR qr(x, a, o.alias_tolerance);
        r.rank_deficient = qr.rank() < x.cols;
        if (r.rank_deficient) {
            r.aliased = qr.aliased;
            r.rank = qr.rank();
            if (!o.singular_ok || qr.rank() == 0) {
                r.status = FitStatus::rank_deficient;
                return r;
            }
            auto reduced = o;
            if (!reduced.start.empty())
                reduced.start = [&] {
                    Vector v;
                    for (auto j : qr.columns)
                        v.push_back(o.start[j]);
                    return v;
                }();
            return detail::expand_aliased(fit(detail::select_columns(x, qr.columns), y, reduced),
                                          qr.columns, qr.aliased);
        }
    }
    if (o.ridge == 0 &&
        ((o.family != Family::gaussian && all_zero) || (o.family == Family::binomial && all_one))) {
        r.status = FitStatus::boundary_response;
        return r;
    }
    if (r.observations <= x.cols && o.family == Family::gaussian) {
        r.status = FitStatus::rank_deficient;
        return r;
    }
    Vector beta = o.start.empty() ? Vector(x.cols) : o.start;
    require(beta.size() == x.cols, "warm start size mismatch");
    for (double b : beta)
        require(std::isfinite(b), "nonfinite warm start");
    const bool count_family = o.family == Family::poisson || o.family == Family::negative_binomial;
    // Grouped designs take per-group Fisher steps; see RowGroups.
    const bool grouped = count_family && o.ridge == 0 && o.solver == Solver::automatic;
    const detail::RowGroups own_groups =
        grouped && !shared ? detail::RowGroups(x) : detail::RowGroups();
    const detail::RowGroups &groups = grouped && shared ? shared->groups : own_groups;
    bool closed_form = false;
    if (groups.ok && o.start.empty()) {
        // The Poisson MLE of a grouped design, exp(b_g) sum_g a e^offset = sum_g a y;
        // for NB it is the MLE at any theta when offsets are constant within groups.
        const std::size_t p = x.cols;
        Vector total(p), shift(p, -std::numeric_limits<double>::infinity()), exposure(p);
        for (std::size_t i = 0; i < x.rows; ++i)
            if (a[i] > 0) {
                total[groups.group[i]] += a[i] * y[i];
                shift[groups.group[i]] = std::max(shift[groups.group[i]], offset[i]);
            }
        for (std::size_t i = 0; i < x.rows; ++i)
            if (a[i] > 0)
                exposure[groups.group[i]] += a[i] * std::exp(offset[i] - shift[groups.group[i]]);
        Vector b(p);
        closed_form = true;
        for (std::size_t g = 0; g < p; ++g) {
            b[g] = std::log(total[g]) - shift[g] - std::log(exposure[g]);
            closed_form &= total[g] > 0 && std::isfinite(b[g]);
        }
        if (closed_form)
            beta = detail::matvec(groups.inverse, b);
    }
    if (o.start.empty() && !closed_form) {
        Vector z(x.rows);
        for (std::size_t i = 0; i < x.rows; ++i) {
            double m = (y[i] + 0.1 * sumy / suma + 0.1) / 1.1;
            double link = o.family == Family::gaussian   ? y[i]
                          : o.family == Family::binomial ? std::log((y[i] + 0.5) / (1.5 - y[i]))
                                                         : std::log(m);
            z[i] = a[i] * (link - offset[i]);
        }
        // Without a ridge the factor of the rank check is the one needed here.
        std::optional<detail::Cholesky> ridged;
        if (o.ridge != 0) {
            auto h = prior_information;
            for (std::size_t j = 0; j < x.cols; ++j)
                h(j, j) += o.ridge;
            ridged.emplace(h, o.rank_tolerance);
        }
        const detail::Cholesky &ch = ridged ? *ridged : design_check;
        if (ch.ok && o.solver != Solver::qr)
            beta = ch.solve(detail::xtv(x, z, rows));
        else {
            detail::WeightedQR qr(x, a, o.alias_tolerance, o.ridge);
            if (qr.ok && qr.rank() == x.cols) {
                Vector u(qr.rows);
                for (std::size_t i = 0; i < x.rows; ++i)
                    u[i] = a[i] > 0 ? z[i] / std::sqrt(a[i]) : 0;
                beta = qr.solve(std::move(u));
            }
        }
    }
    double theta = std::clamp(o.theta, o.theta_min, o.theta_max);
    if (o.family == Family::negative_binomial && o.dispersion != Dispersion::fixed &&
        !o.nb_warm_start) {
        auto pilot = o;
        pilot.family = Family::poisson;
        pilot.retain_observations = true;
        pilot.robust = false;
        pilot.cox_reid = false;
        auto pr = fit(x, y, pilot);
        if (!pr.ok())
            return pr.family = o.family, pr;
        beta = pr.coefficients;
        r.boundary_coefficients = pr.boundary_coefficients;
        // A moment estimate at the upper bound (no overdispersion beyond the Poisson
        // fit, common in nearly saturated designs) is a point where the likelihood is
        // flat in theta and Newton would stop at once; start from o.theta instead, and
        // fit() also runs from the bound and keeps the better result.
        const double moment = detail::moment_theta(y, pr.fitted, a, o);
        if (moment_at_bound)
            *moment_at_bound = moment >= o.theta_max;
        theta = moment < o.theta_max || start_at_bound
                    ? moment
                    : std::clamp(o.theta, o.theta_min, o.theta_max);
    }
    const auto mean_values = [&](const Vector &b) {
        return groups.ok ? detail::means(groups, b, offset, o.family)
                         : detail::means(x, b, offset, o.family, rows);
    };
    Vector mu = mean_values(beta), w, s;
    const detail::CountTerms terms =
        count_family ? detail::CountTerms(y, a) : detail::CountTerms(Vector(), Vector());
    bool complete = false;
    int positive_full_rank = -1; // unknown until a boundary check is needed
    // Without zero counts and prior weights, the design restricted to positive
    // counts is the design itself, whose cross-product was factored above.
    const auto positive_rank = [&] {
        return !zero_count && o.weights.empty()
                   ? design_check.ok
                   : detail::positive_counts_full_rank(x, y, a, o.rank_tolerance, rows);
    };
    // Unpenalized NB ML without the Cox-Reid adjustment maximizes one objective,
    // so (beta, u = log theta) take joint Newton steps with the observed
    // information. For the log link -d2l/deta2 = a theta mu (theta + y)/(theta + mu)^2
    // > 0, so A = X'W X is positive definite. With the scores s and g_u, the mixed
    // derivatives c = X' d2l/(deta du) and the curvature h_u, the Schur complement
    // h_u + c'A^-1 c < 0 gives the Newton step du = -(g_u + c'A^-1 s)/(h_u + c'A^-1 c);
    // otherwise du is a unit step up that slope. Then dbeta = A^-1 (s + c du).
    // Both directions ascend, and step halving on the joint likelihood keeps every
    // iterate an improvement. Where the Schur complement is not negative, theta is
    // held until the coefficients settle (see below). Other NB ML fits alternate coefficient fits
    // with theta updates below.
    const bool joint = o.family == Family::negative_binomial && o.dispersion == Dispersion::ml &&
                       !o.cox_reid && o.ridge == 0;
    if (joint) {
        const std::size_t n = x.rows, p = x.cols;
        const double lower = std::log(o.theta_min), upper = std::log(o.theta_max);
        double ll = detail::likelihood(y, mu, a, o.family, theta, terms);
        // Scale of the relative objective change, as for coefficient-only fits.
        const double scale = 0.1 + std::abs(detail::count_deviance(y, mu, a, o.family, theta));
        Vector mixed(n);
        bool converged = false, previous_stable = false;
        bool reduced = false; // diverging coefficients held (see below)
        Vector last_direction;
        std::size_t iteration = 0;
        for (; iteration < o.max_iterations; ++iteration) {
            ++r.iterations;
            ++r.dispersion_iterations;
            const double t = theta, u = std::log(theta);
            w.resize(n);
            s.resize(n);
            // Response-only polygamma sums come from the count histogram.
            auto [g, h] = terms.polygamma(t);
            h += terms.weight / t;
            for (std::size_t i = 0; i < n; ++i) {
                const double m = mu[i], v = y[i], d = t + m;
                w[i] = a[i] * t * m * (t + v) / (d * d);
                s[i] = a[i] * t * (v - m) / d;
                mixed[i] = a[i] * t * m * (v - m) / (d * d);
                if (a[i] > 0) {
                    g += a[i] * ((m - v) / d - log1p_by_log(m / t));
                    h += a[i] * ((v - m) / (d * d) - 1 / d);
                }
            }
            const double gu = static_cast<double>(g * t),
                         hu = static_cast<double>(h * t * t + g * t);
            // delta = A^-1 X's and coupling = A^-1 c, by the same solvers as below.
            Vector delta, coupling, c;
            double resolution = 0;
            if (groups.ok) {
                Vector weight(p), score(p), cross(p);
                for (std::size_t i = 0; i < n; ++i) {
                    weight[groups.group[i]] += w[i];
                    score[groups.group[i]] += s[i];
                    cross[groups.group[i]] += mixed[i];
                }
                bool positive = true;
                c.assign(p, 0);
                for (std::size_t k = 0; k < p; ++k) {
                    positive &= weight[k] > 0 && std::isfinite(weight[k]);
                    for (std::size_t j = 0; j < p; ++j)
                        c[j] += groups.rows(k, j) * cross[k];
                    score[k] /= weight[k];
                    cross[k] /= weight[k];
                }
                if (positive) {
                    delta = detail::matvec(groups.inverse, score);
                    coupling = detail::matvec(groups.inverse, cross);
                    resolution = groups.resolution;
                }
            }
            if (c.empty())
                c = detail::xtv(x, mixed, rows);
            if (delta.empty() && o.solver != Solver::qr) {
                detail::Cholesky ch(detail::crossprod(x, w, rows), o.rank_tolerance);
                if (ch.ok && ch.rcond >= 1e-8) {
                    delta = ch.solve(detail::xtv(x, s, rows));
                    coupling = ch.solve(c);
                    resolution = 100 * std::numeric_limits<double>::epsilon() / std::sqrt(ch.rcond);
                }
            }
            // Diverging coefficients (e.g. an all-zero group) drive their weights, and
            // the information, to zero. Once the full system is singular they are
            // identified from the last direction and held, and Newton continues on the
            // other coefficients and theta (their fitted means are already negligible,
            // the maximum-likelihood limit). The final covariance excludes them.
            const auto solve_qr = [&](const std::vector<std::size_t> &columns) {
                const bool all = columns.size() == p;
                const Matrix xk = all ? Matrix() : detail::select_columns(x, columns);
                detail::WeightedQR qr(all ? x : xk, w, o.alias_tolerance);
                if (!qr.ok || qr.rank() < columns.size())
                    return false;
                r.used_qr = true;
                resolution = 100 * std::numeric_limits<double>::epsilon() / qr.rcond();
                Vector z(qr.rows), ck(columns.size());
                for (std::size_t i = 0; i < n; ++i)
                    z[i] = w[i] > 0 ? s[i] / std::sqrt(w[i]) : 0;
                for (std::size_t j = 0; j < columns.size(); ++j)
                    ck[j] = c[columns[j]];
                const auto dk = qr.solve(std::move(z));
                const auto uk = detail::matvec(qr.inverse(), ck);
                delta.assign(p, 0);
                coupling.assign(p, 0);
                for (std::size_t j = 0; j < columns.size(); ++j) {
                    delta[columns[j]] = dk[j];
                    coupling[columns[j]] = uk[j];
                }
                return true;
            };
            std::vector<std::size_t> all_columns(p);
            std::iota(all_columns.begin(), all_columns.end(), 0);
            if (!reduced && delta.empty() && !solve_qr(all_columns) && !last_direction.empty()) {
                if (positive_full_rank < 0)
                    positive_full_rank = positive_rank() ? 1 : 0;
                auto flags = detail::count_boundary(x, y, a, last_direction, o.rank_tolerance,
                                                    positive_full_rank == 1);
                for (std::size_t j = 0; j < p; ++j)
                    r.boundary_coefficients[j] |= flags[j];
                reduced = r.has_boundary();
            }
            if (reduced) {
                std::vector<std::size_t> kept;
                for (std::size_t j = 0; j < p; ++j)
                    if (!r.boundary_coefficients[j])
                        kept.push_back(j);
                delta.clear();
                if (!kept.empty())
                    solve_qr(kept);
            }
            if (delta.empty()) {
                if (previous_stable)
                    converged = true;
                else
                    r.status = FitStatus::numerical_failure;
                break;
            }
            const double slope = gu + detail::dot(c, delta), schur = hu + detail::dot(c, coupling);
            // Where the joint likelihood is not concave in log theta (schur >= 0, often
            // while the coefficients are still far from their conditional optimum), a
            // unit step in theta can carry it past a maximum of the profile likelihood
            // onto the slope towards the Poisson limit. Hold theta and take the
            // coefficient Newton step (an ascent direction) until the coefficients
            // settle; only then move theta up its slope by a unit step.
            double coefficient_change = 0;
            for (std::size_t j = 0; j < p; ++j)
                if (!r.boundary_coefficients[j]) // diverging coefficients never settle
                    coefficient_change =
                        std::max(coefficient_change, std::abs(delta[j]) / (1 + std::abs(beta[j])));
            double du = std::abs(slope) < 1e-10                       ? 0
                        : schur < 0                                   ? -slope / schur
                        : coefficient_change > std::sqrt(o.tolerance) ? 0
                                                                      : (slope > 0 ? 1 : -1);
            // Held: theta could move but waits for the coefficients (not at a bound the
            // slope pushes against). A held iteration never counts as converged.
            const bool pinned = (slope > 0 && u >= upper) || (slope < 0 && u <= lower);
            const bool held = schur >= 0 && du == 0 && std::abs(slope) >= 1e-10 && !pinned;
            du = std::clamp(u + std::clamp(du, -2.0, 2.0), lower, upper) - u;
            Vector direction(p), next(p), next_mu;
            for (std::size_t j = 0; j < p; ++j)
                direction[j] = delta[j] + coupling[j] * du;
            last_direction = direction;
            // Theta keeps its exact value unless the step moves it: near the upper bound
            // the likelihood's lgamma differences carry rounding noise that a
            // one-ulp change of theta would expose to the acceptance test.
            double step = 1, next_theta = theta, ln = -std::numeric_limits<double>::infinity();
            bool accepted = false;
            for (unsigned halve = 0; halve < 40; ++halve) {
                for (std::size_t j = 0; j < p; ++j)
                    next[j] = beta[j] + step * direction[j];
                next_theta =
                    du == 0 ? theta : std::clamp(std::exp(u + step * du), o.theta_min, o.theta_max);
                next_mu = mean_values(next);
                ln = detail::likelihood(y, next_mu, a, o.family, next_theta, terms);
                if (std::isfinite(ln) && ln >= ll - 1e-12 * (1 + std::abs(ll))) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                // No step improves on a point whose objective had already settled, or
                // whose Newton step promises less than the objective tolerance: the fit
                // is at the maximum to working precision (seen near the theta bound).
                const double predicted =
                    std::abs(detail::dot(detail::xtv(x, s, rows), direction) + gu * du);
                if (previous_stable || predicted < o.tolerance * scale)
                    converged = true;
                else
                    r.status = FitStatus::numerical_failure;
                break;
            }
            const bool objective_stable = 2 * std::abs(ln - ll) < o.tolerance * scale;
            previous_stable = objective_stable;
            if (objective_stable) {
                if (positive_full_rank < 0)
                    positive_full_rank = positive_rank() ? 1 : 0;
                auto flags = detail::count_boundary(x, y, a, direction, o.rank_tolerance,
                                                    positive_full_rank == 1);
                for (std::size_t j = 0; j < p; ++j)
                    r.boundary_coefficients[j] |= flags[j];
            }
            double change = 0;
            for (std::size_t j = 0; j < p; ++j)
                if (!r.boundary_coefficients[j])
                    change =
                        std::max(change, std::abs(next[j] - beta[j]) / (1 + std::abs(beta[j])));
            const double u_change = std::abs(std::log(next_theta / theta));
            beta = std::move(next);
            mu = std::move(next_mu);
            theta = next_theta;
            ll = ln;
            if (objective_stable && !held && change < std::max(o.tolerance, resolution) &&
                u_change < o.tolerance) {
                converged = true;
                break;
            }
        }
        complete = converged;
        if (!converged && iteration == o.max_iterations)
            r.status = FitStatus::iteration_limit;
    }
    auto outer_max = joint ? 0
                     : o.family == Family::negative_binomial && o.dispersion == Dispersion::ml
                         ? o.max_dispersion_iterations
                         : 1;
    std::vector<double> chain; // log theta iterates since the last extrapolation (Cox-Reid)
    // Working weights and scores of a coefficient step. NB steps use the observed
    // information, a theta mu (theta + y) / (theta + mu)^2 > 0 for the log link:
    // Newton's method converges quadratically where Fisher scoring (the expected
    // information) converges only linearly. The reported covariance still uses the
    // expected information.
    const auto step_weights = [&](const Vector &at, Vector &weights, Vector &scores) {
        detail::working(y, at, a, o.family, theta, weights, scores);
        if (o.family == Family::negative_binomial)
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const double d = theta + at[i];
                weights[i] = a[i] * theta * at[i] * (theta + y[i]) / (d * d);
            }
    };
    double final_likelihood = nan; // the log likelihood at the final estimate, when known
    for (std::size_t outer = 0; outer < outer_max; ++outer) {
        bool converged = false;
        double ll = detail::likelihood(y, mu, a, o.family, theta, terms) -
                    0.5 * o.ridge * detail::dot(beta, beta);
        double deviance = count_family ? detail::count_deviance(y, mu, a, o.family, theta) +
                                             o.ridge * detail::dot(beta, beta)
                                       : nan;
        // Counted per coefficient loop; r.iterations accumulates across dispersion rounds.
        std::size_t iteration = 0;
        bool have_weights = false; // w and s already hold the step weights at mu
        for (; iteration < o.max_iterations; ++iteration) {
            ++r.iterations;
            if (!have_weights)
                step_weights(mu, w, s);
            have_weights = false;
            std::optional<detail::Cholesky>
                factor; // of this step's information, if it gave the step
            Vector delta;
            // Relative rounding level of the step, about eps * cond(sqrt(W)X); coefficient
            // changes below it cannot be resolved, so it floors the change criterion.
            double resolution = 0;
            if (groups.ok) {
                const std::size_t p = x.cols;
                Vector weight(p), score(p);
                for (std::size_t i = 0; i < x.rows; ++i) {
                    weight[groups.group[i]] += w[i];
                    score[groups.group[i]] += s[i];
                }
                bool positive = true;
                for (std::size_t g = 0; g < p; ++g) {
                    positive &= weight[g] > 0 && std::isfinite(weight[g]);
                    score[g] /= weight[g];
                }
                if (positive) {
                    delta = detail::matvec(groups.inverse, score);
                    resolution = groups.resolution;
                }
            }
            if (delta.empty() && o.solver != Solver::qr) {
                Matrix h = detail::crossprod(x, w, rows);
                auto u = detail::xtv(x, s, rows);
                for (std::size_t j = 0; j < x.cols; ++j) {
                    h(j, j) += o.ridge;
                    u[j] -= o.ridge * beta[j];
                }
                factor.emplace(h, o.rank_tolerance);
                // The fixed point is exact either way, but steps from an ill-conditioned
                // X'WX converge only linearly; QR keeps Newton's quadratic convergence.
                if (factor->ok && factor->rcond >= 1e-8) {
                    delta = factor->solve(u);
                    resolution =
                        100 * std::numeric_limits<double>::epsilon() / std::sqrt(factor->rcond);
                } else
                    factor.reset();
            }
            if (delta.empty()) {
                detail::WeightedQR qr(x, w, o.alias_tolerance, o.ridge);
                if (!qr.ok || qr.rank() < x.cols) {
                    r.status = o.family == Family::binomial ? FitStatus::separation
                                                            : FitStatus::numerical_failure;
                    break;
                }
                r.used_qr = true;
                resolution = 100 * std::numeric_limits<double>::epsilon() / qr.rcond();
                Vector u(qr.rows);
                for (std::size_t i = 0; i < x.rows; ++i)
                    u[i] = w[i] > 0 ? s[i] / std::sqrt(w[i]) : 0;
                for (std::size_t j = 0; j + x.rows < qr.rows; ++j)
                    u[x.rows + j] = -std::sqrt(o.ridge) * beta[j];
                delta = qr.solve(std::move(u));
            }
            Vector next(beta.size()), next_mu;
            double step = 1, ln = -std::numeric_limits<double>::infinity();
            bool accepted = false;
            for (unsigned halve = 0; halve < 40; ++halve) {
                for (std::size_t j = 0; j < beta.size(); ++j)
                    next[j] = beta[j] + step * delta[j];
                next_mu = mean_values(next);
                ln = detail::likelihood(y, next_mu, a, o.family, theta, terms) -
                     0.5 * o.ridge * detail::dot(next, next);
                if (std::isfinite(ln) && ln >= ll - 1e-12 * (1 + std::abs(ll))) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                r.status = FitStatus::numerical_failure;
                break;
            }
            double change = 0;
            // At fixed theta the (penalized) deviance changes by -2 times the log
            // likelihood change, so it is updated rather than recomputed.
            double next_deviance = count_family ? deviance - 2 * (ln - ll) : nan;
            bool objective_stable = !count_family || (std::isfinite(next_deviance) &&
                                                      std::abs(next_deviance - deviance) /
                                                              (0.1 + std::abs(next_deviance)) <
                                                          o.tolerance);
            if (count_family && o.ridge == 0 && objective_stable) {
                if (positive_full_rank < 0)
                    positive_full_rank = positive_rank() ? 1 : 0;
                auto flags = detail::count_boundary(x, y, a, delta, o.rank_tolerance,
                                                    positive_full_rank == 1);
                for (std::size_t j = 0; j < x.cols; ++j)
                    r.boundary_coefficients[j] |= flags[j];
            }
            for (std::size_t j = 0; j < beta.size(); ++j)
                if (!r.boundary_coefficients[j])
                    change =
                        std::max(change, std::abs(next[j] - beta[j]) / (1 + std::abs(beta[j])));
            beta = std::move(next);
            mu = std::move(next_mu);
            ll = ln;
            deviance = next_deviance;
            if (o.family == Family::binomial && o.ridge == 0) {
                double largest = 0;
                for (double b : beta)
                    largest = std::max(largest, std::abs(b));
                bool boundary = std::any_of(mu.begin(), mu.end(),
                                            [](double m) { return m < 1e-12 || m > 1 - 1e-12; });
                if (boundary && largest > 30) {
                    r.status = FitStatus::separation;
                    break;
                }
            }
            if (objective_stable && change < std::max(o.tolerance, resolution)) {
                converged = true;
                break;
            }
            if (!factor)
                continue;
            // A further Newton step from an estimate already within the tolerance would
            // only confirm it, at the cost of a cross-product, a factorization and a
            // likelihood. This step's factor predicts that step instead: with the score
            // at the new estimate, the chord step d = H^-1 u and the deviance change
            // u'd it promises. The factor is trusted only if no working weight has moved
            // by more than a quarter. If both are within the tolerance the fit has
            // converged, and the chord step is taken as a last refinement unless it
            // lowers the likelihood or a coefficient is at a boundary.
            Vector next_w, next_s;
            step_weights(mu, next_w, next_s);
            bool close = true;
            for (std::size_t i = 0; i < w.size() && close; ++i)
                close = std::abs(next_w[i] - w[i]) <= 0.25 * w[i];
            if (close) {
                auto u = detail::xtv(x, next_s, rows);
                for (std::size_t j = 0; j < x.cols; ++j)
                    u[j] -= o.ridge * beta[j];
                const Vector chord = factor->solve(u);
                const double gain = detail::dot(u, chord);
                const bool stable =
                    !count_family ||
                    (std::isfinite(gain) && gain / (0.1 + std::abs(deviance)) < o.tolerance);
                if (count_family && o.ridge == 0 && stable) {
                    if (positive_full_rank < 0)
                        positive_full_rank = positive_rank() ? 1 : 0;
                    auto flags = detail::count_boundary(x, y, a, chord, o.rank_tolerance,
                                                        positive_full_rank == 1);
                    for (std::size_t j = 0; j < x.cols; ++j)
                        r.boundary_coefficients[j] |= flags[j];
                }
                double chord_change = 0;
                for (std::size_t j = 0; j < beta.size(); ++j)
                    if (!r.boundary_coefficients[j])
                        chord_change =
                            std::max(chord_change, std::abs(chord[j]) / (1 + std::abs(beta[j])));
                if (stable && std::isfinite(chord_change) &&
                    chord_change < std::max(o.tolerance, resolution)) {
                    if (!r.has_boundary()) {
                        Vector refined(beta.size());
                        for (std::size_t j = 0; j < beta.size(); ++j)
                            refined[j] = beta[j] + chord[j];
                        Vector refined_mu = mean_values(refined);
                        // The chord step raises the objective by about u'd/2 (it ascends
                        // while the weights are within a quarter of the factor's). When
                        // that is below the rounding of the objective, its value is known
                        // without another evaluation.
                        const double rise = 0.5 * gain;
                        const bool known = count_family && rise <= 1e-13 * (1 + std::abs(ll)) &&
                                           std::all_of(refined_mu.begin(), refined_mu.end(),
                                                       [](double m) { return std::isfinite(m); });
                        const double lr =
                            known ? ll + rise
                                  : detail::likelihood(y, refined_mu, a, o.family, theta, terms) -
                                        0.5 * o.ridge * detail::dot(refined, refined);
                        if (std::isfinite(lr) && lr >= ll - 1e-12 * (1 + std::abs(ll))) {
                            beta = std::move(refined);
                            mu = std::move(refined_mu);
                            ll = lr;
                        }
                    }
                    converged = true;
                    break;
                }
            }
            w = std::move(next_w);
            s = std::move(next_s);
            have_weights = true;
        }
        if (converged && outer_max == 1 && o.ridge == 0)
            final_likelihood = ll;
        if (!converged) {
            // Only an exhausted loop is an iteration limit; explicit failures keep their status.
            if (iteration == o.max_iterations)
                r.status = FitStatus::iteration_limit;
            break;
        }
        if (outer_max == 1) {
            complete = true;
            break;
        }
        ++r.dispersion_iterations;
        double new_theta;
        r.cox_reid = false;
        if (o.cox_reid) {
            // Coefficients at a boundary carry no information about theta.
            std::vector<std::size_t> kept;
            for (std::size_t j = 0; j < x.cols; ++j)
                if (!r.boundary_coefficients[j])
                    kept.push_back(j);
            const auto adjusted = detail::select_columns(x, kept);
            r.cox_reid = detail::cox_reid(adjusted, mu, a, theta, false).ok;
            new_theta =
                detail::estimate_theta(y, mu, a, theta, o, terms, r.cox_reid ? &adjusted : nullptr);
        } else
            new_theta = detail::estimate_theta(y, mu, a, theta, o, terms);
        double change = std::abs(std::log(new_theta / theta));
        if (o.cox_reid) {
            if (chain.empty())
                chain.push_back(std::log(theta));
            chain.push_back(std::log(new_theta));
        }
        theta = new_theta;
        if (change < o.tolerance) {
            complete = true;
            break;
        }
        // Alternating coefficient fits with adjusted-likelihood theta steps can
        // converge slowly and linearly where the adjusted likelihood is flat; Aitken
        // extrapolation of three successive log theta values accelerates them.
        if (chain.size() == 3) {
            const double d1 = chain[1] - chain[0], d2 = chain[2] - chain[1], ratio = d2 / d1;
            if (std::isfinite(ratio) && ratio != 0 && std::abs(ratio) < 1) {
                theta =
                    std::clamp(std::exp(chain[2] - d2 * d2 / (d2 - d1)), o.theta_min, o.theta_max);
                chain.clear();
            } else
                chain.erase(chain.begin());
        }
        r.status = FitStatus::iteration_limit;
    }
    r.coefficients = beta;
    r.theta = o.family == Family::negative_binomial ? theta : nan;
    r.theta_boundary = o.family == Family::negative_binomial &&
                       (theta <= o.theta_min * 1.00001 || theta >= o.theta_max / 1.00001);
    r.fitted = mu;
    detail::working(y, mu, a, o.family, theta, r.working_weights, r.score_residuals);
    r.log_likelihood = std::isnan(final_likelihood)
                           ? detail::likelihood(y, mu, a, o.family, theta, terms)
                           : final_likelihood;
    if (o.family == Family::gaussian) {
        double rss = -2 * r.log_likelihood;
        r.scale = rss / static_cast<double>(r.observations - x.cols);
        double ml = rss / r.observations;
        r.log_likelihood =
            ml > 0 ? -0.5 * (r.observations * (std::log(2 * 3.14159265358979323846 * ml) + 1))
                   : std::numeric_limits<double>::infinity();
        // Analytic Gaussian prior precision contributes a parameter-independent constant.
        for (double ai : a)
            if (ai > 0)
                r.log_likelihood += 0.5 * std::log(ai);
    }
    auto h = detail::crossprod(x, r.working_weights, rows);
    for (std::size_t j = 0; j < x.cols; ++j)
        h(j, j) += o.ridge;
    detail::Cholesky ch(h, o.rank_tolerance);
    // Covariance from X'WX only while it is well conditioned; otherwise from QR,
    // whose error grows with cond(sqrt(W)X) rather than its square.
    Matrix inverse;
    if (complete) {
        if (o.solver != Solver::qr && ch.ok && ch.rcond >= 1e-8)
            inverse = ch.inverse();
        else {
            detail::WeightedQR qr(x, r.working_weights, o.alias_tolerance, o.ridge);
            if (qr.ok && qr.rank() == x.cols) {
                inverse = qr.inverse();
                r.used_qr = true;
            }
        }
        if (inverse.rows != x.cols && r.has_boundary()) {
            // Coefficients at a boundary carry vanishing weight: the covariance of the
            // others is the inverse of their own information (boundary entries are
            // reported as missing below).
            std::vector<std::size_t> kept;
            for (std::size_t j = 0; j < x.cols; ++j)
                if (!r.boundary_coefficients[j])
                    kept.push_back(j);
            auto hk = detail::crossprod(detail::select_columns(x, kept), r.working_weights);
            for (std::size_t j = 0; j < kept.size(); ++j)
                hk(j, j) += o.ridge;
            detail::Cholesky ck(hk, o.rank_tolerance);
            if (!kept.empty() && ck.ok) {
                const auto ik = ck.inverse();
                inverse = Matrix(x.cols, x.cols, 0.0);
                for (std::size_t j = 0; j < kept.size(); ++j)
                    for (std::size_t k = 0; k < kept.size(); ++k)
                        inverse(kept[j], kept[k]) = ik(j, k);
            }
        }
        if (inverse.rows != x.cols)
            r.status = FitStatus::numerical_failure; // singular final information
    }
    if (complete && inverse.rows == x.cols) {
        r.status = r.has_boundary() ? FitStatus::converged_boundary : FitStatus::converged;
        r.covariance = inverse;
        for (double &v : r.covariance.values)
            v *= r.scale;
        for (std::size_t j = 0; j < x.cols; ++j)
            r.standard_errors[j] = std::sqrt(r.covariance(j, j));
        if (o.robust) {
            Vector s2 = r.score_residuals;
            for (double &v : s2)
                v *= v;
            auto meat = detail::crossprod(x, s2, rows);
            r.robust_covariance = Matrix(x.cols, x.cols);
            for (std::size_t j = 0; j < x.cols; ++j)
                for (std::size_t k = 0; k < x.cols; ++k)
                    for (std::size_t b = 0; b < x.cols; ++b)
                        for (std::size_t c = 0; c < x.cols; ++c)
                            r.robust_covariance(j, k) += inverse(j, b) * meat(b, c) * inverse(c, k);
        }
        for (std::size_t j = 0; j < x.cols; ++j)
            if (r.boundary_coefficients[j]) {
                r.standard_errors[j] = nan;
                for (std::size_t k = 0; k < x.cols; ++k) {
                    r.covariance(j, k) = r.covariance(k, j) = nan;
                    if (o.robust)
                        r.robust_covariance(j, k) = r.robust_covariance(k, j) = nan;
                }
            }
    }
    if (!o.retain_observations)
        r.compact();
    return r;
}
} // namespace detail
// When the moment start shows no overdispersion, the profile likelihood in theta
// can have a local maximum below the Poisson limit, or the limit can lie beyond a
// dip that hides an interior maximum. Such fits run from both o.theta and the upper
// bound and keep the higher likelihood; reported iterations count both.
inline FitResult fit(const Matrix &x, const Vector &y, const FitOptions &o) {
    bool at_bound = false;
    auto r = detail::fit_from(x, y, o, false, &at_bound);
    if (!at_bound)
        return r;
    auto alternative = detail::fit_from(x, y, o, true, nullptr);
    const bool better =
        alternative.ok() && (!r.ok() || alternative.log_likelihood > r.log_likelihood);
    auto &chosen = better ? alternative : r;
    chosen.iterations = r.iterations + alternative.iterations;
    chosen.dispersion_iterations = r.dispersion_iterations + alternative.dispersion_iterations;
    return std::move(chosen);
}
// SCEPTRE's documented two-stage response-null recipe. The coefficients are
// Poisson estimates; theta maximizes the NB likelihood with those means fixed.
// This is not a joint NB fit and supplies no coefficient Wald/LR inference.
inline FitResult fit_sceptre_null(const Matrix &x, const Vector &y, const FitOptions &o = {}) {
    require(o.family == Family::negative_binomial && o.dispersion == Dispersion::ml &&
                o.ridge == 0 && !o.nb_warm_start && !o.cox_reid,
            "SCEPTRE null requires unpenalized NB with ML dispersion, no NB warm start and no "
            "Cox-Reid adjustment");
    auto pilot = o;
    pilot.family = Family::poisson;
    pilot.robust = false;
    pilot.retain_observations = true;
    auto r = fit(x, y, pilot);
    r.method = FitMethod::sceptre_null;
    r.family = Family::negative_binomial;
    r.standard_errors.assign(x.cols, nan);
    r.covariance = Matrix();
    r.robust_covariance = Matrix();
    if (r.ok() && !r.has_boundary()) {
        const Vector a = o.weights.empty() ? Vector(y.size(), 1) : o.weights;
        double theta = detail::moment_theta(y, r.fitted, a, o);
        const detail::CountTerms terms(y, a);
        bool converged = false;
        for (std::size_t iteration = 0; iteration < o.max_dispersion_iterations; ++iteration) {
            ++r.dispersion_iterations;
            const double next = detail::estimate_theta(y, r.fitted, a, theta, o, terms);
            const double change = std::abs(std::log(next / theta));
            theta = next;
            if (change < o.tolerance) {
                converged = true;
                break;
            }
        }
        r.theta = theta;
        r.theta_boundary = theta <= o.theta_min * 1.00001 || theta >= o.theta_max / 1.00001;
        r.log_likelihood = detail::likelihood(y, r.fitted, a, r.family, theta, terms);
        detail::working(y, r.fitted, a, r.family, theta, r.working_weights, r.score_residuals);
        r.status = converged ? FitStatus::converged : FitStatus::iteration_limit;
    } else {
        // A failed/boundary Poisson pilot is not a usable NB response null.
        r.log_likelihood = nan;
        r.working_weights.clear();
        r.score_residuals.clear();
    }
    if (!o.retain_observations)
        r.compact();
    return r;
}
} // namespace rapidglm
