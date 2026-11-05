#pragma once
#include "model.hpp"
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace rapidglm {
// Observations x features, in compressed sparse column format.
struct SparseCSC {
    std::size_t rows = 0, cols = 0;
    std::vector<std::size_t> column_ptr, row_index;
    Vector values;
    void validate() const {
        require(cols < std::numeric_limits<std::size_t>::max(), "CSC column count overflow");
        require(column_ptr.size() == cols + 1 && column_ptr.front() == 0 &&
                    column_ptr.back() == values.size() && values.size() == row_index.size(),
                "invalid CSC storage");
        for (std::size_t j = 0; j < cols; ++j) {
            require(column_ptr[j] <= column_ptr[j + 1], "invalid CSC pointers");
            for (std::size_t k = column_ptr[j]; k < column_ptr[j + 1]; ++k) {
                require(row_index[k] < rows && std::isfinite(values[k]), "invalid CSC entry");
                require(k == column_ptr[j] || row_index[k - 1] < row_index[k],
                        "CSC rows must be sorted and unique");
            }
        }
    }
    Vector column(std::size_t j) const {
        require(j < cols, "CSC column out of range");
        Vector y(rows);
        for (auto k = column_ptr[j]; k < column_ptr[j + 1]; ++k)
            y[row_index[k]] = values[k];
        return y;
    }
    // function(row, value) for the stored entries of column j, rows ascending.
    template <class Function> void for_each(std::size_t j, Function function) const {
        for (auto k = column_ptr[j]; k < column_ptr[j + 1]; ++k)
            function(row_index[k], values[k]);
    }
};
namespace detail {
// Leaves trivially constructible elements uninitialized, so that a large buffer
// is first written, and its memory first touched, by the threads that fill it.
template <class T> struct DefaultInitAllocator {
    using value_type = T;
    DefaultInitAllocator() = default;
    template <class U> DefaultInitAllocator(const DefaultInitAllocator<U> &) noexcept {}
    T *allocate(std::size_t n) {
        return std::allocator<T>().allocate(n);
    }
    void deallocate(T *p, std::size_t n) noexcept {
        std::allocator<T>().deallocate(p, n);
    }
    template <class U> void construct(U *p) {
        ::new (static_cast<void *>(p)) U;
    }
    template <class U, class... Args> void construct(U *p, Args &&...args) {
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
    template <class U> bool operator==(const DefaultInitAllocator<U> &) const noexcept {
        return true;
    }
    template <class U> bool operator!=(const DefaultInitAllocator<U> &) const noexcept {
        return false;
    }
};
} // namespace detail
// Counts, observations (cells) x features (genes), for the workflows that read
// one feature at a time: the nonzero counts of each feature, ascending by cell,
// in one buffer of eight-byte entries, and where each feature's entries start.
// Half the size of the double CSC, and a feature is one contiguous run.
struct CountMatrix {
    struct Entry {
        std::uint32_t cell, count;
    };
    std::size_t rows = 0, cols = 0; // cells, features
    // Feature j's entries are entries[start[j]] ... entries[start[j + 1] - 1].
    std::vector<std::uint64_t> start;
    std::vector<Entry, detail::DefaultInitAllocator<Entry>> entries;
    void validate() const {
        require(rows <= std::numeric_limits<std::uint32_t>::max(), "too many cells for counts");
        require(start.size() == cols + 1 && start.front() == 0 && start.back() == entries.size(),
                "invalid count storage");
        for (std::size_t j = 0; j < cols; ++j) {
            require(start[j] <= start[j + 1], "invalid count offsets");
            for (auto k = start[j]; k < start[j + 1]; ++k) {
                require(entries[k].cell < rows && entries[k].count > 0, "invalid count entry");
                require(k == start[j] || entries[k - 1].cell < entries[k].cell,
                        "count cells must be sorted and unique");
            }
        }
    }
    Vector column(std::size_t j) const {
        require(j < cols, "count feature out of range");
        Vector y(rows);
        for (auto k = start[j]; k < start[j + 1]; ++k)
            y[entries[k].cell] = entries[k].count;
        return y;
    }
    // function(cell, count) for the nonzero counts of feature j, cells ascending.
    template <class Function> void for_each(std::size_t j, Function function) const {
        for (auto k = start[j]; k < start[j + 1]; ++k)
            function(static_cast<std::size_t>(entries[k].cell),
                     static_cast<double>(entries[k].count));
    }
    // From a CSC matrix of non-negative integers below 2^32 (zeros are dropped).
    static CountMatrix from(const SparseCSC &csc) {
        csc.validate();
        require(csc.rows <= std::numeric_limits<std::uint32_t>::max(), "too many cells for counts");
        CountMatrix out;
        out.rows = csc.rows;
        out.cols = csc.cols;
        out.start.assign(1, 0);
        out.entries.reserve(csc.values.size());
        for (std::size_t j = 0; j < csc.cols; ++j) {
            for (auto k = csc.column_ptr[j]; k < csc.column_ptr[j + 1]; ++k) {
                const double v = csc.values[k];
                require(v >= 0 && v <= std::numeric_limits<std::uint32_t>::max() &&
                            v == std::floor(v),
                        "counts must be non-negative integers below 2^32");
                if (v > 0)
                    out.entries.push_back({static_cast<std::uint32_t>(csc.row_index[k]),
                                           static_cast<std::uint32_t>(v)});
            }
            out.start.push_back(out.entries.size());
        }
        return out;
    }
    SparseCSC csc() const {
        SparseCSC out;
        out.rows = rows;
        out.cols = cols;
        out.column_ptr.assign(start.begin(), start.end());
        out.row_index.reserve(entries.size());
        out.values.reserve(entries.size());
        for (const auto &entry : entries) {
            out.row_index.push_back(entry.cell);
            out.values.push_back(entry.count);
        }
        return out;
    }
};
// Runs function(i) for i in [0, count) on `threads` OpenMP threads. Items are
// handed out one at a time as threads become free: per-feature costs vary widely
// (boundary fits can take hundreds of iterations), and each item is independent,
// so results do not depend on the schedule.
template <class Function>
inline void parallel_for(std::size_t count, int threads, Function function) {
    require(threads > 0, "threads must be positive");
    require(count <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
            "batch too large");
    const auto division = vmath::division_mode();
    std::exception_ptr failure;
    std::mutex mutex;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
        try {
            const vmath::DivisionScope scope(division);
            function(static_cast<std::size_t>(i));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure)
                failure = std::current_exception();
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}
// Work within the caller's existing team. The implicit taskgroup keeps the
// callback and its referenced inputs alive until all items have completed.
template <class Function>
inline void task_for(std::size_t count, bool parallel, Function function) {
#if defined(_OPENMP) && _OPENMP >= 201511
    if (parallel && omp_in_parallel()) {
        const auto division = vmath::division_mode();
        std::exception_ptr failure;
        std::mutex mutex;
#pragma omp taskloop grainsize(1) shared(function, failure, mutex)
        for (std::size_t i = 0; i < count; ++i) {
            try {
                const vmath::DivisionScope scope(division);
                function(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure)
                    failure = std::current_exception();
            }
        }
        if (failure)
            std::rethrow_exception(failure);
        return;
    }
#else
    (void)parallel;
#endif
    for (std::size_t i = 0; i < count; ++i)
        function(i);
}
inline std::vector<FitResult> fit_batch(const Matrix &design, const SparseCSC &responses,
                                        FitOptions options = {}, int threads = 1,
                                        const Vector &theta_trend = {}) {
    const vmath::DivisionScope division(options.division);
    responses.validate();
    design.validate();
    require(design.rows == responses.rows, "batch dimensions mismatch");
    require(theta_trend.empty() || theta_trend.size() == responses.cols,
            "dispersion trend size mismatch");
    options.retain_observations = false;
    if (!options.design && options.weights.empty())
        options.design = prepare_design(design, options.rank_tolerance);
    std::vector<FitResult> out(responses.cols);
    parallel_for(responses.cols, threads, [&](auto j) {
        auto o = options;
        if (!theta_trend.empty()) {
            o.theta = theta_trend[j];
            o.dispersion = Dispersion::fixed;
            o.cox_reid = false; // no dispersion is estimated
        }
        out[j] = fit(design, responses.column(j), o);
    });
    return out;
}
inline Vector estimate_dispersion_trend(const SparseCSC &responses,
                                        const std::vector<FitResult> &fits) {
    responses.validate();
    require(fits.size() == responses.cols, "trend fits mismatch");
    Vector means(responses.cols), logt;
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (std::size_t j = 0; j < responses.cols; ++j) {
        double m = 0;
        for (auto k = responses.column_ptr[j]; k < responses.column_ptr[j + 1]; ++k)
            m += responses.values[k];
        means[j] = std::log(std::max(m / responses.rows, 1e-12));
        if (fits[j].ok() && !fits[j].has_boundary() &&
            fits[j].family == Family::negative_binomial && !fits[j].theta_boundary) {
            double x = means[j], y = std::log(1 / fits[j].theta);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            ++n;
        }
    }
    require(n >= 2 && n * sxx - sx * sx > 1e-12,
            "insufficient valid features for dispersion trend");
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx), intercept = (sy - slope * sx) / n;
    for (double &m : means)
        m = std::exp(std::clamp(-intercept - slope * m, std::log(1e-6), std::log(1e8)));
    return means;
}
// Profile ML shared theta; failures invalidate the objective rather than disappear.
inline double estimate_shared_theta(const Matrix &design, const SparseCSC &responses,
                                    FitOptions options = {}, int threads = 1) {
    responses.validate();
    require(responses.cols > 0, "no features for shared dispersion");
    require(!options.cox_reid, "the Cox-Reid adjustment is not available for a shared dispersion");
    options.family = Family::negative_binomial;
    options.dispersion = Dispersion::fixed;
    auto objective = [&](double u) {
        options.theta = std::exp(u);
        auto fits = fit_batch(design, responses, options, threads);
        double ll = 0;
        for (const auto &f : fits) {
            require(f.ok(), "shared dispersion requires valid fits for every supplied feature");
            ll += f.log_likelihood;
        }
        return ll;
    };
    double lo = std::log(options.theta_min), hi = std::log(options.theta_max),
           r = 0.6180339887498948482;
    double a = hi - r * (hi - lo), b = lo + r * (hi - lo), fa = objective(a), fb = objective(b);
    for (unsigned it = 0; it < 80 && hi - lo > options.tolerance; ++it) {
        if (fa > fb) {
            hi = b;
            b = a;
            fb = fa;
            a = hi - r * (hi - lo);
            fa = objective(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + r * (hi - lo);
            fb = objective(b);
        }
    }
    return std::exp((lo + hi) / 2);
}
} // namespace rapidglm
