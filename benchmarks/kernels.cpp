// Deterministic inputs, setup outside timed regions, and observable checksums.
// Run several processes and compare medians; these are not end-to-end timings.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <rapidglm/rapidglm.hpp>

using namespace rapidglm;
namespace {
volatile double sink = 0;
template <class F>
void measure(const char *kernel, std::size_t n, std::size_t p, double probability,
             std::size_t iterations, F operation) {
    sink = operation(0); // Untimed warmup.
    double checksum = 0;
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        checksum += operation(i);
    double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    sink = checksum;
    std::cout << kernel << ',' << n << ',' << p << ',' << probability << ',' << iterations << ','
              << seconds << ',' << checksum << '\n';
}
Matrix design(std::size_t n, std::size_t p) {
    Matrix x(n, p, 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 1; j < p; ++j)
            x(i, j) = std::sin((i + 1) * (j + 1) * 0.731);
    return x;
}
} // namespace
int main() {
    std::cout << std::setprecision(17) << "kernel,n,p,probability,iterations,seconds,checksum\n";
    for (std::size_t p : {4u, 16u, 64u, 128u}) {
        Matrix a = detail::crossprod(design(2 * p, p), Vector(2 * p, 1));
        for (std::size_t j = 0; j < p; ++j)
            a(j, j) += p;
        detail::Cholesky factor(a);
        require(factor.ok, "benchmark factorization failed");
        Vector b(p, 1);
        const std::size_t count = p <= 16 ? 20000 : 500;
        measure("factor", p, p, 0, count, [&](auto i) {
            detail::Cholesky f(a);
            require(f.ok, "benchmark factorization failed");
            return f.l(i % p, i % p);
        });
        measure("solve", p, p, 0, count, [&](auto i) { return factor.solve(b)[i % p]; });
        measure("inverse", p, p, 0, count, [&](auto i) { return factor.inverse()(i % p, i % p); });
    }
    for (std::size_t n : {4096u, 100286u}) {
        auto x = design(n, 4);
        Vector y(n);
        for (std::size_t i = 0; i < n; ++i)
            y[i] = i % 7;
        FitOptions options;
        options.family = Family::poisson;
        auto fitted = fit(x, y, options);
        require(fitted.ok(), "benchmark fit failed");
        BinaryScoreWorkspace score{ScoreWorkspace(x, fitted)};
        measure("uniform", n, 0, 0, 128, [&](auto draw) {
            CounterRng rng(97, 13, draw);
            double total = 0;
            for (std::size_t i = 0; i < n; ++i)
                total += rng.uniform();
            return total;
        });
        for (double probability : {0.01, 0.3}) {
            Vector probabilities(n);
            for (std::size_t i = 0; i < n; ++i)
                probabilities[i] = probability * (0.5 + (i % 101) / 100.0);
            auto sampler = crt_sampler(probabilities);
            std::vector<std::size_t> selected;
            // Bank is for isolating score cost only, not a production draw cache.
            std::vector<std::vector<std::size_t>> bank(128);
            for (std::size_t d = 0; d < bank.size(); ++d) {
                CounterRng rng(97, 13, d);
                sampler(rng, bank[d]);
            }
            measure("crt_indices", n, 4, probability, 128, [&](auto d) {
                CounterRng rng(97, 13, d);
                sampler(rng, selected);
                return static_cast<double>(selected.size());
            });
            measure("score_only", n, 4, probability, 128,
                    [&](auto d) { return score.statistic(bank[d]); });
            measure("crt_score", n, 4, probability, 128, [&](auto d) {
                CounterRng rng(97, 13, d);
                sampler(rng, selected);
                return score.statistic(selected);
            });
        }
    }
    auto x = design(2000, 4);
    for (auto family :
         {Family::gaussian, Family::poisson, Family::binomial, Family::negative_binomial}) {
        Vector y(x.rows);
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] = family == Family::binomial ? i % 3 == 0 : i % 7;
        FitOptions options;
        options.family = family;
        options.dispersion = Dispersion::fixed;
        const char *name = family == Family::gaussian   ? "fit_gaussian"
                           : family == Family::poisson  ? "fit_poisson"
                           : family == Family::binomial ? "fit_binomial"
                                                        : "fit_nb_fixed";
        measure(name, x.rows, x.cols, 0, 20, [&](auto) {
            auto f = fit(x, y, options);
            require(f.ok(), "benchmark fit failed");
            return f.coefficients[0];
        });
    }
}
