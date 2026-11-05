// Special functions of double (include/rapidglm/vmath.hpp) against glibc's libm
// and boost::math: the largest error in units in the last place against long double
// references, and nanoseconds per element, over standard input ranges and the arguments
// of NB likelihoods, y + 1/alpha, for arrays (rapidglm_avx2, rapidglm_portable) and one
// value at a time (rapidglm_scalar); and, for digamma and trigamma, the scalar calls
// of the count paths (one argument repeated, and calls whose result the next needs).
// One thread.
//   rapidglm_special_functions accuracy N        JSON on stdout
//   rapidglm_special_functions speed SECONDS     JSON on stdout
//   rapidglm_special_functions values            the known values of
//   tests/fixtures/vmath_known_values.tsv
#include "boost_polygamma.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/version.hpp>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <gnu/libc-version.h>
#include <random>
#include <rapidglm/vmath.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace vm = rapidglm::vmath;
using LD = long double;
using Array = std::function<void(const double *, double *, std::size_t)>;
namespace {
// boost::math as callers use it by default (double promoted to long double
// internally), and without that promotion.
using NoPromotion = boost::math::policies::policy<boost::math::policies::promote_double<false>>;
template <double (*F)(double)> void loop(const double *x, double *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = F(x[i]);
}
double libm_log(double x) {
    return std::log(x);
}
double libm_lgamma_r(double x) {
    int sign;
    return ::lgamma_r(x, &sign);
}
double libm_log1p(double x) {
    return std::log1p(x);
}
double libm_exp(double x) {
    return std::exp(x);
}
double libm_lgamma(double x) {
    return std::lgamma(x);
}
// boost::math throws on overflow and poles under its default policy; those
// arguments count as +-inf (the sign of the true value).
template <class F> double caught(F f, double x, double overflow) {
    try {
        return f(x);
    } catch (const std::overflow_error &) {
        return overflow;
    } catch (const std::domain_error &) {
        return NAN;
    }
}
double boost_lgamma(double x) {
    return caught([](double v) { return boost::math::lgamma(v); }, x, INFINITY);
}
double boost_digamma(double x) {
    return caught([](double v) { return boost::math::digamma(v); }, x, -INFINITY);
}
double boost_trigamma(double x) {
    return caught([](double v) { return boost::math::trigamma(v); }, x, INFINITY);
}
double boost_lgamma_double(double x) {
    return caught([](double v) { return boost::math::lgamma(v, NoPromotion()); }, x, INFINITY);
}
double boost_digamma_double(double x) {
    return caught([](double v) { return boost::math::digamma(v, NoPromotion()); }, x, -INFINITY);
}
double boost_trigamma_double(double x) {
    return caught([](double v) { return boost::math::trigamma(v, NoPromotion()); }, x, INFINITY);
}

#ifdef RAPIDGLM_VMATH_AVX2
// boost::math's double-precision digamma and trigamma algorithms in four AVX2 lanes
// (boost_polygamma.hpp): one division per recurrence step, or one in all.
__attribute__((target("avx2,fma"))) void boost_algorithm_digamma(const double *x, double *out,
                                                                 std::size_t n) {
    vm::detail::avx2::apply<boost_polygamma::avx2::digamma<false>>(x, out, n);
}
__attribute__((target("avx2,fma"))) void
boost_algorithm_digamma_one_division(const double *x, double *out, std::size_t n) {
    vm::detail::avx2::apply<boost_polygamma::avx2::digamma<true>>(x, out, n);
}
__attribute__((target("avx2,fma"))) void boost_algorithm_trigamma(const double *x, double *out,
                                                                  std::size_t n) {
    vm::detail::avx2::apply<boost_polygamma::avx2::trigamma>(x, out, n);
}
#endif
struct Implementation {
    std::string name;
    Array f;
};
struct Function {
    std::string name;
    std::function<LD(LD)> reference;
    std::vector<Implementation> implementations;
    std::vector<std::string> ranges;
};
std::vector<Function> functions() {
    std::vector<Function> f = {{"log",
                                [](LD x) { return logl(x); },
                                {{"glibc", loop<libm_log>},
                                 {"rapidglm_portable", vm::log_array_portable},
                                 {"rapidglm_scalar", loop<vm::log>}},
                                {"tiny", "unit", "moderate", "large", "nb"}},
                               {"log1p",
                                [](LD x) { return log1pl(x); },
                                {{"glibc", loop<libm_log1p>},
                                 {"rapidglm_portable", vm::log1p_array_portable},
                                 {"rapidglm_scalar", loop<vm::log1p>}},
                                {"negative", "tiny", "unit", "moderate", "large"}},
                               {"exp",
                                [](LD x) { return expl(x); },
                                {{"glibc", loop<libm_exp>},
                                 {"rapidglm_portable", vm::exp_array_portable},
                                 {"rapidglm_scalar", loop<vm::exp>}},
                                {"exp_small", "exp_wide"}},
                               {"lgamma",
                                [](LD x) { return lgammal(x); },
                                {{"glibc", loop<libm_lgamma>},
                                 {"glibc_lgamma_r", loop<libm_lgamma_r>},
                                 {"rapidglm_portable", vm::lgamma_array_portable},
                                 {"rapidglm_scalar", loop<vm::lgamma>},
                                 {"rapidglm_dispatched", vm::lgamma_array},
                                 {"boost", loop<boost_lgamma>},
                                 {"boost_double", loop<boost_lgamma_double>}},
                                {"tiny", "unit", "moderate", "large", "nb"}},
                               {"digamma",
                                [](LD x) { return boost::math::digamma(x); },
                                {{"boost", loop<boost_digamma>},
                                 {"boost_double", loop<boost_digamma_double>},
                                 {"rapidglm_portable", vm::digamma_array_portable},
                                 {"rapidglm_scalar", loop<vm::digamma>}},
                                {"tiny", "unit", "moderate", "large", "nb"}},
                               {"trigamma",
                                [](LD x) { return boost::math::trigamma(x); },
                                {{"boost", loop<boost_trigamma>},
                                 {"boost_double", loop<boost_trigamma_double>},
                                 {"rapidglm_portable", vm::trigamma_array_portable},
                                 {"rapidglm_scalar", loop<vm::trigamma>}},
                                {"tiny", "unit", "moderate", "large", "nb"}}};
#ifdef RAPIDGLM_VMATH_AVX2
    if (vm::avx2_active()) {
        f[0].implementations.push_back({"rapidglm_avx2", vm::log_array_avx2});
        f[1].implementations.push_back({"rapidglm_avx2", vm::log1p_array_avx2});
        f[2].implementations.push_back({"rapidglm_avx2", vm::exp_array_avx2});
        f[4].implementations.push_back({"rapidglm_avx2", vm::digamma_array_avx2});
        f[5].implementations.push_back({"rapidglm_avx2", vm::trigamma_array_avx2});
        f[4].implementations.push_back({"boost_algorithm_avx2", boost_algorithm_digamma});
        f[4].implementations.push_back(
            {"boost_algorithm_avx2_one_division", boost_algorithm_digamma_one_division});
        f[5].implementations.push_back({"boost_algorithm_avx2", boost_algorithm_trigamma});
    }
#endif
    return f;
}
// Arguments: log-uniform tiny (1e-300, 1e-5) and large (100, 1e300), uniform
// unit (1e-5, 3) and moderate (3, 100), negative (-1, 0) for log1p, exp over
// (-1, 1) and (-700, 700), and nb: y + 1/alpha with y geometric (mean 20) and
// alpha log-uniform in (1e-8, 1e3).
std::vector<double> arguments(const std::string &range, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    std::geometric_distribution<int> counts(1.0 / 21);
    const auto logu = [&](double lo, double hi) {
        return std::exp(std::log(lo) + u(rng) * (std::log(hi) - std::log(lo)));
    };
    std::vector<double> x(n);
    for (auto &v : x)
        v = range == "tiny"        ? logu(1e-300, 1e-5)
            : range == "unit"      ? 1e-5 + u(rng) * (3 - 1e-5)
            : range == "moderate"  ? 3 + u(rng) * 97
            : range == "large"     ? logu(100, 1e300)
            : range == "negative"  ? -u(rng)
            : range == "exp_small" ? -1 + 2 * u(rng)
            : range == "exp_wide"  ? -700 + 1400 * u(rng)
                                   : counts(rng) + 1 / logu(1e-8, 1e3);
    return x;
}
double ulp_error(double v, LD reference) {
    const double r = static_cast<double>(reference);
    if (std::isnan(v) && std::isnan(r))
        return 0;
    if (!std::isfinite(r))
        return v == r ? 0 : INFINITY;
    const int e = std::max(r == 0 ? -1074 + 52 : std::ilogb(r), -1022);
    return static_cast<double>(fabsl(v - reference) / std::ldexp(1.0L, e - 52));
}
void accuracy(std::size_t n) {
    std::printf("{\n \"accuracy\": [");
    bool first = true;
    for (const auto &f : functions())
        for (const auto &range : f.ranges) {
            const auto x = arguments(range, n, 1 + std::hash<std::string>()(f.name + range) % 1000);
            std::vector<LD> reference(n);
            for (std::size_t i = 0; i < n; ++i)
                reference[i] = f.reference(x[i]);
            for (const auto &imp : f.implementations) {
                std::vector<double> out(n);
                imp.f(x.data(), out.data(), n);
                double worst = 0, at = 0, sum = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double e = ulp_error(out[i], reference[i]);
                    sum += e;
                    if (!(e <= worst)) {
                        worst = e;
                        at = x[i];
                    }
                }
                std::printf(
                    "%s\n  {\"function\": \"%s\", \"range\": \"%s\", \"implementation\": \"%s\", "
                    "\"n\": %zu, \"max_ulp\": %.4f, \"at\": %.17g, \"mean_ulp\": %.4f}",
                    first ? "" : ",", f.name.c_str(), range.c_str(), imp.name.c_str(), n, worst, at,
                    sum / n);
                first = false;
            }
        }
    std::printf("\n ]\n}\n");
}
// The least time per element over repeated calls on 4096 arguments (in cache),
// within the time budget per function and range.
void speed(double seconds) {
    using Clock = std::chrono::steady_clock;
    std::printf("{\n \"speed\": [");
    bool first = true;
    const std::size_t n = 4096;
    for (const auto &f : functions())
        for (const auto &range : f.ranges) {
            const auto x = arguments(range, n, 7);
            std::vector<double> out(n);
            const double budget = seconds / (f.implementations.size() * f.ranges.size());
            for (const auto &imp : f.implementations) {
                double best = INFINITY;
                const auto start = Clock::now();
                std::size_t calls = 0;
                while (calls < 5 ||
                       std::chrono::duration<double>(Clock::now() - start).count() < budget) {
                    const auto t0 = Clock::now();
                    imp.f(x.data(), out.data(), n);
                    best = std::min(
                        best,
                        std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / n);
                    ++calls;
                }
                std::printf(
                    "%s\n  {\"function\": \"%s\", \"range\": \"%s\", \"implementation\": \"%s\", "
                    "\"ns_per_element\": %.3f, \"calls\": %zu}",
                    first ? "" : ",", f.name.c_str(), range.c_str(), imp.name.c_str(), best, calls);
                first = false;
            }
        }
    // digamma and trigamma as the count paths call them, one value at a time:
    // "repeated", one argument for every call (edgeR-QL's squeezed variances, at
    // half the residual degrees of freedom); "dependent", each call's argument
    // depending on the previous result (the theta Newton steps).
    std::printf("\n ],\n \"scalar_calls\": [");
    first = true;
    using Scalar = double (*)(double);
    const std::vector<std::pair<std::string, std::vector<std::pair<std::string, Scalar>>>> calls = {
        {"digamma",
         {{"boost", boost_digamma},
          {"boost_double", boost_digamma_double},
          {"rapidglm", vm::digamma}}},
        {"trigamma",
         {{"boost", boost_trigamma},
          {"boost_double", boost_trigamma_double},
          {"rapidglm", vm::trigamma}}}};
    for (const auto &[name, implementations] : calls)
        for (double argument : {1.5, 4.0, 15.0, 300.0})
            for (const char *pattern : {"repeated", "dependent"})
                for (const auto &[label, g] : implementations) {
                    const std::vector<double> x(n, argument);
                    std::vector<double> out(n);
                    const bool dependent = pattern[0] == 'd';
                    double best = INFINITY;
                    const auto start = Clock::now();
                    std::size_t calls_made = 0;
                    while (calls_made < 5 ||
                           std::chrono::duration<double>(Clock::now() - start).count() <
                               seconds / 100) {
                        const auto t0 = Clock::now();
                        if (dependent) {
                            double y = 0;
                            for (std::size_t i = 0; i < n; ++i)
                                y = g(x[i] + y * 0);
                            out[0] = y;
                        } else
                            for (std::size_t i = 0; i < n; ++i)
                                out[i] = g(x[i]);
                        best = std::min(
                            best,
                            std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                                n);
                        ++calls_made;
                    }
                    std::printf(
                        "%s\n  {\"function\": \"%s\", \"argument\": %g, \"pattern\": \"%s\", "
                        "\"implementation\": \"%s\", \"ns_per_call\": %.3f}",
                        first ? "" : ",", name.c_str(), argument, pattern, label.c_str(), best);
                    first = false;
                }
    std::printf("\n ],\n \"versions\": {\"glibc\": \"%s\", \"boost\": \"%s\", "
                "\"compiler\": \"%s\", \"vmath_path\": \"%s\"}\n}\n",
                gnu_get_libc_version(), BOOST_LIB_VERSION, __VERSION__, vm::path());
}
// x and f(x) as bit patterns, for the binding tests' bit-for-bit comparison of
// RapidGLM's functions (not lgamma, whose bits follow the C library).
void values() {
    const double inf = INFINITY;
    std::vector<double> x = {
        5e-324, 1e-300, 1e-10, 0.001, 0.25, 0.5,  0.75, 1,     1.0000001, 1.4616321449683622,
        1.5,    2,      2.5,   3,     4.75, 5,    6,    10,    33.5,      100,
        1234.5, 1e6,    1e15,  1e200, 0,    -0.5, -1,   -2.25, -700,      700,
        709.5,  inf};
    for (int y : {0, 1, 5, 40})
        for (double alpha : {1e-8, 1e-3, 0.1, 1.0, 10.0, 1e3})
            x.push_back(y + 1 / alpha);
    const std::vector<std::pair<const char *, void (*)(const double *, double *, std::size_t)>> f =
        {{"log", vm::log_array},
         {"log1p", vm::log1p_array},
         {"exp", vm::exp_array},
         {"digamma", vm::digamma_array},
         {"trigamma", vm::trigamma_array}};
    std::printf("function\tx\tvalue\n");
    for (const auto &[name, g] : f) {
        std::vector<double> out(x.size());
        g(x.data(), out.data(), x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            std::uint64_t a, b;
            std::memcpy(&a, &x[i], 8);
            std::memcpy(&b, &out[i], 8);
            std::printf("%s\t%016" PRIx64 "\t%016" PRIx64 "\n", name, a, b);
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "accuracy")
        accuracy(argc > 2 ? std::stoul(argv[2]) : 1000000);
    else if (mode == "speed")
        speed(argc > 2 ? std::stod(argv[2]) : 60);
    else if (mode == "values")
        values();
    else {
        std::fprintf(stderr,
                     "usage: rapidglm_special_functions accuracy N | speed SECONDS | values\n");
        return 2;
    }
}
