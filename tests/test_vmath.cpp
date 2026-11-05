#include "vmath_reference.hpp"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <rapidglm/sparse.hpp>
#include <rapidglm/vmath.hpp>
#include <string>
#include <thread>
#include <vector>

namespace vm = rapidglm::vmath;
namespace {
using Array = void (*)(const double *, double *, std::size_t);
struct Function {
    std::string name;
    Array portable, dispatched;
    double (*scalar)(double);
    bool libm; // the C library's (bits and NaN payloads platform-dependent) or RapidGLM's
};
const std::vector<Function> &functions() {
    static const std::vector<Function> f = {
        {"log", vm::log_array_portable, vm::log_array, vm::log, false},
        {"log1p", vm::log1p_array_portable, vm::log1p_array, vm::log1p, false},
        {"exp", vm::exp_array_portable, vm::exp_array, vm::exp, false},
        {"lgamma", vm::lgamma_array_portable, vm::lgamma_array, vm::lgamma, true},
        {"digamma", vm::digamma_array_portable, vm::digamma_array, vm::digamma, false},
        {"trigamma", vm::trigamma_array_portable, vm::trigamma_array, vm::trigamma, false}};
    return f;
}
std::uint64_t bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
double from_bits(std::uint64_t u) {
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
// The same inputs on every platform, from integer arithmetic and exact (or
// correctly rounded) operations only: random bit patterns of positive doubles
// (every binade, subnormals included), fine grids around 0, 1, 2 and the
// digamma root, NB shapes y + 1/alpha for alpha = 2^(e/4), 1e-8 to 1e3, and
// special values, and the negatives of some.
std::vector<double> inputs() {
    std::vector<double> x;
    std::uint64_t state = 20261024;
    const auto next = [&] { // splitmix64
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < 200000; ++i)
        x.push_back(from_bits(next() % 0x7FF0000000000000ULL));
    for (int k = -4000; k <= 4000; ++k) {
        x.push_back(k * 0x1p-12 + 6);                  // [5, 7)
        x.push_back(1 + k * 0x1p-30);                  // around 1
        x.push_back(2 + k * 0x1p-30);                  // around 2
        x.push_back(1.4616321449683622 + k * 0x1p-42); // around the digamma root
        x.push_back(std::abs(k) * 0x1p-14);            // [0, 0.25)
    }
    for (int y = 0; y <= 300; ++y)
        for (int e = -108; e <= 43; ++e) // 1/alpha from 2^-10 to 1.75 x 2^27
            x.push_back(y + std::ldexp(1 + 0.25 * (e & 3), -(e >> 2)));
    const double inf = std::numeric_limits<double>::infinity();
    for (double v :
         {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 5e-324,
          2.2250738585072014e-308, 1.7976931348623157e308, -1.0, -2.0, -0.5, 709.78, -745.2})
        x.push_back(v);
    const std::size_t positive = x.size();
    for (std::size_t i = 0; i < positive; i += 7)
        x.push_back(-x[i]);
    return x;
}
bool same(double a, double b, bool libm) {
    return bits(a) == bits(b) || (libm && std::isnan(a) && std::isnan(b));
}
// FNV-1a over the output bits, every NaN as one pattern.
std::uint64_t digest(const std::vector<double> &v) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (double d : v) {
        std::uint64_t u = std::isnan(d) ? 0x7ff8000000000000ULL : bits(d);
        for (int b = 0; b < 8; ++b) {
            h ^= (u >> (8 * b)) & 0xff;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}
} // namespace

// Every path of every function gives the same bits: the portable (scalar FMA)
// path, the dispatched one (AVX2 or Advanced SIMD where available) and the
// scalar form (lgamma, the C library's, has one path).
BOOST_AUTO_TEST_CASE(vmath_paths_are_bit_identical) {
    for (auto mode : {vm::DivisionMode::slow, vm::DivisionMode::fast}) {
        const vm::DivisionScope scope(mode);
        BOOST_TEST_CONTEXT("division " << vm::division_name()) {

            const auto x = inputs();
            BOOST_TEST_MESSAGE("vmath path: " << vm::path());
            for (const auto &f : functions()) {
                std::vector<double> a(x.size()), b(x.size());
                f.portable(x.data(), a.data(), x.size());
                f.dispatched(x.data(), b.data(), x.size());
                std::size_t differ = 0, scalar_differ = 0;
                for (std::size_t i = 0; i < x.size(); ++i) {
                    differ += !same(a[i], b[i], f.libm);
                    scalar_differ += !same(a[i], f.scalar(x[i]), f.libm);
                }
                BOOST_TEST_INFO(f.name);
                BOOST_TEST(differ == 0u);
                BOOST_TEST_INFO(f.name);
                BOOST_TEST(scalar_differ == 0u);
                // In place, and every length of the masked tail.
                std::vector<double> c(x.begin(), x.begin() + 11), d(11);
                for (std::size_t n = 0; n <= 11; ++n) {
                    std::vector<double> e(c.begin(), c.begin() + n);
                    f.dispatched(e.data(), e.data(), n);
                    f.portable(c.data(), d.data(), n);
                    for (std::size_t i = 0; i < n; ++i)
                        BOOST_TEST(same(e[i], d[i], f.libm));
                }
            }
        }
    }
}

// The same digests on every platform (x86-64 AVX2 and portable, AArch64 Advanced
// SIMD and portable): RapidGLM's results do not depend on the hardware (lgamma's
// follow the C library and are not checked).
BOOST_AUTO_TEST_CASE(vmath_known_answer_digests) {
    for (auto mode : {vm::DivisionMode::slow, vm::DivisionMode::fast}) {
        const vm::DivisionScope scope(mode);
        BOOST_TEST_CONTEXT("division " << vm::division_name()) {

            const auto x = inputs();
            const std::vector<std::pair<std::string, std::uint64_t>> expected = {
                {"log", 0x041e2d41d9503f5bULL},
                {"log1p",
                 mode == vm::DivisionMode::slow ? 0x9f1c6f67a18fba58ULL : 0xa704f61f54fc0c52ULL},
                {"exp", 0x3b6f865d9b3d2499ULL},
                {"digamma",
                 mode == vm::DivisionMode::slow ? 0x9bef21efa1f73abeULL : 0x68e917844005145ULL},
                {"trigamma", 0x5249ee3947805d62ULL}};
            for (const auto &f : functions()) {
                std::vector<double> out(x.size());
                f.dispatched(x.data(), out.data(), x.size());
                const auto h = digest(out);
                BOOST_TEST_MESSAGE(f.name << " digest 0x" << std::hex << h << std::dec);
                for (const auto &[name, value] : expected)
                    if (name == f.name) {
                        BOOST_TEST_INFO(f.name);
                        BOOST_TEST(h == value);
                    }
            }
        }
    }
}

// Errors against mpmath (scripts/vmath_reference.py), in units in the last place
// of the correctly rounded value, within the bounds METHODS.md documents (lgamma:
// the C library's, 2.5 ulp at x = -2.5 with glibc 2.35).
BOOST_AUTO_TEST_CASE(vmath_matches_mpmath_reference) {
    for (auto mode : {vm::DivisionMode::slow, vm::DivisionMode::fast}) {
        const vm::DivisionScope scope(mode);
        BOOST_TEST_CONTEXT("division " << vm::division_name()) {

            const auto check = [](const char *name, auto &table, double (*f)(double),
                                  double bound) {
                double worst = 0, at = 0;
                for (const auto &r : table) {
                    const double v = f(r.x);
                    const double ulp = std::ldexp(1.0, std::max(std::ilogb(r.hi), -1022) - 52);
                    const double error = std::abs(static_cast<long double>(v) - r.hi - r.lo) / ulp;
                    if (!(error <= worst)) {
                        worst = error;
                        at = r.x;
                    }
                }
                BOOST_TEST_MESSAGE(name << ": max " << worst << " ulp at " << at);
                BOOST_TEST_INFO(name << " at " << at);
                BOOST_TEST(worst <= bound);
            };
            check("log", vmath_reference_log, vm::log, 1.0);
            check("log1p", vmath_reference_log1p, vm::log1p, 1.0);
            check("exp", vmath_reference_exp, vm::exp, 1.0);
            check("lgamma", vmath_reference_lgamma, vm::lgamma, 4.0);
            check("digamma", vmath_reference_digamma, vm::digamma, 3.5);
            check("trigamma", vmath_reference_trigamma, vm::trigamma, 3.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(vmath_special_values) {
    for (auto mode : {vm::DivisionMode::slow, vm::DivisionMode::fast}) {
        const vm::DivisionScope scope(mode);
        BOOST_TEST_CONTEXT("division " << vm::division_name()) {

            const double inf = std::numeric_limits<double>::infinity();
            const double na =
                from_bits(0x7FF00000000007A2ULL); // R's NA_real_: a NaN with payload 1954
            for (const auto &f : {vm::log, vm::log1p, vm::exp})
                BOOST_TEST(bits(f(na)) == bits(na)); // payload kept
            for (const auto &f : {vm::digamma, vm::trigamma}) {
                BOOST_TEST(bits(f(na)) == bits(na)); // payload kept
                BOOST_TEST(std::isnan(f(-1.0)));
                BOOST_TEST(std::isnan(f(-0.5)));
                BOOST_TEST(std::isnan(f(-inf)));
            }
            BOOST_TEST(vm::digamma(0.0) == -inf);
            BOOST_TEST(vm::digamma(-0.0) == -inf);
            BOOST_TEST(vm::digamma(inf) == inf);
            BOOST_TEST(vm::trigamma(0.0) == inf);
            BOOST_TEST(vm::trigamma(inf) == 0.0);
            BOOST_TEST(vm::trigamma(1e-160) == inf); // 1/x^2 overflows
            BOOST_TEST(vm::digamma(1e-310) == -inf); // -1/x overflows
            BOOST_TEST(vm::log(0.0) == -inf);
            BOOST_TEST(vm::log(-0.0) == -inf);
            BOOST_TEST(std::isnan(vm::log(-1.0)));
            BOOST_TEST(vm::log(inf) == inf);
            BOOST_TEST(bits(vm::log(1.0)) == bits(0.0));
            BOOST_TEST(vm::log(5e-324) == -744.4400719213812); // subnormal, correctly rounded
            BOOST_TEST(vm::log1p(-1.0) == -inf);
            BOOST_TEST(std::isnan(vm::log1p(-1.5)));
            BOOST_TEST(vm::log1p(inf) == inf);
            BOOST_TEST(bits(vm::log1p(-0.0)) == bits(-0.0));
            BOOST_TEST(vm::log1p(-0x1p-61) == -0x1p-61); // rounds to x below 2^-60
            BOOST_TEST(vm::log1p(1e300) == 690.7755278982137);
            BOOST_TEST(vm::exp(710.0) == inf);
            BOOST_TEST(vm::exp(inf) == inf);
            BOOST_TEST(vm::exp(-inf) == 0.0);
            BOOST_TEST(vm::exp(-746.0) == 0.0);
            BOOST_TEST(vm::exp(-745.0) == 5e-324); // gradual underflow
            BOOST_TEST(vm::exp(709.5) == 1.3549863193146328e308);
            BOOST_TEST(vm::lgamma(1.0) == 0.0);
            BOOST_TEST(vm::lgamma(2.0) == 0.0);
            BOOST_TEST(vm::lgamma(0.0) == inf);
            BOOST_TEST(vm::lgamma(-2.0) == inf);
            // Small exact cases: digamma(1) = -gamma, trigamma(1) = pi^2/6.
            BOOST_TEST(std::abs(vm::digamma(1.0) + 0.57721566490153286) <= 2e-16);
            BOOST_TEST(std::abs(vm::trigamma(1.0) - 1.6449340668482264) <= 4e-16);
        }
    }
}

BOOST_AUTO_TEST_CASE(vmath_division_scopes_and_workers) {
    using Mode = vm::DivisionMode;
    const auto initial = vm::division_mode();
    BOOST_CHECK_THROW(vm::parse_division("auto"), std::invalid_argument);
    BOOST_CHECK_THROW(vm::set_division(static_cast<Mode>(8)), std::invalid_argument);
    for (auto mode : {Mode::slow, Mode::fast}) {
        const vm::DivisionScope scope(mode);
        BOOST_CHECK(vm::division_mode() == mode);
        std::vector<int> actual(32);
        rapidglm::parallel_for(actual.size(), 4, [&](auto i) {
            actual[i] = static_cast<int>(vm::division_mode());
            const vm::DivisionScope nested(mode == Mode::slow ? Mode::fast : Mode::slow);
        });
        for (auto value : actual)
            BOOST_TEST(value == static_cast<int>(mode));
        const auto opposite = mode == Mode::slow ? Mode::fast : Mode::slow;
        std::thread other([&] { const vm::DivisionScope independent(opposite); });
        other.join();
        BOOST_CHECK(vm::division_mode() == mode);
        try {
            const vm::DivisionScope failed(opposite);
            throw std::runtime_error("restore after error");
        } catch (const std::runtime_error &) {
        }
        BOOST_CHECK(vm::division_mode() == mode);
    }
    BOOST_CHECK(vm::division_mode() == initial);
}

BOOST_AUTO_TEST_CASE(vmath_fast_positive_lgamma_digest) {
    const vm::DivisionScope scope(vm::DivisionMode::fast);
    auto x = inputs();
    x.erase(std::remove_if(x.begin(), x.end(), [](double v) { return !(v >= 0); }), x.end());
    std::vector<double> out(x.size());
    vm::lgamma_array(x.data(), out.data(), x.size());
    BOOST_TEST(digest(out) == 0x2468864ca0a96375ULL);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = vm::fast::trigamma(x[i]);
        if (!same(out[i], vm::fast::trigamma_kernel(x[i]), false)) {
            BOOST_ERROR("engine trigamma paths differ at " << x[i]);
            break;
        }
    }
    BOOST_TEST(digest(out) == 0x27a1cca60c27cefaULL);
}
