// Binary128 reference maximum-likelihood fits for the GLM comparison suite.
// Independent of the engine: damped Newton on the exact log likelihood, with a
// profile search over the NB size. Used only to score other tools' accuracy.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
extern "C" {
#include <quadmath.h>
}
using Q = __float128;
namespace {
struct Problem {
    std::size_t n = 0, p = 0;
    std::string family;
    std::vector<double> x, y, offset, weights;
};
std::vector<double> read(const std::string &path, std::size_t count) {
    std::vector<double> v(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(count * 8));
    if (!f)
        throw std::runtime_error("cannot read " + path);
    return v;
}
std::string field(const std::string &json, const std::string &key) {
    auto k = json.find("\"" + key + "\"");
    auto c = json.find(':', k);
    auto e = json.find_first_of(",\n}", c + 1);
    std::string v = json.substr(c + 1, e - c - 1);
    v.erase(0, v.find_first_not_of(" \""));
    v.erase(v.find_last_not_of(" \"") + 1);
    return v;
}
Problem load(const std::string &dir) {
    std::ifstream m(dir + "/meta.json");
    std::stringstream s;
    s << m.rdbuf();
    Problem pr;
    pr.family = field(s.str(), "family");
    pr.n = std::stoul(field(s.str(), "n"));
    pr.p = std::stoul(field(s.str(), "p"));
    pr.x = read(dir + "/X.f64", pr.n * pr.p);
    pr.y = read(dir + "/y.f64", pr.n);
    pr.offset = read(dir + "/offset.f64", pr.n);
    pr.weights = read(dir + "/weights.f64", pr.n);
    return pr;
}
Q digammaq(Q x) {
    Q r = 0;
    while (x < 40) {
        r -= 1 / x;
        x += 1;
    }
    static const double b[] = {1.0 / 6,       -1.0 / 30,      1.0 / 42, -1.0 / 30,
                               5.0 / 66,      -691.0 / 2730,  7.0 / 6,  -3617.0 / 510,
                               43867.0 / 798, -174611.0 / 330};
    Q x2 = 1 / (x * x), term = x2, s = 0;
    for (int k = 1; k <= 10; ++k, term *= x2)
        s += Q(b[k - 1]) / (2 * k) * term;
    return r + logq(x) - 1 / (2 * x) - s;
}
struct Fit {
    std::vector<Q> beta;
    Q ll = 0, theta = 0;
    int iterations = 0;
    bool converged = false;
};
// Log likelihood, gradient and (negative) Hessian for fixed theta.
Q evaluate(const Problem &pr, const std::vector<Q> &beta, Q theta, std::vector<Q> *g,
           std::vector<Q> *h, bool expected = false) {
    const auto n = pr.n, p = pr.p;
    Q ll = 0;
    if (g)
        g->assign(p, 0);
    if (h)
        h->assign(p * p, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const Q a = pr.weights[i];
        if (a == 0)
            continue;
        Q eta = pr.offset[i];
        for (std::size_t j = 0; j < p; ++j)
            eta += Q(pr.x[i * p + j]) * beta[j];
        const Q y = pr.y[i];
        Q score, weight;
        if (pr.family == "gaussian") {
            ll -= a * (y - eta) * (y - eta) / 2;
            score = y - eta;
            weight = 1;
        } else if (pr.family == "binomial") {
            const Q mu = eta > 0 ? 1 / (1 + expq(-eta)) : expq(eta) / (1 + expq(eta));
            ll += a * (y * eta - (eta > 0 ? eta + log1pq(expq(-eta)) : log1pq(expq(eta))));
            score = y - mu;
            weight = mu * (1 - mu);
        } else if (pr.family == "poisson") {
            const Q mu = expq(eta);
            ll += a * (y * eta - mu - lgammaq(y + 1));
            score = y - mu;
            weight = mu;
        } else {
            const Q mu = expq(eta), t = theta;
            ll += a * (lgammaq(y + t) - lgammaq(t) - lgammaq(y + 1) + t * logq(t / (t + mu)) +
                       y * logq(mu / (t + mu)));
            score = (y - mu) * t / (t + mu);
            weight = expected ? mu * t / (t + mu) : mu * t * (y + t) / ((t + mu) * (t + mu));
        }
        for (std::size_t j = 0; j < p; ++j) {
            const Q xj = pr.x[i * p + j];
            if (g)
                (*g)[j] += a * score * xj;
            if (h)
                for (std::size_t k = 0; k <= j; ++k)
                    (*h)[j * p + k] += a * weight * xj * Q(pr.x[i * p + k]);
        }
    }
    if (h)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < j; ++k)
                (*h)[k * p + j] = (*h)[j * p + k];
    return ll;
}
// Solves h d = g for symmetric positive definite h with Jacobi scaling.
bool solve(std::vector<Q> h, std::vector<Q> g, std::size_t p, std::vector<Q> &d) {
    std::vector<Q> s(p);
    for (std::size_t j = 0; j < p; ++j) {
        if (!(h[j * p + j] > 0))
            return false;
        s[j] = sqrtq(h[j * p + j]);
    }
    for (std::size_t j = 0; j < p; ++j) {
        g[j] /= s[j];
        for (std::size_t k = 0; k < p; ++k)
            h[j * p + k] /= s[j] * s[k];
    }
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k < j; ++k)
            h[j * p + j] -= h[j * p + k] * h[j * p + k];
        if (!(h[j * p + j] > Q(1e-60)))
            return false;
        h[j * p + j] = sqrtq(h[j * p + j]);
        for (std::size_t i = j + 1; i < p; ++i) {
            for (std::size_t k = 0; k < j; ++k)
                h[i * p + j] -= h[i * p + k] * h[j * p + k];
            h[i * p + j] /= h[j * p + j];
        }
    }
    d = g;
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k < j; ++k)
            d[j] -= h[j * p + k] * d[k];
        d[j] /= h[j * p + j];
    }
    for (std::size_t j = p; j-- > 0;) {
        for (std::size_t k = j + 1; k < p; ++k)
            d[j] -= h[k * p + j] * d[k];
        d[j] /= h[j * p + j];
    }
    for (std::size_t j = 0; j < p; ++j)
        d[j] /= s[j];
    return true;
}
Fit newton(const Problem &pr, Q theta, std::vector<Q> beta) {
    Fit f;
    std::vector<Q> g, h, d;
    Q ll = evaluate(pr, beta, theta, &g, &h);
    for (f.iterations = 0; f.iterations < 500; ++f.iterations) {
        if (!solve(h, g, pr.p, d))
            break;
        Q step = 1, lnew = 0;
        std::vector<Q> next(pr.p);
        for (int halve = 0; halve < 200; ++halve, step /= 2) {
            for (std::size_t j = 0; j < pr.p; ++j)
                next[j] = beta[j] + step * d[j];
            lnew = evaluate(pr, next, theta, nullptr, nullptr);
            if (lnew >= ll)
                break;
        }
        Q change = 0;
        for (std::size_t j = 0; j < pr.p; ++j)
            change = fmaxq(change, fabsq(next[j] - beta[j]) / (1 + fabsq(beta[j])));
        beta = next;
        ll = evaluate(pr, beta, theta, &g, &h);
        if (change < Q(1e-30)) {
            f.converged = true;
            break;
        }
    }
    f.beta = beta;
    f.ll = ll;
    f.theta = theta;
    return f;
}
// Partial derivative of the NB log likelihood in log(theta) at fixed means.
Q theta_score(const Problem &pr, const std::vector<Q> &beta, Q theta) {
    Q s = 0;
    for (std::size_t i = 0; i < pr.n; ++i) {
        const Q a = pr.weights[i];
        if (a == 0)
            continue;
        Q eta = pr.offset[i];
        for (std::size_t j = 0; j < pr.p; ++j)
            eta += Q(pr.x[i * pr.p + j]) * beta[j];
        const Q mu = expq(eta), y = pr.y[i];
        s += a * (digammaq(y + theta) - digammaq(theta) + logq(theta / (theta + mu)) +
                  (mu - y) / (theta + mu));
    }
    return s * theta;
}
std::string text(Q v) {
    char buffer[64];
    quadmath_snprintf(buffer, sizeof buffer, "%.30Qe", v);
    return buffer;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: glm_reference problem_dir output.json");
        auto pr = load(argv[1]);
        std::vector<Q> start(pr.p, 0);
        Fit best;
        if (pr.family == "nb") {
            // Profile search in u = log(theta) over RapidGLM's documented range.
            Q lo = logq(Q(1e-6)), hi = logq(Q(1e8));
            auto profile = [&](Q u) {
                auto f = newton(pr, expq(u), start);
                start = f.beta;
                return std::make_pair(theta_score(pr, f.beta, expq(u)), f);
            };
            auto [slo, flo] = profile(lo);
            auto [shi, fhi] = profile(hi);
            if (shi > 0)
                best = fhi; // likelihood still increasing at the upper bound
            else if (slo < 0)
                best = flo;
            else {
                for (int it = 0; it < 200 && hi - lo > Q(1e-28); ++it) {
                    Q mid = (lo + hi) / 2;
                    auto [s, f] = profile(mid);
                    (s > 0 ? lo : hi) = mid;
                    best = f;
                }
            }
        } else
            best = newton(pr, 1, start);
        std::vector<Q> h;
        evaluate(pr, best.beta, best.theta, nullptr, &h, true);
        Q scale = 1, observations = 0;
        if (pr.family == "gaussian") {
            Q rss = 0;
            for (std::size_t i = 0; i < pr.n; ++i)
                if (pr.weights[i] > 0) {
                    Q eta = pr.offset[i];
                    for (std::size_t j = 0; j < pr.p; ++j)
                        eta += Q(pr.x[i * pr.p + j]) * best.beta[j];
                    rss += Q(pr.weights[i]) * (pr.y[i] - eta) * (pr.y[i] - eta);
                    observations += 1;
                }
            scale = rss / (observations - pr.p);
            best.ll = -observations / 2 * (logq(2 * M_PIq * rss / observations) + 1);
            for (std::size_t i = 0; i < pr.n; ++i)
                if (pr.weights[i] > 0)
                    best.ll += logq(Q(pr.weights[i])) / 2;
        }
        std::ofstream out(argv[2]);
        out << "{\"converged\": " << (best.converged ? "true" : "false")
            << ", \"iterations\": " << best.iterations << ", \"loglik\": " << text(best.ll)
            << ", \"theta\": " << (pr.family == "nb" ? text(best.theta) : "null")
            << ", \"coefficients\": [";
        for (std::size_t j = 0; j < pr.p; ++j)
            out << (j ? ", " : "") << text(best.beta[j]);
        out << "], \"standard_errors\": [";
        for (std::size_t j = 0; j < pr.p; ++j) {
            std::vector<Q> e(pr.p, 0), col;
            e[j] = 1;
            bool ok = solve(h, e, pr.p, col);
            out << (j ? ", " : "") << (ok ? text(sqrtq(col[j] * scale)) : std::string("null"));
        }
        out << "]}\n";
    } catch (const std::exception &e) {
        std::cerr << "glm_reference: " << e.what() << '\n';
        return 2;
    }
}
