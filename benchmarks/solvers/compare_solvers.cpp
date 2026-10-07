// RapidGLM's negative binomial solvers against Ceres Solver (Apache-2.0) on the same
// problems, for docs/SOLVERS.md.
//
// fixed: coefficient fits at a fixed dispersion per gene with log size-factor
//   offsets (DESeq2's final GLM). RapidGLM: safeguarded Newton (observed
//   information, step halving). Ceres: Levenberg-Marquardt and dogleg trust
//   regions on deviance residuals (the sum of their squares is the deviance, so
//   the least-squares minimum is the MLE), and L-BFGS on the negative log
//   likelihood.
// joint: coefficients and log theta by maximum likelihood. RapidGLM: joint Newton
//   with step halving. Ceres: L-BFGS and BFGS on the negative log likelihood.
//
// Every gene sees identical data, offsets and dispersions (from bulk_de, not
// timed); every answer is scored by the same log likelihood. External solvers
// start from the intercept-only fit; RapidGLM uses its own start. RapidGLM is
// timed two ways: rapidglm_newton shares among the genes what a fit derives from
// the design alone (prepare_design: X'X, its rank check and the rows' nonzero
// entries, computed once and not timed), as fit_batch and bulk_de do;
// rapidglm_newton_plain calls fit() for each gene on its own. The last argument
// chooses one of them or both. Results: per
// method and tolerance setting, wall time, iterations, solver-reported
// convergence and the log-likelihood shortfall from the best answer per gene,
// separately for genes whose MLE is finite and for those RapidGLM reports at a
// coefficient boundary (diverging estimates, e.g. an all-zero group).
//
// Usage: rapidglm_solver_benchmark <dir with x.tsv and counts.mtx> <fixed|joint>
//        [threads] [max_genes (0: all)] [shared|plain|both (default)] > result.json
#include "io.hpp"
#include <rapidglm/rapidglm.hpp>

#include <boost/math/special_functions/digamma.hpp>
#include <ceres/ceres.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace rapidglm;

namespace {
struct Problem {
    const Matrix *x;
    Vector y, offset;
    double theta; // fixed problems
    // The design-only part of RapidGLM's fits (prepare_design, not timed), shared
    // by every gene as in fit_batch and bulk_de.
    std::shared_ptr<const detail::DesignSummary> design;
};

// lgamma(y + theta) - lgamma(theta) and digamma(y + theta) - digamma(theta). The
// plain differences lose precision at the large theta of near-Poisson genes;
// above 1e7 they come from y log theta + sum_k log1p(k / theta) and
// sum_k 1 / (theta + k), k < y, with k / theta small, by the power sums of k.
double lgamma_ratio(double y, double theta) {
    if (theta <= 1e7 || y == 0)
        return std::lgamma(y + theta) - std::lgamma(theta);
    const double s1 = y * (y - 1) / 2, s2 = (y - 1) * y * (2 * y - 1) / 6, s3 = s1 * s1;
    return y * std::log(theta) + s1 / theta - s2 / (2 * theta * theta) +
           s3 / (3 * theta * theta * theta);
}
double digamma_difference(double y, double theta) {
    if (theta <= 1e7 || y == 0)
        return boost::math::digamma(y + theta) - boost::math::digamma(theta);
    const double s1 = y * (y - 1) / 2, s2 = (y - 1) * y * (2 * y - 1) / 6, s3 = s1 * s1;
    return (y - s1 / theta + s2 / (theta * theta) - s3 / (theta * theta * theta)) / theta;
}
// One observation's log likelihood without -lgamma(y + 1).
double nb_term(double y, double mu, double theta) {
    return lgamma_ratio(y, theta) - theta * std::log1p(mu / theta) +
           (y > 0 ? y * std::log(mu / (theta + mu)) : 0);
}

double nb_loglik(const Problem &q, const double *beta, double theta) {
    const std::size_t n = q.x->rows, p = q.x->cols;
    double ll = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double eta = q.offset[i];
        for (std::size_t k = 0; k < p; ++k)
            eta += (*q.x)(i, k) * beta[k];
        const double mu = std::exp(eta), y = q.y[i];
        ll += nb_term(y, mu, theta) - std::lgamma(y + 1);
    }
    return ll;
}

// Deviance residuals r_i = sign(y - mu) sqrt(D_i) at fixed theta.
class DevianceResiduals : public ceres::CostFunction {
  public:
    explicit DevianceResiduals(const Problem &q) : q_(q) {
        set_num_residuals(static_cast<int>(q.x->rows));
        mutable_parameter_block_sizes()->push_back(static_cast<int>(q.x->cols));
    }
    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override {
        const std::size_t n = q_.x->rows, p = q_.x->cols;
        const double *beta = parameters[0], theta = q_.theta;
        for (std::size_t i = 0; i < n; ++i) {
            double eta = q_.offset[i];
            for (std::size_t k = 0; k < p; ++k)
                eta += (*q_.x)(i, k) * beta[k];
            if (!(std::abs(eta) < 700))
                return false;
            const double mu = std::exp(eta), y = q_.y[i];
            const double d = 2 * ((y > 0 ? y * std::log(y / mu) : 0) -
                                  (y + theta) * std::log((y + theta) / (mu + theta)));
            const double r = (y > mu ? 1 : -1) * std::sqrt(std::max(d, 0.0));
            residuals[i] = r;
            if (jacobians && jacobians[0]) {
                // dr/deta = theta (mu - y) / ((mu + theta) r); at r = 0 its limit
                // -sqrt(theta mu / (theta + mu)).
                const double g = std::abs(r) > 1e-8 ? theta * (mu - y) / ((mu + theta) * r)
                                                    : -std::sqrt(theta * mu / (theta + mu));
                for (std::size_t k = 0; k < p; ++k)
                    jacobians[0][i * p + k] = g * (*q_.x)(i, k);
            }
        }
        return true;
    }

  private:
    const Problem &q_;
};

// Negative log likelihood (without -sum lgamma(y + 1)) in beta, or in (beta, log
// theta) when `joint`.
class NegativeLoglik : public ceres::FirstOrderFunction {
  public:
    NegativeLoglik(const Problem &q, bool joint) : q_(q), joint_(joint) {}
    int NumParameters() const override {
        return static_cast<int>(q_.x->cols + (joint_ ? 1 : 0));
    }
    bool Evaluate(const double *parameters, double *cost, double *gradient) const override {
        const std::size_t n = q_.x->rows, p = q_.x->cols;
        if (joint_ && !(std::abs(parameters[p]) < 30))
            return false;
        const double theta = joint_ ? std::exp(parameters[p]) : q_.theta;
        double f = 0, gu = 0;
        if (gradient)
            std::fill(gradient, gradient + NumParameters(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double eta = q_.offset[i];
            for (std::size_t k = 0; k < p; ++k)
                eta += (*q_.x)(i, k) * parameters[k];
            if (!(std::abs(eta) < 700))
                return false;
            const double mu = std::exp(eta), y = q_.y[i];
            f -= nb_term(y, mu, theta);
            if (gradient) {
                const double deta = theta * (y - mu) / (theta + mu);
                for (std::size_t k = 0; k < p; ++k)
                    gradient[k] -= deta * (*q_.x)(i, k);
                if (joint_)
                    gu += digamma_difference(y, theta) - std::log1p(mu / theta) +
                          (mu - y) / (theta + mu);
            }
        }
        *cost = f;
        if (gradient && joint_)
            gradient[p] = -theta * gu;
        return true;
    }

  private:
    const Problem &q_;
    bool joint_;
};

struct Outcome {
    Vector beta;
    double theta = rapidglm::nan, loglik = rapidglm::nan, seconds = 0;
    std::size_t iterations = 0;
    bool converged = false;
};

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

Vector intercept_start(const Problem &q) {
    Vector beta(q.x->cols, 0.0);
    double sy = 0, so = 0;
    for (std::size_t i = 0; i < q.y.size(); ++i) {
        sy += q.y[i];
        so += std::exp(q.offset[i]);
    }
    beta[0] = std::log(std::max(sy, 0.5) / so);
    return beta;
}

struct Tolerances {
    const char *name;
    double function, gradient, parameter;
};
const Tolerances defaults{"ceres_default", 1e-6, 1e-10, 1e-8};
const Tolerances tight{"tight", 1e-12, 1e-12, 1e-10};

Outcome rapidglm_fit(const Problem &q, bool joint, bool shared) {
    Outcome out;
    const auto t = Clock::now();
    FitOptions fo;
    fo.family = Family::negative_binomial;
    fo.dispersion = joint ? Dispersion::ml : Dispersion::fixed;
    fo.theta = joint ? 10 : q.theta;
    fo.offset = q.offset;
    fo.max_iterations = 1000;
    if (shared)
        fo.design = q.design;
    const auto f = fit(*q.x, q.y, fo);
    out.seconds = since(t);
    out.converged = f.ok();
    out.iterations = f.iterations;
    if (f.coefficients.size() == q.x->cols) {
        out.beta = f.coefficients;
        out.theta = joint ? f.theta : q.theta;
    }
    return out;
}

Outcome trust_region(const Problem &q, ceres::TrustRegionStrategyType strategy,
                     const Tolerances &tol) {
    Outcome out;
    const auto t = Clock::now();
    out.beta = intercept_start(q);
    ceres::Problem problem;
    problem.AddResidualBlock(new DevianceResiduals(q), nullptr, out.beta.data());
    ceres::Solver::Options o;
    o.trust_region_strategy_type = strategy;
    o.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    o.max_num_iterations = 1000;
    o.function_tolerance = tol.function;
    o.gradient_tolerance = tol.gradient;
    o.parameter_tolerance = tol.parameter;
    o.logging_type = ceres::SILENT;
    o.num_threads = 1;
    ceres::Solver::Summary s;
    ceres::Solve(o, &problem, &s);
    out.seconds = since(t);
    out.converged = s.termination_type == ceres::CONVERGENCE;
    out.iterations = s.iterations.size();
    out.theta = q.theta;
    return out;
}

Outcome line_search(const Problem &q, bool joint, ceres::LineSearchDirectionType direction,
                    const Tolerances &tol) {
    Outcome out;
    const auto t = Clock::now();
    Vector x = intercept_start(q);
    if (joint)
        x.push_back(std::log(10.0));
    ceres::GradientProblem problem(new NegativeLoglik(q, joint));
    ceres::GradientProblemSolver::Options o;
    o.line_search_direction_type = direction;
    o.max_num_iterations = 1000;
    o.function_tolerance = tol.function;
    o.gradient_tolerance = tol.gradient;
    o.parameter_tolerance = tol.parameter;
    o.logging_type = ceres::SILENT;
    ceres::GradientProblemSolver::Summary s;
    ceres::Solve(o, problem, x.data(), &s);
    out.seconds = since(t);
    out.converged = s.termination_type == ceres::CONVERGENCE;
    out.iterations = s.iterations.size();
    out.theta = joint ? std::exp(x.back()) : q.theta;
    x.resize(q.x->cols);
    out.beta = x;
    return out;
}

// A JSON number, or null where none is defined (an empty gene subset).
std::string json_number(double v) {
    if (!std::isfinite(v))
        return "null";
    std::ostringstream out;
    out.precision(10);
    out << v;
    return out.str();
}
double sample_quantile(Vector v, double p) {
    if (v.empty())
        return rapidglm::nan;
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(p * (v.size() - 1))];
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: rapidglm_solver_benchmark <dir> <fixed|joint> [threads] [max_genes] "
                     "[shared|plain|both]\n";
        return 2;
    }
    const std::string dir = argv[1], mode = argv[2];
    const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
    const std::size_t max_genes = argc > 4 ? std::stoul(argv[4]) : 0;
    const std::string design = argc > 5 ? argv[5] : "both";
    if (design != "shared" && design != "plain" && design != "both") {
        std::cerr << "the last argument must be shared, plain or both\n";
        return 2;
    }
    const bool joint = mode == "joint";
    const Matrix x = cli::read_matrix(dir + "/x.tsv");
    auto counts = cli::read_market(dir + "/counts.mtx"); // features x samples (CLI layout)
    // The CLI reads feature-by-observation files into observations x features.
    const auto t_setup = Clock::now();
    BulkOptions bo;
    bo.coefficient = x.cols - 1;
    bo.threads = threads;
    // Median-of-ratios size factors need a gene without zeros; otherwise DESeq2's
    // documented alternative, poscounts.
    BulkResult bulk;
    try {
        bulk = bulk_de(x, counts, bo);
    } catch (const std::invalid_argument &) {
        bo.size_factor_method = SizeFactorMethod::poscounts;
        bulk = bulk_de(x, counts, bo);
    }
    std::vector<Problem> problems;
    const auto summary = prepare_design(x);
    Vector offset(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i)
        offset[i] = std::log(bulk.size_factors[i]);
    for (std::size_t j = 0; j < counts.cols && (!max_genes || problems.size() < max_genes); ++j) {
        const auto &g = bulk.genes[j];
        if (g.status == "all_zero" || !(g.dispersion > 0))
            continue;
        problems.push_back({&x, counts.column(j), offset, 1 / g.dispersion, summary});
    }
    const double setup_seconds = since(t_setup);
    struct Method {
        std::string name;
        std::function<Outcome(const Problem &)> run;
    };
    std::vector<Method> methods;
    if (design != "plain")
        methods.push_back(
            {"rapidglm_newton", [&](const Problem &q) { return rapidglm_fit(q, joint, true); }});
    if (design != "shared")
        methods.push_back({"rapidglm_newton_plain",
                           [&](const Problem &q) { return rapidglm_fit(q, joint, false); }});
    for (const auto *tol : {&defaults, &tight}) {
        const std::string suffix = std::string("/") + tol->name;
        if (!joint) {
            methods.push_back({"ceres_levenberg_marquardt" + suffix, [tol](const Problem &q) {
                                   return trust_region(q, ceres::LEVENBERG_MARQUARDT, *tol);
                               }});
            methods.push_back({"ceres_dogleg" + suffix, [tol](const Problem &q) {
                                   return trust_region(q, ceres::DOGLEG, *tol);
                               }});
        } else
            methods.push_back({"ceres_bfgs" + suffix, [tol](const Problem &q) {
                                   return line_search(q, true, ceres::BFGS, *tol);
                               }});
        methods.push_back({"ceres_lbfgs" + suffix, [tol, joint](const Problem &q) {
                               return line_search(q, joint, ceres::LBFGS, *tol);
                           }});
    }
    const std::size_t m = problems.size();
    std::vector<std::vector<Outcome>> outcomes(methods.size(), std::vector<Outcome>(m));
    std::vector<double> wall(methods.size());
    for (std::size_t k = 0; k < methods.size(); ++k) {
        const auto t = Clock::now();
        parallel_for(m, threads, [&](auto j) {
            auto o = methods[k].run(problems[j]);
            if (o.beta.size() == x.cols && std::isfinite(o.theta))
                o.loglik = nb_loglik(problems[j], o.beta.data(), o.theta);
            outcomes[k][j] = std::move(o);
        });
        wall[k] = since(t);
        std::cerr << methods[k].name << ": " << wall[k] << " s\n";
    }
    // Boundary genes: RapidGLM reports a coefficient at a boundary (diverging MLE).
    std::vector<unsigned char> boundary(m, 0);
    parallel_for(m, threads, [&](auto j) {
        FitOptions fo;
        fo.family = Family::negative_binomial;
        fo.dispersion = joint ? Dispersion::ml : Dispersion::fixed;
        fo.theta = joint ? 10 : problems[j].theta;
        fo.offset = problems[j].offset;
        fo.max_iterations = 1000;
        boundary[j] = fit(x, problems[j].y, fo).has_boundary();
    });
    Vector best(m, -std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k < methods.size(); ++k)
        for (std::size_t j = 0; j < m; ++j)
            if (std::isfinite(outcomes[k][j].loglik))
                best[j] = std::max(best[j], outcomes[k][j].loglik);
    // Diagnostics: the genes where RapidGLM falls furthest short (stderr).
    if (std::getenv("RAPIDGLM_SOLVER_WORST")) {
        std::vector<std::size_t> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return best[a] - outcomes[0][a].loglik > best[b] - outcomes[0][b].loglik;
        });
        std::size_t shown[2] = {0, 0};
        for (std::size_t r = 0; r < m; ++r) {
            const auto j = order[r];
            if (shown[boundary[j]]++ >= 3)
                continue;
            std::cerr << "gene " << j << " boundary " << int(boundary[j]) << " y";
            for (double v : problems[j].y)
                std::cerr << ' ' << v;
            std::cerr << '\n';
            for (std::size_t k = 0; k < methods.size(); ++k)
                std::cerr << "  " << methods[k].name << " loglik " << outcomes[k][j].loglik
                          << " theta " << outcomes[k][j].theta << " converged "
                          << outcomes[k][j].converged << " iterations " << outcomes[k][j].iterations
                          << '\n';
        }
    }
    std::cout << "{\n  \"dataset\": \"" << dir << "\",\n  \"mode\": \"" << mode
              << "\",\n  \"threads\": " << threads << ",\n  \"observations\": " << x.rows
              << ",\n  \"columns\": " << x.cols << ",\n  \"genes\": " << m
              << ",\n  \"boundary_genes\": " << std::accumulate(boundary.begin(), boundary.end(), 0)
              << ",\n  \"setup_seconds\": " << setup_seconds << ",\n  \"methods\": [\n";
    for (std::size_t k = 0; k < methods.size(); ++k) {
        std::cout << "    {\"name\": \"" << methods[k].name << "\", \"wall_seconds\": " << wall[k];
        for (int b = 0; b < 2; ++b) {
            Vector shortfall, iterations, seconds;
            std::size_t converged = 0, failed = 0, worse = 0;
            for (std::size_t j = 0; j < m; ++j) {
                if (boundary[j] != b)
                    continue;
                const auto &o = outcomes[k][j];
                seconds.push_back(o.seconds);
                iterations.push_back(static_cast<double>(o.iterations));
                converged += o.converged;
                if (!std::isfinite(o.loglik)) {
                    ++failed;
                    continue;
                }
                const double s = best[j] - o.loglik;
                shortfall.push_back(s);
                worse += s > 1e-6 * (1 + std::abs(best[j]));
            }
            std::cout << ",\n      \"" << (b ? "boundary" : "finite")
                      << "\": {\"genes\": " << seconds.size() << ", \"cpu_seconds\": "
                      << std::accumulate(seconds.begin(), seconds.end(), 0.0)
                      << ", \"converged\": " << converged << ", \"failed\": " << failed
                      << ", \"shortfall_over_1e-6_relative\": " << worse
                      << ", \"shortfall_median\": " << json_number(sample_quantile(shortfall, 0.5))
                      << ", \"shortfall_99\": " << json_number(sample_quantile(shortfall, 0.99))
                      << ", \"shortfall_max\": " << json_number(sample_quantile(shortfall, 1.0))
                      << ", \"iterations_median\": "
                      << json_number(sample_quantile(iterations, 0.5))
                      << ", \"iterations_99\": " << json_number(sample_quantile(iterations, 0.99))
                      << "}";
        }
        std::cout << "}" << (k + 1 < methods.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n}\n";
}
