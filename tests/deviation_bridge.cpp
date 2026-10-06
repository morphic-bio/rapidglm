// Validation-only access to public RapidGLM APIs; no comparator implementation.
#include <Rcpp.h>
#include <rapidglm/rapidglm.hpp>
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
namespace {
rapidglm::Matrix matrix(const Rcpp::NumericMatrix &x) {
    rapidglm::Matrix out(x.nrow(), x.ncol());
    for (std::size_t i = 0; i < out.rows; ++i)
        for (std::size_t j = 0; j < out.cols; ++j)
            out(i, j) = x(i, j);
    return out;
}
rapidglm::Vector vector(const Rcpp::NumericVector &x) {
    return Rcpp::as<rapidglm::Vector>(x);
}
} // namespace

// [[Rcpp::export]]
Rcpp::List audit_fits(Rcpp::NumericMatrix X, Rcpp::NumericMatrix Y, Rcpp::NumericVector offset,
                      Rcpp::NumericVector weights, Rcpp::NumericVector theta, int family = 3,
                      double tolerance = 1e-9, int threads = 1) {
    using namespace rapidglm;
    const auto x = matrix(X);
    require(family >= 0 && family <= 3 && X.ncol() > 0, "invalid model specification");
    require(Y.nrow() == X.nrow(), "response dimension mismatch");
    require(theta.size() == 0 || theta.size() == Y.ncol(), "theta dimension mismatch");
    FitOptions o;
    o.family = static_cast<Family>(family);
    o.offset = vector(offset);
    o.weights = vector(weights);
    o.tolerance = tolerance;
    o.max_iterations = 1000;
    o.max_dispersion_iterations = 200;
    o.retain_observations = false;
    // Workers use only C++ data and immutable raw response storage. Allocate
    // and populate all R objects on the calling thread after workers finish.
    const auto thetas = vector(theta);
    const double *responses = Y.begin();
    const std::size_t observations = Y.nrow(), features = Y.ncol();
    std::vector<FitResult> fits(features);
    parallel_for(features, threads, [&](std::size_t j) {
        auto options = o;
        if (!thetas.empty()) {
            options.dispersion = Dispersion::fixed;
            options.theta = thetas[j];
        }
        Vector response(responses + j * observations, responses + (j + 1) * observations);
        fits[j] = fit(x, response, options);
    });
    Rcpp::NumericMatrix beta(Y.ncol(), X.ncol()), se(Y.ncol(), X.ncol());
    Rcpp::NumericVector dispersion(Y.ncol()), ll(Y.ncol()), p(Y.ncol());
    Rcpp::CharacterVector status(Y.ncol());
    for (int j = 0; j < Y.ncol(); ++j) {
        const auto &f = fits[j];
        for (int k = 0; k < X.ncol(); ++k) {
            beta(j, k) = f.coefficients.at(k);
            se(j, k) = f.standard_errors.at(k);
        }
        dispersion[j] = f.theta;
        ll[j] = f.log_likelihood;
        p[j] = wald_test(f, X.ncol() - 1).p_value;
        status[j] = status_name(f.status);
    }
    return Rcpp::List::create(Rcpp::_["beta"] = beta, Rcpp::_["se"] = se,
                              Rcpp::_["theta"] = dispersion, Rcpp::_["loglik"] = ll,
                              Rcpp::_["p"] = p, Rcpp::_["status"] = status);
}

// [[Rcpp::export]]
int audit_workers(int threads) {
    rapidglm::require(threads > 0, "threads must be positive");
    int actual = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
#endif
    return actual;
}

// [[Rcpp::export]]
Rcpp::List audit_null_model(Rcpp::NumericVector y, Rcpp::NumericMatrix X, Rcpp::NumericMatrix G,
                            bool sceptre_model = true) {
    using namespace rapidglm;
    const auto x = matrix(X);
    require(G.nrow() == X.nrow(), "assignment dimension mismatch");
    FitOptions options;
    auto f = sceptre_model ? fit_sceptre_null(x, vector(y), options) : fit(x, vector(y), options);
    require(f.ok() && !f.has_boundary(), "invalid response null");
    ScoreWorkspace score(x, f);
    Rcpp::NumericVector statistics(G.ncol()), binary_statistics(G.ncol()), ratios(G.ncol());
    BinaryScoreWorkspace binary(score);
    for (int j = 0; j < G.ncol(); ++j) {
        Rcpp::NumericVector column = G(Rcpp::_, j);
        auto g = vector(column);
        statistics[j] = score.statistic(g);
        double observed = 0, expected = 0;
        std::vector<std::size_t> selected;
        for (std::size_t i = 0; i < g.size(); ++i) {
            require(g[i] == 0 || g[i] == 1, "assignments must be binary");
            if (g[i]) {
                selected.push_back(i);
                observed += y[i];
                expected += f.fitted[i];
            }
        }
        binary_statistics[j] = binary.statistic(selected);
        ratios[j] = observed / expected;
    }
    return Rcpp::List::create(Rcpp::_["beta"] = f.coefficients, Rcpp::_["mu"] = f.fitted,
                              Rcpp::_["theta"] = f.theta, Rcpp::_["loglik"] = f.log_likelihood,
                              Rcpp::_["statistic"] = statistics,
                              Rcpp::_["binary_statistic"] = binary_statistics,
                              Rcpp::_["ratio"] = ratios, Rcpp::_["status"] = status_name(f.status));
}

// Known and estimated propensity banks use the same Philox uniform streams.
// The empirical and staged estimates below use identical prefixes of each bank.
// [[Rcpp::export]]
Rcpp::DataFrame audit_tails(Rcpp::NumericVector y, Rcpp::NumericMatrix X, Rcpp::NumericVector g,
                            Rcpp::NumericVector truth_p, double seed, int test_id = 0,
                            int draws = 4999) {
    using namespace rapidglm;
    require(draws >= 499, "audit requires at least 499 draws");
    require(y.size() == X.nrow() && g.size() == X.nrow() &&
                (truth_p.size() == 0 || truth_p.size() == X.nrow()),
            "audit input dimension mismatch");
    auto x = matrix(X);
    FitOptions o;
    auto null = fit(x, vector(y), o);
    o.family = Family::binomial;
    auto propensity = fit(x, vector(g), o);
    require(null.ok() && !null.has_boundary() && propensity.ok(), "invalid audit fit");
    ScoreWorkspace score(x, null);
    Rcpp::CharacterVector law, method;
    Rcpp::NumericVector pvalue, statistic, discrepancy;
    Rcpp::IntegerVector count;
    Rcpp::LogicalVector approximate;
    for (int known = 0; known <= (truth_p.size() ? 1 : 0); ++known) {
        auto probabilities = known ? vector(truth_p) : propensity.fitted;
        ResampleOptions ro;
        ro.seed = static_cast<std::uint64_t>(seed);
        ro.test_id = test_id;
        ro.budgets = {static_cast<std::size_t>(draws)};
        ro.approximate = false;
        ro.retain_null = true;
        auto bank = resample(score, vector(g), crt_sampler(probabilities), ro);
        require(bank.valid && bank.null_statistics.size() == static_cast<std::size_t>(draws),
                "invalid score bank");
        bool stopped = false;
        double staged = rapidglm::nan, ks = rapidglm::nan;
        std::size_t staged_draws = draws;
        for (std::size_t budget : {std::size_t(499), static_cast<std::size_t>(draws)}) {
            Vector prefix(bank.null_statistics.begin(), bank.null_statistics.begin() + budget);
            double emp = empirical_tail(prefix, bank.statistic, Side::both);
            law.push_back(known ? "known" : "fitted");
            method.push_back(budget == 499 ? "empirical_499" : "empirical_full");
            pvalue.push_back(emp);
            count.push_back(budget);
            approximate.push_back(false);
            statistic.push_back(bank.statistic);
            discrepancy.push_back(NA_REAL);
            if (!stopped) {
                auto sf = fit_skew_normal(prefix);
                ks = sf.discrepancy;
                if (sf.valid) {
                    boost::math::skew_normal_distribution<double> sn(sf.location, sf.scale,
                                                                     sf.shape);
                    staged = std::min(1.0, 2 * std::min(boost::math::cdf(sn, bank.statistic),
                                                        boost::math::cdf(boost::math::complement(
                                                            sn, bank.statistic))));
                    staged_draws = budget;
                    stopped = true;
                } else
                    staged = emp;
            }
        }
        law.push_back(known ? "known" : "fitted");
        method.push_back("staged");
        pvalue.push_back(staged);
        count.push_back(staged_draws);
        approximate.push_back(stopped);
        statistic.push_back(bank.statistic);
        discrepancy.push_back(ks);
    }
    return Rcpp::DataFrame::create(
        Rcpp::_["law"] = law, Rcpp::_["method"] = method, Rcpp::_["p_value"] = pvalue,
        Rcpp::_["draws"] = count, Rcpp::_["approximate"] = approximate,
        Rcpp::_["statistic"] = statistic, Rcpp::_["discrepancy"] = discrepancy);
}
