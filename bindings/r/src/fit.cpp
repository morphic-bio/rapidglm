// R entry point for the RapidGLM fitter. The R layer (R/glm_fit.R) supplies
// family handling and assembles a stats::glm.fit-compatible result.
#include <cpp11.hpp>
#include <rapidglm/model.hpp>

using namespace rapidglm;

[[cpp11::register]] cpp11::writable::list
rapidglm_fit_(cpp11::doubles_matrix<> x, cpp11::doubles y, cpp11::doubles weights,
              cpp11::doubles offset, cpp11::doubles start, std::string family, double theta,
              bool estimate_theta, double tolerance, int max_iterations, bool singular_ok,
              double alias_tolerance) {
    const std::size_t n = x.nrow(), p = x.ncol();
    Matrix design(n, p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < n; ++i)
            design(i, j) = x(i, j);
    FitOptions o;
    o.family = family == "gaussian"   ? Family::gaussian
               : family == "poisson"  ? Family::poisson
               : family == "binomial" ? Family::binomial
                                      : Family::negative_binomial;
    if (o.family == Family::negative_binomial) {
        o.dispersion = estimate_theta ? Dispersion::ml : Dispersion::fixed;
        o.theta = theta;
    }
    o.weights.assign(weights.begin(), weights.end());
    o.offset.assign(offset.begin(), offset.end());
    o.start.assign(start.begin(), start.end());
    o.tolerance = tolerance;
    o.max_iterations = static_cast<std::size_t>(max_iterations);
    o.singular_ok = singular_ok;
    o.alias_tolerance = alias_tolerance;
    Vector response(y.begin(), y.end());
    const FitResult r = fit(design, response, o);

    cpp11::writable::doubles coefficients(static_cast<R_xlen_t>(p)), eta(static_cast<R_xlen_t>(n)),
        mu(static_cast<R_xlen_t>(n)), working(static_cast<R_xlen_t>(n));
    cpp11::writable::logicals aliased(static_cast<R_xlen_t>(p));
    for (std::size_t j = 0; j < p; ++j) {
        coefficients[j] = r.coefficients[j];
        aliased[j] = r.aliased.size() == p && r.aliased[j] ? TRUE : FALSE;
    }
    // The R layer builds glm.fit's qr and effects from these with base R's QR.
    const bool have_state = r.working_weights.size() == n && r.fitted.size() == n;
    for (std::size_t i = 0; i < n; ++i) {
        double e = o.offset.empty() ? 0 : o.offset[i];
        for (std::size_t j = 0; j < p; ++j)
            if (have_state && !(r.aliased.size() == p && r.aliased[j]))
                e += design(i, j) * r.coefficients[j];
        eta[i] = have_state ? e : NA_REAL;
        mu[i] = have_state ? r.fitted[i] : NA_REAL;
        working[i] = have_state ? r.working_weights[i] : NA_REAL;
    }
    using namespace cpp11::literals;
    return cpp11::writable::list(
        {"status"_nm = status_name(r.status), "converged"_nm = r.ok(),
         "iterations"_nm = static_cast<int>(r.iterations), "coefficients"_nm = coefficients,
         "aliased"_nm = aliased, "rank"_nm = static_cast<int>(r.rank), "theta"_nm = r.theta,
         "theta_boundary"_nm = r.theta_boundary, "loglik"_nm = r.log_likelihood,
         "boundary"_nm = r.has_boundary(), "eta"_nm = eta, "mu"_nm = mu,
         "working_weights"_nm = working, "used_qr"_nm = r.used_qr});
}
