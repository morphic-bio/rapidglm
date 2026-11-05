// R entry point for the special functions of the RapidGLM C interface
// (rapidglm_<f>_array; include/rapidglm/vmath.hpp). As base R's math functions:
// a numeric argument (integers converted), its attributes kept, a NaN argument
// (NA among them) returned as it is, and one "NaNs produced" warning when other
// arguments give NaN. The values come in blocks that stay in cache for that check.
#include <cpp11.hpp>
#include <rapidglm/rapidglm_c.h>

#include <algorithm>
#include <cmath>
#include <string>

[[cpp11::register]] SEXP rapidglm_special_(SEXP x, std::string name) {
    using Function = int (*)(const double *, double *, size_t);
    const Function f = name == "log"        ? rapidglm_log_array
                       : name == "log1p"    ? rapidglm_log1p_array
                       : name == "exp"      ? rapidglm_exp_array
                       : name == "lgamma"   ? rapidglm_lgamma_array
                       : name == "digamma"  ? rapidglm_digamma_array
                       : name == "trigamma" ? rapidglm_trigamma_array
                                            : nullptr;
    if (!f)
        cpp11::stop("unknown special function '%s'", name.c_str());
    const int type = TYPEOF(x);
    if (!(type == REALSXP || (type == INTSXP && !Rf_inherits(x, "factor"))))
        cpp11::stop("x must be numeric");
    const cpp11::sexp in = type == REALSXP ? x : Rf_coerceVector(x, REALSXP);
    const R_xlen_t n = Rf_xlength(in);
    const cpp11::sexp out = Rf_allocVector(REALSXP, n);
    SHALLOW_DUPLICATE_ATTRIB(out, x);
    const double *a = REAL(in);
    double *y = REAL(out);
    bool produced = false;
    constexpr R_xlen_t block = 2048;
    for (R_xlen_t i = 0; i < n; i += block) {
        const R_xlen_t m = std::min(block, n - i);
        if (f(a + i, y + i, static_cast<size_t>(m)) != RAPIDGLM_OK)
            cpp11::stop("%s", rapidglm_last_error());
        for (R_xlen_t j = i; j < i + m; ++j)
            if (std::isnan(y[j])) {
                if (std::isnan(a[j]))
                    y[j] = a[j];
                else
                    produced = true;
            }
    }
    if (produced)
        cpp11::warning("NaNs produced");
    return out;
}

[[cpp11::register]] int rapidglm_division_set_(int mode) {
    int previous;
    if (rapidglm_set_division(mode, &previous) != RAPIDGLM_OK)
        cpp11::stop("%s", rapidglm_last_error());
    return previous;
}

[[cpp11::register]] std::string rapidglm_special_path_() {
    return rapidglm_special_path();
}
