// Python entry point for the RapidGLM fitter (pybind11). The Python layer
// (rapidglm/statsmodels.py) builds statsmodels-compatible results from it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rapidglm/model.hpp>

namespace py = pybind11;
using namespace rapidglm;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {
Vector vector_of(const Array &a) {
    return Vector(a.data(), a.data() + a.size());
}
py::array_t<double> array_of(const Vector &v) {
    return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}
} // namespace

PYBIND11_MODULE(_rapidglm, m) {
    m.doc() = "RapidGLM engine bindings";
    m.def(
        "fit",
        [](Array x, Array y, Array weights, Array offset, Array start, const std::string &family,
           double theta, bool estimate_theta, double tolerance, std::size_t max_iterations,
           bool singular_ok, double alias_tolerance, const std::string &division) {
            const vmath::DivisionScope scope(division.empty() ? DivisionMode::inherit
                                                              : vmath::parse_division(division));
            if (x.ndim() != 2)
                throw std::invalid_argument("exog must be two-dimensional");
            Matrix design(static_cast<std::size_t>(x.shape(0)),
                          static_cast<std::size_t>(x.shape(1)));
            design.values = vector_of(x); // C order is RapidGLM's row-major layout
            FitOptions o;
            o.family = family == "gaussian"   ? Family::gaussian
                       : family == "poisson"  ? Family::poisson
                       : family == "binomial" ? Family::binomial
                                              : Family::negative_binomial;
            if (o.family == Family::negative_binomial) {
                o.dispersion = estimate_theta ? Dispersion::ml : Dispersion::fixed;
                o.theta = theta;
            }
            o.weights = vector_of(weights);
            o.offset = vector_of(offset);
            o.start = vector_of(start);
            o.tolerance = tolerance;
            o.max_iterations = max_iterations;
            o.singular_ok = singular_ok;
            o.alias_tolerance = alias_tolerance;
            FitResult r;
            {
                py::gil_scoped_release release;
                r = rapidglm::fit(design, vector_of(y), o);
            }
            const std::size_t p = design.cols;
            py::array_t<double> covariance({p, p});
            auto c = covariance.mutable_unchecked<2>();
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = 0; k < p; ++k)
                    c(j, k) = r.covariance.rows == p ? r.covariance(j, k) : rapidglm::nan;
            std::vector<bool> aliased(p, false);
            for (std::size_t j = 0; j < p && j < r.aliased.size(); ++j)
                aliased[j] = r.aliased[j] != 0;
            py::dict out;
            out["division"] = vmath::division_name();
            out["status"] = status_name(r.status);
            out["converged"] = r.ok();
            out["iterations"] = r.iterations;
            out["coefficients"] = array_of(r.coefficients);
            out["standard_errors"] = array_of(r.standard_errors);
            out["covariance"] = covariance; // includes the dispersion scale
            out["scale"] = r.scale;
            out["aliased"] = aliased;
            out["rank"] = r.rank;
            out["theta"] = r.theta;
            out["theta_boundary"] = r.theta_boundary;
            out["loglik"] = r.log_likelihood;
            out["boundary"] = r.has_boundary();
            out["fitted"] = array_of(r.fitted);
            out["working_weights"] = array_of(r.working_weights);
            out["used_qr"] = r.used_qr;
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("weights"), py::arg("offset"), py::arg("start"),
        py::arg("family"), py::arg("theta") = 10.0, py::arg("estimate_theta") = false,
        py::arg("tolerance") = 1e-9, py::arg("max_iterations") = 100, py::arg("singular_ok") = true,
        py::arg("alias_tolerance") = 1e-11, py::arg("division") = "");
}
