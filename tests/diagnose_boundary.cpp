// Diagnostic executable: observe the production fitter through deterministic
// iteration-budget replays. No copied or substituted solver.
#include "../src/io.hpp"
#include <iostream>

using namespace rapidglm;
namespace {
Vector read_binary(const std::string &path, std::size_t count) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "diagnostic input requires IEEE binary64");
    std::ifstream in(path, std::ios::binary);
    Vector values(count);
    for (auto &v : values) {
        unsigned char bytes[8];
        in.read(reinterpret_cast<char *>(bytes), 8);
        require(in.good(), "truncated diagnostic binary input");
        std::uint64_t bits = 0;
        for (unsigned j = 0; j < 8; ++j)
            bits |= static_cast<std::uint64_t>(bytes[j]) << (8 * j);
        std::memcpy(&v, &bits, 8);
    }
    require(in.peek() == std::char_traits<char>::eof(), "extra diagnostic binary input");
    return values;
}
void echo_binary(std::ostream &out, const Vector &values) {
    for (double v : values) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, 8);
        for (unsigned j = 0; j < 8; ++j)
            out.put(static_cast<char>((bits >> (8 * j)) & 255));
    }
}
void header(std::ostream &out, std::size_t p, std::size_t n) {
    out << "model\tstart\tmaxit\titerations\tstatus\ttheta\tlog_likelihood\tdeviance"
           "\tboundary_beta\tboundary_se\tboundary_mu\tmin_eta\tclipped_means\ttheta_boundary";
    for (std::size_t j = 0; j < p; ++j)
        out << "\tbeta" << j << "\tse" << j << "\tboundary" << j << "\twald_p" << j;
    for (std::size_t i = 0; i < n; ++i)
        out << "\tmu" << i;
    out << '\n';
}
void row(std::ostream &out, const std::string &model, const std::string &start, std::size_t maxit,
         const Matrix &x, const Vector &y, const FitOptions &o, const FitResult &f,
         std::size_t boundary) {
    double deviance = 0, boundary_mu = 0;
    auto mu = f.fitted;
    mu.resize(y.size(), rapidglm::nan);
    auto eta = detail::matvec(x, f.coefficients);
    std::size_t clipped = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        eta[i] += o.offset[i];
        clipped += eta[i] < -700;
        if (x(i, boundary) != 0)
            boundary_mu += mu[i];
        double term = y[i] == 0 ? 0 : y[i] * std::log(y[i] / mu[i]);
        if (model == "poisson")
            term += mu[i] - y[i];
        else if (std::isfinite(f.theta))
            term -= (y[i] + f.theta) * std::log1p((y[i] - mu[i]) / (f.theta + mu[i]));
        else
            term = rapidglm::nan; // Failed NB pilot has not estimated theta.
        deviance += 2 * o.weights[i] * term;
    }
    out << model << '\t' << start << '\t' << maxit << '\t' << f.iterations << '\t'
        << status_name(f.status);
    for (double v :
         {f.theta, f.log_likelihood, deviance, f.coefficients[boundary],
          f.standard_errors[boundary], boundary_mu, *std::min_element(eta.begin(), eta.end())}) {
        out << '\t';
        cli::value(out, v);
    }
    out << '\t' << clipped << '\t' << f.theta_boundary;
    for (std::size_t j = 0; j < x.cols; ++j) {
        for (double v :
             {f.coefficients[j], f.standard_errors[j],
              static_cast<double>(f.boundary_coefficients[j]), wald_test(f, j).p_value}) {
            out << '\t';
            cli::value(out, v);
        }
    }
    for (double v : mu) {
        out << '\t';
        cli::value(out, v);
    }
    out << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: rapidglm_boundary_diagnostic manifest.json");
        cli::Tree manifest;
        boost::property_tree::read_json(argv[1], manifest);
        for (const auto &entry : manifest.get_child("cases")) {
            const auto &c = entry.second;
            Matrix x(c.get<std::size_t>("observations"), c.get<std::size_t>("parameters"));
            auto binary = [&](const std::string &name, std::size_t count) {
                return read_binary(c.get<std::string>("binary_inputs." + name), count);
            };
            const auto column_major = binary("design", x.rows * x.cols);
            for (std::size_t i = 0; i < x.rows; ++i)
                for (std::size_t j = 0; j < x.cols; ++j)
                    x(i, j) = column_major[j * x.rows + i];
            auto y = binary("response", x.rows);
            auto start = binary("start", x.cols);
            FitOptions o;
            o.offset = binary("offset", x.rows);
            o.weights = binary("weights", x.rows);
            std::ofstream echo(c.get<std::string>("directory") + "/native_inputs.f64le",
                               std::ios::binary);
            for (const auto *v : {&x.values, &y, &o.offset, &o.weights, &start})
                echo_binary(echo, *v);
            require(echo.good(), "input echo write failed");
            o.tolerance = manifest.get<double>("tolerance");
            const auto boundary = c.get<std::size_t>("boundary_column");
            std::ofstream prefixes(c.get<std::string>("directory") + "/native_prefix.tsv"),
                summaries(c.get<std::string>("directory") + "/native_runs.tsv");
            header(prefixes, x.cols, x.rows);
            header(summaries, x.cols, x.rows);
            for (const std::string model : {"poisson", "nb_fixed", "nb_ml"}) {
                o.family = model == "poisson" ? Family::poisson : Family::negative_binomial;
                o.dispersion = model == "nb_fixed" ? Dispersion::fixed : Dispersion::ml;
                o.theta = manifest.get<double>("fixed_theta");
                for (const std::string initialization : {"default", "shared"}) {
                    o.start = initialization == "shared" ? start : Vector{};
                    for (std::size_t maxit : {100u, 1000u}) {
                        o.max_iterations = maxit;
                        const auto f = fit(x, y, o);
                        row(summaries, model, initialization, maxit, x, y, o, f, boundary);
                    }
                }
                if (model == "nb_ml")
                    continue; // Isolate coefficient iterations with theta held fixed.
                o.start = start;
                for (const auto &budget : manifest.get_child("prefix_budgets")) {
                    o.max_iterations = budget.second.get_value<std::size_t>();
                    const auto f = fit(x, y, o);
                    row(prefixes, model, "shared", o.max_iterations, x, y, o, f, boundary);
                }
            }
            require(prefixes.good() && summaries.good(), "diagnostic output write failed");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
