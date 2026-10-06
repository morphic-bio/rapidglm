// Fits one comparison problem with the RapidGLM engine and reports timing.
// Data are loaded before timing; each repeat is a complete fit.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <rapidglm/rapidglm.hpp>
#include <sstream>
using namespace rapidglm;
namespace {
Vector read(const std::string &path, std::size_t count) {
    Vector v(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(count * 8));
    require(static_cast<bool>(f), "cannot read problem data");
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
void number(std::ostream &out, double v) {
    if (std::isfinite(v))
        out << std::setprecision(17) << v;
    else
        out << "null";
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 4 || argc == 5,
                "usage: glm_compare problem_dir output.json max_seconds [singular_ok|qr,...]");
        const std::string flags = argc == 5 ? argv[4] : "";
        const std::string dir = argv[1];
        std::ifstream m(dir + "/meta.json");
        std::stringstream s;
        s << m.rdbuf();
        const auto family = field(s.str(), "family");
        const std::size_t n = std::stoul(field(s.str(), "n")), p = std::stoul(field(s.str(), "p"));
        Matrix x(n, p);
        x.values = read(dir + "/X.f64", n * p);
        const auto y = read(dir + "/y.f64", n);
        FitOptions o;
        o.family = family == "gaussian"   ? Family::gaussian
                   : family == "poisson"  ? Family::poisson
                   : family == "binomial" ? Family::binomial
                                          : Family::negative_binomial;
        o.offset = read(dir + "/offset.f64", n);
        o.weights = read(dir + "/weights.f64", n);
        o.retain_observations = false;
        o.singular_ok = flags.find("singular_ok") != std::string::npos;
        if (flags.find("qr") != std::string::npos)
            o.solver = Solver::qr;
        const double budget = std::stod(argv[3]);
        std::vector<double> seconds;
        FitResult f;
        do {
            const auto start = std::chrono::steady_clock::now();
            f = fit(x, y, o);
            seconds.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        } while (seconds.size() < 7 &&
                 std::accumulate(seconds.begin(), seconds.end(), 0.0) < budget);
        std::ofstream out(argv[2]);
        out << "{\"tool\": \"rapidglm\", \"flags\": \"" << flags << "\", \"rank\": " << f.rank
            << ", \"used_qr\": " << (f.used_qr ? "true" : "false") << ", \"status\": \""
            << status_name(f.status) << "\", \"converged\": " << (f.ok() ? "true" : "false")
            << ", \"iterations\": " << f.iterations << ", \"loglik\": ";
        number(out, f.log_likelihood);
        out << ", \"theta\": ";
        number(out, f.theta);
        out << ", \"coefficients\": [";
        for (std::size_t j = 0; j < p; ++j) {
            out << (j ? ", " : "");
            number(out, f.coefficients[j]);
        }
        out << "], \"standard_errors\": [";
        for (std::size_t j = 0; j < p; ++j) {
            out << (j ? ", " : "");
            number(out, f.standard_errors[j]);
        }
        out << "], \"seconds\": [";
        for (std::size_t r = 0; r < seconds.size(); ++r)
            out << (r ? ", " : "") << seconds[r];
        out << "]}\n";
    } catch (const std::exception &e) {
        std::cerr << "glm_compare: " << e.what() << '\n';
        return 2;
    }
}
