// Batch benchmark: many responses sharing one design (the genomics use case).
// Reads a problem directory with X.f64 (n x p), Y.f64 (m responses x n,
// feature-major) and an optional offset.f64 (n), fits every response with
// fit_batch at the given thread count and reports wall time and coefficients.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
int main(int argc, char **argv) {
    try {
        require(argc == 8, "usage: batch dir n p m family threads output.tsv");
        const std::string dir = argv[1], family = argv[5];
        const std::size_t n = std::stoul(argv[2]), p = std::stoul(argv[3]), m = std::stoul(argv[4]);
        const int threads = std::stoi(argv[6]);
        Matrix x(n, p);
        Vector y(n * m);
        std::ifstream fx(dir + "/X.f64", std::ios::binary), fy(dir + "/Y.f64", std::ios::binary);
        fx.read(reinterpret_cast<char *>(x.values.data()), static_cast<std::streamsize>(n * p * 8));
        fy.read(reinterpret_cast<char *>(y.data()), static_cast<std::streamsize>(n * m * 8));
        require(fx && fy, "cannot read batch data");
        Vector offset;
        if (std::ifstream fo(dir + "/offset.f64", std::ios::binary); fo) {
            offset.resize(n);
            fo.read(reinterpret_cast<char *>(offset.data()), static_cast<std::streamsize>(n * 8));
            require(static_cast<bool>(fo), "cannot read offsets");
        }
        SparseCSC responses;
        responses.rows = n;
        responses.cols = m;
        responses.column_ptr = {0};
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t i = 0; i < n; ++i)
                if (y[j * n + i] != 0) {
                    responses.row_index.push_back(i);
                    responses.values.push_back(y[j * n + i]);
                }
            responses.column_ptr.push_back(responses.values.size());
        }
        FitOptions o;
        o.family = family == "poisson" ? Family::poisson : Family::negative_binomial;
        o.offset = offset;
        const auto start = std::chrono::steady_clock::now();
        auto fits = fit_batch(x, responses, o, threads);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream out(argv[7]);
        out << std::setprecision(17);
        std::size_t failed = 0;
        for (const auto &f : fits) {
            failed += !f.ok();
            for (std::size_t j = 0; j < p; ++j)
                out << (j ? "\t" : "") << f.coefficients[j];
            out << '\t' << f.theta << '\n';
        }
        std::cout << "{\"seconds\": " << seconds << ", \"threads\": " << threads
                  << ", \"failed\": " << failed << "}\n";
    } catch (const std::exception &e) {
        std::cerr << "batch: " << e.what() << '\n';
        return 2;
    }
}
