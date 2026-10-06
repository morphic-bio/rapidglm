// Validation-only null calibration of known-propensity CRT at sparse assignment
// rates, where the thinned sampler uses geometric gaps. Each dataset is tested
// with both CRT draw methods on identical data.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
int main(int argc, char **argv) {
    try {
        require(argc == 5 || argc == 6,
                "usage: calibration_sparse output.tsv datasets draws threads [offset]");
        const std::size_t datasets = std::stoul(argv[2]), draws = std::stoul(argv[3]);
        // A nonzero offset keys an independent extension (data seeds and RNG test IDs).
        const std::uint64_t offset = argc == 6 ? std::stoull(argv[5]) : 0;
        const int threads = std::stoi(argv[4]);
        const std::size_t n = 2000;
        struct Row {
            double p[2] = {rapidglm::nan, rapidglm::nan};
            double treated = 0;
        };
        std::vector<Row> rows(4 * datasets);
        parallel_for(rows.size(), threads, [&](std::size_t index) {
            const auto family = static_cast<Family>(index / datasets);
            std::mt19937_64 rng(202609290000ULL + offset + index);
            std::normal_distribution<double> normal;
            Matrix x(n, 2, 1);
            Vector y(n), g(n), probability(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double z = normal(rng), eta = 0.5 + 0.4 * z;
                x(i, 1) = z;
                // Mean assignment rate about 2%, varying with the covariate.
                probability[i] = 1 / (1 + std::exp(4 - 0.5 * z));
                g[i] = std::bernoulli_distribution(probability[i])(rng);
                if (family == Family::gaussian)
                    y[i] = eta + normal(rng);
                else if (family == Family::binomial)
                    y[i] = std::bernoulli_distribution(1 / (1 + std::exp(-eta)))(rng);
                else if (family == Family::poisson)
                    y[i] = std::poisson_distribution<int>(std::exp(eta))(rng);
                else
                    y[i] = std::poisson_distribution<int>(
                        std::gamma_distribution<double>(2, std::exp(eta) / 2)(rng))(rng);
            }
            auto &row = rows[index];
            row.treated = std::accumulate(g.begin(), g.end(), 0.0);
            FitOptions o;
            o.family = family;
            const auto null = fit(x, y, o);
            if (!null.ok() || row.treated == 0)
                return;
            const ScoreWorkspace score(x, null);
            ResampleOptions r;
            r.seed = 20260929;
            r.test_id = offset + index;
            r.budgets = {draws};
            r.approximate = false;
            for (int m = 0; m < 2; ++m) {
                const auto method = m == 0 ? CrtDraw::thinned : CrtDraw::direct;
                const auto result = resample(score, g, crt_sampler(probability, method), r);
                if (result.valid)
                    row.p[m] = result.p_value;
            }
        });
        std::ofstream out(argv[1]);
        out << "family\tdataset\ttreated\tp_thinned\tp_direct\n" << std::setprecision(17);
        for (std::size_t i = 0; i < rows.size(); ++i)
            out << i / datasets << '\t' << i % datasets << '\t' << rows[i].treated << '\t'
                << rows[i].p[0] << '\t' << rows[i].p[1] << '\n';
        require(static_cast<bool>(out), "failed writing calibration output");
    } catch (const std::exception &e) {
        std::cerr << "calibration_sparse: " << e.what() << '\n';
        return 2;
    }
}
