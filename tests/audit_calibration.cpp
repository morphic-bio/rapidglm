#include "calibration_data.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;

// Independent two-column weighted projection in extended precision. This uses
// a scalar 2x2 inverse rather than the production Cholesky/score workspaces.
double reference_score(const rapidglm_test::NullData &d, const FitResult &f,
                       const std::vector<std::size_t> &selected) {
    long double a = 0, b = 0, c = 0, r0 = 0, r1 = 0, g0 = 0, g1 = 0, u = 0;
    for (std::size_t i = 0; i < d.design.rows; ++i) {
        long double z = d.design(i, 1), w = f.working_weights[i], r = f.score_residuals[i];
        a += w;
        b += w * z;
        c += w * z * z;
        r0 += r;
        r1 += z * r;
    }
    for (auto i : selected) {
        g0 += f.working_weights[i];
        g1 += static_cast<long double>(f.working_weights[i]) * d.design(i, 1);
        u += f.score_residuals[i];
    }
    long double determinant = a * c - b * b;
    long double q0 = (c * g0 - b * g1) / determinant, q1 = (a * g1 - b * g0) / determinant;
    return static_cast<double>((u - q0 * r0 - q1 * r1) /
                               std::sqrt((g0 - g0 * q0 - g1 * q1) * f.scale));
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: audit_calibration output.tsv datasets");
        const auto n = std::stoul(argv[2]);
        require(n >= 1000, "retain all original 1000 cases");
        struct Row {
            double p, independent = rapidglm::nan, error = 0;
        };
        std::vector<Row> rows(n);
        parallel_for(n, 8, [&](std::size_t i) {
            auto index = 5000 + i; // Preserve the exact original Poisson CRT panel.
            auto d = rapidglm_test::generate_null(5, index);
            FitOptions o;
            o.family = Family::poisson;
            auto f = fit(d.design, d.response, o);
            require(f.ok() && !f.has_boundary(), "invalid calibration fit");
            ScoreWorkspace s(d.design, f);
            ResampleOptions ro;
            ro.seed = 20260926;
            ro.test_id = index;
            ro.approximate = false;
            ro.budgets = {499};
            auto result = resample(s, d.assignment, crt_sampler(d.probability), ro);
            require(result.valid, "invalid calibration test");
            rows[i].p = result.p_value;
            if (i >= 1000)
                return;
            auto sampler = crt_sampler(d.probability);
            BinaryScoreWorkspace binary(s);
            std::vector<std::size_t> selected;
            for (std::size_t j = 0; j < d.assignment.size(); ++j)
                if (d.assignment[j])
                    selected.push_back(j);
            double observed = reference_score(d, f, selected);
            rows[i].error = std::abs(observed - result.statistic);
            std::size_t left = 1, right = 1;
            for (std::size_t draw = 0; draw < 499; ++draw) {
                CounterRng rng(ro.seed, ro.test_id, draw);
                sampler(rng, selected);
                double score = reference_score(d, f, selected);
                rows[i].error =
                    std::max(rows[i].error, std::abs(score - binary.statistic(selected)));
                left += score <= observed;
                right += score >= observed;
            }
            rows[i].independent = std::min(1.0, 2.0 * std::min(left, right) / 500);
        });
        std::ofstream out(argv[1]);
        out << "dataset\tindex\tp_value\tindependent_p\tscore_max_abs\n" << std::setprecision(17);
        for (std::size_t i = 0; i < n; ++i)
            out << i << '\t' << 5000 + i << '\t' << rows[i].p << '\t' << rows[i].independent << '\t'
                << rows[i].error << '\n';
        return out ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
