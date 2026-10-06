#include "calibration_data.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;
int main(int argc, char **argv) {
    try {
        require(argc == 5 || argc == 6,
                "usage: calibration output.tsv datasets draws threads [thinned|direct]");
        std::size_t datasets = std::stoul(argv[2]), draws = std::stoul(argv[3]);
        int threads = std::stoi(argv[4]);
        // direct reproduces the CRT streams used before thinned sampling (b0acdb8 and earlier).
        const std::string method = argc == 6 ? argv[5] : "thinned";
        require(method == "thinned" || method == "direct", "CRT draw must be thinned/direct");
        const auto crt_draw = method == "direct" ? CrtDraw::direct : CrtDraw::thinned;
        require(datasets > 0 && draws > 0, "positive simulation counts required");
        struct Row {
            unsigned scenario;
            std::size_t id;
            double p = rapidglm::nan;
            bool valid = false;
        };
        std::vector<Row> rows(8 * datasets);
        parallel_for(rows.size(), threads, [&](std::size_t index) {
            auto &row = rows[index];
            row.scenario = index / datasets;
            row.id = index % datasets;
            Family family = static_cast<Family>(row.scenario % 4);
            bool crt = row.scenario >= 4;
            auto data = rapidglm_test::generate_null(row.scenario, index);
            const auto &x = data.design;
            const auto &y = data.response;
            const auto &g = data.assignment;
            const auto &p = data.probability;
            FitOptions o;
            o.family = family;
            auto null = fit(x, y, o);
            if (!null.ok())
                return;
            ResampleOptions r;
            r.seed = 20260926;
            r.test_id = index;
            r.budgets = {draws};
            r.approximate = false;
            auto result = resample(ScoreWorkspace(x, null), g,
                                   crt ? crt_sampler(p, crt_draw) : permutation_sampler(g), r);
            row.valid = result.valid;
            row.p = result.p_value;
        });
        std::ofstream out(argv[1]);
        out << "scenario\tdataset\tvalid\tp_value\n";
        for (const auto &r : rows)
            out << r.scenario << '\t' << r.id << '\t' << r.valid << '\t' << std::setprecision(17)
                << r.p << '\n';
        return out ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
