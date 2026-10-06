// Frozen complement-control panels: cache nulls, compare full effect fits.
#include "../src/io.hpp"
#include <chrono>
#include <iostream>

using namespace rapidglm;
namespace io = rapidglm::cli;
using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
    try {
        require(argc == 5, "usage: effect_benchmark config.json output.tsv threads repetitions");
        io::Tree config;
        boost::property_tree::read_json(argv[1], config);
        const int threads = std::stoi(argv[3]), repetitions = std::stoi(argv[4]);
        require(threads > 0 && repetitions > 0, "invalid benchmark dimensions");
        require(config.get<std::string>("control_group", "") == "complement" &&
                    !config.get<bool>("cell_qc", true) &&
                    (!config.get<bool>("assignment_qc", true) ||
                     config.get<std::string>("moi", "low") == "high") &&
                    config.get<std::string>("integration", "union") == "union",
                "benchmark requires frozen complement controls and union integration");
        auto x = config.get_optional<std::string>("design")
                     ? io::read_matrix(config.get<std::string>("design"))
                     : io::formula_design(config.get<std::string>("covariates"),
                                          config.get<std::string>("formula"));
        auto responses = io::read_market(config.get<std::string>("responses"));
        auto assignments = io::read_market(config.get<std::string>("assignments"));
        auto guides = io::read_table(config.get<std::string>("guides"));
        auto pairs = io::read_table(config.get<std::string>("pairs"));
        auto options = io::fit_options(config);
        require(options.family == Family::negative_binomial, "NB benchmark expected");
        std::map<std::string, Vector> treatment;
        for (std::size_t j = 1; j < guides.size(); ++j) {
            auto &g = treatment[guides[j][1]];
            if (g.empty())
                g.resize(x.rows);
            for (auto k = assignments.column_ptr[j - 1]; k < assignments.column_ptr[j]; ++k)
                if (assignments.values[k])
                    g[assignments.row_index[k]] = 1;
        }
        std::vector<std::pair<std::size_t, std::string>> tasks;
        std::map<std::size_t, Vector> outcomes;
        std::map<std::size_t, FitResult> nulls;
        for (std::size_t i = 1; i < pairs.size(); ++i) {
            auto j = static_cast<std::size_t>(io::number(pairs[i][0]));
            if (!outcomes.count(j)) {
                outcomes[j] = responses.column(j);
                nulls[j] = fit(x, outcomes.at(j), options);
                require(nulls.at(j).ok() && !nulls.at(j).has_boundary(), "invalid cached null");
            }
            tasks.emplace_back(j, pairs[i][1]);
        }
        std::ofstream output(argv[2]);
        require(bool(output), "cannot open benchmark output");
        output
            << std::setprecision(17)
            << "repetition\tthreads\twarm\tpair\tresponse\ttarget\tstatus\tseconds\tbatch_seconds"
               "\titerations\tdispersion_iterations\ttheta\tloglik\tlog2fc\tlog2fc_se"
               "\tmax_beta_delta\tmax_se_delta\tmax_theta_delta\tloglik_delta\n";
        std::vector<FitResult> reference(tasks.size());
        for (int repetition = 0; repetition <= repetitions; ++repetition) {
            for (int mode = 0; mode < 2; ++mode) {
                const bool warm = (mode + repetition) % 2 != 0;
                std::vector<FitResult> fits(tasks.size());
                Vector seconds(tasks.size());
                auto start = Clock::now();
                parallel_for(tasks.size(), threads, [&](std::size_t i) {
                    const auto &task = tasks[i];
                    auto before = Clock::now();
                    fits[i] =
                        fit_effect_model(x, outcomes.at(task.first), treatment.at(task.second),
                                         nulls.at(task.first), options, warm);
                    seconds[i] = std::chrono::duration<double>(Clock::now() - before).count();
                });
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                if (repetition == 0 && !warm)
                    reference = fits;
                for (std::size_t i = 0; i < tasks.size(); ++i) {
                    const auto &f = fits[i], &cold = reference[i];
                    require(f.ok() && f.status == cold.status &&
                                f.boundary_coefficients == cold.boundary_coefficients,
                            "effect-fit status changed");
                    auto effect = log2_fold_change(f, x.cols, options.robust);
                    double beta_delta = 0, se_delta = 0;
                    for (std::size_t k = 0; k < f.coefficients.size(); ++k) {
                        beta_delta = std::max(beta_delta,
                                              std::abs(f.coefficients[k] - cold.coefficients[k]));
                        se_delta = std::max(
                            se_delta, std::abs(f.standard_errors[k] - cold.standard_errors[k]));
                    }
                    output << repetition << '\t' << threads << '\t' << warm << '\t' << i << '\t'
                           << tasks[i].first << '\t' << tasks[i].second << '\t'
                           << status_name(f.status) << '\t' << seconds[i] << '\t' << elapsed << '\t'
                           << f.iterations << '\t' << f.dispersion_iterations << '\t' << f.theta
                           << '\t' << f.log_likelihood << '\t' << effect.estimate << '\t'
                           << effect.standard_error << '\t' << beta_delta << '\t' << se_delta
                           << '\t' << std::abs(f.theta - cold.theta) << '\t'
                           << f.log_likelihood - cold.log_likelihood << '\n';
                }
                output.flush();
                std::cerr << "repetition " << repetition << " warm " << warm << " seconds "
                          << elapsed << '\n';
            }
        }
        require(bool(output), "cannot write benchmark output");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
