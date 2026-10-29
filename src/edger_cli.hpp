#pragma once
#include "io.hpp"
#include <chrono>

// `rapidglm edger-ql`: the quasi-likelihood pipeline of edger_ql on a count
// matrix and a design.
namespace rapidglm::cli {
template <class Hashes>
inline int run_edger_ql(const Tree &config, const std::string &config_path, Hashes &hashes,
                        std::chrono::steady_clock::time_point start) {
    for (const auto &key : {"covariates", "formula", "offset", "weights", "family", "units"})
        require(!config.count(key), (std::string(key) + " is not an edger-ql setting").c_str());
    const std::vector<std::string> settings = {
        "responses",    "design",        "output",     "coefficient",     "contrast",
        "coefficients", "filter",        "min_count",  "min_total_count", "large_n",
        "min_prop",     "normalization", "dispersion", "prior_count",     "threads"};
    for (const auto &entry : config)
        require(std::find(settings.begin(), settings.end(), entry.first) != settings.end(),
                ("unknown edger-ql setting: " + entry.first).c_str());
    const auto responses_path = config.get<std::string>("responses");
    const auto design_path = config.get<std::string>("design");
    const auto output = config.get<std::string>("output");
    require(!output.empty(), "output prefix required");
    const auto counts = read_market(responses_path);
    const auto design = read_matrix(design_path);
    const auto hypothesis = cli::hypothesis(config, design.cols);
    QlOptions o;
    o.filter = config.get<bool>("filter", true);
    o.min_count = config.get<double>("min_count", o.min_count);
    o.min_total_count = config.get<double>("min_total_count", o.min_total_count);
    o.large_n = config.get<double>("large_n", o.large_n);
    o.min_prop = config.get<double>("min_prop", o.min_prop);
    const auto normalization = config.get<std::string>("normalization", "tmm");
    require(normalization == "tmm" || normalization == "none", "normalization must be tmm/none");
    o.tmm = normalization == "tmm";
    if (config.count("dispersion"))
        o.dispersion = config.get<double>("dispersion");
    o.prior_count = config.get<double>("prior_count", o.prior_count);
    o.threads = config.get<int>("threads", 1);
    for (const auto &path : {responses_path, design_path})
        hashes[path] = file_sha1(path);
    const auto result = edger_ql(design, counts, hypothesis, o);
    {
        std::ofstream out(output + ".tsv");
        out << "gene\tstatus\ttotal\tAveLogCPM\tlogFC\tcoefficient\tunshrunk_coefficient\tF\t"
               "PValue\tFDR\tdeviance\tdeviance_adj\tdf_residual_adj\ts2\ts2_prior\ts2_post\t"
               "df_total\n";
        for (std::size_t j = 0; j < result.genes.size(); ++j) {
            const auto &g = result.genes[j];
            out << j << '\t' << g.status << '\t';
            value(out, g.total);
            for (double v : {g.ave_log_cpm, g.log_fc, g.coefficient, g.unshrunk_coefficient, g.f,
                             g.p_value, g.fdr, g.deviance, g.deviance_adj, g.df_residual_adj, g.s2,
                             g.s2_prior, g.s2_post, g.df_total}) {
                out << '\t';
                value(out, v);
            }
            out << '\n';
        }
    }
    {
        std::ofstream out(output + ".samples.tsv");
        out << "sample\tlib_size\tlib_size_filtered\tnorm_factor\teffective_lib_size\n";
        for (std::size_t i = 0; i < counts.rows; ++i) {
            out << i;
            for (double v : {result.library_sizes[i], result.filtered_library_sizes[i],
                             result.norm_factors[i], result.effective_library_sizes[i]}) {
                out << '\t';
                value(out, v);
            }
            out << '\n';
        }
    }
    std::size_t discoveries = 0;
    for (const auto &g : result.genes)
        discoveries += g.fdr <= 0.1;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream record(output + ".run.json");
    // JSON numbers; null for a value not computed.
    const auto number = [](std::ostream &out, double v) {
        if (std::isfinite(v))
            value(out, v);
        else
            out << "null";
    };
    record << "{\"command\": \"edger-ql\", \"version\": \"" << version << "\", \"config\": ";
    quoted(record, config_path);
    record << ",\n  \"division\": ";
    quoted(record, vmath::division_name());
    record << ", \"samples\": " << counts.rows << ", \"genes\": " << counts.cols
           << ", \"kept\": " << result.kept << ", \"hypothesis\": ";
    hypothesis_record(record, hypothesis);
    record << ", \"filter\": " << (o.filter ? "true" : "false") << ", \"minimum_group\": ";
    number(record, result.minimum_group);
    record << ", \"normalization\": ";
    quoted(record, normalization);
    record << ", \"dispersion\": ";
    number(record, result.dispersion);
    record << ", \"top_proportion\": ";
    number(record, result.top_proportion);
    record << ", \"top_genes\": " << result.top_genes << ", \"average_ql_dispersion\": ";
    number(record, result.scale);
    record << ", \"df_prior\": ";
    number(record, result.df_prior);
    record << ", \"prior_count\": ";
    number(record, o.prior_count);
    record << ", \"discoveries_fdr_0_1\": " << discoveries << ", \"threads\": " << o.threads
           << ", \"input_sha1\": {";
    bool first = true;
    for (const auto &[path, sha] : hashes) {
        record << (first ? "" : ", ");
        quoted(record, path);
        record << ": ";
        quoted(record, sha);
        first = false;
    }
    record << "}, \"wall_seconds\": ";
    number(record, seconds);
    record << "}\n";
    return 0;
}
} // namespace rapidglm::cli
