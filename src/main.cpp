#include "io.hpp"
#include "tiles.hpp"
#include <chrono>
#include <ctime>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <sys/resource.h>
#include <unistd.h>
using namespace rapidglm;
namespace io = rapidglm::cli;
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << version << '\n';
            return 0;
        }
        if (argc < 2 || (argc == 2 && std::string(argv[1]) == "--help")) {
            std::cout
                << "rapidglm " << version
                << "\nUsage: rapidglm <fit|scan|sceptre|bulk-de|pseudobulk-de|da-lr|dose-trend> "
                   "config.json\nResponses: feature-by-observation MatrixMarket; design: "
                   "observation-by-predictor numeric TSV.\nSee docs/CLI.md for the "
                   "configuration contract.\n";
            return 0;
        }
        require(argc == 3, "expected command and JSON configuration");
        std::string command = argv[1];
        require(command == "fit" || command == "scan" || command == "sceptre" ||
                    command == "bulk-de" || command == "pseudobulk-de" || command == "da-lr" ||
                    command == "dose-trend",
                "unknown command");
        auto start = std::chrono::steady_clock::now();
        io::Tree config;
        auto config_file = io::input(argv[2]);
        std::string config_text((std::istreambuf_iterator<char>(config_file)),
                                std::istreambuf_iterator<char>());
        std::istringstream config_stream(config_text);
        boost::property_tree::read_json(config_stream, config);
        std::map<std::string, std::string> hashes;
        hashes[argv[2]] = io::file_sha1(argv[2]);
        // A tiled run keeps its inputs' fingerprints in the checkpoint directory:
        // a file whose size and modification time are those recorded there is not
        // read again to hash it (each tile task is a process of its own).
        const auto fingerprint_directory =
            command == "sceptre" ? config.get<std::string>("checkpoint_dir", "") : std::string();
        auto known_inputs = fingerprint_directory.empty()
                                ? std::map<std::string, io::tiles::InputFingerprint>()
                                : io::tiles::read_input_fingerprints(fingerprint_directory);
        bool new_fingerprints = false;
        // The recorded fingerprint of `path`, or its size and time for a new record.
        const auto recorded = [&](const std::string &path) {
            auto stamp = io::tiles::input_stamp(path);
            const auto found = known_inputs.find(path);
            if (stamp && found != known_inputs.end() && found->second.same_file(*stamp))
                stamp->sha1 = found->second.sha1;
            return stamp;
        };
        const auto remember = [&](const std::string &path,
                                  std::optional<io::tiles::InputFingerprint> stamp,
                                  const std::string &sha1) {
            hashes[path] = sha1;
            // A file modified in the last seconds is not recorded: a further change
            // could leave its size and time as they are now.
            if (stamp && stamp->sha1.empty() &&
                stamp->seconds + 2 <= static_cast<std::int64_t>(std::time(nullptr))) {
                stamp->sha1 = sha1;
                known_inputs[path] = *stamp;
                new_fingerprints = true;
            }
        };
        for (const char *key :
             {"design", "covariates", "offset", "weights", "assignments", "guides", "pairs",
              "targets", "grid_responses", "response_ids", "units", "group", "qc_total",
              "qc_detected", "qc_mito", "predictors", "strata", "size_factor_values"})
            if (auto path = config.get_optional<std::string>(key))
                if (!path->empty()) {
                    const auto stamp = recorded(*path);
                    remember(*path, stamp,
                             stamp && !stamp->sha1.empty() ? stamp->sha1 : io::file_sha1(*path));
                }
        // The responses are the large input: their fingerprint is computed on its
        // own thread while they are parsed and analysed, and collected when it is
        // first needed (the tile plan, or the run record).
        const auto responses_path = config.get<std::string>("responses");
        const auto responses_stamp = recorded(responses_path);
        std::future<std::string> responses_hash;
        if (responses_stamp && !responses_stamp->sha1.empty())
            hashes[responses_path] = responses_stamp->sha1;
        else
            responses_hash = std::async(std::launch::async, io::file_sha1, responses_path);
        const auto finish_hashes = [&] {
            if (responses_hash.valid())
                remember(responses_path, responses_stamp, responses_hash.get());
            if (new_fingerprints && !fingerprint_directory.empty()) {
                io::tiles::write_input_fingerprints(fingerprint_directory, known_inputs);
                new_fingerprints = false;
            }
        };
        // The perturbation workflow reads counts one response at a time, from the
        // compact count matrix; `responses` then carries the dimensions only.
        // Responses that are not counts (other response models) use the general
        // container.
        SparseCSC responses;
        CountMatrix counts;
        bool count_matrix = command == "sceptre";
        if (count_matrix)
            try {
                counts = io::read_market_counts(responses_path, config.get<int>("threads", 1));
                responses.rows = counts.rows;
                responses.cols = counts.cols;
            } catch (const io::NotCountTable &) {
                count_matrix = false;
            }
        if (!count_matrix)
            responses = io::read_market(responses_path);
        Matrix design = config.count("design")
                            ? io::read_matrix(config.get<std::string>("design"))
                            : io::formula_design(config.get<std::string>("covariates"),
                                                 config.get<std::string>("formula"));
        auto options = io::fit_options(config);
        const auto dispersion_mode = config.get<std::string>("dispersion", "ml");
        require((dispersion_mode != "shared" && dispersion_mode != "trend") ||
                    (command != "sceptre" && command != "da-lr" && command != "scan"),
                "dispersion=shared/trend is not supported by the scan, sceptre or da-lr commands");
        int threads = config.get<int>("threads", 1);
        require(threads > 0, "threads must be positive");
        std::string prefix = config.get<std::string>("output");
        require(!prefix.empty(), "output prefix required");
        auto directory = std::filesystem::path(prefix).parent_path();
        if (!directory.empty())
            std::filesystem::create_directories(directory);
        std::ofstream out(prefix + ".tsv.tmp");
        require(static_cast<bool>(out), "cannot create result table");
        std::size_t failures = 0, retained = responses.rows, calibration = 0, power = 0,
                    discoveries = 0, samplers_cached = 0, samplers_rebuilt = 0,
                    sampler_cache_mb = 0, null_fits = 0;
        // By kind (discovery, negative control, positive control).
        std::array<std::size_t, 3> pairs_listed{}, pairs_passing_qc{}, family_sizes{};
        std::uint64_t seed = config.get<std::uint64_t>("seed", 1);
        QuasiLikelihoodResult quasi;
        BulkResult bulk;
        std::optional<io::TileRun> tiled;
        const auto null_model =
            config.get<std::string>("null_model", command == "scan" ? "direct" : "sceptre");
        require(command == "sceptre" || command == "scan" || !config.count("null_model"),
                "null_model is a scan or sceptre setting");
        // Per-feature ML Wald and LR tests were anticonservative in the pseudobulk
        // mock-comparison null with eight replicates; the quasi-likelihood test was
        // calibrated (validation/QL_TEST.json).
        const auto test =
            config.get<std::string>("test", command != "pseudobulk-de"                    ? "wald"
                                            : options.family == Family::negative_binomial ? "ql"
                                                                                          : "lr");
        const auto effect_fits = config.get<std::string>("effect_fits", "significant");
        const auto crt_draw = config.get<std::string>("crt_draw", "thinned");
        const bool reuse_draws = config.get<bool>("reuse_draws", false);
        const bool shared_screen = config.get<bool>("shared_screen", false);
        const bool sequential_screen = config.get<bool>("sequential_screen", false);
        const double early_stop_p = config.get<double>("early_stop_p", 0.0);
        const auto stream_keys = config.get<std::string>("stream_keys", "identity");
        const auto adjustment_family = config.get<std::string>("adjustment_family", "tested");
        // A grid request: targets against responses, without a list of the pairs.
        const auto targets_path = config.get<std::string>("targets", "");
        const bool grid_request = !targets_path.empty();
        const auto response_ids_path = config.get<std::string>("response_ids", "");
        for (const char *key : {"reuse_draws",
                                "shared_screen",
                                "shared_screen_p",
                                "sequential_screen",
                                "early_stop_p",
                                "draw_cache_mb",
                                "sampler_cache_mb",
                                "checkpoint_dir",
                                "phase",
                                "tile",
                                "tile_targets",
                                "tile_responses",
                                "stream_keys",
                                "response_ids",
                                "adjustment_family",
                                "family_size_discovery",
                                "family_size_negative",
                                "family_size_positive",
                                "targets",
                                "grid_responses"})
            io::require(command == "sceptre" || !config.count(key),
                        std::string(key) + " is a sceptre setting");
        for (const char *key : {"effect_fits", "crt_draw", "fit_check", "confirm_draws"})
            io::require(command == "sceptre" || command == "scan" || !config.count(key),
                        std::string(key) + " is a scan or sceptre setting");
        if (crt_draw == "hazard" && !aes_instructions())
            std::cerr << "warning: this processor has no AES instructions, so crt_draw hazard "
                         "draws with the thinned sampler (the same law, from thinned's stream)\n";
        const auto calibration_name = config.get<std::string>("calibration", "permutation");
        for (const char *key : {"calibration", "predictors", "strata"})
            require(command == "scan" || !config.count(key),
                    "calibration, predictors and strata are scan settings");
        for (const char *key : {"coefficients", "contrast"})
            require(command == "bulk-de" || command == "pseudobulk-de" || command == "dose-trend" ||
                        !config.count(key),
                    "coefficients and contrast are bulk-de, pseudobulk-de and dose-trend settings");
        std::optional<Hypothesis> tested;
        double screen = 0;       // empirical screen of resampled tests (scan, sceptre)
        std::size_t confirm = 0; // their confirmation draws in effect
        require(command == "scan" || command == "sceptre" || !config.count("screen"),
                "screen is a scan or sceptre setting");
        for (const char *key :
             {"size_factors", "size_factor_values", "trend", "min_mean", "keep_minimum_start",
              "outlier_sd", "cooks_filter", "cooks_cutoff", "min_replicates_for_replace",
              "independent_filtering", "lfc_shrink", "legacy_cooks"})
            require(command == "bulk-de" || !config.count(key),
                    "size_factors, size_factor_values, trend, min_mean, keep_minimum_start, "
                    "outlier_sd, cooks_filter, cooks_cutoff, min_replicates_for_replace, "
                    "independent_filtering, lfc_shrink and legacy_cooks are bulk-de settings");
        if (command == "scan") {
            require(null_model == "direct" || null_model == "sceptre",
                    "null_model must be direct/sceptre");
            require(calibration_name == "asymptotic" || calibration_name == "permutation" ||
                        calibration_name == "crt",
                    "calibration must be asymptotic/permutation/crt");
            require(effect_fits == "all" || effect_fits == "significant" || effect_fits == "none",
                    "effect_fits must be all/significant/none");
            require(crt_draw == "thinned" || crt_draw == "direct" || crt_draw == "hazard",
                    "crt_draw must be thinned/direct/hazard");
            auto predictors = io::read_market(config.get<std::string>("predictors"));
            ScanOptions o;
            o.model = options;
            o.null_model = null_model == "sceptre" ? FitMethod::sceptre_null : FitMethod::direct;
            o.calibration = calibration_name == "asymptotic" ? Calibration::asymptotic
                            : calibration_name == "crt"      ? Calibration::crt
                                                             : Calibration::permutation;
            o.crt_draw = crt_draw == "direct"   ? CrtDraw::direct
                         : crt_draw == "hazard" ? CrtDraw::hazard
                                                : CrtDraw::thinned;
            o.effect_fits = effect_fits == "all"           ? EffectFits::all
                            : effect_fits == "significant" ? EffectFits::significant
                                                           : EffectFits::none;
            o.resampling = io::resample_options(config, seed);
            screen = o.resampling.screen;
            confirm = o.resampling.confirm;
            for (double v : io::read_vector(config.get<std::string>("strata", ""))) {
                require(v >= 0 && std::floor(v) == v, "strata must be nonnegative integers");
                o.strata.push_back(static_cast<std::size_t>(v));
            }
            o.alpha = config.get<double>("alpha", 0.1);
            o.min_nonzero = config.get<std::size_t>("min_nonzero", 0);
            auto adjustment = config.get<std::string>("adjustment", "BH");
            require(adjustment == "BH" || adjustment == "bonferroni", "invalid adjustment");
            o.bonferroni = adjustment == "bonferroni";
            o.parallel_draws = config.get<bool>("parallel_draws", true);
            o.parallel_pairs = config.get<bool>("parallel_pairs", true);
            o.warm_start_effects = config.get<bool>("warm_start_effects", true);
            o.threads = threads;
            std::vector<ScanPair> pairs;
            if (auto path = config.get_optional<std::string>("pairs"); path && !path->empty()) {
                auto table = io::read_table(*path);
                require(!table.empty() && table[0].size() >= 2 && table[0][0] == "response" &&
                            table[0][1] == "predictor" &&
                            (table[0].size() == 2 || table[0][2] == "family"),
                        "pairs header must be response,predictor[,family] (tab separated)");
                for (std::size_t i = 1; i < table.size(); ++i) {
                    const auto &row = table[i];
                    require(row.size() == table[0].size(), "pairs row has the wrong column count");
                    double response = io::number(row[0]), predictor = io::number(row[1]);
                    require(response >= 0 && std::floor(response) == response && predictor >= 0 &&
                                std::floor(predictor) == predictor,
                            "pair indices must be nonnegative integers");
                    pairs.push_back({static_cast<std::size_t>(response),
                                     static_cast<std::size_t>(predictor),
                                     row.size() > 2 ? row[2] : "all"});
                }
                require(!pairs.empty(), "pairs file has no rows");
            }
            auto results = score_scan(design, responses, predictors, pairs, o);
            io::scan_table(out, results);
            for (const auto &r : results) {
                failures += !std::isfinite(r.p_value);
                discoveries += r.status.rfind("ok", 0) == 0 && r.adjusted_p <= o.alpha;
            }
        } else if (command == "bulk-de") {
            require(options.family == Family::negative_binomial && dispersion_mode == "ml" &&
                        !config.count("cox_reid") && options.ridge == 0 && !options.robust &&
                        options.offset.empty() && options.weights.empty(),
                    "bulk-de fits NB models with its own dispersion estimates and size factors; "
                    "omit dispersion, cox_reid, ridge, robust, offset and weights");
            BulkOptions o;
            tested = io::hypothesis(config, design.cols);
            o.hypothesis = *tested;
            require(test == "wald" || test == "lrt", "bulk-de test must be wald/lrt");
            o.lrt = test == "lrt";
            const auto sf = config.get<std::string>("size_factors", "ratio");
            require(sf == "ratio" || sf == "poscounts", "size_factors must be ratio/poscounts");
            o.size_factor_method =
                sf == "poscounts" ? SizeFactorMethod::poscounts : SizeFactorMethod::ratio;
            o.size_factors = io::read_vector(config.get<std::string>("size_factor_values", ""));
            const auto trend = config.get<std::string>("trend", "parametric");
            require(trend == "parametric" || trend == "local" || trend == "mean",
                    "trend must be parametric/local/mean");
            o.trend = trend == "local"  ? DispersionTrend::local
                      : trend == "mean" ? DispersionTrend::mean
                                        : DispersionTrend::parametric;
            o.min_mean = config.get<double>("min_mean", 0.5);
            o.keep_minimum_start = config.get<bool>("keep_minimum_start", true);
            o.outlier_sd = config.get<double>("outlier_sd", 2);
            o.cooks_filter = config.get<bool>("cooks_filter", true);
            o.cooks_cutoff = config.get<double>("cooks_cutoff", 0);
            o.legacy_cooks = config.get<bool>("legacy_cooks", true);
            o.min_replicates_for_replace = config.get<std::size_t>("min_replicates_for_replace", 7);
            o.independent_filtering = config.get<bool>("independent_filtering", true);
            o.alpha = config.get<double>("alpha", 0.1);
            const auto shrink = config.get<std::string>("lfc_shrink", "none");
            require(shrink == "none" || shrink == "normal", "lfc_shrink must be none/normal");
            o.lfc_shrinkage = shrink == "normal" ? LfcShrinkage::normal : LfcShrinkage::none;
            o.seed = seed;
            o.threads = threads;
            bulk = bulk_de(design, responses, o);
            io::bulk_table(out, bulk.genes, o.lfc_shrinkage == LfcShrinkage::normal);
            std::ofstream filter_table(prefix + ".filter.tsv");
            filter_table << "theta\trejections\tfit\n";
            for (std::size_t k = 0; k < bulk.filter_rejections.size(); ++k)
                filter_table << std::setprecision(17) << bulk.filter_rejections[k].first << '\t'
                             << bulk.filter_rejections[k].second << '\t' << bulk.filter_fit[k]
                             << '\n';
            require(static_cast<bool>(filter_table), "failed writing filter table");
            for (const auto &g : bulk.genes) {
                failures += !std::isfinite(g.p_value);
                discoveries += std::isfinite(g.adjusted_p) && g.adjusted_p < o.alpha;
            }
        } else if (command == "sceptre") {
            require(null_model == "direct" || null_model == "sceptre", "unknown null_model");
            require(null_model != "sceptre" ||
                        (config.get<std::string>("dispersion", "ml") == "ml" &&
                         options.family == Family::negative_binomial && options.ridge == 0),
                    "null_model=sceptre (the default) requires family=nb, dispersion=ml and "
                    "ridge=0; set null_model=direct for other response models");
            auto assignments = io::read_market(config.get<std::string>("assignments"));
            auto guide_table = io::read_table(config.get<std::string>("guides"));
            // With a grid request the listed pairs are optional (positive controls, say).
            const auto pair_table =
                grid_request && config.get<std::string>("pairs", "").empty()
                    ? std::vector<std::vector<std::string>>({{"response", "target", "kind"}})
                    : io::read_table(config.get<std::string>("pairs"));
            require(guide_table[0] == std::vector<std::string>({"id", "target", "non_targeting"}),
                    "guides header must be id,target,non_targeting (tab separated)");
            require(pair_table[0] == std::vector<std::string>({"response", "target", "kind"}),
                    "pairs header must be response,target,kind (tab separated)");
            std::vector<Guide> guides;
            for (std::size_t i = 1; i < guide_table.size(); ++i) {
                auto &r = guide_table[i];
                require(r[2] == "0" || r[2] == "1", "non_targeting must be 0/1");
                guides.push_back({r[0], r[1], r[2] == "1"});
            }
            std::vector<Pair> pairs;
            for (std::size_t i = 1; i < pair_table.size(); ++i) {
                auto &r = pair_table[i];
                require(r[2] == "discovery" || r[2] == "negative" || r[2] == "positive",
                        "unknown pair kind");
                double index = io::number(r[0]);
                require(index >= 0 && std::floor(index) == index && index < responses.cols,
                        "invalid response index");
                pairs.push_back({static_cast<std::size_t>(index), r[1],
                                 r[2] == "negative"   ? PairKind::negative_control
                                 : r[2] == "positive" ? PairKind::positive_control
                                                      : PairKind::discovery});
            }
            SceptreOptions o;
            o.model = options;
            o.null_model = null_model == "sceptre" ? FitMethod::sceptre_null : FitMethod::direct;
            o.threads = threads;
            auto moi = config.get<std::string>("moi", "low");
            require(moi == "low" || moi == "high", "MOI must be low/high");
            o.high_moi = moi == "high";
            auto control =
                config.get<std::string>("control_group", o.high_moi ? "complement" : "nt_cells");
            require(control == "complement" || control == "nt_cells", "invalid control group");
            o.nt_controls = control == "nt_cells";
            auto mechanism = config.get<std::string>("resampling_mechanism",
                                                     o.high_moi ? "crt" : "permutations");
            require(mechanism == "crt" || mechanism == "permutations",
                    "invalid resampling mechanism");
            o.crt = mechanism == "crt";
            require(crt_draw == "thinned" || crt_draw == "direct" || crt_draw == "hazard",
                    "crt_draw must be thinned/direct/hazard");
            o.crt_draw = crt_draw == "direct"   ? CrtDraw::direct
                         : crt_draw == "hazard" ? CrtDraw::hazard
                                                : CrtDraw::thinned;
            auto integration = config.get<std::string>("integration", "union");
            require(integration == "union" || integration == "singleton" ||
                        integration == "bonferroni",
                    "invalid guide integration");
            o.integration = integration == "union"       ? Integration::union_guides
                            : integration == "singleton" ? Integration::singleton
                                                         : Integration::bonferroni;
            auto side = config.get<std::string>("side", "both");
            require(side == "left" || side == "right" || side == "both", "invalid side");
            o.resampling.side = side == "left"    ? Side::left
                                : side == "right" ? Side::right
                                                  : Side::both;
            auto approx = config.get<std::string>("approximation", "skew_normal");
            require(approx == "skew_normal" || approx == "no_approximation",
                    "invalid approximation");
            o.resampling.approximate = approx == "skew_normal";
            o.resampling.fit_check = io::fit_check(config);
            o.resampling.seed = seed;
            if (auto budgets = config.get_child_optional("budgets")) {
                o.resampling.budgets.clear();
                for (auto &v : *budgets)
                    o.resampling.budgets.push_back(v.second.get_value<std::size_t>());
            }
            o.resampling.screen = config.get<double>("screen", o.resampling.screen);
            screen = o.resampling.screen;
            o.resampling.confirm = io::confirm_draws(config, o.resampling);
            confirm = o.resampling.confirm;
            o.cell_qc = config.get<bool>("cell_qc", true);
            o.assignment_qc = config.get<bool>("assignment_qc", true);
            o.parallel_draws = config.get<bool>("parallel_draws", true);
            o.parallel_pairs = config.get<bool>("parallel_pairs", true);
            // Every target's CRT sampler is kept unless that would take more than
            // half of the machine's memory.
            const auto pages = ::sysconf(_SC_PHYS_PAGES), page = ::sysconf(_SC_PAGE_SIZE);
            const std::size_t half_memory_mb =
                pages > 0 && page > 0
                    ? static_cast<std::size_t>(pages) / 2 * static_cast<std::size_t>(page) >> 20
                    : 4096;
            sampler_cache_mb = config.get<std::size_t>("sampler_cache_mb", half_memory_mb);
            o.sampler_cache_bytes = sampler_cache_mb << 20;
            const auto warn_samplers = [&](std::size_t rebuilt) {
                if (rebuilt)
                    std::cerr << "warning: the CRT samplers of " << rebuilt
                              << " target(s) did not fit sampler_cache_mb (" << sampler_cache_mb
                              << ") and were rebuilt for every pair, which is slow; raise "
                                 "sampler_cache_mb, or use checkpoint_dir without tile sizes, "
                                 "which splits the targets into blocks that fit\n";
            };
            o.reuse_draws = reuse_draws;
            o.shared_screen = shared_screen;
            o.resampling.sequential_screen = sequential_screen;
            io::require(std::isfinite(early_stop_p) && early_stop_p >= 0 && early_stop_p < 1,
                        "early_stop_p must be in [0, 1); 0 turns it off");
            o.resampling.early_stop = early_stop_p;
            o.shared_screen_p = config.get<double>("shared_screen_p", o.shared_screen_p);
            io::require(shared_screen || !config.count("shared_screen_p"),
                        "shared_screen_p needs shared_screen");
            o.draw_cache_bytes = config.get<std::size_t>("draw_cache_mb", 8192) << 20;
            o.warm_start_effects = config.get<bool>("warm_start_effects", true);
            require(effect_fits == "all" || effect_fits == "significant" || effect_fits == "none",
                    "effect_fits must be all/significant/none");
            o.effect_fits = effect_fits == "all"           ? EffectFits::all
                            : effect_fits == "significant" ? EffectFits::significant
                                                           : EffectFits::none;
            o.min_nonzero = config.get<std::size_t>("min_nonzero", 7);
            o.alpha = config.get<double>("alpha", 0.1);
            o.lower_quantile = config.get<double>("qc_lower", 0.01);
            o.upper_quantile = config.get<double>("qc_upper", 0.99);
            o.max_mito = config.get<double>("max_mito", 0.2);
            auto adjustment = config.get<std::string>("adjustment", "BH");
            require(adjustment == "BH" || adjustment == "bonferroni", "invalid adjustment");
            o.bonferroni_adjustment = adjustment == "bonferroni";
            require(stream_keys == "identity" || stream_keys == "position",
                    "stream_keys must be identity/position");
            o.stream_keys = stream_keys == "position" ? StreamKeys::position : StreamKeys::identity;
            if (!response_ids_path.empty())
                o.response_ids = io::read_lines(response_ids_path);
            require(adjustment_family == "tested" || adjustment_family == "listed",
                    "adjustment_family must be tested/listed");
            o.family =
                adjustment_family == "listed" ? AdjustmentFamily::listed : AdjustmentFamily::tested;
            o.family_size = {config.get<std::size_t>("family_size_discovery", 0),
                             config.get<std::size_t>("family_size_negative", 0),
                             config.get<std::size_t>("family_size_positive", 0)};
            CellQC qc;
            qc.total_counts = io::read_vector(config.get<std::string>("qc_total", ""));
            qc.detected_features = io::read_vector(config.get<std::string>("qc_detected", ""));
            qc.mitochondrial_fraction = io::read_vector(config.get<std::string>("qc_mito", ""));
            // A grid request is run as sceptre_grid() runs it: pair QC of the whole
            // grid first, then the pairs that pass it and the listed pairs as a
            // list, with the families of listing every pair of the grid.
            io::require(grid_request || !config.count("grid_responses"),
                        "grid_responses needs targets");
            std::vector<PairTally> tally;
            std::array<std::size_t, 3> grid_pairs{};
            if (grid_request) {
                require(o.stream_keys == StreamKeys::identity,
                        "a grid request needs identity stream keys (a list has the positions)");
                const auto target_table = io::read_table(targets_path);
                require(target_table[0] == std::vector<std::string>({"target", "kind"}),
                        "targets header must be target,kind (tab separated)");
                SceptreGrid grid;
                for (std::size_t i = 1; i < target_table.size(); ++i) {
                    const auto &r = target_table[i];
                    require(r[1] == "discovery" || r[1] == "negative",
                            "target kind must be discovery/negative");
                    (r[1] == "negative" ? grid.negative_targets : grid.discovery_targets)
                        .push_back(r[0]);
                }
                if (config.count("grid_responses")) {
                    for (double v : io::read_vector(config.get<std::string>("grid_responses"))) {
                        require(v >= 0 && std::floor(v) == v && v < responses.cols,
                                "invalid response index");
                        grid.responses.push_back(static_cast<std::size_t>(v));
                    }
                    require(!grid.responses.empty(), "grid_responses file has no rows");
                }
                auto checked =
                    count_matrix
                        ? sceptre_pair_qc(design, counts, assignments, guides, grid, qc, o)
                        : sceptre_pair_qc(design, responses, assignments, guides, grid, qc, o);
                // The rows of listing every pair of the grid and the listed pairs
                // (a listed pair may not repeat a pair of the grid in its kind).
                grid.pairs = pairs;
                grid_pairs = rapidglm::detail::grid_rows(guides, grid, checked.tally, o);
                if (o.family == AdjustmentFamily::listed)
                    for (std::size_t k = 0; k < 3; ++k)
                        if (!o.family_size[k])
                            o.family_size[k] = grid_pairs[k];
                checked.passing.insert(checked.passing.end(), pairs.begin(), pairs.end());
                pairs = std::move(checked.passing);
                tally = std::move(checked.tally);
            }
            SceptreResult result;
            const auto checkpoint = config.get<std::string>("checkpoint_dir", "");
            if (checkpoint.empty()) {
                for (const char *key : {"phase", "tile", "tile_targets", "tile_responses"})
                    io::require(!config.count(key), std::string(key) + " needs checkpoint_dir");
                result = count_matrix
                             ? sceptre(design, counts, assignments, guides, pairs, qc, o)
                             : sceptre(design, responses, assignments, guides, pairs, qc, o);
                samplers_cached = result.samplers_cached;
                samplers_rebuilt = result.samplers_rebuilt;
                null_fits = result.null_fits;
            } else {
                tiled.emplace();
                tiled->directory = checkpoint;
                tiled->phase = config.get<std::string>("phase", "all");
                tiled->tile = config.get<std::size_t>("tile", 0);
                tiled->tile_targets =
                    config.get<std::size_t>("tile_targets", o.reuse_draws ? 64 : 0);
                tiled->tile_responses =
                    config.get<std::size_t>("tile_responses", o.reuse_draws ? 0 : 256);
                tiled->automatic = !o.reuse_draws && !config.count("tile_targets") &&
                                   !config.count("tile_responses");
                // Inputs and settings that change results; not threads, scheduling,
                // cache sizes, paths of the output or of the configuration itself.
                std::vector<std::string> parts;
                const std::function<void(const std::string &, const io::Tree &)> flatten =
                    [&](const std::string &path, const io::Tree &tree) {
                        parts.push_back(path + "=" + tree.data());
                        std::size_t k = 0;
                        for (const auto &child : tree)
                            flatten(path + "/" + std::to_string(k++) + ":" + child.first,
                                    child.second);
                    };
                static const std::set<std::string> volatile_keys = {
                    "phase",          "tile",           "threads",        "output",
                    "checkpoint_dir", "parallel_draws", "parallel_pairs", "sampler_cache_mb",
                    "draw_cache_mb"};
                for (const auto &entry : config)
                    if (!volatile_keys.count(entry.first))
                        flatten(entry.first, entry.second);
                // A default that changes results is part of the plan, so that a
                // checkpoint written under another default is not resumed.
                if (!config.count("fit_check"))
                    parts.push_back("default:fit_check=strict");
                if (!config.count("confirm_draws"))
                    parts.push_back("default:confirm_draws=" + std::to_string(confirm));
                if (!config.count("stream_keys"))
                    parts.push_back("default:stream_keys=identity");
                if (!config.count("adjustment_family"))
                    parts.push_back("default:adjustment_family=tested");
                finish_hashes();
                for (const auto &entry : hashes)
                    if (entry.first != argv[2])
                        parts.push_back(entry.first + "#" + entry.second);
                std::sort(parts.begin(), parts.end());
                io::Sha1 fingerprint;
                for (const auto &part : parts)
                    fingerprint.update(part.data(), part.size() + 1);
                tiled->fingerprint = fingerprint.hex();
                auto merged = count_matrix ? io::sceptre_tiled(*tiled, design, counts, assignments,
                                                               guides, pairs, qc, o)
                                           : io::sceptre_tiled(*tiled, design, responses,
                                                               assignments, guides, pairs, qc, o);
                samplers_rebuilt = tiled->samplers_rebuilt;
                null_fits = tiled->null_fits;
                if (!merged) {
                    warn_samplers(samplers_rebuilt);
                    out.close();
                    std::filesystem::remove(prefix + ".tsv.tmp");
                    std::cout << "phase " << tiled->phase << ": " << tiled->computed
                              << " tile(s) computed of " << tiled->tiles << " in "
                              << tiled->directory << '\n';
                    return 0;
                }
                result = std::move(*merged);
            }
            warn_samplers(samplers_rebuilt);
            io::pair_table(out, result.tests);
            retained = result.retained_cells;
            calibration = result.calibration_rejections;
            power = result.power_rejections;
            discoveries = result.discovery_rejections;
            for (const auto &r : result.tests)
                failures += !std::isfinite(r.p_value);
            // A grid request lists every pair of the grid, tested or not.
            pairs_listed = grid_request ? grid_pairs : result.pairs_listed;
            pairs_passing_qc = result.pairs_passing_qc;
            family_sizes = result.family_size;
            if (grid_request) {
                std::ofstream tally_file(prefix + ".tally.tsv");
                io::tally_table(tally_file, tally);
                require(static_cast<bool>(tally_file), "failed writing tally output");
            }
            std::ofstream qq(prefix + ".qq.tsv");
            qq << "expected\tobserved\tlower_95\tupper_95\n";
            for (const auto &q : result.calibration_qq)
                qq << std::setprecision(17) << q.expected << '\t' << q.observed << '\t' << q.lower
                   << '\t' << q.upper << '\n';
            require(static_cast<bool>(qq), "failed writing QQ output");
        } else {
            if (command == "pseudobulk-de" && config.count("units")) {
                auto values = io::read_vector(config.get<std::string>("units"));
                std::vector<std::size_t> units;
                for (double v : values) {
                    require(v >= 0 && std::floor(v) == v && v < design.rows,
                            "invalid aggregation unit");
                    units.push_back(static_cast<std::size_t>(v));
                }
                responses = aggregate_pseudobulk(responses, units, design.rows);
            }
            require(design.rows == responses.rows, "design and response observations differ");
            const auto &mode = dispersion_mode;
            Vector trend;
            if (mode == "shared") {
                options.theta = estimate_shared_theta(design, responses, options, threads);
                options.dispersion = Dispersion::fixed;
            }
            if (mode == "trend") {
                require(command == "fit",
                        "estimated trend currently requires the fit command; supply fixed feature "
                        "dispersions through the C++ API for tests");
                trend = estimate_dispersion_trend(responses,
                                                  fit_batch(design, responses, options, threads));
            }
            if (command == "fit") {
                auto fits = fit_batch(design, responses, options, threads, trend);
                io::fits_table(out, fits);
                for (const auto &f : fits)
                    failures += !f.ok();
            } else if (command == "da-lr") {
                auto group = io::read_vector(config.get<std::string>("group"));
                auto tests = differential_accessibility(design, responses, group, options, threads);
                io::regression_table(out, tests);
                for (const auto &t : tests)
                    failures += !t.test.valid;
            } else {
                tested = io::hypothesis(config, design.cols);
                require(test == "wald" || test == "lr" || test == "ql", "test must be wald/lr/ql");
                if (test == "ql") {
                    require(dispersion_mode == "ml" && !config.count("cox_reid"),
                            "test=ql estimates Cox-Reid adjusted ML dispersions itself; leave "
                            "dispersion at ml and omit cox_reid");
                    quasi = quasi_likelihood_tests(design, responses, options, *tested, threads);
                    io::regression_table(out, quasi.tests, &quasi.features);
                    for (const auto &t : quasi.tests)
                        failures += !t.test.valid;
                } else {
                    auto tests = regression_tests(design, responses, options, *tested, test == "lr",
                                                  threads);
                    io::regression_table(out, tests);
                    for (const auto &t : tests)
                        failures += !t.test.valid;
                }
            }
        }
        out.close();
        require(static_cast<bool>(out), "failed writing results");
        std::filesystem::rename(prefix + ".tsv.tmp", prefix + ".tsv");
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        double load[3] = {0, 0, 0};
        getloadavg(load, 3);
        std::ofstream record(prefix + ".run.json");
        require(static_cast<bool>(record), "cannot create run record");
        record << "{\n  \"version\": ";
        io::quoted(record, version);
        record << ",\n  \"command\": ";
        io::quoted(record, command);
#ifdef RAPIDGLM_LAPACK_SOLVES
        record << ",\n  \"lapack_solves\": true";
#else
        record << ",\n  \"lapack_solves\": false";
#endif
        record << ",\n  \"config_path\": ";
        io::quoted(record, argv[2]);
        record << ",\n  \"parameters\": ";
        record << config_text;
        if (command == "bulk-de") {
            record << ",\n  \"test\": ";
            io::quoted(record, test);
            record << std::setprecision(17) << ",\n  \"size_factors\": [";
            for (std::size_t i = 0; i < bulk.size_factors.size(); ++i)
                record << (i ? "," : "") << bulk.size_factors[i];
            record << "],\n  \"trend\": ";
            io::quoted(record, bulk.trend);
            const auto number = [&](const char *name, double v) {
                record << ",\n  \"" << name << "\": ";
                if (std::isfinite(v))
                    record << v;
                else
                    record << "null";
            };
            number("asymptotic_dispersion", bulk.asymptotic_dispersion);
            number("extra_poisson", bulk.extra_poisson);
            number("log_residual_variance", bulk.log_residual_variance);
            number("prior_variance", bulk.prior_variance);
            number("cooks_cutoff", bulk.cooks_cutoff);
            record << ",\n  \"cooks_dispersion\": \""
                   << (config.get<bool>("legacy_cooks", true) ? "legacy_pydeseq2" : "rapidglm")
                   << "\"";
            number("filter_threshold", bulk.filter_threshold);
            number("filter_theta", bulk.filter_theta);
            record << ",\n  \"replaced_genes\": " << bulk.replaced_genes;
            record << ",\n  \"lfc_prior_variance\": [";
            for (std::size_t c = 0; c < bulk.lfc_prior_variance.size(); ++c)
                record << (c ? "," : "") << bulk.lfc_prior_variance[c];
            record << "]";
        }
        if (command == "scan") {
            record << ",\n  \"null_model\": ";
            io::quoted(record, null_model);
            record << ",\n  \"calibration\": ";
            io::quoted(record, calibration_name);
            record << ",\n  \"effect_fits\": ";
            io::quoted(record, effect_fits);
            record << ",\n  \"screen\": " << screen;
        }
        if (command == "sceptre") {
            record << ",\n  \"null_model\": ";
            io::quoted(record, null_model);
            record << ",\n  \"null_fit_method\": ";
            io::quoted(record, null_model == "sceptre" ? "poisson_coefficients_conditional_nb_ml"
                                                       : "direct_glm");
            record << ",\n  \"score_numerator\": ";
            io::quoted(record,
                       null_model == "sceptre" ? "sceptre_uncorrected" : "nuisance_projected");
            record << ",\n  \"effect_fit_method\": \"direct_glm\"";
            record << ",\n  \"effect_fits\": ";
            io::quoted(record, effect_fits);
            record << ",\n  \"fit_check\": ";
            io::quoted(record, config.get<std::string>("fit_check", "strict"));
            record << ",\n  \"confirm_draws\": " << confirm;
            record << ",\n  \"crt_draw\": ";
            io::quoted(record, crt_draw);
            // hazard needs the processor's AES instructions; without them it is thinned.
            record << ",\n  \"crt_draw_used\": ";
            io::quoted(record, crt_draw == "hazard" && !aes_instructions() ? "thinned" : crt_draw);
            if (command == "sceptre") {
                record << ",\n  \"count_matrix\": " << (count_matrix ? "true" : "false");
                record << ",\n  \"reuse_draws\": " << (reuse_draws ? "true" : "false");
                record << ",\n  \"shared_screen\": " << (shared_screen ? "true" : "false");
                record << ",\n  \"sequential_screen\": " << (sequential_screen ? "true" : "false");
                record << ",\n  \"early_stop_p\": " << early_stop_p;
                if (shared_screen)
                    record << ",\n  \"shared_screen_p\": "
                           << config.get<double>("shared_screen_p", 0.05);
                record << ",\n  \"samplers\": {\"cache_mb\": " << sampler_cache_mb
                       << ", \"cached_targets\": " << samplers_cached
                       << ", \"targets_rebuilt_per_pair\": " << samplers_rebuilt << "}";
                record << ",\n  \"stream_keys\": ";
                io::quoted(record, stream_keys);
                record << ",\n  \"response_ids\": "
                       << (response_ids_path.empty() ? "false" : "true");
                record << ",\n  \"adjustment_family\": ";
                io::quoted(record, adjustment_family);
                const auto by_kind = [&](const char *name, const std::array<std::size_t, 3> &n) {
                    record << ",\n  \"" << name << "\": {\"discovery\": " << n[0]
                           << ", \"negative\": " << n[1] << ", \"positive\": " << n[2] << "}";
                };
                by_kind("pairs_listed", pairs_listed);
                by_kind("pairs_passing_qc", pairs_passing_qc);
                by_kind("family_size", family_sizes);
                record << ",\n  \"null_fits\": " << null_fits;
                if (grid_request)
                    record << ",\n  \"grid\": true";
            }
            if (tiled)
                record << ",\n  \"tiles\": {\"count\": " << tiled->tiles
                       << ", \"computed_here\": " << tiled->computed
                       << ", \"computed_seconds\": " << std::setprecision(6)
                       << tiled->computed_seconds << ", \"tile_targets\": " << tiled->tile_targets
                       << ", \"tile_responses\": " << tiled->tile_responses << "}";
            record << ",\n  \"resampling_policy\": "
                   << (screen > 0 ? "\"empirical_screen\"" : "\"rapidglm\"")
                   << ",\n  \"screen\": " << screen;
        }
        if (tested) {
            record << ",\n  \"hypothesis\": ";
            io::hypothesis_record(record, *tested);
        }
        if (command == "pseudobulk-de" || command == "dose-trend") {
            record << ",\n  \"test\": ";
            io::quoted(record, test);
            if (test == "ql") {
                record << ",\n  \"ql_dispersion\": \"cox_reid_adjusted_ml\""
                       << ",\n  \"ql_dispersion_features\": " << quasi.dispersion_features
                       << ",\n  \"ql_trend_window\": " << quasi.trend_window
                       << ",\n  \"ql_abundance_trend\": "
                       << (quasi.abundance_trend ? "true" : "false")
                       << ",\n  \"ql_df_prior\": " << std::setprecision(17) << quasi.df_prior;
            }
        }
        record << ",\n  \"input_sha1\": {";
        bool first = true;
        finish_hashes();
        for (const auto &entry : hashes) {
            if (!first)
                record << ',';
            first = false;
            io::quoted(record, entry.first);
            record << ':';
            io::quoted(record, entry.second);
        }
        record << "}";
        record << ",  \"seed\": " << seed << ",\n  \"threads\": " << threads
               << ",\n  \"observations\": " << responses.rows
               << ",\n  \"features\": " << responses.cols << ",\n  \"retained_cells\": " << retained
               << ",\n  \"invalid_tests_or_fits\": " << failures
               << ",\n  \"calibration_rejections\": " << calibration
               << ",\n  \"power_rejections\": " << power
               << ",\n  \"discovery_rejections\": " << discoveries
               << ",\n  \"wall_seconds\": " << std::setprecision(17) << seconds
               << ",\n  \"peak_rss_kib\": " << usage.ru_maxrss << ",\n  \"load_average\": ["
               << load[0] << ',' << load[1] << ',' << load[2] << "],\n  \"api_frozen\": false\n}\n";
        record.close();
        require(static_cast<bool>(record), "failed writing run record");
        std::cout << prefix << ".tsv (" << failures << " invalid rows; " << seconds << " s)\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "rapidglm: " << e.what() << '\n';
        return 2;
    }
}
