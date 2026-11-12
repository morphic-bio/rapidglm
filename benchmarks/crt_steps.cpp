// The steps that made the null draws of conditional-resampling (CRT) tests
// faster, each timed with the library as released: this program calls RapidGLM
// and changes none of it.
//
//   first     the draws as they were first implemented (the code before commit
//             e3a2da3, kept below in namespace first): a thread per response,
//             its pairs tasks of that thread's team; a sampler built for every
//             pair; each draw generated and then scored on its own, 16 draws to
//             a task; random numbers one Philox block at a time; a cell's score
//             terms in three arrays.
//   single    the same schedule and still one draw at a time, with the released
//             sampler, generator and score records.
//   batched   step 1: a pair's draws are generated in batches of 256 and scored
//             together in blocks of cells (detail::BinaryDrawBatch).
//   pipeline  step 2: the pair pipeline (detail::pair_pipeline): all threads on a
//             few pairs of one response at a time, and one sampler per target.
//
// Step 3, the generator's Philox blocks eight at a time with AVX2, is the
// difference between the two executables built from this file.
// rapidglm_crt_steps_portable is compiled with RAPIDGLM_FORCE_PORTABLE, which
// selects the library's portable code throughout. In the timed part that changes
// the generator and, where a sampler is built, the exponentials of its
// probabilities; the scoring of the pair pipeline
// (BinaryScoreWorkspace::statistics_of) and the thinned sampler are the same
// code in both. (crt_draw hazard draws its gaps from the AES generator instead,
// so the comparison of the two executables is for thinned or direct.)
//
// The native pipeline is what sceptre() runs with batch_scoring false. Its
// default from 1.6.0, batch scoring, and the default sampler, hazard, came after
// these steps; both are settings of `rapidglm sceptre` and are timed with it.
//
// Only the null draws are timed. Reading the inputs, cell and pair QC, the null
// fits, the propensity fits, the score records and the observed statistics are
// done first, as sceptre() does them, a chunk of responses at a time. A pair's
// draws are those of sceptre() with the same settings (its stream is keyed by
// the response and the target), so the four modes write the same table byte for
// byte, and `verify` compares it with the records of sceptre() itself.
//
// Usage: rapidglm_crt_steps <first|single|batched|pipeline> config.json
// The configuration uses the keys of `rapidglm sceptre` (docs/CLI.md) for a
// high-MOI screen with complement controls, CRT and union integration, and
//   chunk_responses  responses prepared before their draws are timed (256);
//                    their score records are held in memory together
//   verify           also run sceptre() and compare every pair (false)
// It writes <output>.tsv (one row per pair) and <output>.run.json (the timings).
#include "io.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>

using namespace rapidglm;
namespace io = rapidglm::cli;
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point from, Clock::time_point to = Clock::now()) {
    return std::chrono::duration<double>(to - from).count();
}
[[noreturn]] void fail(const std::string &message) {
    throw std::invalid_argument(message);
}
// The draw code of the first implementation, from this repository's history
// (include/rapidglm/resampling.hpp before commit e3a2da3), so that it can still
// be timed. Its draws are the released ones: the same Philox words in the same
// order, the same candidates and acceptances, and the same sums in the same
// order.
namespace first {
// Philox4x32-10, one block of four words at a time.
class Rng {
    std::uint64_t seed_, block_ = 0;
    std::uint32_t test_, draw_;
    std::array<std::uint32_t, 4> output_{};
    unsigned cursor_ = 4;

  public:
    Rng(std::uint64_t seed, std::uint64_t test, std::uint64_t draw) : seed_(seed) {
        require(test <= UINT32_MAX && draw <= UINT32_MAX, "RNG test/draw ID exceeds 32 bits");
        test_ = static_cast<std::uint32_t>(test);
        draw_ = static_cast<std::uint32_t>(draw);
    }
    std::uint32_t next32() {
        if (cursor_ == 4) {
            require(block_ != UINT64_MAX, "RNG block counter exhausted");
            output_ = CounterRng::philox(
                {static_cast<std::uint32_t>(block_), static_cast<std::uint32_t>(block_ >> 32),
                 draw_, test_},
                {static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(seed_ >> 32)});
            ++block_;
            cursor_ = 0;
        }
        return output_[cursor_++];
    }
    std::uint64_t next64() {
        std::uint64_t a = next32();
        return (a << 32) | next32();
    }
    double uniform() {
        return static_cast<double>(next64() >> 11) * 0x1.0p-53;
    }
};
using Sampler = std::function<void(Rng &, std::vector<std::size_t> &)>;
// Independent Bernoulli(p_i) labels. direct uses one uniform per observation.
// thinned groups observations by q = 2^-b >= p_i > q/2, visits candidates with
// geometric gaps of rate q and accepts each with probability p_i/q.
Sampler crt_sampler(Vector probabilities, CrtDraw method) {
    double expected = 0;
    for (double p : probabilities) {
        require(std::isfinite(p) && p >= 0 && p <= 1, "invalid CRT probability");
        expected += p;
    }
    if (method == CrtDraw::direct || 16 * expected >= probabilities.size())
        return [p = std::move(probabilities)](Rng &rng, std::vector<std::size_t> &out) {
            out.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                if (rng.uniform() < p[i])
                    out.push_back(i);
        };
    struct Level {
        double bound, log_miss;
        std::vector<std::size_t> members; // increasing observation indices
    };
    std::map<int, Level> levels;
    for (std::size_t i = 0; i < probabilities.size(); ++i)
        if (probabilities[i] > 0) {
            int exponent;
            std::frexp(probabilities[i], &exponent); // p in [2^(e-1), 2^e)
            const int b = std::max(0, -exponent);
            auto &level = levels[b];
            level.bound = std::ldexp(1.0, -b);
            level.members.push_back(i);
        }
    std::vector<Level> ordered;
    for (auto &entry : levels) {
        entry.second.log_miss = std::log1p(-entry.second.bound);
        ordered.push_back(std::move(entry.second));
    }
    return [p = std::move(probabilities),
            levels = std::move(ordered)](Rng &rng, std::vector<std::size_t> &out) {
        out.clear();
        for (const auto &level : levels) {
            const auto &members = level.members;
            const auto merged = static_cast<std::ptrdiff_t>(out.size());
            if (level.bound >= 0.25) {
                for (auto i : members)
                    if (rng.uniform() < p[i])
                        out.push_back(i);
            } else
                for (std::size_t next = 0;;) {
                    // Gap before the next candidate: P(gap >= k) = (1 - q)^k.
                    const double gap = std::floor(std::log(1 - rng.uniform()) / level.log_miss);
                    if (!(gap < static_cast<double>(members.size() - next)))
                        break;
                    next += static_cast<std::size_t>(gap);
                    const auto i = members[next++];
                    if (rng.uniform() * level.bound < p[i])
                        out.push_back(i);
                }
            // Each level's picks are increasing; merge keeps the output sorted.
            std::inplace_merge(out.begin(), out.begin() + merged, out.end());
        }
    };
}
// A cell's score terms in three arrays (weighted design row, weight, residual),
// copied from the released records; the sums end as the released score does.
struct Scores {
    const BinaryScoreWorkspace &records;
    Matrix weighted_design;
    Vector weights, residuals;
    explicit Scores(const BinaryScoreWorkspace &r)
        : records(r), weighted_design(r.observations(), r.columns), weights(r.observations()),
          residuals(r.observations()) {
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const double *record = &r.cells[i * r.stride];
            residuals[i] = record[0];
            weights[i] = record[1];
            for (std::size_t j = 0; j < r.columns; ++j)
                weighted_design(i, j) = record[2 + j];
        }
    }
    double statistic(const std::vector<std::size_t> &selected, BinaryScoreScratch &scratch) const {
        auto &c = scratch.cross;
        c.assign(weighted_design.cols, 0);
        double u = 0, v = 0;
        for (std::size_t k = 0; k < selected.size(); ++k) {
            auto i = selected[k];
            require(i < weights.size() && (k == 0 || selected[k - 1] < i),
                    "sampler indices must be sorted, unique and in range");
            u += residuals[i];
            v += weights[i];
            for (std::size_t j = 0; j < c.size(); ++j)
                c[j] += weighted_design(i, j);
        }
        return records.statistic_from(records.finish(u, v, c, scratch));
    }
};
// A task's draws: each with its own generator, generated and then scored.
struct Draws {
    const Scores &scores;
    const Sampler &sampler;
    std::vector<std::size_t> selected;
    BinaryScoreScratch scratch;
    Draws(const Scores &scores, const Sampler &sampler, std::size_t reserve)
        : scores(scores), sampler(sampler) {
        selected.reserve(reserve);
    }
    void operator()(const Stream &stream, std::size_t first, std::size_t count, double *null) {
        for (std::size_t k = 0; k < count; ++k) {
            Rng rng(stream.seed, stream.test, first + k);
            sampler(rng, selected);
            null[k] = scores.statistic(selected, scratch);
        }
    }
};
} // namespace first
// A target's treated cells, its propensity model and, in the pipeline, the
// sampler its pairs share.
struct Target {
    // Positions among the retained cells, ascending.
    std::vector<std::size_t> positions;
    FitStatus propensity_status = FitStatus::numerical_failure;
    Vector propensity_coefficients;
    std::once_flag sampler_ready;
    std::shared_ptr<const Sampler> sampler;
    bool fitted() const {
        return propensity_status == FitStatus::converged ||
               propensity_status == FitStatus::converged_boundary;
    }
};
// A pair ready for its null draws.
struct Test {
    std::size_t task = 0;
    double statistic = rapidglm::nan;
    ResampleOptions resampling;
};
// A response's score records and its pairs that reach the null draws.
struct Response {
    std::unique_ptr<BinaryScoreWorkspace> records;
    std::unique_ptr<first::Scores> first_scores; // `first` only
    std::vector<Test> tests;
};
// `single`: one draw at a time with the released code: the draw's own
// generator, the sampler's per-draw call, then the score of that draw alone.
struct OneDraw {
    const BinaryScoreWorkspace &workspace;
    const Sampler &sampler;
    std::vector<std::size_t> selected;
    SamplerScratch sampler_scratch;
    BinaryScoreScratch score_scratch;
    OneDraw(const BinaryScoreWorkspace &w, const Sampler &s, std::size_t reserve)
        : workspace(w), sampler(s) {
        selected.reserve(reserve);
    }
    void operator()(const Stream &stream, std::size_t first, std::size_t count, double *null) {
        for (std::size_t k = 0; k < count; ++k) {
            CounterRng rng(stream.seed, stream.test, first + k);
            sampler(rng, selected, sampler_scratch);
            null[k] = workspace.statistic(selected, score_scratch);
        }
    }
};
// What sceptre() records of a pair's null draws.
void record(PairResult &r, const ResampleResult &result) {
    if (!result.valid) {
        r.status = "invalid_score";
        return;
    }
    r.status = "ok";
    r.statistic = result.statistic;
    r.p_value = result.p_value;
    r.draws = result.draws;
    r.approximation = result.used_approximation;
}
struct Run {
    const SceptreOptions &o;
    const detail::SceptreSetup &setup;
    const Matrix &design; // rows of the retained cells
    std::vector<std::unique_ptr<Target>> &targets;
    std::vector<PairResult> &results;
    Target &target_of(const Test &test) const {
        return *targets[setup.tasks[test.task].target_context];
    }
    // A target's fitted propensities, from its model's coefficients.
    Vector propensities(const Target &target) const {
        return detail::means(design, target.propensity_coefficients, Vector(design.rows, 0),
                             Family::binomial);
    }
    // The sampler of sceptre(): Bernoulli draws at the fitted propensities.
    Sampler sampler(const Target &target) const {
        return crt_sampler(propensities(target), o.crt_draw);
    }
};
// `first`, `single` and `batched`: the first implementation's schedule.
// Responses are shared out among the threads; a response's pairs, and the chunks
// of a pair's draws (staged_null), are tasks. Each pair builds its own sampler
// (make_sampler); make_batch returns what generates and scores the
// `draws_per_task` draws of a task.
template <class MakeSampler, class MakeBatch>
void thread_per_response(const Run &run, const std::vector<std::shared_ptr<Response>> &responses,
                         std::size_t draws_per_task, MakeSampler make_sampler,
                         MakeBatch make_batch) {
    parallel_for(responses.size(), run.o.threads, [&](std::size_t b) {
        const Response &response = *responses[b];
        task_for(response.tests.size(), run.o.parallel_pairs, [&](std::size_t k) {
            const Test &test = response.tests[k];
            const Target &target = run.target_of(test);
            const auto sampler = make_sampler(target);
            ResampleResult result;
            result.statistic = test.statistic;
            detail::staged_null(
                result, test.resampling,
                [&] { return make_batch(response, sampler, target.positions.size()); },
                draws_per_task);
            record(run.results[test.task], result);
        });
    });
}
// `pipeline`: the library's pair pipeline over the same prepared pairs. A
// target's sampler is built by its first pair and shared by the others.
void pipeline(const Run &run, const std::vector<std::shared_ptr<Response>> &responses) {
#if defined(_OPENMP) && _OPENMP >= 201511
    struct Job {
        const Response *response;
        const Test *test;
        const Sampler *sampler;
    };
    std::vector<std::vector<std::size_t>> batches;
    for (const auto &response : responses) {
        batches.emplace_back(response->tests.size());
        std::iota(batches.back().begin(), batches.back().end(), 0);
    }
    detail::pair_pipeline(
        batches, run.o.threads, run.o.parallel_draws, false,
        [&](std::size_t b) { return responses[b]; },
        [&](std::size_t k, Response &response) {
            const Test &test = response.tests[k];
            Target &target = run.target_of(test);
            std::call_once(target.sampler_ready, [&] {
                target.sampler = std::make_shared<const Sampler>(run.sampler(target));
            });
            return std::make_unique<Job>(Job{&response, &test, target.sampler.get()});
        },
        [](Job &job) {
            return detail::PairDraws{job.response->records.get(),
                                     job.sampler,
                                     &job.test->resampling,
                                     job.test->statistic,
                                     {}};
        },
        [&](Job &job, const ResampleResult &result) {
            record(run.results[job.test->task], result);
        });
#else
    (void)run;
    (void)responses;
    fail("the pipeline mode needs OpenMP 4.5 tasks");
#endif
}
// The keys the harness reads; any other key is refused, since a setting it
// ignored would make its table differ from that of `rapidglm sceptre`.
void check_keys(const io::Tree &config) {
    std::istringstream list(
        "responses assignments design covariates formula guides pairs response_ids qc_total "
        "qc_detected qc_mito output family dispersion theta ridge tolerance max_iterations "
        "null_model division moi control_group resampling_mechanism integration stream_keys "
        "crt_draw side approximation fit_check confirm_draws budgets screen seed cell_qc "
        "assignment_qc min_nonzero qc_lower qc_upper max_mito parallel_draws parallel_pairs "
        "threads chunk_responses verify");
    const std::set<std::string> known{std::istream_iterator<std::string>(list), {}};
    for (const auto &entry : config)
        if (!known.count(entry.first))
            fail("the harness does not support the setting " + entry.first);
}
// The options of `rapidglm sceptre` for the same configuration.
SceptreOptions options_of(const io::Tree &config, int threads) {
    SceptreOptions o;
    o.model = io::fit_options(config);
    const auto dispersion = config.get<std::string>("dispersion", "ml");
    require(dispersion != "shared" && dispersion != "trend",
            "dispersion=shared/trend is not supported by sceptre");
    const auto null_model = config.get<std::string>("null_model", "sceptre");
    require(null_model == "direct" || null_model == "sceptre", "unknown null_model");
    require(null_model != "sceptre" ||
                (dispersion == "ml" && o.model.family == Family::negative_binomial &&
                 o.model.ridge == 0),
            "null_model=sceptre (the default) requires family=nb, dispersion=ml and ridge=0; "
            "set null_model=direct for other response models");
    o.null_model = null_model == "sceptre" ? FitMethod::sceptre_null : FitMethod::direct;
    o.threads = threads;
    require(config.get<std::string>("moi", "low") == "high" &&
                config.get<std::string>("control_group", "complement") == "complement" &&
                config.get<std::string>("resampling_mechanism", "crt") == "crt" &&
                config.get<std::string>("integration", "union") == "union" &&
                config.get<std::string>("stream_keys", "identity") == "identity",
            "the harness times CRT tests of a high-MOI screen: set moi to high, and leave "
            "control_group (complement), resampling_mechanism (crt), integration (union) and "
            "stream_keys (identity) at their values for it");
    o.high_moi = true;
    o.nt_controls = false;
    o.crt = true;
    const auto crt_draw = config.get<std::string>("crt_draw", "hazard");
    require(crt_draw == "thinned" || crt_draw == "direct" || crt_draw == "hazard",
            "crt_draw must be thinned/direct/hazard");
    o.crt_draw = crt_draw == "direct"   ? CrtDraw::direct
                 : crt_draw == "hazard" ? CrtDraw::hazard
                                        : CrtDraw::thinned;
    const auto side = config.get<std::string>("side", "both");
    require(side == "left" || side == "right" || side == "both", "invalid side");
    o.resampling.side = side == "left" ? Side::left : side == "right" ? Side::right : Side::both;
    const auto approximation = config.get<std::string>("approximation", "skew_normal");
    require(approximation == "skew_normal" || approximation == "no_approximation",
            "invalid approximation");
    o.resampling.approximate = approximation == "skew_normal";
    o.resampling.fit_check = io::fit_check(config);
    o.resampling.seed = config.get<std::uint64_t>("seed", 1);
    if (auto budgets = config.get_child_optional("budgets")) {
        o.resampling.budgets.clear();
        for (auto &v : *budgets)
            o.resampling.budgets.push_back(v.second.get_value<std::size_t>());
    }
    o.resampling.screen = config.get<double>("screen", o.resampling.screen);
    o.resampling.confirm = io::confirm_draws(config, o.resampling);
    o.cell_qc = config.get<bool>("cell_qc", true);
    o.assignment_qc = config.get<bool>("assignment_qc", true);
    o.parallel_draws = config.get<bool>("parallel_draws", true);
    o.parallel_pairs = config.get<bool>("parallel_pairs", true);
    o.min_nonzero = config.get<std::size_t>("min_nonzero", 7);
    o.lower_quantile = config.get<double>("qc_lower", 0.01);
    o.upper_quantile = config.get<double>("qc_upper", 0.99);
    o.max_mito = config.get<double>("max_mito", 0.2);
    const auto response_ids = config.get<std::string>("response_ids", "");
    if (!response_ids.empty())
        o.response_ids = io::read_lines(response_ids);
    // No effect estimates: the harness ends with the p-values.
    o.effect_fits = EffectFits::none;
    return o;
}
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}
bool same_record(const PairResult &a, const PairResult &b) {
    return a.response == b.response && a.target == b.target && a.guide == b.guide &&
           a.kind == b.kind && a.status == b.status && a.cells == b.cells &&
           a.nonzero_treatment == b.nonzero_treatment && a.nonzero_control == b.nonzero_control &&
           a.draws == b.draws && a.approximation == b.approximation &&
           same_bits(a.statistic, b.statistic) && same_bits(a.p_value, b.p_value);
}
} // namespace
int main(int argc, char **argv) {
    try {
        const std::string mode = argc == 3 ? argv[1] : "";
        if (mode != "first" && mode != "single" && mode != "batched" && mode != "pipeline") {
            std::cerr << "Usage: rapidglm_crt_steps <first|single|batched|pipeline> config.json\n"
                         "Times the null draws of CRT tests as first implemented, one at a time "
                         "with the released code, in batches, or in the pair pipeline "
                         "(docs/CRT_STEPS.md).\n";
            return 2;
        }
        const auto start = Clock::now();
        io::Tree config;
        auto config_file = io::input(argv[2]);
        boost::property_tree::read_json(config_file, config);
        check_keys(config);
        const vmath::DivisionScope division(
            config.count("division") ? vmath::parse_division(config.get<std::string>("division"))
                                     : DivisionMode::inherit);
        const int threads = config.get<int>("threads", 1);
        require(threads > 0, "threads must be positive");
        const auto chunk_responses = config.get<std::size_t>("chunk_responses", 256);
        require(chunk_responses > 0, "chunk_responses must be positive");
        const std::string prefix = config.get<std::string>("output", "");
        require(!prefix.empty(), "output prefix required");
        const SceptreOptions o = options_of(config, threads);
        require(mode != "first" || o.crt_draw != CrtDraw::hazard,
                "the first implementation drew thinned or direct: set crt_draw for mode first");
        // Inputs, as `rapidglm sceptre` reads them.
        const CountMatrix counts =
            io::read_market_counts(config.get<std::string>("responses"), threads);
        const Matrix design = config.count("design")
                                  ? io::read_matrix(config.get<std::string>("design"))
                                  : io::formula_design(config.get<std::string>("covariates"),
                                                       config.get<std::string>("formula"));
        const auto assignments = io::read_market(config.get<std::string>("assignments"));
        const auto guide_table = io::read_table(config.get<std::string>("guides"));
        const auto pair_table = io::read_table(config.get<std::string>("pairs"));
        require(guide_table[0] == std::vector<std::string>({"id", "target", "non_targeting"}),
                "guides header must be id,target,non_targeting (tab separated)");
        require(pair_table[0] == std::vector<std::string>({"response", "target", "kind"}),
                "pairs header must be response,target,kind (tab separated)");
        std::vector<Guide> guides;
        for (std::size_t i = 1; i < guide_table.size(); ++i) {
            const auto &r = guide_table[i];
            require(r[2] == "0" || r[2] == "1", "non_targeting must be 0/1");
            guides.push_back({r[0], r[1], r[2] == "1"});
        }
        std::vector<Pair> pairs;
        for (std::size_t i = 1; i < pair_table.size(); ++i) {
            const auto &r = pair_table[i];
            require(r[2] == "discovery" || r[2] == "negative" || r[2] == "positive",
                    "unknown pair kind");
            const double index = io::number(r[0]);
            require(index >= 0 && std::floor(index) == index && index < counts.cols,
                    "invalid response index");
            pairs.push_back({static_cast<std::size_t>(index), r[1],
                             r[2] == "negative"   ? PairKind::negative_control
                             : r[2] == "positive" ? PairKind::positive_control
                                                  : PairKind::discovery});
        }
        CellQC qc;
        qc.total_counts = io::read_vector(config.get<std::string>("qc_total", ""));
        qc.detected_features = io::read_vector(config.get<std::string>("qc_detected", ""));
        qc.mitochondrial_fraction = io::read_vector(config.get<std::string>("qc_mito", ""));
        const double load_seconds = seconds(start);

        // The cells, the tasks and the stream keys of sceptre(), and its design
        // and model options for the retained cells.
        const auto setup_start = Clock::now();
        const vmath::DivisionScope division_scope(o.model.division);
        const auto setup = detail::sceptre_setup(design, counts, assignments, guides, pairs, qc, o);
        const auto &cells = setup.cells;
        const auto &tasks = setup.tasks;
        const Matrix common_design = subset_rows(design, cells);
        auto common_options = o.model;
        common_options.retain_observations = true;
        if (common_options.weights.empty())
            common_options.design = prepare_design(common_design, common_options.rank_tolerance);
        // Every target's treated cells and propensity model, once.
        std::vector<std::size_t> first_task(setup.target_contexts, tasks.size());
        for (std::size_t i = 0; i < tasks.size(); ++i)
            if (first_task[tasks[i].target_context] == tasks.size())
                first_task[tasks[i].target_context] = i;
        std::vector<std::unique_ptr<Target>> targets(setup.target_contexts);
        parallel_for(targets.size(), threads, [&](std::size_t t) {
            auto target = std::make_unique<Target>();
            boost::dynamic_bitset<std::uint64_t> treatment(counts.rows);
            for (auto j : tasks[first_task[t]].guides)
                for (auto k = assignments.column_ptr[j]; k < assignments.column_ptr[j + 1]; ++k)
                    if (assignments.values[k])
                        treatment[assignments.row_index[k]] = 1;
            Vector labels(cells.size());
            for (std::size_t i = 0; i < cells.size(); ++i)
                if (treatment[cells[i]]) {
                    target->positions.push_back(i);
                    labels[i] = 1;
                }
            auto po = common_options;
            po.family = Family::binomial;
            po.offset.clear();
            po.weights.clear();
            po.start.clear();
            po.retain_observations = false;
            auto propensity = fit(common_design, labels, po);
            target->propensity_status = propensity.status;
            target->propensity_coefficients = std::move(propensity.coefficients);
            targets[t] = std::move(target);
        });
        std::map<std::size_t, std::vector<std::size_t>> response_tasks;
        for (std::size_t i = 0; i < tasks.size(); ++i)
            response_tasks[tasks[i].pair.response].push_back(i);
        std::vector<std::vector<std::size_t>> batches;
        for (auto &entry : response_tasks)
            batches.push_back(std::move(entry.second));
        std::vector<PairResult> results(tasks.size());
        const Run run{o, setup, common_design, targets, results};
        const double setup_seconds = seconds(setup_start);

        // A response up to its pairs' null draws, in the order of sceptre(): pair
        // QC, the null fit (if a pair passes), the propensity model, the score
        // records and the observed statistics.
        const auto prepare = [&](const std::vector<std::size_t> &batch) {
            auto response = std::make_shared<Response>();
            const Vector y = subset(counts.column(tasks[batch[0]].pair.response), cells);
            std::size_t nonzero = 0;
            for (double v : y)
                nonzero += v != 0;
            std::vector<std::size_t> passing, drawn;
            for (auto i : batch) {
                const auto &task = tasks[i];
                auto &r = results[i];
                r.response = task.pair.response;
                r.target = task.pair.target;
                r.kind = task.pair.kind;
                r.guide = task.label;
                r.cells = cells.size();
                for (auto position : targets[task.target_context]->positions)
                    r.nonzero_treatment += y[position] != 0;
                r.nonzero_control = nonzero - r.nonzero_treatment;
                if (r.nonzero_treatment >= o.min_nonzero && r.nonzero_control >= o.min_nonzero)
                    passing.push_back(i);
                else
                    r.status = "failed_pair_qc";
            }
            if (passing.empty())
                return response;
            const FitResult null = detail::sceptre_fit_null(o, common_design, y, common_options);
            for (auto i : passing) {
                const auto &target = *targets[tasks[i].target_context];
                if (!null.ok() || null.has_boundary())
                    results[i].status = std::string("null_") + status_name(null.status);
                else if (!target.fitted())
                    results[i].status =
                        std::string("propensity_") + status_name(target.propensity_status);
                else
                    drawn.push_back(i);
            }
            if (drawn.empty())
                return response;
            response->records = detail::score_records(common_design, null,
                                                      detail::score_state(common_design, null));
            for (auto i : drawn) {
                const auto &task = tasks[i];
                Test test;
                test.task = i;
                test.statistic = detail::observed_statistic_of(
                    *response->records, targets[task.target_context]->positions);
                if (!std::isfinite(test.statistic)) {
                    results[i].status = "invalid_score";
                    continue;
                }
                const Stream own = keyed_stream(
                    o.resampling.seed, 1, setup.response_keys[task.pair.response], task.target_key);
                test.resampling = o.resampling;
                test.resampling.seed = own.seed;
                test.resampling.test_id = own.test;
                test.resampling.retain_null = false;
                test.resampling.parallel_draws = o.parallel_draws;
                detail::check_budgets(test.resampling);
                response->tests.push_back(std::move(test));
            }
            if (response->tests.empty())
                response->records.reset();
            else if (mode == "first")
                response->first_scores = std::make_unique<first::Scores>(*response->records);
            return response;
        };
        double prepare_seconds = 0, draw_seconds = 0;
        std::size_t drawn_pairs = 0, drawn_responses = 0;
        for (std::size_t begin = 0; begin < batches.size(); begin += chunk_responses) {
            const std::size_t count = std::min(chunk_responses, batches.size() - begin);
            const auto prepare_start = Clock::now();
            std::vector<std::shared_ptr<Response>> prepared(count), responses;
            parallel_for(count, threads,
                         [&](std::size_t q) { prepared[q] = prepare(batches[begin + q]); });
            for (auto &response : prepared)
                if (!response->tests.empty()) {
                    drawn_pairs += response->tests.size();
                    responses.push_back(std::move(response));
                }
            drawn_responses += responses.size();
            const auto draw_start = Clock::now();
            if (mode == "first")
                thread_per_response(
                    run, responses, 16,
                    [&](const Target &target) {
                        return first::crt_sampler(run.propensities(target), o.crt_draw);
                    },
                    [](const Response &response, const first::Sampler &sampler,
                       std::size_t reserve) {
                        return first::Draws(*response.first_scores, sampler, reserve);
                    });
            else if (mode == "single")
                thread_per_response(
                    run, responses, 16, [&](const Target &target) { return run.sampler(target); },
                    [](const Response &response, const Sampler &sampler, std::size_t reserve) {
                        return OneDraw(*response.records, sampler, reserve);
                    });
            else if (mode == "batched")
                thread_per_response(
                    run, responses, 256, [&](const Target &target) { return run.sampler(target); },
                    [](const Response &response, const Sampler &sampler, std::size_t) {
                        return detail::BinaryDrawBatch(*response.records, sampler);
                    });
            else
                pipeline(run, responses);
            const auto draw_stop = Clock::now();
            prepare_seconds += seconds(prepare_start, draw_start);
            draw_seconds += seconds(draw_start, draw_stop);
        }

        bool verified = false;
        double verify_seconds = 0;
        if (config.get<bool>("verify", false)) {
            const auto verify_start = Clock::now();
            const auto reference = sceptre(design, counts, assignments, guides, pairs, qc, o).tests;
            require(reference.size() == results.size(), "sceptre() returned other pairs");
            for (std::size_t i = 0; i < results.size(); ++i)
                if (!same_record(results[i], reference[i]))
                    throw std::runtime_error("the record of pair " + std::to_string(i) +
                                             " (response " + std::to_string(results[i].response) +
                                             ", target " + results[i].target +
                                             ") differs from that of sceptre()");
            verified = true;
            verify_seconds = seconds(verify_start);
        }

        const auto directory = std::filesystem::path(prefix).parent_path();
        if (!directory.empty())
            std::filesystem::create_directories(directory);
        std::size_t tested = 0, null_draws = 0;
        {
            std::ofstream out(prefix + ".tsv");
            require(static_cast<bool>(out), "cannot create result table");
            out << "response\ttarget\tguide\tkind\tstatus\tcells\tnonzero_treatment\t"
                   "nonzero_control\tdraws\tapproximation\tstatistic\tp_value\n";
            for (const auto &r : results) {
                out << r.response << '\t' << r.target << '\t' << r.guide << '\t'
                    << static_cast<int>(r.kind) << '\t' << r.status << '\t' << r.cells << '\t'
                    << r.nonzero_treatment << '\t' << r.nonzero_control << '\t' << r.draws << '\t'
                    << r.approximation << '\t';
                io::value(out, r.statistic);
                out << '\t';
                io::value(out, r.p_value);
                out << '\n';
                tested += r.status == "ok";
                null_draws += r.draws;
            }
            out.close();
            require(static_cast<bool>(out), "cannot write result table");
        }
#ifdef RAPIDGLM_PHILOX_AVX2
        const bool philox_avx2 = detail::philox_avx2_available();
#else
        const bool philox_avx2 = false;
#endif
#ifdef RAPIDGLM_FORCE_PORTABLE
        const char *build = "portable";
#else
        const char *build = "native";
#endif
        std::ofstream record_file(prefix + ".run.json");
        require(static_cast<bool>(record_file), "cannot create run record");
        const auto number = [&](const char *key, double v, const char *end = ",\n") {
            record_file << "    \"" << key << "\": ";
            io::value(record_file, v);
            record_file << end;
        };
        record_file << "{\n  \"program\": \"rapidglm_crt_steps\",\n  \"rapidglm\": \"" << version
                    << "\",\n  \"mode\": \"" << mode << "\",\n  \"build\": \"" << build
                    << "\",\n  \"philox_avx2\": " << (philox_avx2 ? "true" : "false")
                    << ",\n  \"aes_instructions\": " << (aes_instructions() ? "true" : "false")
                    << ",\n  \"crt_draw\": \"" << config.get<std::string>("crt_draw", "hazard")
                    << "\",\n  \"threads\": " << threads
                    << ",\n  \"chunk_responses\": " << chunk_responses
                    << ",\n  \"retained_cells\": " << cells.size()
                    << ",\n  \"targets\": " << targets.size()
                    << ",\n  \"responses\": " << batches.size()
                    << ",\n  \"responses_drawn\": " << drawn_responses
                    << ",\n  \"pairs\": " << results.size()
                    << ",\n  \"pairs_drawn\": " << drawn_pairs
                    << ",\n  \"pairs_tested\": " << tested << ",\n  \"null_draws\": " << null_draws
                    << ",\n  \"verified_against_sceptre\": " << (verified ? "true" : "false")
                    << ",\n  \"seconds\": {\n";
        number("load", load_seconds);
        number("setup", setup_seconds);
        number("prepare", prepare_seconds);
        number("draws", draw_seconds);
        number("verify", verify_seconds);
        number("total", seconds(start), "\n");
        record_file << "  }\n}\n";
        record_file.close();
        require(static_cast<bool>(record_file), "cannot write run record");
        std::cout << mode << ' ' << build << ": " << null_draws << " null draws of " << tested
                  << " pairs in " << draw_seconds << " s\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "rapidglm_crt_steps: " << e.what() << '\n';
        return 1;
    }
}
