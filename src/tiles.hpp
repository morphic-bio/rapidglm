#pragma once
// Tiled, checkpointed sceptre runs for the command-line program (docs/CLI.md,
// "Tiles and checkpoints"). A run is split into tiles of target contexts x
// responses. Each finished tile is written to the checkpoint directory, so a
// stopped run resumes where it left off and tiles can run on separate workers;
// the merge then adjusts, summarizes and fits deferred effects exactly as a
// single run would.
#include "io.hpp"
#include <cerrno>
#include <cstdlib>
#include <optional>
namespace rapidglm::cli {
inline void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::invalid_argument(message);
}
struct TileRun {
    std::string directory, phase = "all";
    std::size_t tile = 0, tile_targets = 0, tile_responses = 0;
    // The tile size was not given: targets are split into blocks whose CRT
    // samplers fit SceptreOptions::sampler_cache_bytes when they do not all
    // fit, and later phases take the size from the plan (the budget may differ
    // between the machines that run them).
    bool automatic = false;
    std::string fingerprint;
    // Filled in: tiles of the run, those computed by this process, the
    // targets (summed over those tiles) that rebuilt their sampler per pair,
    // and the null models this process fitted (response nulls and the tiles').
    std::size_t tiles = 0, computed = 0, samplers_rebuilt = 0, null_fits = 0;
    double computed_seconds = 0;
};
namespace tiles {
inline std::string tile_path(const TileRun &t, std::size_t k) {
    std::ostringstream name;
    name << t.directory << "/tile_" << std::setw(6) << std::setfill('0') << k << ".tsv";
    return name.str();
}
// Writes through a temporary file that is renamed into place when complete, so
// a file under the final name is always whole.
template <class Write> void write_atomically(const std::string &path, Write write) {
    const auto temporary = path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        require(static_cast<bool>(out), "cannot create " + temporary);
        write(out);
        out.flush();
        require(static_cast<bool>(out), "failed writing " + temporary);
    }
    std::filesystem::rename(temporary, path);
}
// The fingerprints of a run's input files, kept in its checkpoint directory with
// each file's size and modification time: a later phase or tile task takes a
// fingerprint from here when the file's size and time are unchanged, instead of
// reading the whole file again to hash it.
struct InputFingerprint {
    std::uint64_t size = 0;
    std::int64_t seconds = 0, nanoseconds = 0;
    std::string sha1;
    bool same_file(const InputFingerprint &other) const {
        return size == other.size && seconds == other.seconds && nanoseconds == other.nanoseconds;
    }
};
// Size and modification time of `path` (no fingerprint), if it can be read.
inline std::optional<InputFingerprint> input_stamp(const std::string &path) {
    struct stat status{};
    if (::stat(path.c_str(), &status) != 0)
        return std::nullopt;
    InputFingerprint stamp;
    stamp.size = static_cast<std::uint64_t>(status.st_size);
    stamp.seconds = static_cast<std::int64_t>(status.st_mtim.tv_sec);
    stamp.nanoseconds = static_cast<std::int64_t>(status.st_mtim.tv_nsec);
    return stamp;
}
inline std::string input_fingerprints_path(const std::string &directory) {
    return directory + "/inputs.tsv";
}
inline std::map<std::string, InputFingerprint>
read_input_fingerprints(const std::string &directory) {
    std::map<std::string, InputFingerprint> known;
    std::ifstream in(input_fingerprints_path(directory));
    std::string line;
    if (!in || !std::getline(in, line) || line != "rapidglm-inputs-1")
        return known;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        InputFingerprint entry;
        std::string path;
        if (fields >> entry.sha1 >> entry.size >> entry.seconds >> entry.nanoseconds &&
            fields.get() == '\t' && std::getline(fields, path) && entry.sha1.size() == 40)
            known[path] = entry;
    }
    return known;
}
inline void write_input_fingerprints(const std::string &directory,
                                     const std::map<std::string, InputFingerprint> &known) {
    std::filesystem::create_directories(directory);
    write_atomically(input_fingerprints_path(directory), [&](std::ostream &out) {
        out << "rapidglm-inputs-1\n";
        for (const auto &entry : known)
            if (entry.first.find_first_of("\t\n") == std::string::npos)
                out << entry.second.sha1 << '\t' << entry.second.size << '\t'
                    << entry.second.seconds << '\t' << entry.second.nanoseconds << '\t'
                    << entry.first << '\n';
    });
}
inline void put_double(std::ostream &out, double v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof v);
}
inline double get_double(std::istream &in) {
    double v;
    in.read(reinterpret_cast<char *>(&v), sizeof v);
    return v;
}
inline void put_size(std::ostream &out, std::uint64_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof v);
}
inline std::uint64_t get_size(std::istream &in) {
    std::uint64_t v = 0;
    in.read(reinterpret_cast<char *>(&v), sizeof v);
    return v;
}
inline void put_vector(std::ostream &out, const Vector &v) {
    put_size(out, v.size());
    out.write(reinterpret_cast<const char *>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(double)));
}
inline Vector get_vector(std::istream &in) {
    Vector v(get_size(in));
    in.read(reinterpret_cast<char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
    return v;
}
// Response null fits: the fields the score state, the rebuild and the effect
// warm starts use, bit for bit; per-observation vectors only where kept.
inline void write_nulls(const std::string &path, const std::vector<FitResult> &nulls) {
    write_atomically(path, [&](std::ostream &out) {
        out << "rapidglm-nulls-1\n";
        std::uint64_t present = 0;
        for (const auto &f : nulls)
            present += f.parameters != 0;
        put_size(out, nulls.size());
        put_size(out, present);
        for (std::size_t j = 0; j < nulls.size(); ++j) {
            const auto &f = nulls[j];
            if (!f.parameters)
                continue;
            put_size(out, j);
            put_size(out, static_cast<std::uint64_t>(f.family));
            put_size(out, static_cast<std::uint64_t>(f.method));
            put_size(out, static_cast<std::uint64_t>(f.status));
            put_size(out, f.parameters);
            put_double(out, f.theta);
            put_double(out, f.scale);
            put_double(out, f.ridge);
            put_size(out, f.theta_boundary);
            put_vector(out, f.coefficients);
            put_size(out, f.boundary_coefficients.size());
            out.write(reinterpret_cast<const char *>(f.boundary_coefficients.data()),
                      static_cast<std::streamsize>(f.boundary_coefficients.size()));
            put_vector(out, f.fitted);
            put_vector(out, f.working_weights);
            put_vector(out, f.score_residuals);
        }
    });
}
inline std::vector<FitResult> read_nulls(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    require(static_cast<bool>(in), "cannot read " + path);
    std::string magic;
    std::getline(in, magic);
    require(magic == "rapidglm-nulls-1", path + " is not a RapidGLM null file");
    std::vector<FitResult> nulls(get_size(in));
    const auto present = get_size(in);
    for (std::uint64_t k = 0; k < present; ++k) {
        const auto j = get_size(in);
        require(j < nulls.size(), "corrupt null file " + path);
        auto &f = nulls[j];
        f.family = static_cast<Family>(get_size(in));
        f.method = static_cast<FitMethod>(get_size(in));
        f.status = static_cast<FitStatus>(get_size(in));
        f.parameters = get_size(in);
        f.theta = get_double(in);
        f.scale = get_double(in);
        f.ridge = get_double(in);
        f.theta_boundary = get_size(in) != 0;
        f.coefficients = get_vector(in);
        f.boundary_coefficients.resize(get_size(in));
        in.read(reinterpret_cast<char *>(f.boundary_coefficients.data()),
                static_cast<std::streamsize>(f.boundary_coefficients.size()));
        f.fitted = get_vector(in);
        f.working_weights = get_vector(in);
        f.score_residuals = get_vector(in);
    }
    require(static_cast<bool>(in), "corrupt null file " + path);
    return nulls;
}
// Tile records: one row per task, doubles in hexadecimal so they round-trip
// exactly.
inline void write_tile(const std::string &path, const std::vector<std::size_t> &tasks,
                       const std::vector<PairResult> &records, double seconds, int threads) {
    write_atomically(path, [&](std::ostream &out) {
        out << "# rapidglm-tile-1 tasks=" << tasks.size() << " seconds=" << std::setprecision(6)
            << seconds << " threads=" << threads << '\n';
        out << std::hexfloat;
        for (std::size_t k = 0; k < tasks.size(); ++k) {
            const auto &r = records[k];
            out << tasks[k] << '\t' << r.response << '\t' << r.target << '\t' << r.guide << '\t'
                << static_cast<int>(r.kind) << '\t' << r.status << '\t' << r.cells << '\t'
                << r.nonzero_treatment << '\t' << r.nonzero_control << '\t' << r.draws << '\t'
                << r.approximation << '\t' << r.statistic << '\t' << r.p_value << '\t' << r.log2fc
                << '\t' << r.log2fc_se << '\t' << r.sceptre_mean_ratio << '\t' << r.mean_ratio_log2
                << '\n';
        }
    });
}
inline double hex_number(const std::string &s) {
    errno = 0;
    char *end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    require(end && *end == '\0' && errno != ERANGE, "corrupt tile value " + s);
    return v;
}
inline std::size_t read_tile(const std::string &path, std::vector<PairResult> &records,
                             std::vector<unsigned char> &seen) {
    auto in = input(path);
    std::string line;
    std::size_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::vector<std::string> f;
        std::size_t begin = 0;
        for (std::size_t i = 0; i <= line.size(); ++i)
            if (i == line.size() || line[i] == '\t') {
                f.push_back(line.substr(begin, i - begin));
                begin = i + 1;
            }
        require(f.size() == 17, "corrupt tile row in " + path);
        const auto task = static_cast<std::size_t>(std::stoull(f[0]));
        require(task < records.size() && !seen[task], "tile task out of range or repeated");
        seen[task] = 1;
        auto &r = records[task];
        r.response = std::stoull(f[1]);
        r.target = f[2];
        r.guide = f[3];
        r.kind = static_cast<PairKind>(std::stoi(f[4]));
        r.status = f[5];
        r.cells = std::stoull(f[6]);
        r.nonzero_treatment = std::stoull(f[7]);
        r.nonzero_control = std::stoull(f[8]);
        r.draws = std::stoull(f[9]);
        r.approximation = f[10] == "1";
        r.statistic = hex_number(f[11]);
        r.p_value = hex_number(f[12]);
        r.log2fc = hex_number(f[13]);
        r.log2fc_se = hex_number(f[14]);
        r.sceptre_mean_ratio = hex_number(f[15]);
        r.mean_ratio_log2 = hex_number(f[16]);
        ++rows;
    }
    return rows;
}
} // namespace tiles
// Runs `phase` of a tiled run: prepare (layout and response nulls), one tile,
// the merge, or all of them, resuming from the tiles already written. Returns
// the full result for "all" and "merge", nothing for "prepare" and "tile".
template <class Responses>
inline std::optional<SceptreResult>
sceptre_tiled(TileRun &t, const Matrix &design, const Responses &responses,
              const SparseCSC &assignments, const std::vector<Guide> &guides,
              const std::vector<Pair> &pairs, const CellQC &qc, const SceptreOptions &o) {
    std::filesystem::create_directories(t.directory);
    const auto layout = sceptre_layout(design, responses, assignments, guides, pairs, qc, o);
    const auto task_count = layout.task_response.size();
    std::vector<std::size_t> used;
    for (auto j : layout.task_response)
        used.push_back(j);
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    std::vector<std::size_t> rank(responses.cols, 0);
    for (std::size_t k = 0; k < used.size(); ++k)
        rank[used[k]] = k;
    std::size_t tt = t.tile_targets ? t.tile_targets : std::max<std::size_t>(1, layout.targets);
    std::size_t tr = t.tile_responses ? t.tile_responses : std::max<std::size_t>(1, used.size());
    const auto plan_path = t.directory + "/plan.txt";
    if (t.automatic) {
        std::ifstream plan(plan_path);
        std::string line, key;
        std::size_t planned_targets = 0, planned_responses = 0;
        while (std::getline(plan, line)) {
            std::istringstream fields(line);
            std::size_t value = 0;
            if (!(fields >> key >> value))
                continue;
            if (key == "tile_targets")
                planned_targets = value;
            if (key == "tile_responses")
                planned_responses = value;
        }
        const std::size_t sampler = crt_sampler_bytes_per_observation * layout.retained_cells;
        if (planned_targets && planned_responses) {
            tt = planned_targets;
            tr = planned_responses;
        } else if (o.crt && !o.nt_controls && sampler > 0 &&
                   layout.targets > o.sampler_cache_bytes / sampler) {
            // The samplers of all targets do not fit: blocks of targets that do,
            // each across every response, instead of rebuilding samplers per pair.
            tt = std::max<std::size_t>(1, o.sampler_cache_bytes / sampler);
            tr = std::max<std::size_t>(1, used.size());
        }
        t.tile_targets = tt;
        t.tile_responses = tr;
    }
    const std::size_t target_blocks = (layout.targets + tt - 1) / tt;
    const std::size_t response_blocks = (used.size() + tr - 1) / tr;
    t.tiles = target_blocks * response_blocks;
    // The plan fixes the tiling and the inputs; a directory from another run is refused.
    std::ostringstream plan_text;
    plan_text << "rapidglm-tiles-1\nfingerprint " << t.fingerprint << "\ntasks " << task_count
              << "\ntile_targets " << tt << "\ntile_responses " << tr << "\ntiles " << t.tiles
              << '\n';
    if (std::filesystem::exists(plan_path)) {
        auto in = input(plan_path);
        std::string existing((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
        require(existing == plan_text.str(),
                "checkpoint_dir " + t.directory +
                    " holds a run with other inputs, options or tiling; use a new directory");
    } else
        tiles::write_atomically(plan_path, [&](std::ostream &out) { out << plan_text.str(); });
    // Tasks of tile k = (target block, response block), target blocks outermost.
    const auto tile_tasks = [&](std::size_t k) {
        const auto tb = k / response_blocks, rb = k % response_blocks;
        std::vector<std::size_t> tasks;
        for (std::size_t i = 0; i < task_count; ++i)
            if (layout.task_target[i] / tt == tb && rank[layout.task_response[i]] / tr == rb)
                tasks.push_back(i);
        return tasks;
    };
    // Response nulls (complement controls): fitted once, then rebuilt per tile.
    std::vector<FitResult> nulls;
    const auto nulls_path = t.directory + "/nulls.bin";
    if (!o.nt_controls) {
        if (std::filesystem::exists(nulls_path))
            nulls = tiles::read_nulls(nulls_path);
        else {
            require(t.phase == "all" || t.phase == "prepare",
                    "run phase prepare first: " + nulls_path + " is missing");
            nulls = sceptre_nulls(design, responses, assignments, guides, pairs, qc, o, used);
            tiles::write_nulls(nulls_path, nulls);
            t.null_fits += used.size();
        }
        require(nulls.size() == responses.cols, "null file does not match the responses");
    }
    if (t.phase == "prepare")
        return std::nullopt;
    SceptreShard shard;
    shard.nulls = o.nt_controls ? nullptr : &nulls;
    const auto run_tile = [&](std::size_t k) {
        const auto path = tiles::tile_path(t, k);
        if (std::filesystem::exists(path))
            return;
        const auto tasks = tile_tasks(k);
        const auto begin = std::chrono::steady_clock::now();
        std::vector<PairResult> records;
        if (!tasks.empty()) {
            auto s = shard;
            s.tasks = &tasks;
            auto part = sceptre(design, responses, assignments, guides, pairs, qc, o, s);
            records = std::move(part.tests);
            t.samplers_rebuilt += part.samplers_rebuilt;
            t.null_fits += part.null_fits;
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        tiles::write_tile(path, tasks, records, seconds, o.threads);
        ++t.computed;
        t.computed_seconds += seconds;
    };
    if (t.phase == "tile") {
        require(t.tile < t.tiles, "tile index out of range");
        run_tile(t.tile);
        return std::nullopt;
    }
    require(t.phase == "all" || t.phase == "merge", "phase must be all/prepare/tile/merge");
    if (t.phase == "all")
        for (std::size_t k = 0; k < t.tiles; ++k)
            run_tile(k);
    std::vector<PairResult> records(task_count);
    std::vector<unsigned char> seen(task_count, 0);
    std::size_t rows = 0;
    for (std::size_t k = 0; k < t.tiles; ++k) {
        const auto path = tiles::tile_path(t, k);
        require(std::filesystem::exists(path), "tile missing for the merge: " + path);
        rows += tiles::read_tile(path, records, seen);
    }
    require(rows == task_count, "tiles do not cover every task");
    shard.results = &records;
    return sceptre(design, responses, assignments, guides, pairs, qc, o, shard);
}
} // namespace rapidglm::cli
