#pragma once
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <rapidglm/rapidglm.hpp>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif
namespace rapidglm::cli {
using Tree = boost::property_tree::ptree;
inline std::ifstream input(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    return f;
}
// SHA-1 from FIPS 180-4, used only as an input fingerprint in run records.
// Boost's uuids::detail::sha1 is an internal interface without a stable digest type.
class Sha1 {
    std::uint32_t h_[5] = {0x67452301U, 0xEFCDAB89U, 0x98BADCFEU, 0x10325476U, 0xC3D2E1F0U};
    unsigned char block_[64] = {};
    std::size_t used_ = 0;
    std::uint64_t bytes_ = 0;
    static std::uint32_t rotate(std::uint32_t x, unsigned n) {
        return (x << n) | (x >> (32 - n));
    }
    void compress() {
        std::uint32_t w[80];
        for (unsigned t = 0; t < 16; ++t)
            w[t] = std::uint32_t(block_[4 * t]) << 24 | std::uint32_t(block_[4 * t + 1]) << 16 |
                   std::uint32_t(block_[4 * t + 2]) << 8 | std::uint32_t(block_[4 * t + 3]);
        for (unsigned t = 16; t < 80; ++t)
            w[t] = rotate(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
        for (unsigned t = 0; t < 80; ++t) {
            std::uint32_t f, k;
            if (t < 20)
                f = (b & c) | (~b & d), k = 0x5A827999U;
            else if (t < 40)
                f = b ^ c ^ d, k = 0x6ED9EBA1U;
            else if (t < 60)
                f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDCU;
            else
                f = b ^ c ^ d, k = 0xCA62C1D6U;
            std::uint32_t next = rotate(a, 5) + f + e + k + w[t];
            e = d;
            d = c;
            c = rotate(b, 30);
            b = a;
            a = next;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
    }

  public:
    void update(const char *data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            block_[used_++] = static_cast<unsigned char>(data[i]);
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
        bytes_ += size;
    }
    std::string hex() {
        const std::uint64_t bits = bytes_ * 8;
        const char one = static_cast<char>(0x80), zero = 0;
        update(&one, 1);
        while (used_ != 56)
            update(&zero, 1);
        for (int shift = 56; shift >= 0; shift -= 8) {
            const char byte = static_cast<char>((bits >> shift) & 0xFFU);
            update(&byte, 1);
        }
        std::ostringstream out;
        for (auto word : h_)
            out << std::hex << std::setw(8) << std::setfill('0') << word;
        return out.str();
    }
};
inline std::string file_sha1(const std::string &path) {
    auto f = input(path);
    Sha1 hash;
    char buffer[65536];
    while (f.read(buffer, sizeof(buffer)) || f.gcount())
        hash.update(buffer, static_cast<std::size_t>(f.gcount()));
    if (!f.eof())
        throw std::runtime_error("failed hashing " + path);
    return hash.hex();
}
inline Vector read_vector(const std::string &path) {
    if (path.empty())
        return {};
    auto f = input(path);
    Vector v;
    double x;
    while (f >> x)
        v.push_back(x);
    require(f.eof(), "invalid vector token");
    return v;
}
inline Matrix read_matrix(const std::string &path) {
    auto f = input(path);
    std::string line;
    Matrix x;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::istringstream row(line);
        Vector values;
        double v;
        while (row >> v)
            values.push_back(v);
        require(row.eof() && !values.empty(), "invalid design row");
        if (x.cols == 0)
            x.cols = values.size();
        require(values.size() == x.cols, "ragged design");
        x.values.insert(x.values.end(), values.begin(), values.end());
        ++x.rows;
    }
    x.validate();
    return x;
}
inline SparseCSC read_market(const std::string &path) {
    auto f = input(path);
    std::string line;
    std::getline(f, line);
    std::istringstream header(line);
    std::string banner, object, storage, type, symmetry;
    header >> banner >> object >> storage >> type >> symmetry;
    require(banner == "%%MatrixMarket" && object == "matrix" && storage == "coordinate" &&
                (type == "real" || type == "integer" || type == "pattern") && symmetry == "general",
            "expected general real/integer coordinate MatrixMarket");
    do {
        require(static_cast<bool>(std::getline(f, line)), "missing MatrixMarket dimensions");
    } while (line.empty() || line[0] == '%');
    std::size_t features, cells, nnz;
    std::istringstream dimensions(line);
    require(static_cast<bool>(dimensions >> features >> cells >> nnz) && features > 0 && cells > 0,
            "invalid MatrixMarket dimensions");
    std::vector<std::tuple<std::size_t, std::size_t, double>> entries;
    entries.reserve(nnz);
    // The entries are parsed from memory with std::from_chars, which rounds values
    // exactly as the stream does and is much faster on large matrices.
    const std::string body{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    const char *cursor = body.data(), *const end = cursor + body.size();
    const auto skip = [&] {
        while (cursor < end && std::isspace(static_cast<unsigned char>(*cursor)))
            ++cursor;
    };
    const auto parse = [&](auto &value) {
        skip();
        if (cursor < end && *cursor == '+')
            ++cursor;
        const auto result = std::from_chars(cursor, end, value);
        if (result.ec != std::errc() ||
            (result.ptr < end && !std::isspace(static_cast<unsigned char>(*result.ptr))))
            return false;
        cursor = result.ptr;
        return true;
    };
    for (std::size_t i = 0; i < nnz; ++i) {
        std::size_t feature, cell;
        double v = 1;
        require(parse(feature) && parse(cell), "truncated MatrixMarket");
        if (type != "pattern")
            require(parse(v), "truncated MatrixMarket value");
        require(feature > 0 && feature <= features && cell > 0 && cell <= cells && std::isfinite(v),
                "invalid MatrixMarket entry");
        entries.emplace_back(feature - 1, cell - 1, v);
    }
    skip();
    require(cursor == end, "extra MatrixMarket entries");
    if (!std::is_sorted(entries.begin(), entries.end()))
        std::sort(entries.begin(), entries.end());
    SparseCSC out;
    out.rows = cells;
    out.cols = features;
    out.column_ptr = {0};
    std::size_t k = 0;
    for (std::size_t j = 0; j < features; ++j) {
        while (k < entries.size() && std::get<0>(entries[k]) == j) {
            auto row = std::get<1>(entries[k]);
            double v = 0;
            do {
                v += std::get<2>(entries[k++]);
            } while (k < entries.size() && std::get<0>(entries[k]) == j &&
                     std::get<1>(entries[k]) == row);
            if (v != 0) {
                out.row_index.push_back(row);
                out.values.push_back(v);
            }
        }
        out.column_ptr.push_back(out.values.size());
    }
    out.validate();
    return out;
}
// A file mapped read-only, so that its text is parsed in place.
class MappedFile {
    const char *data_ = nullptr;
    std::size_t size_ = 0;

  public:
    explicit MappedFile(const std::string &path) {
        const int descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor < 0)
            throw std::runtime_error("cannot open " + path);
        struct stat status{};
        if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
            ::close(descriptor);
            throw std::runtime_error("cannot map " + path);
        }
        size_ = static_cast<std::size_t>(status.st_size);
        if (size_) {
            void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (mapped == MAP_FAILED) {
                ::close(descriptor);
                throw std::runtime_error("cannot map " + path);
            }
            data_ = static_cast<const char *>(mapped);
        }
        ::close(descriptor);
    }
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    ~MappedFile() {
        if (data_)
            ::munmap(const_cast<char *>(data_), size_);
    }
    const char *begin() const {
        return data_;
    }
    const char *end() const {
        return data_ + size_;
    }
};
// Thrown by read_market_counts for a well-formed file that is not a table of
// counts, one entry per line; read_market reads such a file.
struct NotCountTable : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
// A feature-by-observation MatrixMarket file of counts, read straight into a
// CountMatrix. The text (one entry per line) is split at line ends into one
// piece per thread and parsed once; each thread appends its entries to its own
// list per feature, and each feature's lists are then copied, in file order,
// into that feature's part of the one buffer. A file ordered by observation,
// as writers produce it, therefore needs no sorting: a feature's cells are
// checked to be ascending and sorted only if they are not. Repeated entries
// are summed and zeros dropped, as read_market does.
inline CountMatrix read_market_counts(const std::string &path, int threads = 1) {
    require(threads > 0, "threads must be positive");
    const MappedFile file(path);
    const char *cursor = file.begin(), *const end = file.end();
    const auto line = [&] {
        const char *stop = static_cast<const char *>(
            std::memchr(cursor, '\n', static_cast<std::size_t>(end - cursor)));
        std::string text(cursor, stop ? stop : end);
        cursor = stop ? stop + 1 : end;
        return text;
    };
    require(cursor < end, "empty MatrixMarket file");
    std::istringstream header(line());
    std::string banner, object, storage, type, symmetry;
    header >> banner >> object >> storage >> type >> symmetry;
    require(banner == "%%MatrixMarket" && object == "matrix" && storage == "coordinate" &&
                (type == "real" || type == "integer" || type == "pattern") && symmetry == "general",
            "expected general real/integer coordinate MatrixMarket");
    std::string dimension_line;
    do {
        require(cursor < end, "missing MatrixMarket dimensions");
        dimension_line = line();
    } while (dimension_line.find_first_not_of(" \t\r") == std::string::npos ||
             dimension_line[0] == '%');
    std::size_t features, cells, nnz;
    std::istringstream dimensions(dimension_line);
    require(static_cast<bool>(dimensions >> features >> cells >> nnz) && features > 0 && cells > 0,
            "invalid MatrixMarket dimensions");
    require(cells <= std::numeric_limits<std::uint32_t>::max(), "too many observations for counts");
    const bool pattern = type == "pattern";
    // Pieces of the body, each starting at the beginning of a line.
    const char *const body = cursor;
    const auto pieces = static_cast<std::size_t>(
        std::max<std::ptrdiff_t>(1, std::min<std::ptrdiff_t>(threads, (end - body) / 65536 + 1)));
    std::vector<const char *> edge(pieces + 1, end);
    edge[0] = body;
    for (std::size_t t = 1; t < pieces; ++t) {
        const char *guess = body + (end - body) / static_cast<std::ptrdiff_t>(pieces) *
                                       static_cast<std::ptrdiff_t>(t);
        guess = std::max(guess, edge[t - 1]);
        const char *stop = static_cast<const char *>(
            std::memchr(guess, '\n', static_cast<std::size_t>(end - guess)));
        edge[t] = stop ? stop + 1 : end;
    }
    using Entry = CountMatrix::Entry;
    std::vector<std::vector<std::vector<Entry>>> lists(pieces);
    std::vector<std::size_t> seen(pieces, 0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto guarded = [&](const auto &work) {
        try {
            work();
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure)
                failure = std::current_exception();
        }
    };
    const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    const auto parse_piece = [&](std::size_t t) {
        auto &mine = lists[t];
        mine.resize(features);
        const char *at = edge[t], *const stop = edge[t + 1];
        const auto index = [&](std::size_t limit) {
            while (at < stop && blank(*at))
                ++at;
            std::uint64_t value = 0;
            const auto result = std::from_chars(at, stop, value);
            require(result.ec == std::errc() && value >= 1 && value <= limit &&
                        (result.ptr == stop || blank(*result.ptr) || *result.ptr == '\n'),
                    "invalid MatrixMarket entry");
            at = result.ptr;
            return static_cast<std::size_t>(value - 1);
        };
        while (at < stop) {
            while (at < stop && blank(*at))
                ++at;
            if (at == stop)
                break;
            if (*at == '\n') {
                ++at;
                continue;
            }
            const auto feature = index(features), cell = index(cells);
            std::uint64_t count = 1;
            if (!pattern) {
                while (at < stop && blank(*at))
                    ++at;
                require(at < stop && *at != '\n', "truncated MatrixMarket value");
                if (*at == '+')
                    ++at;
                const auto whole = std::from_chars(at, stop, count);
                const bool integral =
                    whole.ec == std::errc() &&
                    (whole.ptr == stop || blank(*whole.ptr) || *whole.ptr == '\n');
                if (integral)
                    at = whole.ptr;
                else {
                    // A real that holds a whole number, such as 3.0 or 2e1.
                    double value = 0;
                    const auto real = std::from_chars(at, stop, value);
                    require(real.ec == std::errc() &&
                                (real.ptr == stop || blank(*real.ptr) || *real.ptr == '\n'),
                            "invalid MatrixMarket entry");
                    if (!(value >= 0 && value == std::floor(value) &&
                          value <= static_cast<double>(std::numeric_limits<std::uint32_t>::max())))
                        throw NotCountTable("counts must be non-negative integers below 2^32");
                    count = static_cast<std::uint64_t>(value);
                    at = real.ptr;
                }
                if (count > std::numeric_limits<std::uint32_t>::max())
                    throw NotCountTable("counts must be non-negative integers below 2^32");
            }
            while (at < stop && blank(*at))
                ++at;
            if (!(at == stop || *at == '\n'))
                throw NotCountTable("expected one MatrixMarket entry per line");
            ++seen[t];
            if (count)
                mine[feature].push_back(
                    {static_cast<std::uint32_t>(cell), static_cast<std::uint32_t>(count)});
        }
    };
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(pieces)) schedule(static, 1)
#endif
    for (std::int64_t t = 0; t < static_cast<std::int64_t>(pieces); ++t)
        guarded([&] { parse_piece(static_cast<std::size_t>(t)); });
    if (failure)
        std::rethrow_exception(failure);
    std::size_t total_seen = 0;
    for (auto count : seen)
        total_seen += count;
    require(total_seen >= nnz, "truncated MatrixMarket");
    require(total_seen == nnz, "extra MatrixMarket entries");
    CountMatrix out;
    out.rows = cells;
    out.cols = features;
    out.start.assign(features + 1, 0);
    for (std::size_t j = 0; j < features; ++j) {
        std::uint64_t size = 0;
        for (std::size_t t = 0; t < pieces; ++t)
            size += lists[t][j].size();
        out.start[j + 1] = out.start[j] + size;
    }
    out.entries.resize(out.start[features]);
    // Each feature's lists, in file order, into its part of the buffer.
    std::vector<std::uint64_t> kept(features);
    const auto place = [&](std::size_t j) {
        Entry *const first = out.entries.data() + out.start[j];
        Entry *to = first;
        for (std::size_t t = 0; t < pieces; ++t) {
            auto &list = lists[t][j];
            if (!list.empty())
                std::memcpy(to, list.data(), list.size() * sizeof(Entry));
            to += list.size();
            std::vector<Entry>().swap(list);
        }
        bool ascending = true;
        for (Entry *entry = first; entry + 1 < to && ascending; ++entry)
            ascending = entry[0].cell < entry[1].cell;
        if (!ascending) {
            std::stable_sort(first, to,
                             [](const Entry &a, const Entry &b) { return a.cell < b.cell; });
            Entry *last = first;
            for (Entry *entry = first; entry < to; ++entry) {
                if (last != first && last[-1].cell == entry->cell) {
                    const std::uint64_t sum = std::uint64_t(last[-1].count) + entry->count;
                    if (sum > std::numeric_limits<std::uint32_t>::max())
                        throw NotCountTable("counts must be non-negative integers below 2^32");
                    last[-1].count = static_cast<std::uint32_t>(sum);
                } else
                    *last++ = *entry;
            }
            to = last;
        }
        kept[j] = static_cast<std::uint64_t>(to - first);
    };
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
#endif
    for (std::int64_t j = 0; j < static_cast<std::int64_t>(features); ++j)
        guarded([&] { place(static_cast<std::size_t>(j)); });
    if (failure)
        std::rethrow_exception(failure);
    lists.clear();
    // Features that held repeated entries are shorter now: close the gaps.
    std::uint64_t write = 0;
    for (std::size_t j = 0; j < features; ++j) {
        const std::uint64_t read = out.start[j];
        if (write != read && kept[j])
            std::memmove(out.entries.data() + write, out.entries.data() + read,
                         kept[j] * sizeof(Entry));
        out.start[j] = write;
        write += kept[j];
    }
    out.start[features] = write;
    out.entries.resize(write);
#ifdef __GLIBC__
    // The per-thread lists were freed by other threads than their allocators.
    malloc_trim(0);
#endif
    return out;
}
inline std::vector<std::vector<std::string>> read_table(const std::string &path) {
    auto f = input(path);
    std::vector<std::vector<std::string>> table;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::istringstream in(line);
        std::vector<std::string> row;
        std::string cell;
        while (std::getline(in, cell, '\t'))
            row.push_back(cell);
        if (!table.empty())
            require(row.size() == table[0].size(), "ragged TSV");
        table.push_back(row);
    }
    require(!table.empty(), "empty TSV");
    return table;
}
inline std::string trim(std::string s) {
    auto start = s.find_first_not_of(" \t");
    if (start == std::string::npos)
        return {};
    auto end = s.find_last_not_of(" \t");
    return s.substr(start, end - start + 1);
}
inline double number(const std::string &s) {
    std::size_t used = 0;
    double v = std::stod(s, &used);
    require(used == s.size() && std::isfinite(v), "invalid numeric value");
    return v;
}
inline Matrix formula_design(const std::string &path, std::string formula) {
    auto table = read_table(path);
    require(table.size() > 1, "empty covariate table");
    auto header = table.front();
    std::vector<Vector> columns;
    formula = trim(formula);
    require(!formula.empty() && formula[0] == '~', "formula must begin with ~");
    formula = trim(formula.substr(1));
    std::vector<std::string> terms;
    std::istringstream in(formula);
    std::string term;
    bool intercept = true;
    while (std::getline(in, term, '+')) {
        term = trim(term);
        if (term == "0" || term == "-1")
            intercept = false;
        else if (term != "1")
            terms.push_back(term);
    }
    if (intercept)
        columns.emplace_back(table.size() - 1, 1);
    for (auto name : terms) {
        bool log = name.rfind("log(", 0) == 0, factor = name.rfind("factor(", 0) == 0;
        if (log || factor) {
            require(name.back() == ')', "invalid formula term");
            name = name.substr(log ? 4 : 7, name.size() - (log ? 5 : 8));
        }
        auto pos = std::find(header.begin(), header.end(), name);
        require(pos != header.end(),
                "unknown covariate/formula term; supported: name, log(name), factor(name)");
        auto j = static_cast<std::size_t>(pos - header.begin());
        if (factor) {
            std::set<std::string> levels;
            for (std::size_t i = 1; i < table.size(); ++i)
                levels.insert(table[i][j]);
            auto first = levels.begin();
            if (intercept && first != levels.end())
                ++first;
            for (auto it = first; it != levels.end(); ++it) {
                Vector col;
                for (std::size_t i = 1; i < table.size(); ++i)
                    col.push_back(table[i][j] == *it);
                columns.push_back(col);
            }
        } else {
            Vector col;
            for (std::size_t i = 1; i < table.size(); ++i) {
                double v = number(table[i][j]);
                if (log) {
                    require(v > 0, "log covariate must be positive");
                    v = std::log(v);
                }
                col.push_back(v);
            }
            columns.push_back(col);
        }
    }
    Matrix x(table.size() - 1, columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j)
        for (std::size_t i = 0; i < x.rows; ++i)
            x(i, j) = columns[j][i];
    return x;
}
inline Family family(const std::string &name) {
    if (name == "gaussian")
        return Family::gaussian;
    if (name == "poisson")
        return Family::poisson;
    if (name == "binomial")
        return Family::binomial;
    if (name == "negative_binomial" || name == "nb")
        return Family::negative_binomial;
    throw std::invalid_argument("unknown family: " + name);
}
inline FitOptions fit_options(const Tree &c) {
    FitOptions o;
    o.family = family(c.get<std::string>("family", "nb"));
    auto dispersion = c.get<std::string>("dispersion", "ml");
    require(dispersion == "ml" || dispersion == "moment" || dispersion == "fixed" ||
                dispersion == "shared" || dispersion == "trend",
            "unknown dispersion option");
    o.dispersion = dispersion == "moment"  ? Dispersion::moment
                   : dispersion == "fixed" ? Dispersion::fixed
                                           : Dispersion::ml;
    o.theta = c.get<double>("theta", 10);
    o.ridge = c.get<double>("ridge", 0);
    o.tolerance = c.get<double>("tolerance", 1e-9);
    o.max_iterations = c.get<std::size_t>("max_iterations", 100);
    o.robust = c.get<bool>("robust", false);
    o.cox_reid = c.get<bool>("cox_reid", false);
    o.offset = read_vector(c.get<std::string>("offset", ""));
    o.weights = read_vector(c.get<std::string>("weights", ""));
    return o;
}
// %.17g, as the stream would print it, through the faster std::to_chars.
inline void value(std::ostream &out, double v) {
    if (!std::isfinite(v)) {
        out << "NA";
        return;
    }
    char buffer[32];
    const auto end =
        std::to_chars(buffer, buffer + sizeof buffer, v, std::chars_format::general, 17).ptr;
    out.write(buffer, end - buffer);
}
inline void quoted(std::ostream &out, const std::string &s) {
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c == '\n')
            out << "\\n";
        else if (c == '\r')
            out << "\\r";
        else if (c == '\t')
            out << "\\t";
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
        else
            out << c;
    }
    out << '"';
}
inline void fits_table(std::ostream &out, const std::vector<FitResult> &fits) {
    out << "feature\tstatus\ttheta\ttheta_boundary\tlog_likelihood\titerations";
    std::size_t p = fits.empty() ? 0 : fits[0].parameters;
    for (std::size_t j = 0; j < p; ++j)
        out << "\tbeta" << j << "\tse" << j << "\twald_p" << j << "\tboundary" << j;
    out << '\n';
    for (std::size_t i = 0; i < fits.size(); ++i) {
        const auto &f = fits[i];
        out << i << '\t' << status_name(f.status) << '\t';
        value(out, f.theta);
        out << '\t' << f.theta_boundary << '\t';
        value(out, f.log_likelihood);
        out << '\t' << f.iterations;
        for (std::size_t j = 0; j < p; ++j) {
            out << '\t';
            value(out, f.coefficients[j]);
            out << '\t';
            value(out, f.standard_errors[j]);
            out << '\t';
            value(out, wald_test(f, j).p_value);
            out << '\t' << (!f.boundary_coefficients.empty() && f.boundary_coefficients[j]);
        }
        out << '\n';
    }
}
// With quasi-likelihood details, `theta` is the trended value used by the tested
// fits and the per-feature estimates follow in extra columns.
// `coefficient` and `standard_error` are the tested coefficient or contrast
// estimate and its standard error (missing for hypotheses with several rows).
inline void regression_table(std::ostream &out, const std::vector<FeatureTest> &tests,
                             const std::vector<QuasiLikelihoodFeature> *ql = nullptr) {
    out << "feature\tstatus\tstatistic\tp_value\tadjusted_p\tcoefficient\tstandard_"
           "error\tlog2fc\tlog2fc_se\ttheta\tcoefficient_boundary";
    if (ql)
        out << "\ttheta_feature\tmean_fitted\tql_dispersion\tql_prior\tql_posterior\tdf_"
               "residual";
    out << '\n';
    for (std::size_t i = 0; i < tests.size(); ++i) {
        const auto &r = tests[i];
        out << i << '\t' << status_name(r.fit.status);
        for (double v : {r.test.statistic, r.test.p_value, r.adjusted_p, r.estimate,
                         r.standard_error, r.log2fc, r.log2fc_se, r.fit.theta}) {
            out << '\t';
            value(out, v);
        }
        out << '\t' << r.boundary;
        if (ql) {
            const auto &q = (*ql)[i];
            for (double v : {q.dispersion > 0 ? 1 / q.dispersion : nan, q.mean, q.ql, q.prior,
                             q.posterior, q.df_residual}) {
                out << '\t';
                value(out, v);
            }
        }
        out << '\n';
    }
}
// The tested hypothesis: `coefficient` (one column, default the last),
// `coefficients` (columns tested jointly) or `contrast` (one weight per column).
inline Hypothesis hypothesis(const Tree &c, std::size_t p) {
    require(c.count("coefficient") + c.count("coefficients") + c.count("contrast") <= 1,
            "give at most one of coefficient, coefficients and contrast");
    if (auto list = c.get_child_optional("coefficients")) {
        std::vector<std::size_t> columns;
        for (auto &v : *list)
            columns.push_back(v.second.get_value<std::size_t>());
        auto h = Hypothesis::coefficients(p, columns);
        reduced_design(Matrix(1, p), h); // validates the columns
        return h;
    }
    if (auto list = c.get_child_optional("contrast")) {
        Vector weights;
        for (auto &v : *list)
            weights.push_back(v.second.get_value<double>());
        require(weights.size() == p, "contrast needs one weight per design column");
        auto h = Hypothesis::contrast(weights);
        h.null_space(); // validates the weights
        return h;
    }
    const auto column = c.get<std::size_t>("coefficient", p - 1);
    require(column < p, "coefficient out of range");
    return Hypothesis::coefficient(p, column);
}
inline void hypothesis_record(std::ostream &out, const Hypothesis &h) {
    out << std::setprecision(17) << "[";
    for (std::size_t r = 0; r < h.contrasts.rows; ++r) {
        out << (r ? ",[" : "[");
        for (std::size_t j = 0; j < h.contrasts.cols; ++j)
            out << (j ? "," : "") << h.contrasts(r, j);
        out << "]";
    }
    out << "]";
}
// The acceptance check of the skew-normal fit: strict (default) or ks.
inline FitCheck fit_check(const Tree &c) {
    const auto name = c.get<std::string>("fit_check", "strict");
    require(name == "strict" || name == "ks", "fit_check must be strict/ks");
    return name == "ks" ? FitCheck::ks : FitCheck::strict;
}
// Resampling keys shared by scan: budgets, approximation, fit_check, side and seed.
inline ResampleOptions resample_options(const Tree &c, std::uint64_t seed) {
    ResampleOptions o;
    auto side = c.get<std::string>("side", "both");
    require(side == "left" || side == "right" || side == "both", "invalid side");
    o.side = side == "left" ? Side::left : side == "right" ? Side::right : Side::both;
    auto approx = c.get<std::string>("approximation", "skew_normal");
    require(approx == "skew_normal" || approx == "no_approximation", "invalid approximation");
    o.approximate = approx == "skew_normal";
    o.fit_check = fit_check(c);
    o.seed = seed;
    if (auto budgets = c.get_child_optional("budgets")) {
        o.budgets.clear();
        for (auto &v : *budgets)
            o.budgets.push_back(v.second.get_value<std::size_t>());
    }
    o.screen = c.get<double>("screen", 0);
    return o;
}
inline void bulk_table(std::ostream &out, const std::vector<BulkGene> &genes, bool shrunk = false) {
    out << "gene\tstatus\tbaseMean\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj\tdispGeneEst\t"
           "dispFit\tdispMAP\tdispersion\tdispOutlier\tmaxCooks\treplaced"
        << (shrunk ? "\tlog2FoldChangeShrunk\tlfcSEShrunk" : "") << '\n';
    for (std::size_t j = 0; j < genes.size(); ++j) {
        const auto &g = genes[j];
        out << j << '\t' << g.status;
        for (double v :
             {g.base_mean, g.log2_fold_change, g.lfc_se, g.statistic, g.p_value, g.adjusted_p,
              g.dispersion_gene, g.dispersion_trend, g.dispersion_map, g.dispersion}) {
            out << '\t';
            value(out, v);
        }
        out << '\t' << g.dispersion_outlier << '\t';
        value(out, g.max_cooks);
        out << '\t' << g.replaced;
        if (shrunk)
            for (double v : {g.shrunken_lfc, g.shrunken_lfc_se}) {
                out << '\t';
                value(out, v);
            }
        out << '\n';
    }
}
inline void scan_table(std::ostream &out, const std::vector<ScanResult> &results) {
    out << "response\tpredictor\tfamily\tstatus\tstatistic\tp_value\tadjusted_p\testimate\t"
           "standard_error\teffect\teffect_se\tdraws\tapproximation\n";
    for (const auto &r : results) {
        out << r.response << '\t' << r.predictor << '\t' << r.family << '\t' << r.status;
        for (double v : {r.statistic, r.p_value, r.adjusted_p, r.estimate, r.standard_error,
                         r.effect, r.effect_se}) {
            out << '\t';
            value(out, v);
        }
        out << '\t' << r.draws << '\t' << r.approximation << '\n';
    }
}
inline void pair_table(std::ostream &out, const std::vector<PairResult> &tests) {
    out << "response\ttarget\tguide\tkind\tstatus\tcells\tnonzero_treatment\tnonzero_"
           "control\tdraws\tapproximation\tstatistic\tp_value\tadjusted_p\tlog2fc\tlog2fc_"
           "se\tsceptre_mean_ratio\tmean_ratio_log2\n";
    for (const auto &r : tests) {
        out << r.response << '\t' << r.target << '\t' << r.guide << '\t' << static_cast<int>(r.kind)
            << '\t' << r.status << '\t' << r.cells << '\t' << r.nonzero_treatment << '\t'
            << r.nonzero_control << '\t' << r.draws << '\t' << r.approximation;
        for (double v : {r.statistic, r.p_value, r.adjusted_p, r.log2fc, r.log2fc_se,
                         r.sceptre_mean_ratio, r.mean_ratio_log2}) {
            out << '\t';
            value(out, v);
        }
        out << '\n';
    }
}
} // namespace rapidglm::cli
