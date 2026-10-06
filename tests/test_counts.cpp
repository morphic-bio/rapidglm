// The compact count matrix and its MatrixMarket loader against the general
// CSC path (read_market, then CountMatrix::from).
#include "../src/io.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdio>
#include <rapidglm/rapidglm.hpp>

using namespace rapidglm;
namespace io = rapidglm::cli;

namespace {
struct TemporaryFile {
    std::string path;
    explicit TemporaryFile(const std::string &text) {
        path = (std::filesystem::temp_directory_path() /
                ("rapidglm_counts_" + std::to_string(::getpid()) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".mtx"))
                   .string();
        std::ofstream out(path, std::ios::binary);
        out << text;
    }
    ~TemporaryFile() {
        std::remove(path.c_str());
    }
};
bool same(const CountMatrix &a, const CountMatrix &b) {
    if (a.rows != b.rows || a.cols != b.cols || a.start != b.start ||
        a.entries.size() != b.entries.size())
        return false;
    for (std::size_t k = 0; k < a.entries.size(); ++k)
        if (a.entries[k].cell != b.entries[k].cell || a.entries[k].count != b.entries[k].count)
            return false;
    return true;
}
// Entries (feature, cell, count), one-based, of a deterministic sparse table.
std::vector<std::array<std::uint64_t, 3>> table(std::size_t features, std::size_t cells) {
    std::vector<std::array<std::uint64_t, 3>> entries;
    for (std::size_t c = 1; c <= cells; ++c)
        for (std::size_t f = 1; f <= features; ++f) {
            const std::uint64_t h = (c * 2654435761u + f * 40503u) >> 5;
            if (h % 3 == 0)
                continue;
            // Mostly small counts, some above 16 bits, some explicit zeros.
            const std::uint64_t count = h % 97 == 0   ? 70000 + h % 1000
                                        : h % 89 == 0 ? 0
                                                      : 1 + h % 6;
            entries.push_back({f, c, count});
        }
    return entries;
}
std::string market(const std::vector<std::array<std::uint64_t, 3>> &entries, std::size_t features,
                   std::size_t cells, const std::string &type = "integer",
                   const std::string &line_end = "\n") {
    std::string text = "%%MatrixMarket matrix coordinate " + type + " general" + line_end +
                       "% a comment" + line_end + std::to_string(features) + " " +
                       std::to_string(cells) + " " + std::to_string(entries.size()) + line_end;
    for (const auto &e : entries) {
        text += std::to_string(e[0]) + " " + std::to_string(e[1]);
        if (type != "pattern")
            text += " " + std::to_string(e[2]);
        text += line_end;
    }
    return text;
}
} // namespace

BOOST_AUTO_TEST_CASE(count_matrix_converts_from_and_to_csc) {
    SparseCSC csc{4, 3, {0, 2, 2, 5}, {0, 3, 1, 2, 3}, {5, 70000, 1, 0, 2}};
    const auto counts = CountMatrix::from(csc);
    counts.validate();
    BOOST_TEST(counts.entries.size() == 4u); // the explicit zero is dropped
    for (std::size_t j = 0; j < csc.cols; ++j)
        BOOST_TEST(counts.column(j) == csc.column(j));
    std::size_t visited = 0;
    counts.for_each(2, [&](std::size_t cell, double count) {
        BOOST_TEST((cell == 1 || cell == 3));
        BOOST_TEST(count == (cell == 1 ? 1 : 2));
        ++visited;
    });
    BOOST_TEST(visited == 2u);
    const auto back = counts.csc();
    back.validate();
    for (std::size_t j = 0; j < csc.cols; ++j)
        BOOST_TEST(back.column(j) == csc.column(j));
    for (double bad : {1.5, -1.0, 4294967296.0}) {
        auto other = csc;
        other.values[0] = bad;
        BOOST_CHECK_THROW(CountMatrix::from(other), std::invalid_argument);
    }
}

BOOST_AUTO_TEST_CASE(count_loader_equals_the_csc_reader) {
    const std::size_t features = 60, cells = 2000;
    const auto entries = table(features, cells);
    // In cell order, as writers produce it: several pieces, no sorting needed.
    const TemporaryFile sorted(market(entries, features, cells));
    const auto reference = CountMatrix::from(io::read_market(sorted.path));
    reference.validate();
    BOOST_TEST(reference.entries.size() > 50000u);
    for (int threads : {1, 3, 8, 16}) {
        const auto counts = io::read_market_counts(sorted.path, threads);
        counts.validate();
        BOOST_TEST(same(counts, reference));
    }
    // Out of order and with repeated entries: sorted and summed like the reader.
    auto shuffled = entries;
    for (std::size_t k = 0; k < shuffled.size(); ++k)
        std::swap(shuffled[k], shuffled[(k * 7919 + 13) % shuffled.size()]);
    for (std::size_t k = 0; k < 500; ++k)
        shuffled.push_back(entries[k * 37 % entries.size()]);
    const TemporaryFile unordered(market(shuffled, features, cells));
    const auto summed = CountMatrix::from(io::read_market(unordered.path));
    BOOST_TEST(!same(summed, reference));
    for (int threads : {1, 4, 16})
        BOOST_TEST(same(io::read_market_counts(unordered.path, threads), summed));
    // Other spellings of the same counts.
    const TemporaryFile windows(market(entries, features, cells, "integer", "\r\n"));
    BOOST_TEST(same(io::read_market_counts(windows.path, 4), reference));
    std::string real = "%%MatrixMarket matrix coordinate real general\n3 4 4\n\n1 1 3.0\n"
                       "  2 2 +2e1  \n3 4 7\n1 3 0.0\n";
    const TemporaryFile reals(real);
    const auto parsed = io::read_market_counts(reals.path, 2);
    BOOST_TEST(same(parsed, CountMatrix::from(io::read_market(reals.path))));
    BOOST_TEST(parsed.entries.size() == 3u);
    BOOST_TEST(parsed.column(1)[1] == 20);
    const TemporaryFile marks(market(entries, features, cells, "pattern"));
    BOOST_TEST(same(io::read_market_counts(marks.path, 4),
                    CountMatrix::from(io::read_market(marks.path))));
}

BOOST_AUTO_TEST_CASE(count_loader_rejects_what_is_not_a_count_table) {
    const std::string head = "%%MatrixMarket matrix coordinate integer general\n";
    for (const std::string body :
         {"2 2 1\n1 1 -1\n",                         // negative
          "2 2 1\n1 1 1.5\n",                        // fractional
          "2 2 1\n1 1 4294967296\n",                 // above 32 bits
          "2 2 1\n0 1 1\n",                          // feature index below one
          "2 2 1\n1 3 1\n",                          // cell index out of range
          "2 2 2\n1 1 1\n",                          // fewer entries than declared
          "2 2 1\n1 1 1\n2 2 1\n",                   // more entries than declared
          "2 2 2\n1 1 1 2 2 1\n",                    // two entries on a line
          "2 2 1\n1 1\n",                            // missing value
          "2 2 2\n1 1 3000000000\n1 1 3000000000\n", // repeated entries overflow
          ""}) {
        const TemporaryFile file(head + body);
        BOOST_CHECK_THROW(io::read_market_counts(file.path, 2), std::exception);
    }
    const TemporaryFile symmetric(
        "%%MatrixMarket matrix coordinate integer symmetric\n2 2 1\n1 1 1\n");
    BOOST_CHECK_THROW(io::read_market_counts(symmetric.path), std::exception);
    const TemporaryFile empty("");
    BOOST_CHECK_THROW(io::read_market_counts(empty.path), std::exception);
    BOOST_CHECK_THROW(io::read_market_counts("/nonexistent/counts.mtx"), std::exception);
}

BOOST_AUTO_TEST_CASE(count_loader_names_tables_that_are_not_counts) {
    // Well-formed files the general reader accepts are reported as such, so
    // that a caller can read them into the general container instead.
    const std::string head = "%%MatrixMarket matrix coordinate real general\n";
    for (const std::string body :
         {"2 2 1\n1 1 -1\n", "2 2 1\n1 1 1.5\n", "2 2 1\n1 1 4294967296\n", "2 2 2\n1 1 1 2 2 1\n",
          "2 2 2\n1 1 3000000000\n1 1 3000000000\n"}) {
        const TemporaryFile file(head + body);
        BOOST_CHECK_THROW(io::read_market_counts(file.path, 2), io::NotCountTable);
        BOOST_CHECK_NO_THROW(io::read_market(file.path));
    }
    const TemporaryFile outside(head + "2 2 1\n1 3 1\n");
    BOOST_CHECK_EXCEPTION(io::read_market_counts(outside.path), std::exception,
                          [](const std::exception &e) {
                              return dynamic_cast<const io::NotCountTable *>(&e) == nullptr;
                          });
}
