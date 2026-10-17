// The compact count matrix and its MatrixMarket loader against the general
// CSC path (read_market, then CountMatrix::from).
#include "../src/io.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdio>
#include <map>
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

BOOST_AUTO_TEST_CASE(market_reader_is_independent_of_entry_order) {
    // The CSC matrix built directly: entries summed per (feature, cell), zeros dropped.
    const std::size_t features = 40, cells = 300;
    auto entries = table(features, cells);
    for (std::size_t k = 0; k < 300; ++k) // repeated entries, to be summed
        entries.push_back({entries[k * 13][0], entries[k * 13][1], k % 2 ? entries[k * 13][2] : 3});
    std::map<std::pair<std::uint64_t, std::uint64_t>, double> sums;
    for (const auto &e : entries)
        sums[{e[0] - 1, e[1] - 1}] += static_cast<double>(e[2]);
    SparseCSC expected{cells, features, {0}, {}, {}};
    for (std::size_t f = 0; f < features; ++f) {
        for (const auto &[key, value] : sums)
            if (key.first == f && value != 0) {
                expected.row_index.push_back(key.second);
                expected.values.push_back(value);
            }
        expected.column_ptr.push_back(expected.values.size());
    }
    const auto check = [&](std::vector<std::array<std::uint64_t, 3>> order) {
        const TemporaryFile file(market(order, features, cells));
        const auto m = io::read_market(file.path);
        BOOST_TEST(m.rows == cells);
        BOOST_TEST(m.cols == features);
        BOOST_TEST(m.column_ptr == expected.column_ptr);
        BOOST_TEST(m.row_index == expected.row_index);
        BOOST_TEST(m.values == expected.values);
    };
    // In feature order (read straight into the matrix), in cell order and shuffled.
    auto by_feature = entries;
    std::stable_sort(by_feature.begin(), by_feature.end(), [](const auto &a, const auto &b) {
        return std::tie(a[0], a[1]) < std::tie(b[0], b[1]);
    });
    check(by_feature);
    auto by_cell = entries;
    std::stable_sort(by_cell.begin(), by_cell.end(), [](const auto &a, const auto &b) {
        return std::tie(a[1], a[0]) < std::tie(b[1], b[0]);
    });
    check(by_cell);
    auto shuffled = entries;
    for (std::size_t k = 0; k < shuffled.size(); ++k)
        std::swap(shuffled[k], shuffled[(k * 7919 + 13) % shuffled.size()]);
    check(shuffled);
    // A feature-ordered file that falls out of order only at its last entry.
    auto late = by_feature;
    const auto middle = late.begin() + static_cast<std::ptrdiff_t>(late.size() / 2);
    std::rotate(middle, middle + 1, late.end());
    check(late);
}

BOOST_AUTO_TEST_CASE(sha1_matches_published_digests) {
    // FIPS 180-2 examples, whole and in uneven pieces (both compression paths
    // where the processor has the SHA extensions: blocks straight from the input
    // and through the buffer).
    const auto digest = [](const std::string &text, std::size_t piece) {
        io::Sha1 hash;
        for (std::size_t at = 0; at < text.size(); at += piece)
            hash.update(text.data() + at, std::min(piece, text.size() - at));
        return hash.hex();
    };
    const std::string million(1000000, 'a');
    for (std::size_t piece : {1u, 7u, 64u, 100u, 65536u, 1000000u}) {
        BOOST_TEST(digest("abc", piece) == "a9993e364706816aba3e25717850c26c9cd0d89d");
        BOOST_TEST(digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", piece) ==
                   "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
        BOOST_TEST(digest(million, piece) == "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }
    BOOST_TEST(digest("", 1) == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    const TemporaryFile file(million);
    BOOST_TEST(io::file_sha1(file.path) == "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
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
