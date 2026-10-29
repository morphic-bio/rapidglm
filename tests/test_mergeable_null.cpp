// The refine start of NT-control pair nulls (SceptreOptions::null_start,
// mergeable_null.hpp): mergeable totals, label invariance, agreement with the
// cold fit to the convergence tolerance (with the cold fallback exercised), and
// thread-count and pair-order invariance.
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <map>
#include <random>
#include <rapidglm/rapidglm.hpp>
using namespace rapidglm;

namespace {
struct Screen {
    Matrix design;
    Vector offset, weights;
    SparseCSC responses, assignments;
    std::vector<Guide> guides;
    std::vector<Pair> pairs;
    std::vector<int> label; // -1, -2: NT guides; k: target k
};
// Cells of two NT guides and of targets of the given sizes, in shuffled order.
// Design: intercept, a continuous covariate, a nearly separable indicator (60
// cells where every gene's mean is about 8% of the others', mostly zeros) and a
// three-level factor. NB counts; each target lowers a third of the genes.
Screen simulate(std::size_t nt1, std::size_t nt2, const std::vector<std::size_t> &sizes,
                std::size_t genes, unsigned seed, bool offsets, bool weights = false) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;
    Screen s;
    s.label.insert(s.label.end(), nt1, -1);
    s.label.insert(s.label.end(), nt2, -2);
    for (std::size_t k = 0; k < sizes.size(); ++k)
        s.label.insert(s.label.end(), sizes[k], static_cast<int>(k));
    std::shuffle(s.label.begin(), s.label.end(), rng);
    const std::size_t n = s.label.size();
    s.design = Matrix(n, 5);
    s.offset.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        s.design(i, 0) = 1;
        s.design(i, 1) = normal(rng);
        s.design(i, 2) = i % 23 == 0 && i / 23 < 60 ? 1 : 0;
        s.design(i, 3) = i % 3 == 1;
        s.design(i, 4) = i % 3 == 2;
        s.offset[i] = offsets ? 0.3 * normal(rng) : 0;
        if (weights)
            s.weights.push_back(i % 17 == 0 ? 0 : 0.5 + (i % 4) * 0.25);
    }
    s.responses = SparseCSC{n, genes, {0}, {}, {}};
    for (std::size_t g = 0; g < genes; ++g) {
        const double level = -1 + 2.5 * uniform(rng), slope = 0.4 * normal(rng),
                     b1 = 0.2 * normal(rng), b2 = 0.2 * normal(rng), theta = 1 + 20 * uniform(rng);
        for (std::size_t i = 0; i < n; ++i) {
            double eta = level + slope * s.design(i, 1) - 2.5 * s.design(i, 2) +
                         b1 * s.design(i, 3) + b2 * s.design(i, 4) + s.offset[i];
            if (s.label[i] >= 0 && (g + static_cast<std::size_t>(s.label[i])) % 3 == 0)
                eta -= 0.8;
            std::gamma_distribution<double> gamma(theta, std::exp(eta) / theta);
            std::poisson_distribution<int> poisson(std::max(1e-12, gamma(rng)));
            if (const int y = poisson(rng)) {
                s.responses.row_index.push_back(i);
                s.responses.values.push_back(y);
            }
        }
        s.responses.column_ptr.push_back(s.responses.values.size());
    }
    const std::size_t guides = sizes.size() + 2;
    s.assignments = SparseCSC{n, guides, {0}, {}, {}};
    for (std::size_t j = 0; j < guides; ++j) {
        const int wanted = j < sizes.size() ? static_cast<int>(j) : (j == sizes.size() ? -1 : -2);
        for (std::size_t i = 0; i < n; ++i)
            if (s.label[i] == wanted) {
                s.assignments.row_index.push_back(i);
                s.assignments.values.push_back(1);
            }
        s.assignments.column_ptr.push_back(s.assignments.values.size());
    }
    for (std::size_t k = 0; k < sizes.size(); ++k)
        s.guides.push_back({"g" + std::to_string(k), "T" + std::to_string(k), false});
    s.guides.push_back({"nt1", "NT1", true});
    s.guides.push_back({"nt2", "NT2", true});
    for (std::size_t g = 0; g < genes; ++g) {
        for (std::size_t k = 0; k < sizes.size(); ++k)
            s.pairs.push_back({g, "T" + std::to_string(k), PairKind::discovery});
        s.pairs.push_back({g, "NT1", PairKind::negative_control});
    }
    return s;
}
SceptreOptions options(const Screen &s, NullStart start) {
    SceptreOptions o;
    o.cell_qc = false;
    o.min_nonzero = 5;
    o.resampling.budgets = {199, 1999};
    o.model.offset = s.offset;
    o.model.weights = s.weights;
    o.null_start = start;
    return o;
}
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}
// FNV-1a over the records' statuses, draws, statistics and p-values, for
// comparing runs across builds and processors (printed, not asserted). With
// `discrete`, over the statuses, draws, approximation flags and empirical
// p-values only.
std::uint64_t digest(const SceptreResult &r, bool discrete = false) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&](const void *data, std::size_t size) {
        const auto *b = static_cast<const unsigned char *>(data);
        for (std::size_t k = 0; k < size; ++k)
            h = (h ^ b[k]) * 1099511628211ULL;
    };
    for (const auto &t : r.tests) {
        mix(t.status.data(), t.status.size());
        mix(&t.draws, sizeof t.draws);
        mix(&t.approximation, sizeof t.approximation);
        if (!discrete)
            mix(&t.statistic, sizeof t.statistic);
        if (!discrete || !t.approximation)
            mix(&t.p_value, sizeof t.p_value);
    }
    return h;
}
// Pair cells of target k: the NT cells and the target's, ascending.
std::vector<std::size_t> pair_cells(const Screen &s, int k, std::vector<std::size_t> *nt,
                                    std::vector<std::size_t> *block) {
    std::vector<std::size_t> cells;
    for (std::size_t i = 0; i < s.label.size(); ++i) {
        if (s.label[i] < 0 && nt)
            nt->push_back(i);
        if (s.label[i] == k && block)
            block->push_back(i);
        if (s.label[i] < 0 || s.label[i] == k)
            cells.push_back(i);
    }
    return cells;
}
} // namespace

BOOST_AUTO_TEST_CASE(mergeable_totals_are_split_order_and_label_invariant) {
    namespace m = detail::mergeable;
    const auto s = simulate(800, 80, {30, 200}, 3, 77, true);
    const std::size_t n = s.design.rows, p = s.design.cols;
    FitOptions o;
    o.offset = s.offset;
    const m::Rows rows(s.design);
    const Vector no_weights;
    const m::Cells cells{&rows, &o.offset, &no_weights, p};
    std::vector<std::size_t> nt, block;
    const auto pair = pair_cells(s, 1, &nt, &block);
    for (std::size_t g = 0; g < 3; ++g) {
        const Vector y = s.responses.column(g);
        const auto reference = m::reference(fit_sceptre_null(s.design, y, o), cells, nt, y);
        BOOST_REQUIRE(reference.ok);
        const auto canonical =
            m::one_step(reference, m::pair_totals(reference, cells, block, y), o);
        BOOST_REQUIRE(canonical.ok);
        // The fixed summation order gives the same bits every time.
        const auto again = m::one_step(reference, m::pair_totals(reference, cells, block, y), o);
        for (std::size_t j = 0; j < p; ++j)
            BOOST_TEST(same_bits(canonical.beta[j], again.beta[j]));
        BOOST_TEST(same_bits(canonical.theta, again.theta));
        // The pair's totals summed in other splits and orders: the pair's cells
        // reversed in two halves, and relabeled (another set of the same size as
        // the target taken as "treated", the rest as "NT"). The start is the same
        // function of the pair's cells, to summation rounding.
        std::mt19937_64 rng(g);
        auto shuffled = pair;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        const std::vector<std::vector<std::size_t>> orders = {{pair.rbegin(), pair.rend()},
                                                              {shuffled.begin(), shuffled.end()}};
        for (const auto &order : orders) {
            m::Totals first(p), second(p);
            const std::size_t split = block.size(); // a relabeled "target" of the target's size
            for (std::size_t k = 0; k < order.size(); ++k)
                reference.add(cells, order[k], y[order[k]], k < split ? first : second);
            m::mirror(first.information);
            m::mirror(second.information);
            m::Totals merged = second;
            for (std::size_t j = 0; j < p; ++j) {
                merged.score[j] += first.score[j];
                merged.cross[j] += first.cross[j];
            }
            for (std::size_t k = 0; k < p * p; ++k)
                merged.information.values[k] += first.information.values[k];
            merged.theta_score += first.theta_score;
            merged.theta_curvature += first.theta_curvature;
            merged.ylogy += first.ylogy;
            const auto other = m::one_step(reference, merged, o);
            BOOST_REQUIRE(other.ok);
            double worst = std::abs(other.theta - canonical.theta) / canonical.theta;
            for (std::size_t j = 0; j < p; ++j)
                worst = std::max(worst, std::abs(other.beta[j] - canonical.beta[j]) /
                                            (1 + std::abs(canonical.beta[j])));
            BOOST_TEST_MESSAGE("gene " << g << ": other split vs canonical, " << worst);
            BOOST_TEST(worst < 1e-12);
        }
        // Refined, the start reaches the cold fit of the pair's cells.
        auto po = o;
        po.offset = subset(o.offset, pair);
        const auto x = subset_rows(s.design, pair);
        const auto y_pair = subset(y, pair);
        const auto cold = fit_sceptre_null(x, y_pair, po);
        const auto refined = m::refine(cells, pair, y_pair, Vector(pair.size(), 1), canonical, o);
        BOOST_REQUIRE(cold.ok());
        BOOST_REQUIRE(refined.ok);
        for (std::size_t j = 0; j < p; ++j)
            BOOST_TEST(std::abs(refined.fit.coefficients[j] - cold.coefficients[j]) <=
                       1e-8 * (1 + std::abs(cold.coefficients[j])));
        BOOST_TEST(std::abs(refined.fit.theta - cold.theta) <= 1e-6 * cold.theta);
    }
}

BOOST_AUTO_TEST_CASE(mergeable_refine_matches_cold_and_falls_back) {
    for (int variant = 0; variant < 3; ++variant) {
        // Target sizes from 6 cells to 600 (40% of the cells); offsets; prior weights.
        const auto s = simulate(700, 60, {6, 25, 120, 600}, 12, 17 + 894 * variant, variant >= 1,
                                variant == 2);
        const auto cold = sceptre(s.design, s.responses, s.assignments, s.guides, s.pairs, {},
                                  options(s, NullStart::cold));
        const auto warm = sceptre(s.design, s.responses, s.assignments, s.guides, s.pairs, {},
                                  options(s, NullStart::refine));
        BOOST_TEST((cold.null_start_used == NullStart::cold));
        BOOST_TEST((warm.null_start_used == NullStart::refine));
        BOOST_TEST(cold.null_refined == 0u);
        BOOST_TEST(warm.null_refined > 30u);
        // The nearly separable column gives some pairs a boundary: they are fitted cold.
        BOOST_TEST(warm.null_cold_refits > 0u);
        BOOST_TEST(warm.null_fits == cold.null_fits);
        BOOST_REQUIRE_EQUAL(cold.tests.size(), warm.tests.size());
        std::size_t ok = 0, empirical = 0;
        double statistic = 0;
        for (std::size_t k = 0; k < cold.tests.size(); ++k) {
            const auto &a = cold.tests[k], &b = warm.tests[k];
            BOOST_TEST(a.status == b.status);
            if (a.status != "ok" || b.status != "ok")
                continue;
            ++ok;
            statistic = std::max(statistic, std::abs(a.statistic - b.statistic) /
                                                std::max(1e-3, std::abs(a.statistic)));
            BOOST_TEST(a.approximation == b.approximation);
            BOOST_TEST(a.draws == b.draws);
            if (!a.approximation) {
                ++empirical;
                BOOST_TEST(a.p_value == b.p_value);
            } else // a tail fit, continuous in the draws' statistics
                BOOST_TEST(std::abs(a.p_value - b.p_value) <= 1e-4 * a.p_value);
            BOOST_TEST(std::abs(a.sceptre_mean_ratio - b.sceptre_mean_ratio) <=
                       1e-8 * a.sceptre_mean_ratio);
            if (std::isfinite(a.log2fc))
                BOOST_TEST(std::abs(a.log2fc - b.log2fc) <= 1e-6 * (1 + std::abs(a.log2fc)));
        }
        BOOST_TEST_MESSAGE(
            "variant " << variant << ": " << ok << " pairs, " << empirical
                       << " empirical p-values, largest relative statistic "
                       << "difference " << statistic << ", refined " << warm.null_refined
                       << ", cold refits " << warm.null_cold_refits << "; digests cold " << std::hex
                       << digest(cold) << " refine " << digest(warm) << ", discrete cold "
                       << digest(cold, true) << " refine " << digest(warm, true) << std::dec);
        BOOST_TEST(ok > 30u);
        BOOST_TEST(statistic < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(mergeable_refine_does_not_depend_on_threads_or_pair_order) {
    const auto s = simulate(500, 50, {8, 40, 300}, 8, 4242, true);
    auto o = options(s, NullStart::refine);
    const auto one = sceptre(s.design, s.responses, s.assignments, s.guides, s.pairs, {}, o);
    o.threads = 4;
    auto reversed = s.pairs;
    std::reverse(reversed.begin(), reversed.end());
    const auto four = sceptre(s.design, s.responses, s.assignments, s.guides, reversed, {}, o);
    std::map<std::pair<std::size_t, std::string>, const PairResult *> by_pair;
    for (const auto &t : four.tests)
        by_pair[{t.response, t.target}] = &t;
    BOOST_REQUIRE_EQUAL(one.tests.size(), four.tests.size());
    std::size_t differ = 0;
    for (const auto &x : one.tests) {
        const auto &y = *by_pair.at({x.response, x.target});
        differ += x.status != y.status || !same_bits(x.statistic, y.statistic) ||
                  !same_bits(x.p_value, y.p_value) || !same_bits(x.log2fc, y.log2fc);
    }
    BOOST_TEST(differ == 0u);
    BOOST_TEST(one.null_refined == four.null_refined);
    BOOST_TEST(one.null_cold_refits == four.null_cold_refits);
}

BOOST_AUTO_TEST_CASE(mergeable_refine_with_crt_and_all_effect_fits) {
    // The refined null with the paths that also need the pair's design rows.
    const auto s = simulate(400, 40, {20, 150}, 5, 31, true);
    for (bool crt : {false, true}) {
        auto o = options(s, NullStart::cold);
        o.crt = crt;
        o.effect_fits = EffectFits::all;
        const auto cold = sceptre(s.design, s.responses, s.assignments, s.guides, s.pairs, {}, o);
        o.null_start = NullStart::refine;
        const auto warm = sceptre(s.design, s.responses, s.assignments, s.guides, s.pairs, {}, o);
        BOOST_TEST(warm.null_refined > 0u);
        BOOST_REQUIRE_EQUAL(cold.tests.size(), warm.tests.size());
        for (std::size_t k = 0; k < cold.tests.size(); ++k) {
            const auto &a = cold.tests[k], &b = warm.tests[k];
            BOOST_TEST(a.status == b.status);
            if (a.status != "ok")
                continue;
            BOOST_TEST(std::abs(a.statistic - b.statistic) <= 1e-6 * (1 + std::abs(a.statistic)));
            if (!a.approximation)
                BOOST_TEST(a.p_value == b.p_value);
            if (std::isfinite(a.log2fc))
                BOOST_TEST(std::abs(a.log2fc - b.log2fc) <= 1e-6 * (1 + std::abs(a.log2fc)));
        }
    }
}
