#include <boost/test/unit_test.hpp>
#include <rapidglm/rapidglm.hpp>
#include <set>
using namespace rapidglm;

BOOST_AUTO_TEST_CASE(permutation_scratch_preserves_sets_and_rng_across_word_boundaries) {
    SamplerScratch scratch;
    for (std::size_t n : {0u, 1u, 63u, 64u, 65u, 193u, 1025u, 17u}) {
        Vector labels(n);
        std::vector<std::size_t> strata(n);
        std::map<std::size_t, std::pair<std::vector<std::size_t>, std::size_t>> groups;
        for (std::size_t i = 0; i < n; ++i) {
            strata[i] = i < n / 2 ? 0 : i % 5 + 1;
            // Include empty, full and partially treated strata.
            labels[i] = strata[i] == 1 || (strata[i] != 2 && i % 3 == 0);
            groups[strata[i]].first.push_back(i);
            groups[strata[i]].second += labels[i] != 0;
        }
        auto sampler = permutation_sampler(labels, strata);
        for (std::size_t draw = 0; draw < 100; ++draw) {
            CounterRng actual_rng(781, 19, draw), expected_rng(781, 19, draw);
            std::vector<std::size_t> actual, expected;
            sampler(actual_rng, actual, scratch);
            // Tree-set reference follows the same Floyd draws, independently
            // of bitmap membership and its reused storage.
            for (const auto &group : groups) {
                const auto &indices = group.second.first;
                const auto k = group.second.second;
                std::set<std::size_t> picked;
                for (auto j = indices.size() - k; j < indices.size(); ++j) {
                    auto t = static_cast<std::size_t>(expected_rng.bounded(j + 1));
                    if (!picked.insert(t).second)
                        picked.insert(j);
                }
                for (auto i : picked)
                    expected.push_back(indices[i]);
            }
            std::sort(expected.begin(), expected.end());
            BOOST_TEST(actual == expected, boost::test_tools::per_element());
            BOOST_TEST(actual_rng.next64() == expected_rng.next64());
            BOOST_TEST(scratch.membership.none());
        }
    }
}

BOOST_AUTO_TEST_CASE(prepared_scores_share_only_immutable_state) {
    Matrix x(193, 2, 1);
    Vector y(193), g(193), probability(193);
    for (std::size_t i = 0; i < y.size(); ++i) {
        x(i, 1) = std::sin(i * 0.1);
        y[i] = (i % 7) * 2;
        g[i] = i % 3 == 0;
        probability[i] = 0.15 + (i % 5) * 0.1;
    }
    for (auto fit : {rapidglm::fit(x, y), fit_sceptre_null(x, y)}) {
        BOOST_REQUIRE(fit.ok());
        ResamplingWorkspace prepared(x, fit);
        for (const auto &sampler : {crt_sampler(probability), permutation_sampler(g)}) {
            ResampleOptions options;
            options.budgets = {99};
            options.approximate = false;
            options.retain_null = true;
            const auto reference = resample(ScoreWorkspace(x, fit), g, sampler, options);
            options.parallel_draws = true;
            for (int threads : {1, 2, 8, 16}) {
                parallel_for(7, threads, [&](auto) {
                    auto actual = resample(prepared, g, sampler, options);
                    // Throw inside worker callbacks; Boost.Test's logging is
                    // deliberately kept outside the parallel region.
                    require(actual.null_statistics == reference.null_statistics &&
                                actual.statistic == reference.statistic &&
                                actual.p_value == reference.p_value,
                            "shared prepared scores changed results");
                });
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(cached_target_and_response_contexts_match_independent_pair_fits) {
    const std::size_t n = 193;
    Matrix design(n, 2, 1);
    SparseCSC responses{n, 2, {0}, {}, {}}, assignments{n, 4, {0}, {}, {}};
    std::vector<Vector> calls(4, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        design(i, 1) = std::cos(i * 0.13);
        calls[0][i] = i % 5 == 0;
        calls[1][i] = i % 7 == 0;
        calls[2][i] = i % 3 == 0;
        calls[3][i] = i % 4 < 2;
    }
    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            responses.row_index.push_back(i);
            responses.values.push_back((i * (j + 1) % 11) * (j + 1));
        }
        responses.column_ptr.push_back(responses.values.size());
    }
    for (const auto &call : calls) {
        for (std::size_t i = 0; i < n; ++i)
            if (call[i] || i == 64) { // Retain an explicit zero at a word boundary.
                assignments.row_index.push_back(i);
                assignments.values.push_back(call[i]);
            }
        assignments.column_ptr.push_back(assignments.values.size());
    }
    std::vector<Guide> guides{
        {"a1", "A", false}, {"a2", "A", false}, {"b", "B", false}, {"nt", "NT", true}};
    std::vector<Pair> pairs{{0, "A"}, {0, "B"}, {1, "A"}, {1, "B"}};
    CellQC qc;
    qc.keep.resize(n);
    SceptreOptions options;
    options.cell_qc = options.assignment_qc = false;
    options.crt = true;
    options.min_nonzero = 1;
    options.resampling.budgets = {99};
    options.resampling.approximate = false;
    options.model.offset.resize(n);
    options.model.weights.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        qc.keep[i] = i % 13 != 0;
        options.model.offset[i] = 0.02 * std::sin(i);
        options.model.weights[i] = i % 29 == 0 ? 0 : 0.5 + i % 3;
    }
    for (bool nt_controls : {false, true})
        for (auto mode : {FitMethod::direct, FitMethod::sceptre_null}) {
            options.nt_controls = nt_controls;
            options.null_model = mode;
            std::vector<ResampleResult> expected;
            for (std::size_t j = 0; j < pairs.size(); ++j) {
                std::vector<std::size_t> cells;
                Vector g;
                for (std::size_t i = 0; i < n; ++i) {
                    const bool treated = j % 2 == 0 ? calls[0][i] || calls[1][i] : calls[2][i];
                    if (qc.keep[i] && (!nt_controls || treated || calls[3][i])) {
                        cells.push_back(i);
                        g.push_back(treated);
                    }
                }
                auto x = subset_rows(design, cells);
                auto y = subset(responses.column(pairs[j].response), cells);
                auto model = options.model;
                model.offset = subset(model.offset, cells);
                model.weights = subset(model.weights, cells);
                auto null =
                    mode == FitMethod::direct ? fit(x, y, model) : fit_sceptre_null(x, y, model);
                auto po = model;
                po.family = Family::binomial;
                po.offset.clear();
                po.weights.clear();
                auto propensity = fit(x, g, po);
                BOOST_REQUIRE(null.ok() && propensity.ok());
                auto ro = options.resampling;
                ro.test_id = j;
                expected.push_back(
                    resample(ScoreWorkspace(x, null), g, crt_sampler(propensity.fitted), ro));
            }
            for (int threads : {1, 16}) {
                options.threads = threads;
                auto actual = sceptre(design, responses, assignments, guides, pairs, qc, options);
                for (std::size_t j = 0; j < pairs.size(); ++j) {
                    BOOST_REQUIRE(actual.tests[j].status == "ok");
                    BOOST_TEST(actual.tests[j].statistic == expected[j].statistic);
                    BOOST_TEST(actual.tests[j].p_value == expected[j].p_value);
                }
            }
        }
}
