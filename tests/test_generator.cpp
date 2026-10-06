#include "calibration_data.hpp"
#include <boost/test/unit_test.hpp>
#include <rapidglm/sparse.hpp>

using namespace rapidglm;

BOOST_AUTO_TEST_CASE(generator_reproducibility_and_scenario_contract) {
    for (unsigned scenario = 0; scenario < 8; ++scenario) {
        auto a = rapidglm_test::generate_null(scenario, 73);
        auto b = rapidglm_test::generate_null(scenario, 73);
        auto c = rapidglm_test::generate_null(scenario, 74);
        BOOST_TEST(a.design.rows == 256u);
        BOOST_TEST(a.design.cols == 2u);
        BOOST_TEST(a.design.values == b.design.values, boost::test_tools::per_element());
        BOOST_TEST(a.response == b.response, boost::test_tools::per_element());
        BOOST_TEST(a.assignment == b.assignment, boost::test_tools::per_element());
        BOOST_TEST(a.probability == b.probability, boost::test_tools::per_element());
        BOOST_CHECK(a.design.values != c.design.values);
        BOOST_CHECK(a.response != c.response);
        // Each dataset owns both its engine and distribution state. Verify the
        // same dataset IDs across schedules, including cached normal variates.
        std::vector<rapidglm_test::NullData> baseline(32);
        for (int threads : {1, 2, 8, 16}) {
            std::vector<rapidglm_test::NullData> data(32);
            parallel_for(data.size(), threads, [&](std::size_t id) {
                data[id] = rapidglm_test::generate_null(scenario, id, 32);
            });
            if (threads == 1)
                baseline = data;
            for (std::size_t id = 0; id < data.size(); ++id) {
                BOOST_TEST(data[id].design.values == baseline[id].design.values,
                           boost::test_tools::per_element());
                BOOST_TEST(data[id].response == baseline[id].response,
                           boost::test_tools::per_element());
                BOOST_TEST(data[id].assignment == baseline[id].assignment,
                           boost::test_tools::per_element());
            }
        }
        for (unsigned i = 0; i < a.design.rows; ++i) {
            BOOST_TEST(a.design(i, 0) == 1);
            double expected = scenario >= 4 ? 1 / (1 + std::exp(1 - a.design(i, 1) / 2)) : 0.3;
            BOOST_TEST(std::abs(a.probability[i] - expected) < 1e-14);
        }
    }
    BOOST_CHECK_THROW(rapidglm_test::generate_null(8, 0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(generator_conditional_moments_assignment_and_null_independence) {
    // These are seeded generator diagnostics, not G3/G5 acceptance thresholds.
    // Seven-standard-error bounds are fixed before running, across 24 settings.
    // Conditioning on an exact z avoids mistaking confounding for null dependence.
    constexpr unsigned n = 100000;
    for (unsigned scenario = 0; scenario < 8; ++scenario)
        for (double z : {-2.0, 0.0, 2.0}) {
            auto family = static_cast<Family>(scenario % 4);
            rapidglm_test::NullGenerator generator(family, scenario >= 4, 19473 + scenario);
            double eta = 0.5 + 0.4 * z, mean = std::exp(eta), variance, fourth;
            if (family == Family::gaussian) {
                mean = eta;
                variance = 1;
                fourth = 3;
            } else if (family == Family::binomial) {
                mean = 1 / (1 + std::exp(-eta));
                variance = mean * (1 - mean);
                fourth = variance * (1 - 3 * variance);
            } else if (family == Family::poisson) {
                variance = mean;
                fourth = mean + 3 * mean * mean;
            } else {
                // NB2 theta=2: fourth central moment = fourth cumulant + 3 variance^2.
                variance = mean + mean * mean / 2;
                fourth = mean + 7 * mean * mean / 2 + 3 * std::pow(mean, 3) +
                         0.75 * std::pow(mean, 4) + 3 * variance * variance;
            }
            double p = scenario >= 4 ? 1 / (1 + std::exp(1 - z / 2)) : 0.3;
            long double residual = 0, squared = 0, assigned = 0, cross = 0;
            bool support_ok = true, probabilities_ok = true;
            for (unsigned i = 0; i < n; ++i) {
                auto row = generator.observation(z);
                double r = row.response - mean;
                residual += r;
                squared += r * r;
                assigned += row.assignment;
                cross += r * (row.assignment - p);
                support_ok &=
                    std::isfinite(row.response) && (row.assignment == 0 || row.assignment == 1);
                if (family != Family::gaussian)
                    support_ok &= row.response >= 0 && std::floor(row.response) == row.response;
                if (family == Family::binomial)
                    support_ok &= row.response <= 1;
                probabilities_ok &= std::abs(row.probability - p) < 1e-14;
            }
            BOOST_TEST_CONTEXT("scenario=" << scenario << " z=" << z) {
                BOOST_TEST(support_ok);
                BOOST_TEST(probabilities_ok);
                BOOST_TEST(std::abs(static_cast<double>(residual / n)) <=
                           7 * std::sqrt(variance / n));
                BOOST_TEST(std::abs(static_cast<double>(squared / n) - variance) <=
                           7 * std::sqrt((fourth - variance * variance) / n));
                BOOST_TEST(std::abs(static_cast<double>(assigned / n) - p) <=
                           7 * std::sqrt(p * (1 - p) / n));
                BOOST_TEST(std::abs(static_cast<double>(cross / n)) <=
                           7 * std::sqrt(variance * p * (1 - p) / n));
            }
        }
}
