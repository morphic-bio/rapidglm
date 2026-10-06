#pragma once
#include <random>
#include <rapidglm/model.hpp>

// Validation-only E1 null generator. Keep the order of RNG calls unchanged:
// covariate, assignment, response. Gaussian responses share the normal sampler
// (including its cached variate) with covariates, as in the original harness.
namespace rapidglm_test {
struct NullObservation {
    double response, assignment, probability;
};
struct NullData {
    rapidglm::Matrix design;
    rapidglm::Vector response, assignment, probability;
};
class NullGenerator {
    rapidglm::Family family_;
    bool crt_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_;

  public:
    NullGenerator(rapidglm::Family family, bool crt, std::uint64_t seed)
        : family_(family), crt_(crt), rng_(seed) {}

    NullObservation observation(double covariate) {
        double eta = 0.5 + 0.4 * covariate, mu = std::exp(eta);
        double p = crt_ ? 1 / (1 + std::exp(1 - 0.5 * covariate)) : 0.3;
        double g = std::bernoulli_distribution(p)(rng_), y;
        if (family_ == rapidglm::Family::gaussian)
            y = eta + normal_(rng_);
        else if (family_ == rapidglm::Family::binomial)
            y = std::bernoulli_distribution(1 / (1 + std::exp(-eta)))(rng_);
        else if (family_ == rapidglm::Family::poisson)
            y = std::poisson_distribution<int>(mu)(rng_);
        else {
            double lambda = std::gamma_distribution<double>(2, mu / 2)(rng_);
            y = std::poisson_distribution<int>(lambda)(rng_);
        }
        return {y, g, p};
    }

    NullData dataset(std::size_t n) {
        NullData data{rapidglm::Matrix(n, 2, 1), rapidglm::Vector(n), rapidglm::Vector(n),
                      rapidglm::Vector(n)};
        for (std::size_t i = 0; i < n; ++i) {
            data.design(i, 1) = normal_(rng_);
            auto row = observation(data.design(i, 1));
            data.response[i] = row.response;
            data.assignment[i] = row.assignment;
            data.probability[i] = row.probability;
        }
        return data;
    }
};
inline NullData generate_null(unsigned scenario, std::uint64_t index, std::size_t n = 256) {
    rapidglm::require(scenario < 8, "unknown calibration scenario");
    return NullGenerator(static_cast<rapidglm::Family>(scenario % 4), scenario >= 4,
                         202609260000ULL + index)
        .dataset(n);
}
} // namespace rapidglm_test
