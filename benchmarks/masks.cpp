// Isolate construction, cell selection and numerical-label materialization.
// Compare dense-double masks with the workflow's packed representation.
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Dense = std::vector<double>;
using Packed = boost::dynamic_bitset<std::uint64_t>;
struct Labels {
    std::vector<std::size_t> cells;
    Dense labels;
};
template <class Mask>
Labels select(std::size_t n, const std::vector<std::size_t> &treated,
              const std::vector<std::size_t> &retained, const Mask &nt, bool nt_controls) {
    Mask treatment(n);
    for (auto i : treated)
        treatment[i] = 1;
    Labels out;
    for (auto i : retained)
        if (!nt_controls || treatment[i] || nt[i])
            out.cells.push_back(i);
    out.labels.resize(out.cells.size());
    for (std::size_t i = 0; i < out.cells.size(); ++i)
        out.labels[i] = treatment[out.cells[i]];
    return out;
}
volatile double sink = 0;
template <class Mask>
double measure(std::size_t n, const std::vector<std::size_t> &treated,
               const std::vector<std::size_t> &retained, const Mask &nt, bool nt_controls,
               std::size_t repetitions) {
    double checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < repetitions; ++i) {
        auto result = select(n, treated, retained, nt, nt_controls);
        const auto index = i % result.cells.size();
        checksum += result.cells[index] + result.labels[index] + result.cells.size();
    }
    const auto stop = std::chrono::steady_clock::now();
    sink = checksum;
    return std::chrono::duration<double>(stop - start).count();
}
} // namespace
int main() {
    std::cout << std::setprecision(17)
              << "cells,density,nt_controls,repetition,mode,iterations,seconds,mask_bytes\n";
    for (std::size_t n : {100286u, 213412u, 1000000u})
        for (unsigned percentage : {1u, 30u}) {
            std::vector<std::size_t> treated, retained;
            Dense dense_nt(n);
            Packed packed_nt(n);
            for (std::size_t i = 0; i < n; ++i) {
                // Deterministic scattered calls; data preparation is not timed.
                const auto hash = (i * UINT64_C(11400714819323198485)) >> 32;
                if (hash % 100 < percentage)
                    treated.push_back(i);
                if (i % 17)
                    retained.push_back(i);
                if (hash % 7 == 0)
                    dense_nt[i] = packed_nt[i] = 1;
            }
            for (bool nt_controls : {false, true}) {
                const auto dense = select(n, treated, retained, dense_nt, nt_controls);
                const auto packed = select(n, treated, retained, packed_nt, nt_controls);
                if (dense.cells != packed.cells || dense.labels != packed.labels)
                    throw std::runtime_error("packed selection differs from dense labels");
                const std::size_t iterations = n < 1000000 ? 100 : 20;
                // Repetition zero warms both paths. Alternate execution order.
                for (unsigned repetition = 0; repetition <= 3; ++repetition)
                    for (unsigned turn = 0; turn < 2; ++turn) {
                        const bool use_packed = (turn + repetition) % 2;
                        const double seconds =
                            use_packed
                                ? measure(n, treated, retained, packed_nt, nt_controls, iterations)
                                : measure(n, treated, retained, dense_nt, nt_controls, iterations);
                        const std::size_t bytes = use_packed ? packed_nt.num_blocks() * 8 : n * 8;
                        std::cout << n << ',' << percentage / 100.0 << ',' << nt_controls << ','
                                  << repetition << ',' << (use_packed ? "packed" : "dense") << ','
                                  << iterations << ',' << seconds << ',' << bytes << '\n';
                    }
            }
        }
}
