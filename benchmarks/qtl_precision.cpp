// Times cis_qtl on one synthetic Gaussian gene (462 samples, an intercept and
// 42 covariates, 4,500 cis variants, 10,000 permutations, one thread) with
// double- and single-precision permutation scores, alternating the two and
// keeping each one's fastest run, then the packed products of one 512-response
// batch alone, the fused single-precision maxima and the shuffles.
// Deterministic inputs; setup is outside the timed regions. Pin the process to
// one core: `rapidglm_qtl_precision_benchmark [permutations] [repeats]`.
//
// `rapidglm_qtl_precision_benchmark genes GENES THREADS [permutations] [repeats]`
// instead times cis_qtl on GENES such genes at once (each with its own 4,500
// variants, stored as hard calls) on THREADS threads in single precision,
// keeping the fastest of the repeats: seconds per gene, and score multiply-adds
// per second per thread, which shows what running many threads costs each one.
// Pin it to THREADS cores.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <rapidglm/rapidglm.hpp>

using namespace rapidglm;
namespace {
volatile double sink = 0;
double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
} // namespace
// GENES genes of 4,500 variants each (hard calls), on THREADS threads.
int many_genes(int argc, char **argv) {
    const std::size_t n = 462, covariates = 42, m = 4500;
    const std::size_t genes = argc > 2 ? std::stoul(argv[2]) : 48;
    const int threads = argc > 3 ? std::stoi(argv[3]) : 12;
    const std::size_t permutations = argc > 4 ? std::stoul(argv[4]) : 10000;
    const int repeats = argc > 5 ? std::stoi(argv[5]) : 2;
    std::mt19937_64 rng(20261024);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> frequency(0.05, 0.5);
    Matrix x(n, covariates + 1, 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 1; k <= covariates; ++k)
            x(i, k) = normal(rng);
    Genotypes g;
    g.samples = n;
    g.variants = genes * m;
    g.hard_calls.assign(g.variants * g.call_bytes(), 0);
    std::vector<CisPhenotype> phenotypes;
    SparseCSC y{n, genes, {0}, {}, {}};
    std::vector<int> dosage(n);
    for (std::size_t f = 0; f < genes; ++f) {
        const long long anchor = 10'000'000LL * static_cast<long long>(f + 1);
        phenotypes.push_back({f, "gene" + std::to_string(f), "chr1", anchor, anchor});
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t v = f * m + k;
            std::binomial_distribution<int> allele(2, frequency(rng));
            for (std::size_t i = 0; i < n; ++i) {
                dosage[i] = allele(rng);
                // PLINK 1 codes: 00 two copies, 10 one, 11 none.
                const unsigned code = dosage[i] == 2 ? 0 : dosage[i] == 1 ? 2 : 3;
                g.hard_calls[v * g.call_bytes() + i / 4] |=
                    static_cast<std::uint8_t>(code << (2 * (i % 4)));
            }
            g.chrom.push_back("chr1");
            g.position.push_back(anchor - 450'000 + 200 * static_cast<long long>(k));
            if (k == 1234)
                for (std::size_t i = 0; i < n; ++i) {
                    y.row_index.push_back(i);
                    y.values.push_back(5 + 0.3 * x(i, 1) + 0.1 * dosage[i] + normal(rng));
                }
        }
        y.column_ptr.push_back(y.row_index.size());
    }
    CisQtlOptions o;
    o.model.family = Family::gaussian;
    o.permutations = permutations;
    o.seed = 20261016;
    o.threads = threads;
    const double score_fma = double(n) * m * permutations * genes;
    std::cout << std::setprecision(6)
              << "genes,threads,variants,permutations,seconds,seconds_per_gene,"
              << "thread_seconds_per_gene,score_gfma_per_s_per_thread\n";
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = cis_qtl(x, y, g, phenotypes, o);
        const double seconds = seconds_since(start);
        best = std::min(best, seconds);
        for (const auto &gene : result.genes)
            require(gene.status == "ok" && gene.variants == m, "benchmark gene failed");
        sink = result.genes[0].beta_p;
        std::cout << genes << ',' << threads << ',' << m << ',' << permutations << ',' << seconds
                  << ',' << seconds / genes << ',' << seconds * threads / genes << ','
                  << score_fma / seconds / threads / 1e9 << '\n';
    }
    std::cout << "best_seconds_per_gene," << best / genes << "\nbest_thread_seconds_per_gene,"
              << best * threads / genes << "\nbest_score_gfma_per_s_per_thread,"
              << score_fma / best / threads / 1e9 << '\n';
    return 0;
}
int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "genes")
        return many_genes(argc, argv);
    const std::size_t n = 462, covariates = 42, m = 4500;
    const std::size_t permutations = argc > 1 ? std::stoul(argv[1]) : 10000;
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 2;
    std::mt19937_64 rng(20261024);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> frequency(0.05, 0.5);
    Matrix x(n, covariates + 1, 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 1; k <= covariates; ++k)
            x(i, k) = normal(rng);
    Genotypes g;
    g.samples = n;
    g.variants = m;
    for (std::size_t v = 0; v < m; ++v) {
        std::binomial_distribution<int> allele(2, frequency(rng));
        for (std::size_t i = 0; i < n; ++i)
            g.dosage.push_back(allele(rng));
        g.chrom.push_back("chr1");
        g.position.push_back(1'000'000 + 100 * static_cast<long long>(v));
        g.id.push_back("v" + std::to_string(v));
    }
    SparseCSC y{n, 1, {0, n}, {}, {}};
    for (std::size_t i = 0; i < n; ++i) {
        y.row_index.push_back(i);
        y.values.push_back(5 + 0.3 * x(i, 1) + 0.1 * g.dosage[1234 * n + i] + normal(rng));
    }
    const std::vector<CisPhenotype> phenotypes{{0, "gene", "chr1", 1'225'000, 1'225'000}};
    CisQtlOptions o;
    o.model.family = Family::gaussian;
    o.permutations = permutations;
    o.seed = 20261016;
    o.threads = 1;
    // Multiply-adds of the score product Y'G_res alone, and of everything the
    // batches multiply: Y'Q for the scales and, in single precision, Q T' for
    // the residuals.
    const double score_fma = double(n) * m * permutations,
                 scale_fma = double(n) * (covariates + 1) * permutations;
    std::cout << std::setprecision(6) << "precision,seconds,score_gfma_per_s,all_gfma_per_s,"
              << "variants,permutations,permutation_p,beta_p\n";
    double best[2] = {1e300, 1e300};
    for (int r = 0; r < repeats; ++r)
        for (int single = 0; single < 2; ++single) {
            o.permutation_precision = single ? PermutationPrecision::single_precision
                                             : PermutationPrecision::double_precision;
            const auto start = std::chrono::steady_clock::now();
            const auto result = cis_qtl(x, y, g, phenotypes, o);
            const double seconds = seconds_since(start);
            best[single] = std::min(best[single], seconds);
            const auto &gene = result.genes[0];
            require(gene.status == "ok" && gene.variants == m, "benchmark gene failed");
            sink = gene.beta_p;
            const double all = score_fma + scale_fma * (single ? 2 : 1);
            std::cout << (single ? "single" : "double") << ',' << seconds << ','
                      << score_fma / seconds / 1e9 << ',' << all / seconds / 1e9 << ','
                      << gene.variants << ',' << gene.permutations << ',' << std::setprecision(17)
                      << gene.permutation_p << ',' << gene.beta_p << std::setprecision(6) << '\n';
        }
    std::cout << "double_over_single," << best[0] / best[1] << '\n';
    // One batch's score product alone: 512 responses by 4,500 variants.
    const std::size_t batch = 512;
    std::vector<double> a(n * batch), b(n * m), c(batch * m);
    for (auto &v : a)
        v = normal(rng);
    for (auto &v : b)
        v = normal(rng);
    const std::vector<float> af(a.begin(), a.end()), bf(b.begin(), b.end());
    // The same factors packed for the fused maxima (unit inverse information).
    std::vector<float, CacheAlignedAllocator<float>> packed_a(n * ((batch + 5) / 6 * 6)),
        packed_b(n * ((m + 15) / 16 * 16));
    detail::pack_rows_float(af.data(), n, batch, packed_a.data());
    detail::pack_columns_float(bf.data(), n, m, packed_b.data());
    std::vector<double> inverse_info((m + 15) / 16 * 16, rapidglm::nan), maxima(batch);
    std::fill_n(inverse_info.begin(), m, 1.0);
    const int calls = 10;
    const double batch_fma = double(n) * batch * m * calls;
    std::cout << "kernel,seconds,gfma_per_s\n";
    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::blocked_product(a.data(), b.data(), n, batch, m, c.data());
        double seconds = seconds_since(start);
        sink = c[7];
        std::cout << "blocked_product," << seconds << ',' << batch_fma / seconds / 1e9 << '\n';
        start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::blocked_product_generic(a.data(), b.data(), n, batch, m, c.data());
        seconds = seconds_since(start);
        sink = c[7];
        std::cout << "blocked_product_generic," << seconds << ',' << batch_fma / seconds / 1e9
                  << '\n';
        start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::blocked_product_float(af.data(), bf.data(), n, batch, m, c.data());
        seconds = seconds_since(start);
        sink = c[7];
        std::cout << "blocked_product_float," << seconds << ',' << batch_fma / seconds / 1e9
                  << '\n';
        start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::blocked_product_float_generic(af.data(), bf.data(), n, batch, m, c.data());
        seconds = seconds_since(start);
        sink = c[7];
        std::cout << "blocked_product_float_generic," << seconds << ',' << batch_fma / seconds / 1e9
                  << '\n';
        start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::fused_maxima(packed_a.data(), packed_b.data(), n, batch, m, inverse_info.data(),
                                 maxima.data());
        seconds = seconds_since(start);
        sink = maxima[7];
        std::cout << "fused_maxima," << seconds << ',' << batch_fma / seconds / 1e9 << '\n';
        start = std::chrono::steady_clock::now();
        for (int k = 0; k < calls; ++k)
            detail::fused_maxima_generic(packed_a.data(), packed_b.data(), n, batch, m,
                                         inverse_info.data(), maxima.data());
        seconds = seconds_since(start);
        sink = maxima[7];
        std::cout << "fused_maxima_generic," << seconds << ',' << batch_fma / seconds / 1e9 << '\n';
    }
    // The shuffles of the permutations (CounterRng value by value, then with
    // the values in bulk), which must agree.
    const Stream stream = keyed_stream(o.seed, 7, 0, name_key("gene"));
    std::vector<std::size_t> scalar(n), bulk(n);
    detail::QtlShuffler shuffler;
    shuffler.prepare(n);
    std::size_t checksum[2] = {0, 0};
    std::cout << "shuffles,seconds\n";
    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (std::size_t b = 0; b < permutations; ++b) {
            detail::qtl_shuffle(stream, b, scalar);
            checksum[0] += scalar[b % n];
        }
        double seconds = seconds_since(start);
        std::cout << "qtl_shuffle," << seconds << '\n';
        start = std::chrono::steady_clock::now();
        for (std::size_t b = 0; b < permutations; ++b) {
            shuffler.shuffle(stream, b, bulk.data());
            checksum[1] += bulk[b % n];
        }
        seconds = seconds_since(start);
        std::cout << "QtlShuffler," << seconds << '\n';
    }
    require(checksum[0] == checksum[1], "shuffles differ");
    return 0;
}
