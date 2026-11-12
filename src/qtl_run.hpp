#pragma once
// The parts of cis-QTL mapping that `rapidglm qtl` (src/qtl_cli.hpp) and the C
// interface (rapidglm_qtl in src/c_api.cpp) share: how a phenotype is anchored,
// which variants min_maf leaves out, the design, the library-size offsets, the
// response matrix, the summary of a run and the names of the output columns.
// Both call cis_qtl() (include/rapidglm/qtl.hpp) with what these build, so the
// two interfaces run the same analysis. Nothing here depends on the CLI's
// readers or on the C ABI.
#include <rapidglm/qtl.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rapidglm::qtl_run {
// Columns of the cis table (one row per phenotype) and of the nominal table
// (one row per variant-phenotype pair), in output order.
inline constexpr std::array<const char *, 22> cis_columns = {"phenotype_id",
                                                             "chrom",
                                                             "start",
                                                             "end",
                                                             "variants",
                                                             "status",
                                                             "lead_variant",
                                                             "lead_position",
                                                             "allele1",
                                                             "allele2",
                                                             "statistic",
                                                             "p_value",
                                                             "estimate",
                                                             "standard_error",
                                                             "theta",
                                                             "permutations",
                                                             "permutations_failed",
                                                             "permutation_p",
                                                             "beta_p",
                                                             "beta_k",
                                                             "beta_n",
                                                             "reference_ncp"};
inline constexpr std::array<const char *, 8> nominal_columns = {
    "phenotype_id", "variant", "position", "distance",
    "statistic",    "p_value", "estimate", "standard_error"};
// Whether a cis column holds text (the others are numbers).
inline bool cis_column_is_text(const std::string &name) {
    return name == "phenotype_id" || name == "chrom" || name == "status" ||
           name == "lead_variant" || name == "allele1" || name == "allele2";
}
inline bool nominal_column_is_text(const std::string &name) {
    return name == "phenotype_id" || name == "variant";
}
// A phenotype anchored as FastQTL and QTLtools read a BED record: coordinates
// are zero-based and half-open, so the anchor is [start + 1, end] (the TSS when
// end = start + 1); a point record (start = end) is anchored at that coordinate.
inline CisPhenotype bed_phenotype(std::size_t response, const std::string &id,
                                  const std::string &chrom, long long start, long long end) {
    require(start >= 0 && start <= end && end < std::numeric_limits<long long>::max(),
            ("invalid BED coordinates for " + id).c_str());
    CisPhenotype ph;
    ph.response = response;
    ph.chrom = chrom;
    ph.start = start == end ? start : start + 1;
    ph.end = end;
    ph.id = id;
    return ph;
}
// Leaves out the variants whose allele frequency in these samples (half the
// mean dosage, the halving being exact) is below min_maf, keeping the order of
// the rest and releasing the spare capacity. Returns the flags of the variants
// kept (one per variant before filtering), or an empty vector when min_maf is
// zero and nothing is filtered, so that a caller can filter what it holds per
// variant (IDs, alleles) in the same way.
inline std::vector<unsigned char> filter_min_maf(Genotypes &genotypes, double min_maf) {
    require(std::isfinite(min_maf) && min_maf >= 0 && min_maf <= 0.5,
            "min_maf must be in [0, 0.5]");
    if (min_maf <= 0)
        return {};
    const std::size_t stride = genotypes.call_bytes();
    std::vector<unsigned char> keep(genotypes.variants, 0);
    std::size_t kept = 0;
    for (std::size_t v = 0; v < genotypes.variants; ++v) {
        const double af = genotypes.mean_dosage(v) / 2;
        if (std::min(af, 1 - af) < min_maf)
            continue;
        keep[v] = 1;
        if (kept != v) {
            std::copy_n(genotypes.hard_calls.begin() + v * stride, stride,
                        genotypes.hard_calls.begin() + kept * stride);
            genotypes.chrom[kept] = std::move(genotypes.chrom[v]);
            genotypes.position[kept] = genotypes.position[v];
        }
        ++kept;
    }
    // The capacity of the unfiltered variants is released (each array briefly
    // held twice, before the scan allocates its gene buffers).
    genotypes.variants = kept;
    genotypes.hard_calls.resize(kept * stride);
    genotypes.chrom.resize(kept);
    genotypes.position.resize(kept);
    genotypes.hard_calls.shrink_to_fit();
    genotypes.chrom.shrink_to_fit();
    genotypes.position.shrink_to_fit();
    return keep;
}
// Phenotype values as the sparse response matrix cis_qtl takes: samples x
// phenotypes, the nonzero values only. value(p, i) is sample i of phenotype p.
template <class Value>
inline SparseCSC nonzero_responses(std::size_t samples, std::size_t phenotypes, Value value) {
    SparseCSC responses{samples, phenotypes, {0}, {}, {}};
    for (std::size_t p = 0; p < phenotypes; ++p) {
        for (std::size_t i = 0; i < samples; ++i) {
            const double v = value(p, i);
            if (v != 0) {
                responses.row_index.push_back(i);
                responses.values.push_back(v);
            }
        }
        responses.column_ptr.push_back(responses.values.size());
    }
    return responses;
}
// The design without covariates: an intercept.
inline Matrix intercept_design(std::size_t samples) {
    Matrix x(samples, 1);
    for (std::size_t i = 0; i < samples; ++i)
        x(i, 0) = 1;
    return x;
}
// Library sizes are for count models only.
inline void require_count_family(const FitOptions &model) {
    require(model.family != Family::gaussian,
            "library_sizes is not supported with family gaussian");
}
// Offsets of a count model: the logarithms of the library sizes, which must be
// positive.
inline Vector log_library_sizes(const Vector &sizes) {
    Vector offset(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        require(sizes[i] > 0, "missing or invalid library size");
        offset[i] = std::log(sizes[i]);
    }
    return offset;
}
// What a run records about itself besides its tables.
struct RunInfo {
    // The permutation scores were computed in single precision (only projected
    // Gaussian permutations have a single-precision path; count models refit
    // every permutation in double).
    bool single_scores = false;
    // Permutations of a count model were refitted: the fits asked for, and how
    // many lockstep fits fit() redid.
    bool refits = false;
    std::size_t tested = 0, refitted = 0;
};
inline RunInfo run_info(const CisQtlOptions &o, const CisQtlResult &result) {
    RunInfo info;
    info.single_scores = o.permutation_precision == PermutationPrecision::single_precision &&
                         detail::qtl_projected_permutations(o.model);
    info.refits = o.permutations > 0 && !detail::qtl_projected_permutations(o.model);
    for (const auto &g : result.genes) {
        info.tested += g.status.rfind("ok", 0) == 0;
        info.refitted += g.permutations_refitted;
    }
    return info;
}
// Calls as the C interface takes them (0, 1 or 2 copies of allele 1; negative:
// missing) into the two-bit PLINK codes of Genotypes::hard_calls, the first
// sample in the lowest bits: 00 two copies, 01 missing, 10 one copy, 11 none.
// `out` has (samples + 3) / 4 bytes. Returns false at a value that is not a call.
inline bool pack_calls(const std::int8_t *calls, std::size_t samples, std::uint8_t *out) {
    std::fill_n(out, (samples + 3) / 4, std::uint8_t(0));
    for (std::size_t i = 0; i < samples; ++i) {
        const int c = calls[i];
        unsigned code;
        if (c < 0)
            code = 1;
        else if (c == 2)
            code = 0;
        else if (c == 1)
            code = 2;
        else if (c == 0)
            code = 3;
        else
            return false;
        out[i / 4] |= static_cast<std::uint8_t>(code << (2 * (i % 4)));
    }
    return true;
}
} // namespace rapidglm::qtl_run
