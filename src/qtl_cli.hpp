#pragma once
// `rapidglm qtl config.json`: cis-QTL mapping from a phenotype BED, PLINK 1
// binary genotypes and an optional covariate table (the inputs FastQTL and
// QTLtools take), with count or Gaussian response models.
#include "io.hpp"
#include <rapidglm/qtl.hpp>
#include <set>
namespace rapidglm::cli {
inline std::vector<std::string> fields(const std::string &line) {
    std::istringstream in(line);
    std::vector<std::string> out;
    std::string cell;
    while (in >> cell)
        out.push_back(cell);
    return out;
}
// PLINK 1 binary genotypes as the PLINK documentation describes them: prefix.bim
// (chromosome, ID, cM, position, allele 1, allele 2), prefix.fam (family ID,
// sample ID, ...) and a variant-major prefix.bed that starts with 0x6c 0x1b 0x01
// and holds two bits per sample, the first sample in the lowest bits: 00 two
// copies of allele 1, 01 missing, 10 one copy, 11 none. The dosage counts allele
// 1. Samples are taken in the order of `samples` (IDs from the .fam).
inline Genotypes read_plink(const std::string &prefix, const std::vector<std::string> &samples,
                            std::vector<std::pair<std::string, std::string>> *alleles = nullptr,
                            const std::set<std::string> &chromosomes = {}) {
    std::vector<std::string> fam;
    {
        auto f = input(prefix + ".fam");
        std::string line;
        while (std::getline(f, line)) {
            const auto row = fields(line);
            require(row.size() >= 2, "a .fam line needs family and sample IDs");
            fam.push_back(row[1]);
        }
    }
    std::map<std::string, std::size_t> fam_index;
    for (std::size_t i = 0; i < fam.size(); ++i)
        require(fam_index.emplace(fam[i], i).second, "duplicate sample ID in .fam");
    std::vector<std::size_t> take;
    for (const auto &s : samples) {
        const auto found = fam_index.find(s);
        require(found != fam_index.end(), "a phenotype sample is missing from the genotypes");
        take.push_back(found->second);
    }
    Genotypes g;
    g.samples = samples.size();
    std::vector<unsigned char> selected;
    {
        auto f = input(prefix + ".bim");
        std::string line;
        while (std::getline(f, line)) {
            const auto row = fields(line);
            if (row.empty())
                continue;
            require(row.size() >= 6, "a .bim line needs six fields");
            const bool keep = chromosomes.empty() || chromosomes.count(row[0]);
            selected.push_back(keep);
            if (!keep)
                continue;
            g.chrom.push_back(row[0]);
            g.id.push_back(row[1]);
            g.position.push_back(std::stoll(row[3]));
            if (alleles)
                alleles->push_back({row[4], row[5]});
        }
    }
    g.variants = g.chrom.size();
    std::ifstream bed(prefix + ".bed", std::ios::binary);
    require(bed.good(), "cannot open the .bed file");
    unsigned char magic[3] = {};
    bed.read(reinterpret_cast<char *>(magic), 3);
    require(bed.gcount() == 3 && magic[0] == 0x6c && magic[1] == 0x1b && magic[2] == 0x01,
            "the .bed file is not a variant-major PLINK 1 file");
    const std::size_t bytes = (fam.size() + 3) / 4;
    bed.seekg(0, std::ios::end);
    require(static_cast<std::uint64_t>(bed.tellg()) == 3 + std::uint64_t(bytes) * selected.size(),
            "the .bed length does not match .fam and .bim");
    bed.seekg(3);
    std::vector<unsigned char> block(bytes);
    g.dosage.resize(g.samples * g.variants);
    static constexpr double code[4] = {2, std::numeric_limits<double>::quiet_NaN(), 1, 0};
    std::size_t kept = 0;
    for (std::size_t v = 0; v < selected.size(); ++v) {
        if (!selected[v]) {
            bed.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
            continue;
        }
        bed.read(reinterpret_cast<char *>(block.data()), static_cast<std::streamsize>(bytes));
        require(static_cast<std::size_t>(bed.gcount()) == bytes, "the .bed file is truncated");
        double *out = &g.dosage[kept++ * g.samples];
        for (std::size_t i = 0; i < take.size(); ++i) {
            const std::size_t s = take[i];
            out[i] = code[(block[s / 4] >> (2 * (s % 4))) & 3];
        }
    }
    return g;
}
// A phenotype BED as FastQTL and QTLtools read it: a header line starting with
// "#" (chromosome, start, end, ID, then one column per sample) and one row per
// phenotype. BED coordinates are zero-based and half-open, so the phenotype's
// anchor is [start + 1, end] (the TSS when end = start + 1).
struct PhenotypeBed {
    std::vector<std::string> samples;
    std::vector<CisPhenotype> phenotypes;
    std::vector<std::vector<double>> values; // per phenotype, per sample
};
inline PhenotypeBed read_phenotype_bed(const std::string &path) {
    auto table = read_table(path);
    require(table.size() >= 2 && !table[0].empty() && table[0][0].rfind("#", 0) == 0 &&
                table[0].size() > 4,
            "a phenotype BED needs a header (#chr, start, end, ID, samples) and rows");
    PhenotypeBed bed;
    bed.samples.assign(table[0].begin() + 4, table[0].end());
    for (std::size_t r = 1; r < table.size(); ++r) {
        const auto &row = table[r];
        CisPhenotype ph;
        ph.response = r - 1;
        ph.chrom = row[0];
        const auto start = std::stoll(row[1]), end = std::stoll(row[2]);
        require(start >= 0 && start <= end && end < std::numeric_limits<long long>::max(),
                ("invalid BED coordinates for " + row[3]).c_str());
        // Also accept point records: start=end anchors at that coordinate.
        ph.start = start == end ? start : start + 1;
        ph.end = end;
        ph.id = row[3];
        std::vector<double> v;
        for (std::size_t c = 4; c < row.size(); ++c)
            v.push_back(number(row[c]));
        bed.phenotypes.push_back(ph);
        bed.values.push_back(std::move(v));
    }
    return bed;
}
// A covariate table as FastQTL reads it: a header "id" then sample IDs, and one
// row per covariate. The design is an intercept and the covariates, sample by
// sample in the order given.
inline Matrix covariate_design(const std::string &path, const std::vector<std::string> &samples) {
    if (path.empty()) {
        Matrix x(samples.size(), 1);
        for (std::size_t i = 0; i < samples.size(); ++i)
            x(i, 0) = 1;
        return x;
    }
    std::vector<std::vector<std::string>> rows;
    {
        auto f = input(path);
        std::string line;
        while (std::getline(f, line))
            if (auto row = fields(line); !row.empty())
                rows.push_back(row);
    }
    require(rows.size() >= 2, "a covariate table needs a header and rows");
    std::map<std::string, std::size_t> column;
    for (std::size_t c = 1; c < rows[0].size(); ++c)
        column[rows[0][c]] = c;
    Matrix x(samples.size(), rows.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto found = column.find(samples[i]);
        require(found != column.end(), "a phenotype sample is missing from the covariates");
        x(i, 0) = 1;
        for (std::size_t k = 1; k < rows.size(); ++k) {
            require(rows[k].size() == rows[0].size(), "ragged covariate table");
            x(i, k) = number(rows[k][found->second]);
        }
    }
    return x;
}
// Library sizes: "sample<TAB>size" rows (a header line is skipped); offsets are
// their logarithms.
inline Vector library_offsets(const std::string &path, const std::vector<std::string> &samples) {
    std::map<std::string, double> size;
    for (const auto &row : read_table(path)) {
        require(row.size() == 2, "library sizes need two columns: sample and size");
        try {
            size[row[0]] = number(row[1]);
        } catch (const std::exception &) {
        }
    }
    Vector offset(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto found = size.find(samples[i]);
        require(found != size.end() && found->second > 0, "missing or invalid library size");
        offset[i] = std::log(found->second);
    }
    return offset;
}
template <class Hashes>
inline int run_qtl(const Tree &config, const std::string &config_path, Hashes &hashes,
                   std::chrono::steady_clock::time_point start) {
    const auto phenotype_path = config.get<std::string>("phenotypes");
    const auto genotype_prefix = config.get<std::string>("genotypes");
    const auto covariate_path = config.get<std::string>("covariates", "");
    const auto library_path = config.get<std::string>("library_sizes", "");
    const auto output = config.get<std::string>("output");
    for (const auto &key : {"responses", "design", "pairs", "offset", "weights"})
        require(!config.count(key), (std::string(key) + " is not a qtl setting").c_str());
    const auto bed = read_phenotype_bed(phenotype_path);
    std::vector<std::pair<std::string, std::string>> alleles;
    std::set<std::string> chromosomes;
    for (const auto &ph : bed.phenotypes)
        chromosomes.insert(ph.chrom);
    auto genotypes = read_plink(genotype_prefix, bed.samples, &alleles, chromosomes);
    const auto x = covariate_design(covariate_path, bed.samples);
    const std::size_t n = bed.samples.size();
    // Variants below min_maf (in these samples) are left out.
    const double min_maf = config.get<double>("min_maf", 0.0);
    require(std::isfinite(min_maf) && min_maf >= 0 && min_maf <= 0.5,
            "min_maf must be in [0, 0.5]");
    if (min_maf > 0) {
        std::size_t kept = 0;
        for (std::size_t v = 0; v < genotypes.variants; ++v) {
            double s = 0, k = 0;
            for (std::size_t i = 0; i < n; ++i)
                if (const double d = genotypes.dosage[v * n + i]; std::isfinite(d)) {
                    s += d;
                    ++k;
                }
            const double af = k ? s / (2 * k) : 0;
            if (std::min(af, 1 - af) < min_maf)
                continue;
            if (kept != v) {
                std::copy_n(genotypes.dosage.begin() + v * n, n,
                            genotypes.dosage.begin() + kept * n);
                genotypes.chrom[kept] = std::move(genotypes.chrom[v]);
                genotypes.id[kept] = std::move(genotypes.id[v]);
                genotypes.position[kept] = genotypes.position[v];
                alleles[kept] = std::move(alleles[v]);
            }
            ++kept;
        }
        genotypes.variants = kept;
        genotypes.dosage.resize(kept * n);
        genotypes.chrom.resize(kept);
        genotypes.id.resize(kept);
        genotypes.position.resize(kept);
        alleles.resize(kept);
    }
    SparseCSC responses{n, bed.phenotypes.size(), {0}, {}, {}};
    for (const auto &values : bed.values) {
        for (std::size_t i = 0; i < n; ++i)
            if (values[i] != 0) {
                responses.row_index.push_back(i);
                responses.values.push_back(values[i]);
            }
        responses.column_ptr.push_back(responses.values.size());
    }
    CisQtlOptions o;
    o.model = fit_options(config);
    require(library_path.empty() || o.model.family != Family::gaussian,
            "library_sizes is not supported with family gaussian");
    if (!library_path.empty())
        o.model.offset = library_offsets(library_path, bed.samples);
    o.window = config.get<long long>("window", 1'000'000);
    o.permutations = config.get<std::size_t>("permutations", 1000);
    o.nominal = config.get<bool>("nominal", false);
    o.seed = config.get<std::uint64_t>("seed", 1);
    o.threads = config.get<int>("threads", 1);
    for (auto path : {phenotype_path, genotype_prefix + ".bed", genotype_prefix + ".bim",
                      genotype_prefix + ".fam", covariate_path, library_path})
        if (!path.empty())
            hashes[path] = file_sha1(path);
    const auto result = cis_qtl(x, responses, genotypes, bed.phenotypes, o);
    {
        std::ofstream out(output + ".cis.tsv");
        require(out.good(), "cannot open QTL cis.tsv output");
        out << "phenotype_id\tchrom\tstart\tend\tvariants\tstatus\tlead_variant\tlead_position\t"
               "allele1\tallele2\tstatistic\tp_value\testimate\tstandard_"
               "error\ttheta\tpermutations\tpermutations_failed\t"
               "permutation_p\tbeta_p\tbeta_k\tbeta_n\treference_ncp\n";
        for (std::size_t f = 0; f < result.genes.size(); ++f) {
            const auto &ph = bed.phenotypes[f];
            const auto &g = result.genes[f];
            const bool lead = std::isfinite(g.statistic);
            out << ph.id << '\t' << ph.chrom << '\t' << ph.start << '\t' << ph.end << '\t'
                << g.variants << '\t' << g.status << '\t' << (lead ? genotypes.id[g.lead] : "NA")
                << '\t';
            if (lead)
                out << genotypes.position[g.lead] << '\t' << alleles[g.lead].first << '\t'
                    << alleles[g.lead].second;
            else
                out << "NA\tNA\tNA";
            for (double v : {g.statistic, g.p_value, g.estimate, g.standard_error, g.theta}) {
                out << '\t';
                value(out, v);
            }
            out << '\t' << g.permutations << '\t' << g.permutations_failed;
            for (double v : {g.permutation_p, g.beta_p, g.beta_k, g.beta_n, g.reference_ncp}) {
                out << '\t';
                value(out, v);
            }
            out << '\n';
        }
        out.close();
        require(!out.fail(), "cannot write QTL cis output");
    }
    if (o.nominal) {
        std::ofstream out(output + ".nominal.tsv");
        require(out.good(), "cannot open QTL nominal.tsv output");
        out << "phenotype_id\tvariant\tposition\tdistance\tstatistic\tp_value\testimate\tstandard_"
               "error\n";
        for (const auto &pair : result.nominal) {
            const auto &ph = bed.phenotypes[pair.phenotype];
            out << ph.id << '\t' << genotypes.id[pair.variant] << '\t'
                << genotypes.position[pair.variant] << '\t'
                << genotypes.position[pair.variant] - ph.start;
            for (double v : {pair.statistic, pair.p_value, pair.estimate, pair.standard_error}) {
                out << '\t';
                value(out, v);
            }
            out << '\n';
        }
        out.close();
        require(!out.fail(), "cannot write QTL nominal output");
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::size_t tested = 0;
    for (const auto &g : result.genes)
        tested += g.status.rfind("ok", 0) == 0;
    std::ofstream record(output + ".run.json");
    require(record.good(), "cannot open QTL run record");
    record << "{\"command\": \"qtl\", \"version\": \"" << version << "\", \"config\": ";
    quoted(record, config_path);
    record << ", \"samples\": " << n << ", \"phenotypes\": " << bed.phenotypes.size()
           << ", \"phenotypes_tested\": " << tested << ", \"variants\": " << genotypes.variants
           << ", \"covariates\": " << x.cols - 1 << ", \"window\": " << o.window
           << ", \"permutations\": " << o.permutations << ", \"seed\": " << o.seed
           << ", \"threads\": " << o.threads << ", \"min_maf\": ";
    value(record, min_maf);
    record << ", \"family\": ";
    quoted(record, config.get<std::string>("family", "nb"));
    record << ", \"input_sha1\": {";
    bool first = true;
    for (const auto &[path, sha] : hashes) {
        record << (first ? "" : ", ");
        quoted(record, path);
        record << ": ";
        quoted(record, sha);
        first = false;
    }
    record << "}, \"wall_seconds\": ";
    value(record, seconds);
    record << "}\n";
    record.close();
    require(!record.fail(), "cannot write QTL run record");
    return 0;
}
} // namespace rapidglm::cli
