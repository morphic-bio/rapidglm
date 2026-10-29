#pragma once
#include "kernels.hpp"
#include "workflows.hpp"
#include <array>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <optional>
// Cis-QTL mapping: for each phenotype (a response with a genomic anchor), the
// variants within a window around it are tested one at a time against a
// covariate-only null model, by the efficient score statistic, and a gene-level
// p-value for the strongest variant is calibrated by permutation with a Beta
// approximation. The design follows jaxQTL (MIT; Zhang, Kim, Suboc, Mancuso and
// Gazal 2025), whose permutation scheme and Beta approximation in turn follow
// FastQTL (Ongen, Buil, Brown, Dermitzakis and Delaneau 2016).
namespace rapidglm {
// Variant genotypes, in one of two forms. Dosages: column v holds variant v's
// dosage for every sample (dosage[v * samples + i]), and NaN marks a missing
// call. Hard calls (with `dosage` empty): the two-bit codes of a variant-major
// PLINK 1 .bed, call_bytes() bytes per variant, sample i of variant v in the two
// bits from bit 2 (i % 4) of byte v * call_bytes() + i / 4: 00 two copies of
// allele 1, 01 missing, 10 one copy, 11 none (hard_call_dosage), so a call takes
// a quarter of a byte instead of eight. Bits past the last sample are ignored.
// Either way a missing call is replaced by the variant's mean dosage.
struct Genotypes {
    std::size_t samples = 0, variants = 0;
    std::vector<double> dosage;
    std::vector<std::uint8_t> hard_calls;
    std::vector<std::string> chrom, id;
    std::vector<long long> position;
    static constexpr double hard_call_dosage[4] = {2, nan, 1, 0};
    // For each byte of four hard calls: their dosage sum plus 2^32 times the
    // number missing, so that one addition per byte accumulates both counts
    // (apart while the dosage sum stays below 2^32, at fewer than 2^31 samples).
    static constexpr std::array<std::uint64_t, 256> hard_call_byte_sums = [] {
        std::array<std::uint64_t, 256> t{};
        for (unsigned byte = 0; byte < 256; ++byte) {
            std::uint64_t sum = 0, missing = 0;
            for (unsigned k = 0; k < 4; ++k) {
                const unsigned code = (byte >> (2 * k)) & 3;
                sum += code == 0 ? 2 : code == 2 ? 1 : 0;
                missing += code == 1;
            }
            t[byte] = sum | missing << 32;
        }
        return t;
    }();
    std::size_t call_bytes() const {
        return (samples + 3) / 4;
    }
    // The mean of variant v's finite dosages, summed over the samples in order (0
    // when there are none). Hard calls are summed a whole byte at a time
    // (hard_call_byte_sums) and the last partial byte call by call. Their sums are
    // whole numbers and so exact: the mean is that of the same calls as dosages,
    // bit for bit.
    double mean_dosage(std::size_t v) const {
        if (!hard_calls.empty()) {
            const std::uint8_t *calls = &hard_calls[v * call_bytes()];
            const std::size_t whole = samples / 4;
            std::uint64_t both = 0;
            for (std::size_t b = 0; b < whole; ++b)
                both += hard_call_byte_sums[calls[b]];
            std::size_t sum = both & 0xffffffffu, missing = both >> 32;
            for (std::size_t i = 4 * whole; i < samples; ++i) {
                const unsigned code = (calls[whole] >> (2 * (i % 4))) & 3;
                sum += code == 0 ? 2 : code == 2 ? 1 : 0;
                missing += code == 1;
            }
            const std::size_t called = samples - missing;
            return called ? static_cast<double>(sum) / static_cast<double>(called) : 0.0;
        }
        double s = 0;
        std::size_t k = 0;
        for (std::size_t i = 0; i < samples; ++i) {
            const double d = dosage[v * samples + i];
            if (std::isfinite(d)) {
                s += d;
                ++k;
            }
        }
        return k ? s / static_cast<double>(k) : 0.0;
    }
    void validate() const {
        require(((dosage.size() == samples * variants && hard_calls.empty()) ||
                 (hard_calls.size() == call_bytes() * variants && dosage.empty())) &&
                    chrom.size() == variants && position.size() == variants &&
                    (id.empty() || id.size() == variants),
                "genotype dimensions mismatch");
    }
};
// A phenotype: response column `response`, anchored at [start, end] on `chrom`
// (the transcription start site, or the gene body). Its cis window is
// [start - window, end + window].
struct CisPhenotype {
    std::size_t response = 0;
    std::string id, chrom;
    long long start = 0, end = 0;
};
// Arithmetic of the projected Gaussian permutation scores (see cis_qtl).
enum class PermutationPrecision { single_precision, double_precision };
struct CisQtlOptions {
    // The response model; offsets (log library size for counts) in model.offset.
    FitOptions model;
    long long window = 1'000'000;
    // Permutations for the gene-level p-value; 0 reports the nominal scan only.
    std::size_t permutations = 1000;
    // Report every variant-phenotype pair as well as each phenotype's lead.
    bool nominal = false;
    // Permutation k of phenotype `id` is drawn from a stream keyed by the seed and
    // the phenotype's ID, so results do not depend on threads or on the other
    // phenotypes of the run.
    std::uint64_t seed = 1;
    int threads = 1;
    // The permutation scores of a Gaussian null without prior weights, offset,
    // ridge or robust covariance (scored by projection, without refits): the
    // product of the residualized permuted responses and the projected
    // genotypes in single precision, summed in double across 128-sample panels,
    // or entirely in double precision. Observed statistics, nominal results,
    // refitted permutations and the Beta calibration are double either way.
    PermutationPrecision permutation_precision = PermutationPrecision::single_precision;
};
struct CisQtlPair {
    std::size_t phenotype = 0, variant = 0;
    // statistic: efficient score z; estimate, standard_error: the one-step score
    // estimate u / I of the variant's coefficient and sqrt(scale / I).
    double statistic = nan, p_value = nan, estimate = nan, standard_error = nan;
};
struct CisQtlGene {
    std::size_t phenotype = 0, variants = 0, lead = 0;
    std::string status = "not_run";
    double statistic = nan, p_value = nan, estimate = nan, standard_error = nan, theta = nan;
    // The Beta-approximated gene-level p-value and the permutation p-value
    // (1 + #{max |z| at least the observed}) / (permutations + 1).
    double beta_p = nan, permutation_p = nan, beta_k = nan, beta_n = nan, reference_ncp = nan;
    std::size_t permutations = 0, permutations_failed = 0;
};
struct CisQtlResult {
    std::vector<CisQtlGene> genes;
    std::vector<CisQtlPair> nominal;
};
namespace detail {
// What a gene's scores take from the null's working weights alone: the factor
// of X'WX, s = L^-1 D^-1 X'WG (p x m, row k at s + k m) and the information
// v - |s|^2 of each variant. With weights that do not change between fits (a
// Gaussian null without prior weights) it serves every permutation.
struct QtlBasis {
    std::optional<Cholesky> factor;
    std::vector<double> s, information;
    bool ready = false;
};
// Working storage of qtl_scores, which a caller may keep from call to call.
struct QtlScoreScratch {
    std::vector<double> u, v, a, st;
};
// u = G'r (G n x m, row-major), summed over the samples in order.
inline void qtl_residual_scores(const double *g, const double *r, std::size_t n, std::size_t m,
                                double *u) {
    std::fill(u, u + m, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double *gi = g + i * m, ri = r[i];
        for (std::size_t q = 0; q < m; ++q)
            u[q] += ri * gi[q];
    }
}
// Score statistics of many variants against one regular null fit, at once.
// With W the working weights, r the score residuals and L L' = X'WX:
// u = g'r - (L^-1 X'Wg)'(L^-1 X'r), I = g'Wg - |L^-1 X'Wg|^2 and
// z = u / sqrt(I scale): the per-variant arithmetic of ScoreWorkspace, with the
// sums over samples taken for all variants at once (cross_products). `g` holds the
// variants sample by sample (g[i m + q]). z (and, if asked, estimates and SEs)
// for every variant; `basis` (may be null) holds the weight-only part: computed
// when not ready, reused when it is. `scratch` (may be null) holds the working
// vectors.
inline bool qtl_scores(const Matrix &x, const FitResult &null, const std::vector<double> &g,
                       std::size_t m, std::vector<double> &z, std::vector<double> *estimate,
                       std::vector<double> *standard_error, QtlBasis *basis = nullptr,
                       QtlScoreScratch *scratch = nullptr) {
    const std::size_t n = x.rows, p = x.cols;
    const auto &w = null.working_weights;
    const auto &r = null.score_residuals;
    QtlBasis own;
    QtlBasis &b = basis ? *basis : own;
    QtlScoreScratch own_scratch;
    QtlScoreScratch &work = scratch ? *scratch : own_scratch;
    auto &u = work.u;
    u.resize(m);
    if (!b.ready) {
        // The factor is of the equilibrated matrix D^-1 (X'WX) D^-1 (D: the root
        // diagonal), so vectors are divided by D before the forward substitution.
        b.factor.emplace(crossprod(x, w));
        if (!b.factor->ok)
            return false;
        const auto &ch = *b.factor;
        const auto &l = ch.l;
        // c = X'WG, u = G'r and v = G'WG (diagonal) by detail::cross_products.
        auto &v = work.v, &a = work.a;
        v.resize(m);
        a.resize(n * p);
        b.s.assign(p * m, 0.0);
        auto &c = b.s;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < p; ++k)
                a[i * p + k] = w[i] * x(i, k);
        cross_products(a.data(), w.data(), r.data(), g.data(), n, p, m, c.data(), u.data(),
                       v.data());
        // Forward substitution L s = D^-1 c, all variants at once (rows of c by covariate).
        for (std::size_t k = 0; k < p; ++k) {
            double *ck = &c[k * m];
            const double inverse_scale = 1 / ch.scale[k];
            for (std::size_t q = 0; q < m; ++q)
                ck[q] *= inverse_scale;
            for (std::size_t j = 0; j < k; ++j) {
                const double lkj = l(k, j);
                const double *cj = &c[j * m];
                for (std::size_t q = 0; q < m; ++q)
                    ck[q] -= lkj * cj[q];
            }
            const double d = 1 / l(k, k);
            for (std::size_t q = 0; q < m; ++q)
                ck[q] *= d;
        }
        b.information.assign(m, 0.0);
        for (std::size_t q = 0; q < m; ++q) {
            double ss = 0;
            for (std::size_t k = 0; k < p; ++k)
                ss += c[k * m + q] * c[k * m + q];
            const double info = v[q] - ss;
            b.information[q] = info > 1e-12 * std::max(1.0, v[q]) ? info : 0;
        }
        b.ready = true;
    } else
        qtl_residual_scores(g.data(), r.data(), n, m, u.data());
    const auto &ch = *b.factor;
    const auto &l = ch.l;
    Vector t = xtv(x, r);
    for (std::size_t k = 0; k < p; ++k) {
        t[k] /= ch.scale[k];
        for (std::size_t j = 0; j < k; ++j)
            t[k] -= l(k, j) * t[j];
        t[k] /= l(k, k);
    }
    z.assign(m, nan);
    if (estimate)
        estimate->assign(m, nan);
    if (standard_error)
        standard_error->assign(m, nan);
    auto &st = work.st;
    st.assign(m, 0.0);
    for (std::size_t k = 0; k < p; ++k) {
        const double *sk = &b.s[k * m];
        const double tk = t[k];
        for (std::size_t q = 0; q < m; ++q)
            st[q] += sk[q] * tk;
    }
    for (std::size_t q = 0; q < m; ++q) {
        const double info = b.information[q], score = u[q] - st[q];
        if (!(info > 0))
            continue;
        z[q] = score / std::sqrt(info * null.scale);
        if (estimate)
            (*estimate)[q] = score / info;
        if (standard_error)
            (*standard_error)[q] = std::sqrt(null.scale / info);
    }
    return true;
}
// The survival function of a noncentral chi-squared distribution with one
// degree of freedom (central at ncp = 0).
inline double chi1_sf(double x, double ncp) {
    if (!(x > 0))
        return 1;
    if (ncp <= 0)
        return boost::math::cdf(
            boost::math::complement(boost::math::chi_squared_distribution<>(1), x));
    return boost::math::cdf(
        boost::math::complement(boost::math::non_central_chi_squared_distribution<>(1, ncp), x));
}
// Beta(k, n) by maximum likelihood, from p-values in (0, 1): Newton steps with
// step halving on the concave log likelihood, from the method of moments.
inline bool beta_mle(const std::vector<double> &p, double &k, double &n) {
    const double count = static_cast<double>(p.size());
    double s1 = 0, s2 = 0, mean = 0, var = 0;
    for (double v : p) {
        s1 += std::log(v);
        s2 += std::log1p(-v);
        mean += v;
    }
    mean /= count;
    for (double v : p)
        var += (v - mean) * (v - mean);
    var /= count;
    const double common = var > 0 ? mean * (1 - mean) / var - 1 : 1;
    k = std::max(1e-3, mean * common);
    n = std::max(1e-3, (1 - mean) * common);
    auto loglik = [&](double a, double b) {
        return (a - 1) * s1 + (b - 1) * s2 -
               count * (vmath::detail::lgamma(a) + vmath::detail::lgamma(b) -
                        vmath::detail::lgamma(a + b));
    };
    double current = loglik(k, n);
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double dab = boost::math::digamma(k + n), tab = boost::math::trigamma(k + n);
        const double gk = s1 - count * (boost::math::digamma(k) - dab),
                     gn = s2 - count * (boost::math::digamma(n) - dab);
        const double hkk = -count * (boost::math::trigamma(k) - tab),
                     hnn = -count * (boost::math::trigamma(n) - tab), hkn = count * tab;
        const double det = hkk * hnn - hkn * hkn;
        if (!(det > 0))
            return false;
        double dk = -(hnn * gk - hkn * gn) / det, dn = -(hkk * gn - hkn * gk) / det;
        double step = 1;
        bool moved = false;
        for (int half = 0; half < 60; ++half, step /= 2) {
            const double a = k + step * dk, b = n + step * dn;
            if (a > 0 && b > 0) {
                const double candidate = loglik(a, b);
                if (candidate >= current - 1e-12 * (1 + std::abs(current))) {
                    moved = std::abs(a - k) > 1e-12 * k || std::abs(b - n) > 1e-12 * n;
                    k = a;
                    n = b;
                    current = candidate;
                    break;
                }
            }
        }
        if (!moved || (std::abs(step * dk) < 1e-10 * k && std::abs(step * dn) < 1e-10 * n))
            return std::isfinite(k) && std::isfinite(n);
    }
    return std::isfinite(k) && std::isfinite(n);
}
// The Beta approximation of a gene's permutation maxima (jaxQTL's calibration):
// the maxima T_b are referred to a noncentral chi-squared with one degree of
// freedom whose noncentrality makes the moment estimate of the first Beta shape
// equal 1 (one effective test), and Beta(k, n) is fitted to the resulting
// p-values by maximum likelihood. The gene-level p-value of an observed
// statistic z is then BetaCDF(sf(z^2)). Maxima whose variance is at most
// `constant` times their squared mean are taken as constant: 64 double epsilons
// by default, and more for maxima from single-precision scores, whose rounding
// alone spreads maxima that are equal in exact arithmetic.
struct BetaCalibration {
    bool ok = false;
    double ncp = nan, k = nan, n = nan;
};
inline BetaCalibration beta_calibration(const std::vector<double> &maxima,
                                        double constant = 64 *
                                                          std::numeric_limits<double>::epsilon()) {
    BetaCalibration out;
    if (maxima.size() < 2 || !std::all_of(maxima.begin(), maxima.end(),
                                          [](double v) { return std::isfinite(v) && v >= 0; }))
        return out;
    double mean_max = std::accumulate(maxima.begin(), maxima.end(), 0.0) / maxima.size();
    double variance_max = 0;
    for (double v : maxima)
        variance_max += (v - mean_max) * (v - mean_max) / maxima.size();
    if (variance_max <=
        constant * std::max(mean_max * mean_max, std::numeric_limits<double>::min()))
        return out;
    std::vector<double> stats;
    for (double t : maxima)
        stats.push_back(std::isfinite(t) ? t * t : 0.0);
    const double tiny = std::numeric_limits<double>::min(),
                 eps = std::numeric_limits<double>::epsilon();
    auto pvalues = [&](double ncp) {
        std::vector<double> p;
        for (double s : stats)
            p.push_back(std::clamp(chi1_sf(s, ncp), tiny, 1 - eps));
        return p;
    };
    auto shape = [&](double ncp) {
        const auto p = pvalues(ncp);
        double mean = 0, var = 0;
        for (double v : p)
            mean += v;
        mean /= static_cast<double>(p.size());
        for (double v : p)
            var += (v - mean) * (v - mean);
        var /= static_cast<double>(p.size());
        return var > 0 ? mean * (mean * (1 - mean) / var - 1) - 1 : nan;
    };
    // Bracket the root of shape(ncp) on [0, 100] and bisect; without a sign change
    // keep the endpoint closer to the root.
    double lo = 0, hi = 100, flo = shape(lo), fhi = shape(hi);
    if (!std::isfinite(flo) || !std::isfinite(fhi))
        return out;
    if (flo * fhi > 0)
        out.ncp = std::abs(flo) <= std::abs(fhi) ? lo : hi;
    else {
        for (int iteration = 0; iteration < 60 && hi - lo > 1e-6 * (1 + lo); ++iteration) {
            const double mid = 0.5 * (lo + hi), fmid = shape(mid);
            if (!std::isfinite(fmid))
                break;
            if ((fmid > 0) == (flo > 0)) {
                lo = mid;
                flo = fmid;
            } else
                hi = mid;
        }
        out.ncp = 0.5 * (lo + hi);
    }
    out.ok = beta_mle(pvalues(out.ncp), out.k, out.n);
    if (!out.ok)
        return {};
    return out;
}
inline double beta_adjust(const BetaCalibration &calibration, double z) {
    if (!calibration.ok || !std::isfinite(z))
        return nan;
    const double p = chi1_sf(z * z, calibration.ncp);
    return boost::math::cdf(boost::math::beta_distribution<>(calibration.k, calibration.n),
                            std::clamp(p, 0.0, 1.0));
}
// Permutation b of a phenotype: a Fisher-Yates shuffle of the sample indices by
// CounterRng(stream, b).
inline void qtl_shuffle(const Stream &stream, std::size_t b, std::vector<std::size_t> &index) {
    std::iota(index.begin(), index.end(), 0);
    CounterRng rng(stream.seed, stream.test, b);
    for (std::size_t r = index.size(); r > 1; --r)
        std::swap(index[r - 1], index[static_cast<std::size_t>(rng.bounded(r))]);
}
// The shuffles of qtl_shuffle with the generator's words computed in bulk.
// CounterRng(stream, b).next64() returns the words of Philox blocks 0, 1, 2, ...
// of draw b two at a time, in block order, and bounded(r) draws again while a
// value is below 2^64 mod r. QtlShuffler computes the blocks that a shuffle of
// n samples uses when nothing is rejected (n - 1 values, rounded up to eight
// blocks) with the vectorized Philox, then runs the swaps over those values,
// with each range's rejection threshold from a table. A shuffle that rejects a
// value (at most n / 2^64 likely per value) continues with the following blocks,
// eight at a time, so every shuffle is that of qtl_shuffle. Value mod r comes
// without a division from c_r = ceil(2^128 / r) (Lemire, Kaser and Kurz 2019,
// doi:10.1002/spe.2689): x mod r is the integer part of ((c_r x) mod 2^128) r /
// 2^128, exactly, for every 64-bit x and 2 <= r <= 2^64; two 64-bit divisions
// per swap took more than half of the shuffle's time on the efficiency cores.
#ifdef __SIZEOF_INT128__
__extension__ typedef unsigned __int128 uint128;
#endif
class QtlShuffler {
    std::size_t n_ = 0, groups_ = 0;
    std::vector<std::uint64_t> threshold_, values_;
#ifdef __SIZEOF_INT128__
    std::vector<uint128> inverse_;
#endif
    // The sixteen values of blocks 8 group .. 8 group + 7 into out.
    static void values(const Stream &stream, std::uint32_t draw, std::size_t group,
                       std::uint64_t *out) {
        alignas(32) std::uint32_t words[32];
        const auto test = static_cast<std::uint32_t>(stream.test);
#ifdef RAPIDGLM_PHILOX_AVX2
        if (philox_avx2_available())
            philox8_avx2(8 * group, draw, test, stream.seed, words);
        else
#endif
            philox_blocks<8>(8 * group, draw, test, stream.seed, words);
        // Word j of block l is words[8 j + l]; value 2 l + h is words 2 h and
        // 2 h + 1 of block l.
        for (std::size_t l = 0; l < 8; ++l)
            for (std::size_t h = 0; h < 2; ++h)
                out[2 * l + h] =
                    static_cast<std::uint64_t>(words[16 * h + l]) << 32 | words[16 * h + 8 + l];
    }

  public:
    // For shuffles of n samples. `groups` (tests): the groups of eight blocks
    // computed before the swaps, by default enough for n - 1 values.
    void prepare(std::size_t n, std::size_t groups = 0) {
        n_ = n;
        groups_ = groups ? groups : (n + 14) / 16;
        threshold_.assign(n + 1, 0);
        for (std::size_t r = 1; r <= n; ++r)
            threshold_[r] = (-static_cast<std::uint64_t>(r)) % r;
#ifdef __SIZEOF_INT128__
        inverse_.assign(n + 1, 0);
        for (std::size_t r = 2; r <= n; ++r)
            inverse_[r] = ~static_cast<uint128>(0) / r + 1;
#endif
        values_.assign(16 * groups_, 0);
    }
    std::size_t samples() const {
        return n_;
    }
    // value mod r for 2 <= r <= n.
    std::uint64_t remainder(std::uint64_t value, std::size_t r) const {
#ifdef __SIZEOF_INT128__
        const uint128 low = inverse_[r] * value;
        const auto high = static_cast<std::uint64_t>(low >> 64),
                   bottom = static_cast<std::uint64_t>(low);
        return static_cast<std::uint64_t>(
            (static_cast<uint128>(high) * r + (static_cast<uint128>(bottom) * r >> 64)) >> 64);
#else
        return value % r;
#endif
    }
    // Permutation b: index[0 .. n - 1] as qtl_shuffle(stream, b, index) leaves it.
    void shuffle(const Stream &stream, std::size_t b, std::size_t *index) {
        require(stream.test <= UINT32_MAX && b <= UINT32_MAX, "RNG test/draw ID exceeds 32 bits");
        const auto draw = static_cast<std::uint32_t>(b);
        for (std::size_t k = 0; k < groups_; ++k)
            values(stream, draw, k, &values_[16 * k]);
        std::iota(index, index + n_, 0);
        std::size_t next = 0, available = 16 * groups_;
        for (std::size_t r = n_; r > 1; --r) {
            std::uint64_t value;
            do {
                if (next == available) {
                    if (values_.size() < available + 16)
                        values_.resize(available + 16);
                    values(stream, draw, available / 16, &values_[available]);
                    available += 16;
                }
                value = values_[next++];
            } while (value < threshold_[r]);
            std::swap(index[r - 1], index[static_cast<std::size_t>(remainder(value, r))]);
        }
    }
};
// Whether cis_qtl scores the permutations of this null by projection: a
// Gaussian null without prior weights, offset, ridge or robust covariance has
// the same working weights and hat matrix for every permuted response.
inline bool qtl_projected_permutations(const FitOptions &model) {
    return model.family == Family::gaussian && model.weights.empty() && model.offset.empty() &&
           model.ridge == 0 && !model.robust;
}
// The largest finite |c_q| inverse_info_q inverse_scale (each product rounded
// left to right, as written), or NaN when none is finite. A maximum does not
// depend on the order of its terms, so it is kept in eight running lanes.
inline double qtl_finite_maximum_generic(const double *c, const double *inverse_info, std::size_t m,
                                         double inverse_scale) {
    constexpr double largest = std::numeric_limits<double>::max();
    double lane[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::size_t q = 0;
    for (; q + 8 <= m; q += 8)
        for (std::size_t l = 0; l < 8; ++l) {
            const double value = std::abs(c[q + l]) * inverse_info[q + l] * inverse_scale;
            // NaN and infinity fail the second comparison.
            lane[l] = value > lane[l] && value <= largest ? value : lane[l];
        }
    for (; q < m; ++q) {
        const double value = std::abs(c[q]) * inverse_info[q] * inverse_scale;
        lane[0] = value > lane[0] && value <= largest ? value : lane[0];
    }
    double maximum = -1;
    for (double v : lane)
        maximum = std::max(maximum, v);
    return maximum >= 0 ? maximum : nan;
}
#ifdef RAPIDGLM_PHILOX_AVX2
__attribute__((target("avx2,fma"))) inline double
qtl_finite_maximum_avx2(const double *c, const double *inverse_info, std::size_t m,
                        double inverse_scale) {
    const __m256d sign = _mm256_set1_pd(-0.0), none = _mm256_set1_pd(-1),
                  largest = _mm256_set1_pd(std::numeric_limits<double>::max()),
                  scale = _mm256_set1_pd(inverse_scale);
    __m256d m0 = none, m1 = none;
    std::size_t q = 0;
    for (; q + 8 <= m; q += 8) {
        __m256d v0 = _mm256_mul_pd(_mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c + q)),
                                                 _mm256_loadu_pd(inverse_info + q)),
                                   scale);
        __m256d v1 = _mm256_mul_pd(_mm256_mul_pd(_mm256_andnot_pd(sign, _mm256_loadu_pd(c + q + 4)),
                                                 _mm256_loadu_pd(inverse_info + q + 4)),
                                   scale);
        v0 = _mm256_blendv_pd(none, v0, _mm256_cmp_pd(v0, largest, _CMP_LE_OQ));
        v1 = _mm256_blendv_pd(none, v1, _mm256_cmp_pd(v1, largest, _CMP_LE_OQ));
        m0 = _mm256_max_pd(m0, v0);
        m1 = _mm256_max_pd(m1, v1);
    }
    alignas(32) double lanes[8];
    _mm256_store_pd(lanes, m0);
    _mm256_store_pd(lanes + 4, m1);
    double maximum = -1;
    for (double v : lanes)
        maximum = std::max(maximum, v);
    for (; q < m; ++q) {
        const double value = std::abs(c[q]) * inverse_info[q] * inverse_scale;
        maximum = value > maximum && value <= std::numeric_limits<double>::max() ? value : maximum;
    }
    return maximum >= 0 ? maximum : nan;
}
#endif
inline double qtl_finite_maximum(const double *c, const double *inverse_info, std::size_t m,
                                 double inverse_scale) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return qtl_finite_maximum_avx2(c, inverse_info, m, inverse_scale);
#endif
    return qtl_finite_maximum_generic(c, inverse_info, m, inverse_scale);
}
// Buffers of a gene worker, reused from gene to gene (cis_qtl keeps one per
// thread). Each grows to the largest gene the worker has met and is never
// shrunk, so later genes allocate, fault in and clear no memory.
struct QtlWorkspace {
    std::vector<std::size_t> cis;
    std::vector<double> g, z, estimate, standard_error, maxima;
    QtlBasis basis;
    QtlScoreScratch scores;
    // qtl_projected_maxima's.
    std::vector<double> qbasis, qbasis_t, block, projection, shuffled, t, t_t, fitted,
        inverse_scales, inverse_info, cross, row_maxima;
    std::vector<float, CacheAlignedAllocator<float>> packed_g, packed_r;
    std::array<std::vector<std::size_t>, 8> index;
    std::vector<std::size_t> source;
    QtlShuffler shuffler;
    // Bytes of the blocks that grown() has freed since release().
    std::size_t freed = 0;
    // v.data() with room for `size` elements. v is replaced (cleared) only when
    // it is smaller, and never shrunk.
    template <class T, class Allocator> T *grown(std::vector<T, Allocator> &v, std::size_t size) {
        if (v.size() < size) {
            freed += v.capacity() * sizeof(T);
            std::vector<T, Allocator>().swap(v);
            v.resize(size);
        }
        return v.data();
    }
    // glibc keeps a freed block below its mmap threshold (which rises to 32 MB
    // as large blocks are freed) resident in the arena of the thread that
    // freed it, where it serves only smaller requests; as workers met ever
    // larger genes, the blocks they had outgrown added 0.8 GiB to the
    // genome-wide peak. So once a gene's buffers have grown by a megabyte or
    // more, the free pages are returned (rarely: each return interrupts every
    // thread).
    void release() {
#ifdef __GLIBC__
        if (freed >= (std::size_t{1} << 20))
            malloc_trim(0);
#endif
        freed = 0;
    }
};
// Whether a column of x is a nonzero constant, so that the constant vector lies
// in the column space.
inline bool qtl_has_intercept(const Matrix &x) {
    for (std::size_t k = 0; k < x.cols; ++k) {
        bool constant = x.rows > 0 && x(0, k) != 0;
        for (std::size_t i = 1; constant && i < x.rows; ++i)
            constant = x(i, k) == x(0, k);
        if (constant)
            return true;
    }
    return false;
}
// In single precision, a permutation whose response has a sum of squares about
// its centre (see qtl_projected_maxima) of at most this many times its
// residual sum of squares is scored from its centred response rather than its
// residual.
inline constexpr double qtl_centred_ratio = 2;
// The permutation maxima of such a null, without refits. Q = X D^-1 L^-T has
// orthonormal columns. G is projected once: G_res = G - Q (Q'G), using the
// observed basis's Q'G and information. Then G_res'(I-H)y_b = G_res'y_b, so
// scores need no permuted refit. In double precision `g` (n x m, sample by
// sample) is replaced by G_res; in single precision G_res is formed in float
// and `g` is left as it was. Permutations 0 .. permutations - 1 of `stream`
// shuffle y, 512 at a time; the largest finite |z| of each is appended to
// `maxima`, and one without a finite |z| is counted in `failed`. `workspace`
// (may be null) holds the buffers.
//
// In double precision the scores are Y_b'G_res. In single precision each
// permuted response is first residualized in double (or centred, see below),
// r_b = y_b - Q t_b with the t_b = Q'y_b of its scale, so both factors of the
// product are centred and
// orthogonal to the covariates and the float product cancels nothing they
// share. Each residual is divided by its residual standard deviation before its
// conversion to float, so the float entries are O(1) whatever the response's
// units: like the double path, this one does not depend on them. R'G_res is
// then summed in float within 128-sample panels and in double across them, and
// each permutation's largest |z| is taken from those sums in the product's
// register tiles (fused_maxima), so the batch's scores are never stored.
// Scales, information and maxima are double in both.
//
// The residual's purpose is to keep the float factors free of what the
// covariates explain; G_res is orthogonal to the covariates, so in exact
// arithmetic any y_b - c with c in their span has the scores of r_b. A
// permutation leaves the covariates little to explain, so the centred response
// y_b - ybar 1 (ybar: the mean when the design has an intercept, 0 otherwise)
// can stand for r_b: its sum of squares SS_c is the same for every permutation,
// and SS_c <= qtl_centred_ratio rss_b bounds the cost. The bounds on the float
// rounding of the response and on every float sum (gamma |g_res| |response| by
// Cauchy-Schwarz) then grow at most by sqrt(SS_c / rss_b) <= sqrt(2) against
// those for r_b, and the part of y_b - ybar 1 in the covariate span, of norm
// sqrt(SS_c - rss_b) <= sqrt(rss_b), adds at most the rounding of G_res (2^-24
// relative) times that norm. Those permutations skip the product
// Q T' and its subtraction; the others (a permutation that the covariates
// happen to fit well) keep the full residual, computed for them alone with the
// same sums as before, and columns that took the fallback hold theirs.
inline void qtl_projected_maxima(const Matrix &x, const QtlBasis &basis, std::vector<double> &g,
                                 std::size_t m, const Vector &y, const Stream &stream,
                                 std::size_t permutations, PermutationPrecision precision,
                                 std::vector<double> &maxima, std::size_t &failed,
                                 QtlWorkspace *workspace = nullptr) {
    const auto &factor = *basis.factor;
    const std::size_t n = x.rows, p = x.cols, batch = 512, df = n - p;
    QtlWorkspace own;
    QtlWorkspace &w = workspace ? *workspace : own;
    double *qbasis = w.grown(w.qbasis, n * p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < p; ++k) {
            double value = x(i, k) / factor.scale[k];
            for (std::size_t l = 0; l < k; ++l)
                value -= factor.l(k, l) * qbasis[i * p + l];
            qbasis[i * p + k] = value / factor.l(k, k);
        }
    double *shuffled = w.grown(w.shuffled, n * batch), *t = w.grown(w.t, batch * p),
           *inverse_scales = w.grown(w.inverse_scales, batch),
           *row_maxima = w.grown(w.row_maxima, batch);
    // Single precision: G_res in float once per gene, in groups of sixteen
    // variants for fused_maxima (zeros past the last variant, whose inverse
    // information is NaN), Q' (p x n) for the fitted values Q T' of a batch,
    // and the batch's residuals in double and, in tiles of six permutations, in
    // float.
    const bool single = precision == PermutationPrecision::single_precision;
    const std::size_t columns = single ? (m + 15) / 16 * 16 : m;
    double *inverse_info = w.grown(w.inverse_info, columns);
    double *qbasis_t = nullptr, *t_t = nullptr, *fitted = nullptr, *cross = nullptr;
    float *packed_g = nullptr, *packed_r = nullptr;
    if (single) {
        qbasis_t = w.grown(w.qbasis_t, p * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < p; ++k)
                qbasis_t[k * n + i] = qbasis[i * p + k];
        // Q S by packed products, 64 samples at a time (each element's sum is
        // the same whatever the block, and the rounding of G_res in double does
        // not survive its conversion to float). Each sample's sixteen variants
        // of a group fill one cache line.
        const std::size_t block_rows = std::min<std::size_t>(n, 64);
        double *block = w.grown(w.block, p * block_rows),
               *projection = w.grown(w.projection, block_rows * m);
        packed_g = w.grown(w.packed_g, n * columns);
        for (std::size_t i0 = 0; i0 < n; i0 += block_rows) {
            const std::size_t rows = std::min(block_rows, n - i0);
            for (std::size_t k = 0; k < p; ++k)
                std::copy_n(&qbasis_t[k * n + i0], rows, &block[k * rows]);
            blocked_product(block, basis.s.data(), p, rows, m, projection);
            for (std::size_t i = i0; i < i0 + rows; ++i) {
                const double *gi = &g[i * m], *pi = &projection[(i - i0) * m];
                float *row = packed_g + i * 16;
                for (std::size_t q = 0; q < m; ++q)
                    row[q / 16 * n * 16 + q % 16] = static_cast<float>(gi[q] - pi[q]);
                for (std::size_t q = m; q < columns; ++q)
                    row[q / 16 * n * 16 + q % 16] = 0.0f;
            }
        }
        packed_r = w.grown(w.packed_r, n * ((batch + 5) / 6 * 6));
        t_t = w.grown(w.t_t, p * batch);
        fitted = w.grown(w.fitted, n * batch);
    } else {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < p; ++k) {
                const double qik = qbasis[i * p + k];
                for (std::size_t q = 0; q < m; ++q)
                    g[i * m + q] -= qik * basis.s[k * m + q];
            }
        cross = w.grown(w.cross, batch * m);
    }
    for (std::size_t q = 0; q < columns; ++q)
        inverse_info[q] =
            q < m && basis.information[q] > 0 ? 1 / std::sqrt(basis.information[q]) : nan;
    double norm = 0;
    for (double value : y)
        norm += value * value;
    // The centre and the sum of squares about it, the same for every permutation.
    double centre = 0, spread = norm;
    if (single && qtl_has_intercept(x)) {
        for (double value : y)
            centre += value;
        centre /= static_cast<double>(n);
        spread = 0;
        for (double value : y)
            spread += (value - centre) * (value - centre);
    }
    // Single precision: column j's residual is its centred response (centred),
    // already in place (fallback), or y_b minus column source[j] of Q T'.
    constexpr std::size_t centred = std::numeric_limits<std::size_t>::max(), fallback = centred - 1;
    std::size_t *source = single ? w.grown(w.source, batch) : nullptr;
    auto &shuffler = w.shuffler;
    if (shuffler.samples() != n)
        shuffler.prepare(n);
    for (auto &draw : w.index)
        w.grown(draw, n);
    w.release();
    Vector yb(n);
    for (std::size_t first = 0; first < permutations; first += batch) {
        const std::size_t count = std::min(batch, permutations - first);
        // Eight permutations at a time, so that each sample's writes share a
        // cache line.
        for (std::size_t j0 = 0; j0 < count; j0 += 8) {
            const std::size_t width = std::min<std::size_t>(8, count - j0);
            for (std::size_t l = 0; l < width; ++l)
                shuffler.shuffle(stream, first + j0 + l, w.index[l].data());
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t l = 0; l < width; ++l)
                    shuffled[i * count + j0 + l] = y[w.index[l][i]];
        }
        blocked_product(shuffled, qbasis, n, count, p, t);
        std::size_t fits = 0;
        for (std::size_t j = 0; j < count; ++j) {
            double rss = norm;
            for (std::size_t k = 0; k < p; ++k)
                rss -= t[j * p + k] * t[j * p + k];
            // Avoid cancellation when almost the whole response lies in
            // the covariate space. This rare refit uses the original
            // least-squares arithmetic, and scores its residual instead
            // of y_b to avoid cancellation in the score as well.
            const bool refit = rss <= 1e-6 * norm;
            if (single)
                source[j] = refit ? fallback : spread <= qtl_centred_ratio * rss ? centred : fits++;
            if (refit) {
                for (std::size_t i = 0; i < n; ++i)
                    yb[i] = shuffled[i * count + j];
                const auto beta = factor.solve(xtv(x, yb));
                rss = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    double fitted_value = 0;
                    for (std::size_t k = 0; k < p; ++k)
                        fitted_value += x(i, k) * beta[k];
                    const double r = yb[i] - fitted_value;
                    shuffled[i * count + j] = r;
                    rss += r * r;
                }
                // The column already is a residual: nothing to subtract below.
                if (single)
                    std::fill_n(&t[j * p], p, 0.0);
            }
            const double scale = rss / static_cast<double>(df);
            inverse_scales[j] = 1 / std::sqrt(scale);
        }
        if (single) {
            // Q T' (n x fits) for the permutations that keep their residual.
            for (std::size_t j = 0; j < count; ++j)
                if (source[j] < fallback)
                    for (std::size_t k = 0; k < p; ++k)
                        t_t[k * fits + source[j]] = t[j * p + k];
            if (fits)
                blocked_product(qbasis_t, t_t, p, n, fits, fitted);
            // The residuals in units of their standard deviation, six
            // permutations at a time (zeros past the last).
            for (std::size_t j0 = 0; j0 < count; j0 += 6) {
                float *tile = packed_r + j0 * n;
                const std::size_t rows = std::min<std::size_t>(6, count - j0);
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t r = 0; r < rows; ++r) {
                        const std::size_t j = j0 + r, e = i * count + j;
                        const double residual = source[j] == fallback ? shuffled[e]
                                                : source[j] == centred
                                                    ? shuffled[e] - centre
                                                    : shuffled[e] - fitted[i * fits + source[j]];
                        tile[i * 6 + r] = static_cast<float>(residual * inverse_scales[j]);
                    }
                    std::fill(tile + i * 6 + rows, tile + i * 6 + 6, 0.0f);
                }
            }
            fused_maxima(packed_r, packed_g, n, count, m, inverse_info, row_maxima);
        } else {
            blocked_product(shuffled, g.data(), n, count, m, cross);
            for (std::size_t j = 0; j < count; ++j)
                row_maxima[j] =
                    qtl_finite_maximum(&cross[j * m], inverse_info, m, inverse_scales[j]);
        }
        for (std::size_t j = 0; j < count; ++j)
            if (std::isfinite(row_maxima[j]))
                maxima.push_back(row_maxima[j]);
            else
                ++failed;
    }
}
} // namespace detail
// Cis-QTL mapping of the phenotypes against the genotypes. Responses are samples
// x response columns, as for sceptre() and score_scan(); genotypes have the same
// samples in the same order.
template <class Responses>
inline CisQtlResult cis_qtl(const Matrix &design, const Responses &responses,
                            const Genotypes &genotypes, const std::vector<CisPhenotype> &phenotypes,
                            const CisQtlOptions &o = {}) {
    const vmath::DivisionScope division(o.model.division);
    design.validate();
    responses.validate();
    genotypes.validate();
    const std::size_t n = design.rows;
    require(responses.rows == n && genotypes.samples == n, "QTL input dimensions mismatch");
    require(o.window >= 0, "the cis window must be non-negative");
    require(o.permutations == 0 || o.permutations >= 10, "use at least 10 permutations, or none");
    require(o.model.offset.empty() || o.model.offset.size() == n, "offset length mismatch");
    require(o.model.weights.empty(), "QTL mapping does not take prior weights");
    for (const auto &ph : phenotypes)
        require(ph.response < responses.cols && ph.start <= ph.end, "invalid phenotype");
    // Variants sorted by chromosome and position, for the window lookups.
    std::vector<std::size_t> order(genotypes.variants);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::tie(genotypes.chrom[a], genotypes.position[a]) <
               std::tie(genotypes.chrom[b], genotypes.position[b]);
    });
    auto options = o.model;
    options.retain_observations = true;
    options.design = prepare_design(design, options.rank_tolerance);
    CisQtlResult result;
    result.genes.resize(phenotypes.size());
    std::vector<std::vector<CisQtlPair>> nominal(phenotypes.size());
    // One workspace per worker thread, reused by each of its genes.
    std::vector<detail::QtlWorkspace> workspaces(static_cast<std::size_t>(o.threads));
    parallel_for(phenotypes.size(), o.threads, [&](std::size_t f) {
        const auto &ph = phenotypes[f];
        auto &gene = result.genes[f];
        gene.phenotype = f;
#ifdef _OPENMP
        const auto worker = static_cast<std::size_t>(omp_get_thread_num());
#else
        const std::size_t worker = 0;
#endif
        require(worker < workspaces.size(), "QTL worker outside its team");
        auto &w = workspaces[worker];
        // The cis variants.
        const auto lower = std::lower_bound(
            order.begin(), order.end(), std::make_pair(ph.chrom, ph.start - o.window),
            [&](std::size_t v, const std::pair<std::string, long long> &key) {
                return std::tie(genotypes.chrom[v], genotypes.position[v]) <
                       std::tie(key.first, key.second);
            });
        auto &cis = w.cis;
        cis.clear();
        for (auto it = lower; it != order.end() && genotypes.chrom[*it] == ph.chrom &&
                              genotypes.position[*it] <= ph.end + o.window;
             ++it)
            cis.push_back(*it);
        gene.variants = cis.size();
        if (cis.empty()) {
            gene.status = "no_cis_variants";
            return;
        }
        const std::size_t m = cis.size();
        // The mean-imputed dosages of the cis variants, decoded here from hard
        // calls. Sixteen variants at a time, so that each sample's writes fill
        // whole cache lines.
        auto &g = w.g;
        w.grown(g, n * m);
        for (std::size_t q0 = 0; q0 < m; q0 += 16) {
            const std::size_t width = std::min<std::size_t>(16, m - q0);
            double mean[16];
            for (std::size_t l = 0; l < width; ++l)
                mean[l] = genotypes.mean_dosage(cis[q0 + l]);
            if (genotypes.hard_calls.empty()) {
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t l = 0; l < width; ++l) {
                        const double d = genotypes.dosage[cis[q0 + l] * n + i];
                        g[i * m + q0 + l] = std::isfinite(d) ? d : mean[l];
                    }
                continue;
            }
            // Each variant's dosage by code, with its mean for the missing code.
            const std::uint8_t *calls[16];
            double dosage[16][4];
            for (std::size_t l = 0; l < width; ++l) {
                calls[l] = &genotypes.hard_calls[cis[q0 + l] * genotypes.call_bytes()];
                std::copy_n(Genotypes::hard_call_dosage, 4, dosage[l]);
                dosage[l][1] = mean[l];
            }
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t byte = i / 4, shift = 2 * (i % 4);
                for (std::size_t l = 0; l < width; ++l)
                    g[i * m + q0 + l] = dosage[l][(calls[l][byte] >> shift) & 3];
            }
        }
        const Vector y = responses.column(ph.response);
        auto observed_fit = fit(design, y, options);
        if (!observed_fit.ok() || observed_fit.has_boundary()) {
            gene.status = std::string("null_") + status_name(observed_fit.status);
            return;
        }
        gene.theta = observed_fit.theta;
        auto &z = w.z, &estimate = w.estimate, &se = w.standard_error;
        // A Gaussian null without prior weights has the same working weights in
        // every refit, so the weight-only part of the scores is computed once.
        const bool fixed_weights = options.family == Family::gaussian && options.weights.empty();
        auto &basis = w.basis;
        basis.factor.reset();
        basis.ready = false;
        const bool scored = detail::qtl_scores(design, observed_fit, g, m, z, &estimate, &se,
                                               fixed_weights ? &basis : nullptr, &w.scores);
        w.release();
        if (!scored) {
            gene.status = "null_singular_information";
            return;
        }
        std::size_t lead = m;
        for (std::size_t q = 0; q < m; ++q)
            if (std::isfinite(z[q]) && (lead == m || std::abs(z[q]) > std::abs(z[lead])))
                lead = q;
        if (lead == m) {
            gene.status = "no_valid_score";
            return;
        }
        gene.lead = cis[lead];
        gene.statistic = z[lead];
        gene.p_value = normal_tail(z[lead], Side::both);
        gene.estimate = estimate[lead];
        gene.standard_error = se[lead];
        gene.status = "ok";
        if (o.nominal) {
            nominal[f].reserve(m);
            for (std::size_t q = 0; q < m; ++q)
                nominal[f].push_back({f, cis[q], z[q],
                                      std::isfinite(z[q]) ? normal_tail(z[q], Side::both) : nan,
                                      estimate[q], se[q]});
        }
        if (o.permutations == 0)
            return;
        // Permutations: the response and its offset shuffled together against the
        // covariates and genotypes, the null refitted (or, for a Gaussian null
        // without weights or offset, the scores projected), and the largest |z| kept.
        const Stream stream = keyed_stream(o.seed, 7, 0, name_key(ph.id));
        auto &maxima = w.maxima;
        maxima.clear();
        const bool batched = detail::qtl_projected_permutations(options);
        if (batched)
            detail::qtl_projected_maxima(design, basis, g, m, y, stream, o.permutations,
                                         o.permutation_precision, maxima, gene.permutations_failed,
                                         &w);
        std::vector<std::size_t> index(n);
        Vector yb(n);
        auto permuted = options;
        for (std::size_t b = 0; b < (batched ? 0 : o.permutations); ++b) {
            detail::qtl_shuffle(stream, b, index);
            for (std::size_t i = 0; i < n; ++i)
                yb[i] = y[index[i]];
            if (!o.model.offset.empty())
                for (std::size_t i = 0; i < n; ++i)
                    permuted.offset[i] = o.model.offset[index[i]];
            const auto refit = fit(design, yb, permuted);
            if (!refit.ok() || refit.has_boundary()) {
                ++gene.permutations_failed;
                continue;
            }
            if (!detail::qtl_scores(design, refit, g, m, z, nullptr, nullptr,
                                    fixed_weights ? &basis : nullptr, &w.scores)) {
                ++gene.permutations_failed;
                continue;
            }
            double maximum = nan;
            for (double v : z)
                if (std::isfinite(v))
                    maximum = std::isfinite(maximum) ? std::max(maximum, std::abs(v)) : std::abs(v);
            if (std::isfinite(maximum))
                maxima.push_back(maximum);
            else
                ++gene.permutations_failed;
        }
        gene.permutations = maxima.size();
        if (maxima.size() < 10) {
            gene.status = "permutations_failed";
            return;
        }
        // A maximum counts as extreme from |z| (1 - tie) up. The observed statistic
        // and the permutation maxima are rounded along different paths, and some
        // designs make them equal in exact arithmetic (with an intercept only and
        // a private variant in every sample, every permutation has the observed
        // largest |z| of those variants), so without the tolerance rounding would
        // decide on which side such ties fall. It is ten times the spread of such
        // ties in single precision (1e-6 relative) and, in double, covers the
        // rounding of the projected scales unless the phenotype's mean exceeds
        // several hundred residual standard deviations.
        const bool single =
            batched && o.permutation_precision == PermutationPrecision::single_precision;
        const double threshold = std::abs(gene.statistic) * (1 - (single ? 1e-5 : 1e-9));
        std::size_t extreme = 0;
        for (double t : maxima)
            extreme += t >= threshold;
        gene.permutation_p =
            (1.0 + static_cast<double>(extreme)) / (1.0 + static_cast<double>(maxima.size()));
        // Float rounding spreads single-precision maxima that are equal in exact
        // arithmetic by a relative variance of about 3e-14; maxima varying by less
        // than 1e-8 (a relative standard deviation of 1e-4) are taken as constant.
        const auto calibration = detail::beta_calibration(
            maxima, single ? 1e-8 : 64 * std::numeric_limits<double>::epsilon());
        gene.reference_ncp = calibration.ncp;
        gene.beta_k = calibration.k;
        gene.beta_n = calibration.n;
        gene.beta_p = detail::beta_adjust(calibration, gene.statistic);
        if (!calibration.ok)
            gene.status = "ok_beta_failed";
    });
    std::size_t pairs = 0;
    for (const auto &rows : nominal)
        pairs += rows.size();
    result.nominal.reserve(pairs);
    for (auto &rows : nominal) {
        result.nominal.insert(result.nominal.end(), rows.begin(), rows.end());
        std::vector<CisQtlPair>().swap(rows);
    }
    return result;
}
} // namespace rapidglm
