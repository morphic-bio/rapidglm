#pragma once
#include "workflows.hpp"
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
// Variant dosages: column v holds variant v's dosage for every sample
// (dosage[v * samples + i]); NaN marks a missing call, which is replaced by the
// variant's mean dosage.
struct Genotypes {
    std::size_t samples = 0, variants = 0;
    std::vector<double> dosage;
    std::vector<std::string> chrom, id;
    std::vector<long long> position;
    void validate() const {
        require(dosage.size() == samples * variants && chrom.size() == variants &&
                    position.size() == variants && (id.empty() || id.size() == variants),
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
// c = A'G (p x m, row k at c + k m), u = G'r and v[q] = sum_i w_i G(i, q)^2 for A
// (n x p) and G (n x m), both row-major. Each sum runs over the samples in
// order from zero; tiles of four rows of c by eight variants are held in
// registers through all samples, and the variants of a tile are adjacent in G.
inline void qtl_cross_generic(const double *a, const double *w, const double *r, const double *g,
                              std::size_t n, std::size_t p, std::size_t m, double *c, double *u,
                              double *v) {
    std::size_t q0 = 0;
    for (; q0 + 8 <= m; q0 += 8) {
        std::size_t k0 = 0;
        for (; k0 + 4 <= p; k0 += 4) {
            double acc[4][8] = {};
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0, *ai = a + i * p + k0;
                for (int kk = 0; kk < 4; ++kk)
                    for (int qq = 0; qq < 8; ++qq)
                        acc[kk][qq] += ai[kk] * gi[qq];
            }
            for (int kk = 0; kk < 4; ++kk)
                for (int qq = 0; qq < 8; ++qq)
                    c[(k0 + kk) * m + q0 + qq] = acc[kk][qq];
        }
        for (; k0 < p; ++k0) {
            double acc[8] = {};
            for (std::size_t i = 0; i < n; ++i)
                for (int qq = 0; qq < 8; ++qq)
                    acc[qq] += a[i * p + k0] * g[i * m + q0 + qq];
            for (int qq = 0; qq < 8; ++qq)
                c[k0 * m + q0 + qq] = acc[qq];
        }
        double su[8] = {}, sv[8] = {};
        for (std::size_t i = 0; i < n; ++i) {
            const double *gi = g + i * m + q0;
            for (int qq = 0; qq < 8; ++qq) {
                su[qq] += r[i] * gi[qq];
                sv[qq] += w[i] * gi[qq] * gi[qq];
            }
        }
        for (int qq = 0; qq < 8; ++qq) {
            u[q0 + qq] = su[qq];
            v[q0 + qq] = sv[qq];
        }
    }
    for (; q0 < m; ++q0) {
        for (std::size_t k = 0; k < p; ++k) {
            double sum = 0;
            for (std::size_t i = 0; i < n; ++i)
                sum += a[i * p + k] * g[i * m + q0];
            c[k * m + q0] = sum;
        }
        double su = 0, sv = 0;
        for (std::size_t i = 0; i < n; ++i) {
            su += r[i] * g[i * m + q0];
            sv += w[i] * g[i * m + q0] * g[i * m + q0];
        }
        u[q0] = su;
        v[q0] = sv;
    }
}
#ifdef RAPIDGLM_PHILOX_AVX2
// The same with AVX2 fused multiply-adds (each term rounded once with its sum),
// for whole tiles; the rest as in qtl_cross_generic.
__attribute__((target("avx2,fma"))) inline void
qtl_cross_avx2(const double *a, const double *w, const double *r, const double *g, std::size_t n,
               std::size_t p, std::size_t m, double *c, double *u, double *v) {
    const std::size_t tiled = m / 8 * 8;
    for (std::size_t q0 = 0; q0 < tiled; q0 += 8) {
        std::size_t k0 = 0;
        for (; k0 + 4 <= p; k0 += 4) {
            __m256d acc[4][2];
            for (auto &row : acc)
                row[0] = row[1] = _mm256_setzero_pd();
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0, *ai = a + i * p + k0;
                const __m256d g0 = _mm256_loadu_pd(gi), g1 = _mm256_loadu_pd(gi + 4);
                for (int kk = 0; kk < 4; ++kk) {
                    const __m256d b = _mm256_broadcast_sd(ai + kk);
                    acc[kk][0] = _mm256_fmadd_pd(b, g0, acc[kk][0]);
                    acc[kk][1] = _mm256_fmadd_pd(b, g1, acc[kk][1]);
                }
            }
            for (int kk = 0; kk < 4; ++kk) {
                _mm256_storeu_pd(c + (k0 + kk) * m + q0, acc[kk][0]);
                _mm256_storeu_pd(c + (k0 + kk) * m + q0 + 4, acc[kk][1]);
            }
        }
        for (; k0 < p; ++k0) {
            __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
            for (std::size_t i = 0; i < n; ++i) {
                const double *gi = g + i * m + q0;
                const __m256d b = _mm256_broadcast_sd(a + i * p + k0);
                s0 = _mm256_fmadd_pd(b, _mm256_loadu_pd(gi), s0);
                s1 = _mm256_fmadd_pd(b, _mm256_loadu_pd(gi + 4), s1);
            }
            _mm256_storeu_pd(c + k0 * m + q0, s0);
            _mm256_storeu_pd(c + k0 * m + q0 + 4, s1);
        }
        __m256d u0 = _mm256_setzero_pd(), u1 = _mm256_setzero_pd(), v0 = _mm256_setzero_pd(),
                v1 = _mm256_setzero_pd();
        for (std::size_t i = 0; i < n; ++i) {
            const double *gi = g + i * m + q0;
            const __m256d g0 = _mm256_loadu_pd(gi), g1 = _mm256_loadu_pd(gi + 4);
            const __m256d ri = _mm256_broadcast_sd(r + i), wi = _mm256_broadcast_sd(w + i);
            u0 = _mm256_fmadd_pd(ri, g0, u0);
            u1 = _mm256_fmadd_pd(ri, g1, u1);
            v0 = _mm256_fmadd_pd(_mm256_mul_pd(wi, g0), g0, v0);
            v1 = _mm256_fmadd_pd(_mm256_mul_pd(wi, g1), g1, v1);
        }
        _mm256_storeu_pd(u + q0, u0);
        _mm256_storeu_pd(u + q0 + 4, u1);
        _mm256_storeu_pd(v + q0, v0);
        _mm256_storeu_pd(v + q0 + 4, v1);
    }
    if (tiled < m) {
        // The last variants: their columns of G as a block of their own.
        const std::size_t rest = m - tiled;
        std::vector<double> block(n * rest), cr(p * rest), ur(rest), vr(rest);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t q = 0; q < rest; ++q)
                block[i * rest + q] = g[i * m + tiled + q];
        qtl_cross_generic(a, w, r, block.data(), n, p, rest, cr.data(), ur.data(), vr.data());
        for (std::size_t k = 0; k < p; ++k)
            for (std::size_t q = 0; q < rest; ++q)
                c[k * m + tiled + q] = cr[k * rest + q];
        for (std::size_t q = 0; q < rest; ++q) {
            u[tiled + q] = ur[q];
            v[tiled + q] = vr[q];
        }
    }
}
inline bool fma_available() {
    static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return available;
}
#endif
inline void qtl_cross(const double *a, const double *w, const double *r, const double *g,
                      std::size_t n, std::size_t p, std::size_t m, double *c, double *u,
                      double *v) {
#ifdef RAPIDGLM_PHILOX_AVX2
    if (fma_available())
        return qtl_cross_avx2(a, w, r, g, n, p, m, c, u, v);
#endif
    qtl_cross_generic(a, w, r, g, n, p, m, c, u, v);
}
// What a gene's scores take from the null's working weights alone: the factor
// of X'WX, s = L^-1 D^-1 X'WG (p x m, row k at s + k m) and the information
// v - |s|^2 of each variant. With weights that do not change between fits (a
// Gaussian null without prior weights) it serves every permutation.
struct QtlBasis {
    std::optional<Cholesky> factor;
    std::vector<double> s, information;
    bool ready = false;
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
// sums over samples taken for all variants at once (qtl_cross). `g` holds the
// variants sample by sample (g[i m + q]). z (and, if asked, estimates and SEs)
// for every variant; `basis` (may be null) holds the weight-only part: computed
// when not ready, reused when it is.
inline void qtl_scores(const Matrix &x, const FitResult &null, const std::vector<double> &g,
                       std::size_t m, std::vector<double> &z, std::vector<double> *estimate,
                       std::vector<double> *standard_error, QtlBasis *basis = nullptr) {
    const std::size_t n = x.rows, p = x.cols;
    const auto &w = null.working_weights;
    const auto &r = null.score_residuals;
    QtlBasis own;
    QtlBasis &b = basis ? *basis : own;
    std::vector<double> u(m);
    if (!b.ready) {
        // The factor is of the equilibrated matrix D^-1 (X'WX) D^-1 (D: the root
        // diagonal), so vectors are divided by D before the forward substitution.
        b.factor.emplace(crossprod(x, w));
        require(b.factor->ok, "singular null information");
        const auto &ch = *b.factor;
        const auto &l = ch.l;
        // c = X'WG, u = G'r and v = G'WG (diagonal) by detail::qtl_cross.
        std::vector<double> v(m), a(n * p);
        b.s.assign(p * m, 0.0);
        auto &c = b.s;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < p; ++k)
                a[i * p + k] = w[i] * x(i, k);
        qtl_cross(a.data(), w.data(), r.data(), g.data(), n, p, m, c.data(), u.data(), v.data());
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
    std::vector<double> st(m, 0.0);
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
               count * (std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
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
// statistic z is then BetaCDF(sf(z^2)).
struct BetaCalibration {
    bool ok = false;
    double ncp = 0, k = nan, n = nan;
};
inline BetaCalibration beta_calibration(const std::vector<double> &maxima) {
    BetaCalibration out;
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
    return out;
}
inline double beta_adjust(const BetaCalibration &calibration, double z) {
    if (!calibration.ok || !std::isfinite(z))
        return nan;
    const double p = chi1_sf(z * z, calibration.ncp);
    return boost::math::cdf(boost::math::beta_distribution<>(calibration.k, calibration.n),
                            std::clamp(p, 0.0, 1.0));
}
} // namespace detail
// Cis-QTL mapping of the phenotypes against the genotypes. Responses are samples
// x response columns, as for sceptre() and score_scan(); genotypes have the same
// samples in the same order.
template <class Responses>
inline CisQtlResult cis_qtl(const Matrix &design, const Responses &responses,
                            const Genotypes &genotypes, const std::vector<CisPhenotype> &phenotypes,
                            const CisQtlOptions &o = {}) {
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
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::tie(genotypes.chrom[a], genotypes.position[a]) <
               std::tie(genotypes.chrom[b], genotypes.position[b]);
    });
    // Mean-imputed dosages.
    std::vector<double> mean_dosage(genotypes.variants, 0.0);
    for (std::size_t v = 0; v < genotypes.variants; ++v) {
        double s = 0;
        std::size_t k = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = genotypes.dosage[v * n + i];
            if (std::isfinite(d)) {
                s += d;
                ++k;
            }
        }
        mean_dosage[v] = k ? s / static_cast<double>(k) : 0.0;
    }
    auto options = o.model;
    options.retain_observations = true;
    options.design = prepare_design(design, options.rank_tolerance);
    CisQtlResult result;
    result.genes.resize(phenotypes.size());
    std::vector<std::vector<CisQtlPair>> nominal(phenotypes.size());
    parallel_for(phenotypes.size(), o.threads, [&](std::size_t f) {
        const auto &ph = phenotypes[f];
        auto &gene = result.genes[f];
        gene.phenotype = f;
        // The cis variants.
        const auto lower = std::lower_bound(
            order.begin(), order.end(), std::make_pair(ph.chrom, ph.start - o.window),
            [&](std::size_t v, const std::pair<std::string, long long> &key) {
                return std::tie(genotypes.chrom[v], genotypes.position[v]) <
                       std::tie(key.first, key.second);
            });
        std::vector<std::size_t> cis;
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
        std::vector<double> g(n * m);
        for (std::size_t q = 0; q < m; ++q)
            for (std::size_t i = 0; i < n; ++i) {
                const double d = genotypes.dosage[cis[q] * n + i];
                g[i * m + q] = std::isfinite(d) ? d : mean_dosage[cis[q]];
            }
        const Vector y = responses.column(ph.response);
        auto observed_fit = fit(design, y, options);
        if (!observed_fit.ok() || observed_fit.has_boundary()) {
            gene.status = std::string("null_") + status_name(observed_fit.status);
            return;
        }
        gene.theta = observed_fit.theta;
        std::vector<double> z, estimate, se;
        // A Gaussian null without prior weights has the same working weights in
        // every refit, so the weight-only part of the scores is computed once.
        const bool fixed_weights = options.family == Family::gaussian && options.weights.empty();
        detail::QtlBasis basis;
        detail::qtl_scores(design, observed_fit, g, m, z, &estimate, &se,
                           fixed_weights ? &basis : nullptr);
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
        if (o.nominal)
            for (std::size_t q = 0; q < m; ++q)
                nominal[f].push_back({f, cis[q], z[q],
                                      std::isfinite(z[q]) ? normal_tail(z[q], Side::both) : nan,
                                      estimate[q], se[q]});
        if (o.permutations == 0)
            return;
        // Permutations: the response and its offset shuffled together against the
        // covariates and genotypes, the null refitted, and the largest |z| kept.
        const Stream stream = keyed_stream(o.seed, 7, 0, name_key(ph.id));
        std::vector<double> maxima;
        std::vector<std::size_t> index(n);
        Vector yb(n);
        const auto permute = [&](std::size_t b) {
            std::iota(index.begin(), index.end(), 0);
            CounterRng rng(stream.seed, stream.test, b);
            for (std::size_t r = n; r > 1; --r)
                std::swap(index[r - 1], index[static_cast<std::size_t>(rng.bounded(r))]);
            for (std::size_t i = 0; i < n; ++i)
                yb[i] = y[index[i]];
        };
        const bool batched =
            fixed_weights && options.offset.empty() && options.ridge == 0 && !options.robust;
        if (batched) {
            // A Gaussian null refitted to a permuted response is its least-squares
            // residual, r = y - X b with X'X b = X'y, and scale |r|^2 / (n - p); the
            // permutations are scored in batches, G'R for a batch of residuals at
            // once (detail::qtl_cross), with the basis of the observed fit.
            const auto &factor = *basis.factor;
            const std::size_t p = design.cols, batch = 64, df = n - p;
            std::vector<double> residuals(n * batch), cross(batch * m), ignored_u(m), ignored_v(m),
                ones(n, 1.0), zero(n, 0.0), t(batch * p), scales(batch);
            for (std::size_t first = 0; first < o.permutations; first += batch) {
                const std::size_t count = std::min(batch, o.permutations - first);
                for (std::size_t j = 0; j < count; ++j) {
                    permute(first + j);
                    const Vector xy = detail::xtv(design, yb);
                    const auto beta = factor.solve(xy);
                    double rss = 0;
                    for (std::size_t i = 0; i < n; ++i) {
                        double fitted = 0;
                        for (std::size_t k = 0; k < p; ++k)
                            fitted += design(i, k) * beta[k];
                        const double r = yb[i] - fitted;
                        residuals[i * count + j] = r;
                        rss += r * r;
                    }
                    scales[j] = rss / static_cast<double>(df);
                    // t = L^-1 D^-1 X'r, zero but for rounding, as in qtl_scores.
                    Vector xr(p, 0.0);
                    for (std::size_t i = 0; i < n; ++i)
                        for (std::size_t k = 0; k < p; ++k)
                            xr[k] += design(i, k) * residuals[i * count + j];
                    for (std::size_t k = 0; k < p; ++k) {
                        double value = xr[k] / factor.scale[k];
                        for (std::size_t l = 0; l < k; ++l)
                            value -= factor.l(k, l) * t[j * p + l];
                        t[j * p + k] = value / factor.l(k, k);
                    }
                }
                detail::qtl_cross(residuals.data(), ones.data(), zero.data(), g.data(), n, count, m,
                                  cross.data(), ignored_u.data(), ignored_v.data());
                for (std::size_t j = 0; j < count; ++j) {
                    double maximum = 0;
                    for (std::size_t q = 0; q < m; ++q) {
                        const double info = basis.information[q];
                        if (!(info > 0))
                            continue;
                        double st = 0;
                        for (std::size_t k = 0; k < p; ++k)
                            st += basis.s[k * m + q] * t[j * p + k];
                        const double score = cross[j * m + q] - st,
                                     value = std::abs(score / std::sqrt(info * scales[j]));
                        if (std::isfinite(value))
                            maximum = std::max(maximum, value);
                    }
                    maxima.push_back(maximum);
                }
            }
        }
        auto permuted = options;
        for (std::size_t b = 0; b < (batched ? 0 : o.permutations); ++b) {
            permute(b);
            if (!o.model.offset.empty())
                for (std::size_t i = 0; i < n; ++i)
                    permuted.offset[i] = o.model.offset[index[i]];
            const auto refit = fit(design, yb, permuted);
            if (!refit.ok() || refit.has_boundary()) {
                ++gene.permutations_failed;
                continue;
            }
            detail::qtl_scores(design, refit, g, m, z, nullptr, nullptr,
                               fixed_weights ? &basis : nullptr);
            double maximum = 0;
            for (double v : z)
                if (std::isfinite(v))
                    maximum = std::max(maximum, std::abs(v));
            maxima.push_back(maximum);
        }
        gene.permutations = maxima.size();
        if (maxima.size() < 10) {
            gene.status = "permutations_failed";
            return;
        }
        std::size_t extreme = 0;
        for (double t : maxima)
            extreme += t >= std::abs(gene.statistic);
        gene.permutation_p =
            (1.0 + static_cast<double>(extreme)) / (1.0 + static_cast<double>(maxima.size()));
        const auto calibration = detail::beta_calibration(maxima);
        gene.reference_ncp = calibration.ncp;
        gene.beta_k = calibration.k;
        gene.beta_n = calibration.n;
        gene.beta_p = detail::beta_adjust(calibration, gene.statistic);
        if (!calibration.ok)
            gene.status = "ok_beta_failed";
    });
    for (auto &rows : nominal)
        for (auto &row : rows)
            result.nominal.push_back(row);
    return result;
}
} // namespace rapidglm
