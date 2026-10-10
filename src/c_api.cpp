// C interface over the RapidGLM C++ library (include/rapidglm/rapidglm_c.h).
#include <rapidglm/rapidglm.hpp>
#include <rapidglm/rapidglm_c.h>

#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

using namespace rapidglm;

struct rapidglm_table {
    struct Column {
        std::string name;
        int type = RAPIDGLM_COLUMN_NUMERIC;
        Vector numbers;
        std::vector<std::string> text;
    };
    std::size_t rows = 0;
    std::deque<Column> columns; // stable references while columns are added
    std::vector<std::pair<std::string, double>> scalars;
    std::vector<std::pair<std::string, Vector>> vectors;
    Vector &numeric(const std::string &name) {
        columns.push_back({name, RAPIDGLM_COLUMN_NUMERIC, Vector(rows, rapidglm::nan), {}});
        return columns.back().numbers;
    }
    std::vector<std::string> &text(const std::string &name) {
        columns.push_back({name, RAPIDGLM_COLUMN_TEXT, {}, std::vector<std::string>(rows)});
        return columns.back().text;
    }
    const Column *find(const char *name, int type) const {
        if (!name)
            return nullptr;
        for (const auto &c : columns)
            if (c.type == type && c.name == name)
                return &c;
        return nullptr;
    }
};

namespace {
thread_local std::string last_error;

template <class Function> int guarded(Function function) {
    try {
        function();
        last_error.clear();
        return RAPIDGLM_OK;
    } catch (const std::invalid_argument &e) {
        last_error = e.what();
        return RAPIDGLM_ERROR_INVALID;
    } catch (const std::exception &e) {
        last_error = e.what();
        return RAPIDGLM_ERROR_FAILED;
    } catch (...) {
        last_error = "unknown error";
        return RAPIDGLM_ERROR_FAILED;
    }
}
// The caller's options over this version's defaults: a caller built against an
// older header passes a shorter struct, and the fields it lacks keep their defaults.
template <class Options, class... Init>
Options effective(const Options *o, void (*init)(Options *, Init...), Init... arguments) {
    require(o != nullptr && o->struct_size >= sizeof(std::size_t) &&
                o->struct_size <= sizeof(Options),
            "options must be initialized with the matching *_init function");
    Options c;
    init(&c, arguments...);
    std::memcpy(static_cast<void *>(&c), o, o->struct_size);
    c.struct_size = sizeof(Options);
    return c;
}
Matrix dense(const rapidglm_dense *m) {
    require(m != nullptr && m->rows > 0 && m->cols > 0 && m->values != nullptr,
            "invalid dense matrix");
    Matrix x(m->rows, m->cols);
    std::copy(m->values, m->values + m->rows * m->cols, x.values.begin());
    x.validate();
    return x;
}
SparseCSC sparse(const rapidglm_sparse *m) {
    require(m != nullptr && m->column_ptr != nullptr, "invalid sparse matrix");
    SparseCSC s;
    s.rows = m->rows;
    s.cols = m->cols;
    s.column_ptr.assign(m->column_ptr, m->column_ptr + m->cols + 1);
    const auto nnz = s.column_ptr.back();
    require(nnz == 0 || (m->row_index != nullptr && m->values != nullptr), "invalid sparse matrix");
    s.row_index.assign(m->row_index, m->row_index + nnz);
    s.values.assign(m->values, m->values + nnz);
    s.validate();
    return s;
}
Vector optional_vector(const double *values, std::size_t n) {
    return values ? Vector(values, values + n) : Vector();
}
Family family(int f) {
    require(f >= RAPIDGLM_GAUSSIAN && f <= RAPIDGLM_NEGATIVE_BINOMIAL, "unknown family");
    return static_cast<Family>(f);
}
FitOptions fit_options(const rapidglm_fit_options &c, std::size_t n) {
    FitOptions o;
    o.family = family(c.family);
    require(c.dispersion >= RAPIDGLM_DISPERSION_ML && c.dispersion <= RAPIDGLM_DISPERSION_FIXED,
            "unknown dispersion");
    o.dispersion = static_cast<Dispersion>(c.dispersion);
    o.theta = c.theta;
    o.tolerance = c.tolerance;
    o.ridge = c.ridge;
    o.max_iterations = c.max_iterations;
    o.offset = optional_vector(c.offset, n);
    o.weights = optional_vector(c.weights, n);
    o.robust = c.robust != 0;
    o.cox_reid = c.cox_reid != 0;
    o.singular_ok = c.singular_ok != 0;
    return o;
}
Side side(int s) {
    require(s >= RAPIDGLM_SIDE_BOTH && s <= RAPIDGLM_SIDE_RIGHT, "unknown side");
    return s == RAPIDGLM_SIDE_LEFT    ? Side::left
           : s == RAPIDGLM_SIDE_RIGHT ? Side::right
                                      : Side::both;
}
EffectFits effect_fits(int e) {
    require(e >= RAPIDGLM_EFFECTS_SIGNIFICANT && e <= RAPIDGLM_EFFECTS_NONE, "unknown effect fits");
    return e == RAPIDGLM_EFFECTS_ALL    ? EffectFits::all
           : e == RAPIDGLM_EFFECTS_NONE ? EffectFits::none
                                        : EffectFits::significant;
}
CrtDraw crt_draw(int d) {
    require(d >= RAPIDGLM_CRT_THINNED && d <= RAPIDGLM_CRT_HAZARD, "unknown crt_draw");
    return d == RAPIDGLM_CRT_DIRECT   ? CrtDraw::direct
           : d == RAPIDGLM_CRT_HAZARD ? CrtDraw::hazard
                                      : CrtDraw::thinned;
}
FitCheck fit_check(int c) {
    require(c == RAPIDGLM_FIT_CHECK_STRICT || c == RAPIDGLM_FIT_CHECK_KS, "unknown fit_check");
    return c == RAPIDGLM_FIT_CHECK_KS ? FitCheck::ks : FitCheck::strict;
}
void budgets(ResampleOptions &r, const std::size_t *values, std::size_t n) {
    if (values)
        r.budgets.assign(values, values + n);
}
std::string text_or_empty(const char *const *values, std::size_t i) {
    require(values != nullptr && values[i] != nullptr, "missing text value");
    return values[i];
}
StreamKeys stream_keys(int k) {
    require(k == RAPIDGLM_STREAMS_IDENTITY || k == RAPIDGLM_STREAMS_POSITION,
            "unknown stream_keys");
    return k == RAPIDGLM_STREAMS_POSITION ? StreamKeys::position : StreamKeys::identity;
}
AdjustmentFamily adjustment_family(int f) {
    require(f == RAPIDGLM_FAMILY_TESTED || f == RAPIDGLM_FAMILY_LISTED,
            "unknown adjustment family");
    return f == RAPIDGLM_FAMILY_LISTED ? AdjustmentFamily::listed : AdjustmentFamily::tested;
}
// The inputs of the perturbation workflow shared by rapidglm_sceptre and
// rapidglm_sceptre_grid.
std::vector<Guide> sceptre_guides(std::size_t n, const char *const *ids, const char *const *targets,
                                  const int *non_targeting) {
    std::vector<Guide> guides(n);
    for (std::size_t j = 0; j < n; ++j)
        guides[j] = {text_or_empty(ids, j), text_or_empty(targets, j),
                     non_targeting && non_targeting[j] != 0};
    return guides;
}
std::vector<Pair> sceptre_pairs(std::size_t n, const std::size_t *responses,
                                const char *const *targets, const int *kinds) {
    require(n == 0 || (responses && kinds), "pair arrays required");
    std::vector<Pair> pairs(n);
    for (std::size_t k = 0; k < n; ++k) {
        require(kinds[k] >= RAPIDGLM_PAIR_DISCOVERY && kinds[k] <= RAPIDGLM_PAIR_POSITIVE,
                "unknown pair kind");
        pairs[k] = {responses[k], text_or_empty(targets, k),
                    kinds[k] == RAPIDGLM_PAIR_NEGATIVE   ? PairKind::negative_control
                    : kinds[k] == RAPIDGLM_PAIR_POSITIVE ? PairKind::positive_control
                                                         : PairKind::discovery};
    }
    return pairs;
}
SceptreOptions sceptre_options(const rapidglm_sceptre_options &c, std::size_t cells,
                               std::size_t responses) {
    SceptreOptions o;
    o.model = fit_options(c.model, cells);
    require(c.null_model == RAPIDGLM_NULL_SCEPTRE || c.null_model == RAPIDGLM_NULL_DIRECT,
            "unknown null model");
    o.null_model =
        c.null_model == RAPIDGLM_NULL_SCEPTRE ? FitMethod::sceptre_null : FitMethod::direct;
    o.high_moi = c.high_moi != 0;
    o.nt_controls = c.nt_controls != 0;
    o.crt = c.crt != 0;
    require(c.integration >= RAPIDGLM_INTEGRATION_UNION &&
                c.integration <= RAPIDGLM_INTEGRATION_BONFERRONI,
            "unknown integration");
    o.integration = c.integration == RAPIDGLM_INTEGRATION_SINGLETON    ? Integration::singleton
                    : c.integration == RAPIDGLM_INTEGRATION_BONFERRONI ? Integration::bonferroni
                                                                       : Integration::union_guides;
    o.resampling.side = side(c.side);
    o.resampling.approximate = c.approximate != 0;
    o.resampling.fit_check = fit_check(c.fit_check);
    o.resampling.confirm = c.confirm_draws;
    budgets(o.resampling, c.budgets, c.n_budgets);
    o.resampling.screen = c.screen;
    o.resampling.seed = c.seed;
    o.effect_fits = effect_fits(c.effect_fits);
    o.cell_qc = c.cell_qc != 0;
    o.assignment_qc = c.assignment_qc != 0;
    o.min_nonzero = c.min_nonzero;
    o.alpha = c.alpha;
    o.lower_quantile = c.lower_quantile;
    o.upper_quantile = c.upper_quantile;
    o.max_mito = c.max_mito;
    o.threads = c.threads;
    o.bonferroni_adjustment = c.bonferroni != 0;
    o.crt_draw = crt_draw(c.crt_draw);
    o.stream_keys = stream_keys(c.stream_keys);
    if (c.response_ids)
        for (std::size_t j = 0; j < responses; ++j)
            o.response_ids.push_back(text_or_empty(c.response_ids, j));
    o.family = adjustment_family(c.family);
    std::copy(c.family_size, c.family_size + 3, o.family_size.begin());
    return o;
}
CellQC sceptre_qc(const rapidglm_sceptre_options &c, std::size_t cells) {
    CellQC qc;
    qc.total_counts = optional_vector(c.qc_total, cells);
    qc.detected_features = optional_vector(c.qc_detected, cells);
    qc.mitochondrial_fraction = optional_vector(c.qc_mito, cells);
    return qc;
}
std::unique_ptr<rapidglm_table> sceptre_table(const SceptreResult &r) {
    auto t = std::make_unique<rapidglm_table>();
    t->rows = r.tests.size();
    auto &response = t->numeric("response");
    auto &target = t->text("target");
    auto &guide = t->text("guide");
    auto &kind = t->numeric("kind");
    auto &status = t->text("status");
    auto &cells = t->numeric("cells");
    auto &nt = t->numeric("nonzero_treatment");
    auto &nc = t->numeric("nonzero_control");
    auto &draws = t->numeric("draws");
    auto &approximation = t->numeric("approximation");
    auto &statistic = t->numeric("statistic");
    auto &p = t->numeric("p_value");
    auto &q = t->numeric("adjusted_p");
    auto &lfc = t->numeric("log2fc");
    auto &lfc_se = t->numeric("log2fc_se");
    auto &ratio = t->numeric("sceptre_mean_ratio");
    auto &ratio_log2 = t->numeric("mean_ratio_log2");
    for (std::size_t i = 0; i < r.tests.size(); ++i) {
        const auto &e = r.tests[i];
        response[i] = static_cast<double>(e.response);
        target[i] = e.target;
        guide[i] = e.guide;
        kind[i] = static_cast<int>(e.kind);
        status[i] = e.status;
        cells[i] = static_cast<double>(e.cells);
        nt[i] = static_cast<double>(e.nonzero_treatment);
        nc[i] = static_cast<double>(e.nonzero_control);
        draws[i] = static_cast<double>(e.draws);
        approximation[i] = e.approximation;
        statistic[i] = e.statistic;
        p[i] = e.p_value;
        q[i] = e.adjusted_p;
        lfc[i] = e.log2fc;
        lfc_se[i] = e.log2fc_se;
        ratio[i] = e.sceptre_mean_ratio;
        ratio_log2[i] = e.mean_ratio_log2;
    }
    t->scalars = {{"retained_cells", static_cast<double>(r.retained_cells)},
                  {"calibration_rejections", static_cast<double>(r.calibration_rejections)},
                  {"power_rejections", static_cast<double>(r.power_rejections)},
                  {"discovery_rejections", static_cast<double>(r.discovery_rejections)},
                  {"null_fits", static_cast<double>(r.null_fits)}};
    // By kind, in the order of RAPIDGLM_PAIR_*.
    const auto by_kind = [&](const char *name, const std::array<std::size_t, 3> &counts) {
        const char *const kinds[] = {"discovery", "negative", "positive"};
        for (std::size_t k = 0; k < 3; ++k)
            t->scalars.emplace_back(std::string(name) + kinds[k], static_cast<double>(counts[k]));
    };
    by_kind("pairs_listed_", r.pairs_listed);
    by_kind("pairs_passing_qc_", r.pairs_passing_qc);
    by_kind("family_size_", r.family_size);
    return t;
}
} // namespace

extern "C" {

const char *rapidglm_version(void) {
    return rapidglm::version;
}
int rapidglm_c_api_version(void) {
    return RAPIDGLM_C_API_VERSION;
}
const char *rapidglm_last_error(void) {
    return last_error.c_str();
}

size_t rapidglm_table_rows(const rapidglm_table *t) {
    return t ? t->rows : 0;
}
size_t rapidglm_table_columns(const rapidglm_table *t) {
    return t ? t->columns.size() : 0;
}
const char *rapidglm_table_column_name(const rapidglm_table *t, size_t k) {
    return t && k < t->columns.size() ? t->columns[k].name.c_str() : nullptr;
}
int rapidglm_table_column_type(const rapidglm_table *t, size_t k) {
    return t && k < t->columns.size() ? t->columns[k].type : -1;
}
const double *rapidglm_table_numeric(const rapidglm_table *t, const char *name) {
    const auto *c = t ? t->find(name, RAPIDGLM_COLUMN_NUMERIC) : nullptr;
    return c ? c->numbers.data() : nullptr;
}
const char *rapidglm_table_text(const rapidglm_table *t, const char *name, size_t row) {
    const auto *c = t ? t->find(name, RAPIDGLM_COLUMN_TEXT) : nullptr;
    return c && row < c->text.size() ? c->text[row].c_str() : nullptr;
}
size_t rapidglm_table_scalars(const rapidglm_table *t) {
    return t ? t->scalars.size() : 0;
}
const char *rapidglm_table_scalar_name(const rapidglm_table *t, size_t k) {
    return t && k < t->scalars.size() ? t->scalars[k].first.c_str() : nullptr;
}
double rapidglm_table_scalar(const rapidglm_table *t, const char *name) {
    if (t && name)
        for (const auto &s : t->scalars)
            if (s.first == name)
                return s.second;
    return rapidglm::nan;
}
size_t rapidglm_table_vectors(const rapidglm_table *t) {
    return t ? t->vectors.size() : 0;
}
const char *rapidglm_table_vector_name(const rapidglm_table *t, size_t k) {
    return t && k < t->vectors.size() ? t->vectors[k].first.c_str() : nullptr;
}
const double *rapidglm_table_vector(const rapidglm_table *t, const char *name, size_t *length) {
    if (t && name)
        for (const auto &v : t->vectors)
            if (v.first == name) {
                if (length)
                    *length = v.second.size();
                return v.second.data();
            }
    if (length)
        *length = 0;
    return nullptr;
}
void rapidglm_table_free(rapidglm_table *t) {
    delete t;
}

void rapidglm_fit_options_init(rapidglm_fit_options *o) {
    if (!o)
        return;
    std::memset(o, 0, sizeof *o);
    o->struct_size = sizeof *o;
    o->family = RAPIDGLM_NEGATIVE_BINOMIAL;
    o->dispersion = RAPIDGLM_DISPERSION_ML;
    o->theta = 10;
    o->tolerance = 1e-9;
    o->max_iterations = 100;
}

int rapidglm_fit_batch(const rapidglm_dense *design, const rapidglm_sparse *responses,
                       const rapidglm_fit_options *options, int threads, rapidglm_table **result) {
    return guarded([&] {
        require(result != nullptr, "result pointer required");
        const auto c = effective(options, rapidglm_fit_options_init);
        const auto x = dense(design);
        const auto y = sparse(responses);
        const auto fits = fit_batch(x, y, fit_options(c, x.rows), threads);
        auto t = std::make_unique<rapidglm_table>();
        t->rows = fits.size();
        auto &feature = t->numeric("feature");
        auto &status = t->text("status");
        auto &theta = t->numeric("theta");
        auto &theta_boundary = t->numeric("theta_boundary");
        auto &ll = t->numeric("log_likelihood");
        auto &iterations = t->numeric("iterations");
        for (std::size_t i = 0; i < fits.size(); ++i) {
            feature[i] = static_cast<double>(i);
            status[i] = status_name(fits[i].status);
            theta[i] = fits[i].theta;
            theta_boundary[i] = fits[i].theta_boundary;
            ll[i] = fits[i].log_likelihood;
            iterations[i] = static_cast<double>(fits[i].iterations);
        }
        for (std::size_t j = 0; j < x.cols; ++j) {
            auto &beta = t->numeric("beta" + std::to_string(j));
            auto &se = t->numeric("se" + std::to_string(j));
            auto &p = t->numeric("wald_p" + std::to_string(j));
            auto &boundary = t->numeric("boundary" + std::to_string(j));
            for (std::size_t i = 0; i < fits.size(); ++i) {
                const auto &f = fits[i];
                if (f.coefficients.size() == x.cols) {
                    beta[i] = f.coefficients[j];
                    se[i] = f.standard_errors[j];
                    p[i] = wald_test(f, j).p_value;
                }
                boundary[i] = !f.boundary_coefficients.empty() && f.boundary_coefficients[j];
            }
        }
        *result = t.release();
    });
}

void rapidglm_bulk_options_init(rapidglm_bulk_options *o) {
    if (!o)
        return;
    std::memset(o, 0, sizeof *o);
    const BulkOptions d;
    o->struct_size = sizeof *o;
    o->coefficient = SIZE_MAX; // the last column
    o->size_factor_method = RAPIDGLM_SIZE_FACTORS_RATIO;
    o->trend = RAPIDGLM_TREND_PARAMETRIC;
    o->min_mean = d.min_mean;
    o->min_dispersion = d.min_dispersion;
    o->max_dispersion = d.max_dispersion;
    o->legacy_cooks = d.legacy_cooks;
    o->outlier_sd = d.outlier_sd;
    o->cooks_cutoff = d.cooks_cutoff;
    o->alpha = d.alpha;
    o->keep_minimum_start = d.keep_minimum_start;
    o->cooks_filter = d.cooks_filter;
    o->independent_filtering = d.independent_filtering;
    o->min_replicates_for_replace = d.min_replicates_for_replace;
    o->lfc_shrinkage = RAPIDGLM_SHRINK_NONE;
    o->seed = d.seed;
    o->threads = 1;
}

int rapidglm_bulk_de(const rapidglm_dense *design, const rapidglm_sparse *counts,
                     const rapidglm_bulk_options *options, rapidglm_table **result) {
    return guarded([&] {
        require(result != nullptr, "result pointer required");
        const auto c = effective(options, rapidglm_bulk_options_init);
        const auto x = dense(design);
        const auto y = sparse(counts);
        BulkOptions o;
        o.coefficient = c.coefficient == SIZE_MAX ? x.cols - 1 : c.coefficient;
        if (c.contrasts) {
            require(c.contrast_rows > 0, "contrast_rows must be positive");
            o.hypothesis.contrasts = Matrix(c.contrast_rows, x.cols);
            std::copy(c.contrasts, c.contrasts + c.contrast_rows * x.cols,
                      o.hypothesis.contrasts.values.begin());
        }
        o.lrt = c.lrt != 0;
        require(c.size_factor_method == RAPIDGLM_SIZE_FACTORS_RATIO ||
                    c.size_factor_method == RAPIDGLM_SIZE_FACTORS_POSCOUNTS,
                "unknown size factor method");
        o.size_factor_method = c.size_factor_method == RAPIDGLM_SIZE_FACTORS_POSCOUNTS
                                   ? SizeFactorMethod::poscounts
                                   : SizeFactorMethod::ratio;
        o.size_factors = optional_vector(c.size_factors, x.rows);
        require(c.trend >= RAPIDGLM_TREND_PARAMETRIC && c.trend <= RAPIDGLM_TREND_MEAN,
                "unknown trend");
        o.trend = static_cast<DispersionTrend>(c.trend);
        o.min_mean = c.min_mean;
        o.min_dispersion = c.min_dispersion;
        o.max_dispersion = c.max_dispersion;
        o.legacy_cooks = c.legacy_cooks != 0;
        o.outlier_sd = c.outlier_sd;
        o.cooks_cutoff = c.cooks_cutoff;
        o.alpha = c.alpha;
        o.keep_minimum_start = c.keep_minimum_start != 0;
        o.cooks_filter = c.cooks_filter != 0;
        o.independent_filtering = c.independent_filtering != 0;
        o.min_replicates_for_replace = c.min_replicates_for_replace;
        require(c.lfc_shrinkage == RAPIDGLM_SHRINK_NONE ||
                    c.lfc_shrinkage == RAPIDGLM_SHRINK_NORMAL,
                "unknown shrinkage");
        o.lfc_shrinkage =
            c.lfc_shrinkage == RAPIDGLM_SHRINK_NORMAL ? LfcShrinkage::normal : LfcShrinkage::none;
        o.seed = c.seed;
        o.threads = c.threads;
        const auto r = bulk_de(x, y, o);
        auto t = std::make_unique<rapidglm_table>();
        t->rows = r.genes.size();
        auto &gene = t->numeric("gene");
        auto &status = t->text("status");
        std::vector<std::pair<const char *, double BulkGene::*>> fields = {
            {"baseMean", &BulkGene::base_mean},
            {"log2FoldChange", &BulkGene::log2_fold_change},
            {"lfcSE", &BulkGene::lfc_se},
            {"stat", &BulkGene::statistic},
            {"pvalue", &BulkGene::p_value},
            {"padj", &BulkGene::adjusted_p},
            {"dispGeneEst", &BulkGene::dispersion_gene},
            {"dispFit", &BulkGene::dispersion_trend},
            {"dispMAP", &BulkGene::dispersion_map},
            {"dispersion", &BulkGene::dispersion}};
        for (std::size_t i = 0; i < r.genes.size(); ++i) {
            gene[i] = static_cast<double>(i);
            status[i] = r.genes[i].status;
        }
        for (const auto &[name, field] : fields) {
            auto &column = t->numeric(name);
            for (std::size_t i = 0; i < r.genes.size(); ++i)
                column[i] = r.genes[i].*field;
        }
        auto &outlier = t->numeric("dispOutlier");
        auto &cooks = t->numeric("maxCooks");
        auto &replaced = t->numeric("replaced");
        auto &shrunk = t->numeric("log2FoldChangeShrunk");
        auto &shrunk_se = t->numeric("lfcSEShrunk");
        for (std::size_t i = 0; i < r.genes.size(); ++i) {
            const auto &g = r.genes[i];
            outlier[i] = g.dispersion_outlier;
            cooks[i] = g.max_cooks;
            replaced[i] = g.replaced;
            shrunk[i] = g.shrunken_lfc;
            shrunk_se[i] = g.shrunken_lfc_se;
        }
        t->scalars = {{"asymptotic_dispersion", r.asymptotic_dispersion},
                      {"extra_poisson", r.extra_poisson},
                      {"log_residual_variance", r.log_residual_variance},
                      {"prior_variance", r.prior_variance},
                      {"cooks_cutoff", r.cooks_cutoff},
                      {"filter_threshold", r.filter_threshold},
                      {"filter_theta", r.filter_theta},
                      {"replaced_genes", static_cast<double>(r.replaced_genes)}};
        t->vectors = {{"size_factors", r.size_factors},
                      {"lfc_prior_variance", r.lfc_prior_variance}};
        *result = t.release();
    });
}

int rapidglm_bulk_adjust(const double *pvalue, const double *base_mean, size_t n, double alpha,
                         int independent_filtering, double *padj, double *threshold) {
    return guarded([&] {
        require(n == 0 || (pvalue && base_mean && padj), "arrays required");
        require(alpha > 0 && alpha < 1, "alpha must be in (0, 1)");
        const auto r =
            independent_filter(Vector(pvalue, pvalue + n), Vector(base_mean, base_mean + n), alpha,
                               independent_filtering != 0);
        std::copy(r.adjusted_p.begin(), r.adjusted_p.end(), padj);
        if (threshold)
            *threshold = r.threshold;
    });
}

int rapidglm_nb_fixed_fits(const rapidglm_dense *design, const rapidglm_dense *counts,
                           const double *size_factors, const double *dispersions, double tolerance,
                           size_t max_iterations, double min_beta, double max_beta, int threads,
                           double *coefficients, double *fitted, double *hat, int *converged) {
    return guarded([&] {
        const auto x = dense(design);
        const auto y = dense(counts);
        require(size_factors && dispersions && coefficients && fitted && hat && converged,
                "arrays required");
        const auto r = nb_fixed_fits(x, y, Vector(size_factors, size_factors + x.rows),
                                     Vector(dispersions, dispersions + y.cols), tolerance,
                                     max_iterations, min_beta, max_beta, threads);
        std::copy(r.coefficients.values.begin(), r.coefficients.values.end(), coefficients);
        std::copy(r.fitted.values.begin(), r.fitted.values.end(), fitted);
        std::copy(r.hat.values.begin(), r.hat.values.end(), hat);
        std::copy(r.converged.begin(), r.converged.end(), converged);
    });
}

int rapidglm_nb_dispersions(const rapidglm_dense *design, const rapidglm_dense *counts,
                            const rapidglm_dense *mu, const double *start, double min_dispersion,
                            double max_dispersion, double prior_variance, int cox_reid,
                            int keep_minimum_start, int threads, double *dispersions,
                            int *converged) {
    return guarded([&] {
        const auto x = dense(design);
        const auto y = dense(counts);
        const auto means = dense(mu);
        require(start && dispersions && converged, "arrays required");
        const auto [d, ok] = nb_dispersions(x, y, means, Vector(start, start + y.cols),
                                            min_dispersion, max_dispersion, prior_variance,
                                            cox_reid != 0, threads, keep_minimum_start != 0);
        std::copy(d.begin(), d.end(), dispersions);
        std::copy(ok.begin(), ok.end(), converged);
    });
}

void rapidglm_sceptre_options_init(rapidglm_sceptre_options *o, int high_moi) {
    if (!o)
        return;
    std::memset(o, 0, sizeof *o);
    const SceptreOptions d;
    o->struct_size = sizeof *o;
    rapidglm_fit_options_init(&o->model);
    o->null_model = RAPIDGLM_NULL_SCEPTRE;
    o->high_moi = high_moi != 0;
    o->nt_controls = !high_moi;
    o->crt = high_moi != 0;
    o->integration = RAPIDGLM_INTEGRATION_UNION;
    o->side = RAPIDGLM_SIDE_BOTH;
    o->approximate = 1;
    o->effect_fits = RAPIDGLM_EFFECTS_SIGNIFICANT;
    o->screen = d.resampling.screen;
    o->confirm_draws = d.resampling.confirm;
    o->cell_qc = d.cell_qc;
    o->assignment_qc = d.assignment_qc;
    o->min_nonzero = d.min_nonzero;
    o->alpha = d.alpha;
    o->lower_quantile = d.lower_quantile;
    o->upper_quantile = d.upper_quantile;
    o->max_mito = d.max_mito;
    o->seed = 1;
    o->threads = 1;
    o->stream_keys = RAPIDGLM_STREAMS_IDENTITY;
    o->family = RAPIDGLM_FAMILY_TESTED;
}

int rapidglm_sceptre(const rapidglm_dense *design, const rapidglm_sparse *responses,
                     const rapidglm_sparse *assignments, size_t n_guides,
                     const char *const *guide_ids, const char *const *guide_targets,
                     const int *guide_non_targeting, size_t n_pairs, const size_t *pair_responses,
                     const char *const *pair_targets, const int *pair_kinds,
                     const rapidglm_sceptre_options *options, rapidglm_table **result) {
    return guarded([&] {
        require(result != nullptr, "result pointer required");
        require(options != nullptr, "options required");
        const auto c = effective(options, rapidglm_sceptre_options_init, options->high_moi);
        const auto x = dense(design);
        const auto y = sparse(responses);
        const auto a = sparse(assignments);
        const auto guides = sceptre_guides(n_guides, guide_ids, guide_targets, guide_non_targeting);
        const auto pairs = sceptre_pairs(n_pairs, pair_responses, pair_targets, pair_kinds);
        const auto o = sceptre_options(c, x.rows, y.cols);
        const auto r = sceptre(x, y, a, guides, pairs, sceptre_qc(c, x.rows), o);
        *result = sceptre_table(r).release();
    });
}

int rapidglm_sceptre_grid(const rapidglm_dense *design, const rapidglm_sparse *responses,
                          const rapidglm_sparse *assignments, size_t n_guides,
                          const char *const *guide_ids, const char *const *guide_targets,
                          const int *guide_non_targeting, size_t n_discovery,
                          const char *const *discovery_targets, size_t n_negative,
                          const char *const *negative_targets, size_t n_grid_responses,
                          const size_t *grid_responses, size_t n_pairs,
                          const size_t *pair_responses, const char *const *pair_targets,
                          const int *pair_kinds, const rapidglm_sceptre_options *options,
                          rapidglm_table **result, rapidglm_table **tally) {
    return guarded([&] {
        require(result != nullptr, "result pointer required");
        require(options != nullptr, "options required");
        const auto c = effective(options, rapidglm_sceptre_options_init, options->high_moi);
        const auto x = dense(design);
        const auto y = sparse(responses);
        const auto a = sparse(assignments);
        const auto guides = sceptre_guides(n_guides, guide_ids, guide_targets, guide_non_targeting);
        SceptreGrid grid;
        for (std::size_t k = 0; k < n_discovery; ++k)
            grid.discovery_targets.push_back(text_or_empty(discovery_targets, k));
        for (std::size_t k = 0; k < n_negative; ++k)
            grid.negative_targets.push_back(text_or_empty(negative_targets, k));
        require(n_grid_responses == 0 || grid_responses, "grid responses required");
        if (n_grid_responses)
            grid.responses.assign(grid_responses, grid_responses + n_grid_responses);
        grid.pairs = sceptre_pairs(n_pairs, pair_responses, pair_targets, pair_kinds);
        const auto o = sceptre_options(c, x.rows, y.cols);
        const auto r = sceptre_grid(x, y, a, guides, grid, sceptre_qc(c, x.rows), o);
        auto t = sceptre_table(r.result);
        if (tally) {
            auto counts = std::make_unique<rapidglm_table>();
            counts->rows = r.tally.size();
            auto &target = counts->text("target");
            auto &kind = counts->numeric("kind");
            auto &requested = counts->numeric("requested");
            auto &passing = counts->numeric("passing");
            auto &few_treatment = counts->numeric("few_treatment");
            auto &few_control = counts->numeric("few_control");
            for (std::size_t i = 0; i < r.tally.size(); ++i) {
                const auto &e = r.tally[i];
                target[i] = e.target;
                kind[i] = static_cast<int>(e.kind);
                requested[i] = static_cast<double>(e.requested);
                passing[i] = static_cast<double>(e.passing);
                few_treatment[i] = static_cast<double>(e.few_treatment);
                few_control[i] = static_cast<double>(e.few_control);
            }
            *tally = counts.release();
        }
        *result = t.release();
    });
}

void rapidglm_scan_options_init(rapidglm_scan_options *o) {
    if (!o)
        return;
    std::memset(o, 0, sizeof *o);
    const ScanOptions d;
    o->struct_size = sizeof *o;
    rapidglm_fit_options_init(&o->model);
    o->null_model = RAPIDGLM_NULL_DIRECT;
    o->calibration = RAPIDGLM_CALIBRATION_PERMUTATION;
    o->side = RAPIDGLM_SIDE_BOTH;
    o->approximate = 1;
    o->effect_fits = RAPIDGLM_EFFECTS_SIGNIFICANT;
    o->screen = d.resampling.screen;
    o->confirm_draws = d.resampling.confirm;
    o->alpha = d.alpha;
    o->min_nonzero = d.min_nonzero;
    o->seed = 1;
    o->threads = 1;
}

int rapidglm_scan(const rapidglm_dense *design, const rapidglm_sparse *responses,
                  const rapidglm_sparse *predictors, size_t n_pairs, const size_t *pair_responses,
                  const size_t *pair_predictors, const char *const *pair_families,
                  const rapidglm_scan_options *options, rapidglm_table **result) {
    return guarded([&] {
        require(result != nullptr, "result pointer required");
        const auto c = effective(options, rapidglm_scan_options_init);
        const auto x = dense(design);
        const auto y = sparse(responses);
        const auto g = sparse(predictors);
        std::vector<ScanPair> pairs;
        if (n_pairs) {
            require(pair_responses && pair_predictors, "pair arrays required");
            for (std::size_t k = 0; k < n_pairs; ++k)
                pairs.push_back({pair_responses[k], pair_predictors[k],
                                 pair_families && pair_families[k] ? pair_families[k] : "all"});
        }
        ScanOptions o;
        o.model = fit_options(c.model, x.rows);
        require(c.null_model == RAPIDGLM_NULL_SCEPTRE || c.null_model == RAPIDGLM_NULL_DIRECT,
                "unknown null model");
        o.null_model =
            c.null_model == RAPIDGLM_NULL_SCEPTRE ? FitMethod::sceptre_null : FitMethod::direct;
        require(c.calibration >= RAPIDGLM_CALIBRATION_ASYMPTOTIC &&
                    c.calibration <= RAPIDGLM_CALIBRATION_CRT,
                "unknown calibration");
        o.calibration = static_cast<Calibration>(c.calibration);
        o.resampling.side = side(c.side);
        o.resampling.approximate = c.approximate != 0;
        o.resampling.fit_check = fit_check(c.fit_check);
        o.resampling.confirm = c.confirm_draws;
        budgets(o.resampling, c.budgets, c.n_budgets);
        o.resampling.screen = c.screen;
        o.resampling.seed = c.seed;
        o.effect_fits = effect_fits(c.effect_fits);
        if (c.strata)
            o.strata.assign(c.strata, c.strata + x.rows);
        o.alpha = c.alpha;
        o.min_nonzero = c.min_nonzero;
        o.bonferroni = c.bonferroni != 0;
        o.threads = c.threads;
        o.crt_draw = crt_draw(c.crt_draw);
        const auto r = score_scan(x, y, g, pairs, o);
        auto t = std::make_unique<rapidglm_table>();
        t->rows = r.size();
        auto &response = t->numeric("response");
        auto &predictor = t->numeric("predictor");
        auto &family_column = t->text("family");
        auto &status = t->text("status");
        auto &statistic = t->numeric("statistic");
        auto &p = t->numeric("p_value");
        auto &q = t->numeric("adjusted_p");
        auto &estimate = t->numeric("estimate");
        auto &se = t->numeric("standard_error");
        auto &effect = t->numeric("effect");
        auto &effect_se = t->numeric("effect_se");
        auto &draws = t->numeric("draws");
        auto &approximation = t->numeric("approximation");
        for (std::size_t i = 0; i < r.size(); ++i) {
            const auto &e = r[i];
            response[i] = static_cast<double>(e.response);
            predictor[i] = static_cast<double>(e.predictor);
            family_column[i] = e.family;
            status[i] = e.status;
            statistic[i] = e.statistic;
            p[i] = e.p_value;
            q[i] = e.adjusted_p;
            estimate[i] = e.estimate;
            se[i] = e.standard_error;
            effect[i] = e.effect;
            effect_se[i] = e.effect_se;
            draws[i] = static_cast<double>(e.draws);
            approximation[i] = e.approximation;
        }
        *result = t.release();
    });
}

} // extern "C"
