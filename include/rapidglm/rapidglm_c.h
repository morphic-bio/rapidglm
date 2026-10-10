/* RapidGLM C interface: a stable C ABI over the header-only C++ library, for
 * language bindings (Python, Rust, Julia, R, ...). Link against librapidglm_c.
 *
 * Conventions
 * - Every call that can fail returns RAPIDGLM_OK (0) or an error code; the
 *   message of the last error on the calling thread is rapidglm_last_error().
 *   No C++ exception crosses this interface.
 * - Dense matrices are row-major. Sparse matrices are compressed sparse column
 *   (CSC) with observations (samples, cells) as rows and features as columns,
 *   sorted unique row indices within each column, and size_t indices.
 * - Option structs start with `struct_size`; fill them with the matching
 *   *_init function first, then change fields. Fields are only ever appended,
 *   so a caller built against an older header keeps working: fields beyond its
 *   struct_size take their defaults. rapidglm_fit_options is embedded in other
 *   option structs and therefore never grows; new model settings are appended
 *   to the outer structs.
 * - Results are tables (rapidglm_table) owned by the caller and released with
 *   rapidglm_table_free. Columns are numeric (double; NaN for missing) or text;
 *   their names match the CLI's output tables. Run-level values are scalars or
 *   vectors on the table.
 */
#ifndef RAPIDGLM_C_H
#define RAPIDGLM_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAPIDGLM_C_API_VERSION 1

enum { RAPIDGLM_OK = 0, RAPIDGLM_ERROR_INVALID = 1, RAPIDGLM_ERROR_FAILED = 2 };

/* Library version string, C API version and the last error on this thread. */
const char *rapidglm_version(void);
int rapidglm_c_api_version(void);
const char *rapidglm_last_error(void);

typedef struct {
    size_t rows, cols;
    const double *values; /* rows x cols, row-major */
} rapidglm_dense;

typedef struct {
    size_t rows, cols;        /* observations x features */
    const size_t *column_ptr; /* cols + 1 entries */
    const size_t *row_index;  /* column_ptr[cols] entries */
    const double *values;     /* column_ptr[cols] entries */
} rapidglm_sparse;

/* Result tables. */
typedef struct rapidglm_table rapidglm_table;
enum { RAPIDGLM_COLUMN_NUMERIC = 0, RAPIDGLM_COLUMN_TEXT = 1 };
size_t rapidglm_table_rows(const rapidglm_table *table);
size_t rapidglm_table_columns(const rapidglm_table *table);
const char *rapidglm_table_column_name(const rapidglm_table *table, size_t column);
int rapidglm_table_column_type(const rapidglm_table *table, size_t column);
/* Pointer to `rows` doubles owned by the table, or NULL if no such numeric column. */
const double *rapidglm_table_numeric(const rapidglm_table *table, const char *name);
/* Text cell, or NULL if no such text column or row. */
const char *rapidglm_table_text(const rapidglm_table *table, const char *name, size_t row);
/* Run-level values: names, a scalar (NaN if absent) or a vector (NULL if absent). */
size_t rapidglm_table_scalars(const rapidglm_table *table);
const char *rapidglm_table_scalar_name(const rapidglm_table *table, size_t index);
double rapidglm_table_scalar(const rapidglm_table *table, const char *name);
size_t rapidglm_table_vectors(const rapidglm_table *table);
const char *rapidglm_table_vector_name(const rapidglm_table *table, size_t index);
const double *rapidglm_table_vector(const rapidglm_table *table, const char *name, size_t *length);
void rapidglm_table_free(rapidglm_table *table);

/* GLM fitting. */
enum {
    RAPIDGLM_GAUSSIAN = 0,
    RAPIDGLM_POISSON = 1,
    RAPIDGLM_BINOMIAL = 2,
    RAPIDGLM_NEGATIVE_BINOMIAL = 3
};
enum { RAPIDGLM_DISPERSION_ML = 0, RAPIDGLM_DISPERSION_MOMENT = 1, RAPIDGLM_DISPERSION_FIXED = 2 };

typedef struct {
    size_t struct_size;
    int family;            /* default RAPIDGLM_NEGATIVE_BINOMIAL */
    int dispersion;        /* default RAPIDGLM_DISPERSION_ML */
    double theta;          /* NB size: fixed value or starting value (default 10) */
    double tolerance;      /* default 1e-9 */
    double ridge;          /* default 0 */
    size_t max_iterations; /* default 100 */
    const double *offset;  /* NULL or one per observation */
    const double *weights; /* NULL or one per observation */
    int robust;            /* HC0 covariance */
    int cox_reid;          /* Cox-Reid adjusted NB ML dispersion */
    int singular_ok;       /* alias dependent columns (R-style) instead of failing */
} rapidglm_fit_options;
void rapidglm_fit_options_init(rapidglm_fit_options *options);

/* Fits every response column. Columns: feature, status, theta, theta_boundary,
 * log_likelihood, iterations, beta<j>, se<j>, wald_p<j>, boundary<j>. */
int rapidglm_fit_batch(const rapidglm_dense *design, const rapidglm_sparse *responses,
                       const rapidglm_fit_options *options, int threads, rapidglm_table **result);

/* Bulk differential expression (DESeq2 procedure); counts are samples x genes. */
enum { RAPIDGLM_SIZE_FACTORS_RATIO = 0, RAPIDGLM_SIZE_FACTORS_POSCOUNTS = 1 };
enum { RAPIDGLM_TREND_PARAMETRIC = 0, RAPIDGLM_TREND_LOCAL = 1, RAPIDGLM_TREND_MEAN = 2 };
enum { RAPIDGLM_SHRINK_NONE = 0, RAPIDGLM_SHRINK_NORMAL = 1 };

typedef struct {
    size_t struct_size;
    size_t coefficient;      /* tested column (default: the last) */
    const double *contrasts; /* NULL, or contrast_rows x columns, row-major */
    size_t contrast_rows;    /* 1 row: a contrast; several: a joint (LRT) test */
    int lrt;                 /* likelihood ratio test instead of Wald */
    int size_factor_method;
    const double *size_factors; /* NULL or one per sample (used as given) */
    int trend;
    double min_mean, outlier_sd, cooks_cutoff, alpha;
    int keep_minimum_start, cooks_filter, independent_filtering;
    size_t min_replicates_for_replace;
    int lfc_shrinkage;
    uint64_t seed;
    int threads;
    double min_dispersion; /* default 1e-8 */
    double max_dispersion; /* default 0: max(10, samples) */
    int legacy_cooks;      /* default 1: Cook's distances with DESeq2's robust dispersion as in
                              PyDESeq2; 0: RapidGLM's own estimator */
} rapidglm_bulk_options;
void rapidglm_bulk_options_init(rapidglm_bulk_options *options);

/* Columns: gene, status, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
 * dispGeneEst, dispFit, dispMAP, dispersion, dispOutlier, maxCooks, replaced,
 * log2FoldChangeShrunk, lfcSEShrunk. Scalars: asymptotic_dispersion,
 * extra_poisson, log_residual_variance, prior_variance, cooks_cutoff,
 * filter_threshold, filter_theta, replaced_genes. Vectors: size_factors,
 * lfc_prior_variance. */
int rapidglm_bulk_de(const rapidglm_dense *design, const rapidglm_sparse *counts,
                     const rapidglm_bulk_options *options, rapidglm_table **result);

/* Benjamini-Hochberg adjustment after bulk_de's independent filtering on the
 * mean of normalized counts (baseMean), for re-adjusting a bulk_de table at
 * another alpha. padj receives n values (NaN for filtered genes and missing
 * p-values); threshold (may be NULL) the baseMean cut (NaN without filtering). */
int rapidglm_bulk_adjust(const double *pvalue, const double *base_mean, size_t n, double alpha,
                         int independent_filtering, double *padj, double *threshold);

/* Building blocks of the bulk procedure for callers running their own pipeline
 * (for example a PyDESeq2 inference back end). counts, fitted and hat are
 * observations x features (row-major); coefficients features x design columns
 * (natural log scale). Output arrays are allocated by the caller.
 *
 * NB fits at a fixed dispersion per feature with log size-factor offsets;
 * coefficients clamped to [min_beta, max_beta]; fitted = s_i exp(x_i' beta);
 * hat = diag(W^1/2 X (X'WX)^-1 X'W^1/2), W = mu / (1 + dispersion mu).
 * converged[j] = 0 marks failed fits (their outputs are NaN). */
int rapidglm_nb_fixed_fits(const rapidglm_dense *design, const rapidglm_dense *counts,
                           const double *size_factors, const double *dispersions, double tolerance,
                           size_t max_iterations, double min_beta, double max_beta, int threads,
                           double *coefficients, double *fitted, double *hat, int *converged);
/* Dispersions at fixed means `mu` from `start`: NB likelihood, Cox-Reid adjusted
 * when cox_reid, plus a normal prior on log dispersion centred at log(start)
 * when prior_variance > 0 (MAP estimates); clamped to the bounds. With
 * keep_minimum_start (and no prior), features starting at min_dispersion keep it,
 * as the procedure's line search does from that flat start. */
int rapidglm_nb_dispersions(const rapidglm_dense *design, const rapidglm_dense *counts,
                            const rapidglm_dense *mu, const double *start, double min_dispersion,
                            double max_dispersion, double prior_variance, int cox_reid,
                            int keep_minimum_start, int threads, double *dispersions,
                            int *converged);

/* Perturbation (SCEPTRE-method) workflow. */
enum { RAPIDGLM_PAIR_DISCOVERY = 0, RAPIDGLM_PAIR_NEGATIVE = 1, RAPIDGLM_PAIR_POSITIVE = 2 };
enum {
    RAPIDGLM_INTEGRATION_UNION = 0,
    RAPIDGLM_INTEGRATION_SINGLETON = 1,
    RAPIDGLM_INTEGRATION_BONFERRONI = 2
};
enum { RAPIDGLM_SIDE_BOTH = 0, RAPIDGLM_SIDE_LEFT = 1, RAPIDGLM_SIDE_RIGHT = 2 };
enum { RAPIDGLM_NULL_SCEPTRE = 0, RAPIDGLM_NULL_DIRECT = 1 };
enum { RAPIDGLM_EFFECTS_SIGNIFICANT = 0, RAPIDGLM_EFFECTS_ALL = 1, RAPIDGLM_EFFECTS_NONE = 2 };
enum { RAPIDGLM_CRT_THINNED = 0, RAPIDGLM_CRT_DIRECT = 1, RAPIDGLM_CRT_HAZARD = 2 };
/* Acceptance of the skew-normal fit: strict also checks the tail on the
 * statistic's side; ks is the check of releases before 1.0.0. */
enum { RAPIDGLM_FIT_CHECK_STRICT = 0, RAPIDGLM_FIT_CHECK_KS = 1 };
#define RAPIDGLM_CONFIRM_AUTO ((size_t)-1)
/* What a pair's random stream is keyed by. identity: the seed, the response
 * (its ID, or else its column) and the target (the guide for a per-guide
 * test), so a pair's record is the same wherever it is listed and with
 * whatever other pairs; a pair may then be listed once within a kind.
 * position: the pair's place in the list (releases before 1.3.0). */
enum { RAPIDGLM_STREAMS_IDENTITY = 0, RAPIDGLM_STREAMS_POSITION = 1 };
/* The family within which p-values of a kind are adjusted. tested: the pairs
 * that pass pair QC; the other pairs get no adjusted p-value (NaN). listed:
 * every listed pair, those that fail pair QC counting with p = 1 (releases
 * before 1.3.0). */
enum { RAPIDGLM_FAMILY_TESTED = 0, RAPIDGLM_FAMILY_LISTED = 1 };

typedef struct {
    size_t struct_size;
    rapidglm_fit_options model; /* response model (default NB ML) */
    int null_model;             /* default RAPIDGLM_NULL_SCEPTRE */
    int high_moi;               /* default 0 */
    int nt_controls;            /* default: 1 for low MOI, 0 (complement) for high */
    int crt;                    /* default: 0 (permutations) for low MOI, 1 for high */
    int integration, side, approximate, effect_fits;
    const size_t *budgets; /* NULL (499, 4999, 19999) or n_budgets increasing values */
    size_t n_budgets;
    double screen; /* empirical screen of the first stage (default 0.02) */
    int cell_qc, assignment_qc;
    size_t min_nonzero;
    double alpha, lower_quantile, upper_quantile, max_mito;
    const double *qc_total, *qc_detected, *qc_mito; /* NULL or one per cell */
    uint64_t seed;
    int threads;
    int bonferroni; /* Bonferroni instead of BH adjustment of discovery pairs */
    int crt_draw;   /* default RAPIDGLM_CRT_THINNED */
    int fit_check;  /* default RAPIDGLM_FIT_CHECK_STRICT */
    /* Draws on which a skew-normal p-value too small for the last budget to
     * check is confirmed (strict check). Default RAPIDGLM_CONFIRM_AUTO: ten
     * times the last budget (199999 for the default budgets). Off when 0 or not
     * above the last budget. */
    size_t confirm_draws;
    int stream_keys; /* default RAPIDGLM_STREAMS_IDENTITY */
    /* With identity keys: NULL (a response is identified by its column) or one
     * ID per response column, distinct strings that stay with the response
     * (gene names, say) when the matrix changes its columns. */
    const char *const *response_ids;
    int family; /* default RAPIDGLM_FAMILY_TESTED */
    /* Hypotheses in the discovery, negative-control and positive-control
     * families when a family is larger than its pairs at hand (a caller that
     * lists only some of them; those not listed count with p = 1). Default 0:
     * the pairs at hand. */
    size_t family_size[3];
} rapidglm_sceptre_options;
/* high_moi selects the MOI-dependent defaults (control group, mechanism). */
void rapidglm_sceptre_options_init(rapidglm_sceptre_options *options, int high_moi);

/* design: cells x covariates; responses: cells x genes; assignments: cells x
 * guides (binary calls). Guides: id, target and non-targeting flag per column of
 * `assignments`. Pairs: response column, target name and kind. Columns:
 * response, target, guide, kind, status, cells, nonzero_treatment,
 * nonzero_control, draws, approximation, statistic, p_value, adjusted_p (NaN
 * for a pair outside its adjustment family), log2fc, log2fc_se,
 * sceptre_mean_ratio, mean_ratio_log2. Scalars: retained_cells,
 * calibration_rejections, power_rejections, discovery_rejections, null_fits
 * (null models fitted) and, for <kind> in discovery, negative and positive,
 * pairs_listed_<kind>, pairs_passing_qc_<kind> and family_size_<kind> (the
 * size of the adjustment family). */
int rapidglm_sceptre(const rapidglm_dense *design, const rapidglm_sparse *responses,
                     const rapidglm_sparse *assignments, size_t n_guides,
                     const char *const *guide_ids, const char *const *guide_targets,
                     const int *guide_non_targeting, size_t n_pairs, const size_t *pair_responses,
                     const char *const *pair_targets, const int *pair_kinds,
                     const rapidglm_sceptre_options *options, rapidglm_table **result);

/* rapidglm_sceptre for a grid request: every pair of some targets and some
 * responses, without a list of the pairs. Discovery targets are tested against
 * each grid response as discovery pairs, negative targets (of non-targeting
 * guides) as negative controls; grid_responses holds n_grid_responses distinct
 * response columns (0, NULL: every response). Pair QC is decided for the whole
 * grid first, and only the pairs that pass it get a record: `result` has those
 * rows (by response in the order requested, then discovery and negative
 * targets in the order given), then the rows of the n_pairs further pairs
 * (may be 0), which are listed as for rapidglm_sceptre (positive controls,
 * say) and may not repeat a pair of the grid in its kind. The records are
 * those rapidglm_sceptre gives for a list of the same pairs, and the
 * adjustment families those of listing every pair of the grid: with
 * RAPIDGLM_FAMILY_LISTED the pairs that fail pair QC are counted without being
 * listed, and pairs_listed_<kind> counts every pair of the grid. Needs
 * identity stream keys. `result`: columns and scalars as for rapidglm_sceptre.
 * `tally` (may be NULL): one row per grid target and kind, discovery targets
 * first, with the columns target, kind, requested, passing, few_treatment (too
 * few treated cells with a nonzero count) and few_control (enough of those, too
 * few such control cells); requested = passing + few_treatment + few_control. */
int rapidglm_sceptre_grid(const rapidglm_dense *design, const rapidglm_sparse *responses,
                          const rapidglm_sparse *assignments, size_t n_guides,
                          const char *const *guide_ids, const char *const *guide_targets,
                          const int *guide_non_targeting, size_t n_discovery,
                          const char *const *discovery_targets, size_t n_negative,
                          const char *const *negative_targets, size_t n_grid_responses,
                          const size_t *grid_responses, size_t n_pairs,
                          const size_t *pair_responses, const char *const *pair_targets,
                          const int *pair_kinds, const rapidglm_sceptre_options *options,
                          rapidglm_table **result, rapidglm_table **tally);

/* Score scan: each response's null model fitted once, many predictors tested. */
enum {
    RAPIDGLM_CALIBRATION_ASYMPTOTIC = 0,
    RAPIDGLM_CALIBRATION_PERMUTATION = 1,
    RAPIDGLM_CALIBRATION_CRT = 2
};

typedef struct {
    size_t struct_size;
    rapidglm_fit_options model;
    int null_model; /* default RAPIDGLM_NULL_DIRECT */
    int calibration, side, approximate, effect_fits;
    const size_t *budgets;
    size_t n_budgets;
    double screen;        /* default 0 */
    const size_t *strata; /* NULL or one per observation (permutation only) */
    double alpha;
    size_t min_nonzero;
    int bonferroni;
    uint64_t seed;
    int threads;
    int crt_draw;         /* default RAPIDGLM_CRT_THINNED */
    int fit_check;        /* default RAPIDGLM_FIT_CHECK_STRICT */
    size_t confirm_draws; /* as for rapidglm_sceptre_options */
} rapidglm_scan_options;
void rapidglm_scan_options_init(rapidglm_scan_options *options);

/* predictors: observations x predictors. Pairs: NULL (all response x predictor
 * pairs) or n_pairs response/predictor indices with optional family names.
 * Columns: response, predictor, family, status, statistic, p_value, adjusted_p,
 * estimate, standard_error, effect, effect_se, draws, approximation. */
int rapidglm_scan(const rapidglm_dense *design, const rapidglm_sparse *responses,
                  const rapidglm_sparse *predictors, size_t n_pairs, const size_t *pair_responses,
                  const size_t *pair_predictors, const char *const *pair_families,
                  const rapidglm_scan_options *options, rapidglm_table **result);

#ifdef __cplusplus
}
#endif

#endif /* RAPIDGLM_C_H */
