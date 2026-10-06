# Bulk differential expression and the perturbation workflow through the
# RapidGLM C interface (the same code as the CLI and the Python package).

choice <- function(value, choices, name) {
  value <- value[1]
  if (!value %in% names(choices))
    stop(sprintf("%s must be one of %s", name, paste(names(choices), collapse = "/")), call. = FALSE)
  unname(choices[[value]])
}

as_design <- function(design, data, n) {
  if (inherits(design, "formula")) {
    if (is.null(data)) stop("a design formula needs col_data", call. = FALSE)
    design <- stats::model.matrix(design, data)
  }
  design <- as.matrix(design)
  storage.mode(design) <- "double"
  if (nrow(design) != n) stop("design needs one row per sample (column of counts)", call. = FALSE)
  design
}

# Observations x features dgCMatrix from a features x observations input.
as_observations <- function(m) {
  if (!methods::is(m, "sparseMatrix")) m <- Matrix::Matrix(as.matrix(m), sparse = TRUE)
  m <- methods::as(methods::as(methods::as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  Matrix::t(m)
}

as_frame <- function(result, row_names = NULL) {
  frame <- as.data.frame(result$columns, stringsAsFactors = FALSE, check.names = FALSE)
  if (!is.null(row_names)) rownames(frame) <- row_names
  attr(frame, "run") <- c(result$scalars, result$vectors)
  frame
}

#' Bulk differential expression by the DESeq2 procedure
#'
#' @param counts genes x samples counts (matrix, data frame or sparse Matrix).
#' @param design samples x columns design matrix, or a formula with `col_data`.
#' @param col_data sample data for a design formula.
#' @param coefficient tested column (name or 1-based index; default: the last).
#' @param contrast numeric weights per design column, or
#'   `c(variable, tested_level, reference_level)` for treatment-coded factors.
#' @param coefficients several columns tested jointly (use `test = "lrt"`).
#' @param legacy_cooks compute Cook's distances with DESeq2's robust dispersion as
#'   implemented in PyDESeq2 (MIT; Muzellec et al. 2023); reproduces DESeq2's outlier flags.
#' @return A data frame per gene (baseMean, log2FoldChange, lfcSE, stat,
#'   pvalue, padj, dispersions, Cook's distances, ...) with run-level values
#'   (size factors, prior variances) in `attr(, "run")`.
bulk_de <- function(counts, design, col_data = NULL, coefficient = NULL, contrast = NULL,
                    coefficients = NULL, test = c("wald", "lrt"),
                    size_factors = c("ratio", "poscounts"), size_factor_values = NULL,
                    trend = c("parametric", "local", "mean"), lfc_shrink = c("none", "normal"),
                    alpha = 0.1, cooks_filter = TRUE, legacy_cooks = FALSE, independent_filtering = TRUE,
                    min_replicates_for_replace = 7, min_mean = 0.5, outlier_sd = 2,
                    cooks_cutoff = 0, keep_minimum_start = TRUE, min_dispersion = 1e-8,
                    max_dispersion = 0, seed = 1, threads = 1) {
  y <- as_observations(counts)
  x <- as_design(design, col_data, nrow(y))
  if (sum(!is.null(coefficient), !is.null(contrast), !is.null(coefficients)) > 1)
    stop("give one of coefficient, contrast and coefficients", call. = FALSE)
  column <- function(k) {
    if (is.character(k)) {
      j <- match(k, colnames(x))
      if (anyNA(j)) stop(sprintf("no design column %s", k[is.na(j)][1]), call. = FALSE)
      return(j)
    }
    as.integer(k)
  }
  o <- list(lrt = choice(match.arg(test), c(wald = 0, lrt = 1), "test"),
            size_factor_method = choice(match.arg(size_factors), c(ratio = 0, poscounts = 1), "size_factors"),
            trend = choice(match.arg(trend), c(parametric = 0, local = 1, mean = 2), "trend"),
            lfc_shrinkage = choice(match.arg(lfc_shrink), c(none = 0, normal = 1), "lfc_shrink"),
            alpha = alpha, cooks_filter = as.numeric(cooks_filter), legacy_cooks = as.numeric(legacy_cooks),
            independent_filtering = as.numeric(independent_filtering),
            min_replicates_for_replace = min_replicates_for_replace, min_mean = min_mean,
            outlier_sd = outlier_sd, cooks_cutoff = cooks_cutoff,
            keep_minimum_start = as.numeric(keep_minimum_start), min_dispersion = min_dispersion,
            max_dispersion = max_dispersion, seed = seed, threads = threads)
  if (!is.null(coefficient)) o$coefficient <- column(coefficient) - 1
  if (!is.null(contrast)) {
    if (is.character(contrast)) {
      stopifnot(length(contrast) == 3)
      w <- numeric(ncol(x))
      for (k in 2:3) {
        j <- match(paste0(contrast[1], contrast[k]), colnames(x))
        if (!is.na(j)) w[j] <- if (k == 2) 1 else -1
      }
      if (all(w == 0)) stop("no design column codes the contrast", call. = FALSE)
      contrast <- w
    }
    o$contrasts <- matrix(as.numeric(contrast), nrow = 1)
  }
  if (!is.null(coefficients)) {
    j <- column(coefficients)
    o$contrasts <- diag(ncol(x))[j, , drop = FALSE]
  }
  if (!is.null(size_factor_values)) o$size_factor_values <- as.numeric(size_factor_values)
  r <- rapidglm_bulk_de_(x, y@p, y@i, y@x, nrow(y), ncol(y), o)
  as_frame(r, colnames(y))
}

#' Perturbation screen analysis by the SCEPTRE method
#'
#' @param responses genes x cells counts; `assignments` guides x cells calls (nonzero = assigned).
#' @param design cells x covariates design matrix, or a formula with `cell_data`.
#' @param guides data frame with id, target and non_targeting (one row per assignment row;
#'   ids default to the row names of `assignments`, non_targeting to target == "non-targeting").
#' @param pairs data frame with response (gene name or 1-based index), target and kind
#'   (discovery, negative or positive).
#' @param fit_check acceptance of the skew-normal fit: "strict" (default) also checks the
#'   tail where the statistic lies and confirms small p-values on `confirm_draws` draws;
#'   "ks" is the check of releases before 1.0.0.
#' @param confirm_draws draws on which a skew-normal p-value too small for the last budget
#'   to check is confirmed (strict check only). Default: ten times the last budget, 199,999
#'   for the default budgets; 0 turns it off.
#' @return A data frame per pair with run-level values in `attr(, "run")`.
sceptre <- function(responses, assignments, design, guides, pairs, cell_data = NULL,
                    moi = c("low", "high"), control_group = NULL, resampling_mechanism = NULL,
                    integration = c("union", "singleton", "bonferroni"),
                    side = c("both", "left", "right"),
                    approximation = c("skew_normal", "no_approximation"), budgets = NULL,
                    screen = 0.02, null_model = c("sceptre", "direct"),
                    effect_fits = c("significant", "all", "none"),
                    crt_draw = c("thinned", "direct", "hazard"), fit_check = c("strict", "ks"),
                    confirm_draws = NULL,
                    cell_qc = TRUE, assignment_qc = TRUE, min_nonzero = 7, alpha = 0.1,
                    qc_lower = 0.01, qc_upper = 0.99, max_mito = 0.2, adjustment = c("BH", "bonferroni"),
                    family = c("nb", "poisson"), seed = 1, threads = 1) {
  y <- as_observations(responses)
  g <- as_observations(assignments != 0)
  x <- as_design(design, cell_data, nrow(y))
  guides <- as.data.frame(guides)
  if (is.null(guides$id)) guides$id <- rownames(assignments)
  if (is.null(guides$non_targeting)) guides$non_targeting <- guides$target == "non-targeting"
  pairs <- as.data.frame(pairs)
  response <- pairs$response
  if (is.character(response)) response <- match(response, colnames(y))
  if (anyNA(response)) stop("unknown response in pairs", call. = FALSE)
  moi <- match.arg(moi)
  o <- list(high_moi = as.numeric(moi == "high"),
            family = choice(match.arg(family), c(poisson = 1, nb = 3), "family"),
            null_model = choice(match.arg(null_model), c(sceptre = 0, direct = 1), "null_model"),
            integration = choice(match.arg(integration), c(union = 0, singleton = 1, bonferroni = 2), "integration"),
            side = choice(match.arg(side), c(both = 0, left = 1, right = 2), "side"),
            approximate = choice(match.arg(approximation), c(skew_normal = 1, no_approximation = 0), "approximation"),
            effect_fits = choice(match.arg(effect_fits), c(significant = 0, all = 1, none = 2), "effect_fits"),
            crt_draw = choice(match.arg(crt_draw), c(thinned = 0, direct = 1, hazard = 2), "crt_draw"),
            fit_check = choice(match.arg(fit_check), c(strict = 0, ks = 1), "fit_check"),
            bonferroni = choice(match.arg(adjustment), c(BH = 0, bonferroni = 1), "adjustment"),
            screen = screen, cell_qc = as.numeric(cell_qc), assignment_qc = as.numeric(assignment_qc),
            min_nonzero = min_nonzero, alpha = alpha, lower_quantile = qc_lower,
            upper_quantile = qc_upper, max_mito = max_mito, seed = seed, threads = threads)
  if (!is.null(confirm_draws)) o$confirm_draws <- confirm_draws
  if (!is.null(control_group))
    o$nt_controls <- choice(control_group, c(complement = 0, nt_cells = 1), "control_group")
  if (!is.null(resampling_mechanism))
    o$crt <- choice(resampling_mechanism, c(permutations = 0, crt = 1), "resampling_mechanism")
  if (!is.null(budgets)) o$budgets <- as.numeric(budgets)
  kinds <- c(discovery = 0L, negative = 1L, positive = 2L)
  kind <- kinds[as.character(pairs$kind)]
  if (anyNA(kind)) stop("pair kind must be discovery, negative or positive", call. = FALSE)
  r <- rapidglm_sceptre_(x, y@p, y@i, y@x, ncol(y), g@p, g@i, g@x, ncol(g),
                         as.character(guides$id), as.character(guides$target),
                         as.integer(guides$non_targeting), as.integer(response - 1),
                         as.character(pairs$target), unname(kind), o)
  frame <- as_frame(r)
  frame$kind <- names(kinds)[frame$kind + 1]
  if (!is.null(colnames(y))) frame$response_id <- colnames(y)[frame$response + 1]
  frame$response <- frame$response + 1
  frame
}
