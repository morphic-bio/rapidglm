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
#'   implemented in PyDESeq2 (MIT; Muzellec et al. 2023), which reproduces DESeq2's
#'   outlier flags (the default); `FALSE` uses RapidGLM's own estimator.
#' @return A data frame per gene (baseMean, log2FoldChange, lfcSE, stat,
#'   pvalue, padj, dispersions, Cook's distances, ...) with run-level values
#'   (size factors, prior variances) in `attr(, "run")`.
bulk_de <- function(counts, design, col_data = NULL, coefficient = NULL, contrast = NULL,
                    coefficients = NULL, test = c("wald", "lrt"),
                    size_factors = c("ratio", "poscounts"), size_factor_values = NULL,
                    trend = c("parametric", "local", "mean"), lfc_shrink = c("none", "normal"),
                    alpha = 0.1, cooks_filter = TRUE, legacy_cooks = TRUE, independent_filtering = TRUE,
                    min_replicates_for_replace = 7, min_mean = 0.5, outlier_sd = 2,
                    cooks_cutoff = 0, keep_minimum_start = TRUE, min_dispersion = 1e-8,
                    max_dispersion = 0, seed = 1, threads = 1, division = NULL) {
  .division_previous <- .division_set(division)
  on.exit(.division_restore(.division_previous), add = TRUE)
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
    if (!is.numeric(k) || anyNA(k) || any(!is.finite(k)) ||
        any(k != floor(k)) || any(k < 1 | k > ncol(x)))
      stop("coefficient must be a 1-based design column", call. = FALSE)
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
  if (!is.null(coefficient)) {
    if (length(coefficient) != 1) stop("coefficient needs one column", call. = FALSE)
    o$coefficient <- column(coefficient) - 1
  }
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
    if (length(contrast) != ncol(x)) stop("contrast needs one weight per design column", call. = FALSE)
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

#' Quasi-likelihood differential expression following edgeR's procedure
#'
#' The default workflow of edgeR 4 (filterByExpr, TMM normalization, glmQLFit,
#' glmQLFTest), written from its papers, documentation and black-box runs.
#'
#' @param counts genes x samples counts (matrix, data frame or sparse Matrix).
#' @param design samples x columns design matrix, or a formula with `col_data`.
#' @param col_data sample data for a design formula.
#' @param coefficient tested column (name or 1-based index; default: the last).
#' @param contrast numeric weights per design column, or
#'   `c(variable, tested_level, reference_level)` for treatment-coded factors.
#' @param coefficients several columns tested jointly.
#' @param filter filter genes by expression as edgeR's filterByExpr does.
#' @param normalization "tmm" (default) or "none".
#' @param norm_factors normalization factors to use as given (one per sample).
#' @param dispersion the NB dispersion; `NULL`: estimated as edgeR does.
#' @return A data frame per gene (AveLogCPM, logFC, F, PValue, FDR,
#'   quasi-dispersions, ...) with run-level values (dispersion, prior df, library
#'   sizes and factors) in `attr(, "run")`.
edger_ql <- function(counts, design, col_data = NULL, coefficient = NULL, contrast = NULL,
                     coefficients = NULL, filter = TRUE, min_count = 10, min_total_count = 15,
                     large_n = 10, min_prop = 0.7, normalization = c("tmm", "none"),
                     norm_factors = NULL, dispersion = NULL, prior_count = 0.125, threads = 1, division = NULL) {
  .division_previous <- .division_set(division)
  on.exit(.division_restore(.division_previous), add = TRUE)
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
    if (!is.numeric(k) || anyNA(k) || any(!is.finite(k)) ||
        any(k != floor(k)) || any(k < 1 | k > ncol(x)))
      stop("coefficient must be a 1-based design column", call. = FALSE)
    as.integer(k)
  }
  o <- list(filter = as.numeric(filter), min_count = min_count, min_total_count = min_total_count,
            large_n = large_n, min_prop = min_prop,
            tmm = choice(match.arg(normalization), c(tmm = 1, none = 0), "normalization"),
            prior_count = prior_count, threads = threads)
  if (!is.null(dispersion)) o$dispersion <- dispersion
  if (!is.null(norm_factors)) {
    if (length(norm_factors) != nrow(x)) stop("norm_factors needs one value per sample", call. = FALSE)
    o$norm_factors <- as.numeric(norm_factors)
  }
  if (!is.null(coefficient)) {
    if (length(coefficient) != 1) stop("coefficient needs one column", call. = FALSE)
    o$coefficient <- column(coefficient) - 1
  }
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
    if (length(contrast) != ncol(x)) stop("contrast needs one weight per design column", call. = FALSE)
    o$contrasts <- matrix(as.numeric(contrast), nrow = 1)
  }
  if (!is.null(coefficients)) {
    j <- column(coefficients)
    o$contrasts <- diag(ncol(x))[j, , drop = FALSE]
  }
  r <- rapidglm_edger_ql_(x, y@p, y@i, y@x, nrow(y), ncol(y), o)
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
#' @param crt_draw the sampler of CRT labels. "hazard" (default) draws them with an
#'   AES-based generator and needs the processor's AES instructions; without them it draws
#'   as "thinned" does, so CRT p-values depend on the kind of processor. "thinned" (the
#'   default of releases before 1.5.0) and "direct" (one uniform per cell) can be asked for.
#'   All three draw the same law from different random streams, so their p-values differ
#'   within Monte Carlo error. `attr(, "run")$crt_draw_used` names the sampler used.
#' @param fit_check acceptance of the skew-normal fit: "strict" (default) also checks the
#'   tail where the statistic lies and confirms small p-values on `confirm_draws` draws;
#'   "ks" is the check of releases before 1.0.0.
#' @param confirm_draws draws on which a skew-normal p-value too small for the last budget
#'   to check is confirmed (strict check only). Default: ten times the last budget, 199,999
#'   for the default budgets; 0 turns it off.
#' @param stream_keys what a pair's random stream is keyed by. "identity" (default): the
#'   seed, the response and the target (the guide for a per-guide test), so a pair's record
#'   is the same wherever it is listed and with whatever other pairs; a pair may then be
#'   listed once within a kind. "position": the pair's place in the list (releases before
#'   1.3.0).
#' @param response_ids with identity keys, one ID per response (distinct strings that stay
#'   with the response, such as gene names) to identify the responses by; `NULL`: their
#'   positions in `responses`.
#' @param adjustment_family the family within which p-values of a kind are adjusted.
#'   "tested" (default): the pairs that pass pair QC; the other pairs have no adjusted
#'   p-value (`NA`). "listed": every listed pair, those that fail pair QC counting with
#'   p = 1 (releases before 1.3.0).
#' @param family_size the number of hypotheses in each kind's family when it is larger than
#'   the pairs at hand (the pairs not listed count with p = 1): sizes named by kind, such as
#'   `c(discovery = 20000)`, or three sizes in the order discovery, negative, positive;
#'   `NULL` or 0: the pairs at hand.
#' @param null_start how a pair's null is started with NT controls and
#'   `null_model = "sceptre"`. "refine" (default): one Fisher-scoring step from the
#'   response's null on all retained cells, using additive totals, then exact passes over
#'   the pair's cells to the cold fit's convergence criteria; a pair that does not converge
#'   is fitted cold. "cold": every pair fitted from scratch, as before 1.6.0. Both give the
#'   same nulls to the convergence tolerance.
#' @param batch_scoring CRT with complement controls (high MOI): score each response's pairs
#'   together, all their draws of a stage in one pass over the response's score records
#'   (`TRUE`, default). The results are those of scoring the pairs one by one (`FALSE`).
#' @param .grid for `sceptre_grid()`.
#' @return A data frame per pair (`adjusted_p` is `NA` for a pair outside its adjustment
#'   family) with run-level values in `attr(, "run")`: retained_cells, the rejection counts,
#'   null_fits (null models fitted), crt_draw_used, null_start_used, null_refined,
#'   null_cold_refits and, by kind, pairs_listed_<kind>,
#'   pairs_passing_qc_<kind> and family_size_<kind> (the size of the adjustment family).
sceptre <- function(responses, assignments, design, guides, pairs, cell_data = NULL,
                    moi = c("low", "high"), control_group = NULL, resampling_mechanism = NULL,
                    integration = c("union", "singleton", "bonferroni"),
                    side = c("both", "left", "right"),
                    approximation = c("skew_normal", "no_approximation"), budgets = NULL,
                    screen = 0.02, null_model = c("sceptre", "direct"),
                    effect_fits = c("significant", "all", "none"),
                    crt_draw = c("hazard", "thinned", "direct"), fit_check = c("strict", "ks"),
                    confirm_draws = NULL,
                    cell_qc = TRUE, assignment_qc = TRUE, min_nonzero = 7, alpha = 0.1,
                    qc_lower = 0.01, qc_upper = 0.99, max_mito = 0.2, adjustment = c("BH", "bonferroni"),
                    family = c("nb", "poisson"), seed = 1, threads = 1,
                    stream_keys = c("identity", "position"), response_ids = NULL,
                    adjustment_family = c("tested", "listed"), family_size = NULL,
                    null_start = c("refine", "cold"), batch_scoring = TRUE,
                    .grid = NULL, division = NULL) {
  .division_previous <- .division_set(division)
  on.exit(.division_restore(.division_previous), add = TRUE)
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
            stream_keys = choice(match.arg(stream_keys), c(identity = 0, position = 1), "stream_keys"),
            adjustment_family = choice(match.arg(adjustment_family), c(tested = 0, listed = 1),
                                       "adjustment_family"),
            null_start = choice(match.arg(null_start), c(refine = 0, cold = 1), "null_start"),
            screen = screen, cell_qc = as.numeric(cell_qc), assignment_qc = as.numeric(assignment_qc),
            min_nonzero = min_nonzero, alpha = alpha, lower_quantile = qc_lower,
            upper_quantile = qc_upper, max_mito = max_mito, seed = seed, threads = threads,
            batch_scoring = as.numeric(batch_scoring))
  if (!is.null(confirm_draws)) o$confirm_draws <- confirm_draws
  if (!is.null(control_group))
    o$nt_controls <- choice(control_group, c(complement = 0, nt_cells = 1), "control_group")
  if (!is.null(resampling_mechanism))
    o$crt <- choice(resampling_mechanism, c(permutations = 0, crt = 1), "resampling_mechanism")
  if (!is.null(budgets)) o$budgets <- as.numeric(budgets)
  kinds <- c(discovery = 0L, negative = 1L, positive = 2L)
  if (!is.null(response_ids)) {
    if (length(response_ids) != ncol(y))
      stop(sprintf("response_ids needs %d IDs, got %d", ncol(y), length(response_ids)), call. = FALSE)
    o$response_ids <- as.character(response_ids)
  }
  if (!is.null(family_size)) {
    sizes <- unlist(family_size)
    if (is.null(names(sizes))) {
      if (length(sizes) != 3)
        stop("family_size needs three sizes (discovery, negative, positive) or sizes named by kind",
             call. = FALSE)
      names(sizes) <- names(kinds)
    }
    if (!all(names(sizes) %in% names(kinds)))
      stop("family_size kinds must be discovery, negative or positive", call. = FALSE)
    o$family_size <- as.numeric(replace(0 * kinds, names(sizes), sizes))
  }
  kind <- kinds[as.character(pairs$kind)]
  if (anyNA(kind)) stop("pair kind must be discovery, negative or positive", call. = FALSE)
  if (is.null(.grid)) {
    r <- rapidglm_sceptre_(x, y@p, y@i, y@x, ncol(y), g@p, g@i, g@x, ncol(g),
                           as.character(guides$id), as.character(guides$target),
                           as.integer(guides$non_targeting), as.integer(response - 1),
                           as.character(pairs$target), unname(kind), o)
  } else {
    columns <- .grid$responses
    if (is.character(columns)) columns <- match(columns, colnames(y))
    if (anyNA(columns)) stop("unknown response in grid_responses", call. = FALSE)
    if (!is.null(columns) && !length(columns))
      stop("grid_responses is empty; NULL requests every response", call. = FALSE)
    both <- rapidglm_sceptre_grid_(x, y@p, y@i, y@x, ncol(y), g@p, g@i, g@x, ncol(g),
                                   as.character(guides$id), as.character(guides$target),
                                   as.integer(guides$non_targeting), as.character(.grid$discovery),
                                   as.character(.grid$negative), as.integer(columns - 1),
                                   as.integer(response - 1), as.character(pairs$target),
                                   unname(kind), o)
    r <- both$result
  }
  frame <- as_frame(r)
  run <- attr(frame, "run")
  run$crt_draw_used <- c("thinned", "direct", "hazard")[run$crt_draw_used + 1]
  if (!is.null(run$null_start_used)) run$null_start_used <- c("refine", "cold")[run$null_start_used + 1]
  attr(frame, "run") <- run
  frame$kind <- names(kinds)[frame$kind + 1]
  if (!is.null(colnames(y))) frame$response_id <- colnames(y)[frame$response + 1]
  frame$response <- frame$response + 1
  if (!is.null(.grid)) {
    tally <- as.data.frame(both$tally$columns, stringsAsFactors = FALSE)
    tally$kind <- names(kinds)[tally$kind + 1]
    attr(frame, "tally") <- tally
  }
  frame
}

#' Perturbation screen analysis of a grid of targets and responses
#'
#' `sceptre()` for every pair of some targets and some responses, without a list of the
#' pairs. Pair QC is decided for the whole grid first and only the pairs that pass it get a
#' row: by response in the order requested, then discovery and negative targets in the order
#' given, then the listed pairs. The rows are those `sceptre()` gives for a list of the same
#' pairs, and the adjustment families those of listing every pair of the grid: with
#' `adjustment_family = "listed"` the pairs that fail pair QC are counted without being
#' listed, and pairs_listed_<kind> counts every pair of the grid.
#'
#' @param responses,assignments,design,guides as for `sceptre()`.
#' @param discovery_targets targets tested against every grid response as discovery pairs.
#' @param negative_targets targets of non-targeting guides tested against every grid response
#'   as negative controls.
#' @param grid_responses the responses of the grid (distinct gene names or 1-based indices);
#'   `NULL`: every response.
#' @param pairs further pairs listed as for `sceptre()`, such as positive controls (`NULL`:
#'   none). A pair of the grid may not be listed again in its kind.
#' @param ... further arguments of `sceptre()`; `stream_keys` must be "identity".
#' @return The data frame of `sceptre()`, with one row per grid target and kind in
#'   `attr(, "tally")`: target, kind, requested, passing, few_treatment (too few treated
#'   cells with a nonzero count) and few_control (enough of those, too few such control
#'   cells).
sceptre_grid <- function(responses, assignments, design, guides, discovery_targets,
                         negative_targets = character(), grid_responses = NULL, pairs = NULL, ..., division = NULL) {
  .division_previous <- .division_set(division)
  on.exit(.division_restore(.division_previous), add = TRUE)
  if (is.null(pairs)) pairs <- data.frame(response = integer(), target = character(), kind = character())
  sceptre(responses, assignments, design, guides, pairs, ...,
          .grid = list(discovery = discovery_targets, negative = negative_targets,
                       responses = grid_responses))
}
