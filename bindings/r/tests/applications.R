# bulk_de(), sceptre() and sceptre_grid() against the RapidGLM CLI on the same inputs.
# Usage: RAPIDGLM_CLI=build/rapidglm Rscript bindings/r/tests/applications.R
library(rapidglm)
cli <- Sys.getenv("RAPIDGLM_CLI")
stopifnot(nzchar(cli), file.exists(cli))
dir <- tempfile("rapidglm-r-")
dir.create(dir)

write_market <- function(m, path) invisible(Matrix::writeMM(methods::as(Matrix::Matrix(m, sparse = TRUE), "generalMatrix"), path))
write_design <- function(x, path) write.table(x, path, sep = "\t", row.names = FALSE, col.names = FALSE)
run_cli <- function(command, config) {
  config$output <- file.path(dir, "result")
  path <- file.path(dir, "config.json")
  writeLines(jsonlite_like(config), path)
  status <- system2(cli, c(command, path), stdout = FALSE, stderr = FALSE)
  stopifnot(status == 0)
  read_cli("result.tsv")
}
read_cli <- function(name) {
  read.delim(file.path(dir, name), colClasses = "character", check.names = FALSE, na.strings = character())
}
# Minimal JSON writer for flat configs (scalars, numeric vectors, strings).
arrays <- c("budgets", "contrast", "coefficients")
jsonlite_like <- function(x) {
  value <- function(v, name) {
    if (is.character(v)) return(paste0('"', v, '"'))
    if (is.logical(v)) return(tolower(as.character(v)))
    if (length(v) > 1 || name %in% arrays)
      return(paste0("[", paste(format(v, digits = 17), collapse = ","), "]"))
    format(v, digits = 17)
  }
  paste0("{", paste(sprintf('"%s":%s', names(x), mapply(value, x, names(x))), collapse = ","), "}")
}
# Every CLI column equals the R column (NA = non-finite): exactly by default, or
# to a relative tolerance where R's BLAS may round differently from the CLI's.
# Below `floor` (p-values from the skew-normal tail approximation far out, which
# rounding differences in the null draws move) both values only need to be small.
same <- function(cli_table, frame, text = character(), tolerance = 0, floor = 0) {
  stopifnot(nrow(cli_table) == nrow(frame))
  for (column in names(cli_table)) {
    stopifnot(column %in% names(frame))
    expected <- cli_table[[column]]
    got <- frame[[column]]
    if (column %in% text) {
      ok <- expected == as.character(got)
    } else {
      e <- suppressWarnings(as.numeric(expected))
      ok <- ifelse(expected == "NA", !is.finite(got),
                   abs(e - got) <= tolerance * abs(e) | (abs(e) < floor & abs(got) < floor))
    }
    ok[is.na(ok)] <- FALSE
    if (!all(ok)) stop(sprintf("column %s differs in %d rows (first: CLI %s, R %s)", column, sum(!ok),
                               expected[!ok][1], format(got[!ok][1], digits = 17)))
  }
}

# Bulk DE: two groups and a batch, with an outlier and an all-zero gene.
set.seed(11)
samples <- 10
col_data <- data.frame(group = factor(rep(c("A", "B"), each = 5)), batch = factor(rep(c("x", "y"), 5)))
base <- rgamma(150, 2, 1 / 40)
effect <- ifelse(seq_along(base) %% 6 == 0, 2.5, 1)
mu <- outer(base, rep(1, samples))
mu[, col_data$group == "B"] <- mu[, col_data$group == "B"] * effect
counts <- matrix(rnbinom(length(mu), mu = mu, size = 8), nrow(mu),
                 dimnames = list(paste0("gene", seq_len(nrow(mu))), paste0("s", seq_len(samples))))
counts[4, ] <- 0
counts[8, 1] <- 4000L
x <- model.matrix(~ batch + group, col_data)
write_market(counts, file.path(dir, "counts.mtx"))
write_design(x, file.path(dir, "x.tsv"))
files <- list(responses = file.path(dir, "counts.mtx"), design = file.path(dir, "x.tsv"), family = "nb")
cases <- list(
  list(cli = list(coefficient = 2, lfc_shrink = "normal", threads = 3),
       r = list(coefficient = "groupB", lfc_shrink = "normal", threads = 3)),
  list(cli = list(contrast = c(0, 1, -1)), r = list(contrast = c(0, 1, -1))),
  list(cli = list(coefficient = 2, legacy_cooks = FALSE), r = list(coefficient = "groupB", legacy_cooks = FALSE)),
  list(cli = list(coefficients = c(1, 2), test = "lrt"), r = list(coefficients = c("batchy", "groupB"), test = "lrt")),
  list(cli = list(coefficient = 2, size_factors = "poscounts", trend = "local"),
       r = list(coefficient = 3, size_factors = "poscounts", trend = "local")))
for (case in cases) {
  table <- run_cli("bulk-de", c(files, case$cli))
  frame <- do.call(bulk_de, c(list(counts, ~ batch + group, col_data), case$r))
  same(table, frame, text = "status")
}
stopifnot(identical(rownames(frame), rownames(counts)))
frame <- bulk_de(counts, ~ batch + group, col_data, contrast = c("group", "B", "A"))
reference <- bulk_de(counts, x, coefficient = "groupB")
stopifnot(identical(frame$pvalue, reference$pvalue), length(attr(frame, "run")$size_factors) == samples)
stopifnot(inherits(try(bulk_de(counts, x, coefficients = 2:3), silent = TRUE), "try-error"))

# Perturbation screen: high MOI, three targets and two non-targeting guides.
cells <- 600
guide_names <- c("a1", "a2", "b1", "c1", "nt1", "nt2")
calls <- matrix(rbinom(cells * length(guide_names), 1, 0.08), length(guide_names),
                dimnames = list(guide_names, NULL))
depth <- rgamma(cells, 20, 20)
covariates <- data.frame(log_depth = log(depth))
mu <- outer(rgamma(10, 2, 1 / 3), depth)
mu[1, ] <- mu[1, ] * ifelse(colSums(calls[1:2, ]) > 0, 0.3, 1)
responses <- matrix(rnbinom(length(mu), mu = mu, size = 5), nrow(mu),
                    dimnames = list(paste0("gene", 1:10), NULL))
guides <- data.frame(id = guide_names, target = c("A", "A", "B", "C", "non-targeting", "non-targeting"),
                     non_targeting = c(0, 0, 0, 0, 1, 1))
pairs <- data.frame(response = c(1, 2, 3, 1, 4, 5, 6), target = c("A", "A", "B", "B", "C", "non-targeting", "A"),
                    kind = c("positive", "discovery", "discovery", "discovery", "discovery", "negative", "discovery"))
x <- model.matrix(~ log_depth, covariates)
write_market(responses, file.path(dir, "responses.mtx"))
write_market(calls, file.path(dir, "assignments.mtx"))
write_design(x, file.path(dir, "design.tsv"))
write.table(guides, file.path(dir, "guides.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cli_pairs <- pairs
cli_pairs$response <- cli_pairs$response - 1
write.table(cli_pairs, file.path(dir, "pairs.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
files <- list(responses = file.path(dir, "responses.mtx"), assignments = file.path(dir, "assignments.mtx"),
              design = file.path(dir, "design.tsv"), guides = file.path(dir, "guides.tsv"),
              pairs = file.path(dir, "pairs.tsv"))
cases <- list(list(moi = "high", budgets = c(199, 1999), seed = 3, threads = 4),
              list(moi = "high", integration = "singleton", approximation = "no_approximation",
                   budgets = 199, seed = 9, effect_fits = "all", adjustment = "bonferroni"))
for (case in cases) {
  table <- run_cli("sceptre", c(files, case))
  table$kind <- c("discovery", "negative", "positive")[as.integer(table$kind) + 1]
  table$response <- as.character(as.integer(table$response) + 1)
  frame <- do.call(sceptre, c(list(responses, calls, ~ log_depth, guides, pairs, cell_data = covariates), case))
  # R's reference BLAS rounds differently from the CLI's OpenBLAS.
  same(table, frame, text = c("target", "guide", "kind", "status"), tolerance = 1e-10, floor = 1e-12)
  stopifnot(sum(frame$status == "ok") >= 5)
}
by_name <- sceptre(responses, calls, x, guides[, c("id", "target")],
                   transform(pairs, response = rownames(responses)[response]), moi = "high", budgets = 199)
stopifnot(identical(by_name$response_id[1], "gene1"))

# Stream keys, response IDs and adjustment families: every target against every gene and
# a positive control. With min_nonzero 40 some pairs fail pair QC.
targets <- c("A", "B", "C", "non-targeting")
grid_pairs <- do.call(rbind, lapply(seq_len(nrow(responses)), function(g)
  data.frame(response = g, target = targets, kind = ifelse(targets == "non-targeting", "negative", "discovery"))))
positive <- data.frame(response = 1, target = "A", kind = "positive")
all_pairs <- rbind(grid_pairs, positive)
write_pairs <- function(pairs, name) {
  pairs$response <- pairs$response - 1
  write.table(pairs, file.path(dir, name), sep = "\t", row.names = FALSE, quote = FALSE)
  file.path(dir, name)
}
files$pairs <- write_pairs(all_pairs, "all_pairs.tsv")
writeLines(rownames(responses), file.path(dir, "ids.txt"))
common <- list(moi = "high", budgets = c(199, 999), seed = 3, min_nonzero = 40, alpha = 0.3)
run_r <- function(f, ...) do.call(f, c(list(responses, calls, ~ log_depth, guides, ...), list(cell_data = covariates), common))
from_cli <- function(table) {
  table$kind <- c("discovery", "negative", "positive")[as.integer(table$kind) + 1]
  table$response <- as.character(as.integer(table$response) + 1)
  table
}
text <- c("target", "guide", "kind", "status")
default <- run_r(sceptre, all_pairs)
failed <- default$status == "failed_pair_qc"
run <- attr(default, "run")
stopifnot(sum(failed) > 0, sum(!failed) > 20, is.na(default$adjusted_p[failed]), !is.na(default$adjusted_p[!failed]),
          run$pairs_listed_discovery == sum(default$kind == "discovery"),
          run$pairs_passing_qc_discovery == sum(default$kind == "discovery" & !failed),
          run$family_size_discovery == run$pairs_passing_qc_discovery,
          run$null_fits == length(unique(default$response[!failed])))
cases <- list(
  list(cli = list(), r = list()),
  list(cli = list(stream_keys = "position"), r = list(stream_keys = "position")),
  list(cli = list(response_ids = file.path(dir, "ids.txt")), r = list(response_ids = rownames(responses))),
  list(cli = list(adjustment_family = "listed", threads = 3), r = list(adjustment_family = "listed", threads = 3)),
  list(cli = list(family_size_discovery = 50, family_size_positive = 2),
       r = list(family_size = c(discovery = 50, positive = 2)), sizes = c(50, 0, 2)),
  list(cli = list(stream_keys = "position", adjustment_family = "listed", family_size_negative = 30,
                  integration = "singleton"),
       r = list(stream_keys = "position", adjustment_family = "listed", family_size = c(0, 30, 0),
                integration = "singleton"), sizes = c(0, 30, 0)))
for (case in cases) {
  table <- from_cli(run_cli("sceptre", c(files, common, case$cli)))
  frame <- do.call(run_r, c(list(sceptre, all_pairs), case$r))
  same(table, frame, text = text, tolerance = 1e-10, floor = 1e-12)
  # The families: the given sizes, or else the pairs at hand (tested or listed).
  run <- attr(frame, "run")
  listed <- identical(case$r$adjustment_family, "listed")
  sizes <- setNames(if (is.null(case$sizes)) c(0, 0, 0) else case$sizes, c("discovery", "negative", "positive"))
  for (kind in names(sizes)) {
    rows <- frame$kind == kind
    at_hand <- if (listed) sum(rows) else sum(rows & frame$status != "failed_pair_qc")
    stopifnot(run[[paste0("pairs_listed_", kind)]] == sum(rows),
              run[[paste0("family_size_", kind)]] == if (sizes[[kind]] > 0) sizes[[kind]] else at_hand)
  }
  if (is.null(case$r$integration)) {
    # The keys change the draws; the family, the adjusted p-values alone.
    keys <- !is.null(case$r$stream_keys) || !is.null(case$r$response_ids)
    stopifnot(identical(frame$p_value, default$p_value) == !keys,
              identical(frame$adjusted_p, default$adjusted_p) == (length(case$r) == 0))
  }
}
fails <- function(expr, message) {
  result <- try(expr, silent = TRUE)
  stopifnot(inherits(result, "try-error"), grepl(message, result, fixed = TRUE))
}
fails(run_r(sceptre, all_pairs, stream_keys = "name"), "should be one of")
fails(run_r(sceptre, all_pairs, response_ids = c("a", "b")), "response_ids needs 10 IDs, got 2")
fails(run_r(sceptre, all_pairs, response_ids = rep("a", 10)), "response IDs must be distinct")
fails(run_r(sceptre, all_pairs, family_size = c(other = 3)), "family_size kinds must be")
fails(run_r(sceptre, all_pairs, family_size = 50), "family_size needs three sizes")
fails(run_r(sceptre, all_pairs, family_size = c(3, 0, 0)), "family_size is smaller")
fails(run_r(sceptre, rbind(all_pairs, all_pairs[5, ])), "listed twice")

# A grid request against the CLI, and against the list of its pairs.
write.table(data.frame(target = targets, kind = c("discovery", "discovery", "discovery", "negative")),
            file.path(dir, "targets.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
files$targets <- file.path(dir, "targets.tsv")
files$pairs <- write_pairs(positive, "positive.tsv")
some <- c(8, 5, 1)
writeLines(as.character(some - 1), file.path(dir, "grid_responses.txt"))
cases <- list(
  list(cli = list(), r = list(), responses = seq_len(nrow(responses))),
  list(cli = list(adjustment_family = "listed", threads = 3), r = list(adjustment_family = "listed", threads = 3),
       responses = seq_len(nrow(responses))),
  list(cli = list(grid_responses = file.path(dir, "grid_responses.txt"), integration = "bonferroni"),
       r = list(grid_responses = rownames(responses)[some], integration = "bonferroni"), responses = some))
for (case in cases) {
  table <- from_cli(run_cli("sceptre", c(files, common, case$cli)))
  frame <- do.call(run_r, c(list(sceptre_grid, targets[1:3], targets[4], pairs = positive), case$r))
  same(table, frame, text = text, tolerance = 1e-10, floor = 1e-12)
  tally <- attr(frame, "tally")
  same(read_cli("result.tally.tsv"), tally, text = c("target", "kind"))
  # The grid's rows are the rows of the listed pairs that pass pair QC.
  listed <- rbind(do.call(rbind, lapply(case$responses, function(g) grid_pairs[grid_pairs$response == g, ])), positive)
  options <- case$r[setdiff(names(case$r), "grid_responses")]
  reference <- do.call(run_r, c(list(sceptre, listed), options))
  passing <- reference[reference$status != "failed_pair_qc", ]
  stopifnot(nrow(passing) < nrow(reference), nrow(passing) == nrow(frame),
            all(mapply(identical, as.list(frame), as.list(passing))),
            identical(tally$target, targets), identical(tally$kind, c("discovery", "discovery", "discovery", "negative")),
            tally$requested == length(case$responses),
            tally$requested == tally$passing + tally$few_treatment + tally$few_control,
            sum(tally$passing) + 1 == nrow(frame))
  run <- attr(frame, "run")
  expected <- attr(reference, "run")
  for (name in grep("^(pairs_listed|pairs_passing_qc|family_size)_", names(run), value = TRUE))
    stopifnot(run[[name]] == expected[[name]])
}
frame <- run_r(sceptre_grid, "A", grid_responses = c(6, 3))
stopifnot(identical(frame$response_id, c("gene6", "gene3")), nrow(attr(frame, "tally")) == 1,
          attr(frame, "tally")$requested == 2)
fails(run_r(sceptre_grid, targets[1:3], targets[4], stream_keys = "position"), "identity stream keys")
fails(run_r(sceptre_grid, targets[1:3], targets[4], pairs = grid_pairs[1, ]), "repeats a pair of the grid")
fails(run_r(sceptre_grid, c("A", "A")), "grid targets must be distinct known targets")
fails(run_r(sceptre_grid, "A", grid_responses = character()), "grid_responses is empty")
fails(run_r(sceptre_grid, "A", grid_responses = "gene99"), "unknown response in grid_responses")
cat("applications: ok\n")
