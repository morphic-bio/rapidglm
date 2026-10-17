# Black-box SCEPTRE 0.99.0 run at low MOI on a panel written by
# scripts/prepare_replogle.py (or any panel in the format of
# scripts/prepare_benchmark_data.py). Uses only SCEPTRE's public functions; no
# package source is read.
# Usage: reference_sceptre_low_moi.R panel_dir sceptre_library out_dir workers formula
# The panel holds responses.mtx (genes x cells), assignments.mtx (guides x cells,
# binary calls, one guide per cell), guides.tsv (id, target, non_targeting),
# pairs.tsv (response, target, kind), responses.txt and covariates.tsv. Guide
# calls are passed as counts of 5 with threshold assignment at 5, so SCEPTRE uses
# the same calls; non-targeting guides get SCEPTRE's "non-targeting" label.
# SCEPTRE's low-MOI defaults otherwise: non-targeting cells as controls and
# permutation resampling. Its cell QC is disabled apart from the removal of cells
# without exactly one guide, which the panel already satisfies. The formula
# names columns of covariates.tsv (e.g. "~ log_umi + log_detected + batch").
# Discovery pairs go to run_discovery_analysis and positive pairs to
# run_power_check; run_calibration_check draws as many negative-control pairs as
# the panel has. Timed: each run_* call.
args <- commandArgs(TRUE)
dir <- normalizePath(args[1]); .libPaths(c(args[2], .libPaths())); out <- args[3]
workers <- as.integer(args[4]); formula <- as.formula(args[5])
suppressPackageStartupMessages({library(sceptre); library(Matrix); library(jsonlite)})
stopifnot(as.character(packageVersion("sceptre")) == "0.99.0")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
y <- as(readMM(file.path(dir, "responses.mtx")), "CsparseMatrix")
g <- as(readMM(file.path(dir, "assignments.mtx")), "CsparseMatrix") * 5
guides <- read.delim(file.path(dir, "guides.tsv"))
pairs <- read.delim(file.path(dir, "pairs.tsv"))
cov <- read.delim(file.path(dir, "covariates.tsv"))
if ("batch" %in% names(cov)) cov$batch <- factor(cov$batch)
responses <- readLines(file.path(dir, "responses.txt"))
rownames(y) <- responses; rownames(g) <- guides$id
colnames(y) <- colnames(g) <- paste0("cell_", seq_len(ncol(y)))
targets <- data.frame(grna_id = guides$id,
                      grna_target = ifelse(guides$non_targeting == 1, "non-targeting", guides$target))
as_pairs <- function(kind) {
  p <- pairs[pairs$kind == kind, ]
  data.frame(grna_target = p$target, response_id = responses[p$response + 1])
}
parallel <- workers > 1
set.seed(20260930)
t0 <- proc.time()[["elapsed"]]
obj <- import_data(y, g, targets, moi = "low", extra_covariates = cov)
positive <- as_pairs("positive")
obj <- set_analysis_parameters(obj, discovery_pairs = as_pairs("discovery"),
                               positive_control_pairs = if (nrow(positive)) positive else NULL,
                               formula_object = formula)
obj <- assign_grnas(obj, method = "thresholding", threshold = 5L, print_progress = FALSE)
obj <- run_qc(obj, response_n_umis_range = c(0, 1), response_n_nonzero_range = c(0, 1), p_mito_threshold = 1)
setup_seconds <- proc.time()[["elapsed"]] - t0
timed <- function(expr) { t <- proc.time()[["elapsed"]]; force(expr); proc.time()[["elapsed"]] - t }
n_negative <- sum(pairs$kind == "negative")
calibration_seconds <- if (n_negative) timed(obj <- run_calibration_check(
  obj, n_calibration_pairs = n_negative, calibration_group_size = 1L, output_amount = 2,
  print_progress = FALSE, parallel = parallel, n_processors = if (parallel) workers else "auto")) else 0
power_seconds <- if (nrow(positive)) timed(obj <- run_power_check(
  obj, output_amount = 2, print_progress = FALSE, parallel = parallel,
  n_processors = if (parallel) workers else "auto")) else 0
discovery_seconds <- timed(obj <- run_discovery_analysis(obj, output_amount = 2, print_progress = FALSE,
                                                         parallel = parallel,
                                                         n_processors = if (parallel) workers else "auto"))
write.table(get_result(obj, "run_discovery_analysis"), file.path(out, "discovery.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
if (nrow(positive)) write.table(get_result(obj, "run_power_check"), file.path(out, "power.tsv"),
                                sep = "\t", row.names = FALSE, quote = FALSE)
if (n_negative) write.table(get_result(obj, "run_calibration_check"), file.path(out, "calibration.tsv"),
                            sep = "\t", row.names = FALSE, quote = FALSE)
write_json(list(version = as.character(packageVersion("sceptre")), workers = workers,
                formula = args[5], setup_seconds = setup_seconds,
                calibration_seconds = calibration_seconds, power_seconds = power_seconds,
                discovery_seconds = discovery_seconds,
                seconds = calibration_seconds + power_seconds + discovery_seconds),
           file.path(out, "timing.json"), auto_unbox = TRUE, pretty = TRUE)
