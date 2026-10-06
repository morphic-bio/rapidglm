# Black-box SCEPTRE 0.99.0 timing run for scripts/benchmark_applications.py on a
# panel written by scripts/prepare_benchmark_data.py. Uses only SCEPTRE's public
# functions; no package source is read.
# Usage: reference_sceptre_timing.R panel_dir sceptre_library out_dir workers
# The panel holds responses.mtx (genes x cells), assignments.mtx (guides x cells,
# binary calls), guides.tsv (id, target, non_targeting), pairs.tsv (response,
# target, kind), responses.txt and covariates.tsv. Guide calls are passed as
# counts of 5 with threshold assignment at 5, so SCEPTRE uses the same calls.
# Non-targeting guides get SCEPTRE's "non-targeting" label. SCEPTRE's cell QC is
# disabled (the published cells are already filtered), as it is in the RapidGLM
# run. Discovery pairs go to run_discovery_analysis, positive pairs to
# run_power_check; run_calibration_check draws as many negative-control pairs
# as the panel has (SCEPTRE samples its own). Timed: each run_* call.
args <- commandArgs(TRUE)
dir <- normalizePath(args[1]); .libPaths(c(args[2], .libPaths())); out <- args[3]
workers <- as.integer(args[4])
suppressPackageStartupMessages({library(sceptre); library(Matrix); library(jsonlite)})
stopifnot(as.character(packageVersion("sceptre")) == "0.99.0")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
y <- as(readMM(file.path(dir, "responses.mtx")), "CsparseMatrix")
g <- as(readMM(file.path(dir, "assignments.mtx")), "CsparseMatrix") * 5
guides <- read.delim(file.path(dir, "guides.tsv"))
pairs <- read.delim(file.path(dir, "pairs.tsv"))
cov <- read.delim(file.path(dir, "covariates.tsv"))
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
obj <- import_data(y, g, targets, moi = "high", extra_covariates = cov)
obj <- set_analysis_parameters(obj, discovery_pairs = as_pairs("discovery"),
                               positive_control_pairs = as_pairs("positive"),
                               formula_object = ~ log_umi + log_detected + log_guides + percent_mito + batch2,
                               control_group = "complement", resampling_mechanism = "crt")
obj <- assign_grnas(obj, method = "thresholding", threshold = 5L, print_progress = FALSE)
obj <- run_qc(obj, response_n_umis_range = c(0, 1), response_n_nonzero_range = c(0, 1), p_mito_threshold = 1)
setup_seconds <- proc.time()[["elapsed"]] - t0
timed <- function(expr) { t <- proc.time()[["elapsed"]]; force(expr); proc.time()[["elapsed"]] - t }
n_negative <- sum(pairs$kind == "negative")
calibration_seconds <- timed(obj <- run_calibration_check(obj, n_calibration_pairs = n_negative,
                                                          calibration_group_size = 1L, output_amount = 2,
                                                          print_progress = FALSE, parallel = parallel,
                                                          n_processors = if (parallel) workers else "auto"))
power_seconds <- timed(obj <- run_power_check(obj, output_amount = 2, print_progress = FALSE,
                                              parallel = parallel,
                                              n_processors = if (parallel) workers else "auto"))
discovery_seconds <- timed(obj <- run_discovery_analysis(obj, output_amount = 2, print_progress = FALSE,
                                                         parallel = parallel,
                                                         n_processors = if (parallel) workers else "auto"))
write.table(get_result(obj, "run_discovery_analysis"), file.path(out, "discovery.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
write.table(get_result(obj, "run_power_check"), file.path(out, "power.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
write.table(get_result(obj, "run_calibration_check"), file.path(out, "calibration.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
write_json(list(version = as.character(packageVersion("sceptre")), workers = workers,
                setup_seconds = setup_seconds, calibration_seconds = calibration_seconds,
                power_seconds = power_seconds, discovery_seconds = discovery_seconds,
                seconds = calibration_seconds + power_seconds + discovery_seconds),
           file.path(out, "timing.json"), auto_unbox = TRUE, pretty = TRUE)
