# Timing of the RapidGLM R package (bulk_de, sceptre) for
# scripts/benchmark_applications.py. Timed: the call only; loading is not timed.
# Usage: rapidglm_r_timing.R bulk dataset_dir coefficient_index(0-based) threads out_dir libs
#        rapidglm_r_timing.R sceptre panel_dir threads out_dir libs
args <- commandArgs(TRUE)
mode <- args[1]
.libPaths(c(normalizePath(strsplit(args[length(args)], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(rapidglm); library(Matrix); library(jsonlite)})
dir <- args[2]
if (mode == "bulk") {
  coefficient <- as.integer(args[3]) + 1; threads <- as.integer(args[4]); out <- args[5]
  counts <- readMM(file.path(dir, "counts.mtx"))  # genes x samples
  x <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
  t0 <- proc.time()[["elapsed"]]
  r <- bulk_de(counts, x, coefficient = coefficient, threads = threads)
  seconds <- proc.time()[["elapsed"]] - t0
  table <- data.frame(pvalue = r$pvalue, padj = r$padj)
} else {
  threads <- as.integer(args[3]); out <- args[4]
  responses <- readMM(file.path(dir, "responses.mtx"))  # genes x cells
  calls <- readMM(file.path(dir, "assignments.mtx"))
  x <- as.matrix(read.delim(file.path(dir, "design.tsv"), header = FALSE))
  guides <- read.delim(file.path(dir, "guides.tsv"))
  pairs <- read.delim(file.path(dir, "pairs.tsv"))
  pairs$response <- pairs$response + 1
  t0 <- proc.time()[["elapsed"]]
  table <- sceptre(responses, calls, x, guides, pairs, moi = "high", cell_qc = FALSE, seed = 20260930,
                   threads = threads)
  seconds <- proc.time()[["elapsed"]] - t0
  table$response <- table$response - 1
  table$kind <- match(table$kind, c("discovery", "negative", "positive")) - 1
}
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.table(table, file.path(out, "r.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write_json(list(seconds = seconds, threads = threads), file.path(out, "timing.json"), auto_unbox = TRUE)
