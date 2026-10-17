# Black-box check of edgeR's GLM fits in the quasi-likelihood workflow, for
# scripts/validate_edger.py. Uses only edgeR's public functions; no package source
# is read.
# Usage: reference_edger_fits.R dataset_dir coefficient_name edger_dir libs
# For the genes and offsets of the reference run in edger_dir (filter.tsv,
# samples.tsv, reference.json from scripts/reference_edger.R), fits the full and
# the reduced design (the coefficient dropped) with glmFit at the dispersion
# glmQLFit fits with (dispersion / average.ql.dispersion), all kept genes in one
# call as glmQLFit does, and writes edger_dir/fit_check.tsv: per gene and design,
# the deviance glmFit reports, the deviance of its fitted values
# (nbinomUnitDeviance summed), its iteration count and failure flag.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(edgeR); library(Matrix); library(jsonlite)})
dir <- args[1]; coef <- args[2]; out <- args[3]
counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "double"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
j <- match(coef, colnames(X))
filter <- read.delim(file.path(out, "filter.tsv"))
samples <- read.delim(file.path(out, "samples.tsv"))
ql <- fromJSON(file.path(out, "reference.json"))$ql
phi <- ql$dispersion / ql$average_ql_dispersion
kept <- filter$gene[filter$kept == 1]
y <- counts[kept + 1, , drop = FALSE]
offset <- matrix(log(samples$effective_lib_size), nrow(y), ncol(y), byrow = TRUE)
result <- data.frame(gene = kept)
for (name in c("full", "reduced")) {
  design <- if (name == "full") X else X[, -j, drop = FALSE]
  f <- glmFit(y, design, dispersion = phi, offset = offset, prior.count = 0)
  result[[paste0(name, "_reported")]] <- f$deviance
  result[[paste0(name, "_at_fitted")]] <- rowSums(nbinomUnitDeviance(y, f$fitted.values, phi))
  result[[paste0(name, "_iter")]] <- f$iter
  result[[paste0(name, "_failed")]] <- f$failed
}
write.table(result, file.path(out, "fit_check.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
