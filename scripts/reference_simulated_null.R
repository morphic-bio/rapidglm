# Black-box edgeR QL, glmGamPoi QL and DESeq2 Wald tests of the tested (last)
# design column on every simulated null dataset, for scripts/validate_ql_simulation.py.
# Usage: reference_simulated_null.R simulation_dir output.tsv libs
# Each dataset directory holds counts.mtx (genes x samples), design.tsv (no header)
# and offset.tsv (log size factors); datasets are listed in simulation_dir/datasets.txt.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[3], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(Matrix); library(edgeR); library(DESeq2)})
datasets <- readLines(file.path(args[1], "datasets.txt"))
out <- list()
for (name in datasets) {
  dir <- file.path(args[1], name)
  counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "integer"
  X <- as.matrix(read.table(file.path(dir, "design.tsv")))
  colnames(X) <- c("(Intercept)", paste0("c", seq_len(ncol(X) - 1)))
  offset <- scan(file.path(dir, "offset.tsv"), quiet = TRUE)
  k <- ncol(X)
  y <- estimateDisp(DGEList(counts, lib.size = exp(offset)), X)
  edger <- glmQLFTest(glmQLFit(y, X), coef = k)$table$PValue
  gp <- glmGamPoi::glm_gp(counts, design = X, offset = offset, size_factors = FALSE)
  gamPoi <- glmGamPoi::test_de(gp, reduced_design = X[, -k, drop = FALSE])$pval
  colnames(counts) <- paste0("s", seq_len(ncol(counts)))
  dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
  dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE)
  res <- results(dds, name = resultsNames(dds)[k])
  out[[name]] <- data.frame(dataset = name, gene = seq_len(nrow(counts)) - 1, edgeR_ql = edger,
                            glmGamPoi_ql = gamPoi, DESeq2_wald = res$pvalue, DESeq2_padj = res$padj)
}
write.table(do.call(rbind, out), args[2], sep = "\t", row.names = FALSE, quote = FALSE)
