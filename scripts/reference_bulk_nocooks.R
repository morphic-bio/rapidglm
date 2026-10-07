# Black-box DESeq2 run without the Cook's-distance cutoff, for
# scripts/validate_cooks.py. Uses only DESeq2's public functions; no package
# source is read.
# Usage: reference_bulk_nocooks.R dataset_dir coefficient_name libs out.tsv [workers]
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column names).
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[3], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix)})
dir <- args[1]; coef <- make.names(args[2])
workers <- if (length(args) > 4) as.integer(args[5]) else 8
counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "integer"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
colnames(X) <- make.names(colnames(X))
colnames(X)[colnames(X) %in% c("intercept", "Intercept", "X.Intercept.")] <- "(Intercept)"
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE, parallel = TRUE,
             BPPARAM = BiocParallel::MulticoreParam(workers))
res <- results(dds, name = coef, cooksCutoff = FALSE)
write.table(data.frame(pvalue = res$pvalue, padj = res$padj), args[4], sep = "\t", row.names = FALSE, quote = FALSE)
