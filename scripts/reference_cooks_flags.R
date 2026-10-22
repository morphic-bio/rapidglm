# Black-box DESeq2 run that keeps its Cook's-distance outputs, for
# scripts/validate_cooks_flags.py. Uses only DESeq2's public functions and
# assays; no package source is read.
# Usage: reference_cooks_flags.R dataset_dir coefficient_name libs out_dir
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column names).
# Writes genes.tsv (p-value and maxCooks per gene) and, as genes x samples tables,
# the assays cooks, replaceCooks and replaceCounts.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[3], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix)})
dir <- args[1]; coef <- make.names(args[2]); out <- args[4]
counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "integer"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
colnames(X) <- make.names(colnames(X))
colnames(X)[colnames(X) %in% c("intercept", "Intercept", "X.Intercept.")] <- "(Intercept)"
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE)
res <- results(dds, name = coef)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.table(data.frame(baseMean = res$baseMean, pvalue = res$pvalue, maxCooks = mcols(dds)$maxCooks),
            file.path(out, "genes.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
for (a in intersect(c("cooks", "replaceCooks", "replaceCounts"), assayNames(dds)))
  write.table(assays(dds)[[a]], file.path(out, paste0(a, ".tsv")), sep = "\t", row.names = FALSE,
              col.names = FALSE, quote = FALSE)
cat(as.character(packageVersion("DESeq2")), "\n")
