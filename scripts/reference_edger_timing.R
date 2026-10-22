# Timed black-box run of edgeR's default quasi-likelihood workflow for
# scripts/benchmark_applications.py. Uses only edgeR's public functions; no
# package source is read.
# Usage: reference_edger_timing.R dataset_dir coefficient_name out_dir libs
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column
# names). Timed: DGEList, filterByExpr on the design, the subset with library
# sizes recomputed, normLibSizes (TMM), glmQLFit, glmQLFTest and topTags, as the
# edgeR 4 User's Guide runs them; reading the files is not timed. edgeR runs on
# one thread; edgeR and limma (whose squeezeVar glmQLFit calls) are loaded
# before timing. Writes genes.tsv (one row per gene, NA for filtered genes) with
# pvalue and padj (FDR) and timing.json.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(edgeR); library(limma); library(Matrix); library(jsonlite)})
dir <- args[1]; coef <- args[2]; out <- args[3]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "double"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
j <- match(coef, colnames(X))
if (is.na(j)) stop("coefficient ", coef, " is not a design column")
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
rownames(counts) <- paste0("g", seq_len(nrow(counts)) - 1)
rownames(X) <- colnames(counts)
t0 <- proc.time()[["elapsed"]]
y <- DGEList(counts)
keep <- filterByExpr(y, design = X)
y <- normLibSizes(y[keep, , keep.lib.sizes = FALSE])
fit <- glmQLFit(y, X)
res <- glmQLFTest(fit, coef = j)
tab <- topTags(res, n = Inf, sort.by = "none")$table
seconds <- proc.time()[["elapsed"]] - t0
genes <- data.frame(gene = seq_len(nrow(counts)) - 1, pvalue = NA_real_, padj = NA_real_)
rows <- match(rownames(tab), rownames(counts))
genes$pvalue[rows] <- tab$PValue
genes$padj[rows] <- tab$FDR
write.table(genes, file.path(out, "genes.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write_json(list(seconds = seconds, edger = as.character(packageVersion("edgeR")), kept = sum(keep)),
           file.path(out, "timing.json"), auto_unbox = TRUE, pretty = TRUE)
