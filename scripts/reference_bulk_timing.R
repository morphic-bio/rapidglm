# Black-box DESeq2 timing run for scripts/benchmark_applications.py. Uses only
# DESeq2's public functions; no package source is read.
# Usage: reference_bulk_timing.R dataset_dir coefficient_name out_dir libs fit_type workers
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column names).
# fit_type: parametric (DESeq2's default) or glmGamPoi (fitType = "glmGamPoi").
# workers > 1 runs DESeq() and results() with BiocParallel::MulticoreParam(workers).
# Timed: DESeq() and results(); loading is not timed.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(jsonlite)})
dir <- args[1]; coef <- args[2]; out <- args[3]; fit_type <- args[5]; workers <- as.integer(args[6])
dir.create(out, recursive = TRUE, showWarnings = FALSE)
counts <- as.matrix(readMM(file.path(dir, "counts.mtx")))
storage.mode(counts) <- "integer"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
# DESeq2 needs syntactic coefficient names (e.g. sex[T.female] -> sex.T.female.).
colnames(X) <- make.names(colnames(X)); coef <- make.names(coef)
colnames(X)[colnames(X) %in% c("intercept", "Intercept", "X.Intercept.")] <- "(Intercept)"
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
rownames(counts) <- paste0("g", seq_len(nrow(counts)) - 1)
dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
parallel <- workers > 1
if (parallel) {
  suppressPackageStartupMessages(library(BiocParallel))
  register(MulticoreParam(workers))
}
t0 <- proc.time()[["elapsed"]]
dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE, fitType = fit_type, parallel = parallel)
res <- results(dds, name = coef, parallel = parallel)
seconds <- proc.time()[["elapsed"]] - t0
write.table(data.frame(gene = seq_len(nrow(dds)) - 1, baseMean = res$baseMean,
                       log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE, stat = res$stat,
                       pvalue = res$pvalue, padj = res$padj),
            file.path(out, "genes.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write_json(list(seconds = seconds, fit_type = fit_type, workers = workers,
                deseq2 = as.character(packageVersion("DESeq2")),
                glmGamPoi = if (fit_type == "glmGamPoi") as.character(packageVersion("glmGamPoi")) else NULL),
           file.path(out, "timing.json"), auto_unbox = TRUE)
