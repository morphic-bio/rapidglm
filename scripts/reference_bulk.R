# Black-box DESeq2 run for scripts/validate_bulk.py. Uses only DESeq2's public
# functions and records their documented outputs; no package source is read.
# Usage: reference_bulk.R dataset_dir coefficient_name out_dir libs [min_replicates_for_replace]
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column names).
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(jsonlite)})
dir <- args[1]; coef <- args[2]; out <- args[3]
replace_min <- if (length(args) >= 5) as.numeric(args[5]) else 7
dir.create(out, recursive = TRUE, showWarnings = FALSE)
counts <- as.matrix(readMM(file.path(dir, "counts.mtx")))
storage.mode(counts) <- "integer"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
# DESeq2 recognizes the intercept by R's name for it; the normal LFC prior leaves
# it wide (variance 1e6) only then.
# DESeq2 needs syntactic coefficient names (e.g. sex[T.female] -> sex.T.female.).
colnames(X) <- make.names(colnames(X)); coef <- make.names(coef)
colnames(X)[colnames(X) %in% c("intercept", "Intercept", "X.Intercept.")] <- "(Intercept)"
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
rownames(counts) <- paste0("g", seq_len(nrow(counts)) - 1)
dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
t0 <- proc.time()[["elapsed"]]
dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE, minReplicatesForReplace = replace_min)
res <- results(dds, name = coef)
seconds <- proc.time()[["elapsed"]] - t0
t1 <- proc.time()[["elapsed"]]
shrunk <- lfcShrink(dds, coef = coef, type = "normal", quiet = TRUE)
shrink_seconds <- proc.time()[["elapsed"]] - t1
mc <- mcols(dds)
keep <- intersect(c("baseMean", "baseVar", "allZero", "dispGeneEst", "dispGeneIter", "dispFit",
                    "dispersion", "dispIter", "dispOutlier", "dispMAP", "maxCooks", "replace",
                    "betaConv", "betaIter", "deviance"), colnames(mc))
genes <- data.frame(gene = seq_len(nrow(dds)) - 1, as.data.frame(mc[, keep]),
                    log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE, stat = res$stat,
                    pvalue = res$pvalue, padj = res$padj,
                    shrunkLFC = shrunk$log2FoldChange, shrunkSE = shrunk$lfcSE)
write.table(genes, file.path(out, "genes.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
# Maximum likelihood coefficients (log2) of every design column.
write.table(as.data.frame(mc[, resultsNames(dds)]), file.path(out, "mle.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
write.table(assays(dds)[["cooks"]], file.path(out, "cooks.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(assays(dds)[["mu"]], file.path(out, "mu.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
if (!is.null(assays(dds)[["H"]]))
  write.table(assays(dds)[["H"]], file.path(out, "H.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
if (!is.null(assays(dds)[["replaceCounts"]]))
  write.table(assays(dds)[["replaceCounts"]], file.path(out, "replace_counts.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
f <- dispersionFunction(dds)
md <- metadata(res)
write_json(list(
  version = as.character(packageVersion("DESeq2")),
  seconds = seconds,
  size_factors = unname(sizeFactors(dds)),
  fit_type = attr(f, "fitType"),
  trend_coefficients = as.list(attr(f, "coefficients")),
  prior_variance = attr(f, "dispPriorVar"),
  var_log_disp_ests = attr(f, "varLogDispEsts"),
  filter_threshold = unname(md$filterThreshold),
  filter_theta = md$filterTheta,
  filter_num_rej = md$filterNumRej,
  lo_fit = md$lo.fit,
  alpha = md$alpha,
  results_names = resultsNames(dds),
  lfc_prior_variance = unname(priorInfo(shrunk)$betaPriorVar),
  shrink_seconds = shrink_seconds
), file.path(out, "reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("seconds", seconds, "\n")
