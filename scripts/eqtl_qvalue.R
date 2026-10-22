#!/usr/bin/env Rscript
# Storey q-values with the Bioconductor qvalue package (defaults: lambda = 0.05..0.95, smoother pi0).
# Input: TSV with columns id, p. Output: TSV with id, p, q and a one-line pi0 report on stdout.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(qvalue))
d <- read.delim(args[1], colClasses = c("character", "numeric"))
res <- tryCatch(list(q = qvalue(d$p), method = "smoother"),
                error = function(e) list(q = qvalue(d$p, lambda = 0.85), method = "lambda=0.85"))
d$q <- res$q$qvalues
write.table(d, args[2], sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("qvalue %s pi0 %.6f method %s n %d\n", as.character(packageVersion("qvalue")),
            res$q$pi0, res$method, nrow(d)))
