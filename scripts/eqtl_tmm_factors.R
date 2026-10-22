#!/usr/bin/env Rscript
# TMM normalisation factors (edgeR::calcNormFactors, method "TMM") for a genes x samples count table.
args <- commandArgs(trailingOnly = TRUE)
x <- as.matrix(read.delim(args[1], check.names = FALSE))
suppressPackageStartupMessages(library(edgeR))
f <- calcNormFactors(x, method = "TMM")
write.table(data.frame(iid = colnames(x), norm_factor = f), args[2], sep = "\t", quote = FALSE,
            row.names = FALSE)
cat("edgeR", as.character(packageVersion("edgeR")), "\n")
