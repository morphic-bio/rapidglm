# Black-box pilot edgeR QL workflow: filtering, TMM, QL fits/tests and tables.
# Uses public functions only. Classic edgeR and fold-change shrinkage are not run.
a<-commandArgs(TRUE)
.libPaths(c(strsplit(a[4], ":")[[1]],.libPaths()))
suppressPackageStartupMessages({library(edgeR);library(Matrix);library(jsonlite)})
dir.create(a[3],recursive=TRUE,showWarnings=FALSE)
y<-as.matrix(readMM(file.path(a[1],"counts.mtx")))
x<-as.matrix(read.delim(file.path(a[1],"design.tsv"),check.names=FALSE)); j<-match(a[2],colnames(x))
t0<-proc.time()[["elapsed"]]
d<-DGEList(y); keep<-filterByExpr(d,design=x); d<-d[keep,,keep.lib.sizes=FALSE]
d<-normLibSizes(d); f<-glmQLFit(d,x); q<-glmQLFTest(f,coef=j)
r<-topTags(q,n=Inf,sort.by="none")$table
seconds<-proc.time()[["elapsed"]]-t0
write.table(data.frame(gene=which(keep)-1,r),file.path(a[3],"genes_ql.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
write.table(data.frame(sample=seq_len(ncol(y))-1,lib_size=d$samples$lib.size,
                       norm_factor=d$samples$norm.factors),file.path(a[3],"samples.tsv"),
            sep="\t",quote=FALSE,row.names=FALSE)
write_json(list(ql_seconds=seconds,version=as.character(packageVersion("edgeR"))),
           file.path(a[3],"timing.json"),auto_unbox=TRUE,digits=NA,pretty=TRUE)

# Untimed diagnostics consumed by reference_edger_fits.R.
write.table(data.frame(gene=seq_len(nrow(y))-1,kept=as.integer(keep)),
            file.path(a[3],"filter.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
write.table(data.frame(sample=seq_len(ncol(y))-1,lib_size=d$samples$lib.size,
                       norm_factor=d$samples$norm.factors,
                       effective_lib_size=d$samples$lib.size*d$samples$norm.factors),
            file.path(a[3],"samples.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
write_json(list(ql=list(dispersion=unname(f$dispersion),
                        average_ql_dispersion=unname(f$average.ql.dispersion))),
           file.path(a[3],"reference.json"),auto_unbox=TRUE,digits=NA,pretty=TRUE)
