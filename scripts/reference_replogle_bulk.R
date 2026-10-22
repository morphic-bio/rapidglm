# Black-box pilot DESeq2 workflow: fit, Wald test, filtering, gene table. No shrinkage.
a <- commandArgs(TRUE)
.libPaths(c(strsplit(a[4], ":")[[1]], .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(jsonlite)})
dir.create(a[3],recursive=TRUE,showWarnings=FALSE)
y <- as.matrix(readMM(file.path(a[1],"counts.mtx"))); storage.mode(y)<-"integer"
x <- as.matrix(read.delim(file.path(a[1],"design.tsv"),check.names=FALSE))
colnames(x)<-make.names(colnames(x)); coefficient<-make.names(a[2])
colnames(x)[colnames(x)%in%c("intercept","Intercept","X.Intercept.")]<-"(Intercept)"
colnames(y)<-paste0("s",seq_len(ncol(y)))
t0<-proc.time()[["elapsed"]]
dds<-DESeqDataSetFromMatrix(y,data.frame(row.names=colnames(y),one=rep(1,ncol(y))),~1)
dds<-DESeq(dds,full=x,betaPrior=FALSE,quiet=TRUE)
r<-results(dds,name=coefficient)
seconds<-proc.time()[["elapsed"]]-t0
write.table(data.frame(gene=seq_len(nrow(y))-1,as.data.frame(r)),file.path(a[3],"genes.tsv"),
            sep="\t",quote=FALSE,row.names=FALSE)
write_json(list(seconds=seconds,version=as.character(packageVersion("DESeq2")),
                filter_theta=metadata(r)$filterTheta,filter_threshold=unname(metadata(r)$filterThreshold)),
           file.path(a[3],"reference.json"),auto_unbox=TRUE,digits=NA,pretty=TRUE)
