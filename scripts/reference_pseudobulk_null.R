# Black-box edgeR/DESeq2 (and glmGamPoi, if installed) runs on the mock-comparison
# splits written by scripts/pseudobulk_null.py; no package source is inspected.
args<-commandArgs(TRUE);work<-normalizePath(args[1]);.libPaths(c(normalizePath(strsplit(args[2],':')[[1]]),.libPaths()))
ggp<-requireNamespace('glmGamPoi',quietly=TRUE)
suppressPackageStartupMessages({library(edgeR);library(DESeq2);library(Matrix);library(jsonlite)})
splits<-fromJSON(file.path(work,'splits.json'),simplifyVector=FALSE);out<-list()
for(s in splits){
  counts<-as.matrix(readMM(file.path(work,paste0(s$condition,'.mtx'))));libsize<-exp(scan(file.path(work,paste0(s$condition,'.offset.tsv')),quiet=TRUE))
  group<-factor(unlist(s$group));design<-model.matrix(~group)
  y<-estimateDisp(DGEList(counts,lib.size=libsize),design)
  ql<-glmQLFTest(glmQLFit(y,design),coef=2);lrt<-glmLRT(glmFit(y,design),coef=2)
  dds<-DESeqDataSetFromMatrix(round(counts),data.frame(group=group),~group);sizeFactors(dds)<-libsize/exp(mean(log(libsize)))
  res<-results(DESeq(dds,quiet=TRUE),name='group_1_vs_0',independentFiltering=FALSE,cooksCutoff=FALSE)
  f<-seq_len(nrow(counts))-1
  out[[length(out)+1]]<-rbind(data.frame(condition=s$condition,split=s$split,feature=f,method='edgeR_ql',p=ql$table$PValue),
    data.frame(condition=s$condition,split=s$split,feature=f,method='edgeR_lrt',p=lrt$table$PValue),
    data.frame(condition=s$condition,split=s$split,feature=f,method='DESeq2_wald',p=res$pvalue))
  if(ggp){
    gf<-glmGamPoi::glm_gp(counts,design=design,offset=log(libsize),size_factors=FALSE)
    gq<-glmGamPoi::test_de(gf,contrast=group1)
    out[[length(out)+1]]<-data.frame(condition=s$condition,split=s$split,feature=f,method='glmGamPoi_ql',p=gq$pval)
  }
}
write.table(do.call(rbind,out),file.path(work,'reference.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
writeLines(c(paste('edgeR',packageVersion('edgeR')),paste('DESeq2',packageVersion('DESeq2')),
  if(ggp) paste('glmGamPoi',packageVersion('glmGamPoi')),R.version.string),file.path(work,'versions.txt'))
