# Black-box edgeR and glmGamPoi quasi-likelihood tests of the Kang stimulation
# contrast (donor-paired design) for scripts/validate_ql.py.
# Usage: reference_ql_contrast.R panel_dir coefficient output.tsv libs
args<-commandArgs(TRUE);.libPaths(c(normalizePath(strsplit(args[4],':')[[1]]),.libPaths()))
suppressPackageStartupMessages({library(edgeR);library(Matrix)})
counts<-as.matrix(readMM(file.path(args[1],'responses.mtx')))
design<-as.matrix(read.table(file.path(args[1],'design.tsv')));colnames(design)<-paste0('c',seq_len(ncol(design))-1)
offset<-scan(file.path(args[1],'offset.tsv'),quiet=TRUE);coef<-as.integer(args[2])+1
y<-estimateDisp(DGEList(counts,lib.size=exp(offset)),design)
ql<-glmQLFTest(glmQLFit(y,design),coef=coef)
out<-data.frame(feature=seq_len(nrow(counts))-1,edgeR_ql=ql$table$PValue)
if(requireNamespace('glmGamPoi',quietly=TRUE)){
  fit<-glmGamPoi::glm_gp(counts,design=design,offset=offset,size_factors=FALSE)
  out$glmGamPoi_ql<-eval(bquote(glmGamPoi::test_de(fit,contrast=.(as.name(colnames(design)[coef])))))$pval
}
write.table(out,args[3],sep='\t',row.names=FALSE,quote=FALSE)
