# Separate-process black-box reference packages; no source inspection.
args<-commandArgs(TRUE);root<-normalizePath(args[1],mustWork=FALSE);.libPaths(c(normalizePath(args[2]),.libPaths()))
suppressPackageStartupMessages(library(SingleCellExperiment));suppressPackageStartupMessages(library(edgeR));suppressPackageStartupMessages(library(DESeq2));suppressPackageStartupMessages(library(jsonlite));suppressPackageStartupMessages(library(Matrix))
dir.create(root,recursive=TRUE,showWarnings=FALSE)
write_numeric<-function(x,p)write.table(x,p,sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
env<-new.env();load('validation/local-data/Kang18_EH2259.rds',envir=env);sce<-env$sce
keep<-which(sce$multiplets=='singlet' & sce$cell=='CD14+ Monocytes');sce<-sce[,keep];sample<-paste(sce$ind,sce$stim,sep='_');levels<-sort(unique(sample));units<-match(sample,levels)
counts<-counts(sce);pb<-sapply(seq_along(levels),function(i)Matrix::rowSums(counts[,units==i,drop=FALSE]));rownames(pb)<-rownames(counts);colnames(pb)<-levels
meta<-as.data.frame(colData(sce))[match(levels,sample),,drop=FALSE];meta$ind<-factor(meta$ind);meta$stim<-factor(meta$stim,levels=c('ctrl','stim'));rownames(meta)<-levels
stopifnot(length(levels)==16,all(table(sample)>=20))
libsize<-colSums(pb);eligible<-which(rowSums(pb>=10)>=4);selected<-head(eligible,1000);pb<-pb[selected,,drop=FALSE];design<-model.matrix(~ind+stim,meta)
d<-file.path(root,'pseudobulk');dir.create(d,showWarnings=FALSE)
writeMM(as(Matrix(pb,sparse=TRUE),'generalMatrix'),file.path(d,'responses.mtx'));write_numeric(design,file.path(d,'design.tsv'));write_numeric(log(libsize),file.path(d,'offset.tsv'))
write.table(data.frame(feature=seq_along(selected)-1,gene=rownames(pb)),file.path(d,'genes.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
config<-list(responses=file.path(d,'responses.mtx'),design=file.path(d,'design.tsv'),offset=file.path(d,'offset.tsv'),family='nb',coefficient=ncol(design)-1,output=file.path(d,'native'),threads=2,test='wald');write_json(config,file.path(d,'config.json'),auto_unbox=TRUE,pretty=TRUE)
start<-proc.time()[['elapsed']];edge<-DGEList(pb,lib.size=libsize);edge<-estimateDisp(edge,design);fit<-glmFit(edge,design);lrt<-glmLRT(fit,coef=ncol(design));edge_seconds<-proc.time()[['elapsed']]-start
ref<-data.frame(feature=seq_along(selected)-1,edgeR_statistic=lrt$table$LR,edgeR_p=lrt$table$PValue,edgeR_log2fc=lrt$table$logFC)
start<-proc.time()[['elapsed']];dds<-DESeqDataSetFromMatrix(round(pb),meta,~ind+stim);sizeFactors(dds)<-libsize/exp(mean(log(libsize)));dds<-DESeq(dds,quiet=TRUE);res<-results(dds,contrast=c('stim','stim','ctrl'),independentFiltering=FALSE,cooksCutoff=FALSE);deseq_seconds<-proc.time()[['elapsed']]-start
ref$DESeq2_statistic<-res$stat;ref$DESeq2_p<-res$pvalue;ref$DESeq2_log2fc<-res$log2FoldChange;write.table(ref,file.path(d,'reference.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write_json(list(source='https://experimenthub.bioconductor.org/fetch/2259',accession='GSE96583 / EH2259',cell_type='CD14+ Monocytes',cells=ncol(sce),samples=16,donors=8,genes=nrow(pb),selection='first 1000 genes in public feature order with count >=10 in >=4 samples',normalization='common raw-library-size offsets; no TMM',edgeR=list(version=as.character(packageVersion('edgeR')),seconds=edge_seconds),DESeq2=list(version=as.character(packageVersion('DESeq2')),seconds=deseq_seconds)),file.path(d,'reference.json'),auto_unbox=TRUE,pretty=TRUE)
