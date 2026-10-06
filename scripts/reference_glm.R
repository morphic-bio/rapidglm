# Black-box comparator: only documented R modelling APIs, no package internals.
args <- commandArgs(TRUE)
root <- normalizePath(args[1], mustWork=FALSE)
dir.create(root, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(MASS))
set.seed(20260926)
write_numeric <- function(x,path) write.table(x,path,sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
run_panel <- function(id,X,Y,family,offset=rep(0,nrow(X)),weights=rep(1,nrow(X))) {
 d <- file.path(root,id);dir.create(d,recursive=TRUE,showWarnings=FALSE)
 write_numeric(X,file.path(d,'design.tsv'));writeMM(as(Matrix(t(Y),sparse=TRUE),'generalMatrix'),file.path(d,'responses.mtx'))
 write_numeric(offset,file.path(d,'offset.tsv'));write_numeric(weights,file.path(d,'weights.tsv'))
 Z <- X[,-ncol(X),drop=FALSE]
 fit_model <- function(x,y) {
  keep <- weights>0; x <- x[keep,,drop=FALSE]; y <- y[keep]; offset <- offset[keep]; weights <- weights[keep]
  if(family=='nb') glm.nb(y ~ x-1+offset(offset),weights=weights,control=glm.control(epsilon=1e-11,maxit=200))
  else glm(y ~ x-1+offset(offset),weights=weights,family=switch(family,gaussian=gaussian(),poisson=poisson(),binomial=binomial()),control=glm.control(epsilon=1e-11,maxit=200))
 }
 rows <- list();start <- proc.time()[['elapsed']]
 for(j in seq_len(ncol(Y))) {
  y <- Y[,j];f <- fit_model(X,y);null <- fit_model(Z,y);s <- coef(summary(f));row <- c(feature=j-1,theta=if(family=='nb')f$theta else NA,log_likelihood=as.numeric(logLik(f)),lr_p=pchisq(2*(as.numeric(logLik(f))-as.numeric(logLik(null))),1,lower.tail=FALSE))
  for(k in seq_len(ncol(X))) row <- c(row,setNames(c(coef(f)[k],s[k,2],s[k,4]),c(paste0('beta',k-1),paste0('se',k-1),paste0('wald_p',k-1))))
  wf <- f$prior.weights*f$family$mu.eta(f$linear.predictors)^2/f$family$variance(f$fitted.values)
  final_cov <- solve(crossprod(model.matrix(f),wf*model.matrix(f)))*summary(f)$dispersion
  for(k in seq_len(ncol(X))) row <- c(row,setNames(sqrt(final_cov[k,k]),paste0('final_se',k-1)))
  rows[[j]] <- row
 }
 elapsed <- proc.time()[['elapsed']]-start
 write.table(do.call(rbind,rows),file.path(d,'reference.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
 writeLines(as.character(elapsed),file.path(d,'reference_seconds.txt'))
 writeLines(family,file.path(d,'family.txt'))
}
n <- 512
X <- cbind(1,rnorm(n),rep(0:1,length.out=n))
offset <- runif(n,-0.5,0.5);weights <- sample(c(0,1,1,1,2),n,replace=TRUE)
for(family in c('gaussian','poisson','binomial','nb')) {
 Y <- sapply(seq_len(12),function(j){eta<-drop(X%*%c(0.3+j/20,-0.25,0.1+j/40))+offset
 switch(family,gaussian=rnorm(n,eta,0.8),poisson=rpois(n,exp(eta)),binomial=rbinom(n,1,plogis(eta)),nb=rnbinom(n,mu=exp(eta),size=c(0.1,0.5,2,10)[(j-1)%%4+1]))})
 run_panel(paste0('E1_',family),X,Y,family,offset,weights)
}
if(length(args)>1) {
 .libPaths(c(args[2],.libPaths()))
 for(mode in c('lowmoi','highmoi')) {
  env<-new.env();name<-paste0(mode,'_example_data');if(length(args)>2)load(file.path(args[3],paste0(name,'.rda')),envir=env)else data(list=name,package='sceptre',envir=env);d<-env[[name]]
  d$extra_covariates$batch<-as.factor(d$extra_covariates[[1]])
  X<-model.matrix(~log1p(Matrix::colSums(d$response_matrix))+log1p(Matrix::colSums(d$response_matrix>0))+batch,data=d$extra_covariates)
  eligible<-which(Matrix::rowSums(d$response_matrix>0)>50)[seq_len(12)]
  run_panel(paste0(if(mode=='lowmoi')'S1' else 'S2',if(length(args)>2)'_public' else '', '_nb'),X,t(as.matrix(d$response_matrix[eligible,,drop=FALSE])),'nb')
 }
}
writeLines(c(R.version.string,paste('MASS',packageVersion('MASS')),paste('Matrix',packageVersion('Matrix'))),file.path(root,'versions.txt'))
