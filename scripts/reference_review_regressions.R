suppressPackageStartupMessages(library(DESeq2))
y<-c(rep(0,10),10+(10:19)%%4)
dds<-DESeqDataSetFromMatrix(matrix(y,nrow=1),data.frame(group=factor(rep(c('A','B'),each=10))),~group)
sizeFactors(dds)<-rep(1,20);dispersions(dds)<-.2
dds<-nbinomWaldTest(dds,betaPrior=FALSE,minmu=.5,quiet=TRUE)
r<-results(dds,cooksCutoff=FALSE,independentFiltering=FALSE)
print(as.data.frame(r),digits=16)

# The LR test evaluates the returned means after the floored coefficient fit.
x <- model.matrix(~group, colData(dds))
dds <- nbinomLRT(dds, full=x, reduced=matrix(1,20,1), minmu=.5, quiet=TRUE)
print(as.data.frame(results(dds,cooksCutoff=FALSE,independentFiltering=FALSE)),digits=16)
