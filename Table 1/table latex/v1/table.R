library(xtable)

# read.excel <- function(header=TRUE,...) {
#   read.table("clipboard",sep="\t",header=header,...)
# }
# 
# dat=read.excel()

Xtable_dat<-xtable(dat)
print(Xtable_dat,file="Xtabledat.txt")




