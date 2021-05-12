#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(InvariantCausalPrediction)

CASES = strtoi(args[1])

print(sprintf('\nTOTAL NUMBER OF CASES: %d', CASES))

case_no = 0

while (case_no < CASES) {

    print(sprintf('\n\nTrying test case %d', case_no))
    
    ## Read csv
    path = 'causalicp/test/test_cases/'
    filename = sprintf('%scase_%d.csv', path, case_no)
    raw = read.csv(filename, header=FALSE)
    
    ## Process the data and set up variables for ICP
    target <- 0 # target is always 0
    ExpInd <- raw[,1] + 1
    Y <- raw[,target+2]
    preds <- c(2:ncol(raw))
    preds <- preds[-(target+1)]
    X <- data.matrix(raw[, preds])

    ## Run ICP
    alpha <- 0.001
    icp = ICP(X,Y,ExpInd,selection='all',alpha=alpha, gof=alpha, showCompletion=FALSE)
    ## print(icp)

    ## One-hot encode accepted sets
    one_hot <- matrix(0, length(icp$acceptedSets), ncol(X))
    i = 1
    for (set in icp$acceptedSets) {    
        one_hot[i,set] <- 1
        i = i + 1
    }
    ## print(one_hot)

    ## Write accepted sets
    filename = sprintf('%sicp_result_%d_accepted.csv', path, case_no)
    write.csv(one_hot, filename)
    ## Write confidence intervals
    filename = sprintf('%sicp_result_%d_confints.csv', path, case_no)
    write.csv(icp$ConfInt, filename)
    ## Write p-values
    filename = sprintf('%sicp_result_%d_pvals.csv', path, case_no)
    write.csv(icp$pvalues, filename)
    
    case_no = case_no + 1
    
    }
