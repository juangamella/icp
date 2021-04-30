# Copyright 2021 Juan L. Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""This module contains the finite sample implementation of Invariant
Causal Prediction, with a two-sample t-test and f-test to check the
invariance of the conditional distribution.

TODO  BEFORE PUBLISHING:
  - color output by termcolor is not portable to all OSs, so deactivate it

"""


import numpy as np

from scipy.stats import ttest_ind as ttest
from scipy.stats import f
from scipy.stats import t

from sklearn.linear_model import LinearRegression

from functools import reduce
import itertools

import copy

from termcolor import colored
#---------------------------------------------------------------------
# "Public" API: icp function

def fit(data, target, alpha=0.05, selection=None, max_predictors=None, verbose=False):
    """Run Invariant Causal Prediction on data from different experimental
    settings.

    Parameters
    ----------
    data : list of array-like
        The data from all experimental settings. Each element of the
        list is an array with a sample from a different setting, where
        columns correspond to variables and rows to observations
        (data-points). The data also contains the response variable,
        which is specified with the `target` parameter.
    target : int
        The response or target variable of interest.
    alpha : float, default=0.05
        The level of the test procedure. Defaults to `0.05`.
    selection : iterable of ints, optional
        A pre-selection of the predictors which ICP will consider,
        given as an iterable containing the variable indices, e.g. a
        list or set. If not specified, ICP will consider all
        predictors.
    max_predictors: int, optional
        The maximum number of predictors ICP should consider, i.e. the
        largest size of the sets which are considered. If not
        specified, ICP will consider sets up to size `p-1`, i.e. the
        total number of predictors.
    verbose: bool, default=False
        If ICP should run in verbose mode, i.e. displaying information
        about completion and the result of tests.
    
    Returns
    -------
    

    """
    # Check inputs
    data = Data(data, target)
    # Build set of candidates
    if isinstance(selection, list):
        base = reduce(lambda union, s: set.union(union, s), selection, set())
        candidates = selection
    else:
        max_predictors = data.p-1 if max_predictors is None else max_predictors
        base = set(range(data.p))
        base.remove(target)
        candidates = []
        for set_size in range(max_predictors+1):
            candidates += list(itertools.combinations(base, set_size))
    # Evaluate candidates
    accepted = [] # To store the accepted sets
    rejected = [] # To store the sets that were rejected
    mses = [] # To store the MSE of the accepted sets
    S = base
    print("Tested sets and their p-values") if verbose else None
    for s in candidates:
        s = set(s)
        # Find linear coefficients on pooled data
        (beta, error) = regress(s, data)
        assert((beta[list(base.difference(s))] == 0).all())
        p_value = test_hypothesis(beta, data)
        reject = p_value < alpha
        if reject:
            rejected.append(s)
        else:
            accepted.append(s)
            S = S.intersection(s)
        if verbose:
            color = "red" if reject else "green"
            set_str = "rejected" if reject else "accepted"
            msg = "  " + colored("%s %s" % (s, set_str), color) + " - (p=%0.4f)" % p_value
            print(msg)
    print("\nEstimated parental set: %s\n" % S) if verbose else None
    return Result(S, accepted, rejected, mses, None)

# Support functions to icp

def test_hypothesis(coefs, data):
    """Test hypothesis for a vector of coefficients coefs, using the t-test for the mean
    and f-test for the variances, and returning the p-value

    """
    mean_pvalues = np.zeros(data.n_env)
    var_pvalues = np.zeros(data.n_env)
    #residuals = data.pooled_targets() - data.pooled_data() @ coefs
    for i in range(data.n_env):
        (env_targets, env_data, others_targets, others_data) = data.split(i)
        residuals_env = env_targets - env_data @ coefs
        residuals_others = others_targets - others_data @ coefs
        # residuals_env = residuals[data.idx == i]
        # residuals_others = residuals[data.idx != i]
        mean_pvalues[i] = t_test(residuals_env, residuals_others)
        var_pvalues[i] = f_test(residuals_env, residuals_others)
        assert(mean_pvalues[i] <= 1)
        assert(var_pvalues[i] <= 1)
    # Combine via bonferroni correction
    pvalue_mean = min(mean_pvalues) * data.n_env
    pvalue_var = min(var_pvalues) * data.n_env
    # Return two times the smallest p-value
    return min(pvalue_mean, pvalue_var) * 2

def regress(s, data, pooling=True):
    """
    Perform the linear regression of data.target over the variables indexed by s
    """
    supp = list(s) + [data.p] # support is pred. set + intercept
    if pooling:
        X = data.pooled_data()[:,supp]
        Y = data.pooled_targets()
    coefs = np.zeros(data.p+1)
    coefs[supp] = np.linalg.lstsq(X,Y, None)[0]
    error = 0 #mse(Y, data.pooled_data() @ coefs)
    return coefs, error

def t_test(X,Y):
    """Return the p-value of the two sample f-test for
    the given sample"""
    result = ttest(X, Y, equal_var=False)
    return result.pvalue

def f_test(X,Y):
    """Return the p-value of the two sample t-test for
    the given sample"""
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = f.cdf(F, len(X)-1, len(Y)-1)
    return  2*min(p, 1-p)

def confidence_intervals(s, coefs, data, alpha):
    """Compute the confidence intervals of the regression coefficients
    (coefs) of a predictor set s, given the level alpha.

    Under Gaussian errors, the confidence intervals are given by
    coefs +/- delta, where

    delta = quantile * variance of residuals @ diag(inv. corr. matrix)

    and variance and corr. matrix of residuals are estimates
    """
    s = list(s)
    supp = s + [data.p] # Support is pred. set + intercept
    coefs = coefs[supp]
    # Quantile term
    dof = data.n - len(s) - 1
    quantile = t.ppf(1-alpha/2/len(s), dof)
    # Residual variance term
    Xs = data.pooled_data()[:,supp]
    residuals = data.pooled_targets() - Xs @ coefs
    variance = np.var(residuals)
    # Corr. matrix term
    sigma = np.diag(np.linalg.inv(Xs.T @ Xs))
    # Compute interval
    delta = quantile * np.sqrt(variance) * sigma
    return (coefs - delta, coefs + delta)
    
#---------------------------------------------------------------------
# Data class and its support functions

class Data():
    """Class to handle access to the dataset. Takes a list of
    environments (each environment is an np array containing the
    observations) and the index of the target.

    Parameters:
      - p: the number of variables
      - n: the total number of samples
      - N: list with number of samples in each environment
      - n_env: the number of environments
      - targets: list with the observations of the target in each environment
      - data: list with the observations of the other vars. in each environment
      - target: the index of the target variable

    """
    def __init__(self, environments, target):
        """Initializes the object by separating the observations of the target
        from the rest of the data, and obtaining the number of
        variables, number of samples per environment and total number
        of samples.

        Arguments:
          - environments: list of np.arrays of dim. (n_e, p), each one
            containing the data of an environment. n_e is the number of
            samples for that environment and p is the number of variables.
          - target: the index of the target variable
        """
        environments = copy.deepcopy(environments) # ensure the stored data is immutable
        self.N = np.array(list(map(len, environments)))
        self.p = environments[0].shape[1]
        self.n = np.sum(self.N)
        self.n_env = len(environments)
        # Extract targets and add a col. of 1s for the intercept
        self.targets = list(map(lambda e: e[:,target], environments))
        self.data = list(map(lambda e: np.hstack([e, np.ones((len(e),1))]), environments))
        self.target = target
        # Construct an index array
        self.idx = np.zeros(self.n)
        ends = np.cumsum(self.N)
        starts = np.zeros_like(ends)
        starts[1::] = ends[:-1]
        for i,start in enumerate(starts):
            end = ends[i]
            self.idx[start:end] = i

    def pooled_data(self):
        """Returns the observations of all variables except the target,
        pooled."""
        return pool(self.data, 0)

    def pooled_targets(self):
        """Returns all the observations of the target variable, pooled."""
        return pool(self.targets, 1)

    def split(self, i):
        """Splits the dataset into targets/data of environment i and
        targets/data of other environments pooled together."""
        rest_data = [d for k,d in enumerate(self.data) if k!=i]
        rest_targets = [t for k,t in enumerate(self.targets) if k!=i]
        self.data[i]
        return (self.targets[i], self.data[i], pool(rest_targets, 1), pool(rest_data, 0))

def pool(arrays, axis):
    """Takes a list() of numpy arrays and returns them in an new
    array, stacked along the given axis.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        stack_fun = np.vstack if axis==0 else np.hstack
        return reduce(lambda acc, array: stack_fun([acc, array]), arrays)

#---------------------------------------------------------------------
# Results class

class Result():
    """Class to hold the estimate produced by ICP and any additional information"""
    def __init__(self, estimate, accepted, rejected, mses, conf_intervals=None):
        self.estimate = estimate # The estimate produced by ICP ie. intersection of accepted sets
        self.accepted = accepted # Accepted sets
        self.rejected = rejected # Rejected sets
        self.mses = np.array(mses) # MSE of the accepted sets
        self.conf_intervals = conf_intervals # Confidence intervals
