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

import pandas as pd
from .gaussian_data import GaussianData

import numpy as np
from functools import reduce
import itertools
from termcolor import colored

# For t-test and f-test
import scipy.stats


# ---------------------------------------------------------------------
# "Public" API: icp function


def fit(data, target, alpha=0.05, conf_ints=False, selection=None, max_predictors=None, verbose=False):
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
    data = GaussianData(data)
    # Build set of candidates
    if selection is not None:
        base = set(list(selection))
    else:
        base = set(range(data.p))
    base -= {target}
    max_predictors = data.p - 1 if max_predictors is None else max_predictors
    candidates = []
    for set_size in range(max_predictors + 1):
        candidates += list(itertools.combinations(base, set_size))
    # Evaluate candidates
    accepted = []  # To store the accepted sets
    rejected = []  # To store the sets that were rejected
    confidence_intervals = [] if conf_ints else None
    S = base
    print("Tested sets and their p-values") if verbose else None
    for s in candidates:
        s = set(s)
        # Test hypothesis
        if conf_ints:
            reject, conf_interval = _test_hypothesis(target, s, data, alpha, conf_ints)
            confidence_intervals.append(conf_interval)
        else:
            reject = _test_hypothesis(target, s, data, alpha, conf_ints)
        # Update accepted sets and intersection
        if reject:
            rejected.append(s)
        else:
            accepted.append(s)
            S = S.intersection(s)
        if verbose:
            color = "red" if reject else "green"
            set_str = "rejected" if reject else "accepted"
            msg = "  " + colored("%s %s" % (s, set_str), color)
            print(msg)
    print("\nEstimated parental set: %s\n" % S) if verbose else None
    return Result(S, accepted, rejected, confidence_intervals)


# Support functions to icp


def _test_hypothesis(y, s, data, alpha, conf_ints=False):
    # Compute pooled coefficients
    coefs, intercept = data.regress_pooled(y, s)
    #print(s, coefs, intercept)
    residuals = data.residuals(y, coefs, intercept)
    # Build p-values for the hypothesis that error distribution
    # remains invariant in each environment
    mean_pvalues = np.zeros(data.e)
    var_pvalues = np.zeros(data.e)
    for i in range(data.e):
        residuals_i = residuals[i]
        residuals_others = np.hstack([residuals[j] for j in range(data.e) if j != i])
        # if s == {4}:
        #     pd.DataFrame(residuals_i).to_csv('residuals_%d_a.csv' % i)
        #     pd.DataFrame(residuals_others).to_csv('residuals_%d_b.csv' % i)
        mean_pvalues[i] = t_test(residuals_i, residuals_others)
        var_pvalues[i] = f_test(residuals_i, residuals_others)
    # Combine via bonferroni correction
    smallest_pvalue = min(min(mean_pvalues), min(var_pvalues))
    reject = smallest_pvalue < alpha / 2 / (data.e - 1)  # The -1 term is from the R implementation
    # Optionally, compute confidence intervals. Done here to avoid
    # re-computing residuals
    if conf_ints and (reject or len(s) == 0):
        return reject, (np.ones(data.p) * np.inf, np.ones(data.p) * - np.inf)
    elif conf_ints:
        conf_ints = confidence_intervals(y, coefs, s, residuals, alpha, data)
        return reject, conf_ints
    else:
        return reject


def t_test(X, Y):
    """Return the p-value of the two sample f-test for
    the given sample"""
    result = scipy.stats.ttest_ind(X, Y, alternative='two-sided', equal_var=False)
    return result.pvalue


def f_test(X, Y):
    """Return the p-value of the two sample t-test for
    the given sample"""
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = scipy.stats.f.cdf(F, len(X) - 1, len(Y) - 1)
    return 2 * min(p, 1 - p)


def confidence_intervals(y, coefs, S, residuals, alpha, data):
    # Estimated residual standard deviation
    sigma = np.std(residuals)
    # Quantile term
    S = list(S)
    dof = data.N - len(S) - 1
    quantile = scipy.stats.t.ppf(1 - alpha / 2 / len(S), dof)
    # Correlation matrix term
    corr_term = np.diag(np.linalg.inv(data._pooled_correlation[:, S][S, :]))
    print(dof, quantile, sigma, corr_term)
    delta = quantile * sigma * corr_term
    lo = np.ones(data.p) * np.inf
    hi = np.ones(data.p) * - np.inf
    #    hi = np.zeros(data.p), np.zeros(data.p)
    lo[S] = coefs[S] - delta
    hi[S] = coefs[S] + delta
    print(S, (lo, hi))
    return (lo, hi)


class Result():
    """Class to hold the estimate produced by ICP and any additional information"""

    def __init__(self, estimate, accepted, rejected, conf_intervals=None):
        self.estimate = estimate  # The estimate produced by ICP ie. intersection of accepted sets
        self.accepted = sorted(accepted)  # Accepted sets
        self.rejected = sorted(rejected)  # Rejected sets
        # Compute confidence intervals
        if conf_intervals is not None:
            mins = np.array([i[0] for i in conf_intervals])
            maxs = np.array([i[1] for i in conf_intervals])
            print(mins)
            print()
            print(maxs)
            self.conf_intervals = mins.min(axis=0), maxs.max(axis=0)
        else:
            self.conf_intervals = None
