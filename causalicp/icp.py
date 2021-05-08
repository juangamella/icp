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
    result : icp.Result
        An object containing the result of running ICP, i.e. estimate,
        accepted sets, p-values, etc.

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

    # Evaluate candidate sets
    accepted = []  # To store the accepted sets
    rejected = []  # To store the sets that were rejected
    confidence_intervals = []
    p_values = {}  # To store the p-values of the tested sets
    coefficients = {}  # To store the estimated coefficients of the tested sets
    estimate = base
    print("Tested sets and their p-values:") if verbose else None
    # Test each set
    for S in candidates:
        S = set(S)
        # Test hypothesis of invariance
        reject, conf_interval, p_value, coefs = _test_hypothesis(target, S, data, alpha)
        # Store result appropriately and update estimate (if necessary)
        p_values[tuple(S)] = p_value
        coefficients[tuple(S)] = coefs
        if not reject:
            confidence_intervals.append(conf_interval)
            accepted.append(S)
            estimate &= S
        if reject:
            rejected.append(S)
        # Optionally, print output
        if verbose:
            color = 'red' if reject else 'green'
            set_str = 'rejected' if reject else 'accepted'
            msg = '  ' + colored('%s %s : %s' % (S, set_str, p_value), color)
            print(msg)
    # If no sets are accepted, there is a model violation. Reflect
    # this by setting the estimate to None
    if len(accepted) == 0:
        estimate = None
    print("gEstimated parental set: %s\n" % estimate) if verbose else None
    # Create and return the result object
    result = Result(target,
                    data,
                    estimate,
                    accepted,
                    rejected,
                    confidence_intervals,
                    p_values,
                    coefficients)
    return result


# Support functions to icp


def _test_hypothesis(y, S, data, alpha):
    # Compute pooled coefficients and residuals
    coefs, intercept = data.regress_pooled(y, S)
    residuals = data.residuals(y, coefs, intercept)
    # print(s, coefs, intercept)
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
        mean_pvalues[i] = _t_test(residuals_i, residuals_others)
        var_pvalues[i] = _f_test(residuals_i, residuals_others)
    # Combine p-values via bonferroni correction
    smallest_pvalue = min(min(mean_pvalues), min(var_pvalues))
    p_value = min(1, smallest_pvalue * 2 * (data.e - 1))  # The -1 term is from the R implementation
    reject = p_value < alpha
    # If set is accepted, compute p-values
    if reject:
        return reject, None, p_value, (coefs, intercept)
    else:
        conf_ints = _confidence_intervals(y, coefs, S, residuals, alpha, data)
        return reject, conf_ints, p_value, (coefs, intercept)


def _t_test(X, Y):
    """Return the p-value of the two sample f-test for
    the given sample"""
    result = scipy.stats.ttest_ind(X, Y, alternative='two-sided', equal_var=False)
    return result.pvalue


def _f_test(X, Y):
    """Return the p-value of the two sample t-test for
    the given sample"""
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = scipy.stats.f.cdf(F, len(X) - 1, len(Y) - 1)
    return 2 * min(p, 1 - p)


def _confidence_intervals(y, coefs, S, residuals, alpha, data):
    lo = np.ones(data.p) * np.inf
    hi = np.ones(data.p) * - np.inf
    # No need to compute intervals for the empty set
    if len(S) == 0:
        return (lo, hi)
    # Compute individual terms
    # 1. Estimated residual standard deviation
    sigma = np.std(residuals)
    # 2. Quantile term
    S = list(S)
    dof = data.N - len(S) - 1
    quantile = scipy.stats.t.ppf(1 - alpha / 2 / len(S), dof)
    # 3. Correlation matrix term
    corr_term = np.diag(np.linalg.inv(data._pooled_correlation[:, S][S, :]))
    #print(dof, quantile, sigma, corr_term)
    # Putting it all together
    delta = quantile * sigma * corr_term
    lo[S] = coefs[S] - delta
    hi[S] = coefs[S] + delta
    #print(S, (lo, hi))
    return (lo, hi)


class Result():
    """The result of running Invariant Causal Prediction, i.e. estimate,
    accepted sets, p-values, etc.

    Parameters
    ----------
    p : int
        The total number of variables in the data (including the response/target).
    e : int
        The number of environments/experimental settings.
    target : int
        The index of the target/response.
    estimate : set or None
        The estimated parental set returned by ICP, or `None` if no
        sets of predictors were accepted.
    accepted_sets : list of set
        A list containing the accepted sets of predictors.
    rejected_sets : list of set
        A list containing the rejected sets of predictors.
    pvalues : dict of (int, float)
        A dictionary containing the p-value for the causal effect of
        each individual predictor.
    conf_intervals : numpy.ndarray
        A `2 x p` array of floats representing the confidence interval
        for the causal effect of each variable. Each column
        corresponds to a variable, and the first and second row
        correspond to the lower and upper limit of the interval,
        respectively. The column corresponding to the target/response
        is set to `nan`.
    set_pvalues : dict
        A dictionary containing the p-value for the invariance test
        for each of the tested sets.
    set_coefficients : dict
        A dictionary containing the coefficients + intercept estimated
        for each of the tested sets.

    """

    def __init__(self, target, data, estimate, accepted, rejected, conf_intervals, set_pvalues, set_coefs):
        # Save details of setup
        self.e = data.e
        self.p = data.p
        self.target = target

        # Store estimate, sets, set pvalues and coefficients
        self.estimate = estimate
        self.accepted = sorted(accepted)
        self.rejected = sorted(rejected)
        self.set_coefficients = set_coefs
        self.set_pvalues = set_pvalues

        # Compute p-values for individual variables
        if len(accepted) == 0:
            # If all sets are rejected, set the p-value of all
            # variables to 1
            self.pvalues = dict((j, 1) for j in range(self.p))
            self.pvalues[target] = np.nan
        else:
            # Otherwise, the p-value of each variable is the highest
            # among the p-values of the sets not containing j
            self.pvalues = {}
            for j in range(self.p):
                if j == target:
                    self.p_values[j] = np.nan
                else:
                    # p_values of sets not containing j
                    not_j = [pval for S, pval in set_pvalues.items() if j not in S]
                    self.pvalues[j] = max(not_j)

        # Compute confidence intervals
        # mins = np.array([i[0] for i in conf_intervals])
        # maxs = np.array([i[1] for i in conf_intervals])
        # print(mins)
        # print()
        # print(maxs)
        # self.conf_intervals = mins.min(axis=0), maxs.max(axis=0)
        self.conf_intervals = None
