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

"""This module contains the implementation of the Invariant Causal
Prediction algorithm.
"""

from causalicp.gaussian_data import _GaussianData

import numpy as np
import itertools
from termcolor import colored

# For t-test and f-test
import scipy.stats


# ---------------------------------------------------------------------
# "Public" API: fit function

def fit(data, target, alpha=0.05, sets=None, verbose=False):
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
        The index of the response or target variable of interest.
    alpha : float, default=0.05
        The level of the test procedure. Defaults to `0.05`.
    sets : iterable of set or None, default=None
        The sets for which ICP will test invariance. If `None` all
        possible subsets of predictors will be considered.
    verbose: {False, True, 'color'}, default=False
        If ICP should run in verbose mode, i.e. displaying information
        about completion and the result of tests. If 'color', output
        is additionally is color encoded (not recommended if your
        terminal does not support color output).

    Returns
    -------
    result : icp.Result
        An object containing the result of running ICP, i.e. estimate,
        accepted sets, p-values, etc.

    Example
    -------
    >>> import sempler, sempler.generators
    >>> import numpy as np
    >>> np.random.seed(12)
    >>> W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 2)
    >>> scm = sempler.LGANM(W, (-1,1), (1,2))
    >>> data = [scm.sample(n=100)]
    >>> data += [scm.sample(n=130, shift_interventions = {1: (3.1, 5.4)})]
    >>> data += [scm.sample(n=98, do_interventions = {2: (-1, 3)})]

    Run ICP for the response variable `0`, at a significance level of `0.05` (the default).

    >>> import causalicp as icp
    >>> result = icp.fit(data, 3, verbose=True)
    Tested sets and their p-values:
       set() rejected : 2.355990957880749e-10
       {0} rejected : 7.698846116207467e-16
       {1} rejected : 4.573866047163566e-09
       {2} rejected : 8.374476052441259e-08
       {0, 1} accepted : 0.7330408066181638
       {0, 2} rejected : 2.062882130448634e-15
       {1, 2} accepted : 0.8433000000649277
       {0, 1, 2} accepted : 1
    Estimated parental set: {1}

    Obtain the estimate, accepted sets, etc

    >>> result.estimate
    {1}

    >>> result.accepted_sets
    [{0, 1}, {1, 2}, {0, 1, 2}]

    >>> result.rejected_sets
    [set(), {0}, {1}, {2}, {0, 2}]

    >>> result.pvalues
    {0: 0.8433000000649277, 1: 8.374476052441259e-08, 2: 0.7330408066181638, 3: nan}

    >>> result.conf_intervals
    array([[0.        , 0.57167295, 0.        ,        nan],
           [2.11059461, 0.7865869 , 3.87380337,        nan]])

    """
    data = _GaussianData(data, method='scatter')
    # Build set of candidate sets
    if sets is not None:
        candidates = sets
    else:
        base = set(range(data.p))
        base -= {target}
        # max_predictors = data.p - 1 if max_predictors is None else max_predictors
        max_predictors = data.p - 1
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
            set_str = 'rejected' if reject else 'accepted'
            if verbose == 'color':
                color = 'red' if reject else 'green'
                msg = '  ' + colored('%s %s : %s' % (S, set_str, p_value), color)
            else:
                msg = '   %s %s : %s' % (S, set_str, p_value)
            print(msg)
    # If no sets are accepted, there is a model violation. Reflect
    # this by setting the estimate to None
    if len(accepted) == 0:
        estimate = None
    print("Estimated parental set: %s" % estimate) if verbose else None
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


# --------------------------------------------------------------------
# Auxiliary (private) functions

def _test_hypothesis(y, S, data, alpha):
    """Test the hypothesis for the invariance of the set S for the
    target/response y"""
    # Compute pooled coefficients and environment-wise residuals
    coefs, intercept = data.regress_pooled(y, S)
    residuals = data.residuals(y, coefs, intercept)
    # Build p-values for the hypothesis that error distribution
    # remains invariant in each environment
    mean_pvalues = np.zeros(data.e)
    var_pvalues = np.zeros(data.e)
    for i in range(data.e):
        residuals_i = residuals[i]
        residuals_others = np.hstack([residuals[j] for j in range(data.e) if j != i])
        mean_pvalues[i] = _t_test(residuals_i, residuals_others)
        var_pvalues[i] = _f_test(residuals_i, residuals_others)
    # Combine p-values via bonferroni correction
    smallest_pvalue = min(min(mean_pvalues), min(var_pvalues))
    p_value = min(1, smallest_pvalue * 2 * (data.e - 1))  # The -1 term is from the R implementation
    reject = p_value < alpha
    # If set is accepted, compute confidence intervals
    if reject:
        return reject, None, p_value, (coefs, intercept)
    else:
        conf_ints = _confidence_intervals(y, coefs, S, residuals, alpha, data)
        return reject, conf_ints, p_value, (coefs, intercept)


def _t_test(X, Y):
    """Return the p-value of the two sample t-test for the given samples."""
    result = scipy.stats.ttest_ind(X, Y, alternative='two-sided', equal_var=False)
    return result.pvalue


def _f_test(X, Y):
    """Return the p-value of the two-sided f-test for the given samples."""
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = scipy.stats.f.cdf(F, len(X) - 1, len(Y) - 1)
    return 2 * min(p, 1 - p)


def _confidence_intervals(y, coefs, S, residuals, alpha, data):
    """Compute the confidence intervals for the coefficients estimated for
    a set S for the target/response y"""
    # NOTE: This is done following the R implementation. The paper
    # suggests a different approach using the t-distribution.
    # NOTE: `residuals` could be recomputed from `y, data, coefs`; but
    # why waste compute time
    lo = np.zeros(data.p)
    hi = np.zeros(data.p)
    # No need to compute intervals for the empty set
    if len(S) == 0:
        return (lo, hi)
    # Compute individual terms
    S = list(S)
    # 1.1. Estimate residual variance
    residuals = np.hstack(residuals)
    sigma = residuals @ residuals / (data.N - len(S) - 1)
    # 1.2. Estimate std. errors of the coefficients
    sup = S + [data.p]  # Must include intercept in the computation
    correlation = data._pooled_correlation[:, sup][sup, :]
    corr_term = np.diag(np.linalg.inv(correlation))
    std_errors = np.sqrt(sigma * corr_term)[:-1]
    # 2. Quantile term
    quantile = scipy.stats.norm.ppf(1 - alpha / 4)
    # All together
    delta = quantile * std_errors
    lo[S] = coefs[S] - delta
    hi[S] = coefs[S] + delta
    return (lo, hi)


class Result():
    """The result of running Invariant Causal Prediction, i.e. estimate,
    accepted sets, p-values, etc.

    Attributes
    ----------
    p : int
        The total number of variables in the data (including the response/target).
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
    conf_intervals : numpy.ndarray or None
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

    Example
    -------

    >>> import causalicp as icp
    >>> result = icp.fit(data, 3)

    >>> result.p
    4

    >>> result.target
    3

    >>> result.estimate
    set()

    >>> result.accepted_sets
    [{1}, {2}, {0, 1}, {1, 2}, {0, 1, 2}]

    >>> result.rejected_sets
    [set(), {0}, {0, 2}]

    >>> result.pvalues
    {0: 1, 1: 0.18743059830475126, 2: 1, 3: nan}

    >>> result.conf_intervals
    array([[0.        , 0.        , 0.        ,        nan],
           [2.37257655, 1.95012059, 5.88760917,        nan]])

    Get the p-value for invariance of the set {0,2}:
    >>> result.set_pvalues[(0,2)]
    0.015594782960195

    Get the estimated coefficients and intercept for the set `{0,1,2}`:
    >>> result.set_coefficients[(0,1,2)]
    (array([1.7850804 , 0.68359795, 0.82072487, 0.        ]), 0.4147561743079411)

    When all sets are rejected (e.g. there is a model violation), the estimate and confidence intervals are set to None:
    >>> result = icp.fit(data_bad_model, 3)
    >>> result.estimate
    >>> result.conf_intervals

    And the individual p-value for the causal effect of each variable is set to 1:
    >>> result.pvalues
    {0: 1, 1: 1, 2: 1, 3: nan}


    """

    def __init__(self, target, data, estimate, accepted, rejected, conf_intervals, set_pvalues, set_coefs):
        # Save details of setup
        self.p = data.p
        self.target = target

        # Store estimate, sets, set pvalues and coefficients
        self.estimate = estimate if len(accepted) > 0 else None
        self.accepted_sets = sorted(accepted)
        self.rejected_sets = sorted(rejected)
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
                    self.pvalues[j] = np.nan
                else:
                    # p-values of sets not containing j
                    not_j = [pval for S, pval in set_pvalues.items() if j not in S]
                    self.pvalues[j] = max(not_j)

        # Compute confidence intervals
        if len(accepted) == 0:
            self.conf_intervals = None
        else:
            mins = np.array([i[0] for i in conf_intervals])
            maxs = np.array([i[1] for i in conf_intervals])
            self.conf_intervals = np.array([mins.min(axis=0), maxs.max(axis=0)])
            self.conf_intervals[:, target] = np.nan


# To run the doctests
if __name__ == '__main__':
    import doctest
    data = [np.array([[0.46274901, -0.19975643, 0.76993618, 2.65949677],
                      [0.3749258, -0.98625196, -0.1806925, 1.23991796],
                      [-0.39597772, -1.79540294, -0.39718702, -1.31775062],
                      [2.39332284, -3.22549743, 0.15317657, 1.60679175],
                      [-0.56982823, 0.5084231, 0.41380479, 1.19607095],
                      [1.16091539, -0.96445044, 0.76270882, 0.51170294],
                      [-0.71027189, 2.28161765, -0.54339211, -0.29296185],
                      [-0.23015704, 0.67032094, 0.63633403, 0.70513425],
                      [2.36506089, -0.61056788, 1.1203586, 5.1764434],
                      [5.27454441, 2.19246677, 1.6096836, 11.98686647]]),
            np.array([[2.52840275, 9.74119224, 1.63344786, 13.8841425],
                      [0.09269866, 4.90554223, 0.64241815, 4.48748196],
                      [1.31054844, 4.84174639, 1.19053922, 5.7346316],
                      [0.70733964, 4.10331408, -0.51166568, 4.20832483],
                      [-0.48996439, 5.2330853, 0.02711441, 2.21195425],
                      [3.16044076, 10.64310901, 1.30060263, 13.53714596],
                      [2.1720934, 4.2031839, 1.18886156, 8.80078728],
                      [0.54639676, -1.27693243, 0.02822059, 0.91230006],
                      [2.3336745, 1.10881138, 1.02470468, 7.51460385],
                      [2.95781693, 6.2754452, 2.21326654, 11.41247738]]),
            np.array([[-0.64933363, 0.96971792, 0.03185636, 0.77500007],
                      [-3.06213482, -2.27795766, -3.09378221, -9.41389523],
                      [-3.33804428, -2.12536208, -1.06272299, -9.1018036],
                      [-1.36123901, -2.67043653, -0.4331009, -5.54480717],
                      [-5.66507769, -4.28318742, -3.27676386, -14.64719309],
                      [0.31197272, 0.68739597, -0.53645326, -0.36094479],
                      [1.98068289, 0.66270648, 0.67257494, 5.51255352],
                      [1.65037937, 0.85144602, 0.59248564, 4.32634469],
                      [4.12838539, 2.74329139, 1.52358883, 12.01222851],
                      [-0.99472687, 0.59361809, -0.81380456, 0.38239821]])]
    data_bad_model = [np.array([[0.46274901, -0.19975643, 0.76993618, 2.65949677],
                                [0.3749258, -0.98625196, -0.1806925, 1.23991796],
                                [-0.39597772, -1.79540294, -0.39718702, -1.31775062],
                                [2.39332284, -3.22549743, 0.15317657, 1.60679175],
                                [-0.56982823, 0.5084231, 0.41380479, 1.19607095]]),
                      np.array([[1.45648798, 1.39977262, 1.05992289, 2.83848591],
                                [-1.35654212, 6.69077259, -1.14624494, 1.1123806],
                                [-0.48800913, 4.25112687, 0.48421499, 2.55352996],
                                [2.74901219, 1.92465628, 1.49619723, 7.82673868],
                                [5.35033726, 6.01847915, 1.69812062, 14.75126425]]),
                      np.array([[4.52681786, 3.03254075, 1.69178362, -3.04895463],
                                [1.58041688, -0.48726278, 0.49325566, -2.07307932],
                                [1.97983673, -0.35611173, 1.28921189, -1.42832605],
                                [-0.61288207, 1.91706645, 0.18163322, -1.51303223],
                                [0.31303047, -0.53037518, 0.5094926, -2.87618241]])]
    doctest.testmod(extraglobs={'data': data, 'data_bad_model': data_bad_model}, verbose=True)
