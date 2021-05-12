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
Prediction algorithm (see icp.fit).
"""

import numpy as np
import itertools
from termcolor import colored
from causalicp.data import _Data
import scipy.stats  # For t-test and f-test


# ---------------------------------------------------------------------
# "Public" API: fit function

def fit(data, target, alpha=0.05, sets=None, precompute=True, verbose=False, color=True):
    """Run Invariant Causal Prediction on data from different experimental
    settings.

    Parameters
    ----------
    data : numpy.ndarray or list of array-like
        The data from all experimental settings. Each element of the
        list/array is a 2-dimensional array with a sample from a
        different setting, where columns correspond to variables and
        rows to observations (data-points). The data also contains the
        response variable, which is specified with the `target`
        parameter.
    target : int
        The index of the response or target variable of interest.
    alpha : float, default=0.05
        The level of the test procedure, taken from `[0,1]`. Defaults
        to `0.05`.
    sets : list of set or None, default=None
        The sets for which ICP will test invariance. An error is
        raised if a set is not a subset of `{0,...,p-1}` or it
        contains the target, where `p` is the total number of
        variables (including the target). If `None` all possible
        subsets of predictors will be considered.
    precompute : bool, default=True
        Wether to precompute the sample covariance matrix to speed up
        linear regression during the testing of each predictor
        set. For large sample sizes this drastically reduces the
        overall execution time, but it may result in numerical
        instabilities for highly correlated data. If set to `False`,
        for each set of predictors the regression is done using an
        iterative least-squares solver on the raw data.
    verbose: bool, default=False
        If ICP should run in verbose mode, i.e. displaying information
        about completion and the result of tests.
    color : bool, default=True
        If the output produced when `verbose=True` should be color
        encoded (not recommended if your terminal does not support
        ANSII color formatting), see
        `termcolor <https://pypi.org/project/termcolor/>`__.

    Raises
    ------
    ValueError :
        If the value of some of the parameters is not appropriate,
        e.g. `alpha` is negative, `data` contains samples with
        different number of variables, or `sets` contains invalid
        sets.
    TypeError :
        If the type of some of the parameters was not expected (see
        examples below).

    Returns
    -------
    result : causalicp.Result
        A :class:`causalicp.Result` object containing the result of
        running ICP, i.e. estimate, accepted sets, p-values, etc.

    Example
    -------

    Using interventional from a linear-gaussian SCM (generated using
    `sempler <https://github.com/juangamella/sempler>`__)

    >>> data = [np.array([[0.46274901, -0.19975643, 0.76993618, 2.65949677],
    ...                   [0.3749258, -0.98625196, -0.1806925, 1.23991796],
    ...                   [-0.39597772, -1.79540294, -0.39718702, -1.31775062],
    ...                   [2.39332284, -3.22549743, 0.15317657, 1.60679175],
    ...                   [-0.56982823, 0.5084231, 0.41380479, 1.19607095]]),
    ...         np.array([[1.45648798, 8.29977262, 1.05992289, 7.49191164],
    ...                   [-1.35654212, 13.59077259, -1.14624494, 5.76580633],
    ...                   [-0.48800913, 11.15112687, 0.48421499, 7.20695569],
    ...                   [2.74901219, 8.82465628, 1.49619723, 12.48016441],
    ...                   [5.35033726, 12.91847915, 1.69812062, 19.40468998]]),
    ...         np.array([[-11.73619893, -6.87502658, -6.71775898, -28.2782561],
    ...                   [-16.24118216, -11.26774231, -9.22041168, -42.09076079],
    ...                   [-14.85266731, -11.02688079, -8.71264951, -40.37471919],
    ...                   [-16.08519052, -11.73497156, -10.58198058, -42.55646184],
    ...                   [-17.07817707, -11.29005529, -10.04063011, -45.01702447]])]

    Running ICP for the response variable `3`, at a significance level of `0.05`.

    >>> import causalicp as icp
    >>> result = icp.fit(data, 3, alpha=0.05, precompute=True, verbose=True, color=False)
    Tested sets and their p-values:
      set() rejected : 6.8529852769059795e-06
      {0} rejected : 0.043550405609324994
      {1} rejected : 6.10963528362226e-06
      {2} rejected : 0.009731028782704005
      {0, 1} accepted : 0.9107055098714101
      {0, 2} rejected : 0.004160395025223608
      {1, 2} accepted : 1
      {0, 1, 2} accepted : 1
    Estimated parental set: {1}


    Obtaining the estimate, accepted sets, etc

    >>> result.estimate
    {1}

    >>> result.accepted_sets
    [{0, 1}, {1, 2}, {0, 1, 2}]

    >>> result.rejected_sets
    [set(), {0}, {1}, {2}, {0, 2}]

    >>> result.pvalues
    {0: 1, 1: 0.043550405609324994, 2: 0.9107055098714101, 3: nan}

    >>> result.conf_intervals
    array([[0.        , 0.37617783, 0.        ,        nan],
           [2.3531227 , 0.89116407, 4.25277329,        nan]])

    **Examples of exceptions**

    A `TypeError` is raised for parameters of the wrong type, and
    `ValueError` if they are not valid. For example, if `alpha` is not
    a float between 0 and 1,

    >>> icp.fit(data, 3, alpha = 1)
    Traceback (most recent call last):
      ...
    TypeError: alpha must be a float, not <class 'int'>.

    >>> icp.fit(data, 3, alpha = -0.1)
    Traceback (most recent call last):
      ...
    ValueError: alpha must be in [0,1].

    >>> icp.fit(data, 3, alpha = 1.1)
    Traceback (most recent call last):
      ...
    ValueError: alpha must be in [0,1].

    if the target is not an integer within range,

    >>> icp.fit(data, 3.0)
    Traceback (most recent call last):
      ...
    TypeError: target must be an int, not <class 'float'>.

    >>> icp.fit(data, 5)
    Traceback (most recent call last):
      ...
    ValueError: target must be an integer in [0, p-1].

    if `sets` is of the wrong type or contains an invalid set,

    >>> icp.fit(data, 3, sets = [{2}, {1,3}])
    Traceback (most recent call last):
      ...
    ValueError: Set {1, 3} in sets is not valid: it must be a subset of {0,...,p-1} - {target}.

    >>> icp.fit(data, 3, sets = ({2}, {0,1}))
    Traceback (most recent call last):
      ...
    TypeError: sets must be a list of set, not <class 'tuple'>.

    >>> icp.fit(data, 3, sets = [(2,), (0,1)])
    Traceback (most recent call last):
      ...
    TypeError: sets must be a list of set, not of <class 'tuple'>.

    if `precompute`, `verbose` or `color` are not of type `bool`,

    >>> icp.fit(data, 3, precompute=1)
    Traceback (most recent call last):
      ...
    TypeError: precompute must be bool, not <class 'int'>.

    >>> icp.fit(data, 3, verbose=1)
    Traceback (most recent call last):
      ...
    TypeError: verbose must be bool, not <class 'int'>.

    >>> icp.fit(data, 3, color=1)
    Traceback (most recent call last):
      ...
    TypeError: color must be bool, not <class 'int'>.

    or if the samples from each experimental setting have different numbers of variables,

    >>> data = [[[0.01, 0.02],[0.03,0.04]], [[0.01],[0.03]]]
    >>> icp.fit(data, 3)
    Traceback (most recent call last):
      ...
    ValueError: The samples from each setting have a different number of variables: [2 1].

    """
    # Check inputs: alpha
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float, not %s." % type(alpha))
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1].")

    # Check inputs: data
    data = _Data(data, method='scatter' if precompute else 'raw')

    # Check inputs: target
    if not isinstance(target, int):
        raise TypeError("target must be an int, not %s." % type(target))
    if target < 0 or target >= data.p or int(target) != target:
        raise ValueError("target must be an integer in [0, p-1].")

    # Check inputs: precompute
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be bool, not %s." % type(precompute))

    # Check inputs: verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be bool, not %s." % type(verbose))

    # Check inputs: color
    if not isinstance(color, bool):
        raise TypeError("color must be bool, not %s." % type(color))

    # Check inputs: sets
    base = set(range(data.p))
    base -= {target}
    # If sets is provided, check its validity
    if sets is not None:
        if not isinstance(sets, list):
            raise TypeError("sets must be a list of set, not %s." % type(sets))
        else:
            for s in sets:
                if not isinstance(s, set):
                    raise TypeError("sets must be a list of set, not of %s." % type(s))
                elif len(s - base) > 0:
                    raise ValueError(
                        "Set %s in sets is not valid: it must be a subset of {0,...,p-1} - {target}." % s)
        candidates = sets
    # Build set of candidate sets
    else:
        # max_predictors = data.p - 1 if max_predictors is None else max_predictors
        max_predictors = data.p - 1
        candidates = []
        for set_size in range(max_predictors + 1):
            candidates += list(itertools.combinations(base, set_size))

    # ----------------------------------------------------------------
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
            if color:
                color = 'red' if reject else 'green'
                msg = colored('  %s %s : %s' % (S, set_str, p_value), color)
            else:
                msg = '  %s %s : %s' % (S, set_str, p_value)
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
    reject = p_value <= alpha
    # If set is accepted, compute confidence intervals
    if reject:
        return reject, None, p_value, (coefs, intercept)
    else:
        conf_ints = _confidence_intervals(y, coefs, S, residuals, alpha, data)
        return reject, conf_ints, p_value, (coefs, intercept)


def _t_test(X, Y):
    """Return the p-value of the two sample t-test for the given samples."""
    result = scipy.stats.ttest_ind(X, Y, equal_var=False)
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
    correlation = data.pooled_correlation[:, sup][sup, :]
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
    """The result of running Invariant Causal Prediction, produced as
    output of :meth:`causalicp.fit`.

    Attributes
    ----------
    p : int
        The total number of variables in the data (including the response/target).
    target : int
        The index of the response/target.
    estimate : set or None
        The estimated parental set returned by ICP, or `None` if all
        sets of predictors were rejected.
    accepted_sets : list of set
        A list containing the accepted sets of predictors.
    rejected_sets : list of set
        A list containing the rejected sets of predictors.
    pvalues : dict of (int, float)
        A dictionary containing the p-value for the causal effect of
        each individual predictor. The target/response is included in
        the dictionary and has value `nan`.
    conf_intervals : numpy.ndarray or None
        A `2 x p` array of floats representing the confidence interval
        for the causal effect of each variable. Each column
        corresponds to a variable, and the first and second row
        correspond to the lower and upper limit of the interval,
        respectively. The column corresponding to the target/response
        is set to `nan`.

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
    {0: 1, 1: 0.187430598304751, 2: 1, 3: nan}

    >>> result.conf_intervals
    array([[0.        , 0.        , 0.        ,        nan],
           [2.37257655, 1.95012059, 5.88760917,        nan]])

    When all sets are rejected (e.g. there is a model violation), the
    estimate and confidence intervals are set to `None`:

    >>> result = icp.fit(data_bad_model, 3)
    >>> result.estimate
    >>> result.conf_intervals

    And the individual p-value for the causal effect of each variable
    is set to `1`:

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
    doctest.testmod(extraglobs={'data2': data, 'data': data,
                                'data_bad_model': data_bad_model}, verbose=True)
