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

"""This module contains the _GaussianData class, to store and handle
the data passed to ICP. The regression of each set of predictors is
done through this class.

"""

import numpy as np
import copy

# ---------------------------------------------------------------------
# _GaussianData class and its support functions


class _GaussianData():
    """Class to manipulate multi-environment Gaussian data"""

    def __init__(self, data, method='scatter'):
        if method not in ['scatter', 'raw']:
            raise ValueError('method=%s not recognized' % method)
        else:
            self._method = method
        self._data = copy.deepcopy(data)
        self.p = self._data[0].shape[1]
        self.e = len(self._data)
        self.n_obs = np.array([len(X) for X in data])
        self.N = self.n_obs.sum()
        # Compute the pooled correlation matrix (with columns of 1s for intercept)
        self._pooled_data = np.hstack([np.vstack(self._data), np.ones((self.N, 1))])
        self._pooled_correlation = self._pooled_data.T @ self._pooled_data
        # If using the scatter (covariance) matrix for regression,
        # compute the pooled mean and variance
        if method == 'scatter':
            wo_intercept = self._pooled_data[:, :-1]
            self._pooled_covariance = np.cov(wo_intercept, rowvar=False, ddof=0)
            self._pooled_mean = np.mean(wo_intercept, axis=0)

    def regress_pooled(self, y, S):
        S = list(S)
        if self._method == 'scatter':
            return regress(y, S, self._pooled_mean, self._pooled_covariance)
        elif self._method == 'raw':
            coefs = np.zeros(self.p + 1)
            sup = S + [self.p]
            coefs[sup] = np.linalg.lstsq(self._pooled_data[:, sup],
                                         self._pooled_data[:, y], None)[0]
            return coefs[0:self.p], coefs[self.p]

    def residuals(self, y, coefs, intercept):
        # Return the residuals of regressing y using the coefficients
        # coef for each environment.
        return [X[:, y] - X @ coefs - intercept for X in self._data]


def regress(y, S, mean, cov):
    coefs = np.zeros_like(mean)
    # Compute the regression coefficients from the
    # weighted empirical covariance (scatter) matrix i.e. b =
    # Σ_{y,S} @ Σ_{S,S}^-1
    if len(S) > 0:
        coefs[S] = np.linalg.solve(cov[S, :][:, S], cov[y, S])
    intercept = mean[y] - coefs @ mean
    return coefs, intercept

    # def regress_weighted(self, y, S, weights=None):
    #     # Compute pooled covariance and mean
    #     # If not specified, each environment is weighted according to
    #     # the normalized size of its sample
    #     weights = self.n_obs / self.N if weights is None else weights
    #     pooled_cov = np.sum(self.sample_covariances * np.reshape(weights, (self.e, 1, 1)), axis=0)
    #     pooled_mean = np.sum(self.sample_means * np.reshape(weights, (self.e, 1)), axis=0)
    #     # Compute regression
    #     return regress(y, S, pooled_mean, pooled_cov)
