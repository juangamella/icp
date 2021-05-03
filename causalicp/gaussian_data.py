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

"""
"""

import numpy as np
import copy

# ---------------------------------------------------------------------
# GaussianData class and its support functions


class GaussianData():
    """Class to manipulate multi-environment Gaussian data"""

    def __init__(self, data):
        self._data = copy.deepcopy(data)
        self.p = self._data[0].shape[1]
        self.e = len(self._data)
        # Compute sample covariances, correlation matrices and sample
        # sizes from each environment
        self.sample_covariances = np.array([np.cov(X, rowvar=False, ddof=0) for X in self._data])
        self.sample_means = np.array([np.mean(X, axis=0) for X in self._data])
        correlation_matrices = []
        n_obs = []
        for X in self._data:
            n_obs.append(len(X))
            aux = np.hstack([X, np.ones((len(X), 1))])
            correlation_matrices.append(aux.T @ aux)
        self.correlation_matrices = np.array(correlation_matrices)
        self.n_obs = np.array(n_obs)
        self.N = self.n_obs.sum()

    def regress_pooled(self, y, S, method='scatter'):
        if method == 'scatter':
            return self.regress_weighted(y, S)
        elif method == 'raw':
            S = list(S)
            pooled = np.hstack([np.vstack(self._data), np.ones((self.N, 1))])
            coefs = np.zeros(self.p + 1)
            coefs[S + [self.p]] = np.linalg.lstsq(pooled[:, S + [self.p]], pooled[:, y], None)[0]
            return coefs[0:self.p], coefs[self.p]

    def regress_weighted(self, y, S, weights=None):
        # Compute pooled covariance and mean
        # If not specified, each environment is weighted according to
        # the normalized size of its sample
        weights = self.n_obs / self.N if weights is None else weights
        pooled_cov = np.sum(self.sample_covariances * np.reshape(weights, (self.e, 1, 1)), axis=0)
        pooled_mean = np.sum(self.sample_means * np.reshape(weights, (self.e, 1)), axis=0)
        # Compute regression
        return regress(y, S, pooled_mean, pooled_cov)

    def residuals(self, y, coefs, intercept):
        # Return the residuals of regressing y using the coefficients
        # coef for each environment.
        return [X[:, y] - X @ coefs - intercept for X in self._data]


def regress(y, S, mean, cov):
    S = sorted(S)
    coefs = np.zeros_like(mean)
    # Compute the regression coefficients from the
    # weighted empirical covariance (scatter) matrix i.e. b =
    # Σ_{y,S} @ Σ_{S,S}^-1
    if len(S) > 0:
        coefs[S] = np.linalg.solve(cov[S, :][:, S], cov[y, S])
    intercept = mean[y] - coefs @ mean
    return coefs, intercept
