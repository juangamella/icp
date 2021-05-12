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

# ---------------------------------------------------------------------
# Unit tests

import unittest
import numpy as np
import copy

# Tested functions
from causalicp.data import _Data


class DataTests(unittest.TestCase):

    def setUp(self):
        self.p = 20
        self.n_obs = [2, 3, 4]
        self.N = np.sum(self.n_obs)
        self.e = len(self.n_obs)
        self.target = 3
        XX = []
        for i, ne in enumerate(self.n_obs):
            X = np.tile(np.ones(self.p), (ne, 1))
            X *= (i + 1)
            X[:, self.target] *= -1
            XX.append(X)
        self.XX = XX

    def test_basic(self):
        data = _Data(self.XX)
        self.assertEqual(data.N, self.N)
        self.assertTrue((data.n_obs == self.n_obs).all())
        self.assertEqual(data.p, self.p)
        self.assertEqual(data.e, self.e)
        self.assertEqual(data.e, len(self.XX))

    def test_memory(self):
        # Test that the data is copied into the class
        XX = copy.deepcopy(self.XX)
        data = _Data(XX)
        XX[0][0, 0] = -100
        data_pooled = data._pooled_data
        self.assertFalse(data_pooled[0, 0] == XX[0][0, 0])
