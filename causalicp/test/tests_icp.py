# Copyright 2019 Juan L Gamella

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

#---------------------------------------------------------------------
# Unit tests for module icp.py

import unittest
import numpy as np
import copy

# Tested functions
from ..icp import Data, Result

class DataTests(unittest.TestCase):

    def setUp(self):
        self.p = 20
        self.N = [2, 3, 4]
        self.n = np.sum(self.N)
        self.target = 3
        environments = []
        for i,ne in enumerate(self.N):
            e = np.tile(np.ones(self.p), (ne, 1))
            e *= (i+1)
            e[:, self.target] *= -1
            environments.append(e)
        self.environments = environments

    def test_basic(self):
        data = Data(self.environments, self.target)
        self.assertEqual(data.n, self.n)
        self.assertTrue((data.N == self.N).all())
        self.assertEqual(data.p, self.p)
        self.assertEqual(data.target, self.target)
        self.assertEqual(data.n_env, len(self.environments))

    def test_memory(self):
        environments = copy.deepcopy(self.environments)
        data = Data(environments, self.target)
        environments[0][0,0] = -100
        data_pooled = data.pooled_data()
        self.assertFalse(data_pooled[0,0] == environments[0][0,0])
        
    def test_targets(self):
        data = Data(self.environments, self.target)
        truth = [-(i+1)*np.ones(ne) for i,ne in enumerate(self.N)]
        truth_pooled = []
        for i,target in enumerate(data.targets):
            self.assertTrue((target == truth[i]).all())
            truth_pooled = np.hstack([truth_pooled, truth[i]])
        self.assertTrue((truth_pooled == data.pooled_targets()).all())

    def test_idx(self):
        data = Data(self.environments, self.target)
        truth = []
        for i,ne in enumerate(self.N):
            truth = np.hstack([truth, np.ones(ne)*i])
        self.assertTrue((truth == data.idx).all())
        
    def test_data(self):
        data = Data(self.environments, self.target)
        truth_pooled = []
        for i,ne in enumerate(self.N):
            sample = np.ones(self.p+1)
            sample[:-1] *= (i+1)
            sample[self.target] *= -1
            truth = np.tile(sample, (ne, 1))
            self.assertTrue((truth == data.data[i]).all())
            truth_pooled = truth if i==0 else np.vstack([truth_pooled, truth])
        self.assertTrue((truth_pooled == data.pooled_data()).all())

    def test_split(self):
        data = Data(self.environments, self.target)
        last = 0
        for i,ne in enumerate(self.N):
            (et, ed, rt, rd) = data.split(i)
            # Test that et and ed are correct
            self.assertTrue((et == data.targets[i]).all())
            self.assertTrue((ed == data.data[i]).all())
            # Same as two previous assertions but truth built differently
            truth_et = data.pooled_targets()[last:last+ne]
            truth_ed = data.pooled_data()[last:last+ne, :]
            self.assertTrue((truth_et == et).all())
            self.assertTrue((truth_ed == ed).all())
            # Test that rt and rd are correct
            idx = np.arange(self.n)
            idx = np.logical_or(idx < last, idx >= last+ne)
            truth_rt = data.pooled_targets()[idx]
            truth_rd = data.pooled_data()[idx, :]
            self.assertTrue((truth_rt == rt).all())
            self.assertTrue((truth_rd == rd).all())
            last += ne

class ResultTests(unittest.TestCase):

    def test_init(self):
        mses = [0.1, 0.2, 0.3]
        accepted = [{1}, {2}, {3}]
        rejected = [{4}, {5}, {6}]
        estimate = set()
        result = Result(estimate, accepted, rejected, mses)
        self.assertTrue(isinstance(result.mses, np.ndarray))
        self.assertTrue(np.allclose(result.mses, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(accepted, result.accepted)
        self.assertEqual(rejected, result.rejected)
        self.assertEqual(None, result.conf_intervals)
            
