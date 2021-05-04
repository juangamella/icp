# Copyright 2020 Juan Luis Gamella Martin

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

import unittest
import numpy as np
import pandas as pd
import causalicp as icp


def process_accepted_sets(df):
    one_hot = df.to_numpy()[:, 1:]
    sets = []
    for s in one_hot:
        idx = tuple(np.where(s != 0)[0] + 1)
        sets.append(idx)
    return set(sets)


def process_confints(df):
    if len(df) == 0:
        return None
    else:
        intervals = df.to_numpy()[[1, 0], 1:]
        return np.hstack([np.array([[np.nan, np.nan]]).T, intervals])

# ---------------------------------------------------------------------
#


class TestsVsR(unittest.TestCase):

    def test_vs_r(self):
        path = 'causalicp/test/test_cases/'
        target = 0
        alpha = 0.001
        case_no = 0
        while True:
            # Read data
            try:
                data_path = path + 'case_%d.npy' % case_no
                data = np.load(data_path)
                # Read accepted sets from R's implementation of ICP
                accepted_sets_path = path + 'icp_result_%d_accepted.csv' % case_no
                true_accepted_sets = process_accepted_sets(pd.read_csv(accepted_sets_path))
                # Read confidence intervals from R's implementation of ICP
                confints_path = path + 'icp_result_%d_confints.csv' % case_no
                true_confints = process_confints(pd.read_csv(confints_path))
            except FileNotFoundError:
                print("No more cases remain")
                break
            # Test
            print("TEST CASE %d" % case_no)
            result = icp.fit(data, target, alpha, conf_ints=False, verbose=False)
            accepted_sets = set(tuple(s) for s in result.accepted)
            self.assertEqual(accepted_sets, true_accepted_sets)
            # print(result.conf_intervals)
            # print(true_confints)
            #self.assertTrue(np.allclose(result.conf_intervals, true_confints, equal_nan=True))
            case_no += 1
