# Copyright 2020 Juan L Gamella

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

# TODO
# - Random seed management
# - Decide how to divide results / run (probably each method-wise)

import pickle
import time
from datetime import datetime
import argparse
import os
import numpy as np
import sempler
import sempler.generators
import pandas as pd

# ---------------------------------------------------------------------------------------------------
# Test case generation + evaluation functions


def gen_scms(G, p, k, w_min, w_max, m_min, m_max, v_min, v_max):
    """
    Generate random experimental cases (ie. linear SEMs). Parameters:
      - n: total number of cases
      - p: number of variables in the SCMs
      - k: average node degree
      - w_min, w_max: Weights of the SCMs are sampled at uniform between w_min and w_max
      - v_min, v_max: Variances of the variables are sampled at uniform between v_min and v_max
      - m_min, m_max: Intercepts of the variables of the SCMs are sampled at uniform between m_min and m_max
      - random_state: to fix the random seed for reproducibility
    """
    cases = []
    while len(cases) < G:
        W = sempler.generators.dag_avg_deg(p, k, w_min, w_max)
        W *= np.random.choice([-1, 1], size=W.shape)
        scm = sempler.LGANM(W, (m_min, m_max), (v_min, v_max))
        cases.append(scm)
    return cases


def generate_interventions(scm, target, no_ints, max_int_size, m_min, m_max, v_min, v_max, include_obs=False):
    """Generate a set of interventions for a given scm, randomly sampling
    no_ints sets of targets of size int_size, and sampling the
    intervention means/variances uniformly from [m_min,m_max], [v_min, v_max].

    If include_obs is True, include an empty intervention to represent
    the reference or observational environment.
    """
    interventions = [None] if include_obs else []
    possible_targets = [i for i in range(scm.p) if i != target]
    # For each intervention
    for _ in range(no_ints):
        int_size = np.random.randint(0, max_int_size)
        if int_size == 0:
            interventions.append(None)
        else:
            # sample targets
            targets = np.random.choice(possible_targets, int_size, replace=False)
            # sample parameters
            means = np.random.uniform(m_min, m_max, len(targets)) if m_min != m_max else [
                m_min] * len(targets)
            variances = np.random.uniform(v_min, v_max, len(
                targets)) if v_min != v_max else [v_min] * len(targets)
            # assemble intervention
            intervention = dict((t, (mean, var))
                                for (t, mean, var) in zip(targets, means, variances))
            interventions.append(intervention)
    return interventions

def data_to_csv(data, path, debug):
    """Save the multienvironment data in a csv file ready to be used by the R script that runs ICP"""
    # Combine the data into a single matrix with an extra column indicating the environment
    flattened = []
    for e,X in enumerate(data):
        flagged = np.hstack([np.ones((len(X),1)) * e, X])
        flattened.append(flagged)
    flattened = np.vstack(flattened)
    # Save to .csv
    df = pd.DataFrame(flattened)
    filename = path + '.csv'
    print('  saved test case data to "%s"' % filename) if debug else None
    df.to_csv(filename, header=False, index=False)

def data_to_bin(data, path, debug):
    filename = path + '.npy'
    print('  saved test case data to "%s"' % filename) if debug else None
    np.save(path, data)
    
# --------------------------------------------------------------------
# Parse input parameters

# Definitions and default settings
arguments = {
    # Execution parameters
    'n_workers': {'default': 1, 'type': int},
    # 'batch_size': {'default': 20000, 'type': int},
    'runs': {'default': 1, 'type': int},
    'random_state': {'default': 42, 'type': int},
    'tag': {'type': str},
    'debug': {'default': False, 'type': bool},
    'chunksize': {'type': int},
    # SCM generation parameters
    'G': {'default': 10, 'type': int},
    'k': {'default': 2.7, 'type': float},
    'p': {'default': 5, 'type': int},
    'target': {'default': 0, 'type': int},
    'w_min': {'default': 0.5, 'type': float},
    'w_max': {'default': 1, 'type': float},
    'v_min': {'default': 1, 'type': float},
    'v_max': {'default': 2, 'type': float},
    'm_min': {'default': 0, 'type': float},
    'm_max': {'default': 1, 'type': float},
    # Intervention parameters
    'int_type': {'default': 'shift', 'type': str},
    'int_size': {'default': 3, 'type': int},
    'no_ints': {'default': 3, 'type': int},
    'int_m_min': {'default': 2, 'type': float},
    'int_m_max': {'default': 3, 'type': float},
    'int_v_min': {'default': 1, 'type': float},
    'int_v_max': {'default': 2, 'type': float},
    # Sampling parameters
    'n': {'default': 1000, 'type': int},
    'n_obs': {'type': int},
    # Score parameters
    'fit_means': {'default': False, 'type': bool},
}

# Parse settings from input
parser = argparse.ArgumentParser(description='Run experiments')
for name, params in arguments.items():
    if params['type'] == bool:
        options = {'action': 'store_true'}
    else:
        options = {'action': 'store', 'type': params['type']}
    if 'default' in params:
        options['default'] = params['default']
    parser.add_argument("--" + name,
                        dest=name,
                        required=False,
                        **options)

args = parser.parse_args()

# Parameters that will be excluded from the filename (see parameter_string function above)
excluded_keys = ['debug', 'n_workers', 'chunksize']  # , 'batch_size']
excluded_keys += ['tag'] if args.tag is None else []
excluded_keys += ['n_obs'] if args.n_obs is None else []

print(args)  # For debugging

# Set random seed
np.random.seed(args.random_state)

# --------------------------------------------------------------------
# Generate test cases
#   A test case is an SCM and a set of interventions

SCMs = gen_scms(args.G,
                args.p,
                args.k,
                args.w_min,
                args.w_max,
                args.m_min,
                args.m_max,
                args.v_min,
                args.v_max)

cases = []
for scm in SCMs:
    target = np.random.choice(range(scm.p))
    interventions = generate_interventions(scm,
                                           args.target,
                                           args.no_ints,
                                           args.int_size,
                                           args.int_m_min,
                                           args.int_m_max,
                                           args.int_v_min,
                                           args.int_v_max)
    cases.append((scm, target, interventions))

# --------------------------------------------------------------------
# Generate data and save it

print("\n\nSampling data for %d test cases %s\n\n" % (len(cases), datetime.now()))
start = time.time()

n_obs = args.n_obs if args.n_obs is not None else args.n

for i,(scm, target, interventions) in enumerate(cases):
    start_case = time.time()
    print('\nGenerating data for test case %d' % i)
    print('  p=%d, Interventions=%s' % (scm.p, interventions)) if args.debug else None
    # Sample interventional data
    XX = []
    for intervention in interventions:
        if intervention is None:
            X = scm.sample(n=n_obs)
        elif args.int_type == 'shift':
            X = scm.sample(n=args.n, shift_interventions=intervention)
        elif args.int_type == 'do':
            X = scm.sample(n=args.n, do_interventions=intervention)
        else:
            raise ValueError("Invalid intervention type: %s" % int_type)
        XX.append(X)
    # Save test case
    path = 'causalicp/test/test_cases/case_%d' % i
    data_to_bin(XX, path, args.debug)
    data_to_csv(XX, path, args.debug)
    with open(path + '.pickle', 'wb') as f:
        pickle.dump((scm, target, interventions, XX), f)
        print('  saved test case details to "%s"' % (path + '.pickle')) if args.debug else None
    print('Done (%0.2f seconds)' % (time.time() - start_case))

print(args)  # For debugging
end = time.time()
print("\n\nFinished at %s (elapsed %0.2f seconds)" % (datetime.now(), end - start))
