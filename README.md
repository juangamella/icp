# Invariant Causal Prediction (ICP) Algorithm for Causal Discovery

This is a Python implementation of the Invariant Causal Prediction (ICP) algorithm from the 2016 paper [*"Causal inference using invariant prediction: identification and confidence intervals"*](https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/rssb.12167) by Jonas Peters, Peter Bühlmann and Nicolai Meinshausen.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install causalicp
```

## Documentation


## Running the algorithm

To run the algorithm, the function `fit` is provided:

```python
causalicp.fit(data, target, alpha=0.05, sets=None, precompute=True, verbose=False, color=False):
```

**Parameters**

- **data** (numpy.ndarray or list of array-like): The data from all
  experimental settings. Each element of the list/array is a
  2-dimensional array with a sample from a different setting, where
  columns correspond to variables and rows to observations
  (data-points). The data also contains the response variable, which
  is specified with the `target` parameter.
- **target** (int) The index of the response or target variable of
  interest.
- **alpha** (float, default=0.05 The level of the test procedure,
  taken from `[0,1]`. Defaults to `0.05`.
- **sets** (list of set or None, default=None): The sets for which ICP
  will test invariance. An error is raised if a set is not a subset of
  `{0,...,p-1}` or it contains the target, where `p` is the total
  number of variables (including the target). If `None` all possible
  subsets of predictors will be considered.
- **precompute** (bool, default=True): Wether to precompute the sample
  covariance matrix to speed up linear regression during the testing
  of each predictor set. For large sample sizes this drastically
  reduces the overall execution time, but it may result in numerical
  instabilities for highly correlated data. If set to `False`, for
  each set of predictors the regression is done using an iterative
  least-squares solver on the raw data.
- **verbose** (bool, default=False): If ICP should run in verbose
  mode, i.e. displaying information about completion and the result of
  tests.
- **color** (bool, default=True): If the output produced when
  `verbose=True` should be color encoded (not recommended if your
  terminal does not support ANSII color formatting), see
  [termcolor](https://pypi.org/project/termcolor/).

**Raises**

- **ValueError**: If the value of some of the parameters is not
  appropriate, e.g. `alpha` is negative, `data` contains samples with
  different number of variables, or `sets` contains invalid sets.
- **TypeError** : If the type of some of the parameters was not expected (see examples below).

**Returns**

The result of the algorithm is returned in a `causalicp.Result` object, containing the result the estimate, accepted sets, p-values, etc.

Its attributes are:

**Example**

We generate interventional data from a linear-gaussian SCM using
[`sempler`](https://github.com/juangamella/sempler) (not a
dependency of causalicp).

```python
import sempler, sempler.generators
import numpy as np
np.random.seed(12)
W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 2)
scm = sempler.LGANM(W, (-1,1), (1,2))
data = [scm.sample(n=100)]
data += [scm.sample(n=130, shift_interventions = {1: (3.1, 5.4)})]
data += [scm.sample(n=98, do_interventions = {2: (-1, 3)})]
```

Running ICP for the response variable `0`, at a significance level of `0.01` (the default).

```python
import causalicp as icp
result = icp.fit(data, 3, alpha=0.05, precompute=True, verbose=True, color=False)
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
```

The estimate, accepted sets, etc are attributes of the `causalicp.Result` object:


>>> result.estimate
{1}

[{0, 1}, {1, 2}, {0, 1, 2}]

[set(), {0}, {1}, {2}, {0, 2}]

result.pvalues
{0: 0.8433000000649277, 1: 8.374476052441259e-08, 2: 0.7330408066181638, 3: nan}

result.conf_intervals
array([[0.        , 0.57167295, 0.        ,        nan],
       [2.11059461, 0.7865869 , 3.87380337,        nan]])
```
**Returns**

**Example**

## Code Structure

All the modules can be found inside the `ges/` directory. These include:

  - `ges.ges` which is the main module with the calls to start GES, and contains the implementation of the insert, delete and turn operators.
  - `ges.utils` contains auxiliary functions and the logic to transform a PDAG into a CPDAG, used after each application of an operator.
  - `ges.scores` contains the modules with the score classes:
      - `ges.scores.decomposable_score` contains the base class for decomposable score classes (see that module for more details).
      - `ges.scores.gauss_obs_l0_pen` contains an implementation of the cached Gaussian BIC score, as used in the original GES paper.
   - `ges.test` contains the modules with the unit tests and tests comparing against the algorithm's implementation in the 'pcalg' package.   

## Tests

All components come with unit tests to match, and some property-based tests. The output of the overall procedure has been checked against that of the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) implementation over tens of thousands of random graphs. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There are, however, additional dependencies to run the tests. You can find these in [`requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/requirements_tests.txt) and [`R_requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/R_requirements_tests.txt).

**Test modules**

They are in the sub package `ges.test`, in the directory `ges/test/`:

   - `test_decomposable_score.py`: tests for the decomposable score base class.
   - `test_gauss_bic.py`: tests for the Gaussian bic score.
   - `test_operators.py`: tests for the insert, delete and turn operators.
   - `test_pdag_to_cpdag.py`: tests the conversion from PDAG to CPDAG, which is applied after each application of an operator.
   - `test_utils.py`: tests the other auxiliary functions.
   - `ges.test.test_vs_pcalg`: compares the output of the algorithm vs. that of `pcalg` for randomly generated graphs.

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
