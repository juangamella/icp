# Invariant Causal Prediction (ICP) Algorithm for Causal Discovery

This is a Python implementation of the Invariant Causal Prediction (ICP) algorithm from the 2016 paper [*"Causal inference using invariant prediction: identification and confidence intervals"*](https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/rssb.12167) by Jonas Peters, Peter Bühlmann and Nicolai Meinshausen.

At the point of writing, and to the best of my knowledge, the only other publicly available implementation of the algorithm is in the [R package](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html) written by the original authors.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install causalicp
```

The package is still at its infancy and its API is subject to change. However, this will be done with care: non backward-compatible changes to the API are reflected by a change to the minor or major version number,

> e.g. *code written using causalicp==0.1.2 will run with causalicp==0.1.3, but may not run with causalicp==0.2.0.*

The code has been written with an emphasis on readability and on
keeping the dependency footprint to a minimum; to this end, the only
dependencies outside the standard library are `numpy`, `scipy` and
`termcolor`.

## Documentation

You can find the complete documentation at https://sempler.readthedocs.io/en/latest/. For completeness, we include here an overview and an example.

### Running the algorithm: `causalicp.fit`

To run the algorithm, the function `fit` is provided:

```python
causalicp.fit(data, target, alpha=0.05, sets=None, precompute=True, verbose=False, color=False):
```

**Parameters**

- ***data*** (numpy.ndarray or list of array-like): The data from all
  experimental settings. Each element of the list/array is a
  2-dimensional array with a sample from a different setting, where
  columns correspond to variables and rows to observations
  (data-points). The data also contains the response variable, which
  is specified with the `target` parameter.
- ***target*** (int) The index of the response or target variable of
  interest.
- ***alpha*** (float, default=0.05 The level of the test procedure,
  taken from `[0,1]`. Defaults to `0.05`.
- ***sets*** (list of set or None, default=None): The sets for which ICP
  will test invariance. An error is raised if a set is not a subset of
  `{0,...,p-1}` or it contains the target, where `p` is the total
  number of variables (including the target). If `None` all possible
  subsets of predictors will be considered.
- ***precompute*** (bool, default=True): Wether to precompute the sample
  covariance matrix to speed up linear regression during the testing
  of each predictor set. For large sample sizes this drastically
  reduces the overall execution time, but it may result in numerical
  instabilities for highly correlated data. If set to `False`, for
  each set of predictors the regression is done using an iterative
  least-squares solver on the raw data.
- ***verbose*** (bool, default=False): If ICP should run in verbose
  mode, i.e. displaying information about completion and the result of
  tests.
- ***color*** (bool, default=True): If the output produced when
  `verbose=True` should be color encoded (not recommended if your
  terminal does not support ANSII color formatting), see
  [termcolor](https://pypi.org/project/termcolor/).

**Raises**

- ***ValueError***: If the value of some of the parameters is not
  appropriate, e.g. `alpha` is negative, `data` contains samples with
  different number of variables, or `sets` contains invalid sets.
- ***TypeError*** : If the type of some of the parameters was not expected (see examples below).

**Returns**

The result of the algorithm is returned in a `causalicp.Result` object, with the following attributes:

- ***p*** (int): The total number of variables in the data (including
    the response/target).
- ***target*** (int): The index of the
    response/target.
- ***estimate*** (set or None): The estimated parental set returned by
    ICP, or `None` if all sets of predictors were rejected.
- ***accepted_sets*** (list of set): A list containing the accepted sets
  of predictors.
- ***rejected_sets*** (list of set): 
    A list containing the rejected sets of predictors.
- ***pvalues*** (dict of (int, float)): A dictionary containing the
    p-value for the causal effect of each individual predictor. The
    target/response is included in the dictionary and has value `nan`.
- ***conf_intervals*** (numpy.ndarray or None): A `2 x p` array of
    floats representing the confidence interval for the causal effect
    of each variable. Each column corresponds to a variable, and the
    first and second row correspond to the lower and upper limit of
    the interval, respectively. The column corresponding to the
    target/response is set to `nan`.

### An example

We generate interventional data from a linear-Gaussian SCM using
[`sempler`](https://github.com/juangamella/sempler) (not a
dependency of `causalicp`).

```python
import sempler, sempler.generators
import numpy as np
np.random.seed(12)

# Generate a random graph and construct a linear-Gaussian SCM
W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 2)
scm = sempler.LGANM(W, (-1,1), (1,2))

# Generate a sample for setting 1: Observational setting
data = [scm.sample(n=100)]

# Setting 2: Shift-intervention on X1
data += [scm.sample(n=130, shift_interventions = {1: (3.1, 5.4)})]

# Setting 3: Do-intervention on X2
data += [scm.sample(n=98, do_interventions = {2: (-1, 3)})]
```

Running ICP for the response variable `3`, at a significance level of `0.05`.

```python
import causalicp as icp
result = icp.fit(data, 3, alpha=0.05, precompute=True, verbose=True, color=False)

# Output:

# Tested sets and their p-values:
#   set() rejected : 2.355990957880749e-10
#   {0} rejected : 7.698846116207467e-16
#   {1} rejected : 4.573866047163566e-09
#   {2} rejected : 8.374476052441259e-08
#   {0, 1} accepted : 0.7330408066181638
#   {0, 2} rejected : 2.062882130448634e-15
#   {1, 2} accepted : 0.8433000000649277
#   {0, 1, 2} accepted : 1
# Estimated parental set: {1}
```

The estimate, accepted sets, etc. are attributes of the `causalicp.Result` object:

```python
result.estimate
# {1}

result.accepted_sets
# [{0, 1}, {1, 2}, {0, 1, 2}]

result.rejected_sets
# [set(), {0}, {1}, {2}, {0, 2}]

result.pvalues
# {0: 0.8433000000649277, 1: 8.374476052441259e-08, 2: 0.7330408066181638, 3: nan}

result.conf_intervals
# array([[0.        , 0.57167295, 0.        ,        nan],
#        [2.11059461, 0.7865869 , 3.87380337,        nan]])
```

## Code structure

The code is divided in two modules:

- `icp.py` which contains the implementation of the algorithm (`fit`
  function) and the definition of the `Result` object.
- `data.py` which contains a class to manage the multi-environment
  data and perform the linear regression for each set in an efficient
  way.

## Tests

Unit tests and doctests are included. Additionally, the output of the
overall procedure has been checked against that of the `R` package by
the original authors,
[`InvariantCausalPrediction`](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html)
over tens of thousands of random graphs. Of course, this doesn't mean
there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. This will also execute the
doctests, generate `1000` random SCMs + interventions, and run the `R`
implementation on them for comparison. You can add
`SUITE=<module_name>` to run a particular module only. There are,
however, additional dependencies to run the tests. You can find these
in
[`requirements_tests.txt`](https://github.com/juangamella/icp/blob/master/requirements_tests.txt)
and
[`R_requirements_tests.txt`](https://github.com/juangamella/icp/blob/master/R_requirements_tests.txt).

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
