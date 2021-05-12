# Invariant Causal Prediction (ICP) Algorithm for Causal Discovery

This is a Python implementation of the Invariant Causal Prediction (ICP) algorithm from the 2016 paper [*"Causal inference using invariant prediction: identification and confidence intervals"*](https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/rssb.12167) by Jonas Peters, Peter Bühlmann and Nicolai Meinshausen.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install causalicp
```

## Documentation


## Running the algorithm

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
