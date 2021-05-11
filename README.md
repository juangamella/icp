>>> import sempler, sempler.generators
    >>> import numpy as np
    >>> np.random.seed(12)
    >>> W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 2)
    >>> scm = sempler.LGANM(W, (-1,1), (1,2))
    >>> data = [scm.sample(n=100)]
    >>> data += [scm.sample(n=130, shift_interventions = {1: (3.1, 5.4)})]
    >>> data += [scm.sample(n=98, do_interventions = {2: (-1, 3)})]

    Run ICP for the response variable `0`, at a significance level of `0.05` (the default).

    >>> import causalicp as icp
    >>> result = icp.fit(data, 3, verbose=True)
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

    Obtain the estimate, accepted sets, etc

    >>> result.estimate
    {1}

    >>> result.accepted_sets
    [{0, 1}, {1, 2}, {0, 1, 2}]

    >>> result.rejected_sets
    [set(), {0}, {1}, {2}, {0, 2}]

    >>> result.pvalues
    {0: 0.8433000000649277, 1: 8.374476052441259e-08, 2: 0.7330408066181638, 3: nan}

    >>> result.conf_intervals
    array([[0.        , 0.57167295, 0.        ,        nan],
           [2.11059461, 0.7865869 , 3.87380337,        nan]])
