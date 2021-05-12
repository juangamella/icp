Welcome to causalicp's documentation!
=====================================

This is a Python implementation of the Invariant Causal Prediction
(ICP) algorithm from the 2016 `paper
<https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/rssb.12167>`__
*"Causal inference using invariant prediction: identification and
confidence intervals"* by Jonas Peters, Peter Bühlmann and Nicolai
Meinshausen.

Navigating this documentation
-----------------------------

To run the algorithm, see the function :meth:`causalicp.fit`. The results of the computation are reported through the :class:`causalicp.Result` class.


Installation
------------

You can clone this repo or install the python package via pip:

.. code:: bash

    pip install causalicp

The code has been written with an emphasis on readability and on keeping
the dependency footprint to a minimum; to this end, the only
dependencies outside the standard library are ``numpy``, ``scipy`` and
``termcolor``.


Versioning
----------
    
The package is still at its infancy and its API is subject to change.
However, this will be done with care: non backward-compatible changes to
the API are reflected by a change to the minor or major version number,

    e.g. *code written using causalicp==0.1.2 will run with
    causalicp==0.1.3, but may not run with causalicp==0.2.0.*

License
-------

The implementation is open-source and shared under a BSD 3-Clause License. You can find the source code in the `GitHub repository <https://github.com/juangamella/icp>`__.

Feedback
--------

Feedback is most welcome! You can add an issue in the `repository <https://github.com/juangamella/icp>`__ or send an `email <mailto:juan.gamella@stat.math.ethz.ch>`__.
    

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   fit
   result
