=======================
pyeee Quickstart
=======================

``pyeee``: A Python library for parameter screening of computational models
using Morris' method of Elementary Effects or its extension of
Efficient or Sequential Elementary Effects by Cuntz, Mai et al. (Water Res Research, 2015).

.. toctree::
   :maxdepth: 3
   :caption: Contents:


About
============

``pyeee`` is a Python library for performing parameter screening of
computational models. It uses Morris' method of  Elementary Effects
and also its extension of Efficient or Sequential Elementary Effects published by:

Cuntz M, Mai J *et al.* (2015)  
Computationally inexpensive identification of noninformative model
parameters by sequential screening  
*Water Resources Research* 51, 6417-6441,
doi:`10.1002/2015WR016907 <http://doi.org/10.1002/2015WR016907>`_.

``pyeee`` can be used with Python functions but wrappers are provided
to use it with external executables as well. Function evaluation can be
distributed with Python's multiprocessing or via MPI.

The complete documentation for ``pyeee`` is available from Read The Docs.

   http://pyeee.readthedocs.org/en/latest/

   
Quick usage guide
=================

Simple Python function
----------------------

Consider the Ishigami-Homma function: :math:`y = \sin(x_0) + a\,\sin(x_1)^2 + b\,x_2^4\sin(x_0)`.

Taking :math:`a = b = 1` gives:

.. code-block:: python

    import numpy as np
    def ishigami1(x):
        return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

The three parameters :math:`x_0, x_1, x_2` follow uniform
distributions between :math:`-\pi` and :math:`+\pi`.

Morris' Elementary Effects can then be calculated like:

.. code-block:: python

    npars = 3
    # lower boundaries
    lb = np.ones(npars) * (-np.pi)
    # upper boundaries
    ub = np.ones(npars) * np.pi

    # Elementary Effects
    from pyeee import ee
    np.random.seed(seed=1023) # for reproducibility of examples
    out = ee(ishigami1, lb, ub)

which gives the Elementary Effects (:math:`\mu*`):

.. code-block:: python

    # mu*
    print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
    # gives: 212.4 0.6 102.8

Sequential Elementary Effects distinguish between informative and
uninformative parameters using several times Morris' Elementary Effects:

.. code-block:: python

    # screen
    from pyeee import eee
    np.random.seed(seed=1021) # for reproducibility of examples
    out = eee(ishigami1, lb, ub)

which returns a logical ndarray with True for the informative
parameters and False for the uninformative parameters:

.. code-block:: python

    print(out)
    # gives: [ True False  True]

    
Python function with extra parameters
-------------------------------------

The function for ``pyeee`` must be of the form `func(x)`. Use Python's
:any:`functools.partial` from the :mod:`functools` module to pass other function parameters.

For example pass the parameters :math:`a` and :math:`b` to the Ishigami-Homma function:

.. code-block:: python

    from functools import partial

    def ishigami(x, a, b):
        return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

    def call_ishigami(ishi, a, b, x):
        return ishi(x, a, b)

    # Partialise function with fixed parameters a and b
    a = 0.5
    b = 2.0
    func  = partial(call_ishigami, ishigami, a, b)
    npars = 3

    # lower boundaries
    lb = np.ones(npars) * (-np.pi)
    # upper boundaries
    ub = np.ones(npars) * np.pi

    # Elementary Effects
    np.random.seed(seed=1021) # for reproducibility of examples
    out = ee(func, lb, ub)

`partial` passes :math:`a` and :math:`b` to the
function `call_ishigami` already during definition so that ``pyeee``
can then simply call it as `func(x)`, so that `x` is passed to
`call_ishigami` as well.


Function wrappers
-----------------

``pyeee`` provides wrappers to use with partial.

.. code-block:: python

    from pyeee import func_wrapper
    args = [a, b]
    kwargs = {}
    func = partial(func_wrapper, ishigami, args, kwargs)

    # screen
    np.random.seed(seed=1021) # for reproducibility of examples
    out = eee(func, lb, ub)

There are wrappers to use with Python functions with or without
masking parameters, as well as wrappers for external executables.


Installation
============

The easiest way to install is via `pip`:

.. code-block:: bash

    pip install pyeee

See the `installation instructions <install.html>`_ for more information.


License
=======

``pyeee`` is distributed under the MIT License.  
See the `LICENSE <https://github.com/mcuntz/pyeee/LICENSE>`_ file for details.

Copyright (c) 2014-2019 Matthias Cuntz, Juliane Mai

The project structure is based on a [template](https://github.com/MuellerSeb/template) provided by [Sebastian Müller](https://github.com/MuellerSeb).


Contributing to pyeee
=====================

Users are welcome to submit bug reports, feature requests, and code
contributions to this project through GitHub.

More information is available in the
`Contributing <contributing.html>`_
guidelines.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
