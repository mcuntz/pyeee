Quickstart
==========

``pyeee``: a Python library for parameter screening of computational
models using the extension of Morris' method of Elementary Effects
called Efficient or Sequential Elementary Effects by
`Cuntz, Mai et al.`_ (Water Res Research, 2015).

|DOI| |PyPI version| |Conda version| |License| |Build Status| |Coverage Status|


About pyeee
-----------

``pyeee`` is a Python library for performing parameter screening of
computational models. It uses the extension of Morris' method of
Elementary Effects of so-called Efficient or Sequential Elementary
Effects published by

Cuntz, Mai `et al.` (2015) Computationally inexpensive identification
of noninformative model parameters by sequential screening,
`Water Resources Research` 51, 6417-6441, doi: `10.1002/2015WR016907`_.

``pyeee`` can be used with Python functions but also with external
programs, using for example the library :mod:`partialwrap`. Function
evaluation can be distributed with Python's :mod:`multiprocessing`
module or via the Message Passing Interface (MPI) :mod:`mpi4py`.


Quick usage guide
-----------------

Simple Python function
^^^^^^^^^^^^^^^^^^^^^^

Consider the Ishigami-Homma function:
:math:`y = sin(x_0) + a * sin(x_1)^2 + b * x_2^4 * sin(x_0)`.

Taking :math:`a = b = 1` gives:

.. code:: python

   import numpy as np
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

The three paramters :math:`x_0`, :math:`x_1`, :math:`x_2` follow
uniform distributions between :math:`-\pi` and :math:`+\pi`.

Morris' Elementary Effects can then be calculated using, for example,
the Python library :mod:`pyjams`, giving the Elementary Effects
(:math:`\mu*`):

.. code:: python

   from pyjams import ee

   npars = 3
   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi
   # Elementary Effects
   np.random.seed(seed=1023)  # for reproducibility of examples
   out = ee(ishigami1, lb, ub, 10)   # mu*
   print("{:.1f} {:.1f} {:.1f}".format(*out[:, 0]))
   # gives: 173.1 0.6 61.7

Sequential Elementary Effects distinguish between informative and
uninformative parameters using several times Morris' Elementary
Effects, returning a logical ndarray with True for the informative
parameters and False for the uninformative parameters:

.. code:: python

   from pyeee import eee

   # screen
   np.random.seed(seed=1023)  # for reproducibility of examples
   out = eee(ishigami1, lb, ub, ntfirst=10)
   print(out)
   [ True False  True]


Python function with extra parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function for ``pyeee`` must be of the form :math:`func(x)`. Use
Python's :py:func:`~functools.partial` from the :py:mod:`functools`
module to pass other function parameters. For example pass the
parameters :math:`a` and :math:`b` to the Ishigami-Homma function.

.. code:: python

   import numpy as np
   from pyeee import eee
   from functools import partial

   def ishigami(x, a, b):
      return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

   def call_ishigami(func, a, b, x):
      return func(x, a, b)

   # Partialise function with fixed parameters
   a = 0.5
   b = 2.0
   func  = partial(call_ishigami, ishigami, a, b)

   npars = 3
   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi
   # Elementary Effects
   np.random.seed(seed=1023)  # for reproducibility of examples
   out = eee(func, lb, ub, ntfirst=10)

Figuratively speaking, :py:func:`~functools.partial` passes `a` and
`b` to the function `call_ishigami` already during definition so that
``eee`` can then simply call it as `func(x)`, where `x` is passed to
`call_ishigami` then as well.


Function wrappers
^^^^^^^^^^^^^^^^^

We recommend to use our package :mod:`partialwrap` for external
executables, which allows easy use of external programs and also their
parallel execution. See the `User Guide <userguide.html>`_ for
details. A trivial example is the use of :mod:`partialwrap` for the
above function wrapping:

.. code:: python

   from partialwrap import function_wrapper
   
   args = [a, b]
   kwargs = {}
   func = partial(func_wrapper, ishigami, args, kwargs)
   # screen
   out = eee(func, lb, ub, ntfirst=10)


Installation
------------

The easiest way to install is via `pip`:

.. code-block:: bash

   pip install pyeee

or via `conda`:

.. code-block:: bash

   conda install -c conda-forge pyeee


Requirements
------------

- :mod:`numpy`
- :mod:`scipy`
- :mod:`schwimmbad`
- :mod:`pyjams`


License
-------

``pyeee`` is distributed under the MIT License. See the
`LICENSE`_ file for details.

Copyright (c) 2019-2024 Matthias Cuntz, Juliane Mai

The project structure is based on a `template`_ provided by `Sebastian Müller`_.


Index and Tables
----------------

* :ref:`genindex`
* :ref:`modindex`


.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3620909.svg
   :target: https://doi.org/10.5281/zenodo.3620909
.. |PyPI version| image:: https://badge.fury.io/py/pyeee.svg
   :target: https://badge.fury.io/py/pyeee
.. |Conda version| image:: https://img.shields.io/conda/vn/conda-forge/pyeee.svg
   :target: https://anaconda.org/conda-forge/pyeee
.. |License| image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/mcuntz/pyeee/blob/master/LICENSE
.. |Build Status| image:: https://github.com/mcuntz/pyeee/workflows/Continuous%20Integration/badge.svg?branch=master
   :target: https://github.com/mcuntz/pyeee/actions
.. |Coverage Status| image:: https://coveralls.io/repos/github/mcuntz/pyeee/badge.svg?branch=master
   :target: https://coveralls.io/github/mcuntz/pyeee?branch=master

.. _10.1002/2015WR016907: http://doi.org/10.1002/2015WR016907
.. _Cuntz, Mai et al.: http://doi.org/10.1002/2015WR016907
.. _LICENSE: https://github.com/mcuntz/pyeee/LICENSE
.. _Sebastian Müller: https://github.com/MuellerSeb
.. _template: https://github.com/MuellerSeb/template
