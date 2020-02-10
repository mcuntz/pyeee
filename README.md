# pyeee - Parameter screening of computational models
<!-- pandoc -f gfm -o README.html -t html README.md -->

A Python library for parameter screening of computational models
using Morris' method of Elementary Effects or its extension of
Efficient/Sequential Elementary Effects by Cuntz, Mai et al. (Water
Res Research, 2015).

[![DOI](https://zenodo.org/badge/233405522.svg)](https://zenodo.org/badge/latestdoi/233405522)
[![PyPI version](https://badge.fury.io/py/pyeee.svg)](https://badge.fury.io/py/pyeee)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/mcuntz/pyeee/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/mcuntz/pyeee.svg?branch=master)](https://travis-ci.org/mcuntz/pyeee)
<!-- [![Build status](https://ci.appveyor.com/api/projects/status/bc57psfpa0676i4d/branch/master?svg=true)](https://ci.appveyor.com/project/mcuntz/pyeee) -->
[![Coverage Status](https://coveralls.io/repos/github/mcuntz/pyeee/badge.svg?branch=master)](https://coveralls.io/github/mcuntz/pyeee?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pyeee/badge/?version=latest)](https://pyeee.readthedocs.io/en/latest/?badge=latest)

## About pyeee

*pyeee* is a Python library for performing parameter screening of
computational models. It uses Morris' method of  Elementary Effects
and also its extension of Efficient or Sequential Elementary Effects published of

Cuntz, Mai *et al.* (2015)  
  Computationally inexpensive identification of noninformative model
parameters by sequential screening  
  *Water Resources Research* 51, 6417-6441,
 doi:[10.1002/2015WR016907](http://doi.org/10.1002/2015WR016907).

*pyeee* can be used with Python functions but wrappers are provided
to use it with executables as well. Function evaluation can be
distributed with Python's multiprocessing or via MPI.


## Documentation

The complete documentation for *pyeee* is available from Read The Docs.

   http://pyeee.readthedocs.org/en/latest/


## Quick usage guide

### Simple Python function

Consider the Ishigami-Homma function: $y = \sin(x_0) + a\,\sin(x_1)^2 + b\,x_2^4\sin(x_0)$.

Taking $a = b = 1$ gives:

```python
	import numpy as np
	def ishigami1(x):
	   return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])
```

The three paramters $x_0$, $x_1$, $x_2$ follow uniform distributions between $-\pi$ and $+\pi$.

Morris' Elementary Effects can then be calculated like:

```python
	npars = 3
	# lower boundaries
	lb = np.ones(npars) * (-np.pi)
	# upper boundaries
	ub = np.ones(npars) * np.pi

    # Elementary Effects
	from pyeee import ee
    np.random.seed(seed=1023) # for reproducibility of examples
	out = ee(ishigami1, lb, ub)
```

which gives the Elementary Effects ($\mu*$):

```python
    # mu*
    print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
    # gives: 212.4 0.6 102.8
```

Sequential Elementary Effects distinguish between informative and
uninformative parameters using several times Morris' Elementary Effects:

```python
	# screen
	from pyeee import eee
    np.random.seed(seed=1023) # for reproducibility of examples
	out = eee(ishigami1, lb, ub)
```

which returns a logical ndarray with True for the informative
parameters and False for the uninformative parameters:

```python
    print(out)
    [ True False  True]
```

### Python function with extra parameters

The function for pyeee must be of the form func(x). Use Python's
partial from the functools module to pass other function parameters.

For example pass the parameters $a$ and $b$ to the Ishigami-Homma function.

```python
	from functools import partial

	def ishigami(x, a, b):
	   return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

	def call_ishigami(ishi, a, b, x):
	   return ishi(x, a, b)

	# Partialise function with fixed parameters
	a = 0.5
	b = 2.0
	func   = partial(call_ishigami, ishigami, a, b)
	npars = 3

	# lower boundaries
	lb = np.ones(npars) * (-np.pi)
	# upper boundaries
	ub = np.ones(npars) * np.pi

    # Elementary Effects
	out = ee(func, lb, ub)
```

`partial` passes $a$ and $b$ to the
function `call_ishigami` already during definition so that *pyeee*
can then simply call it as `func(x)`, so that `x` is passed to
`call_ishigami` as well.


### Function wrappers

*pyeee* provides wrappers to use with partial.

```python
	from pyeee import func_wrapper
	args = [a, b]
	kwargs = {}
	func = partial(func_wrapper, ishigami, args, kwargs)

	# screen
	out = eee(func, lb, ub)
```

There are wrappers to use with Python functions with or without
masking parameters, as well as wrappers for external executables. See the
documentation for details.

   http://pyeee.readthedocs.org/en/latest/


## Installation

The easiest way to install is via `pip`::

    pip install pyeee

See the [installation instructions](http://pyeee.readthedocs.io/en/latest/install.html) in the
[documentation](http://pyeee.readthedocs.io) for more information.


## Requirements:

- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org/scipylib)
- [schwimmbad](https://github.com/adrn/schwimmbad)


## License

*pyeee* is distributed under the MIT License.  
See the [LICENSE](https://github.com/mcuntz/pyeee/LICENSE) file for details.

Copyright (c) 2012-2019 Matthias Cuntz, Juliane Mai

The project structure is based on a [template](https://github.com/MuellerSeb/template) provided by [Sebastian Müller](https://github.com/MuellerSeb).

## Contributing to pyeee

Users are welcome to submit bug reports, feature requests, and code
contributions to this project through GitHub.  
More information is available in the
[Contributing](http://pyeee.readthedocs.org/en/latest/contributing.html)
guidelines.
