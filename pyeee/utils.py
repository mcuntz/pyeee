#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
General utility functions.

This module was written by Matthias Cuntz while at Institut National
de Recherche en Agriculture, Alimentation et Environnement (INRAE),
Nancy, France.

Copyright (c) 2019-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Oct 2019 by Matthias Cuntz (mc (at) macu (dot) de)
* Make numpy doctsring format, Dec 2019, Matthias Cuntz
* Distinguish iterable and array_like parameter types, Jan 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following functions are provided

.. autosummary::
   cost_square
"""
import numpy as np


__all__ = ['cost_square']


# -----------------------------------------------------------
# general cost function
def cost_square(p, func, x, y):
    """
    General cost function for least square optimising `func(x,p)` vs `y`.

    Parameters
    ----------
    p : array_like
        Parameters of `func`.
    func : callable
        Python function callable as `func(x,p)`.
    x : array_like
        Independent func parameters.
    y : array_like
        Target values for `func(x,p)`.

    Returns
    -------
    B : float or ndarray
        Squared sum of `y` vs. `func(x,p)`: `sum((y-func(x,p))**2)`
    """
    return np.sum( (y - func(x,p))**2 )
