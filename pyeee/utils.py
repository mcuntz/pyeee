#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
General utility functions.

This module was written by Matthias Cuntz while at Institut National de
Recherche Agronomique (INRA), Nancy, France.

Copyright (c) 2019 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Oct 2019 by Matthias Cuntz (mc (at) macu (dot) de)
* Make numpy doctsring format, Dec 2019, Matthias Cuntz

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
    p : float or iterable
        Parameters of `func`.
    func : callable
        Python function callable as `func(x,p)`.
    x : float or iterable
        Independent func parameters.
    y : float or iterable
        Target values for `func(x,p)`.

    Returns
    -------
    B : float or ndarray
        Squared sum of `y` vs. `func(x,p)`: `sum((y-func(x,p))**2)`
    """
    return np.sum( (y - func(x,p))**2 )
