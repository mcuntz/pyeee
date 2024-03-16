#!/usr/bin/env python
"""
Functions used with Efficient/Sequential Elementary Effects

This module was written by Matthias Cuntz while at Department of
Computational Hydrosystems, Helmholtz Centre for Environmental
Research - UFZ, Leipzig, Germany, and continued while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2015- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

Functions:

.. autosummary::
   cost_square
   curvature

History
    * Written Mar 2015 by Matthias Cuntz (mc (at) macu (dot) de)
    * Changed to Sphinx docstring and numpydoc, Dec 2019, Matthias Cuntz
    * Split logistic and curvature into separate files,
      May 2020, Matthias Cuntz
    * More consistent docstrings, Jan 2022, Matthias Cuntz
    * Transferred curvature and cost_square from pyjams to pyeee,
      Mar 2024, Matthias Cuntz

"""
import numpy as np


__all__ = ['cost_square', 'curvature']


# -----------------------------------------------------------
# general cost function of sum of squared differences
def cost_square(p, func, x, y):
    """
    General cost function for optimising `func(x, p)` vs `y` with sum of
    square deviations.

    Parameters
    ----------
    p : iterable of floats
        parameters
    func : callable
        `fun(x,p) -> float`
    x : float or array_like of floats
        independent variable
    y : float or array_like of floats
        dependent variable, observations

    Returns
    -------
    float
        sum of squared deviations
    """
    return np.sum((y - func(x, p))**2)


# -----------------------------------------------------------
# curvature of a function
def curvature(x, dfunc, d2func, *args, **kwargs):
    """
    Curvature of a function f

    .. math::
       f''/(1+f'^2)^{3/2}

    Parameters
    ----------
    x : array_like
        Independent variable to evalute curvature
    dfunc : callable
        Function giving first derivative of function *f*: *f'*, to be called
        `dfunc(x, *args, **kwargs)`
    d2func : callable
        Function giving second derivative of function *f*: *f''*, to be called
        `d2func(x, *args, **kwargs)`
    args : iterable
        Arguments passed to *dfunc* and *d2func*
    kwargs : dict
        Keyword arguments passed to *dfunc* and *d2func*

    Returns
    -------
    float or ndarray
        Curvature of function *f* at *x*

    Examples
    --------
    .. code-block:: python

       from pyjams.functions import dlogistic_offset, d2logistic_offset
       curvature(1., dlogistic_offset, d2logistic_offset,
                 [1., 2., 2., 1.])

    """
    return ( d2func(x, *args, **kwargs) /
             (1. + dfunc(x, *args, **kwargs)**2)**1.5 )


# -----------------------------------------------------------

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
