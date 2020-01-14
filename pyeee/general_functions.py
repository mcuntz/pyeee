#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Module with general functions, derivatives, etc. for Efficient/Sequential Elementary Effects.

The current functions are:
    curvature             Curvature of function f: f''/(1+f'^2)^3/2
    logistic              Logistic function L/(1+exp(-k(x-x0)))
    logistic_p            logistic(x,*p)
    dlogistic             First derivative of logistic function
    d2logistic            Second derivative of logistic function
    logistic_offset       logistic function with offset L/(1+exp(-k(x-x0))) + a
    logistic_offset_p     logistic_offset(x,*p)
    dlogistic_offset      First derivative of logistic function with offset
    d2logistic_offset     Second derivative of logistic function with offset
    logistic2_offset      Double logistic function with offset L1/(1+exp(-k1(x-x01))) - L2/(1+exp(-k2(x-x02))) + a2
    logistic2_offset_p    logistic2_offset(x,*p)
    dlogistic2_offset     First derivative of double logistic function with offset
    d2logistic2_offset    Second derivative of double logistic function with offset

This module was written by Matthias Cuntz while at Department of
Computational Hydrosystems, Helmholtz Centre for Environmental
Research - UFZ, Leipzig, Germany, and continued while at Institut
National de Recherche en Agriculture, Alimentation et Environnement
(INRAE), Nancy, France.

Copyright (c) 2015-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Mar 2015 by Matthias Cuntz (mc (at) macu (dot) de)
* Added functions logistic_p and logistic_offset_p, Dec 2017, Matthias Cuntz
* Changed to Sphinx docstring and numpydoc, Dec 2019, Matthias Cuntz
* Distinguish iterable and array_like parameter types, Jan 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following wrappers are provided

.. autosummary::
    curvature
    logistic
    logistic_p
    dlogistic
    d2logistic
    logistic_offset
    logistic_offset_p
    dlogistic_offset
    d2logistic_offset
    logistic2_offset
    logistic2_offset_p
    dlogistic2_offset
    d2logistic2_offset
"""
import numpy as np
import scipy.special as sp

__all__ = ['curvature',
           'logistic', 'logistic_p', 'dlogistic', 'd2logistic',
           'logistic_offset', 'logistic_offset_p', 'dlogistic_offset', 'd2logistic_offset',
           'logistic2_offset', 'logistic2_offset_p', 'dlogistic2_offset', 'd2logistic2_offset']

# -----------------------------------------------------------
# curvature of function
def curvature(x, dfunc, d2func, *args, **kwargs):
    """
    Curvature of function:

        `f''/(1+f'^2)^3/2`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute curvature
    dfunc : callable
        Function giving first derivative of function f: f', to be called `dfunc(x, *args, **kwargs)`
    d2func : callable
        Function giving second derivative of function f: f'', to be called `d2func(x, *args, **kwargs)`
    args : iterable
        Arguments passed to `dfunc` and `d2func`
    kwargs : dict
        Keyword arguments passed to `dfunc` and `d2func`

    Returns
    -------
    float or ndarray
        Curvature of function f at `x`
    """
    return ( d2func(x, *args, ** kwargs) /
                 (1. + dfunc(x, *args, **kwargs)**2)**1.5 )


# -----------------------------------------------------------
# a/(1+exp(-b(x-c))) - logistic function
def logistic(x, L, k, x0):
    """
    Logistic function:

        `L/(1+exp(-k(x-x0)))`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function

    Returns
    -------
    float or ndarray
        Logistic function at `x` with maximum `L`, steepness `k` and inflection point `x0`
    """
    return L * sp.expit(k * (x - x0))


def logistic_p(x, p):
    """
    Logistic function:

        `L/(1+exp(-k(x-x0)))`

    where parameters `L`, `k` and `x0` are given as one iterable, i.e.

        `p[0]/(1+exp(-p[1](x-p[2])))`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    p : iterable
        Iterable of length 3 with maximum, steepness and inflection point of logistic function

    Returns
    -------
    float or ndarray
        Logistic function at `x` with maximum `p[0]`, steepness `p[1]` and inflection point `p[2]`
    """
    return logistic(x, p[0], p[1], p[2])


# -----------------------------------------------------------
# 1st derivative of logistic functions
def dlogistic(x, L, k, x0):
    """
    First derivative of logistic function:

        `L/(1+exp(-k(x-x0)))`

    which is

        `k.L/(2(cosh(k(x-x0))+1))`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute derivative of logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function

    Returns
    -------
    float or ndarray
        First derivative of logistic function at `x` with maximum `L`, steepness `k` and inflection point `x0`
    """
    return k * L / (2. * (np.cosh(k * (x - x0)) + 1.))


# -----------------------------------------------------------
# 2nd derivative of logistic functions
def d2logistic(x, L, k, x0):
    """
    Second derivative of logistic function:

        `L/(1+exp(-k(x-x0)))`

    which is

        `-k^2.L.sinh(k(x-x0))/(2(cosh(k(x-x0))+1)^2)`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute derivative of logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function

    Returns
    -------
    float or ndarray
        Second derivative of logistic function at `x` with maximum `L`, steepness `k` and inflection point `x0`
    """
    return ( -k**2 * L * np.sinh(k * (x - x0)) /
                 (2. * (np.cosh(k * (x - x0)) + 1.)**2) )


# -----------------------------------------------------------
# L/(1+exp(-k(x-x0))) + a - logistic function with offset
def logistic_offset(x, L, k, x0, a):
    """
    Logistic function with offset:

        `L/(1+exp(-k(x-x0))) + a`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function
    a : float
        Offset of logistic function

    Returns
    -------
    float or ndarray
        Logistic function at `x` with maximum `L`, steepness `k`, inflection point `x0` and offset `a`
    """
    return L * sp.expit(k * (x - x0)) + a


def logistic_offset_p(x, p):
    """
    Logistic function with offset:

        `L/(1+exp(-k(x-x0))) + a`

    where parameters `L`, `k`, `x0`, and `a` are given as one iterable, i.e.

        `p[0]/(1+exp(-p[1](x-p[2]))) + p[3]`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    p : iterable
        Iterable of length 4 with maximum, steepness, inflection point, and offset of logistic function

    Returns
    -------
    float or ndarray
        Logistic function at `x` with maximum `p[0]`, steepness `p[1]`, inflection point `p[2]`, and offset `p[3]`
    """
    return logistic_offset(x, p[0], p[1], p[2], p[3])


# -----------------------------------------------------------
# 1st derivative of logistic functions with offset
def dlogistic_offset(x, L, k, x0, a):
    """
    First derivative of logistic function with offset:

        `L/(1+exp(-k(x-x0))) + a`

    which is

        `k.L/(2(cosh(k(x-x0))+1))`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute derivative of logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function
    a : float
        Offset of logistic function

    Returns
    -------
    float or ndarray
        First derivative of logistic function with offset at `x` with maximum `L`, steepness `k`,
        inflection point `x0`, and offset `a`
    """
    return k * L / (2. * (np.cosh(k * (x - x0)) + 1.))


# -----------------------------------------------------------
# 2nd derivative of logistic functions with offset
def d2logistic_offset(x, L, k, x0, a):
    """
    Second derivative of logistic function with offset

        `L/(1+exp(-k(x-x0))) + a`

    which is

        `-k^2.L.sinh(k(x-x0))/(2(cosh(k(x-x0))+1)^2)`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute derivative of logistic function
    L : float
        Maximum of logistic function
    k : float
        Steepness of logistic function
    x0 : float
        Inflection point of logistic function
    a : float
        Offset of logistic function

    Returns
    -------
    float or ndarray
        Second derivative of logistic function at `x` with maximum `L`, steepness `k`,
        inflection point `x0`, and offset `a`
    """
    return ( -k**2 * L * np.sinh(k * (x - x0)) /
                 (2. * (np.cosh(k * (x - x0)) + 1.)**2) )


# -----------------------------------------------------------
# L/(1+exp(-k(x-x0))) + a - logistic function with offset
def logistic2_offset(x, L1, k1, x01, L2, k2, x02, a):
    """
    Double logistic function with offset:

        `L1/(1+exp(-k1(x-x01))) - L2/(1+exp(-k2(x-x02))) + a`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    L1 : float
        Maximum of first logistic function
    k1 : float
        Steepness of first logistic function
    x01 : float
        Inflection point of first logistic function
    L2 : float
        Maximum of second logistic function
    k2 : float
        Steepness of second logistic function
    x02 : float
        Inflection point of second logistic function
    a : float
        Offset of double logistic function

    Returns
    -------
    float or ndarray
        Double Logistic function at `x`
    """
    return L1 * sp.expit(k1 * (x - x01)) - L2 * sp.expit(k2 * (x - x02)) + a


def logistic2_offset_p(x, p):
    """
    Double logistic function with offset:

        `L1/(1+exp(-k1(x-x01))) - L2/(1+exp(-k2(x-x02))) + a`

    where parameters `L1/2`, `k1/2` and `x01/2` are given as one iterable, i.e.

        `p[0]/(1+exp(-p[1](x-p[2]))) - p[3]/(1+exp(-p[4](x-p[5]))) + p[6]`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    p : iterable
        Iterable of length 6 with maximum, steepness, inflection point of first logistic function,
        maximum, steepness, inflection point of second logistic function, and offset of double logistic function

    Returns
    -------
    float or ndarray
        Double Logistic function with offset at `x`
    """
    return logistic2_offset(x, p[0], p[1], p[2], p[3], p[4], p[5], p[6])


# -----------------------------------------------------------
# 1st derivative of logistic functions with offset
def dlogistic2_offset(x, L1, k1, x01, L2, k2, x02, a):
    """
    First derivative of double logistic function with offset:

        `L1/(1+exp(-k1(x-x01))) - L2/(1+exp(-k2(x-x02))) + a`

    which is

        `k1.L1/(2(cosh(k1(x-x01))+1)) - k2.L2/(2(cosh(k2(x-x02))+1))`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    L1 : float
        Maximum of first logistic function
    k1 : float
        Steepness of first logistic function
    x01 : float
        Inflection point of first logistic function
    L2 : float
        Maximum of second logistic function
    k2 : float
        Steepness of second logistic function
    x02 : float
        Inflection point of second logistic function
    a : float
        Offset of double logistic function

    Returns
    -------
    float or ndarray
        First derivative of double logistic function with offset at `x`
    """
    return ( k1 * L1 / (2. * (np.cosh(k1 * (x - x01)) + 1.)) -
             k2 * L2 / (2. * (np.cosh(k2 * (x - x02)) + 1.)) )


# -----------------------------------------------------------
# 2nd derivative of logistic functions with offset
def d2logistic2_offset(x, L1, k1, x01, L2, k2, x02, a):
    """
    Second derivative of double logistic function with offset:

        `L1/(1+exp(-k1(x-x01))) - L2/(1+exp(-k2(x-x02))) + a`

    which is

        `-k1^2.L1.sinh(k1(x-x01))/(2(cosh(k1(x-x01))+1)^2) +k2^2.L2.sinh(k2(x-x02))/(2(cosh(k2(x-x02))+1)^2)`

    Parameters
    ----------
    x : array_like
        Independent variable to evalute logistic function
    L1 : float
        Maximum of first logistic function
    k1 : float
        Steepness of first logistic function
    x01 : float
        Inflection point of first logistic function
    L2 : float
        Maximum of second logistic function
    k2 : float
        Steepness of second logistic function
    x02 : float
        Inflection point of second logistic function
    a : float
        Offset of double logistic function

    Returns
    -------
    float or ndarray
        Second derivative of double logistic function with offset at `x`
    """
    return ( -k1**2 * L1 * np.sinh(k1 * (x - x01)) /
                 (2. * (np.cosh(k1 * (x - x01)) + 1.)**2) +
             k2**2 * L2 * np.sinh(k2 * (x - x02)) /
                 (2. * (np.cosh(k2 * (x - x02)) + 1.)**2) )


# -----------------------------------------------------------

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
