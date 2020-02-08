#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Module with wrappers to partialise functions so that they can be called simply as func(x).

from functools import partial
if isinstance(func, (str,list)):
    obj = partial(exe_wrapper, func,
                  parameterfile, parameterwriter, objectivefile, objectivereader,
                  {'shell':bool, 'debug':bool, 'pid':bool, 'pargs':list, 'pkwargs':dict})
else:
    obj = partial(func_wrapper, func, arg, kwarg)
fx = obj(x)

func can be a Python function or executable filename.
If func is not a Python function, it will be passed to subprocess.check_output(func).
Func can then be a string (e.g. './prog -arg') if shell=True
or a list (e.g. ['./prog', '-arg']) if shell=False (default).
Programs without arguments, pipes, etc. can simply be strings with shell=True or False.

This module was written by Matthias Cuntz while at Institut National
de Recherche en Agriculture, Alimentation et Environnement (INRAE),
Nancy, France.

Copyright (c) 2016-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Nov 2016 by Matthias Cuntz (mc (at) macu (dot) de)
* Added x0 and mask to wrapper of external programs, Jan 2018, Matthias Cuntz
* Added that `pid` is passed to parameterwriter,
  and check parameterwriter (getargspec) for number or args, Feb 2018, Matthias Cuntz
* Removed check of number of parameters of parameterwriter (getargspec) but add
  separate wrappers for separate parmeterwriters with different number or arguments, Feb 2018, Matthias Cuntz
* Added `plotfile` and made docstring sphinx compatible option, Jan 2018, Matthias Cuntz
* Changed to Sphinx docstring and numpydoc, Nov 2019, Matthias Cuntz
* Remove that exe_wrappers support also Python functions. User should use func_wrappers, Nov 2019, Matthias Cuntz
* Make one exe_wrapper, passing bounds, mask, etc. via kwarg dictionary to parameterwriter; distinguish iterable and array_like parameter types, Jan 2020, Matthias Cuntz
* Replaced kwarg.pop mechanism because it removed the keywords from subsequent function calls, Feb 2020, Matthias Cuntz
* Change from ValueError to TypeError if function given to exe wrappers, Feb 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following wrappers are provided

.. autosummary::
   exe_wrapper
   exe_mask_wrapper
   func_wrapper
   func_mask_wrapper
"""
import subprocess
import os
import numpy as np

__all__ = ['exe_wrapper', 'exe_mask_wrapper',
           'func_wrapper', 'func_mask_wrapper']

def exe_wrapper(func,
                parameterfile, parameterwriter, objectivefile, objectivereader, kwarg,
                x):
    """
    Wrapper function for external programs using a `parameterwriter` with the interface:

        `parameterwriter(parameterfile, x, *pargs, **pkwargs)`

    or if `pid==True`:

        `parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)`

    Examples of `parameterwriter` with `pid==True` are: :any:`standard_parameter_writer` or
    :any:`sub_ja_params_files`.

    To be used with :any:`functools.partial`:

        `obj = partial(exe_wrapper, func,
        parameterfile, parameterwriter, objectivefile, objectivereader,
        {'shell':bool, 'debug':bool, 'pid':bool, 'pargs':list, 'pkwargs':dict})`

    This allows then calling obj with only the non-masked parameters:

        `fx = obj(x)`

    which translates to:

        `parameterwriter(parameterfile, x, *pargs, **pkwargs)`

        `err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)`

        `obj = objectivereader(objectivefile)`

    or if `pid==True` to:

        `parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)`

        `err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)`

        `obj = objectivereader(objectivefile+'.'+pid)`

    Parameters
    ----------
    func : string or list of strings
        External program to launch by :any:`subprocess`
    parameterfile : string
        Filename of parameter file
    parameterwriter : callable
        Python function writing the `parameterfile`, called as:

            `parameterwriter(parameterfile, x, *pargs, **pkwargs)`

        or if `pid==True` as:

            `parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)`

    objectivefile : string
        Filename of file with objective values written by external executable
    objectivereader : callable
        Python function for reading objective value from `objectivefile`
    kwarg : dict
        Dictionary with keyword arguments for exe_wrapper. Possible arguments are:

            **shell** (bool)

                If True, :any:`subprocess` opens shell for external executable

            **debug** (bool)

                If True, model output is displayed while executable is running

            **pid** (bool)

                If True, append '.RandomNumber' to `parameterfile` and `objectivefile` for parallel calls of `func`

            **pargs** (iterable)

                List of arguments of `parameterwriters`.

            **pkwargs** (dict)

                Dictionary with keyword arguments of `parameterwriters`.

    Returns
    -------
    float
        Objective value calculated by the external executable `func` or via the `objectivereader`


    History
    -------
    Written,  Matthias Cuntz, Mar 2018
    Modified, Matthias Cuntz, Dec 2019 - rm Python function -> use func_wrapper
                                       - Sphinx docstring
              Matthias Cuntz, Jan 2020 - renamed to exe_wrapper
                                       - shell, debug in kwarg dictionary
                                       - remove parameterfile if it exists
                                       - pid switch in kwarg dictionary
                                       - pargs, pkwargs for passing bounds, mask, etc. to parameterwriter
                                         replacing other exe_wrappers
                                       - distinguish iterable and array_like parameter types
              Matthias Cuntz, Feb 2020 - replaced kwarg.pop because ir removed keywords from subsequent calls
                                       - ValueError -> TypeError
    """
    shell   = kwarg['shell']   if 'shell'   in kwarg else False
    debug   = kwarg['debug']   if 'debug'   in kwarg else False
    pid     = kwarg['pid']     if 'pid'     in kwarg else False
    pargs   = kwarg['pargs']   if 'pargs'   in kwarg else []
    pkwargs = kwarg['pkwargs'] if 'pkwargs' in kwarg else {}
    # For multiprocess but not MPI: pid = mp.current_process()._identity[0]
    # seed uses global variables so all processes will produce same random numbers
    # use np.random.RandomState() for each processes for individual seeds in each process
    if pid:
        randst = np.random.RandomState()
        ipid   = str(randst.randint(2147483647))
    if isinstance(func, (str,list)):
        if pid:
            parameterwriter(parameterfile, ipid, x, *pargs, **pkwargs)
            if isinstance(func, str):
                func1 = func+' '+ipid
            else:
                func1 = func+[ipid]
            if debug:
                err = subprocess.check_call(func1, stderr=subprocess.STDOUT, shell=shell)
            else:
                err = subprocess.check_output(func1, stderr=subprocess.STDOUT, shell=shell)
            obj = objectivereader(objectivefile+'.'+ipid)
            if os.path.exists(parameterfile+'.'+ipid): os.remove(parameterfile+'.'+ipid)
            if os.path.exists(objectivefile+'.'+ipid): os.remove(objectivefile+'.'+ipid)
        else:
            parameterwriter(parameterfile, x, *pargs, **pkwargs)
            if debug:
                err = subprocess.check_call(func, stderr=subprocess.STDOUT, shell=shell)
            else:
                err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)
            obj = objectivereader(objectivefile)
            if os.path.exists(parameterfile): os.remove(parameterfile)
            if os.path.exists(objectivefile): os.remove(objectivefile)
        return obj
    else:
        raise TypeError('func must be string or list of strings for subprocess. Use func_wrapper for Python functions.')


def exe_mask_wrapper(func, x0, mask,
                     parameterfile, parameterwriter, objectivefile, objectivereader, kwarg,
                     x):
    """
    Wrapper function for external programs using a mask and a `parameterwriter` with the interface:

        `parameterwriter(parameterfile, x, *pargs, **pkwargs)`

    or if `pid==True`:

        `parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)`

    where `x` are the masked parameters (include mask).

    Examples of `parameterwriter` with `pid==True` are: :any:`standard_parameter_writer_bounds_mask` or
    :any:`sub_ja_params_files`.

    To be used with :any:`functools.partial`:

        `obj = partial(exe_mask_wrapper, func, x0, mask,
        parameterfile, parameterwriter, objectivefile, objectivereader,
        {'shell':bool, 'debug':bool, 'pid':bool, 'pargs':list, 'pkwargs':dict})`

    This allows then calling obj with only the non-masked parameters:

        `fx = obj(x)`

    which translates to:

        `xx = np.copy(x0)`

        `xx[mask] = x`

        `parameterwriter(parameterfile, xx, *pargs, **pkwargs)`

        `err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)`

        `obj = objectivereader(objectivefile)`

    or if `pid==True` to:

        `xx = np.copy(x0)`

        `xx[mask] = x`

        `parameterwriter(parameterfile, pid, xx, *pargs, **pkwargs)`

        `err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)`

        `obj = objectivereader(objectivefile+'.'+pid)`

    Parameters
    ----------
    func : string or list of strings
        External program to launch by :any:`subprocess`
    x0 : array_like
        Initial values of parameters and fixed values of masked parameters
    mask : array_like
        Mask to include (1) or exclude (0) parameter from parameterwriter
    parameterfile : string
        Filename of parameter file
    parameterwriter : callable
        Python function writing the `parameterfile`, called as:

            `parameterwriter(parameterfile, x, *pargs, **pkwargs)`

        or if `pid==True`:

            `parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)`

    objectivefile : string
        Filename of file with objective values written by external executable
    objectivereader : callable
        Python function for reading objective value from `objectivefile`
    kwarg : dict
        Dictionary with keyword arguments for exe_mask_wrapper. Possible arguments are:

            **shell** (bool)

                If True, :any:`subprocess` opens shell for external executable

            **debug** (bool)

                If True, model output is displayed while executable is running

            **pid** (bool)

                If True, append '.RandomNumber' to `parameterfile` and `objectivefile` for parallel calls of `func`

            **pargs** (iterable)

                List of arguments of `parameterwriters`.

            **pkwargs** (dict)

                Dictionary with keyword arguments of `parameterwriters`.

    Returns
    -------
    float
        Objective value calculated by the external executable `func` or via the `objectivereader`


    History
    -------
    Written,  Matthias Cuntz, Mar 2018
    Modified, Matthias Cuntz, Dec 2019 - rm Python function -> use func_mask_wrapper
                                       - Sphinx docstring
              Matthias Cuntz, Jan 2020 - renamed to exe_mask_wrapper
                                       - shell, debug in kwarg dictionary
                                       - remove parameterfile if it exists
                                       - pid switch in kwarg dictionary
                                       - pargs, pkwargs for passing bounds, mask, etc. to parameterwriter
                                         replacing other exe_wrappers
                                       - distinguish iterable and array_like parameter types
              Matthias Cuntz, Feb 2020 - replaced kwarg.pop because ir removed keywords from subsequent calls
                                       - ValueError -> TypeError
    """
    shell   = kwarg['shell']   if 'shell'   in kwarg else False
    debug   = kwarg['debug']   if 'debug'   in kwarg else False
    pid     = kwarg['pid']     if 'pid'     in kwarg else False
    pargs   = kwarg['pargs']   if 'pargs'   in kwarg else []
    pkwargs = kwarg['pkwargs'] if 'pkwargs' in kwarg else {}
    # For multiprocess but not MPI: pid = mp.current_process()._identity[0]
    # seed uses global variables so all processes will produce same random numbers
    # use np.random.RandomState() for each processes for individual seeds in each process
    if pid:
        randst = np.random.RandomState()
        ipid   = str(randst.randint(2147483647))
    xx = np.copy(x0)
    xx[mask] = x
    if isinstance(func, (str,list)):
        if pid:
            parameterwriter(parameterfile, ipid, xx, *pargs, **pkwargs)
            if isinstance(func, str):
                func1 = func+' '+ipid
            else:
                func1 = func+[ipid]
            if debug:
                err = subprocess.check_call(func1, stderr=subprocess.STDOUT, shell=shell)
            else:
                err = subprocess.check_output(func1, stderr=subprocess.STDOUT, shell=shell)
            obj = objectivereader(objectivefile+'.'+ipid)
            if os.path.exists(parameterfile+'.'+ipid): os.remove(parameterfile+'.'+ipid)
            if os.path.exists(objectivefile+'.'+ipid): os.remove(objectivefile+'.'+ipid)
        else:
            parameterwriter(parameterfile, xx, *pargs, **pkwargs)
            if debug:
                err = subprocess.check_call(func, stderr=subprocess.STDOUT, shell=shell)
            else:
                err = subprocess.check_output(func, stderr=subprocess.STDOUT, shell=shell)
            obj = objectivereader(objectivefile)
            if os.path.exists(parameterfile): os.remove(parameterfile)
            if os.path.exists(objectivefile): os.remove(objectivefile)
        return obj
    else:
        raise TypeError('func must be string or list of strings for subprocess. Use func_mask_wrapper for Python functions.')


# Python function wrappers
def func_wrapper(func, arg, kwarg, x):
    """
    Wrapper function for Python function.
    To be used with partial:

        `obj = partial(func_wrapper, func, arg, kwarg)`

    This allows then calling obj with only the non-masked parameters:

        `fx = obj(x)`

    which translates to:

        `fx = func(x, *arg, **kwarg)`

    Parameters
    ----------
    func : callable
        Python function to be called `func(x, *arg, **kwarg)`
    arg : iterable
        Arguments passed to `func`
    kwarg : dictionary
        Keyword arguments passed to `func`

    Returns
    -------
    float
        Objective value calculated by `func`


    History
    -------
    Written,  Matthias Cuntz, Nov 2016
    Modified, Matthias Cuntz, Nov 2019 - Sphinx docstring
    """
    return func(x, *arg, **kwarg)


def func_mask_wrapper(func, x0, mask, arg, kwarg, x):
    """
    Wrapper function for Python function using a mask.
    To be used with partial:

        `obj = partial(func_mask_wrapper, func, x0, mask, arg, kwarg)`

    This allows then calling obj with only the non-masked parameters:

        `fx = obj(x)`

    which translates to:

        `xx = np.copy(x0)`

        `xx[mask] = x`

        `fx = func(xx, *arg, **kwarg)`

    Parameters
    ----------
    func : callable
        Python function to be called `func(x, *arg, **kwarg)`
    x0 : array_like
        Fixed values of masked parameters
    mask : array_like
        Mask to use `x0` values ('mask[i]=1') or use new parameters `x` ('mask[i]=0') in call of function
    arg : iterable
        Arguments passed to `func`
    kwarg : dictionary
        Keyword arguments passed to `func`

    Returns
    -------
    float
        Objective value calculated by `func`


    History
    -------
    Written,  Matthias Cuntz, Nov 2016
    Modified, Matthias Cuntz, Nov 2019 - Sphinx docstring
              Matthias Cuntz, Jan 2020 - distinguish iterable and array_like parameter types
    """
    xx       = np.copy(x0)
    xx[mask] = x
    return func(xx, *arg, **kwarg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
