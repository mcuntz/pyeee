#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
screening : Provides the function screening/ee for Morris' method of Elementary Effects.

This function was written by Matthias Cuntz while at Institut National
de Recherche en Agriculture, Alimentation et Environnement (INRAE),
Nancy, France.

Copyright (c) 2017-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Dec 2017 by Matthias Cuntz (mc (at) macu (dot) de)
* Output also (npara,3)-array for nt=1, Dec 2017, Matthias Cuntz
* Removed call to external programs: use exe_wrappers, Jan 2018, Matthias Cuntz
* Function can return multiple outputs, e.g. time series, Jan 2018, Matthias Cuntz
* Python 3: map returns iterator, so list(map), Jul 2018, Fabio Gennaretti
* Bug: default ntotal was not set if ntotal<0 (but nt instead), Dec 2019, Matthias Cuntz
* Make numpy docstring format, Dec 2019, Matthias Cuntz
* x0 optional, Jan 2020, Matthias Cuntz
* Distinguish iterable and array_like parameter types; added seed keyword to screening/ee, Jan 2020, Matthias Cuntz
* InputError does not exist, use TypeError, Feb 2020, Matthias Cuntz
* Use new names of kwargs of morris_sampling and elementary_effects, Feb 2020, Matthias Cuntz
* Make number of final trajectories an argument instead of a keyword argument, Feb 2020, Matthias Cuntz
* Sample not only from uniform distribution but allow all distributions of scipy.stats, Mar 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following functions are provided

.. autosummary::
   ee
   screening
"""
import numpy as np
from pyeee import morris_sampling, elementary_effects


__all__ = ['screening', 'ee']


def screening(func, lb, ub, nt, x0=None, mask=None,
              nsteps=6, ntotal=None,
              dist=None, distparam=None,
              seed=None,
              processes=1, pool=None,
              verbose=0):
    """
    Parameter screening using Morris' method of Elementary Effects.

    Note, the input function must be callable as `func(x)`.

    Parameters
    ----------
    func : callable
        Python function callable as `func(x)` with `x` the function parameters.
    lb : array_like
        Lower bounds of parameters
        or lower fraction of percent point function ppf if distribution given.

        Be aware that the percent point function ppf of most continuous distributions
        is infinite at 0.
    ub : array_like
        Upper bounds of parameters
        or upper fraction of percent point function ppf if distribution given.

        Be aware that the percent point function ppf of most continuous distributions
        is infinite at 1.
    nt : int
        Number of trajectories used for screening.
    x0 : array_like, optional
        Parameter values used with `mask==0`.
    mask : array_like, optional
        Include (1,True) or exclude (0,False) parameters in screening (default: include all parameters).
    nsteps : int, optional
        Number of steps along one trajectory (default: 6)
    ntotal : int, optional
        Total number of sampled trajectories to select the nt most different trajectories.
        If None: `max(nt**2,10*nt)` (default: None)
    dist : list, optional
        List of None or scipy.stats distribution objects for each factor
        having the method ppf, Percent Point Function (Inverse of CDF) (default: None)

        If None, the uniform distribution will be sampled from lower bound LB to upper bound UB.

        If dist is scipy.stats.uniform, the ppf will be sampled from the lower
        fraction given in LB and the upper fraction in UB. The sampling interval
        is then given by the parameters loc=lower and scale=interval=upper-lower
        in distparam. This means
        dist=None, LB=a, UB=b
        corresponds to
        LB=0, UB=1, dist=scipy.stats.uniform, distparam=[a,b-a]
    distparam : list, optional
        List with tuples with parameters as required for dist (default: (0,1)).

        All distributions of scipy.stats have location and scale parameters, at least.
        loc and scale are implemented as keyword arguments in scipy.stats. Other parameters
        such as the shape parameter of the gamma distribution must hence be given first,
        e.g. (shape,loc,scale) for the gamma distribution.

        distparam is ignored if dist is None.

        The percent point function ppf is called like this: dist(*distparam).ppf(x)
    seed : int or array_like
        Seed for numpy's random number generator (default: None).
    processes : int, optional
        The number of processes to use to evaluate objective function and constraints (default: 1).
    pool : `schwimmbad` pool object, optional
        Generic map function used from module `schwimmbad <https://schwimmbad.readthedocs.io/en/latest/>`_,
        which provides, serial, multiprocessor, and MPI mapping functions (default: None).

        The pool is chosen with:

            schwimmbad.choose_pool(mpi=True/False, processes=processes).

        The pool will be chosen automatically if pool is None.

        MPI pools can only be opened and closed once. If you want to use screening several
        times in one program, then you have to choose the pool, pass it to screening,
        and later close the pool in the calling progam.

    verbose : int, optional
        Print progress report during execution if verbose>0 (default: 0).

    Returns
    -------
    (nparameter,3) ndarray
            if nt>1:

                2D-array - (nparameter,3) with per parameter

                           1. mean of absolute elementary effects over all nt trajectories (mu*)
                           2. mean of elementary effects over all nt trajectories (mu)
                           3. standard deviation of elementary effects over all nt trajectories (sigma)

            else:

                2D-array - (nparameter,3) with per parameter

                           1. absolute elementary effect of each parameter
                           2. elementary effect of each parameter
                           3. zeros

    See Also
    --------
    :func:`~pyeee.eee.eee` : Efficient Elementary Effects, same as

    :func:`~pyeee.eee.see` : Sequential Elementary Effects

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> from pyeee.utils import func_wrapper
    >>> from pyeee.functions import fmorris
    >>> seed = 1023
    >>> np.random.seed(seed=seed)
    >>> npars = 20
    >>> beta0              = 0.
    >>> beta1              = np.random.standard_normal(npars)
    >>> beta1[:10]         = 20.
    >>> beta2              = np.random.standard_normal((npars,npars))
    >>> beta2[:6,:6]       = -15.
    >>> beta3              = np.zeros((npars,npars,npars))
    >>> beta3[:5,:5,:5]    = -10.
    >>> beta4              = np.zeros((npars,npars,npars,npars))
    >>> beta4[:4,:4,:4,:4] = 5.
    >>> arg   = [beta0, beta1, beta2, beta3, beta4]
    >>> kwarg = {}
    >>> func  = partial(func_wrapper, fmorris, arg, kwarg)
    >>> lb    = np.zeros(npars)
    >>> ub    = np.ones(npars)
    >>> nt      = 20
    >>> ntotal  = 10*nt
    >>> nsteps  = 6
    >>> verbose = 0
    >>> out = screening(func, lb, ub, nt, x0=None, mask=None, nsteps=nsteps, ntotal=ntotal, processes=4, verbose=verbose)
    >>> print(out[0:3,0])
    [60.7012889  67.33372626 48.46673528]


    History
    -------
    Written,  Matthias Cuntz,   Dec 2017
    Modified, Matthias Cuntz,   Dec 2017 - output for nt=1 also (npara,3)
              Matthias Cuntz,   Jan 2018 - rm call of external programs
              Matthias Cuntz,   Jan 2018 - function can return multiple output, e.g. time series
              Fabio Gennaretti, Jul 2018 - map of Python3 returns iterator -> make list(map())
              Matthias Cuntz,   Dec 2019 - bug: default ntotal was not set if ntotal<0 (but nt instead)
                                         - numpy docstring format
              Matthias Cuntz,   Jan 2020 - x0 optional
                                         - distinguish iterable and array_like parameter types
                                         - added seed
              Matthias Cuntz,   Feb 2020 - InputError -> TypeError
                                         - use new names of kwargs of moris_sampling
              Matthias Cuntz,   Feb 2020 - number of final trajectories is argument, not keyword argument
              Matthias Cuntz,   Mar 2020 - Sample from distributions: dist, distparam keywords
    """
    # Get MPI communicator
    try:
        from mpi4py import MPI
        comm  = MPI.COMM_WORLD
        csize = comm.Get_size()
        crank = comm.Get_rank()
    except ImportError:
        comm  = None
        csize = 1
        crank = 0

    # Checks
    # Bounds
    assert len(lb)==len(ub), 'Lower- and upper-bounds must have the same lengths.'
    lb = np.array(lb)
    ub = np.array(ub)
    # Mask
    if mask is None:
        assert np.all(ub >= lb), 'All upper-bounds must be greater or equal than lower-bounds.'
    else:
        assert len(mask)==len(ub), 'Mask and bounds must have the same lengths.'
        if x0 is None:
            raise TypeError('x0 must be given if mask is set')
        x0 = np.array(x0)
        if not np.all(mask): assert len(mask)==len(x0), 'Mask and x0 must have the same lengths.'
        assert np.all(ub[mask] >= lb[mask]), 'All unmasked upper-bounds must be greater or equal than lower-bounds.'

    # Set defaults
    npara = len(lb)
    if mask is None:
        imask  = np.ones(npara, dtype=np.bool)
    else:
        imask  = mask
    nmask = np.sum(imask)
    if ntotal is None:
        ntotal = max(nt**2, 10*nt)

    # Seed random number generator
    if seed is not None: np.random.seed(seed=seed)  # same on all ranks because trajectories are sampled on all ranks

    # Sample trajectories
    if (crank==0) and (verbose > 0): print('Sample trajectories')
    if dist is None:
        idist      = None
        idistparam = None
    else:
        idist      = [ dist[i]      for i in np.where(imask)[0] ]
        idistparam = [ distparam[i] for i in np.where(imask)[0] ]
    tmatrix, tvec = morris_sampling(nmask, lb[imask], ub[imask], nt,
                                    nsteps=nsteps, ntotal=ntotal,
                                    dist=idist, distparam=idistparam,
                                    Diagnostic=False)

    if mask is None:
        x = tmatrix
    else:
        # Set input vector to trajectories and masked elements = x0
        x = np.tile(x0,tvec.size).reshape(tvec.size,npara) # default to x0
        x[:,imask] = tmatrix                               # replaced unmasked with trajectorie values

    # Choose the right mapping function: single, multiprocessor or mpi
    if pool is None:
        import schwimmbad
        ipool = schwimmbad.choose_pool(mpi=False if csize==1 else True, processes=processes)
    else:
        ipool = pool

    # Calculate all model runs
    if (crank==0) and (verbose > 0): print('Calculate objective functions')
    fx = np.array(list(ipool.map(func, x)))

    # Calc elementary effects
    if (crank==0) and (verbose > 0): print('Calculate elementary effects')
    if fx.ndim == 1:
        fx  = fx[:,np.newaxis]
        nfx = 1
    else:
        nfx = fx.shape[1]
    out = np.zeros((nfx,npara,3))
    for j in range(nfx):
        sa, res = elementary_effects(nmask, tmatrix, tvec, fx[:,j], nsteps=nsteps, Diagnostic=False)
        # Output with zero for all masked parameters
        if nt == 1:
            out[j,imask,0] = np.abs(sa[:,0])
            out[j,imask,1] = sa[:,0]
        else:
            out[j,imask,:] = res

    if nfx == 1: out = out[0,:,:]

    if pool is None: ipool.close()

    return out


def ee(*args, **kwargs):
    """
    Wrapper function for :func:`~pyeee.screening.screening`.
    """
    return screening(*args, **kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    # import schwimmbad
    # import jams

    # import time as ptime
    # t1 = ptime.time()

    # try:
    #     from mpi4py import MPI
    #     comm  = MPI.COMM_WORLD
    #     csize = comm.Get_size()
    #     crank = comm.Get_rank()
    # except ImportError:
    #     comm  = None
    #     csize = 1
    #     crank = 0


    # seed = 1025

    # if seed is not None:
    #     np.random.seed(seed=seed)

    # if False:
    #     # work with global parameters
    #     def morris(X):
    #         return jams.functions.morris(X, beta0, beta1, beta2, beta3, beta4)
    #     func = morris
    #     npars = 20
    #     beta0              = 0.
    #     beta1              = np.random.standard_normal(npars)
    #     beta1[:10]         = 20.
    #     beta2              = np.random.standard_normal((npars,npars))
    #     beta2[:6,:6]       = -15.
    #     beta3              = np.zeros((npars,npars,npars))
    #     beta3[:5,:5,:5]    = -10.
    #     beta4              = np.zeros((npars,npars,npars,npars))
    #     beta4[:4,:4,:4,:4] = 5.
    #     lb    = np.zeros(npars)
    #     ub    = np.ones(npars)
    #     nt      = 20
    #     ntotal  = 10*nt
    #     nsteps  = 6
    #     verbose = 1

    #     out = screening(func, lb, ub, nt, x0=None, mask=None, nsteps=nsteps, ntotal=ntotal, processes=4, verbose=1)

    #     t2    = ptime.time()

    #     if crank == 0:
    #         strin = '[m]: {:.1f}'.format((t2-t1)/60.) if (t2-t1)>60. else '[s]: {:d}'.format(int(t2-t1))
    #         print('Time elapsed: ', strin)
    #         print('mu*, mu, std: ', out)
    #         out1 = out / out.max(axis=0)
    #         mustar = out1[:,0]
    #         mustar = np.sort(mustar)
    #         import matplotlib.pyplot as plt
    #         plt.plot(mustar, 'ro')
    #         plt.show()

    # if False:
    #     # pass parameters
    #     func = jams.functions.morris
    #     npars = 20
    #     lb    = np.zeros(npars)
    #     ub    = np.ones(npars)
    #     beta0              = 0.
    #     beta1              = np.random.standard_normal(npars)
    #     beta1[:10]         = 20.
    #     beta2              = np.random.standard_normal((npars,npars))
    #     beta2[:6,:6]       = -15.
    #     beta3              = np.zeros((npars,npars,npars))
    #     beta3[:5,:5,:5]    = -10.
    #     beta4              = np.zeros((npars,npars,npars,npars))
    #     beta4[:4,:4,:4,:4] = 5.
    #     args = [beta0, beta1, beta2, beta3, beta4] # Morris
    #     nt = 20
    #     ntotal  = 10*nt
    #     nsteps = 6
    #     verbose = 1

    #     out = screening(func, lb, ub, nt, arg=args, mask=None, nsteps=nsteps, ntotal=ntotal, processes=4, verbose=1)
    #     t2    = ptime.time()

    #     if crank == 0:
    #         strin = '[m]: {:.1f}'.format((t2-t1)/60.) if (t2-t1)>60. else '[s]: {:d}'.format(int(t2-t1))
    #         print('Time elapsed: ', strin)
    #         print('mu*, mu, std: ', out)
    #         out1 = out / out.max(axis=0)
    #         mustar = out1[:,0]
    #         mustar = np.sort(mustar)
    #         import matplotlib.pyplot as plt
    #         plt.plot(mustar, 'ro')
    #         plt.show()

    # if False:
    #     # pass parameters and pool
    #     processes = 4
    #     pool = schwimmbad.choose_pool(mpi=False if csize==1 else True, processes=processes)
    #     func = jams.functions.morris
    #     npars = 20
    #     lb    = np.zeros(npars)
    #     ub    = np.ones(npars)
    #     beta0              = 0.
    #     beta1              = np.random.standard_normal(npars)
    #     beta1[:10]         = 20.
    #     beta2              = np.random.standard_normal((npars,npars))
    #     beta2[:6,:6]       = -15.
    #     beta3              = np.zeros((npars,npars,npars))
    #     beta3[:5,:5,:5]    = -10.
    #     beta4              = np.zeros((npars,npars,npars,npars))
    #     beta4[:4,:4,:4,:4] = 5.
    #     args = [beta0, beta1, beta2, beta3, beta4] # Morris
    #     nt = 20
    #     ntotal  = 10*nt
    #     nsteps = 6
    #     verbose = 1

    #     out = screening(func, lb, ub, nt, arg=args, mask=None, nsteps=nsteps, ntotal=ntotal,
    #                     processes=processes, pool=pool, verbose=1)
    #     t2    = ptime.time()

    #     if crank == 0:
    #         strin = '[m]: {:.1f}'.format((t2-t1)/60.) if (t2-t1)>60. else '[s]: {:d}'.format(int(t2-t1))
    #         print('Time elapsed: ', strin)
    #         print('mu*, mu, std: ', out)
    #         out1 = out / out.max(axis=0)
    #         mustar = out1[:,0]
    #         mustar = np.sort(mustar)
    #         import matplotlib.pyplot as plt
    #         plt.plot(mustar, 'ro')
    #         plt.show()
    #     pool.close()

    # # PYEEE
    # from functools import partial
    # import numpy as np
    # from pyeee.functions import G, Gstar, K, fmorris
    # from pyeee.utils import func_wrapper

    # #
    # # G function
    # # seed for reproducible results
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = G
    # npars  = 6
    # params = [78., 12., 0.5, 2., 97., 33.] # G

    # # Partialise function with fixed parameters
    # arg   = [params]
    # kwarg = {}
    # obj = partial(func_wrapper, func, arg, kwarg)

    # # eee parameters
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nt      = 10
    # ntotal  = 50
    # nsteps  = 6
    # verbose = 1

    # out = ee(obj, lb, ub, nt, x0=None, mask=None, ntotal=ntotal, nsteps=nsteps, processes=4)
    # print('G')
    # print(np.around(out[:,0],3))

    # #
    # # G function
    # # seed for reproducible results
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = G
    # npars  = 6
    # params = [78., 12., 0.5, 2., 97., 33.] # G

    # # Partialise function with fixed parameters
    # arg   = [params]
    # kwarg = {}
    # obj = partial(func_wrapper, func, arg, kwarg)

    # # eee parameters
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nt      = 10
    # ntotal  = 50
    # nsteps  = 6
    # verbose = 1

    # out = ee(obj, lb, ub, nt, x0=None, mask=None, processes=4)
    # print('G')
    # print(np.around(out[:,0],3))

    # #
    # # Gstar function
    # # seed for reproducible results
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = Gstar
    # npars  = 10
    # params = [[np.ones(npars),     np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]], # G*
    #           [np.ones(npars),     np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
    #           [np.ones(npars)*0.5, np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
    #           [np.ones(npars)*0.5, np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
    #           [np.ones(npars)*2.0, np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
    #           [np.ones(npars)*2.0, np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]]
    #          ]

    # # eee parameters
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nsteps = 6
    # verbose = 1

    # for ii in range(len(params)):
    #     # Partialise function with fixed parameters
    #     arg   = params[ii]
    #     kwarg = {}
    #     obj = partial(func_wrapper, func, arg, kwarg)

    #     out = ee(obj, lb, ub, nt, x0=None, mask=None, ntotal=ntotal, nsteps=nsteps, processes=4)
    #     print('G* ', ii)
    #     print(np.around(out[:,0],3))

    # #
    # # Bratley / K function
    # # seed for reproducible results
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = K
    # npars  = 10
    # params = [] # K

    # # eee parameters
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nsteps = 6
    # verbose = 1

    # out = ee(func, lb, ub, nt, x0=None, mask=None, ntotal=ntotal, nsteps=nsteps, processes=4)
    # print('K')
    # print(np.around(out[:,0],3))


    # #
    # # Morris function
    # # seed for reproducible results
    # seed = 1234
    # np.random.seed(seed=seed)

    # func = fmorris
    # npars = 20
    # beta0              = 0.
    # beta1              = np.random.standard_normal(npars)
    # beta1[:10]         = 20.
    # beta2              = np.random.standard_normal((npars,npars))
    # beta2[:6,:6]       = -15.
    # beta3              = np.zeros((npars,npars,npars))
    # beta3[:5,:5,:5]    = -10.
    # beta4              = np.zeros((npars,npars,npars,npars))
    # beta4[:4,:4,:4,:4] = 5.

    # # Partialise Morris function with fixed parameters beta0-4
    # arg   = [beta0, beta1, beta2, beta3, beta4]
    # kwarg = {}
    # obj = partial(func_wrapper, func, arg, kwarg)

    # # eee parameters
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nsteps = 6
    # verbose = 1

    # out = ee(obj, lb, ub, nt, x0=None, mask=None, ntotal=ntotal, nsteps=nsteps, processes=4)
    # print('Morris')
    # print(np.around(out[:,0],3))

    
    # from functools import partial
    # import numpy as np
    # from pyeee.functions import ishigami_homma
    # from pyeee.utils import func_mask_wrapper
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = ishigami_homma
    # npars  = 3

    # x0   = np.ones(npars)
    # mask = np.ones(npars, dtype=np.bool)
    # mask[1] = False

    # arg   = [1., 3.]
    # kwarg = {}

    # obj = partial(func_mask_wrapper, func, x0, mask, arg, kwarg)

    # lb = np.ones(npars) * (-np.pi)
    # ub = np.ones(npars) * np.pi
    # nt      = 10
    # ntotal  = 50
    # nsteps  = 6

    # out = ee(obj, lb[mask], ub[mask],
    #          nt, ntotal=ntotal, nsteps=nsteps,
    #          processes=1)

    # print('Ishigami-Homma')
    # print(np.around(out[:,0],3))

    
    # from functools import partial
    # import numpy as np
    # import scipy.stats as stats
    # from pyeee.functions import ishigami_homma
    # from pyeee.utils import func_mask_wrapper
    # seed = 1234
    # np.random.seed(seed=seed)

    # func   = ishigami_homma
    # npars  = 3

    # x0   = np.ones(npars)
    # mask = np.ones(npars, dtype=np.bool)
    # mask[1] = False

    # arg   = [1., 3.]
    # kwarg = {}

    # obj = partial(func_mask_wrapper, func, x0, mask, arg, kwarg)

    # lb = np.ones(npars) * (-np.pi)
    # ub = np.ones(npars) * np.pi
    # dist      = [ stats.uniform for i in range(npars) ]
    # distparam = [ (lb[i],ub[i]-lb[i]) for i in range(npars) ]
    # lb = np.zeros(npars)
    # ub = np.ones(npars)
    # nt      = 10
    # ntotal  = 50
    # nsteps  = 6

    # out = ee(obj, lb[mask], ub[mask],
    #          nt, ntotal=ntotal, nsteps=nsteps,
    #          dist=[ dist[i] for i in np.where(mask)[0] ],
    #          distparam=[ distparam[i] for i in np.where(mask)[0] ],
    #          processes=1)

    # print('Ishigami-Homma dist')
    # print(np.around(out[:,0],3))
