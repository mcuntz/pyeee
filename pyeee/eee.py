#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
eee : Provides the function eee for Efficient/Sequential Elementary Effects,
      an extension of Morris' method of Elementary Effects by
      Cuntz, Mai et al. (Water Res Research, 2015).

This function was written by Matthias Cuntz while at Institut National
de Recherche en Agriculture, Alimentation et Environnement (INRAE),
Nancy, France.

Copyright (c) 2017-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Nov 2017 by Matthias Cuntz (mc (at) macu (dot) de)
* Added `weight` option, Jan 2018, Matthias Cuntz
* Added `plotfile` and made docstring sphinx compatible option, Jan 2018, Matthias Cuntz
* x0 optional; added verbose keyword; distinguish iterable and array_like parameter types, Jan 2020, Matthias Cuntz
* Rename ntsteps to nsteps to be consistent with screening/ee; and check if logfile is string rather thean checking for file handle, Feb 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following functions are provided

.. autosummary::
   see
   eee
"""
import numpy as np
import scipy.optimize as opt
# from pyeee import tee
# from pyeee import cost_square
# from pyeee import curvature, logistic_offset_p, dlogistic, d2logistic
# from pyeee import screening
from .tee import tee
from .utils import cost_square
from .general_functions import curvature, logistic_offset_p, dlogistic, d2logistic
from .screening import screening
# from tee import tee
# from utils import cost_square
# from general_functions import curvature, logistic_offset_p, dlogistic, d2logistic
# from screening import screening


__all__ = ['eee', 'see']


def _cleanup(lfile, pool, ipool):
    ''' Helper function closing logfile and pool if open. '''
    try:
        lfile.close()
    except:
        pass
    if pool is None:
        ipool.close()


# Python 3
# def eee(func, lb, ub,
#         x0=None, mask=None, ntfirst=5, ntlast=5, nsteps=6, weight=False,
#         seed=None, processes=1, pool=None,
#         verbose=0, logfile=None, plotfile=None):
def eee(func, *args, **kwargs):
    """
    Parameter screening using Efficient/Sequential Elementary Effects of
    Cuntz, Mai et al. (Water Res Research, 2015).

    Note, the input function must be callable as `func(x)`.

    Parameters
    ----------
    func : callable
        Python function callable as `func(x)` with `x` the function parameters.
    lb : array_like
        Lower bounds of parameters.
    ub : array_like
        Upper bounds of parameters.
    x0 : array_like, optional
        Parameter values used with `mask==0`.
    mask : array_like, optional
        Include (1,True) or exclude (0,False) parameters in screening (default: include all parameters).
    ntfirst : int, optional
        Number of trajectories in first step of sequential elementary effects (default: 5).
    ntlast : int, optional
        Number of trajectories in last step of sequential elementary effects (default: 5).
    nsteps : int, optional
        Number of intervals for each trajectory (default: 6)
    weight : boolean, optional
        If False, use the arithmetic mean mu* for each parameter if function has multiple outputs,
        such as the mean mu* of each time step of a time series (default).

        If True, return weighted mean mu*, weighted by sd.
    seed : int or array_like
        Seed for numpy``s random number generator (default: None).
    processes : int, optinal
        The number of processes to use to evaluate objective function and constraints (default: 1).
    pool : `schwimmbad` pool object, optinal
        Generic map function used from module `schwimmbad <https://schwimmbad.readthedocs.io/en/latest/>`_,
        which provides, serial, multiprocessor, and MPI mapping functions (default: None).

        The pool is chosen with:

            schwimmbad.choose_pool(mpi=True/False, processes=processes).

        The pool will be chosen automatically if pool is None.

        MPI pools can only be opened and closed once. If you want to use screening several
        times in one program, then you have to choose the pool, pass it to eee,
        and later close the pool in the calling progam.

    verbose : int, optional
        Print progress report during execution if verbose>0 (default: 0).
    logfile : File handle or logfile name
        File name of possible log file (default: None = no logfile will be written).
    plotfile : Plot file name
        File name of possible plot file with fit of logistic function to mu* of first trajectories
        (default: None = no plot produced).

    Returns
    -------
    mask : ndarray
        (len(lb),) Mask with 1=informative and 0=uninformative model parameters, to be used with '&' on input mask.

    See Also
    --------
    :func:`~pyeee.screening.screening` : Elementary Effects, same as

    :func:`~pyeee.screening.ee` : Elementary Effects

    Examples
    --------
    >>> import numpy as np
    >>> import pyeee
    >>> seed = 1023
    >>> np.random.seed(seed=seed)
    >>> npars = 10
    >>> lb    = np.zeros(npars)
    >>> ub    = np.ones(npars)
    >>> ntfirst = 10
    >>> ntlast  = 5
    >>> nsteps  = 6
    >>> out = pyeee.eee(pyeee.K, lb, ub, x0=None, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps)
    >>> print(np.where(out)[0] + 1)
    [1 2 3 4 6]


    History
    -------
    Written,  Matthias Cuntz, Nov 2017
    Modified, Matthias Cuntz, Jan 2018 - weight
                              Nov 2019 - plotfile, numpy docstring format
              Matthias Cuntz, Jan 2020 - x0 optional
                                       - verbose keyword
                                       - distinguish iterable and array_like parameter types
              Matthias Cuntz, Feb 2020 - ntsteps -> nsteps
                                       - check if logfile is string instead of checking for file handle
    """
    # Get keyword arguments
    # This allows mixing keyword arguments of eee and keyword arguments to be passed to optimiser.
    # The mixed syntax eee(func, *args, logfile=None, **kwargs) is only working in Python 3
    # so need a workaround in Python 2, i.e. read all as keyword args and take out the keywords for eee.
    x0        = kwargs.pop('x0', None)
    mask      = kwargs.pop('mask', None)
    ntfirst   = kwargs.pop('ntfirst', 5)
    ntlast    = kwargs.pop('ntlast', 5)
    nsteps    = kwargs.pop('nsteps', 6)
    weight    = kwargs.pop('weight', False)
    seed      = kwargs.pop('seed', None)
    processes = kwargs.pop('processes', 1)
    pool      = kwargs.pop('pool', None)
    verbose   = kwargs.pop('verbose', 0)
    logfile   = kwargs.pop('logfile', None)
    plotfile  = kwargs.pop('plotfile', None)

    # Set up MPI if available
    try:
        from mpi4py import MPI
        comm  = MPI.COMM_WORLD
        csize = comm.Get_size()
        crank = comm.Get_rank()
    except ImportError:
        comm  = None
        csize = 1
        crank = 0

    # Logfile
    if crank == 0:
        if logfile is None:
            lfile = None
        else:
            # haswrite = getattr(logfile, "write", None)
            # if haswrite is None:
            #     lfile = open(logfile, "w")
            # else:
            #     if not callable(haswrite):
            #         lfile = logfile
            #     else:
            #         raise InputError('x0 must be given if mask is set')
            if isinstance(logfile, str):
                lfile = open(logfile, "w")
            else:
                lfile = logfile
    else:
        lfile = None

    # Start
    if crank == 0:
        if (verbose > 0):
            tee('Start screening in eee.', file=lfile)
        else:
            if lfile is not None:
                print('Start screening in eee.', file=lfile)

    # Check
    assert len(args) == 2, 'lb and ub must be given as arguments.'
    lb, ub = args[:2]
    npara = len(lb)
    if crank == 0:
        assert len(lb) == len(ub), 'Lower and upper bounds have not the same size.'
    lb = np.array(lb)
    ub = np.array(ub)

    # mask
    if mask is None:
        ix0    = np.ones(npara)
        imask  = np.ones(npara, dtype=np.bool)
        iimask = np.arange(npara, dtype=np.int)
        nmask  = npara
    else:
        if x0 is None:
            raise InputError('x0 must be given if mask is set')
        ix0    = np.copy(x0)
        imask  = np.copy(mask)
        iimask = np.where(imask)[0]
        nmask  = iimask.size
        if nmask == 0:
            if crank == 0:
                if (verbose > 0):
                    tee('\nAll parameters masked, nothing to do.', file=lfile)
                    tee('Finished screening in eee.', file=lfile)
                else:
                    if lfile is not None:
                        print('\nAll parameters masked, nothing to do.', file=lfile)
                        print('Finished screening in eee.', file=lfile)
                if logfile is not None: lfile.close()
            # Return all true
            if mask is None:
                return np.ones(len(lb), dtype=np.bool)
            else:
                return mask
    if crank == 0:
        if (verbose > 0):
            tee('\nScreen unmasked parameters: ', nmask, iimask+1, file=lfile)
        else:
            if lfile is not None:
                print('\nScreen unmasked parameters: ', nmask, iimask+1, file=lfile)

    # Seed random number generator
    if seed is not None: np.random.seed(seed=seed)  # same on all ranks because trajectories are sampled on all ranks

    # Choose the right mapping function: single, multiprocessor or mpi
    if pool is None:
        import schwimmbad
        ipool = schwimmbad.choose_pool(mpi=False if csize == 1 else True, processes=processes)
    else:
        ipool = pool

    # Step 1 of Cuntz et al. (2015) - first screening with ntfirst trajectories, calc mu*
    res = screening( # returns (npara,3) with mu*, mu, std if nt>1
        func, lb, ub,
        x0=ix0, mask=imask,
        nt=ntfirst, nsteps=nsteps, ntotal=10*ntfirst,
        processes=processes, pool=ipool,
        verbose=0)
    if res.ndim > 2:
        if weight:
            mustar = np.sum(res[:, iimask, 2] * res[:, iimask, 0],axis=0) / np.sum(res[:, iimask, 2], axis=0)
        else:
            mustar = np.mean(res[:, iimask, 0], axis=0)
    else:
        mustar = res[iimask, 0]

    # Step 2 of Cuntz et al. (2015) - calc eta*
    mumax  = np.amax(mustar)
    xx     = np.arange(nmask) / np.float(nmask-1)
    iisort = np.argsort(mustar)
    yy     = mustar[iisort] / mumax

    if crank == 0:
        if (verbose > 0):
            tee('\nSorted means of absolute elementary effects (mu*): ', mustar[iisort], file=lfile)
            tee('Normalised mu* = eta*: ', yy, file=lfile)
            tee('Corresponding to parameters: ', iimask[iisort] + 1, file=lfile)
        else:
            if lfile is not None:
                print('\nSorted means of absolute elementary effects (mu*): ', mustar[iisort], file=lfile)
                print('Normalised mu* = eta*: ', yy, file=lfile)
                print('Corresponding to parameters: ', iimask[iisort] + 1, file=lfile)

    # Step 3.1 of Cuntz et al. (2015) - fit logistic function
    #               [y-max,    steepness,                       inflection point, offset]
    pini = np.array([yy.max(), (yy.max() - yy.max()) / xx.max(), 0.5 * xx.max(), yy.min()])
    plogistic, f, d = opt.fmin_l_bfgs_b(cost_square,
                                        pini,
                                        args=(logistic_offset_p, xx, yy),
                                        approx_grad=1,
                                        bounds=[(None, None), (None, None), (None, None), (None, None)],
                                        iprint=0,
                                        disp=0)

    # Step 3.2 of Cuntz et al. (2015) - determine point of steepest curvature -> eta*_thresh
    def mcurvature(*args, **kwargs):
        return -curvature(*args, **kwargs)

    x_K = opt.brent(mcurvature,                     # x_K
                    args=(dlogistic, d2logistic, plogistic[0], plogistic[1], plogistic[2]),
                    brack=(xx[0], xx[-1]))
    curvmax    = logistic_offset_p(x_K, plogistic)  # L(x_K)
    eta_thresh = curvmax                            # eta*_thresh = L(x_K) # in range 0-1
    if (curvmax > 0.2) or (x_K < xx[0]):
        x_scaled   = xx[0]                          # x_K = min(x)
        eta_thresh = np.min(mustar) / mumax         # eta*_thresh = min(mu*)/max(mu*)
    mu_thresh = eta_thresh * mumax                  # mu*_thresh = eta*_thresh*max(mu*)

    if crank == 0:
        if (verbose > 0):
            tee('\nThreshold eta*_thresh, mu*_tresh: ', eta_thresh, mu_thresh, file=lfile)
            tee('L(x_K): ', logistic_offset_p(x_K, plogistic), file=lfile)
            tee('p_opt of L: ', plogistic, file=lfile)
        else:
            if lfile is not None:
                print('\nThreshold eta*_thresh, mu*_tresh: ', eta_thresh, mu_thresh, file=lfile)
                print('L(x_K): ', logistic_offset_p(x_K, plogistic), file=lfile)
                print('p_opt of L: ', plogistic, file=lfile)

    # Plot first mu* of elementary effects with logistic function and threshold
    if crank == 0:
        if plotfile is not None:
            try:
                import matplotlib as mpl
                mpl.use('Agg')
                import matplotlib.pyplot as plt
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'  # Arial, Verdana
                mpl.rc('savefig', dpi=300, format='png')
                if npara > 99:
                    mpl.rc('font', size=8)
                else:
                    mpl.rc('font', size=11)
                fig = plt.figure()
                sub = plt.subplot(111)
                xx = xx
                yy = mustar[iisort]
                line1 = sub.plot(xx, yy, 'ro')
                nn = 1000
                xx2 = xx.min() + np.arange(nn) / float(nn - 1) * (xx.max() - xx.min())
                yy2 = logistic_offset_p(xx2, plogistic) * mumax
                line2 = sub.plot(xx2, yy2, 'b-')
                xmin, xmax = sub.get_xlim()
                line3 = sub.plot([xmin, xmax], [mu_thresh, mu_thresh], 'k-')
                if npara > 99:
                    xnames = ['{:03d}'.format(i) for i in iimask[iisort] + 1]
                else:
                    xnames = ['{:02d}'.format(i) for i in iimask[iisort] + 1]
                plt.setp(sub, xticks=xx, xticklabels=xnames)
                plt.setp(sub, xlabel='Parameter')
                plt.setp(sub, ylabel=r'$\mu*$')
                fig.savefig(plotfile, transparent=False, bbox_inches='tight', pad_inches=0.035)
                plt.close(fig)
            except ImportError:
                pass

    # Step 4 of Cuntz et al. (2015) - Discard from next steps all parameters with
    #                                 eta* >= eta*_tresh, i.e. mu* >= mu*_tresh
    imask[iimask] = imask[iimask] & (mustar < mu_thresh)

    if np.all(~imask):
        if crank == 0:
            if (verbose > 0):
                tee('\nNo more parameters to screen, i.e. all (unmasked) parameters are informative.', file=lfile)
                tee('Finished screening in eee.', file=lfile)
            else:
                if lfile is not None:
                    print('\nNo more parameters to screen, i.e. all (unmasked) parameters are informative.', file=lfile)
                    print('Finished screening in eee.', file=lfile)
            _cleanup(lfile, pool, ipool)
        # Return all true
        if mask is None:
            return np.ones(len(lb), dtype=np.bool)
        else:
            return mask

    # Step 5 and 6 of Cuntz et al. (2015) - Next trajectory with remaining parameters.
    #                                       Discard all parameters with |EE| >= mu*_tresh
    #                                     - Repeat until no |EE| >= mu*_tresh
    niter = 1
    donext = True
    while donext:
        if crank == 0:
            if (verbose > 0):
                tee('\nParameters remaining for iteration ', niter, ':', np.where(imask)[0] + 1, file=lfile)
            else:
                if lfile is not None:
                    print('\nParameters remaining for iteration ', niter, ':', np.where(imask)[0] + 1, file=lfile)
        iimask = np.where(imask)[0]
        res = screening( # returns EE(parameters) if nt=1
            func, lb, ub,
            x0=ix0, mask=imask,
            nt=1, nsteps=nsteps, ntotal=10,
            processes=processes, pool=ipool,
            verbose=0)

        # absolute EE
        if res.ndim > 2:
            if weight:
                mustar = np.sum(res[:, iimask, 2] * res[:, iimask, 0], axis=0) / np.sum(res[:, iimask, 2], axis=0)
            else:
                mustar = np.mean(res[:, iimask, 0], axis=0)
        else:
            mustar = res[iimask, 0]

        if crank == 0:
            if (verbose > 0):
                tee('Absolute elementary effects |EE|: ', mustar, file=lfile)
            else:
                if lfile is not None:
                    print('Absolute elementary effects |EE|: ', mustar, file=lfile)

        imask[iimask] = imask[iimask] & (mustar < mu_thresh)

        if np.all(~imask):
            if crank == 0:
                if (verbose > 0):
                    tee('\nNo more parameters to screen, i.e. all (unmasked) parameters are informative.', file=lfile)
                    tee('Finished screening in eee.', file=lfile)
                else:
                    if lfile is not None:
                        print('\nNo more parameters to screen, i.e. all (unmasked) parameters are informative.', file=lfile)
                        print('Finished screening in eee.', file=lfile)
                _cleanup(lfile, pool, ipool)
            # Return all true
            if mask is None:
                return np.ones(len(lb), dtype=np.bool)
            else:
                return mask

        # Step 6 of Cuntz et al. (2015) - Repeat until no |EE| >= mu*_tresh
        if np.all(mustar < mu_thresh): donext = False

        niter += 1

    # Step 7 of Cuntz et al. (2015) - last screening with ntlast trajectories
    #                                 all parameters with mu* < mu*_thresh are final noninformative parameters
    if crank == 0:
        if (verbose > 0):
            tee('\nParameters remaining for last screening:', np.where(imask)[0] + 1, file=lfile)
        else:
            if lfile is not None:
                print('\nParameters remaining for last screening:', np.where(imask)[0] + 1, file=lfile)

    iimask = np.where(imask)[0]

    res = screening( # (npara,3) with mu*, mu, std if nt>1
        func, lb, ub,
        x0=ix0, mask=imask,
        nt=ntlast, nsteps=nsteps, ntotal=10 * ntlast,
        processes=processes, pool=ipool,
        verbose=0)
    if res.ndim > 2:
        if weight:
            mustar = np.sum(res[:, iimask, 2] * res[:, iimask, 0], axis=0) / np.sum(res[:, iimask, 2], axis=0)
        else:
            mustar = np.mean(res[:, iimask, 0], axis=0)
    else:
        mustar = res[iimask, 0]

    if crank == 0:
        if ntlast > 1:
            if (verbose > 0):
                tee('Final mu*: ', mustar, file=lfile)
            else:
                if lfile is not None:
                    print('Final mu*: ', mustar, file=lfile)
        else:
            if (verbose > 0):
                tee('Final absolute elementary effects |EE|: ', mustar, file=lfile)
            else:
                if lfile is not None:
                    print('Final absolute elementary effects |EE|: ', mustar, file=lfile)

    imask[iimask] = imask[iimask] & (mustar < mu_thresh)

    if np.all(~imask):
        if crank == 0:
            if (verbose > 0):
                tee('\nNo more parameters left after screening, i.e. all (unmasked) parameters are informative.',
                    file=lfile)
                tee('Finished screening in eee.', file=lfile)
            else:
                if lfile is not None:
                    print('\nNo more parameters left after screening, i.e. all (unmasked) parameters are informative.',
                    file=lfile)
                    print('Finished screening in eee.', file=lfile)
            _cleanup(lfile, pool, ipool)
        # Return all true
        if mask is None:
            return np.ones(len(lb), dtype=np.bool)
        else:
            return mask

    # Return mask with unmasked informative model parameters (to be used with 'and' on initial mask)
    if mask is None:
        out = ~imask
    else:
        out = (~imask) & mask  # (true where now zero, i.e. were masked or informative) and (initial mask)

    if crank == 0:
        if (verbose > 0):
            tee('\nFinal informative parameters:', np.sum(out), np.where(out)[0] + 1, file=lfile)
            tee('Final noninformative parameters:', np.sum(imask), np.where(imask)[0] + 1, file=lfile)
            tee('\nFinished screening in eee.', file=lfile)
        else:
            if lfile is not None:
                print('\nFinal informative parameters:', np.sum(out), np.where(out)[0] + 1, file=lfile)
                print('Final noninformative parameters:', np.sum(imask), np.where(imask)[0] + 1, file=lfile)
                print('\nFinished screening in eee.', file=lfile)
        # Close logfile and pool
        _cleanup(lfile, pool, ipool)

    return out


def see(func, *args, **kwargs):
    """
    Wrapper function for :func:`~pyeee.eee.eee`.
    """
    return eee(func, *args, **kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    # from sa_functions import fmorris, tee

    # Morris with MPI
    
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

    # seed = None # 1025

    # if seed is not None:
    #     np.random.seed(seed=seed)

    # func = fmorris
    # npars = 20
    # lb    = np.zeros(npars)
    # ub    = np.ones(npars)
    # beta0              = 0.
    # beta1              = np.random.standard_normal(npars)
    # beta1[:10]         = 20.
    # beta2              = np.random.standard_normal((npars,npars))
    # beta2[:6,:6]       = -15.
    # beta3              = np.zeros((npars,npars,npars))
    # beta3[:5,:5,:5]    = -10.
    # beta4              = np.zeros((npars,npars,npars,npars))
    # beta4[:4,:4,:4,:4] = 5.
    # args = [beta0, beta1, beta2, beta3, beta4] # Morris
    # ntfirst = 10
    # ntlast  = 5
    # nsteps  = 6
    # verbose = 1

    # out = eee(func, lb, ub, *args, x0=None, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps, processes=4)

    # t2    = ptime.time()

    # if crank == 0:
    #     strin = '[m]: {:.1f}'.format((t2-t1)/60.) if (t2-t1)>60. else '[s]: {:d}'.format(int(t2-t1))
    #     tee('Time elapsed: ', strin)
    #     tee('mask (1: informative, 0: noninformative): ', out)

    # PYEEE
    # from functools import partial
    # import numpy as np
    # from sa_test_functions import G, Gstar, K, fmorris
    # from function_wrapper import func_wrapper

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
    # ntfirst = 10
    # ntlast  = 5
    # nsteps  = 6
    # verbose = 1

    # out = eee(obj, lb, ub, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps, processes=4, plotfile='g.png')
    # print('G')
    # print(np.where(out)[0] + 1)

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
    # ntfirst = 10
    # ntlast  = 5
    # nsteps  = 6
    # verbose = 1

    # for ii in range(len(params)):
    #     # Partialise function with fixed parameters
    #     arg   = params[ii]
    #     kwarg = {}
    #     obj = partial(func_wrapper, func, arg, kwarg)

    #     out = eee(obj, lb, ub, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps, processes=4, plotfile='gstar'+str(ii)+'.png',logfile='log'+str(ii)+'.txt')
    #     print('G* ', ii)
    #     print(np.where(out)[0] + 1)

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
    # ntfirst = 10
    # ntlast  = 5
    # nsteps  = 6
    # verbose = 1

    # out = eee(func, lb, ub, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps, processes=4, plotfile='k.png')
    # print('K')
    # print(np.where(out)[0] + 1)


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
    # ntfirst = 10
    # ntlast  = 5
    # nsteps  = 6
    # verbose = 1

    # out = eee(obj, lb, ub, mask=None, ntfirst=ntfirst, ntlast=ntlast, nsteps=nsteps, processes=4, plotfile='morris.png', verbose=1)
    # print('Morris')
    # print(np.where(out)[0] + 1)
