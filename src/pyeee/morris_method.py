#!/usr/bin/env python
"""
Morris' Method of Elementary Effects.

Module includes optimised sampling of trajectories including optional groups as
well as calculation of the Morris measures mu, stddev and mu*.

Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis. Wiley
Series in Probability and Statistics, John Wiley & Sons, New York, 1-504,
pages 68ff

Provided functions are:
    morris_sampling - Sample trajectories in parameter space

    elementary_effects - Calculate Elementary Effects from model
    output on trajectories

Note that the functions morris_sampling and elementary_effects are wrappers for
the functions Optimized_Groups and Morris_Measure_Groups of F. Campolongo and
J. Cariboni ported to Python by Stijn Van Hoey.

This module was originally written by Stijn Van Hoey as a translation
from an original Matlab code of F. Campolongo and J. Cariboni, JRC -
IPSC Ispra, Varese, IT. It was adapted by Matthias Cuntz while at
Department of Computational Hydrosystems, Helmholtz Centre for
Environmental Research - UFZ, Leipzig, Germany, and continued while at
Institut National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2012-2022 Stijn Van Hoey, Matthias Cuntz, see AUTHORS.rst
            for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following wrappers are provided

.. autosummary::
   morris_sampling
   elementary_effects

History
    * Written in Matlab by F. Campolongo, J. Cariboni,
      JRC - IPSC Ispra, Varese, IT.
      Last Update 15 November 2005 by J. Cariboni:
      http://sensitivity-analysis.jrc.ec.europa.eu/software/index.htm
      now at: https://ec.europa.eu/jrc/en/samo/simlab
    * Translated to Python in May 2012 by Stijn Van Hoey.
    * Adapted to Python 3, etc., Oct 2013, Matthias Cuntz
    * Went from exponential time increase with number of trajectories to linear
      increase by using in Optimised_Groups one call to cdist from
      scipy.spatial.distance and removed one loop in a loop over total number
      of trajectories.
      Several further little improvements on speed, Dec 2017, Matthias Cuntz
    * Allow single trajectories, Dec 2017, Matthias Cuntz
    * Catch degenerated case where lower bound==upper bound, return 0,
      Feb 2018, Matthias Cuntz
    * Use integer division operator // instead of / for trajectory length r,
      Jul 2018, Fabio Genaretti
    * Distance matrix is not done for all trajectories at once because of very
      large memory requirement, Aug 2018, Matthias Cuntz & Fabio Gennaretti
    * Changed to Sphinx docstring and numpydoc, Dec 2019, Matthias Cuntz
    * Distinguish iterable and array_like parameter types,
      Jan 2020, Matthias Cuntz
    * Remove np.matrix in Sampling_Function_2, called in Optimized_Groups to
      remove numpy deprecation warnings, Jan 2020, Matthias Cuntz
    * Plot diagnostic figures in png files if matplotlib installed, Feb 2020,
      Matthias Cuntz
    * Renamed file to morris_method.py, Feb 2020, Matthias Cuntz
    * Adjusted argument and keyword argument names to be consistent with
      pyeee, Feb 2020, Matthias Cuntz
    * Make number of final trajectories an argument instead of a keyword
      argument and sample default of 10*final trajectories,
      Feb 2020, Matthias Cuntz
    * Sample not only from uniform distribution but allow all distributions of
      scipy.stats, Mar 2020, Matthias Cuntz
    * More consistent docstrings, Jan 2022, Matthias Cuntz
    * Raise Error if more than one component changed at once,
      Jul 2023, Matthias Cuntz

"""
import numpy as np


__all__ = ['morris_sampling', 'elementary_effects']


def Sampling_Function_2(p, k, r, lb, ub, GroupMat=np.array([])):
    """
    Morris sampling function


    Definition
    ----------
    def  Sampling_Function_2(p, k, r, lb, ub, GroupMat=np.array([])):


    Input
    -----
    p : int
        number of intervals considered in [0,1]
    k : int
        number of factors examined (``sizea=k``).
        In case groups are chosen the number of factors is stored in
        ``NumFact`` and ``sizea`` becomes the number of created groups
        (``sizea=GroupMat.shape[1]``).
    r : int
        sample size
    lb : array-like of size (``sizea``,)
        Lower Bound for each factor in list or array
    ub : array-like of size (``sizea``,)
        Upper Bound for each factor in list or array


    Optional Input
    --------------
    GroupMat : ndarray with size (``NumFact``, ``GroupNumber``)
        Array which describes the chosen groups. (default: ``np.array([])``)
        Each column represents a group and its elements are set to 1
        in correspondence of the factors that belong to the fixed group. All
        the other elements are zero.


    Output
    ------
    list
        [``Outmatrix``, ``OutFact``]
        ``Outmatrix(sizeb*r, sizea)`` for the entire sample size computed
            ``In(i,j)`` matrices
        ``OutFact(sizea*r)` `for the entire sample size computed ``Fact(i,1)``
            vectors


    Notes
    -----
    Local Variables
        NumFact : number of factors examined in the case when groups are chosen
        GroupNumber : Number of groups (eventually 0)
        sizeb : sizea+1
        randmult(sizea) : vector of random +1 and -1
        perm_e(sizea) : vector of sizea random permutated indeces
        fact(sizea) : vector containing the factor varied within each traj
        DDo(sizea,sizea) : D* in Morris, 1991
        A(sizeb,sizea) : Jk+1,k in Morris, 1991
        B(sizeb,sizea) : B in Morris, 1991
        Po(sizea,sizea) : P* in Morris, 1991
        Bo(sizeb,sizea) : B* in Morris, 1991
        Ao(sizeb) : Jk+1 in Morris, 1991
        xo(sizea) : x* in Morris, 1991 (starting point for the trajectory)
        In(sizeb,sizea) : for each loop orientation matrix. It corresponds to
            a trajectory of k step in the parameter space and it provides a
            single elementary effect per factor
        Fact(sizea) : For each loop vector indicating which factor or group of
            factors has been changed in each step of the trajectory
        AuxMat(sizeb,sizea) : Delta*0.5*((2*B - A) * DD0 + A) in Morris, 1991.
            The AuxMat is used as in Morris design for single factor analysis,
            while it constitutes an intermediate step for the group analysis.

    Note: B0 is constructed as in Morris design when groups are not considered.

        When groups are considered the routine follows the following steps:

           1. Creation of P0 and DD0 matrices defined in Morris for the
              groups. This means that the dimensions of these 2 matrices are
              (GroupNumber,GroupNumber).

           2. Creation of AuxMat matrix with (GroupNumber+1,GroupNumber)
              elements.

           3. Definition of GroupB0 starting from AuxMat, GroupMat and P0.

           4. The final B0 for groups is obtained as [ones(sizeb,1)*x0' +
              GroupB0]. The P0 permutation is present in GroupB0 and it's
              not necessary to permute the matrix (ones(sizeb,1)*x0')
              because it's already randomly created.


    References
    ----------
    Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis.
        Wiley Series in Probability and Statistics, John Wiley & Sons,
        New York, 1-504. - on page 68ff


    History
    -------
    Written original Matlab code by F. Campolongo, J. Cariboni,
        JRC - IPSC Ispra, Varese, IT
        Last Update: 15 November 2005 by J.Cariboni
        http://sensitivity-analysis.jrc.ec.europa.eu/software/index.htm
        now at: https://ec.europa.eu/jrc/en/samo/simlab
    Modified, Stijn Van Hoey, May 2012 - ported to Python
              Matthias Cuntz, Oct 2013
                  - adapted to JAMS Python package and ported to Python 3
              Matthias Cuntz, Jan 2020 - remove np.matrix
    """
    # Parameters and initialisation of the output matrix
    sizea = k
    Delta = p / (2. * (p - 1.))
    NumFact = sizea
    if GroupMat.shape[0] == GroupMat.size:
        Groupnumber = 0
    else:
        Groupnumber = GroupMat.shape[1]  # size(GroupMat,2)
        sizea = GroupMat.shape[1]

    sizeb = sizea + 1

    Outmatrix = np.zeros(((sizea + 1) * r, NumFact))
    OutFact = np.zeros(((sizea + 1) * r, 1))
    # For each i generate a trajectory
    for i in range(r):
        Fact = np.zeros(sizea + 1)
        # Construct DD0
        DD0 = np.diagflat(np.sign(np.random.random(k) * 2 - 1))

        # Construct B (lower triangular)
        B = np.tri((sizeb), sizea, k=-1, dtype=int)

        # Construct A0, A
        A0 = np.ones((sizeb, 1))
        A  = np.ones((sizeb, NumFact))

        # Construct the permutation matrix P0. In each column of P0 one
        # randomly chosen element equals 1 while all the others equal zero. P0
        # tells the order in which order factors are changed in each Note that
        # P0 is then used reading it by rows.
        I = np.eye(sizea)
        P0 = I[:, np.random.permutation(sizea)]

        # When groups are present the random permutation is done only on B. The
        # effect is the same since the added part (A0*x0') is completely
        # random.
        if Groupnumber != 0:
            B = np.dot(B, np.dot(GroupMat, P0.T).T)

        # Compute AuxMat both for single factors and groups analysis. For
        # Single factors analysis AuxMat is added to (A0*X0) and then
        # permutated through P0. When groups are active AuxMat is used to build
        # GroupB0. AuxMat is created considering DD0. If the element on DD0
        # diagonal is 1 then AuxMat will start with zero and add Delta. If the
        # element on DD0 diagonal is -1 then DD0 will start Delta and goes to
        # zero.
        AuxMat = Delta * 0.5 * (np.dot(2. * B - A, DD0) + A)

        # a --> Define the random vector x0 for the factors. Note that x0 takes
        # value in the hypercube
        # [0,...,1-Delta]*[0,...,1-Delta]*[0,...,1-Delta]*[0,...,1-Delta]
        # Original in Stijn Van Hoey's version
        # xset=np.arange(0.0,1.0-Delta,1.0/(p-1))
        # Jule's version from The Primer
        # xset=np.arange(0.0,1.0-1.0/(p-1),1.0/(p-1))
        # Matthias thinks that the difference between Python and Matlab is that
        # Python is not taking the last element; therefore the following
        # version
        xset = np.arange(0.0, 1.00000001 - Delta, 1.0 / (p - 1))
        x0 = xset.take(list(np.ceil(
            np.random.random(k) * np.floor(p / 2)) - 1))
        x0 = x0[np.newaxis, :]

        # b --> Compute the matrix B*, here indicated as B0. Each row in B0 is
        # a trajectory for Morris Calculations. The dimension of B0 is
        # (Numfactors+1,Numfactors)
        if Groupnumber != 0:
            B0 = np.dot(A0, x0) + AuxMat
        else:
            B0 = np.dot(np.dot(A0, x0) + AuxMat, P0)

        # c --> Compute values in the original intervals
        # B0 has values x(i,j) in [0, 1/(p -1), 2/(p -1), ... , 1].
        # To obtain values in the original intervals [lb, ub] we compute
        # lb(j) + x(i,j)*(ub(j)-lb(j))
        In = np.tile(lb, (sizeb, 1)) + B0 * np.tile((ub - lb), (sizeb, 1))

        # Create the Factor vector. Each component of this vector indicates
        # which factor or group of factors has been changed in each step of the
        # trajectory.
        for j in range(sizea):
            ii = np.where(P0[j, :])[0]
            if ii.size > 1:
                raise ValueError('More than one component changed per step')
            Fact[j] = ii[0]
        # Enkel om vorm logisch te houden. of Fact kleiner maken
        Fact[sizea] = int(-1)

        # append the create traject to the others
        Outmatrix[i * (sizea + 1):(i + 1) * (sizea + 1), :] = In
        OutFact[i * (sizea + 1):(i + 1) * (sizea + 1)] = Fact.reshape(
            (sizea + 1, 1))

    return Outmatrix, OutFact


def Optimized_Groups(NumFact, lb, ub, r,
                     p=6, N=None,
                     dist=None, distparam=None,
                     GroupMat=np.array([]), Diagnostic=0):
    """
    Optimisation in the choice of trajectories for Morris experiment,
    that means elementary effects


    Definition
    ----------
    def Optimized_Groups(NumFact, lb, ub, r, p=6, N=None,
                         GroupMat=np.array([]), Diagnostic=0):


    Input
    -----
    NumFact
        Number of factors
    lb
        [NumFact] Lower bound of the uniform distribution for each factor
        or lower fraction of percent point function ppf if distribution given.
    ub
        [NumFact] Upper bound of the uniform distribution for each factor
        or upper fraction of percent point function ppf if distribution given.
    r
        Final number of optimal trajectories


    Optional Input
    --------------
    p
        Number of levels (default: 6)
    N
        Total number of trajectories. If None: N=10*r (default: None)
    dist
        List of None or scipy.stats distribution objects for each factor
        having the method ppf, Percent Point Function (Inverse of CDF)
        (default: None).

        If None, the uniform distribution will be sampled from lower bound lb
        to upper bound ub.

        If dist is scipy.stats.uniform, the ppf will be sampled from the lower
        fraction given in lb and the upper fraction in ub. The sampling
        interval is then given by the parameters loc=lower and
        scale=interval=upper-lower in param.
    distparam
        List with tuples with parameters as required for dist (default: (0,1)).
        All distributions of scipy.stats have location and scale parameters,
        at least. loc and scale are implemented as keyword arguments in
        scipy.stats. Other parameters such as the shape parameter of the gamma
        distribution must hence be given first,
        e.g. (shape,loc,scale) for the gamma distribution.
        distparam is ignored if dist is None.
        The percent point function ppf is called like this:
        dist(*distparam).ppf(x)
    GroupMat
        [NumFact,NumGroups] Matrix describing the groups.
        (default: np.array([]))

        Each column represents a group and its elements are set to 1 in
        correspondence of the factors that belong to the fixed group. All the
        other elements are zero.
    Diagnostic
        1=plot the histograms and compute the efficiency of the sampling or
        not, 0 otherwise (default)


    Output
    ------
    [OptMatrix, OptOutVec]


    References
    ----------
    Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis. Wiley
    Series in Probability and Statistics, John Wiley & Sons, New York, 1-504. -
    on page 68ff

    """
    from scipy.spatial import distance
    import scipy.stats as stats

    if N is None:
        N = 10 * r

    assert len(lb) == NumFact, 'Lower bound must have length NumFact.'
    assert len(ub) == NumFact, 'Upper bound must have length NumFact.'
    if dist is not None:
        assert len(lb) == len(dist), ('scipy.stats distribution object or'
                                      ' None has to be given for each'
                                      ' parameter.')
        for dd in dist:
            if dd is not None:
                if not isinstance(dd,
                                  (stats.rv_discrete, stats.rv_continuous)):
                    raise TypeError(f'{dd} is not a scipy.stats distribution'
                                    f' object.')

    # np.random.seed(seed=1025)
    # Sample trajectorie between 0 and 1. Will be rescaled to specific
    # distributions later.
    lbt = np.zeros(NumFact)
    ubt = np.ones(NumFact)
    # Version with Groups
    OutMatrix, OutFact = Sampling_Function_2(p, NumFact, N, lbt, ubt, GroupMat)

    try:
        Groupnumber = GroupMat.shape[1]
    except:  # pragma: no cover
        Groupnumber = 0

    if Groupnumber != 0:
        sizeb = Groupnumber + 1
    else:
        sizeb = NumFact + 1

    # Compute the distance between all pair of trajectories (sum of the
    # distances between points) The distance matrix is a matrix N*N The
    # distance is defined as the sum of the distances between all pairs of
    # points if the two trajectories differ, 0 otherwise
    Dist = np.zeros((N, N))
    Diff_Traj = np.arange(0.0, N, 1.0)
    # combine all trajectories: eg N=3: 0&1; 0&2; 1&2 (is not dependent from
    # sequence)
    for j in range(N):
        for z in range(j + 1, N):
            MyDist = distance.cdist(
                OutMatrix[sizeb * j:sizeb * (j + 1), :],
                OutMatrix[sizeb * z:sizeb * (z + 1), :])
            if np.where(MyDist == 0.)[0].size == sizeb:
                # Same trajectory. If the number of zeros in Dist matrix is
                # equal to (NumFact+1) then the trajectory is a replica. In
                # fact (NumFact+1) is the maximum number of points that two
                # trajectories can have in common
                Dist[j, z] = 0.
                Dist[z, j] = 0.
                # Memorise the replicated trajectory
                Diff_Traj[z] = -1.  # the z value identifies the duplicate
            else:
                # Define the distance between two trajectories as
                # the minimum distance among their points
                dd = np.sum(MyDist)
                Dist[j, z] = dd
                Dist[z, j] = dd

    # prepare array with excluded duplicates (alternative would be deleting
    # rows)
    # iidup = np.where(Diff_Traj == -1.)[0]
    # dupli = iidup.size
    iiind = np.where(Diff_Traj != -1.)[0]
    New_N = iiind.size  # N - iidup.size
    New_OutMatrix = np.zeros((sizeb * New_N, NumFact))
    New_OutFact   = np.zeros((sizeb * New_N, 1))

    # Eliminate replicated trajectories in the sampled matrix
    ID = 0
    for i in range(N):
        if Diff_Traj[i] != -1.:
            New_OutMatrix[ID * sizeb:(ID + 1) * sizeb, :] = (
                OutMatrix[i * sizeb:(i + 1) * sizeb, :])
            New_OutFact[ID * sizeb:(ID + 1) * sizeb, :] = (
                OutFact[i * sizeb:(i + 1) * sizeb, :])
            ID += 1

    # Select in the distance matrix only the rows and columns of different
    #   trajectories Dist_Diff =
    #   np.delete(Dist_Diff,np.where(Diff_Traj==-1.)[0])
    # moet 2D matrix zijn... wis rijen ipv hou bij
    Dist_Diff = Dist[iiind, :]
    Dist_Diff = Dist_Diff[:, iiind]

    # Select the optimal set of trajectories
    Traj_Vec = np.zeros((New_N, r), dtype=int)
    OptDist  = np.zeros((New_N, r))
    for m in range(New_N):  # each row in Traj_Vec
        Traj_Vec[m, 0] = m
        for z in range(1, r):  # elements in columns after first
            New_Dist_Diff  = np.sqrt(np.sum(Dist_Diff[Traj_Vec[m, :z], :]**2,
                                            axis=0))
            ii = New_Dist_Diff.argmax()
            Traj_Vec[m, z] = ii
            OptDist[m, z]  = New_Dist_Diff[ii]

    # Construct optimal matrix
    SumOptDist = np.sum(OptDist, axis=1)
    # Find the maximum distance
    Pluto = np.where(SumOptDist == SumOptDist.max())[0]
    Opt_Traj_Vec = Traj_Vec[Pluto[0], :]

    OptMatrix = np.zeros((sizeb * r, NumFact))
    OptOutVec = np.zeros((sizeb * r, 1))

    for k in range(r):
        OptMatrix[k * sizeb:(k + 1) * sizeb, :] = (
            New_OutMatrix[sizeb * Opt_Traj_Vec[k]:
                          sizeb * (Opt_Traj_Vec[k] + 1), :])
        OptOutVec[k * sizeb:(k + 1) * sizeb, :] = (
            New_OutFact[sizeb * Opt_Traj_Vec[k]:
                        sizeb * (Opt_Traj_Vec[k] + 1), :])

    # ----------------------------------------------------------------------
    # Compute values in the original intervals
    # Optmatrix has values x(i,j) in [0, 1/(p -1), 2/(p -1), ... , 1].
    # To obtain values in the original intervals [lb, ub] we compute
    # lb(j) + x(i,j)*(ub(j)-lb(j))
    if Diagnostic:
        OptMatrix_b = OptMatrix.copy()  # save for plot
    if dist is None:
        OptMatrix = (np.tile(lb, (sizeb * r, 1)) +
                     np.tile(ub - lb, (sizeb * r, 1)) * OptMatrix)
    else:
        for i, dd in enumerate(dist):
            OptMatrix[:, i] = lb[i] + (ub[i] - lb[i]) * OptMatrix[:, i]
            if dd is not None:
                if distparam is None:
                    pars = (0., 1.)
                else:
                    if distparam[i] is None:
                        pars = (0., 1.)
                    else:
                        pars = tuple([ float(k) for k in distparam[i] ])
                OptMatrix[:, i] = dd(*pars).ppf(OptMatrix[:, i])

    if Diagnostic:
        # Clean the trajectories from repetitions and plot the histograms
        hplot = np.zeros((2 * r, NumFact))

        for i in range(NumFact):
            for j in range(r):
                # select the first value of the factor
                hplot[j * 2, i] = OptMatrix_b[j * sizeb, i]

                # search the second value
                for ii in range(1, sizeb):
                    if ( OptMatrix_b[j * sizeb + ii, i] !=
                         OptMatrix_b[j * sizeb, i] ):
                        kk = 1
                        hplot[j * 2 + kk, i] = OptMatrix_b[j * sizeb + ii, i]

        try:  # pragma: no cover
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = 'Arial'  # Arial, Verdana
            mpl.rc('savefig', dpi=300, format='png')
            mpl.rc('font', size=9)
            fig = plt.figure()
            fig.suptitle('New Strategy')
            DimPlots = NumFact // 2 + (1 if NumFact % 2 > 0 else 0)
            for i in range(NumFact):
                ax = fig.add_subplot(DimPlots, 2, i + 1)
                ax.hist(hplot[:, i], p)
            fig.savefig('morris_diag_new_strategy.png', transparent=False,
                        bbox_inches='tight', pad_inches=0.035)
            plt.close(fig)
        except ImportError:  # pragma: no cover
            pass

        # Plot the histogram for the original sampling strategy
        # Select the matrix
        OrigSample = OutMatrix[:r * (sizeb), :]
        Orihplot   = np.zeros((2 * r, NumFact))

        for i in range(NumFact):
            for j in range(r):
                # select the first value of the factor
                Orihplot[j * 2, i] = OrigSample[j * sizeb, i]

                # search the second value
                for ii in range(1, sizeb):
                    if OrigSample[j * sizeb + ii, i] != OrigSample[j *
                                                                   sizeb, i]:
                        kk = 1
                        Orihplot[j * 2 + kk, i] = OrigSample[j * sizeb + ii, i]

        try:  # pragma: no cover
            fig = plt.figure()
            fig.suptitle('Old Strategy')
            DimPlots = NumFact // 2 + (1 if NumFact % 2 > 0 else 0)
            for i in range(NumFact):
                ax = fig.add_subplot(DimPlots, 2, i + 1)
                ax.hist(Orihplot[:, i], p)
                # plt.title('Old Strategy')
            fig.savefig('morris_diag_old_strategy.png', transparent=False,
                        bbox_inches='tight', pad_inches=0.035)
            plt.close(fig)
        except:  # pragma: no cover
            pass

        # Measure the quality of the sampling strategy
        levels = np.arange(0.0, 1.1, 1.0 / (p - 1))
        NumSPoint = np.zeros((NumFact, p))
        NumSOrigPoint = np.zeros((NumFact, p))
        for i in range(NumFact):
            for j in range(p):
                # For each factor and each level count the number of times the
                # factor is on the level
                # This for the new and original sampling
                NumSPoint[i, j] = np.where(
                    np.abs(hplot[:, i] - np.tile(levels[j], hplot.shape[0]))
                    < 1e-5)[0].size
                NumSOrigPoint[i, j] = np.where(
                    np.abs(Orihplot[:, i] -
                           np.tile(levels[j], Orihplot.shape[0]))
                    < 1e-5)[0].size

        # The optimal sampling has values uniformly distributed across the
        # levels
        OptSampl       = 2. * r / p
        QualMeasure    = 0.
        QualOriMeasure = 0.
        for i in range(NumFact):
            for j in range(p):
                QualMeasure = QualMeasure + np.abs(NumSPoint[i, j] - OptSampl)
                QualOriMeasure = QualOriMeasure + np.abs(
                    NumSOrigPoint[i, j] - OptSampl)

        QualMeasure = 1. - QualMeasure / (OptSampl * p * NumFact)
        QualOriMeasure = 1. - QualOriMeasure / (OptSampl * p * NumFact)

        print('The quality of the sampling strategy changed from {:f} with '
              'the old strategy to {:f} '
              'for the optimized strategy'.format(QualOriMeasure, QualMeasure))

    return OptMatrix, OptOutVec[:, 0]


def Morris_Measure_Groups(NumFact, Sample, OutFact, Output, p=4,
                          Group=[], Diagnostic=False):
    """
    Given the Morris sample matrix, the output values and the group matrix
    compute the Morris measures.


    Definition
    ----------
    def Morris_Measure_Groups(NumFact, Sample, OutFact, Output, p=4, Group=[],
                              Diagnostic=False):


    Input
    -----
    NumFact
        Number of factors
    Sample
        Matrix of the Morris sampled trajectories
    OutFact
        Matrix with the factor changings as specified in Morris sampling
    Output
        Matrix of the output(s) values in correspondence of each point of each
        trajectory


    Optional Input
    --------------
    p
        Number of levels
    Group
        [NumFact, NumGroups] Matrix describing the groups.
        Each column represents one group. The element of each column are zero
        if the factor is not in the group. Otherwise it is 1.
    Diagnostic
        True:  print out diagnostics

        False: otherwise (default)


    Output
    ------
    SA, OutMatrix
        SA(NumFact*Output.shape[1],N) individual sensitivity measures

        OutMatrix(NumFact*Output.shape[1], 3) = [Mu*, Mu, StDev]
        Morris Measures
        It gives the three measures of each factor for each output.


    References
    ----------
    Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis. Wiley
    Series in Probability and Statistics, John Wiley & Sons, New York, 1-504. -
    on page 68ff

    """
    try:
        NumGroups = Group.shape[1]
        if Diagnostic:
            print('{:d} Groups are used'.format(NumGroups))
    except:  # pragma: no cover
        NumGroups = 0
        if Diagnostic:
            print('No Groups are used')

    Delt = p / (2. * (p - 1.))

    if NumGroups != 0:
        sizea = NumGroups
        GroupMat = Group
        GroupMat = GroupMat.transpose()
        if Diagnostic:
            print('NumGroups', NumGroups)
    else:
        sizea = NumFact
    sizeb = sizea + 1

    # r = Sample.shape[0]/sizeb
    r = Sample.shape[0] // sizeb

    try:
        NumOutp = Output.shape[1]
    except:  # pragma: no cover
        NumOutp = 1
        Output = Output.reshape((Output.size, 1))

    # For each Output
    if NumGroups == 0:
        # for every output: every factor is a line, columns are mu*, mu and std
        OutMatrix = np.zeros((NumOutp * NumFact, 3))
    else:
        # for every output: every factor is a line, column is mu*
        OutMatrix = np.zeros((NumOutp * NumFact, 1))

    SAmeas_out = np.zeros((NumOutp * NumFact, r))

    for k in range(NumOutp):
        OutValues = Output[:, k]

        # For each trajectory
        # vorm afhankelijk maken van group of niet...
        SAmeas = np.zeros((NumFact, r))
        for i in range(r):
            # For each step j in the trajectory
            # Read the orientation matrix fact for the r-th sampling
            # Read the corresponding output values
            # Read the line of changing factors
            Single_Sample = Sample[i * sizeb:(i + 1) * sizeb, :]
            Single_OutValues = OutValues[i * sizeb:(i + 1) * sizeb]
            # gives factor in change (or group)
            Single_Facts = np.array(OutFact[i * sizeb:(i + 1) * sizeb],
                                    dtype=int)

            A = (Single_Sample[1:sizeb, :] -
                 Single_Sample[:sizea, :]).transpose()
            Delta = A[np.where(A)]  # AAN TE PASSEN?
            # If upper bound==lower bound then A==0 in all trajectories. Delta
            # will have not the right dimensions then because these are
            # filtered out with where. Fill in Delta==0 for these cases.
            ii = np.where(np.sum(A, axis=0) == 0.)[0]
            if ii.size > 0:
                Delta = np.insert(Delta, ii, 0.)

            if Diagnostic:
                print('A: ', A)
                print('Delta: ', Delta)
                print('Single_Facts: ', Single_Facts)

            # For each point of the fixed trajectory, i.e. for each factor,
            # compute the values of the Morris function.
            for j in range(sizea):
                if NumGroups != 0:  # work with groups
                    Auxfind = A[:, j]
                    Change_factor = np.where(np.abs(Auxfind) > 1e-010)[0]
                    for gk in Change_factor:
                        SAmeas[gk, i] = np.abs(
                            (Single_OutValues[j] - Single_OutValues[j + 1]) /
                            Delt)  # nog niet volledig goe
                else:
                    if Delta[j] > 0.0:
                        SAmeas[Single_Facts[j], i] = ((Single_OutValues[j+1] -
                                                       Single_OutValues[j]) /
                                                      Delt)
                    else:
                        SAmeas[Single_Facts[j], i] = ((Single_OutValues[j] -
                                                       Single_OutValues[j + 1])
                                                      / Delt)

        # Compute Mu AbsMu and StDev
        if np.isnan(SAmeas).any():
            AbsMu = np.zeros(NumFact)
            Mu = np.zeros(NumFact)
            Stdev = np.zeros(NumFact)

            for j in range(NumFact):
                SAm = SAmeas[j, :]
                SAm = SAm[~np.isnan(SAm)]
                rr = float(SAm.size)
                AbsMu[j] = np.sum(np.abs(SAm)) / rr
                if NumGroups == 0:
                    Mu[j] = SAm.mean()
                    if SAm.size > 1:
                        Stdev[j] = np.std(SAm, ddof=1)
                    else:
                        Stdev[j] = 0.
        else:
            AbsMu = np.sum(np.abs(SAmeas), axis=1) / r
            if NumGroups == 0:
                Mu = SAmeas.mean(axis=1)
                if SAmeas.shape[1] > 1:
                    Stdev = np.std(SAmeas, ddof=1, axis=1)
                else:
                    Stdev = np.zeros(SAmeas.shape[0])
            else:
                Mu = np.zeros(NumFact)
                Stdev = np.zeros(NumFact)

        OutMatrix[k * NumFact:k * NumFact + NumFact, 0] = AbsMu
        if NumGroups == 0:
            OutMatrix[k * NumFact:k * NumFact + NumFact, 1] = Mu
            OutMatrix[k * NumFact:k * NumFact + NumFact, 2] = Stdev

        SAmeas_out[k * NumFact:k * NumFact + NumFact, :] = SAmeas

    return SAmeas_out, OutMatrix


def morris_sampling(nparam, lb, ub, nt,
                    nsteps=6, ntotal=None,
                    dist=None, distparam=None,
                    GroupMat=np.array([]), Diagnostic=0):
    """
    Sample trajectories in parameter space

    Optimisation in the choice of trajectories for Morris experiment, that
    means elementary effects.

    Parameters
    ----------
    nparam : int
        Number of parameters / factors
    lb : array_like
        (nparam,) Lower bound of the uniform distribution for each parameter /
        factor or lower fraction of percent point function ppf if distribution
        given.
        Be aware that the percent point function *ppf* of most continuous
        distributions is infinite at 0.
    ub : array_like
        (nparam,) Upper bound of the uniform distribution for each parameter /
        factor or upper fraction of percent point function ppf if distribution
        given.
        Be aware that the percent point function *ppf* of most continuous
        distributions is infinite at 1.
    nt : int
        Final number of optimal trajectories
    nsteps : int, optional
        Number of levels, i.e. intervals in trajectories (default: 6)
    ntotal : int, optional
        Total number of sampled trajectories. If None: `ntotal=10*nt` (default:
        None)
    dist : list, optional
        List of None or scipy.stats distribution objects for each factor having
        the method ppf, Percent Point Function (Inverse of CDF) (default: None)
        If None, the uniform distribution will be sampled from lower bound `lb`
        to upper bound `ub`.
        If `dist` is scipy.stats.uniform, the ppf will be sampled from the
        lower fraction given in `lb` and the upper fraction in `ub`. The
        sampling interval is then given by the parameters `loc=lower` and
        `scale=interval=upper-lower` in `distparam`. This means
        `dist=None`, `lb=a`, `ub=b`
        corresponds to
        `lb=0`, `ub=1`, `dist=scipy.stats.uniform`, `distparam=[a,b-a]`
    distparam : list, optional
        List with tuples with parameters as required for `dist` (default:
        (0,1)).
        All distributions of scipy.stats have location and scale parameters, at
        least. `loc` and `scale` are implemented as keyword arguments in
        scipy.stats. Other parameters such as the shape parameter of the gamma
        distribution must hence be given first, e.g. `(shape,loc,scale)` for
        the gamma distribution.
        `distparam` is ignored if `dist` is None.
        The percent point function ppf is called like this:
        `dist(*distparam).ppf(x)`
    GroupMat : ndarray, optional
        (nparam,ngroup) Matrix describing the groups. (default: np.array([]))
        Each column represents a group. The elements of each column are zero
        if the parameter / factor is not in the group, otherwise it is 1.
    Diagnostic : bool, optional
        Plot the histograms and compute the efficiency of the sampling if True
        (default: False)

    Returns
    -------
    traj : list
        list [OptMatrix, OptOutVec] with `OptMatrix((nparam+1)*nt,nparam)` and
        `OptOutVec(nparam*nt)`

    References
    ----------
    Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis. Wiley
    Series in Probability and Statistics, John Wiley & Sons, New York, 1-504. -
    on page 68ff

    Examples
    --------
    >>> import numpy as np
    >>> seed = 1023
    >>> np.random.seed(seed=seed)
    >>> npara = 10
    >>> x0    = np.ones(npara)*0.5
    >>> lb    = np.zeros(npara)
    >>> ub    = np.ones(npara)
    >>> mask  = np.ones(npara, dtype=bool)
    >>> mask[5::2] = False
    >>> nmask = np.sum(mask)
    >>> nt     = npara
    >>> ntotal = max(nt**2, 10*nt)
    >>> nsteps = 6
    >>> tmatrix, tvec = morris_sampling(nmask, lb[mask], ub[mask], nt,
    ...                                 nsteps=nsteps, ntotal=ntotal,
    ...                                 Diagnostic=False)
    >>> # Set input vector to trajectories and masked elements to x0
    >>> x = np.tile(x0, tvec.size).reshape(tvec.size, npara)  # default to x0
    >>> x[:,mask] = tmatrix  # replaced unmasked with trajectorie values
    >>> print(x[0,:])
    [0.6 0.4 0.8 0.6 0.6 0.5 0.4 0.5 0.  0.5]

    >>> import scipy.stats as stats
    >>> seed = 1023
    >>> np.random.seed(seed=seed)
    >>> npara = 10
    >>> x0    = np.ones(npara)*0.5
    >>> lb    = np.zeros(npara)
    >>> ub    = np.ones(npara)
    >>> dist  = [ stats.uniform for i in range(npara) ]
    >>> distparam = [ (lb[i], ub[i]-lb[i]) for i in range(npara) ]
    >>> mask  = np.ones(npara, dtype=bool)
    >>> mask[5::2] = False
    >>> nmask = np.sum(mask)
    >>> nt     = npara
    >>> ntotal = max(nt**2, 10*nt)
    >>> nsteps = 6
    >>> tmatrix, tvec = morris_sampling(nmask, lb[mask], ub[mask], nt,
    ...                                 nsteps=nsteps, ntotal=ntotal,
    ...                                 dist=dist, distparam=distparam,
    ...                                 Diagnostic=False)
    >>> # Set input vector to trajectories and masked elements to x0
    >>> x = np.tile(x0, tvec.size).reshape(tvec.size, npara)  # default to x0
    >>> x[:,mask] = tmatrix  # replaced unmasked with trajectory values
    >>> print(x[0,:])
    [0.6 0.4 0.8 0.6 0.6 0.5 0.4 0.5 0.  0.5]

    """
    return Optimized_Groups(nparam, lb, ub, nt,
                            p=nsteps, N=ntotal,
                            dist=dist, distparam=distparam,
                            GroupMat=GroupMat, Diagnostic=Diagnostic)


def elementary_effects(nparam, OptMatrix, OptOutVec, Output,
                       nsteps=4, Group=[], Diagnostic=False):
    """
    Compute the Morris measures given the Morris sample matrix, the output
    values and the group matrix.

    Parameters
    ----------
    nparam : int
        Number of parameters / factors list [OptMatrix, OptOutVec] with
        OptMatrix((nparam+1)*nt,nparam) and OptOutVec(nparam*nt)
    OptMatrix : ndarray
        ((nparam+1)*nt,nparam) Matrix of the Morris sampled trajectories from
        morris_sampling
    OptOutVec : ndarray
        (nparam*nt,) Matrix with the parameter / factor changings from
        morris_sampling
    Output : ndarray
        ((nparam+1)*nt,) Matrix of the output values of each point of each
        trajectory
    nsteps : int, optional
        Number of levels, i.e. intervals in trajectories (default: 4)
    Group : ndarray, optional
        (nparam,NumGroups) Matrix describing the groups. (default: [])
        Each column represents a group. The elements of each column are zero
        if the parameter / factor is not in the group, otherwise it is 1.
    Diagnostic : bool, optional
        Print out diagnostics if True (default: False)

    Returns
    -------
    SA, OutMatrix : list of ndarrays
        SA(nparam*Output.shape[1],N) individual sensitivity measures,
        OutMatrix(nparam*Output.shape[1], 3) = [Mu*, Mu, StDev] Morris Measures

        It gives the three measures of each parameter / factor for each output.

    References
    ----------
    Saltelli, A., Chan, K., & Scott, E. M. (2000). Sensitivity Analysis. Wiley
    Series in Probability and Statistics, John Wiley & Sons, New York, 1-504. -
    on page 68ff

    Examples
    --------
    >>> import numpy as np
    >>> seed = 1023
    >>> np.random.seed(seed=seed)
    >>> npara = 10
    >>> x0    = np.ones(npara)*0.5
    >>> lb    = np.zeros(npara)
    >>> ub    = np.ones(npara)
    >>> mask  = np.ones(npara, dtype=bool)
    >>> mask[5::2] = False
    >>> nmask = np.sum(mask)
    >>> nt     = npara
    >>> ntotal = max(nt**2, 10*nt)
    >>> nsteps = 6
    >>> tmatrix, tvec = morris_sampling(nmask, lb[mask], ub[mask], nt,
    ...                                 ntotal=ntotal, nsteps=nsteps,
    ...                                 Diagnostic=False)
    >>> # Set input vector to trajectories and masked elements to x0
    >>> x = np.tile(x0, tvec.size).reshape(tvec.size, npara)  # default to x0
    >>> x[:,mask] = tmatrix  # replaced unmasked with trajectorie values
    >>> func = np.sum
    >>> fx = np.array(list(map(func,x)))
    >>> out = np.zeros((npara,3))
    >>> sa, res = elementary_effects(nmask, tmatrix, tvec, fx, nsteps=nsteps,
    ...                              Diagnostic=False)
    >>> out[mask,:] = res
    >>> print(out[:,0])
    [1. 1. 1. 1. 1. 0. 1. 0. 1. 0.]

    """
    return Morris_Measure_Groups(nparam, OptMatrix, OptOutVec,
                                 Output, nsteps, Group, Diagnostic)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    # # nt = 10
    # print('nt=10')
    # np.random.seed(seed=1234)
    # nparam = 15
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5,1],3))  #  (nparam,r) individual elementary effects for all parameters

    # # nt = 10, nan
    # print('nt=10, nan')
    # np.random.seed(seed=1234)
    # nparam = 15
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # out[1:nt*nparam:nparam//2] = np.nan
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[~np.isnan(sa[:,1]),1],3))  #  (nparam,r) individual elementary effects for all parameters

    # # nt = 1
    # print('nt=1')
    # np.random.seed(seed=1234)
    # nparam = 15
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # nt     = 1
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=Diagnostic)
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5].squeeze(),3))  #  (nparam,r) individual elementary effects for all parameters

    # # groups
    # print('groups')
    # np.random.seed(seed=1234)
    # nparam   = 15
    # ngroup = 5
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # Groups = np.random.randint(0, 4, (nparam,ngroup))
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, GroupMat=Groups, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps, Group=Groups)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5,1],3))  #  (nparam,r) individual elementary effects for all parameters

    # # groups, nan
    # print('groups, nan')
    # np.random.seed(seed=1234)
    # nparam   = 15
    # ngroup = 5
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # Groups = np.random.randint(0, 4, (nparam,ngroup))
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # out[1:nt*nparam:nparam//2] = np.nan
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, GroupMat=Groups, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps, Group=Groups)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5,1],3))  #  (nparam,r) individual elementary effects for all parameters

    # # nt=10, dist
    # print('nt=10, distributions')
    # import scipy.stats as stats
    # np.random.seed(seed=1234)
    # nparam = 15
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # lb = np.zeros(nparam)
    # ub = np.ones(nparam)
    # dist      = [ stats.uniform for i in range(nparam) ]
    # distparam = [ (lb[i],ub[i]-lb[i]) for i in range(nparam) ]
    # lb = np.zeros(nparam)
    # ub = np.ones(nparam)
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5,1],3))  #  (nparam,r) individual elementary effects for all parameters

    # # groups, dist
    # print('groups, distributions')
    # np.random.seed(seed=1234)
    # nparam   = 15
    # ngroup = 5
    # lb = np.arange(nparam)
    # ub = 2.*lb + 1.
    # dist      = [ stats.uniform for i in range(nparam) ]
    # distparam = [ (lb[i],ub[i]-lb[i]) for i in range(nparam) ]
    # lb = np.zeros(nparam)
    # ub = np.ones(nparam)
    # Groups = np.random.randint(0, 4, (nparam,ngroup))
    # nt     = 10
    # nsteps = 6
    # ntotal = 100
    # Diagnostic = 0
    # out = np.random.random(nt*(nparam+1))
    # mat, vec = morris_sampling(nparam, lb, ub, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam, GroupMat=Groups, Diagnostic=Diagnostic)
    # print(np.around(mat[0,0:5],3))
    # print(np.around(vec[0:5],3))
    # sa, res = elementary_effects(nparam, mat, vec, out, nsteps=nsteps, Group=Groups)
    # print(np.around(res[0:5,0],3)) # (nparam,3) = AbsMu, Mu, Stddev
    # print(np.around(sa[0:5,1],3))  #  (nparam,r) individual elementary effects for all parameters
