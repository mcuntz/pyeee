User Guide
==========

``pyeee`` is a Python library for performing parameter screening of
computational models. It uses Morris' method of Elementary Effects and
its extension called Efficient or Sequential Elementary Effects
published by:

Cuntz, Mai `et al.` (2015) Computationally inexpensive identification
of noninformative model parameters by sequential screening,
`Water Resources Research` 51, 6417-6441, doi:
`10.1002/2015WR016907`_.

The numerical models must be callable as `func(x)`. Use
:func:`functools.partial` from Python's standard library to make any
function callable as `func(x)`. One can use the package
:mod:`partialwrap` to use external programs with
:func:`functools.partial` and hence ``pyeee``.


Elementary Effects
------------------

Consider the Ishigami-Homma function:
:math:`y = \sin(x_0) + a \sin(x_1)^2 + b x_2^4 \sin(x_0)`.

Taking :math:`a = b = 1` gives:

.. code-block:: python

   import numpy as np

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

The three parameters :math:`x_0, x_1, x_2` follow uniform
distributions between :math:`-\pi` and :math:`+\pi`.

Morris' Elementary Effects can then be calculated, using 20
trajectories, as:

.. code-block:: python

   from pyeee import screening

   # function
   func  = ishigami1
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # Elementary Effects with 20 trajectories
   np.random.seed(seed=1023)  # for reproducibility of examples
   out = screening(func, lb, ub, 20)

   # mu*
   print("{:.1f} {:.1f} {:.1f}".format(*out[:, 0]))
   # gives: 72.6 0.6 14.3

:func:`~pyeee.screening.screening` returns a `(npars, 3)` ndarray
with:

   1. `(npars, 0)` the means of the absolute elementary effects over
      all trajectories (:math:`\mu*`)
   2. `(npars, 1)` the means of the elementary effects over all nt
      trajectories (:math:`\mu`)
   3. `(npars, 2)` the standard deviations of the elementary effects
      over all trajectories (:math:`\sigma`)

For Elementary Effects and its sensititvity measures, see
https://en.wikipedia.org/wiki/Elementary_effects_method, or

Saltelli *et al.* (2007) Global Sensitivity Analysis. The Primer,
John Wiley & Sons Ltd, Chichester, UK, ISBN: 978-0470-059-975,
doi: `10.1002/9780470725184`_.

The numerical model `func`, lower parameter boundaries `lb`, upper
parameter boundaries `ub`, and the number of trajectories `nt` are
mandatory arguments to :func:`~pyeee.screening.screening` (or the
identical function :func:`~pyeee.screening.ee` ;-).

Further optional arguments relevant to Elementary Effects are:

   - `nsteps`: int - Number of steps along one trajectory (default: 6)
   - `ntotal`: int - Total number of trajectories to check for the
     `nt` most different trajectories (default: `max(nt**2, 10 * nt)`)

Note that :func:`~pyeee.screening.screening` uses the functions
:func:`~pyeee.morris_method.morris_sampling` and
:func:`~pyeee.morris_method.elementary_effects`, which are the
implementations of Francesca Campolongo and Jessica Cariboni written
in Matlab and translated to Python by Stijn Van Hoey. They support the
notion of parameter groups, which is not taken into account in
:func:`~pyeee.screening.screening`.


Efficient/Sequential Elementary Effects
---------------------------------------

Morris' method of Elementary Effects is not a full sensitivity
analysis. The sensititvity measures of Elementary Effects are rather
used for preliminary screening for noninformative model parameters for
a given model output, so that fewer parameters are needed during a
full sensitivity analysis or during model optimisation.

The numerical model `func` will be evaluated `nt * (npars + 1)` times
for calculating Elementary Effects. The user chooses the number of
trajectories `nt`. A large number of `nt` might be computationally
expensive and a small number might miss areas of the parameter space,
where certain parameters become sensitive. Typical values for `nt` in
the literature are on the order of tens to hundreds. This means that
the method of Elementary Effects needs between 500 and 5000 model
evaluations for a model with 50 parameters.

The extension of Efficient or Sequential Elementary Effects can be
used if one uses Elementary Effects `only` to distinguish between
sensitive (informative) and insensitive (noninformative) model
parameters. It follows the idea: if one knows that a model is
sensitive to a certain parameter, this parameter does not have to be
included anymore in further screening. If a parameter has a large
Elementary Effect in one trajectory it will most probably be
influential. So one does not have to calculate another Elementary
Effect for this parameter and it can be discarded from further
trajectories, saving model evaluations.

The method starts hence with a limited number of trajectories
`ntfirst` for all model parameters, i.e. it performs
`ntfirst * (npars + 1)` model evaluations. Further trajectories are
sampled, calculating Elementary Effects, but without the parameters
that were already found sensitive. This means that subsequent
trajectories need less and less function evaluations. The algorithm
ends if a subsequent trajectory did not yield any sensitive parameters
anymore. A last `ntlast` trajectories are finally sampled, and
Elementary Effects calculated, to assure a large sample for parameters
with little sensitivity, to minimize the possibility that the
parameters are sensitive in a small part of the parameter space, which
was missed due to a small sample.

The call of :func:`~pyeee.eee.eee` (or the identical function
:func:`~pyeee.eee.see`) is very similar to standard Elementary effects
:func:`~pyeee.screening.screening` (or the identical function
:func:`~pyeee.screening.ee` ;-):

.. code-block:: python

   import numpy as np
   from pyeee import eee

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   func  = ishigami1
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # Sequential Elementary Effects
   np.random.seed(seed=1025)  # for reproducibility of examples
   out = eee(func, lb, ub, ntfirst=10)

   print(out)
   # gives: [ True False  True]

:func:`~pyeee.eee.eee` returns an include-mask, being `True` for
sensitive parameters and `False` for noninformative parameters. The
mask can be combined by `logical_and` with an incoming mask.


Check initial fit
^^^^^^^^^^^^^^^^^

Efficient/Sequential Elementary Effects fit a logistic function to
the output of the `ntfirst` trajectories, which determines the
threshold between informative and uninformative parameters for the
following (shorter) trajectories. One can check this initial,
important step by passing the name of an output file to
:func:`~pyeee.eee.eee` with the keyword `plotfile`:

.. code-block:: python

   out = eee(func, lb, ub, ntfirst=10, plotfile='ishigami.png')

Note that :mod:`matplotlib` must be installed to produce the
`plotfile`. The file format of `plotfile` is always `png` independent
on the file name.


Logging
^^^^^^^

Following the same idea, the user can also log progress and
intermediate results of :func:`~pyeee.eee.eee` in a text file giving
the `logfile` keyword:

.. code-block:: python

   out = eee(func, lb, ub, ntfirst=10, plotfile='ishigami.png',
             logfile='ishigami.log')


Advanced usage
--------------

Exclude parameters from calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~pyeee.screening.screening` and :func:`~pyeee.eee.eee` offer
the possibility to mask some model parameters so that they will not be
changed during calculation of Elementary Effects. Inital values `x0`
must be given that will be taken where `mask == False`, i.e. `mask`
could be called an include-mask (opposite of the exclude-mask of
numpy's masked arrays). Note that the size of `x0` must be the size of
`lb`, `ub` and `mask`, i.e. one has to give initial values even if an
element is included in the screening, which means `mask[i] == True`.

For example, if one wants to exclude the second parameter :math:`x_0`
of the above Ishigami-Homma function in the calculation of the
Elementary Effects:

.. code-block:: python

   # function
   mask    = np.ones(npars, dtype=bool)  # True  -> include
   mask[0] = False                       # False -> exclude

   # initial values
   x0 = np.ones(npars) * 0.5

   # Efficient Elementary Effects
   np.random.seed(seed=1024)  # for reproducibility of examples

   out = eee(func, lb, ub, x0=x0, mask=mask)
   print(out)
   # gives: [False False  True]

   mask = mask & out
   print(mask)
   # gives: [False False  True]

   # Elementary Effects
   out = screening(func, lb, ub, 20, x0=x0, mask=mask)
   print("{:.1f} {:.1f} {:.1f}".format(*out[:, 0]))
   # 0.0 0.0 62.2


Function with multiple outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The numerical model `func` might return several outputs per model run,
e.g. returning a time series. The Morris' sensitivity measures are
calculated hence for each output, i.e. for each point in time in a
time series. Efficient/Sequential Elementary Effects
:func:`~pyeee.eee.eee` can either take the arithmetic mean of all
:math:`\mu*` or a weighted mean :math:`\mu*`, weighted by
:math:`\sigma`.

The keyword `weight=False` is appropriate if each single output is
equally important. An example is river runoff where one might be
interested in both, high flows such as floods and low flows indicating
droughts.

An example for `weight=True` are fluxes to and from the atmosphere
such as evapotranspiration. The atmosphere is more strongly influenced
by larger fluxes so that sensitivity measures during periods of little
atmospheric exchange are less interesting. `Cuntz, Mai et al.`_ (2015)
argued that weighting by the standard deviation :math:`\sigma` is
equivalent to flux weighting because parameter variations yield larger
variances for larger fluxes than they yield for smaller fluxes in most
computer models.


Parallel model evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

The numerical model `func` will be evaluated `npars + 1` times for
each trajectory, with `npars` the number of parameters of the
computational model. Multiprocessing can be used for parallel function
evaluation. Setting `processes=nprocs` evaluates `nprocs` parameter
sets in parallel:

.. code-block:: python

   # Elementary Effects using 4 parallel processes
   out = screening(func, lb, ub, processes=4)

``pyeee`` uses the package :mod:`schwimmbad` for
parallelisation. :mod:`schwimmbad` provides a uniform interface to
parallel processing pools and enables switching easily between local
development (e.g. serial processing or :mod:`multiprocessing`) and
deployment on a cluster or supercomputer (e.g. via MPI or JobLib).

Consider the following Python code in a script (e.g. `eeetest.py`):

.. code-block:: python

   # File: eeetest.py
   import sys
   import numpy as np
   from pyeee import eee
   import schwimmbad

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # get number of processes
   if len(sys.argv) > 1:
       nprocs = int(sys.argv[1])
   else:
       nprocs = 1

   # mpi4py is an optional dependency of pyeee
   try:
      from mpi4py import MPI
      comm  = MPI.COMM_WORLD
      csize = comm.Get_size()
      crank = comm.Get_rank()
      if csize > 1:
          nprocs = csize
   except ImportError:
      comm  = None
      csize = 1
      crank = 0

   # function
   func  = ishigami1
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # choose the serial or parallel pool
   ipool = schwimmbad.choose_pool(mpi=False if csize==1 else True,
                                  processes=nprocs)

   # Elementary Effects
   np.random.seed(seed=1023)  # for reproducibility of examples
   out = eee(func, lb, ub, processes=nprocs, pool=ipool)

   if crank == 0:
       print(out)
   ipool.close()

The user gives the number of processors to use on the command line
(`ncpus`).
   
This script can be run in normal serial mode, i.e. all function
evaluations are done one after the other:

.. code-block:: bash

   python eeetest.py

One can give explicitly that the script should run one core only:

.. code-block:: bash

   python eeetest.py 1

Or it can use Python's :mod:`multiprocessing` module, e.g. with 4
parallel processes:

.. code-block:: bash

   python eeetest.py 4

or use the Message Passing Interface (MPI), e.g. with 4 parallel
processes:

.. code-block:: bash

   mpiexec -n 4 python eeetest.py

Note that :mod:`mpi4py` must be installed for the last example.


Sampling parameters with other distributions than the uniform distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Morris' method of Elementary Effects samples parameters along
trajectories through the possible parameter space. It assumes
uniformly distributed parameters between a lower bound and an upper
bound.

This implementation of Morris' Elementary Effects
:func:`~pyeee.screening.screening` allows sampling parameters from
other distributions than uniform distributions. For example, a
parameter :math:`p` might have been determined by repeated
experiments. One can hence determine the mean parameter
:math:`\overline{p}` and calculate the error of the mean
:math:`\epsilon_p`. This error of the mean is actually the standard
deviation of the distribution of the mean. One would thus sample a
normal distribution with mean :math:`\overline{p}` and a standard
deviation :math:`\epsilon_p` for the parameter :math:`p` to calculate
Morris' Elementary Effects.

:func:`~pyeee.screening.screening` allows all distributions of
mod:`scipy.stats`, given with the keyword `dist`. The parameters of
the distributions are given as a list of tuples with the keyword
`distparam`. The lower and upper bounds change their meaning if `dist`
is given for a parameter: :func:`~pyeee.screening.screening` samples
uniformly the Percent Point Function (ppf) of the distribution between
lower and upper bound. The percent point function is the inverse of
the Cumulative Distribution Function (cdf). Lower and upper bounds
must hence be between `0` and `1`. Note the percent point functions of
most continuous distributions will be infinite at the limits `0` and
`1`.

The three parameters :math:`x_0, x_1, x_2` of the Ishigami-Homma
function follow uniform distributions between :math:`-\pi` and
:math:`+\pi`. Say that :math:`x_1` follows a Gaussian distribution
around the mean :math:`0` with a standard deviation of
:math:`1.81`. We want to sample between plus or minus three standard
deviations, which includes about 99.7\% of the total
distribution. This means that the lower bound would be 0.0015
(0.003/2.) and the upper bound 0.9985.

.. code-block:: python

   import scipy.stats as stats
   dist      = [None, stats.norm, stats.uniform]
   distparam = [None, (0., 1.81), (-np.pi, 2.*np.pi)]
   lb        = [-np.pi, 0.0015, 0.]
   ub        = [np.pi, 0.9985, 1.]

   out = screening(func, lb, ub, 20, dist=dist, distparam=distparam)

This example demonstrates that

   1. if `dist` is passed, one has to give a distribution for each
      parameter;
   2. distributions are given as :mod:`scipy.stats` distribution
      objects;
   3. if `dist` is None, :func:`~pyeee.screening.screening` assumes a
      uniform distribution and samples between lower and upper bound;
   4. (almost) all :mod:`scipy.stats` distributions take the keywords
      `loc` and `scale`. Their meaning is *NOT* mean and standard
      deviation in most distributions. For the uniform distribution
      :any:`scipy.stats.uniform`, `loc` is the lower limit and `loc +
      scale` the upper limit. This means the combination `dist=None`,
      `distparam=None`, `lb=a`, `ub=b` corresponds to
      `dist=scipy.stats.uniform`, `distparam=[a, b-a]`, `lb=0`,
      `ub=1`.

Note also that

   5. if `distparam` is None, `loc=0` and `scale=1` will be taken;
   6. `loc` and `scale` are implemented as keywords in
      :mod:`scipy.stats`. Other parameters such as for example the
      shape parameter of the gamma distribution
      :any:`scipy.stats.gamma` must hence be given first,
      i.e. `(shape, loc, scale)`.

Remember that Morris' method of Elementary Effects assumes uniformly
distributed parameters and that other distributions are an extension
of the original method.

:func:`~pyeee.eee.eee` uses :func:`~pyeee.screening.screening`
internally. It consequently also offers the possibility to sample
other distributions than uniform distributions with the keywords
`dist` and `distparams`.

.. code-block:: python

   out = eee(func, lb, ub, ntfirst=10, dist=dist, distparam=distparam)


Python function with extra parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function for :func:`~pyeee.sreening.sreening` and
:func:`~pyeee.eee.eee` must be of the form `func(x)`. Use Python's
:func:`functools.partial` from the Python module :mod:`functools` to
pass other function parameters.

For example pass the parameters :math:`a` and :math:`b` to the
Ishigami-Homma function. One needs a wrapper function that takes the
function and its parameters as arguments. The variable parameters of
the screening must be the last argument, i.e. it must be `x` of
`func(x)`:

.. code:: python

   from functools import partial

   def ishigami(x, a, b):
      return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

   # x has to be the last argument
   def call_ishigami(func, a, b, x):
      return func(x, a, b)

   # Partialise function with fixed parameters
   a = 0.5
   b = 2.0
   func  = partial(call_ishigami, ishigami, a, b)

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   out = eee(func, lb, ub, ntfirst=10)

The parameters :math:`a` and :math:`b` are fixed parameters during
screening. Figuratively speaking, :func:`functools.partial` passes
:math:`a` and :math:`b` to the function `call_ishigami` already during
definition. :func:`~pyeee.eee.eee` can then simply call it as
`func(x)`, where `x` is passed to `call_ishigami` then as well. This
"finishes" the call of `call_ishigami` and `x`, `a` and `b` are passed
to `ishigami`.


Screening of external computer models
-------------------------------------

**Note: this section is pretty much a repetition of the** `User
Guide`_ **of** :mod:`partialwrap`, **which itself is not limited to be
used with** ``pyeee`` **but can be used with any package that calls
functions in the form** `func(x)`. **The finer notions of**
:mod:`partialwrap` **might be better explained in its** `User Guide`_.

``pyeee`` can be used to screen parameters from external computer
models written in any (compiled) language such as C, Fortran or
similar. We use our package :mod:`partialwrap` for this.
:mod:`partialwrap` provides wrapper functions that basically launch
external executables using Python's :mod:`subprocess` module, while
providing functionality to write parameter files for the external
executables and read in output from the executables in return.

This means that the wrappers of :mod:`partialwrap` need a function
`parameterwriter` that writes the parameters in the parameter file(s)
`parameterfile`. The wrappers also need to read model output from
`outputfile` with the function `outputreader`. The latter can also do
further calculations such as calculating an objective function from
the model output.

Take an external program that calculates the Ishigami-Homma function
with :math:`a = b = 1`, reading in the parameters :math:`x_0, x_1,
x_2` from a `parameterfile = params.txt` and writing its output into
an `outputfile = out.txt`. Take for simplicity a Python program first
(e.g. `ishigami1.py`):

.. code-block:: python

   # File: ishigami1.py
   import numpy as np

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # read parameters
   x = np.loadtxt('params.txt')

   # calc function
   y = ishigami1(x)

   # write output file
   np.savetxt('out.txt', y)

The external program, which is in full `python3 ishigami1.py`, can be
used with the wrapper function
:func:`~partialwrap.wrappers.exe_wrapper` of :mod:`partialwrap`:

.. code-block:: python

   from functools import partial
   import numpy as np
   import scipy.optimize as opt
   from partialwrap import exe_wrapper
   from pyeee import eee
        
   ishigami1_exe   = ['python3', 'ishigami1.py']
   parameterfile   = 'params.txt'
   parameterwriter = np.savetxt
   outputfile      = 'out.txt'
   outputreader    = np.loadtxt
   ishigami1_wrap  = partial(exe_wrapper, ishigami1_exe,
                             parameterfile, parameterwriter,
                             outputfile, outputreader, {})

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   out = eee(ishigami1_wrap, lb, ub, ntfirst=10)

The empty dictionary at the end of the partial statement is explained
below.

One can see that the external Ishigami-Homma program could have been
written in a compiled language such as C, Fortran or similar, and then
used with ``pyeee``. A Fortran program could look like this:

.. code-block:: fortran

   program ishigami1

       implicit none

       integer, parameter :: dp = kind(1.0d0)

       character(len=*), parameter :: pfile = 'params.txt'
       character(len=*), parameter :: ofile = 'out.txt'

       integer, parameter :: punit = 99
       integer, parameter :: ounit = 101

       real(dp), dimension(3) :: x ! parameters x_0, x_1, x_2
       real(dp) :: out             ! output value
       integer  :: n

       integer  :: ios

       ! read parameters
       open(punit, file=pfile, status='old', action='read')
       ios = 0
       n = 1
       do while (ios==0)
           read(punit, fmt=*, iostat=ios) x(n)
           n = n + 1
       end do
       n = n - 2
       close(punit)

       ! calc function
       out = sin(x(1)) + sin(x(2))**2 + x(3)**4 * sin(x(1))

       ! write output file
       open(ounit, file=ofile)
       write(ounit,*) out
       close(ounit)

   end program ishigami1

This program can be compiled like:

.. code-block:: bash

   gfortran -o ishigami1.exe ishigami1.f90

and used in Python:

.. code-block:: python

   from functools import partial
   import numpy as np
   import scipy.optimize as opt
   from partialwrap import exe_wrapper
   from pyeee import eee
        
   ishigami1_exe   = ['ishigami1.exe']
   parameterfile   = 'params.txt'
   parameterwriter = np.savetxt
   outputfile      = 'out.txt'
   outputreader    = np.loadtxt
   ishigami1_wrap  = partial(exe_wrapper, ishigami1_exe,
                             parameterfile, parameterwriter,
                             outputfile, outputreader, {})

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   out = eee(ishigami1_wrap, lb, ub, ntfirst=10)

Where the only difference to the Python version is that
`ishigami1_exe = ['./ishigami1.exe']` instead of
`ishigami1_exe = ['python3', 'ishigami1.py']`.


Parallel evaluation of external executables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elementary Effects run the computational model `nt * (npars + 1)`
times. All model runs are independent and can be executated at the
same time if computing ressources permit. Even simple personal
computers have several computing cores nowadays.

However, using 4 worker with `processes=4`, for example, writes 4
times `parameterfile = params.txt` thus overwriting itself. Here the
`pid` keyword of :mod:`partialwrap` comes in handy. Each invocation
would have its own random number `pid` associated, writing
`parameterfile.pid` and reading `outfile.pid`. The Ishigami-Homma
program would need to be changed to (the Python version here):

.. code-block:: python

   # File: ishigami1_pid.py
   import numpy as np
   from partialwrap import standard_parameter_reader, standard_parameter_writer

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # get pid
   if len(sys.argv) > 1:
       pid = int(sys.argv[1])
   else:
       pid = None

   # read parameters
   x = standard_parameter_reader('params.txt', pid=pid)

   # calc function
   y = ishigami1(x)

   # write output file
   standard_parameter_writer('out.txt', pid=pid)

:func:`~partialwrap.std_io.standard_parameter_reader` and
:func:`~partialwrap.std_io.standard_parameter_writer` are convenience
functions that reads and writes parameters from a file just like
:func:`numpy.loadtxt` and :func:`numpy.savetxt`. The difference the
functions support the `pid` keyword. If `True`,
:func:`~partialwrap.std_io.standard_parameter_reader` reads from files
such as `params.txt.158398716` rather than from `params.txt`. To achieve this, the
`pid` keyword simply has to be set to *True* in the call of `partial`:

.. code-block:: python

   from functools import partial
   import numpy as np
   import scipy.optimize as opt
   from partialwrap import exe_wrapper
   from partialwrap import standard_parameter_reader, standard_parameter_writer
   from pyeee import eee
        
   ishigami1_exe   = ['python3', 'ishigami1.py']
   parameterfile   = 'params.txt'
   parameterwriter = standard_parameter_writer
   outputfile      = 'out.txt'
   outputreader    = standard_parameter_reader
   ishigami1_wrap  = partial(exe_wrapper, ishigami1_exe,
                             parameterfile, parameterwriter,
                             outputfile, outputreader,
			     {'pid': True})

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   out = eee(ishigami1_wrap, lb, ub, ntfirst=10. processes=4)


Using launch scripts
^^^^^^^^^^^^^^^^^^^^

If you cannot change your computational model to deal with `pid`, you
can use, for example, a `bash` script or a Python script that launches
each model run in a separate directory. A bash script would be
appropriate on Linux, of course, but a Python script work on Windows
as well. Here we give a Python script as an example but look at the
`User Guide`_ of :mod:`partialwrap` for an example of a `bash` script:

.. code-block:: python

   # File: run_ishigami1.py
   import os
   import shutil
   import subprocess
   import sys

   # get pid
   if len(sys.argv) > 1:
       pid = sys.argv[1]
   else:
       pid = None

   exe   = 'ishigami1.py'
   pfile = 'params.txt'
   ofile = 'out.txt'

   # make individual run directory
   if pid is None:
       rundir = 'tmp'
   else:
       rundir = f'tmp.{pid}'
   os.mkdir(rundir)

   # copy individual parameter file
   if pid is None:
       os.rename(f'{pfile}', f'{rundir}/{pfile}')
   else:
       os.rename(f'{pfile}.{pid}', f'{rundir}/{pfile}')

   # run in individual directory
   shutil.copyfile(exe, f'{rundir}/{exe}')
   os.chdir(rundir)
   err = subprocess.check_output(['python3', exe],
                                 stderr=subprocess.STDOUT)

   # make output available to exe_wrapper
   if pid is None:
       os.rename(ofile, f'../{ofile}')
   else:
       os.rename(ofile, f'../{ofile}.{pid}')

   # clean up
   os.chdir('..')
   shutil.rmtree(rundir)

Note: `exe = 'ishigami1.py'` rather than `exe = ishigami1_pid.py` here
because this example assumes that the executable cannot account for
the `pid` keyword. This Python script can be used with ``pyeee``
exactly like all the scripts above:

.. code-block:: python

   from functools import partial
   import numpy as np
   import scipy.optimize as opt
   from partialwrap import exe_wrapper
   from partialwrap import standard_parameter_reader, standard_parameter_writer
   from pyeee import eee
        
   ishigami1_exe   = ['python3', 'run_ishigami1.py']
   parameterfile   = 'params.txt'
   parameterwriter = standard_parameter_writer
   outputfile      = 'out.txt'
   outputreader    = standard_parameter_reader
   ishigami1_wrap  = partial(exe_wrapper, ishigami1_exe,
                             parameterfile, parameterwriter,
                             outputfile, outputreader,
			     {'pid': True})

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   out = eee(ishigami1_wrap, lb, ub, ntfirst=10. processes=4)

That's all Folks!


.. _10.1002/2015WR016907: http://doi.org/10.1002/2015WR016907
.. _10.1002/9780470725184: http://doi.org/10.1002/9780470725184
.. _Cuntz, Mai et al.: http://doi.org/10.1002/2015WR016907
.. _LICENSE: https://github.com/mcuntz/pyeee/LICENSE
.. _Sebastian Müller: https://github.com/MuellerSeb
.. _template: https://github.com/MuellerSeb/template
.. _User Guide: https://mcuntz.github.io/partialwrap/html/userguide.html
