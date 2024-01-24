**********
User Guide
**********

``pyeee`` is a Python library for performing parameter screening of computational models. It uses
Morris' method of Elementary Effects (*EE*) and also its extension of Efficient or Sequential
Elementary Effects (*EEE* or *SEE*) published by:

Cuntz, Mai *et al.* (2015) Computationally inexpensive
identification of noninformative model parameters by sequential
screening, *Water Resources Research* 51, 6417-6441,
doi:`10.1002/2015WR016907 <http://doi.org/10.1002/2015WR016907>`_.

The numerical models are simply passed to functions `ee` and
:func:`~pyeee.eee.eee` to perform Elementary Effects or Efficient/Sequential Elementary Effects,
respectively.

The numerical models must be callable as `func(x)`. Use :func:`functools.partial` from Python's
standard library to make any function callable as `func(x)`. One can use the package
:mod:`partialwrap` to use external programs with :func:`functools.partial` and hence ``pyeee``.

The package uses several functions of the JAMS Python package

   https://github.com/mcuntz/jams_python

The JAMS package and hesseflux are synchronised irregularly.


Elementary Effects
==================

Simple Python functions
-----------------------

Consider the Ishigami-Homma function: :math:`y = \sin(x_0) + a\,\sin(x_1)^2 + b\,x_2^4\sin(x_0)`.

Taking :math:`a = b = 1` gives:

.. code-block:: python

   import numpy as np

   # Ishigami-Homma function a=b=1
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

The three parameters :math:`x_0, x_1, x_2` follow uniform distributions between :math:`-\pi` and
:math:`+\pi`.

Elementary Effects can be calculated, using 20 trajectories, as follows:

.. code-block:: python

   from pyjams import ee

   # function
   func  = ishigami1
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # Elementary Effects
   np.random.seed(seed=1023) # for reproducibility of examples
   out = ee(func, lb, ub, 20)

:func:`ee` returns a `(npars,3)` ndarray with:

   1. (npars,0) the means of the absolute elementary effects over all trajectories (:math:`\mu*`)
   2. (npars,1) the means of the elementary effects over all nt trajectories (:math:`\mu`)
   3. (npars,2) the standard deviations of the elementary effects over all trajectories (:math:`\sigma`)

For Elementary Effects and its sensititvity measures, see
https://en.wikipedia.org/wiki/Elementary_effects_method, or

Saltelli *et al.* (2007)
Global Sensitivity Analysis. The Primer, John Wiley & Sons Ltd,
Chichester, UK, ISBN: 978-0470-059-975, doi:`10.1002/9780470725184
<http://doi.org/10.1002/9780470725184>`_.

.. code-block:: python

   # mu*
   print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
   # gives: 212.4 0.6 102.8

The numerical model `func`, lower parameter boundaries `lb`, upper parameter boundaries `ub`, and
the number of trajectories `nt` are mandatory arguments to `ee`. Further
optional arguments relevant to Elementary Effects are:

   - `nsteps` : int - Number of steps along one trajectory (default: 6)
   - `ntotal` : int - Total number of trajectories to check for the `nt` most
     different trajectories (default: `max(nt**2,10*nt)`)

Note that the functions `ee` and `screening` are identical.


Exclude parameters from calculations
------------------------------------

`ee` offers the possibility to mask some model parameters so that they will
not be changed during calculation of Elementary Effects. Inital values `x0` must be given that will
be taken where `mask==False`, i.e. `mask` could be called an include-mask. Note that the size of
`x0` must be the size of `lb`, `ub` and `mask`, i.e. one has to give initial values even if an
element is included in the screening, which means `mask[i]==True`.

For example, if one wants to exclude the second parameter :math:`x_1` of the above Ishigami-Homma
function in the calculation of the Elementary Effects:

.. code-block:: python

   # function
   mask    = np.ones(npars, dtype=bool) # True  -> include
   mask[1] = False                      # False -> exclude

   # initial values
   x0 = np.ones(npars) * 0.5

   # Elementary Effects
   np.random.seed(seed=1024) # for reproducibility of examples
   out = ee(func, lb, ub, 10, x0=x0, mask=mask, nsteps=8, ntotal=100)

   print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
   # gives: 114.8 0.0 26.6


Parallel model evaluation
-------------------------

The numerical model `func` will be evaluated `nt*(npars+1)` times, with `npars` the number of
parameters of the computational model. Multiprocessing can be used for parallel function
evaluation. Setting `processes=nprocs` evaluates `nprocs` parameter sets in parallel:

.. code-block:: python

   # Elementary Effects using 4 parallel processes
   np.random.seed(seed=1024) # for reproducibility of examples
   out = ee(func, lb, ub, 10, x0=x0, mask=mask, nsteps=8, ntotal=100,
            processes=4)

``pyeee`` uses the package :mod:`schwimmbad` for parallelisation. :mod:`schwimmbad` provides a
uniform interface to parallel processing pools and enables switching easily between local
development (e.g. serial processing or :mod:`multiprocessing`) and deployment on a cluster or
supercomputer (via e.g. MPI or JobLib).

Consider the following code in a script (e.g. `eeetest.py`):

.. code-block:: python

   # File: eeetest.py

   # get number of processes
   import sys
   if len(sys.argv) > 1:
       nprocs = int(sys.argv[1])
   else:
       nprocs = 1

   # Ishigami-Homma function a=b=1
   import numpy as np
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # mpi4py is an optional dependency of pyeee
   try:
      from mpi4py import MPI
      comm  = MPI.COMM_WORLD
      csize = comm.Get_size()
      crank = comm.Get_rank()
   except ImportError:
      comm  = None
      csize = 1
      crank = 0

   from pyjams import ee

   # function
   func  = ishigami1
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # choose the serial or parallel pool
   import schwimmbad
   ipool = schwimmbad.choose_pool(mpi=False if csize==1 else True, processes=nprocs)

   # Elementary Effects
   np.random.seed(seed=1023) # for reproducibility of examples
   out = ee(func, lb, ub, 20, processes=nprocs, pool=ipool)

   if crank == 0:
       print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
   ipool.close()

This script can be run serially, i.e. that all function evaluations are done one after the other:

.. code-block:: bash

   python eeetest.py

or

.. code-block:: bash

   python eeetest.py 1

It can use Python's :mod:`multiprocessing` module, e.g. with 4 parallel processes:

.. code-block:: bash

   python eeetest.py 4

or use the Message Passing Interface (MPI), e.g. with 4 parallel processes:

.. code-block:: bash

   mpiexec -n 4 python eeetest.py 4

Note that :mod:`mpi4py` must be installed for the latter.


Python functions with additional parameters
-------------------------------------------

The function for ``pyeee`` must be of the form `func(x)`. Use Python's :func:`functools.partial` to
pass other function parameters.

For example pass the parameters :math:`a` and :math:`b` to the Ishigami-Homma function. One needs a
wrapper function that takes the function and its parameters as arguments. The variable parameters
of the screening must be the last argument, i.e. `x` of `func(x)`:

.. code-block:: python

   from functools import partial

   def ishigami(x, a, b):
       return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

   def call_func_ab(func, a, b, x):
       return func(x, a, b)

The parameters :math:`a` and :math:`b` are fixed parameters during
screening. They are hence already passed to `call_func_ab` with
:func:`functools.partial` before start of the screening.

.. code-block:: python

   # Partialise function with fixed parameters a and b
   a    = 0.5
   b    = 2.0
   func = partial(call_func_ab, ishigami, a, b)

   out  = ee(func, lb, ub, 10)

When `func` is called as `func(x)`, the call of `call_func_ab` is finished and `x`, `a` and `b` are
passed to `ishigami`.

We recommend the package :mod:`partialwrap` that provides wrapper functions to work with
:func:`functools.partial`. `call_func_ab` can be replaced by the wrapper function of
:mod:`partialwrap`: :func:`~partialwrap.function_wrapper`:

.. code-block:: python

   from partialwrap import function_wrapper
   arg   = [a, b]
   kwarg = {}
   func  = partial(function_wrapper, ishigami, arg, kwarg)
   out   = ee(func, lb, ub, 10)

where all arguments of the function but the first one must be given as a :any:`list` and keyword
arguments as a :any:`dict`. The function wrapper finally passes `x`, `arg` and `kwarg` to `func(x,
*arg, **kwarg)`.

:mod:`partialwrap` provides also a wrapper function to work with masks as above. To exclude the
second parameter :math:`x_1` from screening of the Ishigami-Homma function, `x0` and `mask` must be
given to :func:`~partialwrap.function_mask_wrapper`. Then Elementary Effects will be calculated
only for the remaining parameters, between `lb[mask]` and `ub[mask]`. All other non-masked
parameters will be taken as `x0`. Remember that `mask` is an include-mask, i.e. all `mask==True`
will be screened and all `mask==False` will not be screened.

.. code-block:: python

   from partialwrap import function_mask_wrapper
   func = partial(function_mask_wrapper, ishigami, x0, mask, arg, kwarg)
   out  = ee(func, lb[mask], ub[mask], 10)


Sampling parameters with other distributions than the uniform distribution
--------------------------------------------------------------------------

Morris' method of Elementary Effects samples parameters along trajectories through the possible
parameter space. It assumes uniformly distributed parameters between a lower bound and an upper
bound.

``pyeee`` allows sampling parameters from other than uniform distributions. For example, a
parameter :math:`p` might have been determined by repeated experiments. One can hence determine the
mean parameter :math:`\overline{p}` and calculate the error of the mean :math:`\epsilon_p`. This
error of the mean is actually the standard deviation of the distribution of the mean. One would
thus sample a normal distribution with mean :math:`\overline{p}` and a standard deviation
:math:`\epsilon_p` for the parameter :math:`p` for determining Morris' Elementary Effects.

``pyeee`` allows all distributions of :mod:`scipy.stats`, given with the keyword `dist`. The
parameter of the distributions are given as tuples with the keyword `distparam`. The lower and
upper bounds change their meaning if `dist` is given for a parameter: ``pyeee`` samples uniformly
the Percent Point Function (ppf) of the distribution between lower and upper bound. The percent
point function is the inverse of the Cumulative Distribution Function (cdf). Lower and upper bound
must hence be between `0` and `1`. Note the percent point functions of most continuous
distributions will be infinite at the limits `0` and `1`.

The three parameters :math:`x_0, x_1, x_2` of the Ishigami-Homma function follow uniform
distributions between :math:`-\pi` and :math:`+\pi`. Say that :math:`x_1` follows a Gaussian
distribution around the mean `0` with a standard deviation of 1.81. We want to sample between plus
or minus three standard deviations, which includes about 99.7\% of the total distribution. This
means that the lower bound would be 0.0015 (0.003/2.) and the upper bound 0.9985.

.. code-block:: python

   import scipy.stats as stats
   dist      = [None, stats.norm, stats.uniform]
   distparam = [None, (0., 1.81), (-np.pi, 2.*np.pi)]
   lb        = [-np.pi, 0.0015, 0.]
   ub        = [np.pi, 0.9985, 1.]

   out = ee(func, lb, ub, 20, dist=dist, distparam=distparam)

This shows that

   1. one has to give a distribution for each parameter;
   2. distributions are given as :mod:`scipy.stats` distribution objects;
   3. if `dist` is None, ``pyeee`` assumes a uniform distribution and samples between lower and
      upper bound;
   4. (almost) all :mod:`scipy.stats` distributions take the keywords `loc` and `scale`. Their
      meaning is *NOT* mean and standard deviation in most distributions. For the uniform
      distribution :any:`scipy.stats.uniform`, `loc` is the lower limit and `loc+scale` the upper
      limit. This means the combination `dist=None`, `lb=a`, `ub=b` corresponds to
      `dist=scipy.stats.uniform`, `distparam=[a,b-a]`, `lb=0`, `ub=1`.

Note also that

   5. if `distparam` is None, `loc=0` and `scale=1` will be taken;
   6. `loc` and `scale` are implemented as keywords in :mod:`scipy.stats`. Other parameters such as
      for example the shape parameter of the gamma distribution :any:`scipy.stats.gamma` must hence be
      given first, i.e. `(shape,loc,scale)`.

Remember that Morris' method of Elementary Effects assumes uniformly distributed parameters and
that other distributions are an extension of the original method.


Efficient/Sequential Elementary Effects
=======================================

Morris' method of Elementary Effects is not a full sensitivity analysis. The sensititvity measures
of Elementary Effects are rather used for preliminary screening for noninformative model parameters
for a given model output, so that fewer parameters are needed during a full sensitivity analysis or
during model optimisation.

The numerical model `func` will be evaluated `nt*(npars+1)` times for calculating Elementary
Effects. The user chooses the number of trajectories `nt`. A large number of `nt` might be
computationally expensive and a small number might miss areas of the parameter space, where certain
parameters become sensitive. Typical values for `nt` in the literature are on the order of tens to
hundreds. This means that the method of Elementary Effects needs between 500 and 5000 model
evaluations for a model with 50 parameters.

The extension of Efficient or Sequential Elementary Effects can be used if one uses Elementary
Effects *only* to distinguish between sensitive (informative) and insensitive (noninformative)
model parameters. It follows the idea: if one knows that a model is sensitive to a certain
parameter, this parameter does not has to be included anymore in the further screening. If a
parameter has a large Elementary Effect in one trajectory it will most probably be influential. So
one does not have to calculate another Elementary Effect for this parameter and it can be discarded
from further trajectories.

The method starts hence with a limited number of trajectories `ntfirst` for all model parameters,
i.e. it performs `ntfirst*(npars+1)` model evaluations. Further trajectories are sampled,
calculating Elementary Effects, but without the parameters that were already found sensitive. This
means that subsequent trajectories need less and less function evaluations. The algorithm ends if a
subsequent trajectory did not yield any sensitive parameters anymore. A last `ntlast` trajectories
are finally sampled, and Elementary Effects calculated, to assure a large sample for little
sensitive parameters, to minimize the possibility that the parameters are sensitive in a small part
of the parameter space, which was missed due to a little sample.

The call of :func:`~pyeee.screening.eee` (or the identical function :func:`~pyeee.screening.see`)
is very similar to standard Elementary effects `ee`:

.. code-block:: python

   def ishigami(x, a, b):
       return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

   from partialwrap import function_wrapper
   arg   = [a, b]
   kwarg = {}
   unc  = partial(function_wrapper, ishigami, arg, kwarg)
   npars = 3

   # lower boundaries
   lb = np.ones(npars) * (-np.pi)
   # upper boundaries
   ub = np.ones(npars) * np.pi

   # Sequential Elementary Effects
   from pyeee import eee
   np.random.seed(seed=1025) # for reproducibility of examples
   out = eee(func, lb, ub, ntfirst=10, ntlast=5, nsteps=6,
             processes=4)

   print(out)
   # gives: [ True False  True]

:func:`~pyeee.screening.eee` returns an include-mask, being `True` for sensitive parameters and
`False` for noninformative parameters. The mask can be combined by `logical_and` with an incoming
mask.

Note if you use :func:`~partialwrap.function_mask_wrapper`, `out` has the dimension of the
`mask==True` elements:

.. code-block:: python

   from partialwrap import function_mask_wrapper
   func = partial(function_mask_wrapper, ishigami, x0, mask, arg, kwarg)
   out  = eee(func, lb[mask], ub[mask])

   # update mask
   mask[mask] = mask[mask] & out

The numerical model `func` might return several outputs per model run, e.g. a time series. The
Morris' sensitivity measures are calculated hence for each output, e.g. each point in time.
Efficient/Sequential Elementary Effects :func:`~pyeee.screening.eee` can either take the arithmetic
mean of all :math:`\mu*` or a weighted mean :math:`\mu*`, weighted by :math:`\sigma`. The keyword
`weight==False` is probably appropriate if each single output is equally important. An example is
river runoff where high flows might be floods and low flows might be droughts. One might want that
the computer model reproduces both circumstances. An example for `weight==True` are fluxes to and
from the atmosphere such as evapotranspiration. The atmosphere is more strongly influenced by
larger fluxes so that sensitivity measures during periods of little atmosphere exchange are less
interesting. Cuntz *et al.* (2015) argued that weighting by standard deviation :math:`\sigma` is
equivalent to flux weighting because parameter variations yield larger variances for large fluxes
than for small fluxes in most computer models.

:func:`~pyeee.screening.eee` offers the same parallel mechanism as `ee`, using the
:func:keywords `processes` and `pool`, which is again a :mod:`schwimmbad` `pool` object.

:func:`~pyeee.screening.eee` also offers the possibility to sample parameters from different
distributions of :mod:`scipy.stats` with the keywords `dist` and `distparam`.

One can give a `plotfile` name to check the initial fit to the `ntfirst` Elementary Effects.

.. code-block:: python

   # Sequential Elementary Effects using all parameters and keywords
   out = eee(func, lb, ub,
             x0=x0, mask=mask, ntfirst=10, ntlast=10, nsteps=6, weight=True,
             processes=4, seed=1025,
             plotfile='ishigami.png', logfile='ishigami.log')

Note that :mod:`matplotlib` must be installed to produce the `plotfile`.


External computer models
========================

**Note that this section is pretty much a repetition of the** `User Guide
<https://partialwrap.readthedocs.io/en/latest/userguide.html>`_ **of** :mod:`partialwrap`, **which is not
limited to be used with** ``pyeee`` **but can be used with any package that calls functions in
the form** `func(x)`. **The notions of** :mod:`partialwrap` **might be better explained in its** `user guide
<https://partialwrap.readthedocs.io/en/latest/userguide.html>`_.

:mod:`partialwrap` provides wrapper functions to work with external executables. :mod:`partialwrap`
writes the sampled parameter sets into files that can be read by the external program. The program
writes its result to a file that will then be read by :mod:`partialwrap` in return. The processing
steps are:

.. code-block:: python

   parameterwriter(parameterfile, parameters)
   err = subprocess.check_output(exe)
   obj = outputreader(outputfile)
   os.remove(parameterfile)
   os.remove(outputfile)

That means :mod:`partialwrap` needs to have a function `parameterwriter` that writes the parameter
file `parameterfile` needed by the executable `exe`. It then needs to have a function
`outputreader` for reading the output file `outputfile` of `exe`, reading or calculating the
objective value used by Elementary Effects.


Simple executables
------------------

Consider for simplicity an external Python program (e.g. `ishiexe.py`)
that calculates the Ishigami-Homma function with :math:`a = b = 1`,
reading in the three parameters :math:`x_0, x_1, x_2` from a
`parameterfile = params.txt` and writing its output into an
`outputfile = obj.txt`:

.. code-block:: python

   # File: ishiexe.py

   # Ishigami-Homma function a=b=1
   import numpy as np
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # read parameters
   from partialwrap import standard_parameter_reader
   pfile = 'params.txt'
   x = standard_parameter_reader(pfile)

   # calc function
   y = ishigami1(x)

   # write objective
   ofile = 'obj.txt'
   with open(ofile, 'w') as ff:
       print(y, file=ff)

This program can be called on the command line with:

.. code-block:: bash

    python ishiexe.py

The external program can be used in ``pyeee`` with :func:`functools.partial` and the
wrapper function :func:`~partialwrap.exe_wrapper`:

.. code-block:: python

   from functools import partial
   from partialwrap import exe_wrapper, standard_parameter_writer, standard_output_reader

   ishi = ['python', 'ishiexe.py']
   parameterfile = 'params.txt'
   outputfile = 'obj.txt'
   func = partial(exe_wrapper, ishi,
                  parameterfile, standard_parameter_writer,
                  outputfile, standard_output_reader, {})

   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi

   from pyjams import ee
   out = ee(func, lb, ub, 10)

:func:`~partialwrap.standard_parameter_reader` and `~partialwrap.standard_parameter_writer` are
convenience functions that read and write one parameter per line in a file without a header. The
function :func:`~partialwrap.standard_output_reader` simply reads one value from a file without
header. The empty dictionary at the end will be explained below at `Further arguments of
wrappers`_.

One can easily imagine to replace the python program `ishiexe.py` by any compiled executable from
C, Fortran or alike.


Exclude parameters from screening
---------------------------------

Similar to :func:`~partialwrap.function_mask_wrapper`, there is also a wrapper to work with masks
and external executables: :func:`~partialwrap.exe_mask_wrapper`. To exclude the second parameter
:math:`x_1` from screening of the Ishigami-Homma function, `x0` and `mask` must be given to
:func:`~partialwrap.exe_mask_wrapper` as well. Remember that `mask` is an include-mask, i.e. all
`mask==True` will be screened and all `mask==False` will not be screened:

.. code-block:: python

   mask    = np.ones(npars, dtype=bool) # True  -> include
   mask[1] = False                      # False -> exclude
   x0      = np.ones(npars) * 0.5
   func = partial(exe_mask_wrapper, ishi, x0, mask,
                  parameterfile, standard_parameter_writer,
                  outputfile, standard_output_reader, {})
   out = ee(func, lb[mask], ub[mask], 10)

:math:`x_1` will then always be the second element of `x0`.


Additional arguments for external executables
---------------------------------------------

Further arguments to the external executable can be given simply by adding it to the call string.
For example, if :math:`a` and :math:`b` were command line arguments to `ishiexe.py`, they could
simply be given in the function name:

.. code-block:: python

   ishi = ['python3', 'ishiexe.py', '-a '+str(a), '-b '+str(b)]


Further arguments of wrappers
-----------------------------

The user can pass further arguments to :func:`~partialwrap.exe_wrapper` and
:func:`~partialwrap.exe_mask_wrapper` via a dictionary at the end of the call. Setting the key
`shell` to `True` passes `shell=True` to :func:`subprocess.check_output`, which makes
:func:`subprocess.check_output` open a shell for running the external executable. Note that the
`args` in :mod:`subprocess` must be a string if `shell=True` and a list if `shell=False`. Setting
the key `debug` to `True` uses :func:`subprocess.check_call` so that any output of the external
executable will be written to the screen (precisely :any:`subprocess.STDOUT`). This especially
prints out also any errors that might have occured during execution:

.. code-block:: python

   ishi = 'python ishiexe.py'
   func = partial(exe_wrapper, ishi,
                  parameterfile, standard_parameter_writer,
                  outputfile, standard_output_reader,
                  {'shell':True, 'debug':True})
   out  = ee(func, lb, ub, 10)

This mechanism allows passing also additional arguments and keyword arguments to the
`parameterwriter`. Setting `pargs` to a list of arguments and `pkwargs` to a dictionary with
keyword arguments passes them to the `parameterwriter` as:

.. code-block:: python

    parameterwriter(parameterfile, x, *pargs, **pkwargs)

Say an external program uses a `parameterfile` that has five
informations per line: 1. identifier, 2. current parameter value, 3. minimum
parameter value, 4. maximum parameter value, 5. parameter mask, e.g.:

.. code-block:: none

    # value min max mask
    1 0.5 -3.1415 3.1415 1
    2 0.0 -3.1415 3.1415 0
    3 1.0 -3.1415 3.1415 1

One can use :func:`~partialwrap.standard_parameter_reader_bounds_mask` in this case. Parameter
bounds and mask can be passed via `pargs`:

.. code-block:: python

   from partialwrap import standard_parameter_reader_bounds_mask

   ishi = ['python', 'ishiexe.py']
   func = partial(exe_wrapper, ishi,
                  parameterfile, standard_parameter_reader_bounds_mask,
                  outputfile, standard_output_reader,
                  {'pargs':[lb,ub,mask]})
   out  = ee(func, lb, ub, 10)

Or in case of exclusion of :math:`x_1`:

.. code-block:: python

   from partialwrap import standard_parameter_reader_bounds_mask
   func = partial(exe_mask_wrapper, ishi, x0, mask,
                  parameterfile, standard_parameter_reader_bounds_mask,
                  outputfile, standard_output_reader,
                  {'pargs':[lb,ub,mask]})
   out  = ee(func, lb[mask], ub[mask], 10)

Another common case is that the parameters are given in the form `parameter = value`, e.g. in
Fortran namelists. :mod:`partialwrap` provides a function that searches parameter names on the
left-hand-side of an equal sign and replaces the values on the right-hand-side of the equal sign
with the sampled parameter values. The `parameterfile` might look like:

.. code-block:: Fortran

   &params
       x0 = 0.5
       x1 = 0.0
       x2 = 1.0
   /

The function :func:`~partialwrap.sub_params_names` (which is identical to
:func:`~partialwrap.sub_params_names_ignorecase`) can be used and parameter names are passed via
`pargs`:

.. code-block:: python

   from partialwrap import sub_params_names

   pnames = ['x0', 'x1', 'x2']
   func = partial(exe_wrapper, ishi,
                  parameterfile, sub_params_names,
                  outputfile, standard_output_reader,
                  {'pargs':[pnames], 'pid':True})
   out = ee(func, lb, ub, 10)

`parameterfile` can be a list of parameterfiles in case of :func:`~partialwrap.sub_params_names`.
`pid` will be explained in the next section. Note that `pargs` is set to `[pnames]`. Setting
`'pargs':pnames` would give `*pnames` to the `parameterwriter`, that means each parameter name as
an individual argument, which would be wrong because :func:`~partialwrap.sub_params_names` wants to
have a list of parameter names. The docstring of :func:`~partialwrap.exe_wrapper` states:

.. code-block:: none

   Wrapper function for external programs using a `parameterwriter` and `outputreader`
   with the interfaces:
       `parameterwriter(parameterfile, x, *pargs, **pkwargs)`
       `outputreader(outputfile, *oargs, **okwargs)`
   or if `pid==True`:
       `parameterwriter(parameterfile, x, *pargs, pid=pid, **pkwargs)`
       `outputreader(outputfile, *oargs, pid=pid, **okwargs)`

And the definition of :func:`~partialwrap.sub_params_names` is:

.. code-block:: python

   def sub_params_names_ignorecase(files, params, names, pid=None):

This means that `*pargs` passes `*[pnames]`, which is `pnames`, as an argument after the
parameters `x` to :func:`~partialwrap.sub_params_names`.

Excluding :math:`x_1` would then be achieved by simply excluding `x1` from `pnames`:

.. code-block:: python

   from partialwrap import sub_params_names

   pnames = ['x0', 'x2']
   func = partial(exe_wrapper, ishi,
                  parameterfile, sub_params_names,
                  outputfile, standard_output_reader,
                  {'pargs':[pnames], 'pid':True})
   out  = ee(func, lb[mask], ub[mask], 10)


Parallel processing of external executables
-------------------------------------------

Elementary Effects run the computational model `nt*(npars+1)` times. All model runs are independent
and can be executated at the same time if computing ressources permit. Even simple personal
computers have several computing cores nowadays. If the computational model is run several times in the
same directory at the same time, all model runs would read the same parameter file and overwrite
the output of each other.

:func:`~partialwrap.exe_wrapper` concatenates an individual integer number to the function string
(or list, see :mod:`subprocess`), adds the integer to call of `parameterwrite` and of
`outputreader`, like:

.. code-block:: python

   pid = str(randst.randint())
   parameterwriter(parameterfile, x, *pargs, pid=pid, **pkwargs)
   err = subprocess.check_output([func, pid])
   obj = outputreader(outputfile, *oargs, pid=pid, **okwargs)
   os.remove(parameterfile+'.'+pid)
   os.remove(outputfile+'.'+pid)

The `parameterwriter` is assumed to write `parameterfile.pid` and the external model is assumed to
write `outputfile.pid`. Only these filenames are cleaned up by :func:`~partialwrap.exe_wrapper`. If
different filenames are used, the user has to clean up herself.

`ishiexe.py` would hence need to read the number from the command line:

.. code-block:: python

   # File: ishiexe1.py

   # read pid if given
   import sys
   pid = None
   if len(sys.argv) > 1:
       pid = sys.argv[1]

   # Ishigami-Homma function a=b=1
   import numpy as np
   def ishigami1(x):
       return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

   # read parameters
   from partialwrap import standard_parameter_reader
   pfile = 'params.txt'
   x = standard_parameter_reader(pfile, pid=pid)

   # calc function
   y = ishigami1(x)

   # write objective
   ofile = 'obj.txt'
   if pid:
       ofile = ofile+'.'+pid
   with open(ofile, 'w') as ff:
       print(y, file=ff)

:func:`~partialwrap.exe_wrapper` would then be used with `'pid':True` and one can use several
parallel processes:

.. code-block:: python

   from partialwrap import exe_wrapper, standard_parameter_writer, standard_output_reader

   ishi = ['python3', 'ishiexe1.py']
   parameterfile = 'params.txt'
   outputfile    = 'obj.txt'
   func = partial(exe_wrapper, ishi,
                  parameterfile, standard_parameter_writer,
                  outputfile, standard_output_reader, {'pid':True})
   npars = 3
   lb  = np.ones(npars) * (-np.pi)
   ub  = np.ones(npars) * np.pi
   out = ee(func, lb, ub, 10, processes=8)

If you cannot change your computational model, you can use, for example, a bash script that
launches each model run in a separate directory, like:

.. code-block:: bash

   #!/bin/bash

   # File: ishiexe.sh

   # get pid
   pid=${1}

   # make individual run directory
   mkdir tmp.${pid}

   # run in individual directory
   cp ishiexe.py tmp.${pid}/
   mv params.txt.${pid} tmp.${pid}/params.txt
   cd tmp.${pid}
   python ishiexe.py

   # make output available to pyeee
   mv obj.txt ../obj.txt.${pid}

   # clean up
   cd ..
   rm -r tmp.${pid}

which would then be used:

.. code-block:: python

   from functools import partial
   from partialwrap import exe_wrapper, standard_parameter_writer, standard_output_reader

   ishi = './ishiexe.sh'
   parameterfile = 'params.txt'
   outputfile = 'obj.txt'
   func = partial(exe_wrapper, ishi,
                  parameterfile, standard_parameter_writer,
                  outputfile, standard_output_reader,
                  {'pid':True, 'shell':True})
   npars = 3
   lb = np.ones(npars) * (-np.pi)
   ub = np.ones(npars) * np.pi
   from pyjams import ee
   out = ee(func, lb, ub, 10, processes=8)

The `User Guide <https://partialwrap.readthedocs.io/en/latest/userguide.html>`_ of
:mod:`partialwrap` gives a similar script written in Python, which could be used if the bash shell is not available, for example on Windows.

That's all Folks!
