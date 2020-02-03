**********************
pyeee User Guide
**********************

``pyeee`` is a Python library for performing parameter screening of
computational models. It uses Morris' method of  Elementary Effects (*EE*)
and also its extension of Efficient or Sequential Elementary Effects
(*EEE* or *SEE*) published by:

Cuntz, Mai *et al.* (2015) Computationally inexpensive
identification of noninformative model parameters by sequential
screening, *Water Resources Research* 51, 6417-6441,
doi:`10.1002/2015WR016907 <http://doi.org/10.1002/2015WR016907>`_.

The numerical models are simply passed to functions :func:`~pyeee.screening.ee`
and :func:`~pyeee.eee.eee` to perform Elementary Effects or
Efficient/Sequential Elementary Effects, respectivley.

The numerical models must be callable as `func(x)`. Use
:func:`functools.partial` from Python's standard library to make any
function callable as `func(x)`. ``pyeee`` provides wrapper functions
to help with this process.


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

The three parameters :math:`x_0, x_1, x_2` follow uniform
distributions between :math:`-\pi` and :math:`+\pi`.

Elementary Effects can be calculated as:

.. code-block:: python

    from pyeee import ee
    ## from screening import ee

    # function
    func  = ishigami1
    npars = 3

    # lower boundaries
    lb = np.ones(npars) * (-np.pi)
    # upper boundaries
    ub = np.ones(npars) * np.pi

    # Elementary Effects
    np.random.seed(seed=1023) # for reproducibility of examples
    out = ee(func, lb, ub)

:func:`~pyeee.screening.ee` returns a `(npars,3)` ndarray with:

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

The numerical model `func`, lower parameter boundaries `lb`, and upper
parameter boundaries `ub` are mandatory arguments to
:func:`~pyeee.screening.ee`. Further optional arguments relevant to
Elementary Effects are:

    - `nt` : int - Number of trajectories used (default: `len(lb)`)
    - `nsteps` : int - Number of steps along one trajectory (default: 6)
    - `ntotal` : int - Total number of trajectories to check for the `nt` most
      different trajectories (default: `max(nt**2,10*nt)`)

Note that the functions :func:`~pyeee.screening.ee` and
:func:`~pyeee.screening.screening` are identical.


Exclude parameters from calculations
------------------------------------

:func:`~pyeee.screening.ee` offers the possibility to mask some model
parameters so that they will not be changed during calculation of
Elementary Effects. Inital values `x0` must be given that will be
taken where `mask==False`, i.e. `mask` could be called an
include-mask. Note that the size of `x0` must be the size of `lb`,
`ub` and `mask`, i.e. one has to give initial values even if an
element is included in the screening, which means `mask[i]==True`.

For example, if one want to exclude the second parameter :math:`x_1`
of the above Ishigami-Homma function in the calculation of the
Elementary Effects:

.. code-block:: python

    # function
    mask    = np.ones(npars, dtype=bool) # True  -> include
    mask[1] = False                      # False -> exclude

    # initial values
    x0 = np.ones(npars) * 0.5

    # Elementary Effects
    np.random.seed(seed=1024) # for reproducibility of examples
    out = ee(func, lb, ub, x0=x0, mask=mask, nt=10, nsteps=8, ntotal=100)

    print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
    # gives: 114.8 0.0 26.6


Parallel model evaluation
-------------------------

The numerical model `func` will be evaluated `nt*(npars+1)` times,
with `npars` the number of parameters of the computational
model. Multiprocessing can be used for parallel function
evaluation. Setting `processes=nprocs` evaluates `nprocs` parameter
sets in parallel:

.. code-block:: python

    # Elementary Effects using 4 parallel processes
    np.random.seed(seed=1024) # for reproducibility of examples
    out = ee(func, lb, ub, x0=x0, mask=mask, nt=10, nsteps=8, ntotal=100,
             processes=4)

``pyeee`` uses the package :any:`schwimmbad` for
parallelisation. :any:`schwimmbad` provides a uniform interface to
parallel processing pools and enables switching easily between local
development (e.g. serial processing or :any:`multiprocessing`) and
deployment on a cluster or supercomputer (via e.g. MPI or JobLib).

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

    from pyeee import ee
    ## from screening import ee

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
    out = ee(func, lb, ub, nt=20, processes=nprocs, pool=ipool)

    if crank == 0:
        print("{:.1f} {:.1f} {:.1f}".format(*out[:,0]))
    ipool.close()

This script can be run serially, i.e. that all function evaluations
are done one after the other:

.. code-block:: bash

    python eeetest.py

or

.. code-block:: bash

    python eeetest.py 1

It can use Python's :any:`multiprocessing` module, e.g. with 4
parallel processes:

.. code-block:: bash

    python eeetest.py 4

or use the Message Passing Interface (MPI), e.g. with 4 parallel processes:

.. code-block:: bash

    mpiexec -n 4 python eeetest.py 4

Note that :mod:`mpi4py` must be installed for the latter.


Python functions with additional parameters
-------------------------------------------

The function for ``pyeee`` must be of the form `func(x)`. Use Python's
:any:`functools.partial` to pass other function parameters.

For example pass the parameters :math:`a` and :math:`b` to the
Ishigami-Homma function. One needs a wrapper function that takes the function
and its parameters as arguments. The variable parameters of the
screening must be the last argument, i.e. `x` of `func(x)`:

.. code-block:: python

    from functools import partial

    def ishigami(x, a, b):
        return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

    def call_ishigami(ishi, a, b, x):
        return ishi(x, a, b)

The parameters :math:`a` and :math:`b` are fixed parameters during
screening. They are hence already passed to `call_ishigami` with
:any:`functools.partial` before start of the screening.

.. code-block:: python

    # Partialise function with fixed parameters a and b
    a 	 = 0.5
    b 	 = 2.0
    func = partial(call_ishigami, ishigami, a, b)

    out  = ee(func, lb, ub)

When `func` is called as `func(x)`, the call of `call_ishigami` is
finished and `x`, `a` and `b` are passed to `ishigami`.

``pyeee`` provides wrapper functions to work with
:any:`functools.partial`. `call_ishigami` can be replaced by the
wrapper function of ``pyeee``:
:func:`~pyeee.function_wrapper.func_wrapper`:

.. code-block:: python

    from pyeee import func_wrapper
    arg   = [a, b]
    kwarg = {}
    func  = partial(func_wrapper, ishigami, arg, kwarg)
    out   = ee(func, lb, ub)

where all arguments of the function but the first one must be given as
a `list` and keyword arguments as a `dictionary`. The function wrapper
finally passes `x`, `arg` and `kwarg` to `func(x, *arg, **kwarg)`.

``pyeee`` provides also a wrapper function to work with masks as
above. To exclude the second parameter :math:`x_1` from screening of
the Ishigami-Homma function again, `x0` and `mask` must be given to
:func:`~pyeee.function_wrapper.func_mask_wrapper` as well. Then
Elementary Effects will be calculated only for the remaining
parameters, between `lb[mask]` and `ub[mask]`. All other
non-masked parameters will be taken as `x0`. Remember that `mask` is
an include-mask, i.e. all `mask==True` will be screened and all
`mask==False` will not be screened.

.. code-block:: python

    from pyeee import func_mask_wrapper
    func = partial(func_mask_wrapper, ishigami, x0, mask, arg, kwarg)
    out  = ee(func, lb[mask], ub[mask])


Efficient/Sequential Elementary Effects
=======================================

Morris' method of Elementary Effects is not a full sensitivity
analysis. The sensititvity measures of Elementary Effects are rather
used for preliminary screening for noninformative model parameters for a
given model output, so that fewer parameters are needed during a full
sensitivity analysis or during model optimisation.

The numerical model `func` will be evaluated `nt*(npars+1)` times for
calculating Elementary Effects. The user can choose the number of
trajectories `nt`. A large number of `nt` might be computationally
expensive and a small number might miss areas of the parameter space,
where certain parameters become sensitive. Typical values for `nt` in
the literature are on the order of tens to hundreds. This means that
the method of Elementary Effects needs between 500 and 5000 model
evaluations for a model with 50 parameters.

The extension of Efficient or Sequential Elementary Effects can be
used if one uses Elementary Effects *only* to distinguish between
sensitive (informative) and insensitive (noninformative) model
parameters. It follows the idea: if one knows that a model is
sensitive to a certain parameter, this parameter does not has to be
included anymore in the further analysis. If a parameter has a large
Elementary Effect in one trajectory it will most probably be
influential. So one does not have to calculate another Elementary
Effect for this parameter and it can be discarded from further
trajectories.

The method starts hence with a limited number of trajectories
`ntfirst` for all model parameters, i.e. it performs
`ntfirst*(npars+1)` model evaluations. Further trajectories are
sampled, calculating Elementary Effects, but without the parameters
that were already found sensitive. This means that subsequent
trajectories need less and less function evaluations. The algorithm
ends if a subsequent trajectory did not yield any sensitive parameters
anymore. A last `ntlast` trajectories are finally sampled, and
Elementary Effects calculated, to assure a large sample for little
sensitive parameters.

The call of :func:`~pyeee.screening.eee` (or the identical function
:func:`~pyeee.screening.see`) is very similar to standard Elementary
effects :func:`~pyeee.screening.ee`:

.. code-block:: python

    def ishigami(x, a, b):
        return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

    from pyeee import func_wrapper
    arg   = [a, b]
    kwarg = {}
    func  = partial(func_wrapper, ishigami, arg, kwarg)
    npars = 3

    # lower boundaries
    lb = np.ones(npars) * (-np.pi)
    # upper boundaries
    ub = np.ones(npars) * np.pi

    # Sequential Elementary Effects
    from pyeee import eee
    ## from eee import eee
    np.random.seed(seed=1025) # for reproducibility of examples
    out = eee(func, lb, ub, ntfirst=10, ntlast=5, nsteps=6,
              processes=4)

    print(out)
    # gives: [ True False  True]

:func:`~pyeee.screening.eee` returns an include-mask, being `True` for
sensitive parameters and `False` for noninformative parameters. The
mask can be combined by `logical_and` with an incoming mask.

Note if you use :func:`~pyeee.function_wrapper.func_mask_wrapper`, `out`
has the dimension of the `mask==True` elements:

.. code-block:: python

    from pyeee import func_mask_wrapper
    func = partial(func_mask_wrapper, ishigami, x0, mask, arg, kwarg)
    out  = eee(func, lb[mask], ub[mask])

    # update mask
    mask[mask] = mask[mask] & out

The numerical model `func` might return several outputs per model run,
e.g. a time series. The Morris' sensitivity measures are calculated
hence for each output, e.g. each point in
time. :func:`~pyeee.screening.eee` can either take the arithmetic mean
of all :math:`\mu*` or a weighted mean :math:`\mu*`, weighted by
:math:`\sigma`. The keyword `weight==False` is probably appropriate if
each single output is equally important. An example is river runoff
where high flows might be floods and low flows might be droughts. One
might want that the computer model reproduces both circumstances. An
example for `weight==True` are fluxes to and from the atmosphere such
as evapotranspiration. The atmosphere is more strongly influenced by
larger fluxes so that sensitivity measures during periods of little
atmosphere exchange are less interesting. Cuntz *et al.* (2015) argued
that weighting by stndard deviation :math:`\sigma` is equivalent to
flux weighting because parameter variations yield larger variances for
large fluxes than for small fluxes in most computer models.

:func:`~pyeee.screening.eee` offers the same parallel mechanism as
:func:`~pyeee.screening.ee`, using the keywords `processes` and
`pool`, which is again a :any:`schwimmbad` `pool` object.

One can give a `plotfile` name to check the initial fit to the
`ntfirst` Elementary Effects.

.. code-block:: python

    # Sequential Elementary Effects using all parameters and keywords
    out = eee(func, lb, ub,
              x0=x0, mask=mask, ntfirst=10, ntlast=10, nsteps=6, weight=True,
              processes=4, seed=1025,
	      plotfile='ishigami.png', logfile='ishigami.log')

Note that :mod:`matplotlib` must be installed to produce the `plotfile`.


External computer models
========================

``pyeee`` provides wrapper functions to work with external
executables. ``pyeee`` writes the sampled parameter sets into files
that can be read by the external program. The program writes its
result to a file that will then be read by ``pyeee`` in return. The
processing steps are:

.. code-block:: python

	parameterwriter(parameterfile, x)
        err = subprocess.check_output(exe)
        obj = objectivereader(objectivefile)
        os.remove(parameterfile)
        os.remove(objectivefile)

That means ``pyeee`` needs to have a function `parameterwriter` that
writes the parameter file `parameterfile` needed by the executable
`exe`. It then needs to have a function `objectivereader` for reading
the output file `objectivefile` of `exe`, reading or calculating the
objective value used by Elementary Effects.


Simple executables
------------------

Consider for simplicity an external Python program (e.g. `ishiexe.py`)
that calculates the Ishigami-Homma function with :math:`a = b = 1`,
reading in the three parameters :math:`x_0, x_1, x_2` from a
`parameterfile = params.txt` and writing its output into an
`objectivefile = obj.txt`:

.. code-block:: python

    # File: ishiexe.py

    # Ishigami-Homma function a=b=1
    import numpy as np
    def ishigami1(x):
        return np.sin(x[0]) + np.sin(x[1])**2 + x[2]**4 * np.sin(x[0])

    # read parameters
    from pyeee import standard_parameter_reader
    ## from std_io import standard_parameter_reader
    pfile = 'params.txt'
    x = standard_parameter_reader(pfile)

    # calc function
    y = ishigami1(x)

    # write objective
    ofile = 'obj.txt'
    ff = open(ofile, 'w')
    print(y, file=ff)
    ff.close()

This program can be called on the command line with:

.. code-block:: bash

    python ishiexe.py

The external program can be used in ``pyeee`` with :any:`functools.partial` and the
wrapper function :func:`~pyeee.function_wrapper.exe_wrapper`:

.. code-block:: python

    from functools import partial
    from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
    ishi = ['python', 'ishiexe.py']
    parameterfile = 'params.txt'
    objectivefile = 'obj.txt'
    func = partial(exe_wrapper, ishi,
                   parameterfile, standard_parameter_writer,
		   objectivefile, standard_objective_reader, {})
    npars = 3
    lb = np.ones(npars) * (-np.pi)
    ub = np.ones(npars) * np.pi
    
    from pyeee import ee
    out = ee(func, lb, ub)

:func:`~pyeee.std_io.standard_parameter_reader` and
:func:`~pyeee.std_io.standard_parameter_writer` are convenience
functions that read and write one parameter per line in a file without
a header. The function :func:`~pyeee.std_io.standard_objective_reader`
simply reads one value from a file without header. The empty directory
at the end will be explained below at `Further arguments of wrappers`_.

One can easily imagine to replace the python program `ishiexe.py` by
any compiled executable from C, Fortran or alike.


Exclude parameters from screening
---------------------------------

Similar to :func:`~pyeee.function_wrapper.func_mask_wrapper`, there is
also a wrapper to work with masks and external executables:
:func:`~pyeee.function_wrapper.exe_mask_wrapper`. To exclude the second parameter :math:`x_1` from screening of
the Ishigami-Homma function again, `x0` and `mask` must be given to
:func:`~pyeee.function_wrapper.exe_mask_wrapper` as well. Remember that `mask` is
an include-mask, i.e. all `mask==True` will be screened and all
`mask==False` will not be screened:

.. code-block:: python

    mask    = np.ones(npars, dtype=bool) # True  -> include
    mask[1] = False                      # False -> exclude
    x0 = np.ones(npars) * 0.5
    func = partial(exe_mask_wrapper, ishi, x0, mask,
                   parameterfile, standard_parameter_writer,
		   objectivefile, standard_objective_reader, {})
    out  = ee(func, lb[mask], ub[mask])

:math:`x_1` will then always be the second element of `x0`.


Additional arguments for external executables
---------------------------------------------

Further arguments to the external executable can be given simply by
adding it to the call string. For example, if :math:`a` and :math:`b`
were command line arguments to `ishiexe.py`, they could simply be given in
the function name:

.. code-block:: python

    ishi = ['python3', 'ishiexe.py', '-a str(a)', '-b str(b)']


Further arguments of wrappers
-----------------------------

The user can pass further arguments to
:func:`~pyeee.function_wrapper.exe_wrapper` and
:func:`~pyeee.function_wrapper.exe_mask_wrapper` via a dictionary at
the end of the call. Setting the key `shell` to `True` passes
`shell=True` to :func:`subprocess.check_output`, which makes
:func:`subprocess.check_output` open a shell for running the external
executable. Note that the `args` in :any:`subprocess` must be a string
if `shell=True` and a list it `shell=False`. Setting the key `debug`
to `True` uses :func:`subprocess.check_call` so that any output of the
external executable will be written to the screen (precisely
:any:`subprocess.STDOUT`). This especially prints out also any errors
that might have occured during execution:

.. code-block:: python

    ishi = 'python ishiexe.py'
    func = partial(exe_wrapper, ishi,
                   parameterfile, standard_parameter_writer,
		   objectivefile, standard_objective_reader,
		   {'shell':True, 'debug':True})
    out  = ee(func, lb, ub)

This mechanism allows passing also additional arguments and keyword
arguments to the `parameterwriter`. Setting `pargs` to a list of
arguments and `pkwargs` to a dictionary with keyword arguments passes
them to the `parameterwriter` as:

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

One can use
:func:`~pyeee.std_io.standard_parameter_reader_bounds_mask` in this
case. Parameter bounds and mask can be passed via `pargs`:

.. code-block:: python

    from pyeee import standard_parameter_reader_bounds_mask
    ishi = ['python', 'ishiexe.py']
    func = partial(exe_wrapper, ishi,
                   parameterfile, standard_parameter_reader_bounds_mask,
		   objectivefile, standard_objective_reader,
		   {'pargs':[lb,ub,mask]})
    out  = ee(func, lb, ub)

Or in case of exclusion of :math:`x_1`:

.. code-block:: python

    from pyeee import standard_parameter_reader_bounds_mask
    func = partial(exe_mask_wrapper, ishi, x0, mask,
                   parameterfile, standard_parameter_reader_bounds_mask,
		   objectivefile, standard_objective_reader,
		   {'pargs':[lb,ub,mask]})
    out  = ee(func, lb[mask], ub[mask])

Another common case is that the parameters are given in the form `parameter
= value`, e.g. in Fortran namelists. ``pyeee`` provides a function
that searches parameter names on the left-hand-side of an equal sign
and replaces the values on the right-hand-side of the equal sign with
the sampled parameter values. The parameterfile might look like:

.. code-block:: Fortran

    &params
      x0 = 0.5
      x1 = 0.0
      x2 = 1.0
    /

The function :func:`~pyeee.std_io.sub_names_params_files` (which is
identical to :func:`~pyeee.std_io.sub_names_params_files_ignorecase`)
can be used and parameter names are passed via `pargs`:

.. code-block:: python

    from pyeee import sub_names_params_files
    pnames = ['x0', 'x1', 'x2']
    func = partial(exe_wrapper, ishi,
                   parameterfile, sub_names_params_files,
		   objectivefile, standard_objective_reader,
		   {'pargs':[pnames], 'pid':True})
    out  = ee(func, lb, ub)

`parameterfile` can be a list of parameterfiles in case of
:func:`~pyeee.std_io.sub_names_params_files`. `pid` will be explained
in the next section. Note that `pargs` is set to `[pnames]`. Setting
`'pargs':pnames` would give `*pnames` to the parameterwriter, that
means each parameter name as an individual argument, which would be
wrong because it wants to have a list of parameter names. The
docstring of :func:`~pyeee.function_wrapper.exe_wrapper` states:

.. code-block:: none

    Wrapper function for external programs using a parameterwriter
    with the interface:
        parameterwriter(parameterfile, x, *pargs, **pkwargs)
    or if pid==True:
        parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)

And the definition of :func:`~pyeee.std_io.sub_names_params_files` is:

.. code-block:: python

    def sub_names_params_files_ignorecase(files, pid, params, names):

so `*pargs` passes `*[pnames]` that means `pnames` as argument after the
parameters to :func:`~pyeee.std_io.sub_names_params_files`.

Excluding :math:`x_1` would then be achieved by simply excluding `x1`
from `pnames`:

.. code-block:: python

    from pyeee import sub_names_params_files
    pnames = ['x0', 'x2']
    func = partial(exe_wrapper, ishi,
                   parameterfile, sub_names_params_files,
		   objectivefile, standard_objective_reader,
		   {'pargs':[pnames], 'pid':True})
    out  = ee(func, lb[mask], ub[mask])


Parallel processing of external executables
-------------------------------------------

Elementary Effects run the computational model `nt*(npars+1)` times. All
model runs are independent and can be executated at the same time if
computing ressources permit. Even simple personal computers have
computing cores nowadays. If the computational model is run several
times in the same directory at the same time, all model runs would
read the same parameter file and overwrite the output of each
other.

:func:`~pyeee.function_wrapper.exe_wrapper` concatenates an individual
integer number to the function string (or list, see
:any:`subprocess`), adds the integer to call of `parameterwrite` and
appends the number to the `objectivefile`, like:

.. code-block:: python

    pid = str(randst.randint())
    parameterwriter(parameterfile, pid, x, *pargs, **pkwargs)
    err = subprocess.check_output([func, pid])
    obj = objectivereader(objectivefile+'.'+pid)
    os.remove(parameterfile+'.'+pid)
    os.remove(objectivefile+'.'+pid)

The `parameterwriter` is supposed to write `parameterfile+'.'+ipid`

`ishiexe.py` would then need to read the number from the command line:

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
    from pyeee import standard_parameter_reader
    ## from std_io import standard_parameter_reader
    pfile = 'params.txt'
    if pid is not None:
        pfile = pfile+'.'+pid
    x = standard_parameter_reader(pfile)

    # calc function
    y = ishigami1(x)

    # write objective
    ofile = 'obj.txt'
    if pid is not None:
        ofile = ofile+'.'+pid
    ff = open(ofile, 'w')
    print(y, file=ff)
    ff.close()

:func:`~pyeee.function_wrapper.exe_wrapper` would then be used with
`'pid':True` and one can use several parallel processes:

.. code-block:: python

    from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
    ishi = ['python3', 'ishiexe1.py']
    parameterfile = 'params.txt'
    objectivefile = 'obj.txt'
    func = partial(exe_wrapper, ishi,
                   parameterfile, standard_parameter_writer,
		   objectivefile, standard_objective_reader, {'pid':True})
    npars = 3
    lb = np.ones(npars) * (-np.pi)
    ub = np.ones(npars) * np.pi
    out = ee(func, lb, ub, processes=8)

Note that :func:`~pyeee.std_io.sub_names_params_files` writes
`parameterfile+'.'+ipid` and does not work with `'pid':False`.

If you cannot change your computational model, you can use, for
example, a bash script that launches each model run in a separate
directory, like:

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
    from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
    ishi = './ishiexe.sh'
    parameterfile = 'params.txt'
    objectivefile = 'obj.txt'
    func = partial(exe_wrapper, ishi,
                   parameterfile, standard_parameter_writer,
		   objectivefile, standard_objective_reader,
		   {'pid':True, 'shell':True})
    npars = 3
    lb = np.ones(npars) * (-np.pi)
    ub = np.ones(npars) * np.pi
    from pyeee import ee
    out = ee(func, lb, ub, processes=8)

Such a script could be written in Python as well, of course, if the
bash shell is not available, e.g. on Windows.
    
That's all Folks!
