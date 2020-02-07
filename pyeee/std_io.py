#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Standard parameter reader and writer functions as well as objective reader functions,
including substitution of tags #JA????# or names in files.

Parameter files can be written in some arbitrary standard formats.
Or a prepared parameter file can be read and parameter values replaced.

This module was written by Matthias Cuntz while at Institut National
de Recherche en Agriculture, Alimentation et Environnement (INRAE),
Nancy, France.

Copyright (c) 2016-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Nov 2016 by Matthias Cuntz (mc (at) macu (dot) de), including standard readers and writers
* Added substitution function for #JA????# in prepared parameter files (sub_ja_params_files), Jan 2018, Matthias Cuntz
* Use .pid as suffix for multiprocessing, Feb 2018, Matthias Cuntz
* Return ndarrays instead of lists and line IDs in standard_parameter_reader, Mar 2018, Matthias Cuntz
* Added substitution functions for arbitrary names in parameter files (sub_names_params_files*), Mar 2018, Matthias Cuntz
* Allow substitution in several parameter files (msub_files), Apr 2018, Matthias Cuntz
* Changed to Sphinx docstring and numpydoc, Dec 2019, Matthias Cuntz
* Renamed standard_parameter_reader/_writer to standard_parameter_reader_bounds_mask/standard_parameter_writer_bounds_mask, Jan 2020, Matthias Cuntz
* New standard_parameter_reader/_writer that read/write single values per parameter, Jan 2020, Matthias Cuntz
* Swapped names and params in call to sub_names_params_files* to be compatible with new generic exe_wrapper, Jan 2020, Matthias Cuntz
* Call standard_parameter_writer with 2 or 3 arguments, i.e. pid given or not, Jan 2020, Matthias Cuntz
* Make all strings raw strings in sub_names_params_files_* routines to deal with regular expressions, Jan 2020, Matthias Cuntz
* Keep formatting of names and spaces with sub_names_params functions; close input file before raising error, Feb 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz

The following functions are provided

.. autosummary::
   sub_ja_params_files
   sub_names_params_files
   sub_names_params_files_case
   sub_names_params_files_ignorecase
   standard_objective_reader
   standard_parameter_reader
   standard_parameter_writer
   standard_parameter_reader_bounds_mask
   standard_parameter_writer_bounds_mask
   standard_timeseries_reader
   standard_time_series_reader
"""
import re
import numpy as np


__all__ = ['sub_ja_params_files',
           'sub_names_params_files', 'sub_names_params_files_case', 'sub_names_params_files_ignorecase',
           'standard_objective_reader',
           'standard_parameter_reader', 'standard_parameter_writer',
           'standard_parameter_reader_bounds_mask', 'standard_parameter_writer_bounds_mask',
           'standard_time_series_reader', 'standard_timeseries_reader']


# ------------------------------------------------------------------------------


def msub(dic, text, flags=0):
    """
    Helper function for substituting several patterns in one string.

    pattern/replacement are given as dictionary: d[pattern] = replacement

    Parameters
    ----------
    dic : dict
        Pattern/replacement dictionary: `dic[pattern] = replacement`
    text : string
        String on which patterns will be replaced
    flags : int
        Flags will be passed to `re.sub()`

    Returns
    -------
    text : string
        Input string text with replaced patterns

    Notes
    -----
    Compiled version as in
        http://code.activestate.com/recipes/81330-single-pass-multiple-replace/
    and
        https://www.safaribooksonline.com/library/view/python-cookbook-2nd/0596007973/ch01s19.h
    do not work because match.string returns the string and not the pattern so that the key
    for the dictionary does not work anymore.

    History
    -------
    Written,  Matthias Cuntz, Mar 2018
    Modified, Matthias Cuntz, Dec 2019 - Sphinx docstring
    """
    for d in dic:
        text = re.sub(d, dic[d], text, flags=flags)
    return text


# ------------------------------------------------------------------------------


def msub_files(files, dic, pid, flags=0):
    """
    Helper function for applying replacement dictionary on several files.
    pattern/replacement are given as dictionary: d[pattern] = replacement

    Parameters
    ----------
    files : list
        List with file names in which pattern replacement will be applied on each line.
    dic : dict
        Pattern/replacement dictionary: dic[pattern] = replacement
    pid : int
        Output files will be input files suffixed by .pid
    flags : int
        Flags will be passed to re.sub()

    Returns
    -------
    files :
        No return value but output files with names of the input files suffixed by .pid, in which all
        occurences of all patterns were replaced.

    History
    -------
    Written,  Matthias Cuntz, Apr 2018
    Modified, Matthias Cuntz, Dec 2019 - Sphinx docstring
    """
    from os.path import exists

    for f in files:
        if not exists(f): raise IOError('File does not exist: '+f)
        tt = open(f).read()
        tt = msub(dic, tt, flags=flags)
        ff = open(f+'.'+str(pid), 'w')
        ff.write(tt)
        ff.close()

    return


# ------------------------------------------------------------------------------


def sub_ja_params_files(files, pid, params):
    """
    Substitute #JA????# with parameter value in several files, i.e.

        #JA0000# with params[0]

        #JA0001# with params[1]

        ...

    Parameters
    ----------
    files : list
        List with file names in which #JA????# will be replaced.
    pid : int
        Output files will be input files suffixed by .pid
    params : iterable
        Parameter values to replace #JA????# patterns.

        params[0] will replace #JA0000#

        params[1] will replace #JA0001#

        ...

    Returns
    -------
    None
        No return value but output files with names of the input files suffixed by .pid,
        in which all #JA????# patterns were replaced by params elements.

    Examples
    --------
    >>> sub_ja_params_files([file1, file2], 1234, [0, 1, 2, 3])


    History
    -------
    Written,  Matthias Cuntz, Jan 2018
    Modified, Matthias Cuntz, Feb 2018 - pid
              Matthias Cuntz, Mar 2018 - use msub
              Matthias Cuntz, Apr 2018 - use msub_files
              Matthias Cuntz, Dec 2019 - Sphinx docstring
    """
    # assert list of files
    if isinstance(files, str): files = [files]

    # make dict for replacement
    dd = {}
    for i, p in enumerate(params):
        k = "#JA{:04d}#".format(i)
        dd[k] = "{:.14e}".format(params[i])

    # replace in each file
    msub_files(files, dd, pid)

    return


# ------------------------------------------------------------------------------


def sub_names_params_files_case(files, pid, params, names):
    """
    Substitute `name = .*` with `name = parameter` value in several files, i.e.

        `names[i] = params[i]`

    Note, `names` are case sensitive.

    Parameters
    ----------
    files : list
        List with file names in which #JA????# will be replaced.
    pid : int
        Output files will be input files suffixed by .pid
    params : iterable
        Parameter values to be given to variables on the right of = sign

        Parameter values to replace #JA????# patterns.

        Variable in names[0] will be assigned value in params[0]

        Variable in names[1] will be assigned value in params[1]

        ...

    names : iterable
        Variable names on left of = sign in files

    Returns
    -------
    None
        No return value but output files with names of the input files suffixed by .pid,
        in which all variables given in names are assigned the values in given in params.

    Examples
    --------
    >>> sub_names_params_files_case([file1, file2], 1234, [0, 1, 2, 3], ['param1', 'param2', 'param3', 'param4'])


    History
    -------
    Written,  Matthias Cuntz, Mar 2018
    Modified, Matthias Cuntz, Apr 2018 - use msub_files
              Matthias Cuntz, Dec 2019 - Sphinx docstring
              Matthias Cuntz, Jan 2020 - swap names and params in argument list
              Matthias Cuntz, Jan 2020 - make all raw strings for regular expressions
              Matthias Cuntz, Feb 2020 - keep formatting of names and spaces
    """
    # assert list of files
    if isinstance(files, str): files = [files]

    # make dict for msub with dict[pattern] = replacement
    dd = {}
    for i, p in enumerate(params):
        nep = r"("+names[i]+r"\s*)=\s*[a-zA-Z0-9_.+-]*" # name = value
        k = r"^(\s*)"+nep                               # beginning of line
        dd[k] = r"\1\2= {:.14e}".format(params[i])      # replacement using substitutions \1, \2, ...
        k = r"(\n+\s*)"+nep                             # after newline
        dd[k] = r"\1\2= {:.14e}".format(params[i])

    # replace in each file
    msub_files(files, dd, pid)

    return


def sub_names_params_files_ignorecase(files, pid, params, names):
    """
    Substitute `name = .*` with `name = parameter` value in several files, i.e.

        `names[i] = params[i]`

    Note, `names` are case insensitive.

    Parameters
    ----------
    files : list
        List with file names in which #JA????# will be replaced.
    pid : int
        Output files will be input files suffixed by .pid
    params : iterable
        Parameter values to be given to variables on the right of = sign

        Parameter values to replace #JA????# patterns.

        Variable in names[0] will be assigned value in params[0]

        Variable in names[1] will be assigned value in params[1]

        ...

    names : iterable
        Variable names on left of = sign in files

    Returns
    -------
    None
        No return value but output files with names of the input files suffixed by .pid,
        in which all variables given in names are assigned the values in given in params.

    Examples
    --------
    >>> sub_names_params_files_ignorecase([file1, file2], 1234, [0, 1, 2, 3], ['param1', 'param2', 'param3', 'param4'])


    History
    -------
    Written,  Matthias Cuntz, Mar 2018
    Modified, Matthias Cuntz, Apr 2018 - use msub_files
              Matthias Cuntz, Dec 2019 - Sphinx docstring
              Matthias Cuntz, Jan 2020 - swap names and params in argument list
              Matthias Cuntz, Jan 2020 - make all raw strings for regular expressions
              Matthias Cuntz, Feb 2020 - keep formatting of names and spaces
    """
    # assert list of files
    if isinstance(files, str): files = [files]

    # make dict for msub with dict[pattern] = replacement
    dd = {}
    for i, p in enumerate(params):
        nep = r"("+names[i]+r"\s*)=\s*[a-zA-Z0-9_.+-]*" # name = value
        k = r"^(\s*)"+nep                               # beginning of line
        dd[k] = r"\1\2= {:.14e}".format(params[i])      # replacement using substitutions \1, \2, ...
        k = r"(\n+\s*)"+nep                             # after newline
        dd[k] = r"\1\2= {:.14e}".format(params[i])

    # replace in each file
    msub_files(files, dd, pid, flags=re.I)

    return


def sub_names_params_files(*args, **kwargs):
    """
    Wrapper for :any:`sub_names_params_files_ignorecase`.
    """
    return sub_names_params_files_ignorecase(*args, **kwargs)


# ------------------------------------------------------------------------------


def standard_objective_reader(filename):
    """
    Standard objective reader.

    The standard objective reader (if objectivereader=None) reads a single value from a file
    without header, comment line or similar.

    That means for example:

        0.0123456789e-02

    Parameters
    ----------
    filename : string
        Filename of with objective value

    Returns
    -------
    float
        Single number read from filename

    Examples
    --------
    >>> subprocess.call(model)
    >>> obj = standard_objective_reader(filename)


    History
    -------
    Written,  Matthias Cuntz, Nov 2016
    Modified, Matthias Cuntz, Dec 2019 - Sphinx docstring
    """
    # read objective value
    f = open(filename, 'r')
    obj = f.readline()
    f.close()

    # return float
    return np.float(obj)


# ------------------------------------------------------------------------------


def standard_parameter_reader(filename):
    """
    Read standard parameter file.

    The standard parameter file is a file containing
    1 line per parameter with the parameter value:

    Lines starting with # will be excluded.

    That means a standard parameter file might look like:

        #par

        3.000000000000000e-01

        2.300000000000000e-01

        1.440000000000000e+01

        3.000000000000000e-01

        ...

    Parameters
    ----------
    filename : string
        Filename with parameter values

    Returns
    -------
    ndarray
        Parameter values

    Examples
    --------
    >>> params = standard_parameter_reader(paramfile)


    History
    -------
    Written,  Matthias Cuntz, Jan 2020
    """
    params = []
    f = open(filename, 'r')
    for line in f:
        if line.startswith('#'): continue
        params.append(np.float(line.strip()))
    f.close()

    return np.array(params, dtype=np.float)


# ------------------------------------------------------------------------------


def standard_parameter_writer(filename, pid, params=None):
    """
    Standard parameter writer.

    The standard parameter writer writes a file containing
    1 line per parameter with the parameter value.

    All values will be written in IEEE double precision: {:.14e}.

    That means:

        3.000000000000000e-01

        2.300000000000000e-01

        1.440000000000000e+01

        3.000000000000000e-01

        ...

    Parameters
    ----------
    filename : string
        Output filename with parameter values
    pid : int
        Output file will be filename.pid
    params : iterable
        Parameter values

        If standard_parameter_writer is called with two arguments, then the second argument will be params.

    Returns
    -------
    None
        No return value but output file written: filename.pid

    Examples
    --------
    >>> randst = np.random.RandomState()
    >>> pid = str(randst.randint(2147483647))
    >>> params = sample_parameter(pis, pmin, pmax, pmask)
    >>> standard_parameter_writer(paramfile, pid, params)


    History
    -------
    Written,  Matthias Cuntz, Jan 2020
    Modified, Matthias Cuntz, Jan 2020 - call with 2 or 3 arguments, i.e. pid given or not
    """
    # Existing file will be overwritten
    if params is None:
        ofile = filename
        iparams = pid
    else:
        ofile = filename+'.'+str(pid)
        iparams = params
    f = open(ofile, 'w')
    for pp in iparams:
        dstr = '{:.14e}'.format(pp)
        print(dstr, file=f)
    f.close()

    return


# ------------------------------------------------------------------------------


def standard_parameter_reader_bounds_mask(filename):
    """
    Read standard parameter file with parameter bounds and mask.

    The standard parameter file is a space separated file containing
    1 line per parameter with the following columns:

    identifier, current parameter value, minimum parameter value,
    maximum parameter value, parameter mask (1: include, 0: exclude).

    Lines starting with # will be excluded.

    That means a standard parameter file might look like:

        # value min max mask

        1 3.000000000000000e-01 0.000000000000000e+00 1.000000000000000e+00 1

        2 2.300000000000000e-01 -1.000000000000000e+00 1.000000000000000e+00 1

        3 1.440000000000000e+01 9.000000000000000e+00 2.000000000000000e+01 1

        4 3.000000000000000e-01 0.000000000000000e+00 1.000000000000000e+00 0

        ...

    Parameters
    ----------
    filename : string
        Filename with parameter values

    Returns
    -------
    list
        List with ndarrays of

            ids - identifier

            params - parameter values

            pmin - minimum parameter value

            pmax - maximum parameter value

            mask - parameter mask (1: include, 0: exclude from optimisation)

    Examples
    --------
    >>> ids, params, pmin, pmax, pmask = standard_parameter_reader_bounds_mask(paramfile)


    History
    -------
    Written,  Matthias Cuntz, Nov 2016
    Modified, Matthias Cuntz, Mar 2018 - return ids
                                       - return numpy.arrays
              Matthias Cuntz, Dec 2019 - Sphinx docstring
              Matthias Cuntz, Jan 2020 - renamed from standard_parameter_reader to standard_parameter_reader_bounds_mask
              Matthias Cuntz, Feb 2020 - close file before raising error
    """
    ids    = []
    params = []
    pmin   = []
    pmax   = []
    pmask  = []
    f = open(filename, 'r')
    for line in f:
        l = line.strip()
        if l.startswith('#'): continue
        el = l.split()
        if len(el) != 5:
            f.close()
            raise IOError('Line has no 5 columns for parameter: '+line)
        ids.append(el[0])
        params.append(np.float(el[1]))
        pmin.append(np.float(el[2]))
        pmax.append(np.float(el[3]))
        pmask.append(int(el[4]))
    f.close()

    return [ids, np.array(params, dtype=np.float), np.array(pmin, dtype=np.float),
            np.array(pmax, dtype=np.float), np.array(pmask, dtype=np.bool)]


# ------------------------------------------------------------------------------


def standard_parameter_writer_bounds_mask(filename, pid, params, pmin, pmax, mask):
    """
    Standard parameter writer with parameter bounds and mask.

    The standard parameter writer writes a space separated file containing
    1 header line (# value min max mask) plus 1 line per parameter with the following columns:

    consecutive parameter number, current parameter value, minimum parameter value,
    maximum parameter value, parameter mask (1: include, 0: exclude).

    All values will be written in IEEE double precision: {:.14e}.

    That means:

        # value min max mask

        1 3.000000000000000e-01 0.000000000000000e+00 1.000000000000000e+00 1

        2 2.300000000000000e-01 -1.000000000000000e+00 1.000000000000000e+00 1

        3 1.440000000000000e+01 9.000000000000000e+00 2.000000000000000e+01 1

        4 3.000000000000000e-01 0.000000000000000e+00 1.000000000000000e+00 0

        ...

    Parameters
    ----------
    filename : string
        Output filename with parameter values
    pid : int
        Output file will be filename.pid if pid is not None
    params : iterable
        Parameter values
    pmin : iterable
        Minimum parameter values
    pmax : iterable
        Maximum parameter values
    mask : iterable
       Parameter mask (1: include, 0: exclude from optimisation)

    Returns
    -------
    None
        No return value but output file written: filename.pid

    Examples
    --------
    >>> randst = np.random.RandomState()
    >>> pid = str(randst.randint(2147483647))
    >>> params = sample_parameter(pis, pmin, pmax, pmask)
    >>> standard_parameter_writer_bounds_mask(paramfile, pid, params, pmin, pmax, pmask)


    History
    -------
    Written,  Matthias Cuntz, Nov 2016
    Modified, Matthias Cuntz, Feb 2018 - pid
              Matthias Cuntz, Dec 2019 - Sphinx docstring
              Matthias Cuntz, Jan 2020 - renamed from standard_parameter_writer to standard_parameter_writer_bounds_mask
                                       - no .pid to filename if pid is None
    """
    # Assert correct call
    assert len(params) == len(pmin), 'Parameter and minima do not have the same length.'
    assert len(params) == len(pmax), 'Parameter and maxima do not have the same length.'
    assert len(params) == len(mask), 'Parameter and mask do not have the same length.'

    # Convert mask to integer if boolean
    pmask = [ int(i) for i in mask ]

    # Existing file will be overwritten
    if pid is None:
        f = open(filename, 'w')
    else:
        f = open(filename+'.'+str(pid), 'w')
    # header
    hstr = '# value min max mask'
    print(hstr, file=f)
    # data
    for i in range(len(params)):
        dstr = '{:d} {:.14e} {:.14e} {:.14e} {:d}'.format(i+1, params[i], pmin[i], pmax[i], pmask[i])
        print(dstr, file=f)
    f.close()

    return


# ------------------------------------------------------------------------------


def standard_time_series_reader(filename):
    """
    Standard reader for time series.

    The standard time series reader reads a time series of arbitrary length
    from a file without header, comment line or similar.

    That means for example:

        0.0123456789e-02

        0.1234567890e-02

        0.2345678900e-02

        0.3456789000e-02

        0.4567890000e-02

        ...

    Parameters
    ----------
    filename : string
        Filename of with time series values

    Returns
    -------
    timeseries : ndarray
        ndarray with values of each line in filename

    Examples
    --------
    >>> subprocess.call(model)
    >>> ts = standard_time_series_reader(filename)


    History
    -------
    Written,  Matthias Cuntz, Jan 2018
    Modified, Matthias Cuntz, Dec 2019 - Sphinx docstring
    """
    # read objective value
    f = open(filename, 'r')
    serie = f.readlines()
    f.close()

    # return float array
    return np.array(serie, dtype=np.float)

def standard_timeseries_reader(filename):
    """
    Wrapper for :any:`standard_time_series_reader`
    """
    return standard_time_series_reader(filename)


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
