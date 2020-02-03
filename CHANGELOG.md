# Changelog

All notable changes after its initial development up to January 2020 (v0.2) are documented in this file.

### v0.6
    - Tests did not work on TravisCI because pyeee not installed: put
      pyeee in PYTHONPATH for tests and in shell script.
    - Added tests for standard IO and documented missing coverage.

### v0.5
    - Added tests for general functions, function and exe wrappers,
      Morris Elementary Effects, SA test functions, screening, and tee
      to increase coverage.
    - Renamed ntsteps to nsteps in eee to be consistent with screening/ee.
    - Change check of logfile in eee: check for string rather than
      file handle to be independent of Python version.
    - Replaced kwarg.pop mechanism in exe wrappers because it removed
      the keywords from subsequent function calls.

### v0.4.2
    - Second release online on Github simply to trigger zenodo.

### v0.4.1
    - First release on zenodo.

### v0.4
    - Replaced numpy.matrix arithmetic with numpy.dot on ndarray in.
      Morris sampling: no numpy deprecation warnings anymore.

### v0.3
    - Added test for see, using logfile and several processes in eee.
    - Added seed keyword to screening/ee.
    - Distinguish iterable and array_like parameter types in all routines.
    - Added verbose keyword to eee / see.
    - Added Elementary Effects (ee) in README.md and Quick usage guide.
    - Corrected error in description of pyeee in setup.py, and set development status to 4 - Beta.

### v0.2
    - Initial release on PyPI.
