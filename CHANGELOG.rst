Changelog
---------

All notable changes after its initial development up to January 2020
(v0.2) are documented in this file.

v4.0 (Feb 2024)
    * Moved all Markdown files to reStructuredText.
    * Moved documentation to sphinxbook.
    * Moved documentation to Github pages.
    * Moved to Github actions.
    * Moved to new pip structure using `pyproject.toml`.

v3.0 (Oct 2021)
    * Use `pyjams` package. Remove all modules, functions, tests, and
      docs of routines that are now in pyjams.
    * Move from travis-ci.org to travis-ci.com.

v2.1 (Sep 2020)
    * Included subpackages const, functions in automatic packaging.
    * Build pure Python wheels without using cibuildwheel.

v2.0 (Jun 2020)
    * Use package partialwrap in docstrings and documentation.
    * Remove utils directory: tee.py is now directly in pyeee
      directory.
    * Sync const and functions of JAMS package.
    * Generalise structure of setup.py.
    * Build only Linux on TravisCI because tests are/were only done on
      Linux.

v1.2 (Apr 2020)
    * Sample not only from uniform distribution but allow all
      distributions of scipy.stats in morris_sampling, screening/eee,
      and eee/see.

v1.1 (Feb 2020)
    * Make number of final trajectories an argument instead of a
      keyword argument in screening/ee.
    * Make number of final trajectories an argument instead of a
      keyword argument and sample by default 10*final trajectories in
      Morris Method, i.e. morris_sampling.

v1.0 (Feb 2020)
    * Restructured package with functions and utils subpackages.

v0.9 (Feb 2020)
    * Added mention to template of Sebastian Mueller in README.md and
      documentation.
    * Renamed morris.py to morris_method.py.
    * Adjusted names of arguments and keyword arguments in
      morris_sampling and elementary_effects to be consistent with
      rest of pyeee.

v0.8 (Feb 2020)
    * Split tests in individual files, one per module.
    * Changed from ValueError to TypeError if function given to exe
      wrappers.
    * InputError does not exist, use TypeError in screening.
    * Use assertRaises for check error handling in tests.
    * Plot diagnostic figures in png files in Morris sampling if
      matplotlib installed.
    * Coverage at maximum except for eee.py.

v0.7 (Feb 2020)
    * Make systematically function_p versions of all logistic
      functions and its derivatives.
    * Keep formatting of names and spaces with sub_names_params
      functions.
    * Close input file before raising error in
      standard_parameter_reader_bounds_mask.
    * Removed missing coverage in function_wrappers, std_io,
      sa_test_functions, and general_functions.

v0.6 (Feb 2020)
    * Tests did not work on TravisCI because pyeee not installed: put
      pyeee in PYTHONPATH for tests and in shell script.
    * Added tests for standard IO and documented missing coverage.

v0.5 (Feb 2020)
    * Added tests for general functions, function and exe wrappers,
      Morris Elementary Effects, SA test functions, screening, and tee
      to increase coverage.
    * Renamed ntsteps to nsteps in eee to be consistent with
      screening/ee.
    * Change check of logfile in eee: check for string rather than
      file handle to be independent of Python version.
    * Replaced kwarg.pop mechanism in exe wrappers because it removed
      the keywords from subsequent function calls.

v0.4.2 (Jan 2020)
    * Second release online on Github, simply to trigger zenodo.

v0.4.1 (Jan 2020)
    * First release on zenodo.

v0.4 (Jan 2020)
    * Replaced numpy.matrix arithmetic with numpy.dot on ndarray in.
      Morris sampling: no numpy deprecation warnings anymore.

v0.3 (Jan 2020)
    * Added test for see, using logfile and several processes in eee.
    * Added seed keyword to screening/ee.
    * Distinguish iterable and array_like parameter types in all
      routines.
    * Added verbose keyword to eee / see.
    * Added Elementary Effects (ee) in README.md and Quick usage
      guide.
    * Corrected error in description of pyeee in setup.py, and set
      development status to 4 - Beta.

v0.2 (Jan 2020)
    * Initial release on PyPI.
