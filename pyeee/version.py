#!/usr/bin/env python
"""
Provide version number for pyeee library.

This module was written by Matthias Cuntz while at Institut National de Recherche
pour l'Agriculture, l'Alimentation et l'Environnement (INRAE), Nancy, France.

Copyright (c) 2019-2020 Matthias Cuntz - mc (at) macu (dot) de
Released under the MIT License; see LICENSE file for details.

* Written Oct 2019 by Matthias Cuntz (mc (at) macu (dot) de)
* v0.2, initial PyPI commit, Jan 2020, Matthias Cuntz
* v0.3, cleaning all minor TODOs, Jan 2020, Matthias Cuntz
* v0.4, removed numpy deprecation warnings, Jan 2020, Matthias Cuntz
* v0.4.1, first zenodo release, Jan 2020, Matthias Cuntz
* v0.4.2, second release to trigger zenodo, Jan 2020, Matthias Cuntz
* v0.5, removed bug in exe wrappers and added number of tests, Feb 2020, Matthias Cuntz
* v0.6, fix tests on TravisCI and have tests for all modules, Feb 2020, Matthias Cuntz
* v0.7, systematically logistic_p versions, and keep formatting with substitution functions, Feb 2020, Matthias Cuntz
* v0.8, Split tests into individual files per module, and small bug fixes in tests and error handling, Feb 2020, Matthias Cuntz
* v0.9, Renamed morris.py to morris_method.py and adapted args and kwargs to common names in pyeee, Feb 2020, Matthias Cuntz
* v1.0, Restructured package with functions and utils subpackages, Feb 2020, Matthias Cuntz
* v1.1, Number of final trajectories is argument instead of keyword, Feb 2020, Matthias Cuntz
* v1.2, Sample not only from uniform distribution but allow all distributions of scipy.stats, Apr 2020, Matthias Cuntz
* v2.0, Restructure: take wrappers and I/O from partialwrap, take function directory from jams, Jun 2020, Matthias Cuntz

.. moduleauthor:: Matthias Cuntz
"""
from __future__ import division, absolute_import, print_function

__author__  = "Matthias Cuntz"
__version__ = "2.0"
