#!/usr/bin/env python
"""Purpose
=======

pyeee provides parameter screening of computational models using the
Morris method of Elementary Effects or the extension of
Efficient/Sequential Elementary Effects of Cuntz, Mai et al. (Water
Res Research, 2015).

:copyright: Copyright 2019-2024 Matthias Cuntz, Juliane Mai; see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

Subpackages
===========
.. autosummary::
   eee

History
    * Written Oct 2019 by Matthias Cuntz (mc (at) macu (dot) de)
    * v0.2, initial PyPI commit, Jan 2020, Matthias Cuntz
    * v0.3, cleaning all minor TODOs, Jan 2020, Matthias Cuntz
    * v0.4, removed numpy deprecation warnings, Jan 2020, Matthias Cuntz
    * v0.4.1, first zenodo release, Jan 2020, Matthias Cuntz
    * v0.4.2, second release to trigger zenodo, Jan 2020, Matthias Cuntz
    * v0.5, removed bug in exe wrappers and added number of tests,
      Feb 2020, Matthias Cuntz
    * v0.6, fix tests on TravisCI and have tests for all modules,
      Feb 2020, Matthias Cuntz
    * v0.7, systematically logistic_p versions, and keep formatting with
      substitution functions, Feb 2020, Matthias Cuntz
    * v0.8, split tests into individual files per module, and small bug fixes
      in tests and error handling, Feb 2020, Matthias Cuntz
    * v0.9, renamed morris.py to morris_method.py and adapted args and kwargs
      to common names in pyeee, Feb 2020, Matthias Cuntz
    * v1.0, restructured package with functions and utils subpackages,
      Feb 2020, Matthias Cuntz
    * v1.1, number of final trajectories is argument instead of keyword,
      Feb 2020, Matthias Cuntz
    * v1.2, sample not only from uniform distribution but allow all
      distributions of scipy.stats, Apr 2020, Matthias Cuntz
    * v2.0, restructure: take wrappers and I/O from partialwrap, take function
      directory from jams, Jun 2020, Matthias Cuntz
    * v2.1, include subpackages in const, functions in setup.py,
      Sep 2020, Matthias Cuntz
    * v3.0, use pyjams package, Oct 2021, Matthias Cuntz
    * v4.0, modernise code structure and documentation,
      Feb 2024, Matthias Cuntz
    * v4.1, add pyeee to conda-forge, Mar 2024, Matthias Cuntz
    * v5.0, remove dependency to pyjams, Apr 2024, Matthias Cuntz

"""
# version, author
try:
    from ._version import __version__
except ImportError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"
__author__  = "Matthias Cuntz, Juliane Mai"

# Efficient Elementary Effects
from .eee import eee, see
# helper functions
from .functions import cost_square, curvature
# logistic functions with first and second derivatives
from .logistic_function import logistic, logistic_p
from .logistic_function import dlogistic, dlogistic_p
from .logistic_function import d2logistic, d2logistic_p
from .logistic_function import logistic_offset, logistic_offset_p
from .logistic_function import dlogistic_offset, dlogistic_offset_p
from .logistic_function import d2logistic_offset, d2logistic_offset_p
from .logistic_function import logistic2_offset, logistic2_offset_p
from .logistic_function import dlogistic2_offset, dlogistic2_offset_p
from .logistic_function import d2logistic2_offset, d2logistic2_offset_p
# Morris' Method
from .morris_method import morris_sampling, elementary_effects
from . screening import screening, ee
# like *nix tee utility
from .tee import tee


__all__ = ['eee', 'see',
           'cost_square', 'curvature',
           'logistic', 'logistic_p',
           'dlogistic', 'dlogistic_p',
           'd2logistic', 'd2logistic_p',
           'logistic_offset', 'logistic_offset_p',
           'dlogistic_offset', 'dlogistic_offset_p',
           'd2logistic_offset', 'd2logistic_offset_p',
           'logistic2_offset', 'logistic2_offset_p',
           'dlogistic2_offset', 'dlogistic2_offset_p',
           'd2logistic2_offset', 'd2logistic2_offset_p',
           'morris_sampling', 'elementary_effects',
           'screening', 'ee',
           'tee']
