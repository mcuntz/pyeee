#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Purpose
=======

Generel set of functions such as the logistic function
as well as test functions for Sensitivity Analysis.

:copyright: Copyright 2019-2020 Matthias Cuntz, see AUTHORS.md for details.
:license: MIT License, see LICENSE for details.

Subpackages
===========
.. autosummary::
    general_functions
    sa_test_functions
    utils
"""

__all__ = ['curvature',
           'logistic', 'logistic_p', 'dlogistic', 'dlogistic_p', 'd2logistic', 'd2logistic_p',
           'logistic_offset', 'logistic_offset_p', 'dlogistic_offset', 'dlogistic_offset_p',
           'd2logistic_offset', 'd2logistic_offset_p', 'logistic2_offset', 'logistic2_offset_p',
           'dlogistic2_offset', 'dlogistic2_offset_p', 'd2logistic2_offset', 'd2logistic2_offset_p',
           'B', 'g', 'G', 'Gstar', 'bratley', 'K', 'fmorris', 'morris', 'oakley_ohagan', 'ishigami_homma',
           'linear', 'product', 'ratio', 'ishigami_homma_easy',
           'cost_square']

# Common functions that are used in curve_fit or fmin parameter estimations
from .general_functions import curvature
from .general_functions import logistic, logistic_p, dlogistic, dlogistic_p, d2logistic, d2logistic_p
from .general_functions import logistic_offset, logistic_offset_p, dlogistic_offset, dlogistic_offset_p
from .general_functions import d2logistic_offset, d2logistic_offset_p, logistic2_offset, logistic2_offset_p
from .general_functions import dlogistic2_offset, dlogistic2_offset_p, d2logistic2_offset, d2logistic2_offset_p
# Test functions for Sensitivity Analysis
from .sa_test_functions import B, g, G, Gstar, bratley, K, fmorris, morris, oakley_ohagan, ishigami_homma
from .sa_test_functions import linear, product, ratio, ishigami_homma_easy
# Utilities
from .utils import cost_square
