#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Purpose
=======

pyeee provides parameter screening of computational models
using the Morris method of elementary effects or
the extension of Efficient/Sequential Elementary Effects of
Cuntz, Mai et al. (Water Res Research, 2015).

:copyright: Copyright 2019 Matthias Cuntz, see AUTHORS.md for details.
:license: MIT License, see LICENSE for details.

Subpackages
===========
.. autosummary::
    eee
    function_wrapper
    general_functions
    morris_method
    sa_test_functions
    screening
    tee
    utils
    version
"""
from .version import __version__
# Common functions that are used in curve_fit or fmin parameter estimations
from .general_functions import curvature
from .general_functions import logistic, logistic_p, dlogistic, dlogistic_p, d2logistic, d2logistic_p
from .general_functions import logistic_offset, logistic_offset_p, dlogistic_offset, dlogistic_offset_p
from .general_functions import d2logistic_offset, d2logistic_offset_p, logistic2_offset, logistic2_offset_p
from .general_functions import dlogistic2_offset, dlogistic2_offset_p, d2logistic2_offset, d2logistic2_offset_p
# Sensitivity analysis test functions
from .sa_test_functions import B, g, G, Gstar, bratley, K, fmorris, morris, oakley_ohagan, ishigami_homma
from .sa_test_functions import linear, product, ratio, ishigami_homma_easy
# Utilities
from .utils import cost_square
from .tee   import tee
# Function wrappers to be used with partial from functools
from .function_wrapper import exe_wrapper, exe_mask_wrapper
from .function_wrapper import func_wrapper, func_mask_wrapper
# Standard parameter reader and writer functions as well as objective reader functions
from .std_io import sub_ja_params_files
from .std_io import sub_names_params_files, sub_names_params_files_case, sub_names_params_files_ignorecase
from .std_io import standard_objective_reader
from .std_io import standard_parameter_reader, standard_parameter_writer
from .std_io import standard_parameter_reader_bounds_mask, standard_parameter_writer_bounds_mask
from .std_io import standard_time_series_reader, standard_timeseries_reader
# Sampling of optimised trajectories for and calculation of Morris Measures / Elementary Effects
from .morris_method import morris_sampling, elementary_effects
# Sample trajectories, run model and return Morris Elementary Effects
from .screening import screening, ee
# Efficient/Sequential Elementary Effects using screening iteratively
from .eee import eee, see
