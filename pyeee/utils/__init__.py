#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
Purpose
=======

Utilities used with and within pyeee.

:copyright: Copyright 2019-2020 Matthias Cuntz, see AUTHORS.md for details.
:license: MIT License, see LICENSE for details.

Subpackages
===========
.. autosummary::
    function_wrapper
    std_io
    tee
"""

__all__ = ['tee',
           'exe_wrapper', 'exe_mask_wrapper',
           'func_wrapper', 'func_mask_wrapper',
           'sub_ja_params_files',
           'sub_names_params_files', 'sub_names_params_files_case', 'sub_names_params_files_ignorecase',
           'standard_objective_reader',
           'standard_parameter_reader', 'standard_parameter_writer',
           'standard_parameter_reader_bounds_mask', 'standard_parameter_writer_bounds_mask',
           'standard_time_series_reader', 'standard_timeseries_reader']

# Utilities
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
