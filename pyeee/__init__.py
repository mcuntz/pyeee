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
    functions
    morris_method
    screening
    utils
    version
"""

# sub-packages without dependencies to rest of pyeee
from . import functions
from . import utils

from .version import __version__
# Sampling of optimised trajectories for and calculation of Morris Measures / Elementary Effects
from .morris_method import morris_sampling, elementary_effects
# Sample trajectories, run model and return Morris Elementary Effects
from .screening import screening, ee
# Efficient/Sequential Elementary Effects using screening iteratively
from .eee import eee, see
