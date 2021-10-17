#!/usr/bin/env python
"""
Purpose
=======

pyeee provides parameter screening of computational models
using the Morris method of elementary effects or
the extension of Efficient/Sequential Elementary Effects of
Cuntz, Mai et al. (Water Res Research, 2015).

The package uses several functions of the JAMS Python package
https://github.com/mcuntz/jams_python
The JAMS package and hesseflux are synchronised irregularly.

:copyright: Copyright 2019-2021 Matthias Cuntz, see AUTHORS.md for details.
:license: MIT License, see LICENSE for details.

Subpackages
===========
.. autosummary::
    eee
    version
"""
from __future__ import division, absolute_import, print_function
from .version import __version__, __author__
from .eee import eee, see


__all__ = ['__version__', '__author__', 'eee', 'see']
