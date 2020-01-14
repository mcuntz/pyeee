#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
   pyeee: A Python library providing parameter screening of computational models
          using the Morris method of Elementary Effects or its extension of
          Efficient Elementary Effects by Cuntz, Mai et al. (Water Res Research, 2015).
"""
import os
import codecs
import re

from setuptools import setup, find_packages

# find __version__

def iread(*fparts):
    """ Read file data. """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *fparts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    """Find version without importing module."""
    version_file = iread(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# setup

DOCLINES = "A Python library providing parameter screening of computational models using the Morris method of Elementary Effects or its extension of Efficient Elementary Effects by Cuntz, Mai et al. (Water Res Research, 2015)."
README = open("README.md").read()
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: MacOS",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Hydrology",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development",
    "Topic :: Utilities",
]

VERSION = find_version("pyeee", "version.py")

setup(
    name="pyeee",
    version=VERSION,
    maintainer="Matthias Cuntz",
    maintainer_email="mc@macu.de",
    description=DOCLINES,
    long_description=README,
    long_description_content_type="text/markdown",
    author="Matthias Cuntz",
    author_email="mc@macu.de",
    url="https://github.com/mcuntz/pyeee",
    license="MIT",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "schwimmbad",
    ],
    extras_require={},
    packages=find_packages(exclude=["tests*", "docs*"]),
)
