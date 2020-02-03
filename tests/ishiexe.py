#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

from pyeee import standard_parameter_reader, ishigami_homma
# from std_io import standard_parameter_reader
# from sa_test_functions import ishigami_homma

# read pid if given
import sys
pid = None
if len(sys.argv) > 1:
    pid = sys.argv[1]

# read parameters with standard_parameter_reader of std_io.py
pfile = 'params.txt'
if pid is not None:
    pfile = pfile+'.'+pid
x = standard_parameter_reader(pfile)

# calc function
y = ishigami_homma(x, 1., 3.)

# write objective
ofile = 'obj.txt'
if pid is not None:
    ofile = ofile+'.'+pid
ff = open(ofile, 'w')
print('{:.14e}'.format(y), file=ff)
ff.close()
