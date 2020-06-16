#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

from partialwrap import standard_parameter_reader
from pyeee.functions import ishigami_homma

# read pid if given
import sys
pid = None
if len(sys.argv) > 1:
    pid = sys.argv[1]

# read parameters with standard_parameter_reader
pfile = 'params.txt'
x = standard_parameter_reader(pfile, pid=pid)

# calc function
y = ishigami_homma(x, 1., 3.)

# write objective
ofile = 'obj.txt'
if pid is not None:
    ofile = ofile+'.'+pid
with open(ofile, 'w') as ff:
    print('{:.14e}'.format(y), file=ff)
