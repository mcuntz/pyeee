#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for the Efficient/Sequential Elementary Effects module.

    python -m unittest -v test_eee.py
    python -m pytest --cov pyeee --cov-report term-missing -v tests/
"""
import unittest


# --------------------------------------------------------------------
# eee.py
# Missing coverage:
#    181-184: ImportError MPI 
#    204: crank!=0 <- MPI
#    230-250: mask
#    276-279: weight
#    339-371: plotfile
#    383-385: logfile
#    391: return after step4
#    415-418: weight
#    445: return after step6
#    459: logfile
#    470-473: weight
#    483-489: logfile
#    494-509: no more parameters after screening
#    515: mask
#    524-526: logfile
class TestEee(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)
        self.ntfirst = 10
        self.ntlast  = 5
        self.nsteps  = 6
        self.verbose = 1


    # G function
    def test_eee_g(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, eee
        from pyeee import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.] # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = partial(func_wrapper, func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = eee(obj, lb, ub, mask=None,
                  ntfirst=self.ntfirst, ntlast=self.ntlast, nsteps=self.nsteps,
                  processes=1)

        # Check
        self.assertEqual(list(np.where(out)[0]+1), [2, 3, 4, 6])


    # Gstar function with different interactions
    def test_see_gstar(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, eee, see
        from pyeee import Gstar

        # Function and parameters
        func   = Gstar
        npars  = 10
        params = [[[1]*npars,          np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]], # G*
                  [np.ones(npars),     np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
                  [np.ones(npars)*0.5, np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
                  [np.ones(npars)*0.5, np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
                  [np.ones(npars)*2.0, np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
                  [np.ones(npars)*2.0, np.random.random(npars), [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]]
                 ]
        iiout  = [[1, 2, 3, 8, 9],
                  [1, 2, 3, 4, 5, 6, 7, 8],
                  [1, 2, 3, 7, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 7]
                 ]
        lb = np.zeros(npars)
        ub = np.ones(npars)

        for ii in range(len(params)):
            # Partialise function with fixed parameters
            arg   = params[ii]
            kwarg = {}
            obj   = partial(func_wrapper, func, arg, kwarg)

            out = see(obj, lb, ub, mask=None,
                      ntfirst=self.ntfirst, ntlast=self.ntlast, nsteps=self.nsteps,
                      processes=1, verbose=1) #, plotfile='gstar'+str(ii)+'.png')
            # Check
            self.assertEqual(list(np.where(out)[0]+1), iiout[ii])


    # Bratley / K function
    def test_eee_k(self):
        from functools import partial
        import os
        import numpy as np
        import schwimmbad
        from pyeee import func_wrapper, eee
        from pyeee import bratley

        # Function and parameters
        func   = bratley
        npars  = 10
        params = [] # k

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        nprocs = 4
        ipool = schwimmbad.choose_pool(mpi=False, processes=nprocs)
        out = eee(func, lb, ub, mask=None,
                  ntfirst=self.ntfirst, ntlast=self.ntlast, nsteps=self.nsteps,
                  processes=nprocs, pool=ipool, logfile='tlog.txt')
        ipool.close()

        # Check
        self.assertEqual(list(np.where(out)[0]+1), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(os.path.exists('tlog.txt'))

        # Clean
        if os.path.exists('tlog.txt'): os.remove('tlog.txt')


    # Morris function
    def test_eee_fmorris(self):
        from functools import partial
        import os
        import numpy as np
        from pyeee import func_wrapper, eee
        from pyeee import fmorris

        # Function and parameters
        func = fmorris
        npars = 20
        beta0              = 0.
        beta1              = np.random.standard_normal(npars)
        beta1[:10]         = 20.
        beta2              = np.random.standard_normal((npars,npars))
        beta2[:6,:6]       = -15.
        beta3              = np.zeros((npars,npars,npars))
        beta3[:5,:5,:5]    = -10.
        beta4              = np.zeros((npars,npars,npars,npars))
        beta4[:4,:4,:4,:4] = 5.

        # Partialise Morris function with fixed parameters beta0-4
        arg   = [beta0, beta1, beta2, beta3, beta4]
        kwarg = {}
        obj   = partial(func_wrapper, func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        # Check
        ff = open('tlog.txt', 'w')
        out = eee(obj, lb, ub, mask=None,
                  ntfirst=self.ntfirst, ntlast=self.ntlast, nsteps=self.nsteps,
                  processes=4, logfile=ff, verbose=1)
        ff.close()

        self.assertEqual(list(np.where(out)[0]+1), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 20])
        self.assertTrue(os.path.exists('tlog.txt'))

        # Clean
        if os.path.exists('tlog.txt'): os.remove('tlog.txt')


if __name__ == "__main__":
    unittest.main()
