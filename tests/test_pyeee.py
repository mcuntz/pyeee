#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for pyeee.
    It checks the functions given in Figure 1 of
    Cuntz, Mai et al. (Water Res Research, 2015).
"""
import unittest

class Test(unittest.TestCase):
    
    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)
        self.ntfirst = 10
        self.ntlast  = 5
        self.ntsteps = 6
        self.verbose = 1


    # G function        
    def test_g(self):
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
                  ntfirst=self.ntfirst, ntlast=self.ntlast, ntsteps=self.ntsteps,
                  processes=1)

        # Check
        self.assertCountEqual(list(np.where(out)[0]+1), [2, 3, 4, 6])

        
    # Gstar function with different interactions
    def test_gstar(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, eee
        from pyeee import Gstar

        # Function and parameters
        func   = Gstar
        npars  = 10
        params = [[np.ones(npars),     np.random.random(npars), [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]], # G*
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
                      ntfirst=self.ntfirst, ntlast=self.ntlast, ntsteps=self.ntsteps,
                      processes=1) #, plotfile='gstar'+str(ii)+'.png')
            # Check
            self.assertCountEqual(list(np.where(out)[0]+1), iiout[ii])


    # Bratley / K function        
    def test_k(self):
        from functools import partial
        import os
        import numpy as np
        from pyeee import func_wrapper, eee
        from pyeee import bratley

        # Function and parameters
        func   = bratley
        npars  = 10
        params = [] # k

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = eee(func, lb, ub, mask=None,
                  ntfirst=self.ntfirst, ntlast=self.ntlast, ntsteps=self.ntsteps,
                  processes=1, logfile='tlog.txt')

        # Check
        self.assertCountEqual(list(np.where(out)[0]+1), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(os.path.exists('tlog.txt'))

        # Clean
        if os.path.exists('tlog.txt'): os.remove('tlog.txt')


    # Morris function
    def test_morris(self):
        from functools import partial
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
        out = eee(obj, lb, ub, mask=None,
                  ntfirst=self.ntfirst, ntlast=self.ntlast, ntsteps=self.ntsteps,
                  processes=4)

        self.assertCountEqual(list(np.where(out)[0]+1), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 20])


if __name__ == "__main__":
    unittest.main()
