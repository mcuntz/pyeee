#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for pyeee.
    It checks the functions given in Figure 1 of
    Cuntz, Mai et al. (Water Res Research, 2015).
"""
import unittest

#
# eee.py
class TestEee(unittest.TestCase):
    
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
                  ntfirst=self.ntfirst, ntlast=self.ntlast, ntsteps=self.ntsteps,
                  processes=1)

        # Check
        self.assertCountEqual(list(np.where(out)[0]+1), [2, 3, 4, 6])

        
    # Gstar function with different interactions
    def test_see_gstar(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, eee, see
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
    def test_eee_k(self):
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
    def test_eee_morris(self):
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

#
# general_functions.py
class TestGeneralFunctions(unittest.TestCase):

    def test_general_functions(self):
        import os
        import numpy as np
        from pyeee import logistic, logistic_p, logistic_offset, logistic_offset_p
        from pyeee import logistic2_offset, logistic2_offset_p
        from pyeee import dlogistic, dlogistic_offset, dlogistic2_offset
        
        self.assertEqual(logistic(1.,  1., 0., 2.), 0.5)
        self.assertEqual(logistic(1.,  1., 2., 1.), 0.5)
        self.assertEqual(logistic(2.,  1., 1., 1.), 1./(1.+np.exp(-1.)))
        self.assertEqual(logistic_p(1., [1., 0., 2.]), 0.5)
        self.assertEqual(logistic_p(1., [1., 2., 1.]), 0.5)
        self.assertEqual(logistic_p(2., [1., 1., 1.]), 1./(1.+np.exp(-1.)))
        self.assertEqual(logistic_offset(1.,  1., 0., 2., 1.), 1.5)
        self.assertEqual(logistic_offset(1.,  1., 2., 1., 1.), 1.5)
        self.assertEqual(logistic_offset(2.,  1., 1., 1., 1.), 1./(1.+np.exp(-1.)) + 1.)
        self.assertEqual(logistic_offset_p(1., [1., 0., 2., 1.]), 1.5)
        self.assertEqual(logistic_offset_p(1., [1., 2., 1., 1.]), 1.5)
        self.assertEqual(logistic_offset_p(2., [1., 1., 1., 1.]), 1./(1.+np.exp(-1.)) + 1.)
        self.assertEqual(logistic2_offset(1.,  1., 2., 1.,  2., 2., 1.,  1.), 0.5)
        self.assertEqual(logistic2_offset_p(1., [1., 2., 1.,  2., 2., 1.,  1.]), 0.5)
        self.assertEqual(dlogistic(1.,  1., 2., 1.), 0.5)
        self.assertEqual(dlogistic_offset(1.,  1., 2., 1., 1.), 0.5)
        self.assertEqual(dlogistic2_offset(1.,  1., 2., 1.,  2., 2., 1.,  1.), -0.5)

#
# sa_test_functions.py
class TestSATestFunctions(unittest.TestCase):

    def test_sa_test_functions(self):
        import os
        import numpy as np
        from pyeee import B, g, G, Gstar, K, bratley, oakley_ohagan, ishigami_homma
        from pyeee import linear, product, ratio, ishigami_homma_easy

        self.assertEqual(B(np.arange(10)), 80)
        self.assertEqual(g(np.ones(5), np.zeros(5)), 32.0)
        self.assertEqual(G(np.ones(5), np.zeros(5)), 32.0)
        self.assertEqual(Gstar(np.ones(5), np.zeros(5), np.ones(5), np.zeros(5)), 1.0)
        self.assertEqual(K(np.arange(5)+1.), -101.0)
        self.assertEqual(bratley(np.arange(5)+1.), -101.0)
        self.assertEqual(oakley_ohagan(np.zeros(15)), 15.75)
        self.assertEqual(ishigami_homma([np.pi/2.,np.pi/2.,1.], 1., 1.), 3.0)
        self.assertEqual(linear(np.ones(1), 1., 1.), 2.0)
        self.assertEqual(product(np.arange(2)+1.), 2.0)
        self.assertEqual(ratio(np.arange(2)+1.), 0.5)
        self.assertEqual(ishigami_homma_easy([np.pi/2.,1.]), 2.0)

        self.assertEqual(list(B(np.arange(12).reshape(6,2))), [56, 89])
        self.assertEqual(list(g(np.ones((5,2)), np.zeros(5))), [32.0, 32.0])
        self.assertEqual(list(G(np.ones((5,2)), np.zeros(5))), [32.0, 32.0])
        self.assertEqual(list(Gstar(np.ones((5,2)), np.zeros(5), np.ones(5), np.zeros(5))), [1.0, 1.0])
        self.assertEqual(list(K(np.arange(8).reshape((4,2))+1.)), [92., 342.])
        self.assertEqual(list(bratley(np.arange(8).reshape((4,2))+1.)), [92., 342.])
        self.assertEqual(list(oakley_ohagan(np.zeros((15,2)))), [15.75, 15.75])
        self.assertEqual(list(ishigami_homma([[np.pi/2.,np.pi/2.],[np.pi/2.,np.pi/2.],[1.,1.]], 1., 1.)), [3.0, 3.0])
        self.assertEqual(list(linear(np.ones((1,2)), 1., 1.)), [2.0, 2.0])
        self.assertEqual(list(product(np.arange(4).reshape((2,2))+1.)), [3.0, 8.0])
        self.assertEqual(list(ratio(np.arange(2).repeat(2).reshape((2,2))+1.)), [0.5, 0.5])
        self.assertEqual(list(ishigami_homma_easy([[np.pi/2.,np.pi/2.],[1.,1.]])), [2.0, 2.0])

#
# screening.py
class TestScreening(unittest.TestCase):
    
    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)
        self.nt      = 10
        self.ntotal  = 50
        self.nsteps  = 6
        self.verbose = 1

    # G function        
    def test_ee_g(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, ee
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

        out = ee(obj, lb, ub, x0=None, mask=None,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:,0],3)), [0.045, 0.24, 1.624, 0.853, 0.031, 0.084])


    # Gstar function with different interactions
    def test_screening_gstar(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, ee, screening
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
        iiout  = [[1.087, 1.807, 0.201, 0.13,  0.08,  0.077, 0.055, 0.198, 0.139, 0.136],
                  [0.924, 1.406, 0.603, 0.876, 1.194, 0.567, 0.642, 0.276, 0.131, 0.34 ],
                  [1.021, 0.875, 0.085, 0.096, 0.116, 0.104, 0.11,  0.07,  0.045, 0.112],
                  [1.096, 0.752, 0.573, 0.762, 0.189, 0.584, 0.259, 0.437, 0.468, 0.169],
                  [4.806, 3.299, 0.614, 0.406, 0.89,  0.628, 0.372, 0.428, 0.64,  0.565],
                  [1.212, 1.076, 3.738, 8.299, 0.636, 0.469, 0.37,  4.856, 0.553, 0.009]
                 ]
        lb = np.zeros(npars)
        ub = np.ones(npars)
            
        for ii in range(len(params)):
            # Partialise function with fixed parameters
            arg   = params[ii]
            kwarg = {}
            obj   = partial(func_wrapper, func, arg, kwarg)

            out = screening(obj, lb, ub, x0=None, mask=None,
                            nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                            processes=1) #, plotfile='gstar'+str(ii)+'.png')
            # Check
            self.assertEqual(list(np.around(out[:,0],3)), iiout[ii])


    # Bratley / K function        
    def test_ee_k(self):
        from functools import partial
        import os
        import numpy as np
        from pyeee import func_wrapper, ee
        from pyeee import bratley

        # Function and parameters
        func   = bratley
        npars  = 10
        params = [] # k

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(func, lb, ub, x0=None, mask=None,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:,0],3)), [0.586, 0.219, 0.082, 0.055, 0.02, 0.068, 0.009, 0.007, 0., 0.])


    # Morris function
    def test_ee_morris(self):
        from functools import partial
        import numpy as np
        from pyeee import func_wrapper, ee
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
        out = ee(obj, lb, ub, x0=None, mask=None,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)),
                              [66.261, 48.363, 20.133, 62.42, 37.87,
                               26.031, 30.211, 39.714, 42.633, 43.776,
                               4.996, 3.701, 4.734, 8.031, 5.734,
                               3.564, 5.068, 7.635, 3.129, 5.224])

#
# tee.py
class TestTee(unittest.TestCase):

    def test_tee(self):
        import os
        from pyeee import tee

        tee('T T T Test 1')
        ff = open('log.txt', 'w')
        tee('T T T Test 2', file=ff)
        ff.close()
        
        self.assertTrue(os.path.exists('log.txt'))

        ff = open('log.txt', 'r')
        inlog = ff.readline()
        ff.close()

        self.assertEqual(inlog.rstrip(), 'T T T Test 2')

        if os.path.exists('log.txt'): os.remove('log.txt')


if __name__ == "__main__":
    unittest.main()
