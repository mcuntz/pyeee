#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for pyeee.
    It checks the functions given in Figure 1 of
    Cuntz, Mai et al. (Water Res Research, 2015).

    python -m pytest --cov pyeee --cov-report term-missing -v tests/
"""
import unittest

# --------------------------------------------------------------------
# eee.py
# Missing coverage:
#    181-184: ImportError MPI 
#    204: crank!=0
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


# --------------------------------------------------------------------
# function_wrappers.py
class TestFunctionWrapper(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)
        # ee
        self.nt      = 10
        self.ntotal  = 50
        # eee
        self.ntfirst = 10
        self.ntlast  = 5
        # both
        self.nsteps  = 6
        self.verbose = 1


    # function wrapper
    def test_func_wrapper(self):
        from functools import partial
        import numpy as np
        from pyeee import ee, ishigami_homma
        from pyeee import func_wrapper

        func  = ishigami_homma
        npars = 3

        arg   = [1., 3.]
        kwarg = {}
        obj   = partial(func_wrapper, func, arg, kwarg)

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(obj, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # function mask wrapper
    def test_func_mask_wrapper(self):
        from functools import partial
        import numpy as np
        from pyeee import ee, ishigami_homma
        from pyeee import func_mask_wrapper

        func  = ishigami_homma
        npars = 3

        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False

        arg   = [1., 3.]
        kwarg = {}

        obj = partial(func_mask_wrapper, func, x0, mask, arg, kwarg)

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(obj, lb[mask], ub[mask],
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        self.assertEqual(list(np.around(out[:,0],3)), [287.258, 0.])


    # exe wrapper
    def test_exe_wrapper(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'

        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe wrapper, error
    def test_exe_wrapper_error(self):
        from functools import partial
        import numpy as np
        from pyeee import ee, ishigami_homma
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ishigami_homma
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'

        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        try:
            out = ee(func, lb, ub,
                     nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                     processes=1)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


    # exe wrapper, processes
    def test_exe_wrapper4(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe wrapper, debug
    def test_exe_wrapper_debug(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'debug':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe wrapper, processes, pid, debug
    def test_exe_wrapper4_pid_debug(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True, 'debug':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe wrapper, processes, shell
    def test_exe_wrapper4_shell(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = './tests/ishiexe.sh'
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True, 'shell':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe wrapper, processes, shell, debug
    def test_exe_wrapper4_shell_debug(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = './tests/ishiexe.sh'
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        func = partial(exe_wrapper, ishi,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True, 'shell':True, 'debug':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb, ub,
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [369.614, 0.576, 184.987])


    # exe mask wrapper
    def test_exe_mask_wrapper(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_mask_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False
        
        func = partial(exe_mask_wrapper, ishi, x0, mask,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb[mask], ub[mask],
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        self.assertEqual(list(np.around(out[:,0],3)), [287.258, 0.])


    # exe mask wrapper, error
    def test_exe_mask_wrapper_error(self):
        from functools import partial
        import numpy as np
        from pyeee import ee, ishigami_homma
        from pyeee import exe_mask_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ishigami_homma
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False
        
        func = partial(exe_mask_wrapper, ishi, x0, mask,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        try:
            out = ee(func, lb[mask], ub[mask],
                     nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                     processes=1)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


    # exe mask wrapper, processes
    def test_exe_mask_wrapper4(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_mask_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False
        
        func = partial(exe_mask_wrapper, ishi, x0, mask,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb[mask], ub[mask],
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [287.258, 0.])


    # exe mask wrapper, processes, shell, debug
    def test_exe_mask_wrapper4_shell_debug(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_mask_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = './tests/ishiexe.sh'
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False
        
        func = partial(exe_mask_wrapper, ishi, x0, mask,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'pid':True, 'shell':True, 'debug':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb[mask], ub[mask],
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:,0],3)), [287.258, 0.])


    # exe mask wrapper, debug
    def test_exe_mask_wrapper_debug(self):
        from functools import partial
        import numpy as np
        from pyeee import ee
        from pyeee import exe_mask_wrapper, standard_parameter_writer, standard_objective_reader
        
        ishi  = ['python3', 'tests/ishiexe.py']
        npars = 3
        parameterfile = 'params.txt'
        objectivefile = 'obj.txt'
        
        x0   = np.ones(npars)
        mask = np.ones(npars, dtype=np.bool)
        mask[1] = False
        
        func = partial(exe_mask_wrapper, ishi, x0, mask,
                       parameterfile, standard_parameter_writer,
                       objectivefile, standard_objective_reader, {'debug':True})

        lb = np.ones(npars) * (-np.pi)
        ub = np.ones(npars) *   np.pi

        out = ee(func, lb[mask], ub[mask],
                 nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        self.assertEqual(list(np.around(out[:,0],3)), [287.258, 0.])


# --------------------------------------------------------------------
# general_functions.py
class TestGeneralFunctions(unittest.TestCase):

    def test_general_functions(self):
        import os
        import numpy as np
        from pyeee import curvature
        from pyeee import logistic, logistic_offset, logistic2_offset
        from pyeee import dlogistic, dlogistic_offset, dlogistic2_offset
        from pyeee import d2logistic, d2logistic_offset, d2logistic2_offset
        from pyeee import logistic_p, logistic_offset_p, logistic2_offset_p
        from pyeee import dlogistic_p, dlogistic_offset_p, dlogistic2_offset_p
        from pyeee import d2logistic_p, d2logistic_offset_p, d2logistic2_offset_p

        self.assertEqual(logistic(1.,  1., 0., 2.), 0.5)
        self.assertEqual(logistic(1.,  1., 2., 1.), 0.5)
        self.assertEqual(logistic(2.,  1., 1., 1.), 1./(1.+np.exp(-1.)))
        self.assertEqual(logistic_offset(1.,  1., 0., 2., 1.), 1.5)
        self.assertEqual(logistic_offset(1.,  1., 2., 1., 1.), 1.5)
        self.assertEqual(logistic_offset(2.,  1., 1., 1., 1.), 1./(1.+np.exp(-1.)) + 1.)
        self.assertEqual(logistic2_offset(1.,  1., 2., 1.,  2., 2., 1.,  1.), 0.5)
        self.assertEqual(dlogistic(1.,  1., 2., 1.), 0.5)
        self.assertEqual(dlogistic_offset(1.,  1., 2., 1., 1.), 0.5)
        self.assertEqual(dlogistic2_offset(1.,  1., 2., 1.,  2., 2., 1.,  1.), -0.5)        
        self.assertEqual(np.around(d2logistic(1., 1., 2., 2.),4), 0.3199)
        self.assertEqual(np.around(d2logistic_offset(1., 1., 2., 2., 1.),4), 0.3199)
        self.assertEqual(np.around(d2logistic2_offset(1., 1., 2., 2.,  2., 2., 2.,  1.),4), -0.3199)
        self.assertEqual(np.around(curvature(1., dlogistic_offset, d2logistic_offset, 1., 2., 2., 1.),4), 0.2998)

        self.assertEqual(logistic_p(1.,  [1., 0., 2.]), 0.5)
        self.assertEqual(logistic_p(1.,  [1., 2., 1.]), 0.5)
        self.assertEqual(logistic_p(2.,  [1., 1., 1.]), 1./(1.+np.exp(-1.)))
        self.assertEqual(logistic_offset_p(1.,  [1., 0., 2., 1.]), 1.5)
        self.assertEqual(logistic_offset_p(1.,  [1., 2., 1., 1.]), 1.5)
        self.assertEqual(logistic_offset_p(2.,  [1., 1., 1., 1.]), 1./(1.+np.exp(-1.)) + 1.)
        self.assertEqual(logistic2_offset_p(1.,  [1., 2., 1.,  2., 2., 1.,  1.]), 0.5)
        self.assertEqual(dlogistic_p(1.,  [1., 2., 1.]), 0.5)
        self.assertEqual(dlogistic_offset_p(1.,  [1., 2., 1., 1.]), 0.5)
        self.assertEqual(dlogistic2_offset_p(1.,  [1., 2., 1.,  2., 2., 1.,  1.]), -0.5)        
        self.assertEqual(np.around(d2logistic_p(1., [1., 2., 2.]),4), 0.3199)
        self.assertEqual(np.around(d2logistic_offset_p(1., [1., 2., 2., 1.]),4), 0.3199)
        self.assertEqual(np.around(d2logistic2_offset_p(1., [1., 2., 2.,  2., 2., 2.,  1.]),4), -0.3199)
        self.assertEqual(np.around(curvature(1., dlogistic_offset_p, d2logistic_offset_p, [1., 2., 2., 1.]),4), 0.2998)


# --------------------------------------------------------------------
# morris.py
# Missing coverage:
#    409-488: Diagnostics=True
#    612-614: Diagnostics=True
#    647: if NumGroups == 0: if SAm.size > 1: Single trajectory?
class TestMorris(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)
        self.NumFact    = 15
        self.LB         = np.arange(self.NumFact)
        self.UB         = 2. * self.LB + 1.
        self.Diagnostic = 0

    def test_r_10(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        N = 100
        p = 6
        r = 10
        out = np.random.random(r*(self.NumFact+1))

        # Check 1
        mat, vec = morris_sampling(self.NumFact, self.LB, self.UB, N=N, p=p, r=r, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.6, 2.2, 2., 4.6, 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [12., 11., 5., 1., 9.])

        # Check 2
        sa, res = elementary_effects(self.NumFact, mat, vec, out, p=p)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.485, 0.445, 0.438, 0.58, 0.645])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.47, 0.502, 0.816, 0.722, 0.418])


    def test_r_10_nan(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        N = 100
        p = 6
        r = 10
        out = np.random.random(r*(self.NumFact+1))
        out[1:r*self.NumFact:self.NumFact//2] = np.nan

        mat, vec = morris_sampling(self.NumFact, self.LB, self.UB, N=N, p=p, r=r, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.NumFact, mat, vec, out, p=p)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.368, 0.309, 0.549, 0.534, 0.65])
        self.assertEqual(list(np.around(sa[~np.isnan(sa[:,1]),1],3)),
                         [0.47, 0.816, 0.722, 0.418, -0.653, -0.941, 0.863, -1.265, -0.424, -0.786, 0.183])

    def test_r_1(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        N = 10
        p = 6
        r = 1
        out = np.random.random(r*(self.NumFact+1))

        mat, vec = morris_sampling(self.NumFact, self.LB, self.UB, N=N, p=p, r=r, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.NumFact, mat, vec, out, p=p)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.579, 0.009, 0.239, 0.864, 0.876])
        self.assertEqual(list(np.around(sa[0:5].squeeze(),3)), [-0.579, -0.009, -0.239, -0.864, 0.876])


    def test_groups(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        NumGroups = 5
        Groups = np.random.randint(0, 4, (self.NumFact,NumGroups))
        N = 100
        p = 6
        r = 10
        out = np.random.random(r*(self.NumFact+1))

        # Check 1
        mat, vec = morris_sampling(self.NumFact, self.LB, self.UB, N=N, p=p, r=r,
                                   GroupMat=Groups, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.2, 1.8, 3.8, 7., 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [3., 0., 1., 4., 2.])

        # Check 2
        sa, res = elementary_effects(self.NumFact, mat, vec, out, p=p, Group=Groups)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.531, 0.43, 0.432, 0.443, 0.443])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.279, 0.557, 0.557, 0.557, 0.557])


    def test_groups_nan(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        NumGroups = 5
        Groups = np.random.randint(0, 4, (self.NumFact,NumGroups))
        N = 100
        p = 6
        r = 10
        out = np.random.random(r*(self.NumFact+1))
        out[1:r*self.NumFact:self.NumFact//2] = np.nan

        mat, vec = morris_sampling(self.NumFact, self.LB, self.UB, N=N, p=p, r=r,
                                   GroupMat=Groups, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.NumFact, mat, vec, out, p=p, Group=Groups)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.49, 0.425, 0.427, 0.441, 0.441])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.279, 0.557, 0.557, 0.557, 0.557])


# --------------------------------------------------------------------
# sa_test_functions.py
# Missing coverage:
#    521: morris as wrapper to fmorris not working because morris is also the name of a module here
class TestSATestFunctions(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # seed for reproducible results
        seed = 1234
        np.random.seed(seed=seed)

    def test_sa_test_functions(self):
        import os
        import numpy as np
        from pyeee import B, g, G, Gstar, K, bratley, oakley_ohagan, ishigami_homma
        from pyeee import linear, product, ratio, ishigami_homma_easy, fmorris

        # scalar
        self.assertEqual(B(np.arange(10)), 80)
        self.assertEqual(g(np.ones(5), np.zeros(5)), 32.0)
        self.assertEqual(G(np.ones(5), np.zeros(5)), 32.0)
        self.assertEqual(Gstar(np.ones(5), np.zeros(5), np.ones(5), np.zeros(5)), 1.0)
        self.assertEqual(Gstar(np.ones(5), [0.,0.,0.,0.,0.], np.ones(5), np.zeros(5)), 1.0)
        self.assertEqual(K(np.arange(5)+1.), -101.0)
        self.assertEqual(bratley(np.arange(5)+1.), -101.0)
        self.assertEqual(oakley_ohagan(np.zeros(15)), 15.75)
        self.assertEqual(ishigami_homma([np.pi/2.,np.pi/2.,1.], 1., 1.), 3.0)
        self.assertEqual(linear(np.ones(1), 1., 1.), 2.0)
        self.assertEqual(product(np.arange(2)+1.), 2.0)
        self.assertEqual(ratio(np.arange(2)+1.), 0.5)
        self.assertEqual(ishigami_homma_easy([np.pi/2.,1.]), 2.0)

        # vector
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

        # Morris
        npars = 20
        x0 = np.ones(npars)*0.5
        lb = np.zeros(npars)
        ub = np.ones(npars)
        beta0              = 0.
        beta1              = np.random.standard_normal(npars)
        beta1[:10]         = 20.
        beta2              = np.random.standard_normal((npars,npars))
        beta2[:6,:6]       = -15.
        beta3              = np.zeros((npars,npars,npars))
        beta3[:5,:5,:5]    = -10.
        beta4              = np.zeros((npars,npars,npars,npars))
        beta4[:4,:4,:4,:4] = 5.
        mm = fmorris(np.linspace(0,2*(npars-1),npars)/float(2*npars-1),
                     beta0, beta1, beta2, beta3, beta4)
        self.assertEqual(np.around(mm,3), -82.711)
        mm = fmorris(np.arange(2*npars,dtype=np.float).reshape((npars,2))/float(2*npars-1),
                     beta0, beta1, beta2, beta3, beta4)
        self.assertEqual(list(np.around(mm,3)), [-82.711, -60.589])


# --------------------------------------------------------------------
# screening.py
# Missing coverage:
#    167-170: MPI
#    183: raise Error if mask is not None but x0 is None
#    231: fx.ndim = 1 - single trajectory?
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

    def test_ee_g_1(self):
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

        out = ee(obj, lb, ub, processes=4)

        # Check
        self.assertEqual(list(np.around(out[:,0],3)), [0.047, 0.233, 1.539, 0.747, 0.025, 0.077])


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
        from pyeee.sa_test_functions import morris as  fmorris

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


# --------------------------------------------------------------------
# std_io.py
#     591-592: IOError is covered below but not recognised in coverage report
class TestStd_io(unittest.TestCase):

    def test_std_io_sub_ja(self):
        import os
        import numpy as np
        from pyeee import sub_ja_params_files

        # standard_parameter_writer without pid
        filename1 = 'params1.txt'
        filename2 = 'params2.txt'
        pid       = 1234
        params    = np.arange(10, dtype=np.float)

        ff = open(filename1, 'w')
        print('param0 = #JA0000#', file=ff)
        print('param1 = #JA0001#', file=ff)
        print('param2 = #JA0002#', file=ff)
        print('param3 = #JA0003#', file=ff)
        print('param4 = #JA0004#', file=ff)
        ff.close()

        ff = open(filename2, 'w')
        print('param4 = #JA0004#', file=ff)
        print('param5 = #JA0005#', file=ff)
        print('param6 = #JA0006#', file=ff)
        print('param7 = #JA0007#', file=ff)
        ff.close()

        sub_ja_params_files([filename1, filename2], pid, params)

        f = open(filename1+'.'+str(pid), 'r')
        lines1 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines1 ],
                         ['param0 = 0.00000000000000e+00',
                          'param1 = 1.00000000000000e+00',
                          'param2 = 2.00000000000000e+00',
                          'param3 = 3.00000000000000e+00',
                          'param4 = 4.00000000000000e+00'])

        f = open(filename2+'.'+str(pid), 'r')
        lines2 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines2 ],
                         ['param4 = 4.00000000000000e+00',
                          'param5 = 5.00000000000000e+00',
                          'param6 = 6.00000000000000e+00',
                          'param7 = 7.00000000000000e+00'])

        if os.path.exists(filename1): os.remove(filename1)
        if os.path.exists(filename2): os.remove(filename2)
        if os.path.exists(filename1+'.'+str(pid)): os.remove(filename1+'.'+str(pid))
        if os.path.exists(filename2+'.'+str(pid)): os.remove(filename2+'.'+str(pid))


    def test_std_io_sub_names_params(self):
        import os
        import numpy as np
        from pyeee import sub_names_params_files_ignorecase, sub_names_params_files_case
        from pyeee import sub_names_params_files

        # ignore case
        filename1 = 'params11.txt'
        filename2 = 'params21.txt'
        pid       = 1234
        params    = np.arange(10, dtype=np.float)
        names     = ['param0', 'param1', 'param2', 'param3', 'param4',
                     'param5', 'param6', 'param7', 'param8', 'param9']

        ff = open(filename1, 'w')
        print('param0 = -6', file=ff)
        print('Param1 = -7', file=ff)
        print('param2 = -8', file=ff)
        print('Param3 = -9', file=ff)
        print('param4 = -10', file=ff)
        ff.close()

        ff = open(filename2, 'w')
        print('param4 = -10', file=ff)
        print('param5 = 3', file=ff)
        print('PARAM6 = 4', file=ff)
        print('param7 = 5', file=ff)
        ff.close()

        sub_names_params_files_ignorecase([filename1, filename2], pid, params, names)

        f = open(filename1+'.'+str(pid), 'r')
        lines1 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines1 ],
                         ['param0 = 0.00000000000000e+00',
                          'Param1 = 1.00000000000000e+00',
                          'param2 = 2.00000000000000e+00',
                          'Param3 = 3.00000000000000e+00',
                          'param4 = 4.00000000000000e+00'])

        f = open(filename2+'.'+str(pid), 'r')
        lines2 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines2 ],
                         ['param4 = 4.00000000000000e+00',
                          'param5 = 5.00000000000000e+00',
                          'PARAM6 = 6.00000000000000e+00',
                          'param7 = 7.00000000000000e+00'])

        if os.path.exists(filename1): os.remove(filename1)
        if os.path.exists(filename2): os.remove(filename2)
        if os.path.exists(filename1+'.'+str(pid)): os.remove(filename1+'.'+str(pid))
        if os.path.exists(filename2+'.'+str(pid)): os.remove(filename2+'.'+str(pid))

        # wrapper for ignore case
        filename1 = 'params12.txt'
        filename2 = 'params22.txt'
        pid       = 1234
        params    = np.arange(10, dtype=np.float)
        names     = ['param0', 'param1', 'param2', 'param3', 'param4',
                     'param5', 'param6', 'param7', 'param8', 'param9']

        ff = open(filename1, 'w')
        print('param0 = -6', file=ff)
        print('Param1 = -7', file=ff)
        print('param2 = -8', file=ff)
        print('Param3 = -9', file=ff)
        print('param4 = -10', file=ff)
        ff.close()

        ff = open(filename2, 'w')
        print('param4 = -10', file=ff)
        print('param5 = 3', file=ff)
        print('PARAM6 = 4', file=ff)
        print('param7 = 5', file=ff)
        ff.close()

        sub_names_params_files([filename1, filename2], pid, params, names)

        f = open(filename1+'.'+str(pid), 'r')
        lines1 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines1 ],
                         ['param0 = 0.00000000000000e+00',
                          'Param1 = 1.00000000000000e+00',
                          'param2 = 2.00000000000000e+00',
                          'Param3 = 3.00000000000000e+00',
                          'param4 = 4.00000000000000e+00'])

        f = open(filename2+'.'+str(pid), 'r')
        lines2 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines2 ],
                         ['param4 = 4.00000000000000e+00',
                          'param5 = 5.00000000000000e+00',
                          'PARAM6 = 6.00000000000000e+00',
                          'param7 = 7.00000000000000e+00'])

        if os.path.exists(filename1): os.remove(filename1)
        if os.path.exists(filename2): os.remove(filename2)
        if os.path.exists(filename1+'.'+str(pid)): os.remove(filename1+'.'+str(pid))
        if os.path.exists(filename2+'.'+str(pid)): os.remove(filename2+'.'+str(pid))

        # case sensitive
        filename1 = 'params13.txt'
        filename2 = 'params23.txt'
        pid       = 1234
        params    = np.arange(10, dtype=np.float)
        names     = ['param0', 'param1', 'param2', 'param3', 'param4',
                     'param5', 'param6', 'param7', 'param8', 'param9']

        ff = open(filename1, 'w')
        print('    param0= -6', file=ff)
        print('   Param1 = -7', file=ff)
        print('  param2  = -8', file=ff)
        print(' Param3   = -9', file=ff)
        print('param4    = -10', file=ff)
        ff.close()

        ff = open(filename2, 'w')
        print('param4    = -10', file=ff)
        print('param5   = 3', file=ff)
        print('PARAM6  = 4', file=ff)
        print('param7 = 5', file=ff)
        ff.close()

        sub_names_params_files_case([filename1, filename2], pid, params, names)

        f = open(filename1+'.'+str(pid), 'r')
        lines1 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines1 ],
                         ['    param0= 0.00000000000000e+00',
                          '   Param1 = -7',
                          '  param2  = 2.00000000000000e+00',
                          ' Param3   = -9',
                          'param4    = 4.00000000000000e+00'])

        f = open(filename2+'.'+str(pid), 'r')
        lines2 = f.readlines()
        f.close()

        self.assertEqual([ i.rstrip() for i in lines2 ],
                         ['param4    = 4.00000000000000e+00',
                          'param5   = 5.00000000000000e+00',
                          'PARAM6  = 4',
                          'param7 = 7.00000000000000e+00'])

        if os.path.exists(filename1): os.remove(filename1)
        if os.path.exists(filename2): os.remove(filename2)
        if os.path.exists(filename1+'.'+str(pid)): os.remove(filename1+'.'+str(pid))
        if os.path.exists(filename2+'.'+str(pid)): os.remove(filename2+'.'+str(pid))


    def test_std_io_standard(self):
        import os
        import numpy as np
        from pyeee import standard_parameter_writer, standard_parameter_reader
        from pyeee import standard_parameter_writer_bounds_mask, standard_parameter_reader_bounds_mask
        from pyeee import standard_objective_reader
        from pyeee import standard_timeseries_reader, standard_time_series_reader
        
        # standard_parameter_reader/writer without pid
        filename = 'params.txt'
        params   = np.arange(10, dtype=np.float)
        standard_parameter_writer(filename, params)
        
        iparams = standard_parameter_reader(filename)

        self.assertEqual(list(iparams), list(params))

        if os.path.exists(filename): os.remove(filename)

        # standard_parameter_writer with pid
        filename = 'params.txt'
        pid      = 1234
        params   = np.arange(10, dtype=np.float)
        standard_parameter_writer(filename, pid, params)
        
        iparams = standard_parameter_reader(filename+'.'+str(pid))

        self.assertEqual(list(iparams), list(params))

        if os.path.exists(filename+'.'+str(pid)): os.remove(filename+'.'+str(pid))

        # standard_parameter_reader/writer_bounds_mask
        filename = 'params.txt'
        pid      = 1234
        params   = np.arange(10, dtype=np.float)
        pmin     = params - 1.
        pmax     = params + 1.
        mask     = np.ones(10, dtype=np.bool)
        standard_parameter_writer_bounds_mask(filename, pid, params, pmin, pmax, mask)
        
        ids, iparams, ipmin, ipmax, imask = standard_parameter_reader_bounds_mask(filename+'.'+str(pid))

        self.assertEqual(list(ids),     list([ str(i) for i in np.arange(10)+1 ]))
        self.assertEqual(list(iparams), list(params))
        self.assertEqual(list(ipmin),   list(pmin))
        self.assertEqual(list(ipmax),   list(pmax))
        self.assertEqual(list(imask),   list(mask))

        if os.path.exists(filename+'.'+str(pid)): os.remove(filename+'.'+str(pid))

        # standard_parameter_reader_bounds_mask - Error
        filename = 'params.txt'
        pid      = 1234
        params   = np.arange(10, dtype=np.float)
        pmin     = params - 1.
        pmax     = params + 1.
        mask     = np.ones(10, dtype=np.bool)
        ff = open(filename, 'w')
        for i in range(10):
            dstr = '{:d} {:.14e} {:.14e} {:.14e}'.format(i+1, params[i], pmin[i], pmax[i])
            print(dstr, file=ff)
        ff.close()
        
        try:
            ids, iparams, ipmin, ipmax, imask = standard_parameter_reader_bounds_mask(filename+'.'+str(pid))
            self.assertTrue(False)
        except IOError:
            self.assertTrue(True)

        if os.path.exists(filename+'.'+str(pid)): os.remove(filename+'.'+str(pid))

        # standard_parameter_writer_bounds_mask - pid=None
        filename = 'params.txt'
        pid      = None
        params   = np.arange(10, dtype=np.float)
        pmin     = params - 1.
        pmax     = params + 1.
        mask     = np.ones(10, dtype=np.bool)
        standard_parameter_writer_bounds_mask(filename, pid, params, pmin, pmax, mask)
        
        ids, iparams, ipmin, ipmax, imask = standard_parameter_reader_bounds_mask(filename)

        self.assertEqual(list(ids),     list([ str(i) for i in np.arange(10)+1 ]))
        self.assertEqual(list(iparams), list(params))
        self.assertEqual(list(ipmin),   list(pmin))
        self.assertEqual(list(ipmax),   list(pmax))
        self.assertEqual(list(imask),   list(mask))

        if os.path.exists(filename): os.remove(filename)

        # standard_objective_reader
        filename = 'obj.txt'

        ff = open(filename, 'w')
        print('{:.14e}'.format(1234.), file=ff)
        ff.close()

        obj = standard_objective_reader(filename)
        self.assertEqual(obj, 1234.)
        
        if os.path.exists(filename): os.remove(filename)

        # standard_time_series_reader
        filename = 'ts.txt'
        params   = np.arange(10, dtype=np.float)

        ff = open(filename, 'w')
        for i in params:
            print('{:.14e}'.format(i), file=ff)
        ff.close()

        ts = standard_time_series_reader(filename)
        self.assertEqual(list(ts), list(params))
        
        if os.path.exists(filename): os.remove(filename)

        # standard_timeseries_reader
        filename = 'ts.txt'
        params   = np.arange(10, dtype=np.float)

        ff = open(filename, 'w')
        for i in params:
            print('{:.14e}'.format(i), file=ff)
        ff.close()

        ts = standard_timeseries_reader(filename)
        self.assertEqual(list(ts), list(params))
        
        if os.path.exists(filename): os.remove(filename)


# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# utils.py
# covered


# --------------------------------------------------------------------
# version.py
# covered


if __name__ == "__main__":
    unittest.main()
