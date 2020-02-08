#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for the Function Wrapper module.

    python -m unittest -v test_function_wrapper.py
    python -m pytest --cov pyeee --cov-report term-missing -v tests/
"""
import unittest


# --------------------------------------------------------------------
# function_wrapper.py
#     336, 340: pid, func is str, and debug in exe_mask_wrapper is covered below but not recognised in coverage report
#     396: function_wrapper is covered below but not recognised in coverage report
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

        self.assertRaises(TypeError, ee, func, lb, ub,
                          nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                          processes=1)


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

        self.assertRaises(TypeError, ee, func, lb[mask], ub[mask],
                          nt=self.nt, ntotal=self.ntotal, nsteps=self.nsteps,
                          processes=1)


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


if __name__ == "__main__":
    unittest.main()
