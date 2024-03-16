#!/usr/bin/env python
"""
This is the unittest for Screening module.

python -m unittest -v tests/test_screening.py
python -m pytest --cov=. --cov-report term-missing -v tests/test_screening.py

"""
import unittest


# extended scipy/_util.py version
class _FunctionWrapper:
    """
    Wrap user function with arguments and keywords, allowing picklability

    Parameters
    ----------
    func : callable
        Function in the form ``func(x, *args, **kwargs)``, where ``x`` are
        the parameters in the form of an iterable.
        ``args`` and ``kwargs`` are passed to the function via the usual
        unpacking operators.

    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.nargs = len(args)
        self.nkwargs = len(kwargs)
        if self.nkwargs == 0:
            if ( (len(self.args) == 2) and
                 isinstance(self.args[-1], dict) and
                 (len(self.args[-1]) == 0) ):
                # if kwargs={} then **kwargs={} and hence counted as args
                self.args = self.args[0]
                self.nargs = len(args)

    def __call__(self, x):
        if (self.nargs > 0) and (self.nkwargs > 0):
            return self.func(x, *self.args, **self.kwargs)
        elif (self.nargs > 0) and (self.nkwargs == 0):
            return self.func(x, *self.args)
        elif (self.nargs == 0) and (self.nkwargs > 0):
            return self.func(x, **self.kwargs)
        else:
            return self.func(x)


class TestScreening(unittest.TestCase):
    """
    Tests for screening.py
    Missing coverage:
        ca. 230: using MPI
        ca. 265: no seed given
        ca. 310: only one trajectory
    """
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
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(obj, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.045, 0.24, 1.624, 0.853, 0.031, 0.084])

    # G function
    def test_ee_g_verbose(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(obj, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1, verbose=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.045, 0.24, 1.624, 0.853, 0.031, 0.084])

    # G function, mask
    def test_ee_g_mask(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb   = np.zeros(npars)
        ub   = np.ones(npars)
        x0   = np.ones(npars)*0.5
        mask = np.ones(npars, dtype=bool)
        mask[1] = False

        out = ee(obj, lb, ub, self.nt, x0=x0, mask=mask,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.038, 0.0, 1.582, 1.136, 0.04, 0.102])

    # G function, mask, error
    def test_ee_g_mask_error(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb   = np.zeros(npars)
        ub   = np.ones(npars)
        x0   = np.ones(npars)*0.5
        mask = np.ones(npars, dtype=bool)
        mask[1] = False

        self.assertRaises(TypeError, ee, obj, lb, ub, self.nt, x0=None,
                          mask=mask, ntotal=self.ntotal, nsteps=self.nsteps,
                          processes=1)

    # G function, nt=1
    def test_ee_g_nt1(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(obj, lb, ub, 1, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.0, 0.134, 1.813, 0.0, 0.017, 0.0])

    # G function, pool
    def test_ee_g_pool(self):
        import numpy as np
        import schwimmbad
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        nprocs = 4
        ipool = schwimmbad.choose_pool(mpi=False, processes=nprocs)
        out = ee(obj, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=nprocs, pool=ipool)
        ipool.close()

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.045, 0.24, 1.624, 0.853, 0.031, 0.084])

    # G function, multiprocesses
    def test_ee_g_1(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import G

        # Function and parameters
        func   = G
        npars  = 6
        params = [78., 12., 0.5, 2., 97., 33.]  # G

        # Partialise function with fixed parameters
        arg   = [params]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(obj, lb, ub, npars, processes=4)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.047, 0.233, 1.539, 0.747, 0.025, 0.077])

    # Gstar function with different interactions
    def test_screening_gstar(self):
        import numpy as np
        from pyeee import screening
        from .sa_test_functions import Gstar

        # Function and parameters
        func   = Gstar
        npars  = 10
        params = [[np.ones(npars),     np.random.random(npars),
                   [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],  # G*
                  [np.ones(npars),     np.random.random(npars),
                   [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
                  [np.ones(npars)*0.5, np.random.random(npars),
                   [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
                  [np.ones(npars)*0.5, np.random.random(npars),
                   [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]],
                  [np.ones(npars)*2.0, np.random.random(npars),
                   [0., 0.,  9.,  9.,  9.,  9.,  9., 9., 9., 9.]],
                  [np.ones(npars)*2.0, np.random.random(npars),
                   [0., 0.1, 0.2, 0.3, 0.4, 0.8, 1., 2., 3., 4.]]]
        iiout  = [[1.087, 1.807, 0.201, 0.13,  0.08,
                   0.077, 0.055, 0.198, 0.139, 0.136],
                  [0.924, 1.406, 0.603, 0.876, 1.194,
                   0.567, 0.642, 0.276, 0.131, 0.34 ],
                  [1.021, 0.875, 0.085, 0.096, 0.116,
                   0.104, 0.11,  0.07,  0.045, 0.112],
                  [1.096, 0.752, 0.573, 0.762, 0.189,
                   0.584, 0.259, 0.437, 0.468, 0.169],
                  [4.806, 3.299, 0.614, 0.406, 0.89,
                   0.628, 0.372, 0.428, 0.64,  0.565],
                  [1.212, 1.076, 3.738, 8.299, 0.636,
                   0.469, 0.37,  4.856, 0.553, 0.009]]
        lb = np.zeros(npars)
        ub = np.ones(npars)

        for ii in range(len(params)):
            # Partialise function with fixed parameters
            arg   = params[ii]
            kwarg = {}
            obj   = _FunctionWrapper(func, arg, kwarg)

            out = screening(obj, lb, ub, self.nt, x0=None, mask=None,
                            ntotal=self.ntotal, nsteps=self.nsteps,
                            processes=1)  # , plotfile='gstar'+str(ii)+'.png')
            # Check
            self.assertEqual(list(np.around(out[:, 0], 3)), iiout[ii])

    # Bratley / K function
    def test_ee_k(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import bratley

        # Function and parameters
        func   = bratley
        npars  = 10
        params = []  # k

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(func, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.586, 0.219, 0.082, 0.055, 0.02,
                          0.068, 0.009, 0.007, 0., 0.])

    # Morris function
    def test_ee_fmorris(self):
        import numpy as np
        from pyeee import ee
        from .sa_test_functions import fmorris

        # Function and parameters
        func = fmorris
        npars = 20
        beta0                 = 0.
        beta1                 = np.random.standard_normal(npars)
        beta1[:10]            = 20.
        beta2                 = np.random.standard_normal((npars, npars))
        beta2[:6, :6]         = -15.
        beta3                 = np.zeros((npars, npars, npars))
        beta3[:5, :5, :5]     = -10.
        beta4                 = np.zeros((npars, npars, npars, npars))
        beta4[:4, :4, :4, :4] = 5.

        # Partialise Morris function with fixed parameters beta0-4
        arg   = [beta0, beta1, beta2, beta3, beta4]
        kwarg = {}
        obj   = _FunctionWrapper(func, arg, kwarg)

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)

        # Check
        out = ee(obj, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 processes=4)

        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [66.261, 48.363, 20.133, 62.42, 37.87,
                          26.031, 30.211, 39.714, 42.633, 43.776,
                          4.996, 3.701, 4.734, 8.031, 5.734,
                          3.564, 5.068, 7.635, 3.129, 5.224])

    # Bratley / K function with distribution
    def test_ee_k_dist(self):
        import numpy as np
        import scipy.stats as stats
        from pyeee import ee
        from .sa_test_functions import bratley

        # Function and parameters
        func   = bratley
        npars  = 10
        params = []  # k

        # Screening
        lb = np.zeros(npars)
        ub = np.ones(npars)
        dist      = [ stats.uniform for i in range(npars) ]
        distparam = [ (lb[i], ub[i]-lb[i]) for i in range(npars) ]
        lb = np.zeros(npars)
        ub = np.ones(npars)

        out = ee(func, lb, ub, self.nt, x0=None, mask=None,
                 ntotal=self.ntotal, nsteps=self.nsteps,
                 dist=dist, distparam=distparam,
                 processes=1)

        # Check
        self.assertEqual(list(np.around(out[:, 0], 3)),
                         [0.586, 0.219, 0.082, 0.055, 0.02,
                          0.068, 0.009, 0.007, 0., 0.])


if __name__ == "__main__":
    unittest.main()
