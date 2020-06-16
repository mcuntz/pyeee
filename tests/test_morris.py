#!/usr/bin/env python
"""
    This is the unittest for the Morris module.

    python -m unittest -v test_morris.py
    python -m pytest --cov-report term-missing -v tests/test_morris.py
"""
from __future__ import division, absolute_import, print_function
import unittest

# --------------------------------------------------------------------
# morris.py
# Missing coverage:
#    477-478, 507-508: ImportError of matplotlib not installed
#    690: if NumGroups == 0: if SAm.size > 1: ?
class TestMorris(unittest.TestCase):

    def setUp(self):
        import numpy as np
        # seed for reproducible results
        self.seed = 1234
        np.random.seed(seed=self.seed)
        self.nparam     = 15
        self.LB         = np.arange(self.nparam)
        self.UB         = 2. * self.LB + 1.
        self.Diagnostic = 0


    def test_r_10(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        # Check 1
        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.6, 2.2, 2., 4.6, 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [12., 11., 5., 1., 9.])

        # Check 2
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.485, 0.445, 0.438, 0.58, 0.645])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.47, 0.502, 0.816, 0.722, 0.418])


    def test_r_10_diag(self):
        import os
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        # Check 1
        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=1)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.6, 2.2, 2., 4.6, 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [12., 11., 5., 1., 9.])
        self.assertTrue(os.path.exists('morris_diag_new_strategy.png'))
        self.assertTrue(os.path.exists('morris_diag_old_strategy.png'))
        if os.path.exists('morris_diag_new_strategy.png'): os.remove('morris_diag_new_strategy.png')
        if os.path.exists('morris_diag_old_strategy.png'): os.remove('morris_diag_old_strategy.png')

        # Check 2
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps, Diagnostic=1)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.485, 0.445, 0.438, 0.58, 0.645])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.47, 0.502, 0.816, 0.722, 0.418])
        self.assertFalse(os.path.exists('morris_diag_new_strategy.png'))
        self.assertFalse(os.path.exists('morris_diag_old_strategy.png'))


    def test_r_10_nan(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))
        out[1:nt*self.nparam:self.nparam//2] = np.nan

        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.368, 0.309, 0.549, 0.534, 0.65])
        self.assertEqual(list(np.around(sa[~np.isnan(sa[:,1]),1],3)),
                         [0.47, 0.816, 0.722, 0.418, -0.653, -0.941, 0.863, -1.265, -0.424, -0.786, 0.183])


    def test_r_1(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        nt     = 1
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.579, 0.009, 0.239, 0.864, 0.876])
        self.assertEqual(list(np.around(sa[0:5].squeeze(),3)), [-0.579, -0.009, -0.239, -0.864, 0.876])


    def test_groups(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        ngroup = 5
        Groups = np.random.randint(0, 4, (self.nparam,ngroup))
        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        # Check 1
        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal,
                                   GroupMat=Groups, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.2, 1.8, 3.8, 7., 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [3., 0., 1., 4., 2.])

        # Check 2
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps, Group=Groups)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.531, 0.43, 0.432, 0.443, 0.443])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.279, 0.557, 0.557, 0.557, 0.557])


    def test_groups_nan(self):
        import numpy as np
        from pyeee import morris_sampling, elementary_effects

        ngroup = 5
        Groups = np.random.randint(0, 4, (self.nparam,ngroup))
        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))
        out[1:nt*self.nparam:self.nparam//2] = np.nan

        mat, vec = morris_sampling(self.nparam, self.LB, self.UB, nt, nsteps=nsteps, ntotal=ntotal,
                                   GroupMat=Groups, Diagnostic=self.Diagnostic)
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps, Group=Groups)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.49, 0.425, 0.427, 0.441, 0.441])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.279, 0.557, 0.557, 0.557, 0.557])


    def test_r_10_dist(self):
        import numpy as np
        import scipy.stats as stats
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        dist      = [ stats.uniform for i in range(self.nparam) ]
        distparam = [ (self.LB[i],self.UB[i]-self.LB[i]) for i in range(self.nparam) ]
        LB = np.zeros(self.nparam)
        UB = np.ones(self.nparam)

        # Check 1
        mat, vec = morris_sampling(self.nparam, LB, UB, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.6, 2.2, 2., 4.6, 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [12., 11., 5., 1., 9.])

        # Check 2
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.485, 0.445, 0.438, 0.58, 0.645])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.47, 0.502, 0.816, 0.722, 0.418])


    def test_r_10_dist_none(self):
        import numpy as np
        import scipy.stats as stats
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        dist      = [ stats.uniform if i%2 == 0 else None for i in range(self.nparam) ]
        distparam = [ (self.LB[i],self.UB[i]-self.LB[i]) for i in range(self.nparam) ]
        LB = self.LB
        UB = self.UB
        LB[::2] = 0.
        UB[::2] = 1.

        # Check 1
        mat, vec = morris_sampling(self.nparam, LB, UB, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam, Diagnostic=self.Diagnostic)
        self.assertEqual(list(np.around(mat[0,0:5],3)), [0.6, 2.2, 2., 4.6, 8.])
        self.assertEqual(list(np.around(vec[0:5],3)), [12., 11., 5., 1., 9.])

        # Check 2
        sa, res = elementary_effects(self.nparam, mat, vec, out, nsteps=nsteps)
        self.assertEqual(list(np.around(res[0:5,0],3)), [0.485, 0.445, 0.438, 0.58, 0.645])
        self.assertEqual(list(np.around(sa[0:5,1],3)), [0.47, 0.502, 0.816, 0.722, 0.418])

        LB = np.zeros(self.nparam)
        UB = np.ones(self.nparam)
        dist       = [ stats.uniform for i in range(self.nparam) ]
        distparam1 = [ (LB[i],UB[i]-LB[i]) for i in range(self.nparam) ]
        distparam2 = [ (LB[i],UB[i]-LB[i]) if i%2 == 0 else None for i in range(self.nparam) ]

        # Check 3
        np.random.seed(seed=self.seed)
        mat1, vec1 = morris_sampling(self.nparam, LB, UB, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam1, Diagnostic=self.Diagnostic)
        np.random.seed(seed=self.seed)
        mat2, vec2 = morris_sampling(self.nparam, LB, UB, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=distparam2, Diagnostic=self.Diagnostic)
        self.assertTrue(np.all((mat1-mat2)==0.))
        self.assertTrue(np.all((vec1-vec2)==0.))

        # Check 3
        np.random.seed(seed=self.seed)
        mat3, vec3 = morris_sampling(self.nparam, LB, UB, nt, nsteps=nsteps, ntotal=ntotal, dist=dist, distparam=None, Diagnostic=self.Diagnostic)
        self.assertTrue(np.all((mat1-mat3)==0.))
        self.assertTrue(np.all((vec1-vec3)==0.))


    def test_r_10_dist_raise(self):
        import numpy as np
        import scipy.stats as stats
        from pyeee import morris_sampling, elementary_effects

        nt     = 10
        nsteps = 6
        ntotal = 100
        out = np.random.random(nt*(self.nparam+1))

        dist      = [ stats.uniform for i in range(self.nparam) ]
        distparam = [ (self.LB[i],self.UB[i]-self.LB[i]) for i in range(self.nparam) ]
        LB = np.zeros(self.nparam)
        UB = np.ones(self.nparam)

        # Check 1
        self.assertRaises(AssertionError, morris_sampling, self.nparam, LB[1:], UB, nt,
                          nsteps=nsteps, ntotal=ntotal,
                          Diagnostic=self.Diagnostic)

        # Check 2
        self.assertRaises(AssertionError, morris_sampling, self.nparam, LB, UB[1:], nt,
                          nsteps=nsteps, ntotal=ntotal,
                          Diagnostic=self.Diagnostic)

        # Check 3
        self.assertRaises(AssertionError, morris_sampling, self.nparam, LB, UB, nt,
                          nsteps=nsteps, ntotal=ntotal,
                          dist=dist[1:], distparam=distparam,
                          Diagnostic=self.Diagnostic)

        # Check 4
        dist[0] = 'stats.uniform'
        self.assertRaises(TypeError, morris_sampling, self.nparam, LB, UB, nt,
                          nsteps=nsteps, ntotal=ntotal,
                          dist=dist, distparam=distparam,
                          Diagnostic=self.Diagnostic)


if __name__ == "__main__":
    unittest.main()
