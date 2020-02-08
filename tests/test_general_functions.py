#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for the General Functions module.

    python -m unittest -v test_general_functions.py
    python -m pytest --cov pyeee --cov-report term-missing -v tests/
"""
import unittest


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


if __name__ == "__main__":
    unittest.main()
