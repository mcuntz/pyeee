#!/usr/bin/env python
"""
This is the unittest for the General Functions module.

python -m unittest -v tests/test_general_functions.py
python -m pytest --cov=. --cov-report term-missing -v tests/test_general_functions.py

"""
import unittest

def cost2_logistic(p, x, y):
    """
    Sum of squared deviations of obs and logistic function
    :math:`L/(1+exp(-k(x-x0)))`

    Parameters
    ----------
    p : iterable of floats
        parameters (`len(p)=3`)
          - `p[0]` = L  = Maximum of logistic function
          - `p[1]` = k  = Steepness of logistic function
          - `p[2]` = x0 = Inflection point of logistic function
    x : float or array_like of floats
        independent variable
    y : float or array_like of floats
        dependent variable, observations

    Returns
    -------
    float
        sum of squared deviations
    """
    import numpy as np
    from pyeee import logistic_p
    return np.sum((y - logistic_p(x, p))**2)


class TestGeneralFunctions(unittest.TestCase):
    """
    Tests for functions/general_functions.py

    """

    def test_cost_square(self):
        import numpy as np
        from pyeee import logistic_p
        from pyeee import cost_square

        p = [1., 1., 0.]
        x = np.arange(2)
        y = np.zeros(2)
        assert cost2_logistic(p, x, y) == cost_square(p, logistic_p, x, y)

    def test_curvature(self):
        import numpy as np
        from pyeee import curvature
        from pyeee import dlogistic_offset, d2logistic_offset
        from pyeee import dlogistic_offset_p, d2logistic_offset_p

        self.assertEqual(np.around(curvature(1., dlogistic_offset,
                                             d2logistic_offset,
                                             1., 2., 2., 1.), 4), 0.2998)
        self.assertEqual(np.around(curvature(1., dlogistic_offset_p,
                                             d2logistic_offset_p,
                                             [1., 2., 2., 1.]), 4), 0.2998)
        self.assertEqual(list(np.around(
            curvature(np.array([1., 1.]), dlogistic_offset, d2logistic_offset,
                      1., 2., 2., 1.), 4)), [0.2998, 0.2998])


if __name__ == "__main__":
    unittest.main()
