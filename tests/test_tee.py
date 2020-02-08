#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
"""
    This is the unittest for tee module.

    python -m unittest -v test_tee.py
    python -m pytest --cov pyeee --cov-report term-missing -v tests/
"""
import unittest


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
