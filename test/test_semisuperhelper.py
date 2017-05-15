# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 01:10:20 2017

@author: jeffrey.gomberg
"""

import unittest

import numpy as np

from creon.semisuperhelper import SemiSupervisedHelper


class TestSemiSuperHelper(unittest.TestCase):
    """
    Some basic testing, just the complicated function pn_assume for now
    """

    def setUp(self):
        self.y = np.asarray([1, 1, 1, 0, 0, 0, -1, -1, -1, -1])
        self.X = np.arange(20).reshape([10,2])
        self.ssh = SemiSupervisedHelper(self.y, random_state=55)

    def test_pn_assume_full(self):
        X, y, X_unused = self.ssh.pn_assume(self.X)
        self.assertEqual(X_unused.size, 0)
        self.assertEqual(list(y), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(X, self.X))

    def test_pn_assume_none(self):
        X, y, X_unused = self.ssh.pn_assume(self.X, unlabeled_pct=0.0)
        self.assertTrue(np.array_equal(y, self.y[:6]))
        self.assertTrue(np.array_equal(X, self.X[:6, :]))
        self.assertTrue(np.array_equal(X_unused, self.X[6:, :]))

    def test_pn_bad_unlabeled_pct(self):
        with self.assertRaises(ValueError):
            X, y, X_unused = self.ssh.pn_assume(self.X, unlabeled_pct=-0.1)

    def test_pn_assume_normal(self):
        X, y, X_unused = self.ssh.pn_assume(self.X, unlabeled_pct=0.5)
        self.assertEqual(list(y), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertTrue(len(X_unused)==2)
        res_not_in_X = np.setdiff1d(self.X.ravel(), X.ravel())
        self.assertEqual(len(res_not_in_X[res_not_in_X >= 12]), 4)


if __name__ == '__main__':
    unittest.main()