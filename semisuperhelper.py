# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 00:03:11 2017

@author: jeffrey.gomberg

"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

class SemiSupervisedHelper:
    """This class will provide helper functions for dealing with semi-supervised learning problems
        Unless otherwise stated, the convention for y is:
            -1 -> Unlabled
            0 -> Negative
            1 -> Positive
    """

    def __init__(self, y, random_state=None):
        self.random_state = random_state
        self.y = y
        self.pn_mask = (y == 0) | (y == 1)
        self.pu_mask = (y == -1) | (y == 1)
        self.nu_mask = (y == -1) | (y == 0)
        self.u_mask = y == -1

    def pn(self, X):
        """
        Return X_pn, y_pn
        """
        return X[self.pn_mask], self.y[self.pn_mask]

    def pu(self, X):
        """
        Return X_pu, y_pu
        """
        return X[self.pu_mask], self.y[self.pu_mask]

    def nu(self, X):
        """
        Return X_nu, y_nu
        """
        return X[self.nu_mask], self.y[self.nu_mask]

    def u(self, X):
        """
        Return all unlabeled X_u, y_u
        """
        return X[self.u_mask], self.y[self.u_mask]

    def pn_assume(self, X, unlabeled_to_class=0, unlabeled_pct=1.0):
        """
        Return X, y with the unlabeled assumed to be unlabled_class and a specified number of unlabeleds
        If unlabaled_pct is float, take that % of unlabeleds, if int, then take that # unlabeleds
        """
        assert unlabeled_pct >= 0.0, "SemiSupervisedHelper.pn_assume unlabeled_pct >= 0"
        random_state = check_random_state(self.random_state)

        X_pn, y_pn = self.pn(X)
        if unlabeled_pct == 0.0:
            return X_pn, y_pn
        X_u, y_u = self.u(X)
        num_u = min(unlabeled_pct if isinstance(unlabeled_pct, int) else int(unlabeled_pct * len(y_u)), len(y_u))
        rand_idx = sample_without_replacement(n_population = X_u.shape[0], n_samples=num_u, random_state=random_state)
        X_u = X_u[rand_idx, :]
        y_u = np.full(num_u, unlabeled_to_class, dtype='int64')

        X_ret = np.vstack((X_pn, X_u))
        y_ret = np.concatenate((y_pn, y_u))
        return X_ret, y_ret

