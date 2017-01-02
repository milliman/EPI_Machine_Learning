# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 00:03:11 2017

@author: jeffrey.gomberg

"""
import pandas as pd

class SemiSupervisedHelper:
    """This class will provide helper functions for dealing with semi-supervised learning problems
        Unless otherwise stated, the convention for y is:
            -1 -> Unlabled
            0 -> Negative
            1 -> Positive
    """

    def __init__(self, y: pd.Series):
        self.y = y
        self.pn_mask = (y == 0) | (y == 1)
        self.pu_mask = (y == -1) | (y == 1)
        self.nu_mask = (y == -1) | (y == 0)
        self.u_mask = y == -1

    def pn(self, X):
        """
        Return X_pn, y_pn
        """
        return X[self.pn_mask.values], self.y[self.pn_mask]

    def pu(self, X):
        """
        Return X_pu, y_pu
        """
        return X[self.pu_mask.values], self.y[self.pu_mask]

    def nu(self, X):
        """
        Return X_nu, y_nu
        """
        return X[self.nu_mask.values], self.y[self.nu_mask]

    def u(self, X):
        """
        Return all unlabeled X, y_u
        """
        return X[self.u_mask.values], self.y[self.u_mask]

    def pn_assume(self, unlabeled_to_class=0):
        """
        Return y with the unalbeled assumed to be unlabled_class
        """
        return self.y.replace(to_replace=-1, value=unlabeled_to_class)

