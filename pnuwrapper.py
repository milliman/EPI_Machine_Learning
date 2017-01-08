# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:40:29 2017

@author: jeffrey.gomberg
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state

from semisuperhelper import SemiSupervisedHelper


class PNUWrapper(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    This will wrap classifiers to be used with unlabeled data

    WARNING: This class is very brittle and should not be used outside this project without more boilerplate
    """

    def __init__(self, base_estimator=None, num_unlabeled=0.0, threshold_set_pct=None, random_state=None):
        """
        If num_unlabeled is float, then use that % of unlabeled in training
        If num_unlabeled is an int, then use that number of unlabeled data to train
        threshold_set_pct is how to set the threshold of the classifier in predict_proba by taking that pct of the
            unused unlabeled data and finding where that pct is predicted. For example, if 10 unused examples are
            predicted and sorted in descending order in terms of predict_proba, then threshold_set_pct-0.5 means we
            look at example #5 and see that predict_proba=0.4, then in the future anything >= 0.4 is predicted positive.
            If it is left None, then calibration on the unseen data will not be used
        All unlabeled data is assumed to be of class "0"
        """
        self.base_estimator = base_estimator
        self.num_unlabeled = num_unlabeled
        self.random_state = random_state
        self.threshold_set_pct = threshold_set_pct
        self.threshold = None

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Check y only has -1, 0, 1
        if np.setdiff1d(y, np.asarray([-1, 0, 1])):
            raise ValueError("y must contain only -1 (unlabeled), 0 (negative), and 1 (positive) labels.")
        # Check to see base_estimator exists
        if self.base_estimator is None:
            raise ValueError("base_estimator must be defined before running fit")
        if self.num_unlabeled < 0:
            raise ValueError("num_unlabeled must be > 0")
        # Store the classes seen during fit
        self.classes_ = np.asarray([0, 1])

        self.X_ = X
        self.y_ = y

        ssh = SemiSupervisedHelper(y, random_state=random_state)
        X_temp, y_temp, X_unlabeled_unused = ssh.pn_assume(X, unlabeled_pct=self.num_unlabeled)
        self.base_estimator.fit(X_temp, y_temp)

        #TODO - error checking on the block below - good threshold_set_pct, enough unlableds, etc.
        if self.threshold_set_pct is not None and len(X_unlabeled_unused > 0):
            unlabeled_pr = self.base_estimator.predict_proba(X_unlabeled_unused)[:, 1]
            unlabeled_pr[::-1].sort()
            u_N = len(unlabeled_pr)
            idx = min(max(int(self.threshold_set_pct * u_N) - 1, 0), u_N - 1)
            self.threshold = unlabeled_pr[idx]
            print("u_N:{} idx={} threshold={} min={} max={}".format(u_N, idx, self.threshold, unlabeled_pr[-1], unlabeled_pr[0]))

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        if self.threshold is None:
            return self.base_estimator.predict(X)
        else:
            pr = self.base_estimator.predict_proba(X)[:, 1]
            return np.asarray(pr >= self.threshold, dtype=np.int)



