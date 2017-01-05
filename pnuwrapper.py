# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:40:29 2017

@author: jeffrey.gomberg
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from semisuperhelper import SemiSupervisedHelper


class PNUWrapper(BaseEstimator, ClassifierMixin):
    """
    This will wrap classifiers to be used with unlabeled data

    WARNING: This class is very brittle and should not be used outside this project without more boilerplate
    """

    #TODO - add in probability fitting of thresholds in the training
    def __init__(self, base_estimator=None, num_unlabeled=0.0):
        """
        If num_unlabeled is float, then use that % of unlabeled in training
        If num_unlabeled is an int, then use that number of unlabeled data to train
        All unlabeled data is assumed to be of class "0"
        """
        self.base_estimator = base_estimator
        self.num_unlabeled = num_unlabeled

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Check y only has -1, 0, 1
        if not np.array_equal(np.unique(y), np.asarray([-1, 0, 1])):
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

        ssh = SemiSupervisedHelper(y)
        X_temp, y_temp = ssh.pn_assume(X, unlabeled_pct=self.num_unlabeled)
        self.base_estimator.fit(X_temp, y_temp)

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.base_estimator.predict(X)

