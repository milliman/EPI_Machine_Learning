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

    WARNING: This class is very brittle and should not be used outside this project without more boilerplate, tests
    and error checking
    """

    def __init__(self, base_estimator=None, num_unlabeled=0.0, threshold_set_pct=None, pu_learning=False,
                 random_state=None):
        """
        If num_unlabeled is float, then use that % of unlabeled in training
        If num_unlabeled is an int, then use that number of unlabeled data to train
        threshold_set_pct is how to set the threshold of the classifier in predict_proba by taking that pct of the
            unused unlabeled data and finding where that pct is predicted. For example, if 10 unused examples are
            predicted and sorted in descending order in terms of predict_proba, then threshold_set_pct-0.5 means we
            look at example #5 and see that predict_proba=0.4, then in the future anything >= 0.4 is predicted positive.
            If it is left None, then calibration on the unseen data will not be used
        All unlabeled data is assumed to be of class "0"
        if pu_learning == True, then throw away negatives in the set and train only on P and U class (default False)
        """
        self.base_estimator = base_estimator
        self.num_unlabeled = num_unlabeled
        self.random_state = random_state
        self.threshold_set_pct = threshold_set_pct
        self.pu_learning = pu_learning

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        # Check y only has -1, 0, 1
        if np.setdiff1d(y, np.asarray([-1, 0, 1])):
            raise ValueError("y must contain only -1 (unlabeled), 0 (negative), and 1 (positive) labels.")
        # Check to see base_estimator exists
        if self.base_estimator is None:
            raise ValueError("base_estimator must be defined before running fit")
        if self.num_unlabeled < 0:
            raise ValueError("num_unlabeled must be >= 0")

        ssh = SemiSupervisedHelper(y, random_state=random_state)
        if self.pu_learning:
            X, y = ssh.pu(X)
            ssh = SemiSupervisedHelper(y, random_state=random_state)

        # Store the classes seen during fit
        self.classes_ = np.asarray([0, 1])
        self.n_features_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        X_temp, y_temp, X_unlabeled_unused = ssh.pn_assume(X, unlabeled_pct=self.num_unlabeled)
        self.base_estimator.fit(X_temp, y_temp)

        if hasattr(self.base_estimator, 'decision_function'):
            self.threshold_fn_ = self.base_estimator.decision_function
        elif hasattr(self.base_estimator, 'predict_proba'):
            self.threshold_fn_ = self.base_estimator.predict_proba
        else:
            self.threshold_fn_ = None
        if self.threshold_fn_ is not None and self.threshold_set_pct is not None and len(X_unlabeled_unused) > 0:
            unlabeled_threshold = self.threshold_fn_(X_unlabeled_unused)
            if len(unlabeled_threshold.shape) > 1:
                unlabeled_threshold = unlabeled_threshold[:, -1]
            unlabeled_threshold[::-1].sort()
            u_N = len(unlabeled_threshold)
            idx = min(max(int(self.threshold_set_pct * u_N) - 1, 0), u_N - 1)
            self.threshold_ = unlabeled_threshold[idx]
        else:
            self.threshold_ = None

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['classes_'])

        # Input validation
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if self.threshold_ is None:
            return self.base_estimator.predict(X)
        else:
            pr = self.base_estimator.predict_proba(X)[:, 1]
            return np.asarray(pr >= self.threshold_, dtype=np.int)

    def predict_proba(self, X):
        check_is_fitted(self, 'classes_')

        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if hasattr(self.base_estimator, "predict_proba"):
            #TODO - need to modify this for the threshold if it exists, calibrate where threshold is
            # and set that to probabilty = 0.5
            proba = self.base_estimator.predict_proba(X)
        else:
            raise AttributeError("predict_prob doesn't exist for: {}".format(self.base_estimator))

        return proba

    @property
    def feature_importances_(self):
        check_is_fitted(self, ['classes_'])
        if hasattr(self.base_estimator, "feature_importances_"):
            return self.base_estimator.feature_importances_
        else:
            raise AttributeError("feature_importances_ doesn't exist for: {}".format(self.base_estimator))

    @property
    def coef_(self):
        check_is_fitted(self, ['classes_'])
        if hasattr(self.base_estimator, "coef_"):
            return self.base_estimator.coef_
        else:
            raise AttributeError("coef_ doesn't exist for: {}".format(self.base_estimator))