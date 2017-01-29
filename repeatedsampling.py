#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:09:12 2017

@author: jgomberg
"""


from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state

import numbers
import itertools
from warnings import warn
import numpy as np
import math

from sklearn.externals.joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.fixes import bincount
from sklearn.utils import indices_to_mask
from sklearn.utils import check_random_state
from sklearn.ensemble.base import _partition_estimators
from sklearn.utils.validation import check_X_y
from sklearn.utils.random import sample_without_replacement, choice

__all__ = ["RepeatedRandomSubSampler"]

def _generate_class_indices(y):
    """ Generate all the indices of each class in ascending order"""
    return [np.where(y==c)[0] for c in np.unique(y)]

def _generate_repeated_sample_indices(random_state, sample_imbalance, y):
    """Draw randomly repeated samples, return an array of arrays of indeces to train models on
    along with a last sample array as the last one may not be the same size as the others.
    """
    class_idxs = _generate_class_indices(y)
    class_len = [len(class_idx) for class_idx in class_idxs]
    majority_class_idx = np.argmax(class_len)
    minority_class_idx = int(not majority_class_idx)
    tot_min_samples = class_len[minority_class_idx]
    tot_maj_samples = class_len[majority_class_idx]
    maj_samples_per_sample = int(tot_min_samples / sample_imbalance)
    estimators = math.ceil(tot_maj_samples / maj_samples_per_sample)
    maj_indices = class_idxs[majority_class_idx]
    # maj_samples is a table of estimator-1 rows by maj_samples_per_sample columns
    maj_samples = choice(maj_indices, size=(estimators-1, maj_samples_per_sample), 
                         replace=False, random_state=random_state)
    # last_maj_sample is a different length than each row of maj_samples to get every examle into a sample
    last_maj_sample = np.setxor1d(maj_samples, maj_indices)
    min_indices = class_idxs[minority_class_idx]
    samples = np.hstack((maj_samples, np.tile(min_indices, (estimators-1, 1))))
    last_sample = np.hstack((last_maj_sample, min_indices))
    return samples, last_sample
    

class RepeatedRandomSubSampler(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    This will wrap classifiers to be used with massively unbalanced data and sample and train base estimators
    on every sample, then combine them with majority voting and / or averaged probability

    WARNING: This class is very brittle and should not be used outside this project without more boilerplate, tests
    and error checking
    """

    def __init__(self, base_estimator=None, sample_imbalance=1.0, random_state=None, n_jobs=1, verbose=0):
        """
        sample_imbalance : optional, default = 1.0
            Number from 1.0 to 0.01.  Represents n_minority_class / n_majority_class in each Bag
        """
        self.base_estimator = base_estimator
        self.sample_imbalance = sample_imbalance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        # Check y only 1,0
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("y must be binary for RepeatedRandomSubSampler at this time")
            
        # Check to see base_estimator exists
        if self.base_estimator is None:
            raise ValueError("base_estimator must be defined before running fit")

        # Store the classes seen during fit
        self.n_features_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        #TODO - generate samples and train all classifiers, store them, do it all in parallel

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['estimators_'])

        # Input validation
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        #TODO - do we want to just do pure voting here? Go through self.estimators_ and make a voting / average pred?
        #pr = self.base_estimator.predict_proba(X)[:, 1]
        return #xxx

    def predict_proba(self, X):
        check_is_fitted(self, 'estimators_')

        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if hasattr(self.base_estimator, "predict_proba"):
            probas = [est.predict_proba(X) for est in self.estimators_]
            #TODO - figure out how to take the mean of every probability column among every estimator
            #proba = np.mean(probas, axis=1)
        else:
            raise AttributeError("predict_prob doesn't exist for: {}".format(self.base_estimator))

        return proba
