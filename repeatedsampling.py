#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:09:12 2017

@author: jgomberg
"""


import numpy as np
import math

from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.random import choice

__all__ = ["RepeatedRandomSubSampler"]

def _generate_class_indices(y):
    """ Generate all the indices of each class in ascending order"""
    return [np.where(y==c)[0] for c in np.unique(y)]

def _generate_repeated_sample_indices(random_state, sample_imbalance, y, verbose):
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
    if verbose > 0:
        print("generating {} samples of indices to use to train multiple estimators, \
              sized {} elements with last being {} elements".format(
                      len(samples) + 1, len(samples[0]), len(last_sample)))
    return samples, last_sample

def _parallel_fit_base_estimator(estimator, X, y):
    estimator.fit(X, y)
    return estimator

class RepeatedRandomSubSampler(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    This will wrap classifiers to be used with massively unbalanced data and sample and train base estimators
    on every sample, then combine them with majority voting and / or averaged probability

    WARNING: This class is very brittle and should not be used outside this project without more boilerplate, tests
    and error checking
    """

    def __init__(self, base_estimator=None, sample_imbalance=1.0, voting='hard',
                 random_state=None, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        """
        sample_imbalance : optional, default = 1.0
            Number from 1.0 to 0.01.  Represents n_minority_class / n_majority_class in each Bag

        voting : str, {'hard', 'soft'} (default = 'hard')
            If 'hard', uses predicted class labels for majroity rule voting.
            Else if 'soft', predicts the class label based on the argmax of the of the predicted probabilities,
            which is recommended for well calibrated classifiers
        """
        self.base_estimator = base_estimator
        self.sample_imbalance = sample_imbalance
        self.voting = voting
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
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

        base_estimator = clone(self.base_estimator)

        # Store the classes seen during fit
        self.n_features_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        samples, last_sample = _generate_repeated_sample_indices(random_state, self.sample_imbalance, y, self.verbose)
        samples_indices = list(samples)
        samples_indices.extend([last_sample])
        self.samples_indices_ = samples_indices

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)
        estimators = parallel(delayed(_parallel_fit_base_estimator)(clone(base_estimator), X[indices,:], y[indices])
                       for indices in samples_indices)

        self.estimators_ = estimators

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

        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        elif self.voting == 'hard':
            predictions = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)(
            delayed(parallel_helper)(estimator, 'predict', X) for estimator in self.estimators_)
            predictions = np.asarray(predictions)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

        return maj

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
            predictions = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)(
                    delayed(parallel_helper)(estimator, 'predict_proba', X) for estimator in self.estimators_)
            predictions = np.array(predictions)

            avg = np.average(predictions, axis=0)
        else:
            raise AttributeError("predict_prob doesn't exist for: {}".format(self.base_estimator))

        return avg

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'estimators_')

        if hasattr(self.estimators_[0], 'feature_importances_'):
            all_feature_importances = np.array([est.feature_importances_ for est in self.estimators_])
            return np.mean(all_feature_importances, axis=0)
        else:
            raise AttributeError("feature_importances_ doesn't exist for: {}".format(self.base_estimator))

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, weights=[0.8, 0.2])
    sub = RepeatedRandomSubSampler(RandomForestClassifier(n_estimators=50), voting='hard', n_jobs=1, verbose=1)
    #sub.fit(X, y)
