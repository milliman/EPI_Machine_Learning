#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 07:44:05 2017

@author: jgomberg
"""

import numpy as np
from contextlib import ContextDecorator
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble.bagging import _generate_indices
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.fixes import bincount
from sklearn.utils import indices_to_mask
from sklearn.utils import check_random_state

__all__ = ["BlaggingClassifier"]

MAX_INT = np.iinfo(np.int32).max

def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features)
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                       seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    print("RIGHT CALL!!!")
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run (total %d)..." %
                  (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        # Draw samples, using a mask, and then fit
        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features

class BaggingTransformedIntoBalancedSampler(ContextDecorator):
    """
    Use this to map functions in Bagging Classifier to custom versions.
    WARNING: This is not threadsafe!

    Example usages::
        @BaggingTransformedIntoBalancedSampler()
        def bag_it(clf, X, y, scorer=frankenscorer)

        with BaggingTransformedIntoBalancedSampler():
            bagger.fit(X, y)
    """
    _orig_parallel_build_estimators = None

    def __enter__(self):
        if self._orig_parallel_build_estimators is None:
            self.set_original_parallel_build_estimators(sklearn.ensemble.bagging._parallel_build_estimators)
            sklearn.ensemble.bagging._parallel_build_estimators =_parallel_build_estimators

    def __exit__(self, *exc):
        sklearn.ensemble.bagging._parallel_build_estimators = self._orig_parallel_build_estimators
        self.reset_original_parallel_build_estimators()

    @classmethod
    def set_original_parallel_build_estimators(cls, fn):
        cls._orig_parallel_build_estimators = fn

    @classmethod
    def reset_original_parallel_build_estimators(cls):
        cls._orig_parallel_build_estimators = None

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    from sklearn.ensemble import RandomForestClassifier
    clf = BaggingClassifier(RandomForestClassifier())
    #with BaggingTransformedIntoBalancedSampler():
    clf.fit(X, y)

