#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 07:44:05 2017

@author: jgomberg
"""
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

__all__ = ["BlaggingClassifier"]

MAX_INT = np.iinfo(np.int32).max

def _generate_class_indices(y):
    return [np.where(y==c)[0] for c in np.unique(y)]

def _generate_indices(random_state, bootstrap, n_population, n_samples, verbose,
                      sample_imbalance=None, y=None):
    """Draw randomly sampled indices."""

    #Find how many samples to take
    if sample_imbalance is not None:
        class_idxs = _generate_class_indices(y)
        class_len = [len(class_idx) for class_idx in class_idxs]
        minority_class_idx = np.argmin(class_len)
        majority_class_idx = np.argmax(class_len)
        min_samples = class_len[minority_class_idx]
        maj_samples = int(min_samples / sample_imbalance)
        maj_n = class_len[majority_class_idx]
        if maj_samples > maj_n:
            raise ValueError("more majority samples than exist in majority class %i > %i" % (maj_samples, maj_n))
        if (maj_samples + min_samples) > n_samples:
            #keep ratio but cut samples
            excess = maj_samples + min_samples - n_samples
            maj_samples = max(maj_samples - math.ceil(excess / 2.0), 1)
            min_samples = max(min_samples - math.floor(excess / 2.0), 1)
        maj_indices = choice(class_idxs[majority_class_idx], size=maj_samples, replace=bootstrap, random_state=random_state)
        min_indices = choice(class_idxs[minority_class_idx], size=min_samples, replace=bootstrap, random_state=random_state)
        if (verbose > 0):
            print("majority_class_smaples: {} minority_class_samples: {} for bag".format(maj_samples, min_samples))
        indices = np.hstack((maj_indices, min_indices))
    elif bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples, sample_imbalance, y, verbose):
    """Randomly draw feature and sample indices."""
    if sample_imbalance is not None and y is None:
        raise ValueError('y cannot be None if sample_imbalance is set')

    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features, verbose)
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples, verbose,
                                       sample_imbalance=sample_imbalance, y=y)

    return feature_indices, sample_indices


def _parallel_build_estimators_balanced(job_number, n_estimators, ensemble, X, y, sample_weight,
                                        sample_imbalance, seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
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
            print("Job: %d - Building estimator %d of %d for this parallel run (total %d)..." %
                  (job_number, i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples,
                                                      sample_imbalance, y, verbose)

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

class BlaggingClassifier(BaggingClassifier):
    """ Blagging Classifier - just like Bagging but with more options to deal with
    imblanced classes.

    See :ref: Bagging in sklearn.ensembles and
    :ref: https://svds.com/learning-imbalanced-classes/
    For similar concepts and ideas

    Parameters
    ----------
    sample_imbalance : optional, default = 1.0
        Number from 1.0 to 0.01.  Represents n_minority_class / n_majority_class in each Bag
        If None then there is no balanced downsampling

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 sample_imbalance=1.0,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        self.sample_imbalance = sample_imbalance

        super(BaggingClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.

        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("y must be binary for blagging at this time")

        # Check parameters
        self._validate_estimator()

        if self.sample_imbalance <= 0.0 or self.sample_imbalance > 1.0:
            raise ValueError("sample_imbalance must not be <= 0.0 or > 1.0")

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or len(self.estimators_) == 0:
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators_balanced)(i,
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                self.sample_imbalance,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    from sklearn.ensemble import RandomForestClassifier
    clf = BlaggingClassifier(RandomForestClassifier(), n_estimators=20)
    #with BaggingTransformedIntoBalancedSampler():
    clf.fit(X, y)

