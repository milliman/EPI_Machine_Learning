# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 23:41:21 2017
"""

from collections import Iterable

import numpy as np
import pandas as pd
import numbers

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection._validation import indexable
from sklearn.model_selection._split import _CVIterableWrapper, StratifiedKFold, KFold
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed

from .jsearchcv import _fit_and_score_with_extra_data, extract_score_grid

def check_cv2(cv=3, y=None, classifier=False, random_state=None):
    """Input checker utility for building a cross-validator

    NOTE: this is the same as sklearn.model_selection._split.check_cv but with an added parameter for random_state
    So that nested CV splits are reproduceable

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return StratifiedKFold(cv, random_state=random_state)
        else:
            return KFold(cv, random_state=random_state)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification

class NestedCV():
    """ Class to perform validation and keep all the models
    """

    def __init__(self, estimator, scoring=None, cv=None, fit_params=None, random_state=None, use_same_random_state=True):
        """
        Parameters
        ----------
        estimator : Should usually be a grid / random parameter searcher
        use_same_random_state : Boolean, optional, default = True
            if true, will make sure each base estimator gets passed the same random state.
            If this is true, then random_state must be an Integer or Integral
            Use this for random searches where you want the same random parameters to be used across all folds of the
                outer cross validations
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.fit_params = fit_params
        self.random_state = random_state
        self.use_same_random_state = use_same_random_state

    def score(self, X, y=None, groups=None, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        """ Will score the estimator and score according to self.cv
        """
        X, y, groups = indexable(X, y, groups)
        if not isinstance(self.random_state, (numbers.Integral, np.integer)) and self.use_same_random_state:
            raise ValueError("If use_same_randome_state, the random state passed in must be an Integer")
        def clone_estimator():
            """Clone the estimator and put in the correct random state for the nested cross validation
            """
            estimator = clone(self.estimator)
            if self.use_same_random_state and ('random_state' in estimator.get_params().keys()):
                estimator.set_params(random_state=self.random_state)
            return estimator

        cv = check_cv2(self.cv, y, classifier=is_classifier(self.estimator), random_state=self.random_state)
        self.cv_iter_ = list(cv.split(X, y, groups))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)
        scores = parallel(delayed(_fit_and_score_with_extra_data)(clone_estimator(), X, y, scorer,
                                                  train, test, verbose, None,
                                                  self.fit_params, return_train_score=True,
                                                  return_times=True, return_estimator=True)
                          for train, test in self.cv_iter_)

        (self.train_score_datas_, self.train_scores_, self.test_score_datas_, self.test_scores_,
                 self.fit_times_, self.score_times_, self.estimators_) = zip(*scores)

        if hasattr(self.estimators_[0], 'best_params_'):
            self.best_params_ = [estimator.best_params_ for estimator in self.estimators_]
        else:
            print("WARN: NestedCV.best_params_ set to None")
            self.best_params_ = None
        if hasattr(self.estimators_[0], 'best_index_'):
            self.best_idxs_ = [estimator.best_index_ for estimator in self.estimators_]
        else:
            print("WARN: NestedCV.best_idxs_ set to None")
            self.best_idxs_ = None

        return self.test_scores_

def rerun_nested_for_scoring(nested: NestedCV, score: str, X, y=None, groups=None,
                             how='max', n_jobs=1, verbose=0, pre_dispatch='2*n_jobs', return_estimators=False):
    """ Rerun a nested CV grid / random hyper param run but very efficiently by using the stored scoring data
    from a previous run

    Parameters
    ----------
    nested : An already "scored" NestedCV
    score : A string of a score calculated during the scoring run of nested
    how : 'max' or 'min', optional, default='max'
        will look for the min or max of the score provided
    return_estimators : if true return a tuple with new estimators in addition to nested cross, optional, default=False
    Returns
    -------
    nested with new values, (optional, new_estimators)
    """
    sub_scores = [extract_score_grid(searcher) for searcher in nested.estimators_]
    sub_scores_means = [sub_score[[c for c in sub_score.columns if 'test' in c and 'mean' in c]] \
                        for sub_score in sub_scores]
    def create_summary(mean_table):
        return pd.DataFrame({'maxidx':mean_table.idxmax(), 'max':mean_table.max(),
                             'min':mean_table.min(), 'minidx':mean_table.idxmin()})
    sub_scores_summary = [create_summary(mean_table) for mean_table in sub_scores_means]
    row = "mean_{}_test".format(score)
    col = how + "idx"
    idxs = [summary.loc[row, col] for summary in sub_scores_summary]
    params = [pd.DataFrame(estimator.cv_results_)['params'][idx] for idx, estimator in zip(idxs, nested.estimators_)]
    nested.best_params_ = params
    nested.best_idxs_ = idxs
    new_estimators = [clone(estimator.estimator).set_params(**param) for param, estimator in zip(params, nested.estimators_)]
    #set the random state so can reproduce results
    for est in new_estimators:
        est.set_params(random_state=nested.random_state)
    if hasattr(nested.scoring, 'change_decision_score'):
        new_scoring = nested.scoring.change_decision_score(score)
    else:
        new_scoring = nested.scoring
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score_with_extra_data)(estimator, X, y, check_scoring(estimator, new_scoring), train, test,
                      verbose, None, nested.fit_params, return_train_score=True, return_times=True, return_estimator=return_estimators)
        for (train, test), estimator in zip(nested.cv_iter_, new_estimators))
    if return_estimators:
        (nested.train_score_datas_, nested.train_scores_, nested.test_score_datas_, nested.test_scores_,
         nested.fit_times_, nested.score_times_, new_estimators) = zip(*scores)
        return nested, new_estimators
    else:
        (nested.train_score_datas_, nested.train_scores_, nested.test_score_datas_, nested.test_scores_,
         nested.fit_times_, nested.score_times_) = zip(*scores)
        return nested

def rerun_nested_for_estimator(nested: NestedCV, estimator, X, y=None, groups=None,
                               n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
    """ Rerun a nested CV grid / random hyper param run but for just the estimator passed in to get an estimation
    of scores - this is basically a fix for the old way of having different random states on the outer folds.
    It should have been same models in every outer fold but ended up being different so estimates are off

    Returns
    -------
    Messes up internal nested state, returns it (but estimators are still dug in there in inner loops)
    """
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score_with_extra_data)(clone(estimator), X, y,
                              check_scoring(estimator, nested.scoring), train, test,
                      verbose, None, nested.fit_params, return_train_score=True, return_times=True)
        for train, test in nested.cv_iter_)
    (nested.train_score_datas_, nested.train_scores_, nested.test_score_datas_, nested.test_scores_,
                 nested.fit_times_, nested.score_times_) = zip(*scores)
    return nested