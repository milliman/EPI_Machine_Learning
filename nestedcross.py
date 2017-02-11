# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 23:41:21 2017

@author: jeffrey.gomberg
"""

import numpy as np
import pandas as pd
import numbers

from sklearn.model_selection._validation import indexable
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state

from jeffsearchcv import _fit_and_score_with_extra_data
from frankenscorer import extract_score_grid

#TODO may want to write some helper functions to re-calc a search, find best estimators from grids
#and then train those models with those parameters and use the self.cv_iter_ to figure out how to get
#a good estimate of other folds

class NestedCV():
    """ Class to perform validation and keep all the models
    """

    def __init__(self, estimator, scoring=None, cv=None, fit_params=None, random_state=None, use_same_random_state=True):
        """
        Parameters
        ----------
        estimator : Should usually be a grid / random parameter searcher
        use_same_random_state : if true, will make sure each base estimator gets passed the same ranomd state.
            If this is true, then random_state must be an Integer or Integral
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
            if self.use_same_random_state:
                estimator.set_params(random_state=self.random_state)
            return estimator

        #TODO - redo this CV logic so that self.random_state is involved
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
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

        self.best_params_ = [estimator.best_params_ for estimator in self.estimators_]
        self.best_idxs_ = [estimator.best_index_ for estimator in self.estimators_]

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