# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 23:41:21 2017

@author: jeffrey.gomberg
"""

import numpy as np

from sklearn.model_selection._validation import indexable
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed

from jeffsearchcv import _fit_and_score_with_extra_data

#TODO may want to write some helper functions to re-calc a search, find best estimators from grids
#and then train those models with those parameters and use the self.cv_iter_ to figure out how to get
#a good estimate of other folds

class NestedCV():
    """ Class to perform validation and keep all the models
    """

    def __init__(self, estimator, scoring=None, cv=None, fit_params=None, random_state=None):
        """
        Parameters
        ----------
        estimator : Should usually be a grid / random parameter searcher
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.fit_params = fit_params
        self.random_state = random_state

    def score(self, X, y=None, groups=None, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        """ Will score the estimator and score according to self.cv
        """
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.cv_iter_ = list(cv.split(X, y, groups))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)
        scores = parallel(delayed(_fit_and_score_with_extra_data)(clone(self.estimator), X, y, scorer,
                                                  train, test, verbose, None,
                                                  self.fit_params, return_train_score=True,
                                                  return_times=True, return_estimator=True)
                          for train, test in self.cv_iter_)

        (self.train_score_datas_, self.train_scores_, self.test_score_datas_, self.test_scores_,
                 self.fit_times_, self.score_times_, self.estimators_) = zip(*scores)

        return self.test_scores_
