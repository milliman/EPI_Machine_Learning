# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:03:22 2017

@author: jeffrey.gomberg
"""

from contextlib import ContextDecorator

import pandas as pd
import sklearn.model_selection._validation as skvalid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, brier_score_loss, fbeta_score

from creonmetrics import labeled_metric, assumed_metric, pu_score, pr_one_unlabeled


def _score_no_number_check(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set. Take out the isNumber check."""
    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, 'item'):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    return score

class TurnOffScoreCheck(ContextDecorator):
    """
    Use this to turn off the check in sklearn that checks if a scorer returns a Number or not.

    Example usages::
        @TurnOffScoreCheck()
        def scorers_gone_wild(clf, X, y, scorer=frankenscorer)

        with TurnOffScoreCheck():
            scorers_gone_wile(clf, X, y, scorer=frankenscorer)
    """
    def __enter__(self):
        self.old_score_fn = skvalid._score
        skvalid._score = _score_no_number_check

    def __exit__(self, *exc):
        skvalid._score = self.old_score_fn

class FrankenScorer():
    """
    This is a sklearn scorer object that returns a dictionary instead of a number

    TODO - fiture out how to override comparison and use some type of customer real score to sort these things?
    Maybe passed in so that it is defined what crazy metric you really want to use
    """
    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict(X)
        y_prob = estimator.predict_proba(X)

        ret = {'labeled_acc' : labeled_metric(y_true, y_pred, accuracy_score),
            'labeled_prec' : labeled_metric(y_true, y_pred, precision_score),
            'labeled_recall' : labeled_metric(y_true, y_pred, recall_score),
            'labeled_f1' : labeled_metric(y_true, y_pred, f1_score),
            'labeled_roc_auc' : labeled_metric(y_true, y_pred, roc_auc_score),
            'labeled_avg_prec' : labeled_metric(y_true, y_pred, average_precision_score),
            'pr_one_unlabeled' : pr_one_unlabeled(y_true, y_pred),
            'labeled_brier' : labeled_metric(y_true, y_prob, brier_score_loss),
            'assumed_brier' : assumed_metric(y_true, y_prob, brier_score_loss),
            'assumed_f1' : assumed_metric(y_true, y_pred, f1_score),
            'assumed_f1beta10' : assumed_metric(y_true, y_pred, fbeta_score, beta=10),
            'pu_score' : pu_score(y_true, y_pred)}
        return pd.Series(ret)

if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    clf = RandomForestClassifier()
    search = GridSearchCV(clf, {'n_estimators':[10,20,30]}, scoring=FrankenScorer())
    with TurnOffScoreCheck():
        search.fit(X, y)

    print(search.cv_results_)

    #LINE 266 of _validation.py can't print out a non-float score!!! (only verbose > 2) (function _fit_and_score())
    #LINE 582 in _search.py can't reshape an array with other arrays, dtype=np.float64 may be a problem (_fit())
    #So I may need to rewrite / copy / paste _fit_and_score() and _fit(), maybe make the FrankenScorer
    #return an object of state along with a number so that I can then store that object
