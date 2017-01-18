# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:03:22 2017

@author: jeffrey.gomberg
"""
import copy
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, brier_score_loss, fbeta_score, confusion_matrix


from creonmetrics import labeled_metric, assumed_metric, pu_score, pr_one_unlabeled
from jeffsearchcv import JeffRandomSearchCV

def extract_score_grid(searcher: JeffRandomSearchCV):
    """
    Take a fitted scorer that used a FrankenScorer() and extract the scoring data into a scoring grid

    The scorer must have cv_results_ as an attribute

    TODO - finish this comment, error checking, and break up into fewer functions
    """
    results = pd.DataFrame(copy.deepcopy(searcher.cv_results_))
    splits = search.cv if search.cv is not None else 3
    rows = len(results)
    #create master_dict of scores
    master_dict = {}
    for row in rows:
        row_dict = defaultdict(dict)
        for split in range(splits):
            for tpe in ['test','train']:
                split_score_dict = copy.deepcopy(results['split{}_{}_score_data'.format(str(split), tpe)].iloc[row])
                d = {}
                for k, v in split_score_dict.items():
                    new_key = "{}_{}{}".format(k,tpe,split)
                    if hasattr(v, 'shape') and v.shape == (2, 2):
                        #confusion matric deconstruction
                        tn, fp, fn, tp = v.ravel()
                        d["tn_%s" % new_key] = tn
                        d["fp_%s" % new_key] = fp
                        d["fn_%s" % new_key] = fn
                        d["tp_%s" % new_key] = tp
                    if FrankenScorer.score_index != k:
                        #don't include the "SCORE" score in the grid
                        d[new_key] = v
                row_dict[row].update(d)
        master_dict.update(row_dict)

    score_grid = pd.DataFrame.from_dict(master_dict, orient="index")
    score_labels = set([s[:-1] for s in score_grid.columns])

    #compute mean and std
    for label in score_labels:
        label_score_grid = score_grid[[s for s in score_grid.columns if label == s[:-1]]]
        mean_for_label = label_score_grid.mean(axis=1)
        std_for_label = label_score_grid.std(axis=1)
        score_grid["mean_{}".format(label)] = mean_for_label
        score_grid["std_{}".format(label)] = std_for_label

    return score_grid

#class TurnOffScoreCheck(ContextDecorator):
#    """
#    Use this to turn off the check in sklearn that checks if a scorer returns a Number or not.
#
#    Example usages::
#        @TurnOffScoreCheck()
#        def scorers_gone_wild(clf, X, y, scorer=frankenscorer)
#
#        with TurnOffScoreCheck():
#            scorers_gone_wile(clf, X, y, scorer=frankenscorer)
#    """
#    def __enter__(self):
#        self.old_score_fn = skvalid._score
#        skvalid._score = _score_no_number_check
#
#    def __exit__(self, *exc):
#        skvalid._score = self.old_score_fn

class FrankenScorer():
    score_index = "SCORE"

    """
    This is a sklearn scorer object that returns a (dictionary, Number) instead of a number
    """
    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict(X)
        y_prob = estimator.predict_proba(X)

        data = {'labeled_acc' : labeled_metric(y_true, y_pred, accuracy_score),
            'labeled_prec' : labeled_metric(y_true, y_pred, precision_score),
            'labeled_recall' : labeled_metric(y_true, y_pred, recall_score),
            'labeled_f1' : labeled_metric(y_true, y_pred, f1_score),
            'labeled_roc_auc' : labeled_metric(y_true, y_pred, roc_auc_score),
            'labeled_avg_prec' : labeled_metric(y_true, y_pred, average_precision_score),
            'confusion_matrix_lab' : labeled_metric(y_true, y_pred, confusion_matrix),
            'pr_one_unlabeled' : pr_one_unlabeled(y_true, y_pred),
            'labeled_brier' : labeled_metric(y_true, y_prob, brier_score_loss),
            'assumed_brier' : assumed_metric(y_true, y_prob, brier_score_loss),
            'assumed_f1' : assumed_metric(y_true, y_pred, f1_score),
            'assumed_f1beta10' : assumed_metric(y_true, y_pred, fbeta_score, beta=10),
            'confusion_matrix_un' : assumed_metric(y_true, y_pred, confusion_matrix),
            'pu_score' : pu_score(y_true, y_pred),
            }

        ret = data['assumed_f1beta10']
        data[self.score_index] = ret

        #TODO: return f_beta10 for now, in future either pass in a way to score this or a custom metric
        return data, ret


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    clf = RandomForestClassifier()
    import scipy as sp
    params = {'n_estimators': sp.stats.randint(low=10, high=500),
              'max_depth':[None, 1, 2, 3, 4, 5, 10, 20]}
    search = JeffRandomSearchCV(clf, params, scoring=FrankenScorer(), n_iter=2, verbose=100)
    search.fit(X, y)

    print(search.cv_results_)

    #LINE 266 of _validation.py can't print out a non-float score!!! (only verbose > 2) (function _fit_and_score())
    #LINE 582 in _search.py can't reshape an array with other arrays, dtype=np.float64 may be a problem (_fit())
    #So I may need to rewrite / copy / paste _fit_and_score() and _fit(), maybe make the FrankenScorer
    #return an object of state along with a number so that I can then store that object

