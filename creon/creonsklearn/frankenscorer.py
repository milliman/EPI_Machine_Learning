# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:03:22 2017

@author: jeffrey.gomberg
"""
from collections import defaultdict

import pandas as pd
from .creonmetrics import labeled_metric, assumed_metric, pu_score, pr_one_unlabeled, brier_score_partial_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, brier_score_loss, fbeta_score, confusion_matrix

def extract_scores_from_nested(scores):
    """ Extract scores from a sequence of dicts
    Returns a DataFrame where rows are CV, columns scores - use in conjuntion with NestedCV
    """
    row_dict = defaultdict(dict)
    for i, split_score_dict in enumerate(scores):
        d = {}
        for k, v in split_score_dict.items():
            if hasattr(v, "shape") and v.shape == (2, 2):
                tn, fp, fn, tp = v.ravel()
                d["tn_%s" % k] = tn
                d["fp_%s" % k] = fp
                d["fn_%s" % k] = fn
                d["tp_%s" % k] = tp
            if FrankenScorer.score_index != k:
                #don't include the "SCORE" score in the grid
                d[k] = v
        row_dict[i].update(d)

    score_grid = pd.DataFrame.from_dict(row_dict, orient="index")
    return score_grid

def get_mean_test_scores(score_grid):
    """ Return the "mean" and "test" columns of the score grid dataset
    """
    score_grid = score_grid.copy()
    return score_grid[[c for c in score_grid.columns if 'test' in c and 'mean' in c]]

class FrankenScorer():
    score_index = "SCORE"

    def __init__(self, decision_score='labeled_f1'):
        self.decision_score = decision_score

    """
    This is a sklearn scorer object that returns a (dictionary, Number) instead of just a Number.
    Will not work without modified sklearn modules that can handle a scorer that returns a dict with a Number
    For example 
    """
    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict(X)
        y_prob = estimator.predict_proba(X)

        pu_score_num = pu_score(y_true, y_pred)
        assumed_f1beta10 = assumed_metric(y_true, y_pred, fbeta_score, beta=10)
        pu_mix_assumed_f1beta10 = (assumed_f1beta10 * 100.0) + pu_score_num

        data = {'labeled_acc' : labeled_metric(y_true, y_pred, accuracy_score),
            'labeled_prec' : labeled_metric(y_true, y_pred, precision_score),
            'labeled_recall' : labeled_metric(y_true, y_pred, recall_score),
            'labeled_f1' : labeled_metric(y_true, y_pred, f1_score),
            'labeled_roc_auc' : labeled_metric(y_true, y_pred, roc_auc_score),
            'labeled_avg_prec' : labeled_metric(y_true, y_pred, average_precision_score),
            'labeled_brier' : labeled_metric(y_true, y_prob, brier_score_loss),
            'labeled_brier_pos' : labeled_metric(y_true, y_prob, brier_score_partial_loss, label=1),
            'labeled_brier_neg' : labeled_metric(y_true, y_prob, brier_score_partial_loss, label=0),
            'confusion_matrix_lab' : labeled_metric(y_true, y_pred, confusion_matrix),
            'pr_one_unlabeled' : pr_one_unlabeled(y_true, y_pred),
            'assumed_brier' : assumed_metric(y_true, y_prob, brier_score_loss),
            'assumed_brier_neg' : assumed_metric(y_true, y_prob, brier_score_partial_loss, label=0),
            'assumed_f1' : assumed_metric(y_true, y_pred, f1_score),
            'assumed_f1beta10' : assumed_f1beta10,
            'confusion_matrix_un' : assumed_metric(y_true, y_pred, confusion_matrix),
            'pu_score' : pu_score_num,
            'pu_mix_assumed_f1beta10' : pu_mix_assumed_f1beta10,
            }

        ret = data[self.decision_score]
        data[self.score_index] = ret

        return data, ret

    def change_decision_score(self, decision_score):
        self.decision_score = decision_score
        return self

