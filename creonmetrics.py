#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 07:41:57 2017

@author: jgomberg

This file will contain our custom scoring metrics
"""

from functools import partial
from sklearn.metrics import brier_score_loss, auc, average_precision_score, f1_score
from sklearn.metrics import make_scorer

def pu_score(y_true, y_pred):
   """
   Take truth vs predicted labels and calculate the pu-score, similar to f1-score.

   The formula for the PU score is::

       pu_score = recall ^ 2 / p(y_pred == 1)

   Assumption, label -1 == unlabeled, 0 == negative, 1 == positive
   """

   tp = sum([t == 1 and p == 1 for t, p in zip(y_true, y_pred)])
   n_pos = (y_true == 1).sum() if tp > 0 else 1
   recall = tp / n_pos

   pr_true = (y_pred == 1).sum() / len(y_pred)

   return recall * recall / pr_true

def prior_squared_error(y_true, y_pred, prior):
   """
   This will evaluate how many of the unlabeled data are pos_label and see as a percentage
   how far away it is from the prior

   Assumption: label -1 == unlabeled, 0 == negative, 1 == positive
   """

   unlabeled_pos = sum([t == -1 and p == 1 for t, p in zip(y_true, y_pred)])
   unlabeled_n = (y_true == -1).sum()

   if unlabeled_n == 0:
      #no unlabeleds, so no error
      return 0.0
   else:
      return ((unlabeled_pos / unlabeled_n) - prior) ** 2

def labeled_metric(y_true, y_pred, metric, **kwargs):
    """
    This will wrap a metric so that you can pass it in and it will compute it on labeled instances

    Assumption: label -1 == unlabled, 0 == negative, 1 == positive
    """
    assert len(y_true) == len(y_pred), "y not same lenght! t->p {} {}".format(len(y_true), len(y_pred))
    labeled_mask = y_true != -1
    y_true_labeled = y_true[labeled_mask]
    # check if a probability, then take the last column and use it (probability of the positive class)
    if (len(y_pred.shape) > 1):
        y_pred = y_pred[:,-1]
    y_pred_labeled = y_pred[labeled_mask]
    import pandas as pd
    assert len(y_true_labeled) == len(y_pred_labeled), \
        "Ugg labeled metric y not same lenght! t->p {} {} orig t-> {} {} len_mask={} y_pred_shape={}".format(len(y_true_labeled), len(y_pred_labeled), len(y_true), len(y_pred), pd.Series(labeled_mask).value_counts(), y_pred.shape)
    return metric(y_true_labeled, y_pred_labeled, **kwargs)

def make_label_scorer(metric, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
    """
    This function will create a callable scorer object similar to "sklearn.metrics.make_scorer" but will
    make sure to only use "labeled" data.  The function assumes unlabeled data == -1, while labeled is 0 or 1.
    """
    fn = partial(labeled_metric, metric=metric, **kwargs)
    return make_scorer(fn, greater_is_better=greater_is_better,
                       needs_proba=needs_proba,
                       needs_threshold=needs_threshold, **kwargs)

def assumed_metric(y_true, y_pred, metric, assume_unlabeled=0, **kwargs):
    """
    This will wrap a metric so that you can pass it in and it will compute it on labeled
    and unlabeled instances converting unlabeled to assume_unlabeled

    Assumption: label -1 == unlabled, 0 == negative, 1 == positive
    """
    unlabeled_mask = y_true == -1
    y_true_assume = y_true.copy()
    y_true_assume[unlabeled_mask] = assume_unlabeled
    # check if a probability, then take the last column and use it (probability of the positive class)
    if (len(y_pred.shape) > 1):
        y_pred = y_pred[:,-1]
    return metric(y_true_assume, y_pred, **kwargs)

def make_assumed_scorer(metric, assume_unlabeled=0,
                        greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
    """
    This function will create a callable scorer object similar to "sklearn.metric.make_scorer" but will
    assume all unlabeled data is assume_unlabeled.  The function assumes unlabeled data == -1, while labeled is 0 or 1.
    """
    fn = partial(assumed_metric, metric=metric, assume_unlabeled=0, **kwargs)
    return make_scorer(fn, greater_is_better=greater_is_better,
                       needs_proba=needs_proba,
                       needs_threshold=needs_threshold, **kwargs)



# Scorers for model selection
pu_scorer = make_scorer(pu_score)
prior_squared_error_scorer_015 = make_scorer(prior_squared_error, greater_is_better=False, prior=0.015)
brier_score_labeled_loss_scorer = make_label_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
brier_score_assumed_loss_scorer = make_assumed_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
f1_labeled_scorer = make_label_scorer(f1_score)
f1_assumed_scorer = make_assumed_scorer(f1_score)