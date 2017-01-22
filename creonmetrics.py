#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 07:41:57 2017

@author: jgomberg

This file will contain our custom scoring metrics
"""

import pandas as pd
import numpy as np
from functools import partial
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, f1_score, fbeta_score, \
    accuracy_score, recall_score, precision_score
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

   if recall == 0.0:
       return 0.0

   pr_true = (y_pred == 1).sum() / len(y_pred)

   return recall * recall / pr_true

def brier_score_partial_loss(y_true, y_prob, sample_weight=None, label=None):
    """ Compute the partial brier score

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    label : int, default=None
        Label of the class we will compute the brier score for. If None, this is a normal brier score

    Returns
    -------
    score : float
        Brier score for label
    """

    if label is not None:
        mask = y_true == label
        y_true = y_true[mask]
        y_prob = y_prob[mask]
        sample_weight = sample_weight[mask] if sample_weight is not None else sample_weight

    return np.average((y_true - y_prob) ** 2, weights=sample_weight)

def report_metrics(clf, X, y_true):
    """
    Take in a fitted classifier and a dataset with true values, output a pd.Series of metrics
    """
    y_prob = clf.predict_proba(X)
    y_pred = clf.predict(X)
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

def pr_one_unlabeled(y_true, y_pred):
    """
    probability that unabeleds are predicted to be class == 1
    """
    unlabeled_pos = sum([t == -1 and p == 1 for t, p in zip(y_true, y_pred)])
    unlabeled_n = (y_true == -1).sum()

    if unlabeled_n == 0:
        #no unlabeleds, so no pr
        return 0.0
    else:
        return unlabeled_pos / unlabeled_n

def prior_squared_error(y_true, y_pred, prior):
    """
    This will evaluate how many of the unlabeled data are pos_label and see as a percentage
    how far away it is from the prior

    Assumption: label -1 == unlabeled, 0 == negative, 1 == positive
    """

    pr = pr_one_unlabeled(y_true, y_pred)
    if pr == 0.0:
      return 0.0
    else:
      return (pr - prior) ** 2

def labeled_metric(y_true, y_pred, metric, **kwargs):
    """
    This will wrap a metric so that you can pass it in and it will compute it on labeled instances

    Assumption: label -1 == unlabled, 0 == negative, 1 == positive
    """
    labeled_mask = y_true != -1
    y_true_labeled = y_true[labeled_mask]
    # check if a probability, then take the last column and use it (probability of the positive class)
    if (len(y_pred.shape) > 1):
        y_pred = y_pred[:,-1]
    y_pred_labeled = y_pred[labeled_mask]
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
f1_assumed_beta10_scorer = make_assumed_scorer(fbeta_score, beta=10)
