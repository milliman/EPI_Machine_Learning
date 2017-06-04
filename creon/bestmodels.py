#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 07:39:06 2017

This module contains all (err some) of the best models from the project

@author: jgomberg
"""

from sklearn.ensemble import RandomForestClassifier

from creon.creonsklearn.repeatedsampling import RepeatedRandomSubSampler
from creon.creonsklearn.pnuwrapper import PNUWrapper


def generate_model_6(rf_random_state=324, subsampler_random_state=83):
    """ Generate model 6 from the memo.  The default random seeds for this function were used for the model
    described in the memo.  The model returned is untrained.

    This model was found using 3x3 nested cross validation for 60 random iterations optimizing to
    PU score + (f1 beta=10 * 100)
    See 'Random Search Nested Cross Repeated Random Sub-Sampling - 3x3x60 with exploration.ipynb'
    """
    rf = RandomForestClassifier(bootstrap=False, class_weight=None,
                  criterion='gini',
                  max_depth=64, max_features=87, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=8,
                  min_samples_split=0.01, min_weight_fraction_leaf=0.0,
                  n_estimators=79, n_jobs=-1, oob_score=False, random_state=rf_random_state,
                  verbose=0, warm_start=False)
    rep = RepeatedRandomSubSampler(base_estimator=rf, voting='thresh', sample_imbalance= 0.44063408204723742,
                                        verbose=1, random_state=subsampler_random_state)
    pnu = PNUWrapper(base_estimator=rep, num_unlabeled=1.0, pu_learning=True, random_state=1)
    return pnu

