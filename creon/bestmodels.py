#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 07:39:06 2017

This module contains all (err some) of the best models from the project

@author: jgomberg
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from creon.creonsklearn.repeatedsampling import RepeatedRandomSubSampler
from creon.creonsklearn.pnuwrapper import PNUWrapper


def generate_model_6(rf_random_state=324, subsampler_random_state=83, verbose=0):
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
                  verbose=verbose, warm_start=False)
    rep = RepeatedRandomSubSampler(base_estimator=rf, voting='thresh', sample_imbalance= 0.44063408204723742,
                                        verbose=verbose, random_state=subsampler_random_state)
    pnu = PNUWrapper(base_estimator=rep, num_unlabeled=1.0, pu_learning=True, random_state=1)
    return pnu

def generate_best_svc(svc_random_state=551, pnu_random_state=721):
    """ This SVC model was discovered in the notebook 'SVC - PN and PNU.ipynb'.  This model is untrained

    It was found using 3x3 nested cross validation for 20 random iterations optimized to f1 beta=10 on all data
    """
    estimators = [('scaler', MaxAbsScaler()),
              ('clf',PNUWrapper(base_estimator=SVC(C=7.1311952396509097, gamma='auto', kernel='linear',
                                                   probability=True, class_weight='balanced',
                                                   random_state=svc_random_state),
        num_unlabeled=5117, pu_learning=True, random_state=pnu_random_state))]
    pipe = Pipeline(estimators)
    return pipe

def generate_rf_pnu_merged(rf_random_state=77, pnu_random_state=42, verbose=0):
    """ This is a good undersampled RF model found in notebook 'RF - PNU Random Search'

    It was found using 3x3 nested cross validation for 100 iterations optimizing to a merged metric of:
    100 * f1 beta=10 + PU Score
    """
    pnu = PNUWrapper(base_estimator=RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
                                                           criterion='entropy', max_depth=31, max_features=68,
                                                           min_samples_leaf=7, min_samples_split=2, n_estimators=298,
                                                           verbose=verbose, random_state=rf_random_state),
        num_unlabeled=12743, pu_learning=True, random_state=pnu_random_state)
    return pnu

def generate_rf_pnu_f1beta10(rf_random_state=77, pnu_random_state=42, verbose=0):
    """ This is a good undersampled RF model found in notebook 'RF - PNU Random Search'

    It was found using 3x3 nested cross validation for 100 iterations optimizing to f1 beta=10 across all data
    assuming unlabeled data is true negative
    """
    pnu = PNUWrapper(base_estimator=RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                                           criterion='gini', max_depth=45, max_features=81,
                                                           min_samples_leaf=7, min_samples_split=0.005, n_estimators=86,
                                                           verbose=verbose, random_state=rf_random_state),
        num_unlabeled=6377, pu_learning=True, random_state=pnu_random_state)
    return pnu