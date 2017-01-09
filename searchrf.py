# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:11:25 2017

@author: jeffrey.gomberg
"""

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from loadcreon import LoadCreon
from creonmetrics import pu_scorer, prior_squared_error_scorer_015, \
    brier_score_labeled_loss_scorer, f1_assumed_scorer, f1_labeled_scorer
from semisuperhelper import SemiSupervisedHelper
from pnuwrapper import PNUWrapper




#TODO - write functions here!







if __name__ == "__main__":
    path = "C:\Data\\010317\membership14_final_0103.txt"
    lc = LoadCreon(path)
    X_train, X_test, y_train, y_test = train_test_split(lc.X, lc.y, test_size=0.2, random_state=771, stratify=lc.y)
    rf_param_search = {'base_estimator__n_estimators':sp.stats.randint(low=10, high=1000),
                       'num_unlabeled':sp.stats.randint(low=2000, high=15000),
                       'threshold_set_pct':[0.0143, None],
                       'base_estimator__max_features':['sqrt','log2',5, 10, 20, 50, None],
                       'base_estimator__max_depth':sp.stats.randint(low=2, high=50),
                       'base_estimator__min_samples_split':sp.stats.uniform(loc=0, scale=1),
                       'base_estimator__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,50,100],
                       'base_estimator__class_weight':[None,'balanced','balanced_subsample']}
    rf = RandomForestClassifier()
    pnu = PNUWrapper(base_estimator=rf, num_unlabeled=5819, threshold_set_pct=0.0143, random_state=77)
    random_rf_searcher = RandomizedSearchCV(pnu, rf_param_search, n_iter=50, scoring=pu_scorer, n_jobs=-1, cv=5)
    random_rf_searcher.fit(X_train.values, y_train.values)