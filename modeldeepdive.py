# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:52:18 2017

@author: jeffrey.gomberg
"""

import pandas as pd
import numpy as np
import scipy as sp

from loadcreon import LoadCreon
from creonmetrics import pu_scorer, prior_squared_error_scorer_015, brier_score_labeled_loss_scorer, \
    f1_assumed_scorer, f1_labeled_scorer, report_metrics, f1_assumed_beta10_scorer, pu_mix_assumed_f1beta10_scorer
from semisuperhelper import SemiSupervisedHelper
from pnuwrapper import PNUWrapper
from jeffsearchcv import JeffRandomSearchCV
from nestedcross import NestedCV
from frankenscorer import FrankenScorer, extract_scores_from_nested, extract_score_grid
from searchrf import save_search, load_search
from repeatedsampling import RepeatedRandomSubSampler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

import lime
import lime.lime_tabular

class ModelDeepDive():
    """
    Wraps a model and can be used to generate all tables, graphs, explanations, etc.
    Allows us to do a deep dive analysis of a model
    """

    def __init__(self, clf, explainer: lime.lime_tabular.LimeTabularExplainer, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Parameters:
        -------------------
        clf : Classifier from sklearn
            If None, will train the model
        explainer : LimeTabularExplainer - used to explain the model
        X_test : n x f pd.DataFrame in same format as clf was trained with
        y_test : n pd.Series in same format as clf was trained with
        """
        self.clf = clf
        self.explainer = explainer
        self.X_test = X_test
        self.y_test = y_test


    def exaplin_example_code(self, row):
        exp = self.explainer.explain_instance(row, self.clf.predict_proba, num_features=30, num_samples=10000)
        with open('xxxx.html', 'w', encoding='utf-8') as t:
            t.write(exp.as_html())

        print(exp.as_list())
        print(exp.as_map())



def create_model_6(X_train, y_train):
    rf = RandomForestClassifier(bootstrap=False, class_weight=None,
                  criterion='gini',
                  max_depth=64, max_features=87, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=8,
                  min_samples_split=0.01, min_weight_fraction_leaf=0.0,
                  n_estimators=79, n_jobs=-1, oob_score=False, random_state=324,
                  verbose=0, warm_start=False)
    rep_test = RepeatedRandomSubSampler(base_estimator=rf, voting='thresh', sample_imbalance= 0.44063408204723742,
                                        verbose=1, random_state=83)
    pnu_test = PNUWrapper(base_estimator=rep_test, num_unlabeled=1.0, pu_learning=True, random_state=1)
    pnu_test.fit(X_train.values, y_train.values)
    return pnu_test

#TODO - fill in feature_names list of LimeTabularExplainer for more conherant explanations
def create_explainer(X_train, y_train):
    return lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values,
                                                  training_labels=y_train.values,
                                                  feature_selection='lasso_path', class_names=['No EPI', 'EPI'],
                                                  discretize_continuous=True, discretizer='entropy')

if __name__ == "__main__":
#    path = "C:\Data\\010317\membership14_final_0103.txt"
#    print("Loading {}".format(path))
#    lc = LoadCreon(path)
#    print("Done loading")
#    X_train, X_test, y_train, y_test = train_test_split(lc.X, lc.y, test_size=0.2, random_state=771, stratify=lc.y)
#    print("Split Data: train_size {}, test_size {}".format(X_train.shape, X_test.shape))
#    model6 = create_model_6(X_train, y_train)
#    scores, _ = FrankenScorer()(model6, X_test.values, y_test.values)
    print(scores)