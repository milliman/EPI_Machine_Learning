#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:30:32 2017

@author: jgomberg
"""

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from creon.creonsklearn.pnuwrapper import PNUWrapper
from creonsklearn.creonmetrics import f1_labeled_scorer


class TestPNUWrapper(unittest.TestCase):

    def setUp(self):
        self.y = np.asarray([1, 1, 1,1, 0,0, 0, 0, -1, -1, -1, -1,-1,-1])
        self.X = np.arange(28).reshape([14,2])

    def test_1(self):
        estimators = [('scaler', MaxAbsScaler()),
              ('clf',PNUWrapper(base_estimator=LogisticRegression(penalty='l1', C=10),
                                num_unlabeled=5819, threshold_set_pct=0.0143))]
        pipe = Pipeline(estimators)
        scores = cross_val_score(pipe, self.X, self.y, cv=2, scoring=f1_labeled_scorer, n_jobs=4)
        pipe.fit(self.X, self.y)
        pipe.predict(self.X)
        pipe.predict_proba(self.X)
        print(scores)



if __name__ == '__main__':
    unittest.main()