"""
This will load data for the Creon project and normalize it.
Also some helper functions for loading and saving searches
"""

import pandas as pd
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib

def save_search(search, filename):
    """
    Save a search to disk in a pickle file using joblib
    """
    joblib.dump(search, filename, compress=True)

def load_search(filename):
    return joblib.load(filename)

class LoadCreon:
    """Manage loading a Creon summarized dataset tab delimited into data
    self.data = original data loaded in
    self.X = cleaned, processed data with a Gender Column,
    """

    def __init__(self, path, sep='\t', call_fit=True):
        """ Load data from file in pathh

        Parameters:
        ------------
        path: str,
        from ..exceptions import NotFittedError     passed into pd.read_csv, a file with delimited data
        sep: str, optional, default='\t'
            delimiter of the file, tab by default
        call_fit: Boolean, optional, default=True
            If true, will call fit right away with default argumants, if not, you must call fit separately
        """
        data = pd.read_csv(path, sep=sep, low_memory=False)

        self.data = data
        self.X = None
        self.y = None
        self._cols_to_drop = None
        self._unused_cols = ['unlabel_flag','true_pos_flag','true_neg_flag','MemberID','epi_related_cond',
                          'epi_related_cond_subgrp','h_rank','pert_flag','mmos','elastase_flag','medical_claim_count',
                          'rx_claim_count','CPT_FLAG44_Sum']
        if call_fit:
            self.fit()

    def fit(self, X: pd.DataFrame=None, y: pd.Series=None):
        """
        Transform the data to clear out unwanted columns and columns that provide no information.

        cleans data and prepares it for use in creon models
        For example, if a feature is all 0, then do not use it
        Will create feature for Gender, drop unused or unwanted features, set unlabeled data to y==-1
        Will remember which columns are used for future data coming in for preprocessing
        Parameters
        ----------
        X: default = None, if None then use self.data
            Data to use to fit, should be the same shape as self.Data
        y: must be None
        """
        if X is None:
            X = self.data.copy()
        else:
            X = X.copy()
        if not np.array_equal(X.columns.values, self.data.columns.values):
            raise ValueError("X must have the columns: {}".format(self.data.columns.values))
        # Binar-i-tize the Gender column to 1 or 0
        X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
        # drop all useless columns
        X_sums = X.sum(numeric_only=True)
        cols_to_drop = list(X_sums[X_sums == 0].index)
        X = X.drop(cols_to_drop, axis=1)
        # store useless column headers to drop on transforms in the future
        self._cols_to_drop = cols_to_drop
        X = X.drop(self._unused_cols, axis=1)
        # set up class from the flags in the data with
        # -1 = unlabeled, 0 = true_negative, 1 = true_positive
        y = (self.data.unlabel_flag * -1) + self.data.true_pos_flag
        self.X = X
        self.y = y

    def transform(self, X):
        """
        Parameters
        ----------
        X: [n_samples, features] of claims data

        Returns
        -------
        A processed matrix that transforms X into something useable by the models generated in this package
        """
        if self.X is None:
            raise NotFittedError("Must fit LoadCreon before transforming data!")
        X = X.copy()
        X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
        X = X.drop(self.cols_to_drop, axis=1)
        X = X.drop(self._unused_cols, axis=1)
        return X


if __name__ == "__main__":
    lc = LoadCreon("C:\Data\\010317\membership14_final_0103.txt");