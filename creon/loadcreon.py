"""
This will load data for the Creon project and normalize it.
Also some helper functions for loading and saving searches
"""

import pandas as pd

from sklearn.exceptions import NotFittedError, ChangedBehaviorWarning
from sklearn.externals import joblib

def save_search(search, filename):
    """
    Save a search to disk in a pickle file using joblib
    """
    joblib.dump(search, filename, compress=True)

def load_search(filename):
    return joblib.load(filename)

# TODO - run a test that, with same column headers and random data,
#lc = LoadCreon()
#lx.X is None
#X = lc.data
#lc.fit(X)
#lc.X == lc.transform(X)

class LoadCreonTransformer:
    """Transforms a dataset by cleaning data, normalizing features, and dropping unused and unnecessary columns
    """

    def __init__(self):
        self.is_fit = False
        self._cols_to_drop = None
        self._unused_cols = None
        self._cols_to_binarize = None
        self._orig_col_headers = None

    def fit(self, X: pd.DataFrame, y = None):
        if self.is_fit:
            raise ChangedBehaviorWarning()
        else:
            self.is_fit = True
        X = X.copy()
        self._orig_col_headers = X.columns.values
        self._cols_to_binarize = ['Gender']
        X = pd.get_dummies(X, columns=self._cols_to_binarize, drop_first=True)
        # drop all useless columns
        self._unused_cols = ['unlabel_flag','true_pos_flag','true_neg_flag','MemberID','epi_related_cond',
                          'epi_related_cond_subgrp','h_rank','pert_flag','mmos','elastase_flag','medical_claim_count',
                          'rx_claim_count','CPT_FLAG44_Sum']
        X = X.drop(self._unused_cols, axis=1)
        X_sums = X.sum(numeric_only=True)
        self._cols_to_drop = list(X_sums[X_sums == 0].index)
        return self

    def transform(self, X):
        if not self.is_fit:
            raise NotFittedError("This LoadCreonTransformer is not fitted yet")
        X = X.copy()
        X_cols = set(X.columns.values)
        data_cols = set(self._orig_col_headers)
        if X_cols != data_cols:
            missing_cols = data_cols - X_cols
            extra_cols = X_cols - data_cols
            raise ValueError("X missing {} cols [{}], and has {} extra cols [{}]".format(len(missing_cols),
                             missing_cols, len(extra_cols), extra_cols))
        #binar-i-tize data
        X = pd.get_dummies(X, columns=self._cols_to_binarize, drop_first=True)
        # drop columns
        X = X.drop(self._cols_to_drop, axis=1)
        X = X.drop(self._unused_cols, axis=1)
        return X


class LoadCreon:
    """Manage loading a Creon summarized dataset tab delimited into data
    self.data = original data loaded in
    self.X = cleaned, processed data with a Gender Column,
    """

    def __init__(self, path, sep='\t', call_fit=True):
        """ Load data from file in path, will set up 'y' for unlabeled data = -1, 0 = negative, 1 = positive

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
        # set up class from the flags in the data with
        # -1 = unlabeled, 0 = true_negative, 1 = true_positive
        y = (self.data.unlabel_flag * -1) + self.data.true_pos_flag
        self.y = y
        self.transformer = LoadCreonTransformer()
        if call_fit:
            self.transformer.fit(self.data, self.y)

    def fit(self, X: pd.DataFrame=None, y: pd.Series=None):
        """
        Transform the data to clear out unwanted columns and columns that provide no information.

        cleans data and prepares it for use in creon models
        For example, if a feature is all 0, then do not use it
        Will create feature for Gender, drop unused or unwanted features
        Will remember which columns are used for future data coming in for preprocessing
        Parameters
        ----------
        X: default = None, required, if left None will raise an exception
            Data to use to fit
        y: must be None
        """
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: [n_samples, features] of claims data

        Returns
        -------
        A processed matrix that transforms X into something useable by the models generated in this package
        """
        return self.transformer.transform(X)

if __name__ == "__main__":
    lc = LoadCreon("C:\Data\\010317\membership14_final_0103.txt");