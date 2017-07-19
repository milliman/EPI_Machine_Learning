"""
This will load data for the Epiml project and normalize it.
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

class LoadEpimlTransformer:
    """Transforms a dataset by cleaning data, normalizing features, and dropping unused and unnecessary columns

    This class can be used as a scikit-learn transformer in a pipeline so that the data cleaning step is used
    consistently across training data and future patient data.
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
        X = X.drop(self._unused_cols, axis=1, errors='ignore')
        X_sums = X.sum(numeric_only=True)
        self._cols_to_drop = list(X_sums[X_sums == 0].index)
        return self

    def transform(self, X):
        if not self.is_fit:
            raise NotFittedError("This LoadEpimlTransformer is not fitted yet")
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
        X = X.drop(self._unused_cols, axis=1, errors='ignore')
        return X


class LoadEpiml:
    """Manage loading a Epiml summarized dataset
    self.data = original data loaded in
    self.X = cleaned, processed data with a Gender Column
    self.y = target varible created with data passed in
    self.transformer = a scikit-learn transformer used to clean and normalize the data passed in
        This can be used in a pipeline so that the cleaning step is remembered in whatever model is used
        for future predictions of data not included in the original training data.
    """

    def __init__(self, path, sep='\t', call_fit=True):
        """ Load data from file in path, will set up 'y' for unlabeled data = -1, 0 = negative, 1 = positive

        Parameters:
        ------------
        path: str,
            passed into pd.read_csv, a file with delimited data
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
        self.transformer = LoadEpimlTransformer()
        if call_fit:
            self.fit(self.data, self.y)

    def fit(self, X: pd.DataFrame=None, y: pd.Series=None):
        """
        Transform the data to clear out unwanted columns and columns that provide no information.

        cleans data and prepares it for use in epiml models
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
        self.X = self.transformer.transform(X)
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
    lc = LoadEpiml("C:\Data\\010317\membership14_final_0103.txt");