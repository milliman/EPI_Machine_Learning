"""
This will load data and provide some summary statistics to the dataset loaded in
"""

import pandas as pd

class LoadCreon:
    """Manage loading a Creon summarized dataset tab delimited into data"""

    def __init__(self, path):
        """Path of the file to load"""
        data = pd.read_csv(path, sep='\t', low_memory=False)
        X = data.copy()
        # Binar-i-tize the Gender column to 1 or 0
        X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
        # drop all useless columns
        X_sums = X.sum(numeric_only=True)
        X = X.drop(list(X_sums[X_sums == 0].index), axis=1)
        X = X.drop(['unlabel_flag','true_pos_flag','true_neg_flag','MemberID','epi_related_cond',
                          'epi_related_cond_subgrp','h_rank','pert_flag','mmos','elastase_flag','medical_claim_count',
                          'rx_claim_count'], axis=1)
        # set up class from the flags in the data with
        # -1 = unlabeled, 0 = true_negative, 1 = true_positive
        y = (data.unlabel_flag * -1) + data.true_pos_flag
        self.data = data
        self.X = X
        self.y = y



if __name__ == "__main__":
    lc = LoadCreon("C:\Data\\010317\membership14_final_0103.txt");