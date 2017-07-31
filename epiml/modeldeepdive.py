# -*- coding: utf-8 -*-
"""
This class helps to explain models using LIME and generate some graphs of the results of a model.

Created on Wed Apr 19 17:52:18 2017
"""
import pickle
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from epiml.loadepiml import LoadEpiml
from epiml.epimlsklearn.frankenscorer import FrankenScorer
from epiml.bestmodels import generate_model_6


class ModelDeepDive():
    """
    Wraps a model and can be used to generate all tables, graphs, explanations, etc.
    Allows us to do a deep dive analysis of a model
    TODO - finish me! - extract code from:
        "RF - PNU Repeated Random Subsampling Random Searn.ipynb"
    """

    def __init__(self, clf, explainer: LimeTabularExplainer, X_test: pd.DataFrame, y_test: pd.Series):
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
        self._explanations = {}

        probas = clf.predict_proba(X_test)[:, -1]
        y_df = pd.DataFrame(y_test, columns=['y_test'])
        y_df['probas'] = probas
        y_df['probas_tens'] = (probas * 10).astype(int)
        self.y_df = y_df

    #DREAM - make the following run in parallel - hard bc LIME library has non-pickle-able classes so can't use joblib
    def generate_explanations(self, n_examples=None, num_features=30, num_samples=10000, random_state=None,
                              use_decile_samples=False):
        """ Generate explanations for n_examples
        Parameters:
        -----------------
        n_examples: int, optional, default=None
            If none generate examples for all in X_test
        num_features: int, optional, default=30
            Number of features to keep for explanations generated (each explanation may have different 30 features)
        num_samples: int, optional, default=10000
            Number of random samples generated for each local explanation that is trained on
        use_decile_samples: Boolean, optional, default=False
            If true, then look at probabilities and choose uniform samples based on buckets of 0-9%,10-19%, etc.
        """
        explanations = {}
        if use_decile_samples:
            y_samples = self.choose_decile_samples(n_examples, random_state=random_state)
            samples = self.X_test.loc[y_samples.index]
        else:
            samples = self.X_test.sample(n_examples, random_state=random_state) if n_examples is not None else self.X_test
        tot = len(samples)
        print("{:%H:%M:%S}: Generating explanations for {} samples".format(datetime.now(), tot))

        for i, (index, row_series) in enumerate(samples.iterrows(), 1):
            print("{:%H:%M:%S}: Generating model {} of {}".format(datetime.now(), i, tot))
            explanation = self.explainer.explain_instance(row_series.values, self.clf.predict_proba,
                                                        num_features=num_features, num_samples=num_samples)
            explanations[index] = explanation

        #update dictionary for more explanations
        self._explanations.update(explanations)
        print('There are now {} explanations available'.format(len(self._explanations)))
        return explanations

    def choose_decile_samples(self, n_examples, random_state=None):
        """ Sample y_df to get a (close to) uniform sample of instances across 10 buckets of probability
        """
        bincounts = np.bincount(self.y_df.probas_tens)
        bucket_weights = bincounts.sum() / bincounts
        bucket_weights_map = dict(enumerate(bucket_weights))
        sample_weights = self.y_df.probas_tens.replace(bucket_weights_map)
        return self.y_df.sample(n_examples, weights=sample_weights, random_state=random_state)

    def save_explanations_to_file(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self._explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def import_explanations_from_file(self, filename):
        with open(filename, 'rb') as handle:
            explanations = pickle.load(handle)
            print("Loaded {} explanations from {}".format(len(explanations), filename))
        self._explanations.update(explanations)
        print('There are now {} explanations available'.format(len(self._explanations)))

    def analyze_features(self, indexes=None):
        """ Analyze the features given the explanations in self._explanations.
        Parameters:
        -----------------------
        indexes: list-like, optional, default=None
            Filter all explanations by an index, this should be used to filter what we are analyzing. For example
            if you wanted to only analyze explanations where y_test was 1, or where predict_prob >= 0.9, you could
            do that with the indexes parameter. If None, then analyze all explanations
        Returns:
        ----------------------------
        rules_df: DataFrame
            Takes the rules in the explanation.as_list() and returns the added weights, added absolute value of weights,
            And the number of explanations that used that specific rule
        features_df: DataFrame
            Takes the features that are used in the rules in the explanations (as_map).  Returns the sum of the absolute
            value of the weights and the number of explanations that used that specific feature
        """
        if not self._explanations:
            raise ValueError("Called analyze_features on ModelDeepDive before explanations are calculated or initialized")
        expl = self._explanations if indexes is None else {k:v for k,v in self._explanations.items() if k in indexes}
        expl_rules = [dict(exp.as_list()) for exp in expl.values()]
        rule_weights = defaultdict(float)
        rule_abs_weights = defaultdict(float)
        rule_counts = defaultdict(int)
        for expl_rule in expl_rules:
            for f in expl_rule:
                rule_weights[f] += expl_rule[f]
                rule_abs_weights[f] += abs(expl_rule[f])
                rule_counts[f] += 1
        rules_df = pd.DataFrame({'weight_sum':rule_weights, 'abs_weight_sum':rule_abs_weights, 'N':rule_counts})
        rules_df['importance'] = np.sqrt(rules_df.abs_weight_sum)
        rules_df['importance_normal'] = rules_df.importance / rules_df.importance.sum()
        rules_df['avg_weight'] = rules_df.weight_sum / rules_df.N
        rules_df.sort_values('importance_normal', ascending=False, inplace=True)

        expl_features = [dict(exp.as_map()[1]) for exp in expl.values()]
        feature_abs_weights = defaultdict(float)
        feature_counts = defaultdict(float)
        for expl_feature in expl_features:
            for f in expl_feature:
                feature_abs_weights[f] += abs(expl_feature[f])
                feature_counts[f] += 1
        features_df = pd.DataFrame({'abs_feature_sum':feature_abs_weights, 'N':feature_counts})
        features_df['importance'] = np.sqrt(features_df.abs_feature_sum)
        features_df['importance_normal'] = features_df.importance / features_df.importance.sum()
        features_df.sort_values('importance_normal', ascending=False, inplace=True)
        return rules_df, features_df

    def analyze_subgroup(self, decile: int):
        """ Call self.analyze_features with an index filter of which decile (0-9) there is
        """
        if decile < 0 or decile > 9:
            raise ValueError("decile must be between 0 and 9")
        mask = self.y_df.probas_tens == decile
        idxs = self.y_df[mask].index
        return self.analyze_features(idxs)

    def generate_calibration_plot(self, c=1.0):
        """ Plot a calibration curve for a classifier passed in.  c = constant to divide all probabilities by if wanted.

        Note: only uses labeled data to calibrate known examples.
        """
        y_test_assumed = self.y_test.values.copy()
        labeled_mask = y_test_assumed != -1
        y_test_assumed = y_test_assumed[labeled_mask]
        X_test_assumed = self.X_test.values.copy()[labeled_mask, :]

        #y_test_assumed[y_test_assumed==-1] = 0
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_assumed,
                                                                        self.clf.predict_proba(X_test_assumed)[:,-1] / c,
                                                                        n_bins=20, normalize=True)
        fig, ax = plt.subplots()
        ax.plot(mean_predicted_value, fraction_of_positives, "s-")
        ax.set_ylabel("Fraction of positives")
        ax.set_ylim([-0.05, 1.05])
        ax.set_title('Calibration plots')
        ax.set_xlabel('Mean predicted value')
        plt.show()

    def generate_probability_plot(self):
        """ Generate a plot that shows the predicted probability with the y_test results of positive, negative, and
        unlabeled
        """
        y_prob = pd.DataFrame(self.y_df.probas)
        y_prob.columns = ['pr_one']
        y_prob['label'] = self.y_test.values
        y_prob['color'] = y_prob.label.map({-1:'b', 0:'r', 1:'g'})
        y_p = y_prob.sort_values(by='pr_one').reset_index(drop=True).reset_index()
        y_p_un = y_p[y_p.label==-1]
        y_p_pos = y_p[y_p.label==1]
        y_p_neg = y_p[y_p.label==0]
        ax = y_p_un.plot.scatter(x='index', y='pr_one', color='DarkBlue', s=1, label='Unlabeled')
        ax = y_p_pos.plot.scatter(x='index', y='pr_one', color='Green', s=400, ax=ax, label='Positive')
        ax = y_p_neg.plot.scatter(x='index', y='pr_one', color='Red', s=25, ax=ax, figsize=(20, 10),
                             xlim=(0, int(len(y_prob)*1.05)), ylim=(0, 1), label='Negative')
        ax.set_ylabel("Probability of Positive")
        ax.set_xlabel("Patient")
        ax.set_title("Positive Probability of Examples")
        plt.legend(loc="upper left")
        plt.show()

    def generate_feature_importance_plot(self, num_features=20):
        """ Generate a plot that shows the feature importance of a model graphically assuming the model has
        the 'feature_importances_' attribute

        Parameters:
        ---------------------
        num_features: int, optional, default=20
            number of features to graph, takes the top num_features in terms of importance
        """
        if not hasattr(self.clf, 'feature_importances_'):
            raise AttributeError("self.clf type {} in ModelDeepDive does not have attribute feature_importances_".
                                 format(type(self.clf)))
        fi = self.clf.feature_importances_
        fi_df = pd.DataFrame(fi, index=self.X_test.columns.values, columns=['Importance']).sort_values(by='Importance',
                             ascending=False)
        fi_df = fi_df.round(5) * 100    #convert to integers that will be between 0-100
        ax = fi_df.iloc[:num_features].iloc[::-1].plot(kind='barh',
                                                    title='Feature Importance: Top {}'.format(num_features))
        ax.legend(loc='right')
        return ax

    def generate_probability_distribution(self):
        """ Generate a plot that shows the probability distribution of the test set
        """
        probas_df = pd.DataFrame(data={'probas':self.y_df.probas}).sort_values(by='probas', ascending=False)
        probas_df['probas'] = probas_df.probas * 100
        ax = probas_df.plot.hist(bins=100, title='Probability Distribution', legend=False)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('# Patients')
        return ax

    def generate_percentile_table(self):
        """ Generate a table with a breakdown of how many patients ended up in each 5% bucket of probability
        of having EPI, along with summary stats on the makeup of each bucket
        """
        probas_df = pd.DataFrame(data={'probas':self.y_df.probas, 'y_test':self.y_test.values}).sort_values(by='probas',
                                                                                                    ascending=False)
        bins = np.linspace(0.0, 1.0, 101)
        percent = pd.cut(probas_df['probas'], bins=bins, include_lowest=True, precision=6, labels=list(range(0,100)))
        probas_df['percent'] = percent
        bins5 = np.linspace(0.0, 1.0, 21)
        percent5 = pd.cut(probas_df['probas'], right=True, bins=bins5, include_lowest=True, precision=6,
                  labels=['0%-5%','5%-10%','10%-15%','15%-20%','20%-25%','25%-30%','30%-35%','35%-40%','40%-45%',
                          '45%-50%','50%-55%','55%-60%','60%-65%','65%-70%','70%-75%','75%-80%','80%-85%','85%-90%',
                          '90%-95%','95%-100%'])
        probas_df['percent5'] = percent5
        dummies = pd.get_dummies(probas_df['y_test'], prefix='y=', prefix_sep='')
        probas_df = pd.concat([probas_df, dummies], axis=1)
        probas_group = probas_df.groupby('percent5')
        percentile_df = probas_group.aggregate({'probas':'count', 'y=-1':'sum', 'y=0':'sum', 'y=1':'sum'})
        labeled_tot = percentile_df['y=1'] + percentile_df['y=0']
        percentile_df['unlabeled_pct'] = percentile_df['y=-1'] / percentile_df.probas
        percentile_df['true_pos_pct'] = percentile_df['y=1'] / labeled_tot
        percentile_df['true_neg_pct'] = percentile_df['y=0'] / labeled_tot
        percentile_df = percentile_df.rename(columns={'y=1':'Positive', 'y=0':'Negative', 'y=-1':'Unlabeled',
                                                      'probas':'# Patients', 'unlabeled_pct':'% Unlabeled',
                                                      'true_pos_pct':'% Positive of Labeled',
                                                      'true_neg_pct':'% Negative of Labeled'})
        percentile_df = percentile_df[['Unlabeled','Positive','Negative','# Patients','% Unlabeled',
                                       '% Positive of Labeled','% Negative of Labeled']]
        percentile_df.index.name = 'Predicted Percentile Bucket'
        return percentile_df


## BEST MODELS
def create_model_6(X_train: pd.DataFrame, y_train: pd.DataFrame):
    pnu_test = generate_model_6()
    pnu_test.fit(X_train.values, y_train.values)
    return pnu_test

#DREAM - fill in feature_names list of LimeTabularExplainer for more conherant explanations
def create_explainer(X_train: pd.DataFrame, y_train: pd.DataFrame):
    return LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values,
                                                  training_labels=y_train.values,
                                                  feature_selection='lasso_path', class_names=['No EPI', 'EPI'],
                                                  discretize_continuous=True, discretizer='entropy')

#EXAMPLE RUN
if __name__ == "__main__":
    path = "C:\Data\\010317\membership14_final_0103.txt"
    print("Loading {}".format(path))
    try:
        lc = LoadEpiml(path)
    except FileNotFoundError:
        #This is for running on a machine not on network that can see the data
        print("File doesn't exist, generating fake data!")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=200, n_informative=50, n_redundant=100, n_classes=2,
                                   n_clusters_per_class=3, weights=[0.9], flip_y=0, hypercube=False,
                                   random_state=101)

        #make this like the unlabeled problem we are solving -change most to unlabeled class == -1
        rnd_unlabeled = np.random.choice([True, False], size=len(y), replace=True, p=[0.8,0.2])
        y[rnd_unlabeled] = -1
        X = pd.DataFrame(X)
        y = pd.Series(y)
    else:
        X = lc.X
        y = lc.y
    print("Done loading")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=771, stratify=y)
    print("Split Data: train_size {}, test_size {}".format(X_train.shape, X_test.shape))
    print("Create and train model")
    model6 = create_model_6(X_train, y_train)
    print("Done with model {}".format(model6))
    scores, _ = FrankenScorer()(model6, X_test.values, y_test.values)
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values,
                                     feature_selection='lasso_path', class_names=['No EPI','EPI'],
                                     discretize_continuous=True)
    deep = ModelDeepDive(model6, explainer, X_test, y_test)
    deep.generate_probability_plot()
    deep.generate_calibration_plot()
    print(scores)