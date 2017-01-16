# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 01:53:51 2017

@author: jeffrey.gomberg
"""

import time
import warnings
import numbers
from collections import Sized, defaultdict
from functools import partial
import numpy as np

from sklearn.model_selection._validation import _index_param_value
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples, indexable
from sklearn.utils.fixes import rankdata, MaskedArray
from sklearn.exceptions import FitFailedWarning
from sklearn.externals.joblib import logger, Parallel, delayed

from sklearn.metrics.scorer import check_scoring

def _fit_and_score_with_extra_data(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split. Allows for scorers that hold more information than
    just a vanilla scorer (ie, Frankenscorer!)

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``. which returns a (dict, score) where the dict is relevant scoring data

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.

    test_score : float
        Score on test set.

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            test_score_data = {}
            if return_train_score:
                train_score = error_score
                train_score_data = {}
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_score_data, test_score = _score_no_number_check(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score_data, train_score = _score_no_number_check(estimator, X_train, y_train, scorer)

    if verbose > 2:
        msg += ", score=%s score_data=%s" % (test_score, test_score_data)
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score_data, train_score, test_score_data, test_score] if return_train_score \
        else [test_score_data, test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret

def _score_no_number_check(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set. Take out the isNumber check."""
    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, 'item'):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    return score

class JeffRandomSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters, but can return a FrankenScorer in the results.

    JeffRandomSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : callable or None, default=None
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. that returns (dict, score) where dict is extra scoring information
        If ``None``, ValueError is raised.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settins, constructed from
        param_distributions.

    """

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super(JeffRandomSearchCV, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def fit(self, X, y=None, groups=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit_base_search_cv_replacement(X, y, groups, sampled_params)


    def _fit_base_search_cv_replacement(self, X, y, groups, parameter_iterable):
            """Actual fitting,  performing the search over parameters."""

            estimator = self.estimator
            cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
            self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

            X, y, groups = indexable(X, y, groups)
            n_splits = cv.get_n_splits(X, y, groups)
            if self.verbose > 0 and isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(n_splits, n_candidates,
                                         n_candidates * n_splits))

            base_estimator = clone(self.estimator)
            pre_dispatch = self.pre_dispatch

            cv_iter = list(cv.split(X, y, groups))
            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score_with_extra_data)(clone(base_estimator), X, y, self.scorer_,
                                      train, test, self.verbose, parameters,
                                      fit_params=self.fit_params,
                                      return_train_score=self.return_train_score,
                                      return_n_test_samples=True,
                                      return_times=True, return_parameters=True,
                                      error_score=self.error_score)
              for parameters in parameter_iterable
              for train, test in cv_iter)

            # if one choose to see train score, "out" will contain train score info
            if self.return_train_score:
                (train_score_datas, train_scores, test_score_datas, test_scores, test_sample_counts,
                 fit_time, score_time, parameters) = zip(*out)
            else:
                (test_score_datas, test_scores, test_sample_counts,
                 fit_time, score_time, parameters) = zip(*out)

            candidate_params = parameters[::n_splits]
            n_candidates = len(candidate_params)

            results = dict()

            def _store_dict(key_name, array):
                array = np.array(array).reshape(n_candidates, n_splits)
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

            def _store(key_name, array, weights=None, splits=False, rank=False):
                """A small helper to store the scores/times to the cv_results_"""
                array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
                if splits:
                    for split_i in range(n_splits):
                        results["split%d_%s"
                                % (split_i, key_name)] = array[:, split_i]

                array_means = np.average(array, axis=1, weights=weights)
                results['mean_%s' % key_name] = array_means
                # Weighted std is not directly available in numpy
                array_stds = np.sqrt(np.average((array -
                                                 array_means[:, np.newaxis]) ** 2,
                                                axis=1, weights=weights))
                results['std_%s' % key_name] = array_stds

                if rank:
                    results["rank_%s" % key_name] = np.asarray(
                        rankdata(-array_means, method='min'), dtype=np.int32)

            # Computed the (weighted) mean and std for test scores alone
            # NOTE test_sample counts (weights) remain the same for all candidates
            test_sample_counts = np.array(test_sample_counts[:n_splits],
                                          dtype=np.int)

            _store('test_score', test_scores, splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            _store_dict('test_score_data', test_score_datas)
            if self.return_train_score:
                _store('train_score', train_scores, splits=True)
                _store_dict('train_score_data', train_score_datas)
            _store('fit_time', fit_time)
            _store('score_time', score_time)

            best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
            best_parameters = candidate_params[best_index]

            # Use one MaskedArray and mask all the places where the param is not
            # applicable for that candidate. Use defaultdict as each candidate may
            # not contain all the params
            param_results = defaultdict(partial(MaskedArray,
                                                np.empty(n_candidates,),
                                                mask=True,
                                                dtype=object))
            for cand_i, params in enumerate(candidate_params):
                for name, value in params.items():
                    # An all masked empty array gets created for the key
                    # `"param_%s" % name` at the first occurence of `name`.
                    # Setting the value at an index also unmasks that index
                    param_results["param_%s" % name][cand_i] = value

            results.update(param_results)

            # Store a list of param dicts at the key 'params'
            results['params'] = candidate_params

            self.cv_results_ = results
            self.best_index_ = best_index
            self.n_splits_ = n_splits

            if self.refit:
                # fit the best estimator using the entire dataset
                # clone first to work around broken estimators
                best_estimator = clone(base_estimator).set_params(
                    **best_parameters)
                if y is not None:
                    best_estimator.fit(X, y, **self.fit_params)
                else:
                    best_estimator.fit(X, **self.fit_params)
                self.best_estimator_ = best_estimator
            return self