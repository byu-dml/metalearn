import warnings

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .common_operations import *


'''

Compute Landmarking meta-features according to Reif et al. 2012.
The accuracy values of the following simple learners are used:
Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor,
Decision Node, Random Node.

'''

def run_pipeline(X, Y, pipeline, n_folds, cv_seed):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress sklearn warnings
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress sklearn warnings
        accuracy_scorer = make_scorer(accuracy_score)
        kappa_scorer = make_scorer(cohen_kappa_score)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        scores = cross_validate(
            pipeline, X.values, Y.values, cv=cv, n_jobs=1, scoring={
                'accuracy': accuracy_scorer, 'kappa': kappa_scorer
            }
        )
        err_rate = 1. - np.mean(scores['test_accuracy'])
        kappa = np.mean(scores['test_kappa'])
        return (err_rate, kappa)

def get_naive_bayes(X, Y, n_folds, cv_seed):
    pipeline = Pipeline([('naive_bayes', GaussianNB())])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)

def get_knn_1(X, Y, n_folds, cv_seed):
    pipeline = Pipeline([(
        'knn_1', KNeighborsClassifier(n_neighbors = 1, n_jobs=1)
    )])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)

def get_decision_stump(X, Y, seed, n_folds, cv_seed):
    pipeline = Pipeline([(
        'decision_stump', DecisionTreeClassifier(
            criterion='entropy', splitter='best', max_depth=1, random_state=seed
        )
    )])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)

def get_random_tree(X, Y, depth, seed, n_folds, cv_seed):
    pipeline = Pipeline([(
        'random_tree', DecisionTreeClassifier(
            criterion='entropy', splitter='random', max_depth=depth,
            random_state=seed
        )
    )])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)

def get_lda(X, Y, n_folds, cv_seed):
    pipeline = Pipeline([(
        'lda', LinearDiscriminantAnalysis()
    )])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)
