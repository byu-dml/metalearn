import warnings
import operator
import numpy as np
from pandas import DataFrame
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, check_cv
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import indexable

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
        results = new_cross_val(pipeline, X.values, Y.values, cv, 1)
        return (results[0], results[1])

def new_cross_val(pipeline, X, y, cv, n_jobs):
    X, y = indexable(X, y)
    cv = check_cv(cv, y, classifier=is_classifier(pipeline))
    accuracy = []
    kappa = []
    for train, test in cv.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        kappa.append(cohen_kappa_score(y_test, y_pred))
    err_rate = 1. - np.mean(accuracy)
    kappa = np.mean(kappa)
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
        'lda', LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto'
        )
    )])
    return run_pipeline(X, Y, pipeline, n_folds, cv_seed)
