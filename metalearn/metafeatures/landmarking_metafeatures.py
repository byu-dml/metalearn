import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .common_operations import *


import warnings
warnings.filterwarnings("ignore", category=UserWarning) # suppress sklearn warnings

'''

Compute Landmarking meta-features according to Reif et al. 2012.
The accuracy values of the following simple learners are used:
Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor,
Decision Node, Random Node.

'''

def run_pipeline(X, Y, estimator, label):
    pipe = Pipeline([('classifiers', estimator)])
    accuracy_scorer = make_scorer(accuracy_score)
    kappa_scorer = make_scorer(cohen_kappa_score)
    scores = cross_validate(pipe, X.as_matrix(), Y.as_matrix(),
        cv=2, n_jobs=-1, scoring={'accuracy': accuracy_scorer, 'kappa': kappa_scorer})
    err_rate = 1. - np.mean(scores['test_accuracy'])
    kappa = np.mean(scores['test_kappa'])

    return (err_rate, kappa)

def get_naive_bayes(X, Y):
    return run_pipeline(X, Y, GaussianNB(), 'NaiveBayes')

def get_knn_1(X, Y):
    return run_pipeline(X, Y, KNeighborsClassifier(n_neighbors = 1), 'kNN1N')

def get_decision_stump(X, Y):
    return run_pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1, random_state=0), 'DecisionStump')

def get_random_tree(X, Y, depth):
    return run_pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=depth, random_state=0), 'RandomNode')

def get_lda(X, Y):
    return run_pipeline(X, Y, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), 'LinearDiscriminantAnalysis')
