import time
import numpy as np

from .metafeatures_base import MetafeaturesBase

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

'''

Compute Landmarking meta-features according to Reif et al. 2012.
The accuracy values of the following simple learners are used: 
Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor, 
Decision Node, Random Node.

'''

class LandmarkingMetafeatures(MetafeaturesBase):

    def __init__(self):
        pass

    def compute(self, X: list, Y: list, attributes: list) -> list:        
        data = np.append(X, Y.reshape(Y.shape[0], -1), axis = 1)
        data = data[(data != np.array(None)).all(axis=1)]
        return get_landmarking_metafeatures(attributes, data, X, Y)

def pipeline(X, Y, estimator):
    pipe = Pipeline([('Imputer', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                     ('classifiers', estimator)])
    score = np.mean(cross_val_score(pipe, X, Y, cv=10, n_jobs=-1))
    return score

def get_landmarking_metafeatures(attributes, data, X, Y):
    metafeatures = {}
    start_time = time.process_time()
    metafeatures['one_nearest_neighbor'] = pipeline(X, Y, KNeighborsClassifier(n_neighbors = 1)) 
    metafeatures['linear_discriminant_analysis'] = pipeline(X, Y, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')) 
    metafeatures['naive_bayes'] = pipeline(X, Y, GaussianNB()) 
    metafeatures['decision_node'] = pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='best', 
                                                                         max_depth=1, random_state=0)) 
    metafeatures['random_node'] = pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='random',
                                                                       max_depth=1, random_state=0))
    metafeatures['landmark_time'] = time.process_time() - start_time
    return metafeatures
