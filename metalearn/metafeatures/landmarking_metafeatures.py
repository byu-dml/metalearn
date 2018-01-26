import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .metafeatures_base import MetafeaturesBase

'''

Compute Landmarking meta-features according to Reif et al. 2012.
The accuracy values of the following simple learners are used: 
Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor, 
Decision Node, Random Node.

'''

class LandmarkingMetafeatures(MetafeaturesBase):

    def __init__(self):
        
        function_dict = {
            'NaiveBayesErrRate': self._get_naive_bayes,
            'NaiveBayesKappa': self._get_naive_bayes,
            'kNN1NErrRate': self._get_knn_1,
            'kNN1NKappa': self._get_knn_1,
            'DecisionStumpErrRate': self._get_decision_stump,
            'DecisionStumpKappa': self._get_decision_stump,
            'RandomNodeErrRate': self._get_random_node,
            'RandomNodeKappa': self._get_random_node,
            'LinearDiscriminantAnalysisErrRate': self._get_lda,
            'LinearDiscriminantAnalysisKappa': self._get_lda
        }

        dependencies_dict = {
            'NaiveBayesErrRate': [],
            'NaiveBayesKappa': [],
            'kNN1NErrRate': [],
            'kNN1NKappa': [],
            'DecisionStumpErrRate': [],
            'DecisionStumpKappa': [],
            'RandomNodeErrRate': [],
            'RandomNodeKappa': [],
            'LinearDiscriminantAnalysisErrRate': [],
            'LinearDiscriminantAnalysisKappa': []
        }

        super().__init__(function_dict, dependencies_dict)

    def compute(self, dataframe: DataFrame, metafeatures: list = None) -> DataFrame:
        dataframe = dataframe.dropna(axis=1, how="all")
        X = dataframe.drop(self.target_name, axis=1)        
        X = self._preprocess_data(X)        
        Y = dataframe[self.target_name]
        if metafeatures is None:
            metafeatures = self.list_metafeatures()
        return self._retrieve_metafeatures(metafeatures, X, Y)        

    def _run_pipeline(self, X, Y, estimator, label):        
        pipe = Pipeline([('classifiers', estimator)])
        accuracy_scorer = make_scorer(accuracy_score)
        kappa_scorer = make_scorer(cohen_kappa_score)
        scores = cross_validate(pipe, X.as_matrix(), Y.as_matrix(), 
            cv=2, n_jobs=-1, scoring={'accuracy': accuracy_scorer, 'kappa': kappa_scorer})
        err_rate = 1. - np.mean(scores['test_accuracy'])
        kappa = np.mean(scores['test_kappa'])

        return {
            label + 'ErrRate': err_rate,
            label + 'Kappa': kappa
        }

    def _get_naive_bayes(self, X, Y):
        values_dict = self._run_pipeline(X, Y, GaussianNB(), 'NaiveBayes')
        return {
            'NaiveBayesErrRate': values_dict['NaiveBayesErrRate'],
            'NaiveBayesKappa': values_dict['NaiveBayesKappa']
        }

    def _get_knn_1(self, X, Y):
        values_dict = self._run_pipeline(X, Y, KNeighborsClassifier(n_neighbors = 1), 'kNN1N')
        return {
            'kNN1NErrRate': values_dict['kNN1NErrRate'],
            'kNN1NKappa': values_dict['kNN1NKappa']
        }

    def _get_decision_stump(self, X, Y):
        values_dict = self._run_pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1, random_state=0), 'DecisionStump')
        return {
            'DecisionStumpErrRate': values_dict['DecisionStumpErrRate'],
            'DecisionStumpKappa': values_dict['DecisionStumpKappa']
        }

    def _get_random_node(self, X, Y):
        values_dict = self._run_pipeline(X, Y, DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=1, random_state=0), 'RandomNode')
        return {
            'RandomNodeErrRate': values_dict['RandomNodeErrRate'],
            'RandomNodeKappa': values_dict['RandomNodeKappa']
        }    

    def _get_lda(self, X, Y):
        values_dict = self._run_pipeline(X, Y, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), 'LinearDiscriminantAnalysis')
        return {
            'LinearDiscriminantAnalysisErrRate': values_dict['LinearDiscriminantAnalysisErrRate'],
            'LinearDiscriminantAnalysisKappa': values_dict['LinearDiscriminantAnalysisKappa']
        }    
