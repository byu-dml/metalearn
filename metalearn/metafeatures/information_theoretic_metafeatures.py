import time
import math
from collections import Counter

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .metafeatures_base import MetafeaturesBase

class InformationTheoreticMetafeatures(MetafeaturesBase):

    def __init__(self):
        
        function_dict = {
            'ClassEntropy': self._get_class_entropy,            
            'MeanAttributeEntropy': self._get_attribute_entropy,            
            'MinAttributeEntropy': self._get_attribute_entropy,            
            'MaxAttributeEntropy': self._get_attribute_entropy,            
            'Quartile1AttributeEntropy': self._get_attribute_entropy,            
            'Quartile2AttributeEntropy': self._get_attribute_entropy,            
            'Quartile3AttributeEntropy': self._get_attribute_entropy,            
            'MeanJointEntropy': self._get_joint_entropy,            
            'MinJointEntropy': self._get_joint_entropy,            
            'MaxJointEntropy': self._get_joint_entropy,            
            'Quartile1JointEntropy': self._get_joint_entropy,            
            'Quartile2JointEntropy': self._get_joint_entropy,            
            'Quartile3JointEntropy': self._get_joint_entropy,               
            'MeanMutualInformation': self._get_mutual_information,            
            'MinMutualInformation': self._get_mutual_information,            
            'MaxMutualInformation': self._get_mutual_information,            
            'Quartile1MutualInformation': self._get_mutual_information,            
            'Quartile2MutualInformation': self._get_mutual_information,            
            'Quartile3MutualInformation': self._get_mutual_information,            
            'EquivalentNumberOfFeatures': self._get_equivalent_number_features,
            'NoiseToSignalRatio': self._get_noise_signal_ratio
        }

        dependencies_dict = {
            'ClassEntropy': [],            
            'MeanAttributeEntropy': [],            
            'MinAttributeEntropy': [],
            'MaxAttributeEntropy': [],
            'Quartile1AttributeEntropy': [],
            'Quartile2AttributeEntropy': [],
            'Quartile3AttributeEntropy': [],
            'MeanJointEntropy': [],            
            'MinJointEntropy': [],
            'MaxJointEntropy': [],
            'Quartile1JointEntropy': [],
            'Quartile2JointEntropy': [],
            'Quartile3JointEntropy': [],                                    
            'MeanMutualInformation': [],            
            'MinMutualInformation': [],
            'MaxMutualInformation': [],
            'Quartile1MutualInformation': [],
            'Quartile2MutualInformation': [],
            'Quartile3MutualInformation': [],
            'EquivalentNumberOfFeatures': ['ClassEntropy','MeanMutualInformation'],
            'NoiseToSignalRatio': ['MeanAttributeEntropy','MeanMutualInformation']
        }                

        super().__init__(function_dict, dependencies_dict)

    def _get_entropy(self, col):
        return entropy(list(Counter(col).values()))

    def _get_attribute_entropy(self, X, Y):
        entropies = []
        
        bins = round(math.sqrt(X.shape[0]))
        numeric_features = self._get_numeric_features(X)    
        for feature in numeric_features:
            col = X[feature].as_matrix()
            try:
                col = pd.cut(col,bins) 
            except:
                col = X[feature].as_matrix()           
            ent = self._get_entropy(col)            
            entropies.append(ent)

        nominal_features = self._get_nominal_features(X)    
        for feature in nominal_features:
            col = X[feature].as_matrix()        
            ent = self._get_entropy(col)            
            entropies.append(ent)
        
        values_dict = self._profile_distribution(entropies, 'AttributeEntropy')
        return {
            'MeanAttributeEntropy': values_dict['MeanAttributeEntropy'],            
            'MinAttributeEntropy': values_dict['MinAttributeEntropy'],            
            'MaxAttributeEntropy': values_dict['MaxAttributeEntropy'],            
            'Quartile1AttributeEntropy': values_dict['Quartile1AttributeEntropy'],            
            'Quartile2AttributeEntropy': values_dict['Quartile2AttributeEntropy'],            
            'Quartile3AttributeEntropy': values_dict['Quartile3AttributeEntropy']
        }

    def _get_class_entropy(self, X, Y):            
        ent = self._get_entropy(Y.as_matrix())        
        return { 'ClassEntropy': ent }

    def _get_joint_entropy(self, X, Y):
        entropies = []
        labels = Y.as_matrix()
        bins = round(math.sqrt(X.shape[0]))
        numeric_features = self._get_numeric_features(X)
        for feature in X.columns:
            col = X[feature].as_matrix()
            if feature in numeric_features:
                col = pd.cut(col,bins)    
            col = np.core.defchararray.add(col.astype(str), labels.astype(str))
            ent = self._get_entropy(col)
            entropies.append(ent)
        values_dict = self._profile_distribution(entropies, 'JointEntropy')
        return {
            'MeanJointEntropy': values_dict['MeanJointEntropy'],            
            'MinJointEntropy': values_dict['MinJointEntropy'],            
            'MaxJointEntropy': values_dict['MaxJointEntropy'],            
            'Quartile1JointEntropy': values_dict['Quartile1JointEntropy'],            
            'Quartile2JointEntropy': values_dict['Quartile2JointEntropy'],            
            'Quartile3JointEntropy': values_dict['Quartile3JointEntropy']
        }

    def _get_mutual_information(self, X, Y):
        mutual_information_scores = []
        labels = Y.as_matrix()
        bins = round(math.sqrt(X.shape[0]))
        numeric_features = self._get_numeric_features(X)
        for feature in X.columns:
            col = X[feature].as_matrix()
            if feature in numeric_features:
                col = pd.cut(col,bins)    
            mutInfo = mutual_info_score(col, labels)
            mutual_information_scores.append(mutInfo)            
        values_dict = self._profile_distribution(mutual_information_scores, 'MutualInformation')
        return {
            'MeanMutualInformation': values_dict['MeanMutualInformation'],            
            'MinMutualInformation': values_dict['MinMutualInformation'],            
            'MaxMutualInformation': values_dict['MaxMutualInformation'],            
            'Quartile1MutualInformation': values_dict['Quartile1MutualInformation'],            
            'Quartile2MutualInformation': values_dict['Quartile2MutualInformation'],            
            'Quartile3MutualInformation': values_dict['Quartile3MutualInformation']
        }

    def _get_equivalent_number_features(self, X, Y, class_entropy, mutual_information):    
        if (class_entropy == 0):
            enf = 0
        elif (mutual_information == 0):
            enf = np.nan
        else:
            enf = class_entropy / mutual_information
        return { 'EquivalentNumberOfFeatures': enf }

    def _get_noise_signal_ratio(self, X, Y, attribute_entropy, mutual_information):    
        if (mutual_information == 0):
            nsr = np.nan
        else:
            nsr = (attribute_entropy - mutual_information) / mutual_information
        return { 'NoiseToSignalRatio': nsr }    
