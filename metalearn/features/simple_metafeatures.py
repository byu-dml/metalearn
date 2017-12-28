import time
import numpy as np

from .metafeatures_base import MetafeaturesBase
from .common_operations import get_numeric, get_min_max_mean_sd


class SimpleMetafeatures(MetafeaturesBase):

    def __init__(self):
        pass

    def compute(self, X: list, Y: list, attributes: list) -> list:        
        data = np.append(X, Y.reshape(Y.shape[0], -1), axis = 1)
        data = data[(data != np.array(None)).all(axis=1)]
        return get_simple_metafeatures(attributes, data, Y)

'''
Helper Methods to eventually be split and/or incorporated in the class
'''

def get_symbol_stats(attributes):
    symbols = []
    features = {}
    for attribute in attributes:
        if ((not 'int' in attribute[1]) and (not 'float' in attribute[1])):        
            symbols.append(len(attribute[1]))    
    features.update(get_min_max_mean_sd(symbols, 'symbols'))    
    features['symbols_sum'] = np.sum(symbols)
    return features

def get_class_stats(Y):
    classes = set(Y)
    counts = [sum(Y == label) for label in classes]
    probs = [count/len(Y) for count in counts]
    
    features = {}
    features.update(get_min_max_mean_sd(probs, 'class_prob'))
    features["default_accuracy"] = max(probs)
    features["majority_class_size"] = max(counts)
    features["minority_class_percentage"] = min(probs)
    features["minority_class_size"] = min(counts)
    return features

def get_simple_metafeatures(attributes, data, Y):
    metafeatures = {}
    start_time = time.process_time()    
    
    #Simple metafeatures about dataset
    metafeatures['number_of_classes'] = len(set(attributes[-1][1])) 
    metafeatures['number_of_instances'] = len(Y)       
    metafeatures['number_of_features'] = len(attributes) - 1   
    metafeatures['dimensionality'] = metafeatures['number_of_features'] / metafeatures['number_of_instances']     
    
    #Simple metafeatures about features
    metafeatures['number_of_numeric_features'] = get_numeric(data, attributes)  
    metafeatures['percentage_of_numeric_features'] = metafeatures['number_of_numeric_features'] / metafeatures['number_of_features']
    metafeatures['number_of_nominal_features'] = metafeatures['number_of_features'] - metafeatures['number_of_numeric_features'] 
    metafeatures['percentage_of_nominal_features'] = metafeatures['number_of_nominal_features'] / metafeatures['number_of_features']
    
    # Missing Values
    # TODO: Determine how missing values are flagged
    # metafeatures['number_of_missing values'] = 
    # metafeatures['number_of_instances_with_missing values'] = 
    # metafeatures['percentage_of_instances_with_missing values'] = metafeatures['number_of_instances_with_missing values'] / metafeatures['number_of_instances']
    
    metafeatures.update(get_symbol_stats(attributes))    
    metafeatures.update(get_class_stats(Y))    
    metafeatures['simple_time'] = time.process_time() - start_time
    return metafeatures