import time
import numpy as np

from .metafeatures import Metafeature
from .common_operations import *

class SimpleMetafeatures(Metafeature):

    def __init__(self):
        pass

    def compute(self, X: list, Y: list, attributes: list) -> list:        
        data = np.append(X, Y.reshape(Y.shape[0], -1), axis = 1)
        data = data[(data != np.array(None)).all(axis=1)]
        return get_simple_metafeatures(attributes, data, Y)

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
        probs = []
        features = {}
        for label in classes:
            count = np.count_nonzero(Y == label)
            probs.append((count)/len(Y))
        features.update(get_min_max_mean_sd(probs, 'class_prob'))
        return features

    def get_simple_metafeatures(attributes, data, Y):
        metafeatures = {}
        start_time = time.process_time()    
        metafeatures['classes'] = len(set(attributes[-1][1]))    
        metafeatures['attributes'] = len(attributes) - 1    
        metafeatures['numeric'] = get_numeric(data, attributes)    
        metafeatures['nominal'] = metafeatures['attributes'] - metafeatures['numeric']    
        metafeatures['samples'] = len(Y)    
        metafeatures['dimensionality'] = metafeatures['attributes'] / metafeatures['samples']    
        metafeatures['numeric_rate'] = metafeatures['numeric'] / metafeatures['attributes']    
        metafeatures['nominal_rate'] = metafeatures['nominal'] / metafeatures['attributes']    
        metafeatures.update(get_symbol_stats(attributes))    
        metafeatures.update(get_class_stats(Y))    
        metafeatures['simple_time'] = time.process_time() - start_time
        return metafeatures
