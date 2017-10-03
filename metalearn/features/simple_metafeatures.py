import time

import numpy as np

from .common_operations import *

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
