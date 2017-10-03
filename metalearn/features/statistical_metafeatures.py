import time
import math

import numpy as np
from scipy.stats import skew, kurtosis

from .common_operations import *
from .rcca import CCA

def get_skewness(data, attributes, preprocessed = False): 
    # suppress errors brought about by code in the scipy skew function    
    np.seterr(divide='ignore', invalid='ignore')
    classes = attributes[-1][1]    
    skw = 0.0
    for label in classes:
        s = 0.0
        n = 0.0        
        for j in range(len(data[0]) - 1):            
            if (preprocessed or is_numeric(j, attributes)):                
                values_of_feature_for_class = get_column_of_class(data, j, label)                                                
                v = skew(values_of_feature_for_class)                                
                if ((v != None) and (not math.isnan(v))):                    
                    s += abs(v)
                    n += 1
        if (n > 0):            
            skw += (s / n)                
    return skw / len(classes)

def get_kurtosis(data, attributes, preprocessed = False):
    classes = attributes[-1][1]
    kurt = 0.0
    for label in classes:
        s = 0.0
        n = 0.0
        for j in range(len(data[0]) - 1):
            if (preprocessed or is_numeric(j, attributes)):
                values_of_feature_for_class = get_column_of_class(data, j, label)
                if (len(set(values_of_feature_for_class)) == 1):
                    v = 0
                else:
                    v = kurtosis(values_of_feature_for_class, fisher = False)
                if ((v != None) and (not math.isnan(v))):
                    s += v
                    n += 1
        if (n > 0):
            kurt += (s / n)
    return (kurt / len(classes))

def get_abs_cor(data, attributes):
    numAtt = len(data[0]) - 1
    if (numAtt > 1):
        classes = attributes[-1][1]
        sums = 0.0
        n = 0.0
        for label in classes:            
            for i in range(numAtt):
                col_i_data_by_class = get_column_of_class(data, i, label)
                if (not is_numeric(i, attributes)):
                    col_i_data_by_class = replace_nominal_column(col_i_data_by_class)
                else:
                    col_i_data_by_class = col_i_data_by_class.reshape(col_i_data_by_class.shape[0], 1)
                for j in range(numAtt):
                    col_j_data_by_class = get_column_of_class(data, j, label)
                    if (not is_numeric(j, attributes)):
                        col_j_data_by_class = replace_nominal_column(col_j_data_by_class)
                    else:
                        col_j_data_by_class = col_j_data_by_class.reshape(col_j_data_by_class.shape[0], 1)
                    cca = CCA(kernelcca = False, reg = 0., numCC = 1, verbose=False)
                    try:                        
                        cca.train([col_i_data_by_class.astype(float), col_j_data_by_class.astype(float)])                        
                        c = cca.cancorrs[0]
                    except:
                        continue
                    if (c):
                        sums += abs(c)
                        n += 1            
        if (n != 0):
            return sums / n
    return 0.0

def get_cancor(data, attributes, n):
    cancor = {}
    c = get_cancors(data, attributes)[0:n]
    for i in range(len(c)):
        cancor['cancor_' + str(i + 1)] = c[i]
    return cancor

def get_cancors(data, attributes):
    att_data = data[:,0:-1]
    preprocess_att_data = replace_nominal(att_data, attributes)
    labels = data[:,-1]
    preprocess_labels = replace_nominal_column(labels)
    cca = CCA(kernelcca = False, reg = 0., numCC = 1)
    try:
        cca.train([preprocess_att_data.astype(float), preprocess_labels.astype(float)])
        return cca.cancorrs
    except:
        return [0., 0.]

def get_statistical_metafeatures(attributes, data, data_preprocessed):
    metafeatures = {}
    start_time = time.process_time()    
    metafeatures['skewness'] = get_skewness(data, attributes)    
    metafeatures['skewness_prep'] = get_skewness(data_preprocessed, attributes, preprocessed = True)    
    metafeatures['kurtosis'] = get_kurtosis(data, attributes)    
    metafeatures['kurtosis_prep'] = get_kurtosis(data_preprocessed, attributes, preprocessed = True)    
    metafeatures['abs_cor'] = get_abs_cor(data, attributes)    
    metafeatures.update(get_cancor(data, attributes, 1))    
    metafeatures['statistical_time'] = time.process_time() - start_time
    return metafeatures
