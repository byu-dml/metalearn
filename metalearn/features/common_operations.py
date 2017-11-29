import math

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""
Compute the min, max, mean, and standard deviation of a vector

Parameters
----------
data: array of real values
label: string with which the data will be associated in the returned dictionary

Returns
-------
features = dictionary containing the min, max, mean, and standard deviation
"""
def get_min_max_mean_sd(data, label):
    features = {}
    features[label + '_min'] = np.amin(data)
    features[label + '_max'] = np.amax(data)
    features[label + '_mean'] = np.mean(data)

    ddof = 1 if len(data) > 1 else 0    
    features[label + '_sd'] = np.std(data, axis = 0, ddof = ddof)
    
    return features

def is_numeric(attribute):    
    colType = attribute[1]
    return 'int' in colType or 'float' in colType

def get_numeric(data, attributes):
    return sum(is_numeric(attr) for attr in attributes)

def replace_nominal_column(col):
    labelledCol = LabelEncoder().fit_transform(col)
    labelledCol = labelledCol.reshape(labelledCol.shape[0],1)
    return OneHotEncoder().fit_transform(labelledCol).toarray()

def replace_nominal(data, attributes):
    data_index_and_feature = list(enumerate(attributes[:len(data[0])]))
    for i, attr in reversed(data_index_and_feature):
        if (not is_numeric(attr)):
            cols = replace_nominal_column(data[:,i])
            data = np.concatenate((data[:,:i], cols, data[:,i+1:]), axis =1)
    return data

def get_column_of_class(data, columnIndex, label):
    return data[:,columnIndex][data[:,-1] == label]

def normalize(data):
    return (data - data.min(axis=0)) / data.ptp(axis=0)
