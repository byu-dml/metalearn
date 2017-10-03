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
    if len(data) > 1:
        std = np.std(data, axis = 0, ddof = 1)
    else:
        std = float('nan')    
    if math.isnan(std):
        std = 0.0
    features[label + '_sd'] = std
    return features

def is_numeric(index, attributes):    
    colType = attributes[index][1]
    return 'int' in colType or 'float' in colType

def get_numeric(data, attributes):
    counter = 0    
    for j in range(len(data[0])):
        if (is_numeric(j, attributes)):
            counter += 1
    return counter

def replace_nominal_column(col):
    le = LabelEncoder()
    le.fit(col)
    labelledCol = le.transform(col)
    labelledCol = labelledCol.reshape(labelledCol.shape[0],1)
    enc = OneHotEncoder()
    enc.fit(labelledCol)
    return enc.transform(labelledCol).toarray()

def replace_nominal(data, attributes):
    newData = np.copy(data)
    oldIndex = 0
    newIndex = 0
    while (newIndex < (len(newData[0]))):
        if (not is_numeric(oldIndex, attributes)):
            cols = replace_nominal_column(newData[:,newIndex])
            newData = np.delete(newData, newIndex, axis = 1)
            for i in range(len(cols[0])):
                newData = np.insert(newData, newIndex, cols[:,i], axis = 1)
                newIndex += 1
        else:
            newIndex += 1
        oldIndex += 1
    return newData

def get_column_of_class(data, columnIndex, label):
    return data[:,columnIndex][data[:,-1] == label]

def normalize(data):
    newData = np.copy(data)
    for j in range(newData.shape[1]):
        col = newData[:,j]
        min_v = np.amin(col)
        max_v = np.amax(col)
        d = max_v - min_v
        if (d == 0):
            d = 1.0
        newData[:,j] = (col - min_v) / d
    return newData
