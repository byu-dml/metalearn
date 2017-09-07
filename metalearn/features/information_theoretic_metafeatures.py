import time
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .common_operations import *

def get_entropy(col):
    return entropy(list(Counter(col).values()))

def get_entropy_average(data, attributes, normalized = False):
    totalEnt = 0.0
    bins = 9
    length = len(data[0])
    for j in range(length):
        if (is_numeric(j, attributes)):
            try:
                col = pd.cut(data[:,j],bins)
            except:
                col = data[:,j]
            if (normalized):
                n = round(math.sqrt(len(data)))
        else:
            col = data[:,j]
            if (normalized):
                n = len(set(col))
                if (n == 1):
                    n = 2
        ent = get_entropy(col)
        if (normalized):
            ent = ent / math.log(n)
        totalEnt = totalEnt + ent
    return totalEnt / len(data[0])

def get_class_entropy(data, normalized = False):
    col = data[:,-1]
    if (normalized):
        n = len(set(col))
        if (n == 1):
            n = 2
    ent = get_entropy(col)
    if (normalized):
        ent = ent / math.log(n)
    return ent

def get_joint_entropy(data, attributes, labels):
    totalEnt = 0.0
    bins = 9
    length = len(data[0])
    for j in range(length):
        if (is_numeric(j, attributes)):
            try:
                col = pd.cut(data[:,j],bins)
            except:
                col = data[:,j]
        else:
            col = data[:,j]
        col = np.core.defchararray.add(col.astype(str), labels)
        ent = get_entropy(col)
        totalEnt = totalEnt + ent
    return totalEnt / len(data[0])

def get_mutual_information(data, attributes, labels):
    totalMutInfo = 0.0
    bins = 9
    length = len(data[0])
    for j in range(length):
        if (is_numeric(j, attributes)):
            try:
                col = pd.cut(data[:,j],bins)
            except:
                col = data[:,j]
        else:
            col = data[:,j]
        mutInfo = mutual_info_score(col, labels)
        totalMutInfo = totalMutInfo + mutInfo
    return totalMutInfo / len(data[0])

def get_equivalent_attributes_number(metafeatures):
    classEnt = metafeatures['class_entropy']
    mutInfo = metafeatures['mutual_information']
    if (classEnt == 0):
        return 0
    elif (mutInfo == 0):
        return pow(10,10)
    else:
        return classEnt / mutInfo

def get_noise_signal_ratio(metafeatures):
    mutInfo = metafeatures['mutual_information']
    attEnt = metafeatures['attribute_entropy']
    if (mutInfo == 0):
        return pow(10,10)
    else:
        return (attEnt - mutInfo) / mutInfo

def get_information_theoretic_metafeatures(attributes, data, X, Y):
    metafeatures = {}
    start_time = time.process_time()
    metafeatures['class_entropy'] = get_class_entropy(data)
    metafeatures['normalized_class_entropy'] = get_class_entropy(data, normalized = True)
    metafeatures['attribute_entropy'] = get_entropy_average(X, attributes)
    metafeatures['normalized_attribute_entropy'] = get_entropy_average(X, attributes, normalized = True)
    metafeatures['joint_entropy'] = get_joint_entropy(X, attributes, Y)
    metafeatures['mutual_information'] = get_mutual_information(X, attributes, Y)
    metafeatures['equivalent_attributes'] = get_equivalent_attributes_number(metafeatures)
    metafeatures['noise_signal_ratio'] = get_noise_signal_ratio(metafeatures)
    metafeatures['infotheo_time'] = time.process_time() - start_time
    return metafeatures
