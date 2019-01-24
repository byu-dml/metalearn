import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .common_operations import *

def get_entropy(col):
    return entropy(col.value_counts())

def get_class_entropy(Y_sample):
    return (get_entropy(Y_sample),)

def get_attribute_entropy(feature_array):
    entropies = [get_entropy(feature) for feature in feature_array]
    return profile_distribution(entropies)

def get_joint_entropy(feature_class_array):
    entropies = [get_entropy(translate_into_tuples(feature_class_pair[0],feature_class_pair[1])) for feature_class_pair in feature_class_array]
    return profile_distribution(entropies)

def get_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]
    return profile_distribution(mi_scores)

def get_equivalent_number_features(class_entropy, mutual_information):
    if mutual_information == 0:
        enf = np.nan
    else:
        enf = class_entropy / mutual_information
    return (enf,)

def get_noise_signal_ratio(attribute_entropy, mutual_information):
    if mutual_information == 0:
        nsr = np.nan
    else:
        nsr = (attribute_entropy - mutual_information) / mutual_information
    return (nsr,)

def translate_into_tuples(col1, col2):
    return pd.Series([x for x in zip(col1, col2)])
