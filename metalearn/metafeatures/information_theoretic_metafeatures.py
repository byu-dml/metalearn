import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .common_operations import *

def get_entropy(col):
    return entropy(col.value_counts())

def get_class_entropy(Y_sample):
    return (get_entropy(Y_sample),)

def get_attribute_entropy(feature_class_array):
    entropies = [get_entropy(feature_class_pair[0]) for feature_class_pair in feature_class_array]
    mean_attribute_entropy, _, min_attribute_entropy, quartile1_attribute_entropy, quartile2_attribute_entropy, quartile3_attribute_entropy, max_attribute_entropy = profile_distribution(entropies)
    return (mean_attribute_entropy, min_attribute_entropy, quartile1_attribute_entropy, quartile2_attribute_entropy, quartile3_attribute_entropy, max_attribute_entropy)

def get_joint_entropy(feature_class_array):
    entropies = [get_entropy(feature_class_pair[0].astype(str) + feature_class_pair[1].astype(str)) for feature_class_pair in feature_class_array]
    mean_joint_entropy, _, min_joint_entropy, quartile1_joint_entropy, quartile2_joint_entropy, quartile3_joint_entropy, max_joint_entropy = profile_distribution(entropies)
    return (mean_joint_entropy, min_joint_entropy, quartile1_joint_entropy, quartile2_joint_entropy, quartile3_joint_entropy, max_joint_entropy)

def get_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]
    mean_mutual_information, _, min_mutual_information, quartile1_mutual_information, quartile2_mutual_information, quartile3_mutual_information, max_mutual_information = profile_distribution(mi_scores)
    return (mean_mutual_information, min_mutual_information, quartile1_mutual_information, quartile2_mutual_information, quartile3_mutual_information, max_mutual_information)

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
