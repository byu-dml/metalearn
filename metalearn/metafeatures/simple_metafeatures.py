import numpy as np
from pandas import DataFrame

from .common_operations import *

def get_dataset_stats(X, Y):
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]
    number_of_classes = Y.unique().shape[0]
    numeric_features = len(get_numeric_features(X))
    nominal_features = number_of_features - numeric_features
    ratio_of_numeric_features = float(numeric_features) / float(number_of_features)
    ratio_of_nominal_features = float(nominal_features) / float(number_of_features)
    return (number_of_instances, number_of_features, number_of_classes, numeric_features, nominal_features, ratio_of_numeric_features, ratio_of_nominal_features)

def get_dimensionality(number_of_features, number_of_instances):
    dimensionality = float(number_of_features) / float(number_of_instances)
    return (dimensionality,)

def get_missing_values(X):
    missing_values = X.shape[0] - X.count()
    number_missing = np.sum(missing_values)
    ratio_missing = float(number_missing) / float(X.shape[0] * X.shape[1])
    number_instances_with_missing = X.shape[1] - np.sum(missing_values == 0)
    ratio_instances_with_missing = float(number_instances_with_missing) / float(X.shape[1])
    return (number_missing, ratio_missing, number_instances_with_missing, ratio_instances_with_missing)

def get_class_stats(Y):
    classes = Y.unique()
    counts = [sum(Y == label) for label in classes]
    probs = [count/Y.shape[0] for count in counts]
    mean_class_probability, stdev_class_probability, min_class_probability, _, _, _, max_class_probability = profile_distribution(probs)
    majority_class_size = max(counts)
    minority_class_size = min(counts)
    return (mean_class_probability, stdev_class_probability, min_class_probability, max_class_probability, minority_class_size, majority_class_size)

def get_nominal_cardinalities(X):
    cardinalities = [X[feature].unique().shape[0] for feature in X.columns if not dtype_is_numeric(X[feature].dtype)]
    mean_cardinality_of_nominal_features, stdev_cardinality_of_nominal_features, min_cardinality_of_nominal_features, _, _, _, max_cardinality_of_nominal_features = profile_distribution(cardinalities)
    return (mean_cardinality_of_nominal_features, stdev_cardinality_of_nominal_features, min_cardinality_of_nominal_features, max_cardinality_of_nominal_features)

def get_numeric_cardinalities(X):
    cardinalities = [X[feature].unique().shape[0] for feature in X.columns if dtype_is_numeric(X[feature].dtype)]
    mean_cardinality_of_numeric_features, stdev_cardinality_of_numeric_features, min_cardinality_of_numeric_features, _, _, _, max_cardinality_of_numeric_features = profile_distribution(cardinalities)
    return (mean_cardinality_of_numeric_features, stdev_cardinality_of_numeric_features, min_cardinality_of_numeric_features, max_cardinality_of_numeric_features)
