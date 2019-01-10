import numpy as np
from pandas import DataFrame

from .common_operations import *

def get_dataset_stats(X, column_types):
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]
    numeric_features = len(get_numeric_features(X, column_types))
    categorical_features = number_of_features - numeric_features
    ratio_of_numeric_features = numeric_features / number_of_features
    ratio_of_categorical_features = categorical_features / number_of_features
    return (number_of_instances, number_of_features, numeric_features, categorical_features, ratio_of_numeric_features, ratio_of_categorical_features)

def get_dimensionality(number_of_features, number_of_instances):
    dimensionality = number_of_features / number_of_instances
    return (dimensionality,)

def get_missing_values(X):
    missing_values_by_instance = X.shape[1] - X.count(axis=1)
    missing_values_by_feature = X.shape[0] - X.count(axis=0)
    number_missing = int(np.sum(missing_values_by_instance)) # int for json compatibility
    ratio_missing = number_missing / (X.shape[0] * X.shape[1])
    number_instances_with_missing = int(np.sum(missing_values_by_instance != 0)) # int for json compatibility
    ratio_instances_with_missing = number_instances_with_missing / X.shape[0]
    number_features_with_missing = int(np.sum(missing_values_by_feature != 0))
    ratio_features_with_missing = number_features_with_missing / X.shape[1]
    return (
        number_missing, ratio_missing, number_instances_with_missing,
        ratio_instances_with_missing, number_features_with_missing,
        ratio_features_with_missing
    )

def get_class_stats(Y):
    classes = Y.unique()
    number_of_classes = classes.shape[0]
    counts = [sum(Y == label) for label in classes]
    probs = [count/Y.shape[0] for count in counts]
    majority_class_size = max(counts)
    minority_class_size = min(counts)
    return (number_of_classes, *profile_distribution(probs), minority_class_size, majority_class_size)

def get_categorical_cardinalities(X, column_types):
    cardinalities = [X[feature].unique().shape[0] for feature in get_categorical_features(X, column_types)]
    return profile_distribution(cardinalities)

def get_numeric_cardinalities(X, column_types):
    cardinalities = [X[feature].unique().shape[0] for feature in get_numeric_features(X, column_types)]
    return profile_distribution(cardinalities)
