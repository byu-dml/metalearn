import numpy as np

from metalearn.metafeatures.common_operations import *
from metalearn.metafeatures.base import build_resources_info, MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup


def get_dataset_stats(X, column_types):
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]
    numeric_features = len(get_numeric_features(X, column_types))
    categorical_features = number_of_features - numeric_features
    ratio_of_numeric_features = numeric_features / number_of_features
    ratio_of_categorical_features = categorical_features / number_of_features
    return (number_of_instances, number_of_features, numeric_features, categorical_features, ratio_of_numeric_features, ratio_of_categorical_features)

get_dataset_stats = MetafeatureComputer(
    get_dataset_stats,
    [
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfNumericFeatures",
        "NumberOfCategoricalFeatures",
        "RatioOfNumericFeatures",
        "RatioOfCategoricalFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.SIMPLE],
    { "X": "X_raw" }
)


def get_dimensionality(number_of_features, number_of_instances):
    dimensionality = number_of_features / number_of_instances
    return (dimensionality,)

get_dimensionality = MetafeatureComputer(
    get_dimensionality,
    ["Dimensionality"],
    ProblemType.ANY,
    [MetafeatureGroup.SIMPLE],
    {
        "number_of_features": "NumberOfFeatures",
        "number_of_instances": "NumberOfInstances"
    }
)


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

get_missing_values = MetafeatureComputer(
    get_missing_values,
    [
        "NumberOfMissingValues",
        "RatioOfMissingValues",
        "NumberOfInstancesWithMissingValues",
        "RatioOfInstancesWithMissingValues",
        "NumberOfFeaturesWithMissingValues",
        "RatioOfFeaturesWithMissingValues"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.SIMPLE],
    { "X": "X_raw" }
)


def get_class_stats(Y):
    classes = Y.unique()
    number_of_classes = classes.shape[0]
    counts = [sum(Y == label) for label in classes]
    probs = [count/Y.shape[0] for count in counts]
    majority_class_size = max(counts)
    minority_class_size = min(counts)
    return (number_of_classes, *profile_distribution(probs), minority_class_size, majority_class_size)

get_class_stats = MetafeatureComputer(
    computer=get_class_stats,
    returns=[
        "NumberOfClasses",
        "MeanClassProbability",
        "StdevClassProbability",
        "SkewClassProbability",
        "KurtosisClassProbability",
        "MinClassProbability",
        "Quartile1ClassProbability",
        "Quartile2ClassProbability",
        "Quartile3ClassProbability",
        "MaxClassProbability",
        "MinorityClassSize",
        "MajorityClassSize"
    ],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[
        MetafeatureGroup.SIMPLE,
        MetafeatureGroup.TARGET_DEPENDENT
    ]
)


def get_categorical_cardinalities(X, column_types):
    cardinalities = [X[feature].unique().shape[0] for feature in get_categorical_features(X, column_types)]
    return profile_distribution(cardinalities)

get_categorical_cardinalities = MetafeatureComputer(
    get_categorical_cardinalities,
    [
        "MeanCardinalityOfCategoricalFeatures",
        "StdevCardinalityOfCategoricalFeatures",
        "SkewCardinalityOfCategoricalFeatures",
        "KurtosisCardinalityOfCategoricalFeatures",
        "MinCardinalityOfCategoricalFeatures",
        "Quartile1CardinalityOfCategoricalFeatures",
        "Quartile2CardinalityOfCategoricalFeatures",
        "Quartile3CardinalityOfCategoricalFeatures",
        "MaxCardinalityOfCategoricalFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.SIMPLE],
)


def get_numeric_cardinalities(X, column_types):
    cardinalities = [X[feature].unique().shape[0] for feature in get_numeric_features(X, column_types)]
    return profile_distribution(cardinalities)

get_numeric_cardinalities = MetafeatureComputer(
    get_numeric_cardinalities,
    [
        "MeanCardinalityOfNumericFeatures",
        "StdevCardinalityOfNumericFeatures",
        "SkewCardinalityOfNumericFeatures",
        "KurtosisCardinalityOfNumericFeatures",
        "MinCardinalityOfNumericFeatures",
        "Quartile1CardinalityOfNumericFeatures",
        "Quartile2CardinalityOfNumericFeatures",
        "Quartile3CardinalityOfNumericFeatures",
        "MaxCardinalityOfNumericFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.SIMPLE]
)


"""
A list of all MetafeatureComputer
instances in this module.
"""
metafeatures_info = build_resources_info(
    get_dataset_stats,
    get_class_stats,
    get_dimensionality,
    get_missing_values,
    get_categorical_cardinalities,
    get_numeric_cardinalities
)
