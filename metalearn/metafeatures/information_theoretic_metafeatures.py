import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from metalearn.metafeatures.common_operations import profile_distribution
from metalearn.metafeatures.base import build_resources_info, MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup


def get_entropy(col):
    return entropy(col.value_counts())

def get_class_entropy(Y_sample):
    return (get_entropy(Y_sample),)

get_class_entropy = MetafeatureComputer(
    get_class_entropy,
    ["ClassEntropy"],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "Y_sample": "YSample"
    }
)


def get_attribute_entropy(feature_array):
    entropies = [get_entropy(feature) for feature in feature_array]
    return profile_distribution(entropies)

get_categorical_attribute_entropy = MetafeatureComputer(
    get_attribute_entropy,
    [
        "MeanCategoricalAttributeEntropy",
        "StdevCategoricalAttributeEntropy",
        "SkewCategoricalAttributeEntropy",
        "KurtosisCategoricalAttributeEntropy",
        "MinCategoricalAttributeEntropy",
        "Quartile1CategoricalAttributeEntropy",
        "Quartile2CategoricalAttributeEntropy",
        "Quartile3CategoricalAttributeEntropy",
        "MaxCategoricalAttributeEntropy"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.INFO_THEORETIC],
    {
        "feature_array": "NoNaNCategoricalFeatures"
    }
)

get_numeric_attribute_entropy = MetafeatureComputer(
    get_attribute_entropy,
    [
        "MeanNumericAttributeEntropy",
        "StdevNumericAttributeEntropy",
        "SkewNumericAttributeEntropy",
        "KurtosisNumericAttributeEntropy",
        "MinNumericAttributeEntropy",
        "Quartile1NumericAttributeEntropy",
        "Quartile2NumericAttributeEntropy",
        "Quartile3NumericAttributeEntropy",
        "MaxNumericAttributeEntropy"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.INFO_THEORETIC],
    {
        "feature_array": "NoNaNBinnedNumericFeatures"
    }
)


def get_joint_entropy(feature_class_array):
    entropies = [get_entropy(translate_into_tuples(feature_class_pair[0],feature_class_pair[1])) for feature_class_pair in feature_class_array]
    return profile_distribution(entropies)

get_categorical_joint_entropy = MetafeatureComputer(
    get_joint_entropy,
    [
        "MeanCategoricalJointEntropy",
        "StdevCategoricalJointEntropy",
        "SkewCategoricalJointEntropy",
        "KurtosisCategoricalJointEntropy",
        "MinCategoricalJointEntropy",
        "Quartile1CategoricalJointEntropy",
        "Quartile2CategoricalJointEntropy",
        "Quartile3CategoricalJointEntropy",
        "MaxCategoricalJointEntropy"
    ],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "feature_class_array": "NoNaNCategoricalFeaturesAndClass"
    }
)

get_numeric_joint_entropy = MetafeatureComputer(
    get_joint_entropy,
    [
        "MeanNumericJointEntropy",
        "StdevNumericJointEntropy",
        "SkewNumericJointEntropy",
        "KurtosisNumericJointEntropy",
        "MinNumericJointEntropy",
        "Quartile1NumericJointEntropy",
        "Quartile2NumericJointEntropy",
        "Quartile3NumericJointEntropy",
        "MaxNumericJointEntropy"
    ],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "feature_class_array": "NoNaNBinnedNumericFeaturesAndClass"
    }
)


def get_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]
    return profile_distribution(mi_scores)

get_categorical_mutual_information = MetafeatureComputer(
    get_mutual_information,
    [
        "MeanCategoricalMutualInformation",
        "StdevCategoricalMutualInformation",
        "SkewCategoricalMutualInformation",
        "KurtosisCategoricalMutualInformation",
        "MinCategoricalMutualInformation",
        "Quartile1CategoricalMutualInformation",
        "Quartile2CategoricalMutualInformation",
        "Quartile3CategoricalMutualInformation",
        "MaxCategoricalMutualInformation"
    ],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "feature_class_array": "NoNaNCategoricalFeaturesAndClass"
    }
)

get_numeric_mutual_information = MetafeatureComputer(
    get_mutual_information,
    [
        "MeanNumericMutualInformation",
        "StdevNumericMutualInformation",
        "SkewNumericMutualInformation",
        "KurtosisNumericMutualInformation",
        "MinNumericMutualInformation",
        "Quartile1NumericMutualInformation",
        "Quartile2NumericMutualInformation",
        "Quartile3NumericMutualInformation",
        "MaxNumericMutualInformation"
    ],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "feature_class_array": "NoNaNBinnedNumericFeaturesAndClass"
    }
)


def get_equivalent_number_features(class_entropy, mutual_information):
    if mutual_information == 0:
        enf = np.nan
    else:
        enf = class_entropy / mutual_information
    return (enf,)

get_equivalent_number_categorical_features = MetafeatureComputer(
    get_equivalent_number_features,
    ["EquivalentNumberOfCategoricalFeatures"],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "class_entropy": "ClassEntropy",
        "mutual_information": "MeanCategoricalMutualInformation"
    }
)

get_equivalent_number_numeric_features = MetafeatureComputer(
    get_equivalent_number_features,
    ["EquivalentNumberOfNumericFeatures"],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "class_entropy": "ClassEntropy",
        "mutual_information": "MeanNumericMutualInformation"
    }
)


def get_noise_signal_ratio(attribute_entropy, mutual_information):
    if mutual_information == 0:
        nsr = np.nan
    else:
        nsr = (attribute_entropy - mutual_information) / mutual_information
    return (nsr,)

get_categorical_noise_signal_ratio = MetafeatureComputer(
    get_noise_signal_ratio,
    ["CategoricalNoiseToSignalRatio"],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "attribute_entropy": "MeanCategoricalAttributeEntropy",
        "mutual_information": "MeanCategoricalMutualInformation"
    }
)

get_numeric_noise_signal_ratio = MetafeatureComputer(
    get_noise_signal_ratio,
    ["NumericNoiseToSignalRatio"],
    ProblemType.CLASSIFICATION,
    [
        MetafeatureGroup.INFO_THEORETIC,
        MetafeatureGroup.TARGET_DEPENDENT
    ],
    {
        "attribute_entropy": "MeanNumericAttributeEntropy",
        "mutual_information": "MeanNumericMutualInformation"
    }
)


def translate_into_tuples(col1, col2):
    return pd.Series([x for x in zip(col1, col2)])

"""
A list of all MetafeatureComputer
instances in this module.
"""
metafeatures_info = build_resources_info(
    get_class_entropy,
    get_categorical_attribute_entropy,
    get_numeric_attribute_entropy,
    get_categorical_joint_entropy,
    get_numeric_joint_entropy,
    get_categorical_mutual_information,
    get_numeric_mutual_information,
    get_equivalent_number_categorical_features,
    get_equivalent_number_numeric_features,
    get_categorical_noise_signal_ratio,
    get_numeric_noise_signal_ratio
)
