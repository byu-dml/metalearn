import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .metafeatures_base import MetafeaturesBase


class InformationTheoreticMetafeatures(MetafeaturesBase):

    def __init__(self):

        function_dict = {
            'ClassEntropy': self._get_class_entropy,
            'MeanAttributeEntropy': self._get_attribute_entropy,
            'MinAttributeEntropy': self._get_attribute_entropy,
            'MaxAttributeEntropy': self._get_attribute_entropy,
            'Quartile1AttributeEntropy': self._get_attribute_entropy,
            'Quartile2AttributeEntropy': self._get_attribute_entropy,
            'Quartile3AttributeEntropy': self._get_attribute_entropy,
            'MeanJointEntropy': self._get_joint_entropy,
            'MinJointEntropy': self._get_joint_entropy,
            'MaxJointEntropy': self._get_joint_entropy,
            'Quartile1JointEntropy': self._get_joint_entropy,
            'Quartile2JointEntropy': self._get_joint_entropy,
            'Quartile3JointEntropy': self._get_joint_entropy,
            'MeanMutualInformation': self._get_mutual_information,
            'MinMutualInformation': self._get_mutual_information,
            'MaxMutualInformation': self._get_mutual_information,
            'Quartile1MutualInformation': self._get_mutual_information,
            'Quartile2MutualInformation': self._get_mutual_information,
            'Quartile3MutualInformation': self._get_mutual_information,
            'EquivalentNumberOfFeatures': self._get_equivalent_number_features,
            'NoiseToSignalRatio': self._get_noise_signal_ratio
        }

        dependencies_dict = {
            'ClassEntropy': [],
            'MeanAttributeEntropy': [],
            'MinAttributeEntropy': [],
            'MaxAttributeEntropy': [],
            'Quartile1AttributeEntropy': [],
            'Quartile2AttributeEntropy': [],
            'Quartile3AttributeEntropy': [],
            'MeanJointEntropy': [],
            'MinJointEntropy': [],
            'MaxJointEntropy': [],
            'Quartile1JointEntropy': [],
            'Quartile2JointEntropy': [],
            'Quartile3JointEntropy': [],
            'MeanMutualInformation': [],
            'MinMutualInformation': [],
            'MaxMutualInformation': [],
            'Quartile1MutualInformation': [],
            'Quartile2MutualInformation': [],
            'Quartile3MutualInformation': [],
            'EquivalentNumberOfFeatures': ['ClassEntropy','MeanMutualInformation'],
            'NoiseToSignalRatio': ['MeanAttributeEntropy','MeanMutualInformation']
        }

        super().__init__(function_dict, dependencies_dict)

    def _get_entropy(self, col):
        return entropy(col.value_counts())

    def _get_attribute_entropy(self, X, Y):
        entropies = []
        for feature in X.columns:
            col = X[feature].dropna(axis=0, how="any")
            if self._dtype_is_numeric(col.dtype):
                col = pd.cut(col, col.shape[0]**.5)
            entropies.append(self._get_entropy(col))
        values_dict = self._profile_distribution(entropies, 'AttributeEntropy')
        return {
            'MeanAttributeEntropy': values_dict['MeanAttributeEntropy'],
            'MinAttributeEntropy': values_dict['MinAttributeEntropy'],
            'MaxAttributeEntropy': values_dict['MaxAttributeEntropy'],
            'Quartile1AttributeEntropy': values_dict['Quartile1AttributeEntropy'],
            'Quartile2AttributeEntropy': values_dict['Quartile2AttributeEntropy'],
            'Quartile3AttributeEntropy': values_dict['Quartile3AttributeEntropy']
        }

    def _get_class_entropy(self, X, Y):
        return {
            "ClassEntropy": self._get_entropy(Y)
        }

    def _get_joint_entropy(self, X, Y):
        entropies = []
        for feature in X.columns:
            df_col_Y = pd.concat([X[feature],Y], axis=1).dropna(axis=0, how="any")
            col = df_col_Y[feature]
            targets = df_col_Y[Y.name]
            if self._dtype_is_numeric(col.dtype):
                col = pd.cut(col, round(col.shape[0]**.5))
            col = col.astype(str) + targets.astype(str)
            entropy = self._get_entropy(col)
            entropies.append(entropy)
        values_dict = self._profile_distribution(entropies, 'JointEntropy')
        return {
            'MeanJointEntropy': values_dict['MeanJointEntropy'],
            'MinJointEntropy': values_dict['MinJointEntropy'],
            'MaxJointEntropy': values_dict['MaxJointEntropy'],
            'Quartile1JointEntropy': values_dict['Quartile1JointEntropy'],
            'Quartile2JointEntropy': values_dict['Quartile2JointEntropy'],
            'Quartile3JointEntropy': values_dict['Quartile3JointEntropy']
        }

    def _get_mutual_information(self, X, Y):
        mi_scores = []
        for feature in X.columns:
            df_col_Y = pd.concat([X[feature],Y], axis=1).dropna(axis=0, how="any")
            col = df_col_Y[feature]
            targets = df_col_Y[Y.name]
            if self._dtype_is_numeric(col.dtype):
                col = pd.cut(col, round(col.shape[0]**.5))
            mi_scores.append(mutual_info_score(col, targets))
        values_dict = self._profile_distribution(mi_scores, 'MutualInformation')
        return {
            'MeanMutualInformation': values_dict['MeanMutualInformation'],
            'MinMutualInformation': values_dict['MinMutualInformation'],
            'MaxMutualInformation': values_dict['MaxMutualInformation'],
            'Quartile1MutualInformation': values_dict['Quartile1MutualInformation'],
            'Quartile2MutualInformation': values_dict['Quartile2MutualInformation'],
            'Quartile3MutualInformation': values_dict['Quartile3MutualInformation']
        }

    def _get_equivalent_number_features(self, X, Y, class_entropy, mutual_information):
        if mutual_information == 0:
            enf = np.nan
        else:
            enf = class_entropy / mutual_information
        return {
            'EquivalentNumberOfFeatures': enf
        }

    def _get_noise_signal_ratio(self, X, Y, attribute_entropy, mutual_information):
        if mutual_information == 0:
            nsr = np.nan
        else:
            nsr = (attribute_entropy - mutual_information) / mutual_information
        return {
            'NoiseToSignalRatio': nsr
        }
