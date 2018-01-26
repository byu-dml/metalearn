import numpy as np
from pandas import DataFrame

from .metafeatures_base import MetafeaturesBase

class SimpleMetafeatures(MetafeaturesBase):

    def __init__(self):
        
        function_dict = {
            'NumberOfInstances': self._get_dataset_stats,
            'NumberOfFeatures': self._get_dataset_stats,
            'NumberOfClasses': self._get_dataset_stats,
            'NumberOfNumericFeatures': self._get_dataset_stats,
            'NumberOfNominalFeatures': self._get_dataset_stats,
            'RatioOfNumericFeatures': self._get_dataset_stats,
            'RatioOfNominalFeatures': self._get_dataset_stats,
            'Dimensionality': self._get_dimensionality,
            'NumberOfMissingValues': self._get_missing_values,
            'RatioOfMissingValues': self._get_missing_values,
            'NumberOfInstancesWithMissingValues': self._get_missing_values,
            'RatioOfInstancesWithMissingValues': self._get_missing_values,
            'MeanClassProbability': self._get_class_stats,
            'StdevClassProbability': self._get_class_stats,
            'MinClassProbability': self._get_class_stats,
            'MaxClassProbability': self._get_class_stats,
            'MinorityClassSize': self._get_class_stats,
            'MajorityClassSize': self._get_class_stats,
            'MeanCardinalityOfNominalFeatures': self._get_nominal_cardinalities,
            'StdevCardinalityOfNominalFeatures': self._get_nominal_cardinalities,
            'MinCardinalityOfNominalFeatures': self._get_nominal_cardinalities,
            'MaxCardinalityOfNominalFeatures': self._get_nominal_cardinalities,
            'MeanCardinalityOfNumericFeatures': self._get_numeric_cardinalities,
            'StdevCardinalityOfNumericFeatures': self._get_numeric_cardinalities,
            'MinCardinalityOfNumericFeatures': self._get_numeric_cardinalities,
            'MaxCardinalityOfNumericFeatures': self._get_numeric_cardinalities,
        }

        dependencies_dict = {
            'NumberOfInstances': [],
            'NumberOfFeatures': [],
            'NumberOfClasses': [],          
            'NumberOfNumericFeatures': [],
            'NumberOfNominalFeatures': [],
            'RatioOfNumericFeatures': [],
            'RatioOfNominalFeatures': [],
            'Dimensionality': ['NumberOfFeatures','NumberOfInstances'],
            'NumberOfMissingValues': [],
            'RatioOfMissingValues': [],
            'NumberOfInstancesWithMissingValues': [],
            'RatioOfInstancesWithMissingValues': [],
            'MeanClassProbability': [],
            'StdevClassProbability': [],
            'MinClassProbability': [],
            'MaxClassProbability': [],
            'MinorityClassSize': [],
            'MajorityClassSize': [],
            'MeanCardinalityOfNominalFeatures': [],
            'StdevCardinalityOfNominalFeatures': [],
            'MinCardinalityOfNominalFeatures': [],
            'MaxCardinalityOfNominalFeatures': [],           
            'MeanCardinalityOfNumericFeatures': [],
            'StdevCardinalityOfNumericFeatures': [],
            'MinCardinalityOfNumericFeatures': [],
            'MaxCardinalityOfNumericFeatures': [],
        }

        super().__init__(function_dict, dependencies_dict)     

    def _get_dataset_stats(self, X, Y):
        number_of_instances = X.shape[0]
        number_of_features = X.shape[1]
        number_of_classes = Y.unique().shape[0]
        numeric_features = len(self._get_numeric_features(X))
        nominal_features = number_of_features - numeric_features
        ratio_of_numeric_features = float(numeric_features) / float(number_of_features)
        ratio_of_nominal_features = float(nominal_features) / float(number_of_features)
        return {
            'NumberOfInstances': number_of_instances,
            'NumberOfFeatures': number_of_features,
            'NumberOfClasses': number_of_classes,
            'NumberOfNumericFeatures': numeric_features,
            'NumberOfNominalFeatures': nominal_features,
            'RatioOfNumericFeatures': ratio_of_numeric_features,
            'RatioOfNominalFeatures': ratio_of_nominal_features
        }

    def _get_dimensionality(self, X, Y, number_of_features, number_of_instances):
        dimensionality = float(number_of_features) / float(number_of_instances)
        return {
            'Dimensionality': dimensionality
        }

    def _get_missing_values(self, X, Y):
        missing_values = X.shape[0] - X.count()
        number_missing = np.sum(missing_values)
        ratio_missing = float(number_missing) / float(X.shape[0] * X.shape[1])
        number_instances_with_missing = X.shape[1] - np.sum(missing_values == 0)
        ratio_instances_with_missing = float(number_instances_with_missing) / float(X.shape[1])
        return {
            'NumberOfMissingValues': number_missing,
            'RatioOfMissingValues': ratio_missing,
            'NumberOfInstancesWithMissingValues': number_instances_with_missing,
            'RatioOfInstancesWithMissingValues': ratio_instances_with_missing
        }

    def _get_class_stats(self, X, Y):
        classes = Y.unique()
        counts = [sum(Y == label) for label in classes]
        probs = [count/Y.shape[0] for count in counts]
        
        values_dict = self._profile_distribution(probs, 'ClassProbability')
        majority_class_size = max(counts)        
        minority_class_size = min(counts)
        return {
            'MeanClassProbability': values_dict['MeanClassProbability'],
            'StdevClassProbability': values_dict['StdevClassProbability'],
            'MinClassProbability': values_dict['MinClassProbability'],
            'MaxClassProbability': values_dict['MaxClassProbability'],
            'MinorityClassSize': minority_class_size,
            'MajorityClassSize': majority_class_size
        }

    def _get_nominal_cardinalities(self, X, Y):              
        cardinalities = [X[feature].unique().shape[0] for feature in X.columns if not self._dtype_is_numeric(X[feature].dtype)]
        values_dict = self._profile_distribution(cardinalities, 'CardinalityOfNominalFeatures')        
        return {
            'MeanCardinalityOfNominalFeatures': values_dict['MeanCardinalityOfNominalFeatures'],
            'StdevCardinalityOfNominalFeatures': values_dict['StdevCardinalityOfNominalFeatures'],
            'MinCardinalityOfNominalFeatures': values_dict['MinCardinalityOfNominalFeatures'],
            'MaxCardinalityOfNominalFeatures': values_dict['MaxCardinalityOfNominalFeatures']
        }

    def _get_numeric_cardinalities(self, X, Y):        
        cardinalities = [X[feature].unique().shape[0] for feature in X.columns if self._dtype_is_numeric(X[feature].dtype)]
        values_dict = self._profile_distribution(cardinalities, 'CardinalityOfNumericFeatures')        
        return {
            'MeanCardinalityOfNumericFeatures': values_dict['MeanCardinalityOfNumericFeatures'],
            'StdevCardinalityOfNumericFeatures': values_dict['StdevCardinalityOfNumericFeatures'],
            'MinCardinalityOfNumericFeatures': values_dict['MinCardinalityOfNumericFeatures'],
            'MaxCardinalityOfNumericFeatures': values_dict['MaxCardinalityOfNumericFeatures']
        }