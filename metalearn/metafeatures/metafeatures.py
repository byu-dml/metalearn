import os
import math
import json
import time
import io
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .common_operations import *
from .simple_metafeatures import *
from .statistical_metafeatures import *
from .information_theoretic_metafeatures import *
from .landmarking_metafeatures import *


class Metafeatures(object):
    """
    Computes metafeatures on a given tabular dataset (pandas.DataFrame) with
    categorical targets. These metafeatures are particularly useful for
    computing summary statistics on a dataset and for machine learning and
    meta-learning applications.
    """

    VALUE_NAME = 'value'
    TIME_NAME = 'time'
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    NO_TARGETS = "NO_TARGETS"
    N_CROSS_VALIDATION_FOLDS = 2
    COMPUTE_TIME_NAME = "_Time"

    def __init__(self):
        self.resource_info_dict = {}
        self.metafeatures_list = []
        mf_info_file_path = os.path.splitext(__file__)[0] + '.json'
        with open(mf_info_file_path, 'r') as f:
            mf_info_json = json.load(f)
            self.function_dict = mf_info_json['functions']
            json_metafeatures_dict = mf_info_json['metafeatures']
            json_resources_dict = mf_info_json['resources']
            self.metafeatures_list = list(json_metafeatures_dict.keys())
            combined_dict = {**json_metafeatures_dict, **json_resources_dict}
            for key in combined_dict:
                self.resource_info_dict[key] = combined_dict[key]

    def list_metafeatures(self):
        """
        Returns a list of metafeatures computable by the Metafeatures class.
        """
        return self.metafeatures_list

    def list_target_dependent_metafeatures(self):
        return list(filter(self._is_target_dependent, self.metafeatures_list))

    def compute(
        self, X: DataFrame, Y: Series = None,
        column_types: Dict[str, str] = None, metafeature_ids: List = None,
        sample_shape=None, seed=None, timer=False
    ) -> DataFrame:
        """
        Parameters
        ----------
        X: pandas.DataFrame, the dataset features
        Y: pandas.Seris, the dataset targets
        column_types: Dict[str, str], dict from column name to column type as
            "NUMERIC" or "CATEGORICAL", must include Y column
        metafeature_ids: list, the metafeatures to compute. default of None
            indicates to compute all metafeatures
        sample_shape: tuple, the shape of X after sampling (X,Y) uniformly.
            Default is (None, None), indicate not to sample rows or columns.
        seed: int, the seed used to generate psuedo-random numbers. when None
            is given, a seed will be generated psuedo-randomly. this can be
            used for reproducibility of metafeatures. a generated seed can be
            accessed through the 'seed' property, after calling this method.
        timer: bool, whether return the computation time of each metafeature in
            addition to the value of each metafeature

        Returns
        -------
        A dataframe containing one row and two columns for each metafeature:
        one for the value and one for the compute time of that metafeature
        value
        """
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_shape, seed, timer
        )
        if column_types is None:
            column_types = self._infer_column_types(X, Y)
        if metafeature_ids is None:
            metafeature_ids = self.list_metafeatures()
        if sample_shape is None:
            sample_shape = (None, None)
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_shape, seed, timer
        )

        self.computed_metafeatures = DataFrame()

        X_raw = X
        X = X_raw.dropna(axis=1, how='all')
        self._set_seed(seed)
        self.resource_results_dict = {
            'XRaw': {self.VALUE_NAME: X_raw, self.TIME_NAME: 0.},
            'X': {self.VALUE_NAME: X, self.TIME_NAME: 0.},
            'Y': {self.VALUE_NAME: Y, self.TIME_NAME: 0.},
            'ColumnTypes': {
                self.VALUE_NAME: column_types, self.TIME_NAME: 0.
            },
            'sample_shape': {
                self.VALUE_NAME: sample_shape, self.TIME_NAME: 0.
            },
            "n_cross_validation_folds": {
                self.VALUE_NAME: self.N_CROSS_VALIDATION_FOLDS, self.TIME_NAME: 0.
            }
        }
        if Y is None:
            target_dependent_metafeatures = self.list_target_dependent_metafeatures()
            # set every target-dependent metafeature that was requested by the user to "NO_TARGETS"
            for metafeature_id in target_dependent_metafeatures:
                if metafeature_id in metafeature_ids:
                    self.computed_metafeatures.at[0, metafeature_id] = self.NO_TARGETS
                    if timer:
                        metafeature_time_id = metafeature_id + self.COMPUTE_TIME_NAME
                        self.computed_metafeatures.at[0, metafeature_time_id] = self.NO_TARGETS
            # remove any target-dependent metafeatures from metafeature_ids so there is no attempt to compute them
            metafeature_ids = [mf for mf in metafeature_ids if mf not in target_dependent_metafeatures]
        self._compute_metafeatures(metafeature_ids, timer)

        return self.computed_metafeatures

    def _is_target_dependent(self, resource_name):
        if resource_name=='Y':
            return True
        elif resource_name=='XSample':
            return False
        else:
            resource_info = self.resource_info_dict[resource_name]
            parameters = resource_info.get('parameters', [])
            for parameter in parameters:
                if self._is_target_dependent(parameter):
                    return True
            function = resource_info['function']
            parameters = self.function_dict[function]['parameters']
            for parameter in parameters:
                if self._is_target_dependent(parameter):
                    return True
            return False

    def _set_seed(self, seed):
        if seed is None:
            self.seed = np.random.randint(2**32)
        else:
            self.seed = seed

    def _get_seed(self):
        return (self.seed + self.seed_offset,)

    def _validate_compute_arguments(
        self, X, Y, column_types, metafeature_ids, sample_shape, seed, timer
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pandas.DataFrame')
        if not isinstance(Y, pd.Series) and not Y is None:
            raise TypeError('Y must be of type pandas.Series')
        if column_types is not None:
            if not Y is None:
                if len(column_types.keys()) != len(X.columns) + 1:
                    raise ValueError(
                        "The number of column_types does not match the number of " +
                        "features plus the target"
                    )
                if column_types[Y.name] == self.NUMERIC:
                    raise TypeError('Regression problems are not supported (target feature is numeric)')
            else:
                if len(column_types.keys()) != len(X.columns):
                    raise ValueError(
                        "The number of column_types does not match the number of " +
                        "features"
                    )
            invalid_column_types = []
            for col_name, col_type in column_types.items():
                if col_type != self.NUMERIC and col_type != self.CATEGORICAL:
                    invalid_column_types.append((col_name, col_type))
            if len(invalid_column_types) > 0:
                raise ValueError(
                    'One or more input column types are not valid: {}. Valid '+
                    'types include {} and {}.'.
                    format(
                        invalid_column_types, self.NUMERIC, self.CATEGORICAL
                    )
                )
        if metafeature_ids is not None:
            invalid_metafeature_ids = [
                mf for mf in metafeature_ids if
                mf not in self.resource_info_dict
            ]
            if len(invalid_metafeature_ids) > 0:
                raise ValueError(
                    'One or more requested metafeatures are not valid: {}'.
                    format(invalid_metafeature_ids)
                )
        if not sample_shape is None:
            if not sample_shape[0] is None and not Y is None:
                min_samples = Y.unique().shape[0] * self.N_CROSS_VALIDATION_FOLDS
                if min_samples > sample_shape[0]:
                    raise ValueError(f"Cannot sample less than {min_samples} rows from Y")
            if not sample_shape[1] is None and sample_shape[1] < 1:
                raise ValueError("Cannot sample less than 1 column")
        if not type(timer) is bool:
            raise ValueError("`timer` must of type `bool`")

    def _infer_column_types(self, X, Y):
        column_types = {}
        for col_name in X.columns:
            if dtype_is_numeric(X[col_name].dtype):
                column_types[col_name] = self.NUMERIC
            else:
                column_types[col_name] = self.CATEGORICAL
        if not Y is None:
            if dtype_is_numeric(Y.dtype):
                column_types[Y.name] = self.NUMERIC
            else:
                column_types[Y.name] = self.CATEGORICAL
        return column_types

    def _compute_metafeatures(self, metafeature_ids, timer):
        for metafeature_id in metafeature_ids:
            value, time_value = self._retrieve_resource(metafeature_id)
            self.computed_metafeatures.at[0, metafeature_id] = value
            if timer:
                metafeature_time_id = metafeature_id + self.COMPUTE_TIME_NAME
                self.computed_metafeatures.at[0, metafeature_time_id] = time_value

    def _retrieve_resource(self, resource_name):
        if resource_name not in self.resource_results_dict:
            retrieved_parameters, total_time = self._retrieve_parameters(
                resource_name
            )
            resource_info = self.resource_info_dict[resource_name]
            f = resource_info['function']
            if 'returns' in resource_info:
                returns = resource_info['returns']
            else:
                returns = self.function_dict[f]['returns']
            if retrieved_parameters is None:
                results = tuple([np.nan] * len(returns))
                total_time = np.nan
            else:
                start = time.time()
                results = eval(f)(*retrieved_parameters)
                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time
            for i in range(len(results)):
                result = results[i]
                result_name = returns[i]
                self.resource_results_dict[result_name] = {
                    self.VALUE_NAME: result, self.TIME_NAME: total_time
                }
        value = self.resource_results_dict[resource_name][self.VALUE_NAME]
        total_time = self.resource_results_dict[resource_name][self.TIME_NAME]
        return (value, total_time)

    def _retrieve_parameters(self, resource_name):
        total_time = 0.0
        retrieved_parameters = []
        resource_info = self.resource_info_dict[resource_name]
        f = resource_info['function']
        if 'parameters' in resource_info:
            parameters = resource_info['parameters']
        else:
            parameters = self.function_dict[f]['parameters']
        if 'seed_offset' in resource_info:
            self.seed_offset = resource_info['seed_offset']
        elif 'seed_offset' in self.function_dict[f]:
            self.seed_offset = self.function_dict[f]['seed_offset']
        for parameter in parameters:
            if isinstance(parameter, float) or isinstance(parameter, int):
                value, time_value = parameter, 0.
            else:
                value, time_value = self._retrieve_resource(parameter)
            if value is np.nan:
                retrieved_parameters = None
                break
            retrieved_parameters.append(value)
            total_time += time_value
        return (retrieved_parameters, total_time)

    def _get_preprocessed_data(self, X_sample, X_sampled_columns, column_types, seed=42):
        series_array = []
        for feature in X_sample.columns:
            feature_series = X_sample[feature].copy()
            col = feature_series.values
            dropped_nan_series = X_sampled_columns[feature].dropna(
                axis=0,how='any'
            )
            num_nan = np.sum(feature_series.isnull())
            np.random.seed(seed)
            col[feature_series.isnull()] = np.random.choice(
                dropped_nan_series, size=num_nan
            )
            if column_types[feature_series.name] == self.CATEGORICAL:
                feature_series = pd.get_dummies(feature_series)
            series_array.append(feature_series)
        return (pd.concat(series_array, axis=1, copy=False),)

    def _sample_columns(self, X, sample_shape, seed):
        if sample_shape[1] is None or X.shape[1] <= sample_shape[1]:
            X_sample = X
        else:
            np.random.seed(seed)
            sampled_column_indices = np.random.choice(
                X.shape[1], size=sample_shape[1], replace=False
            )
            sampled_columns = X.columns[sampled_column_indices]
            X_sample = X[sampled_columns]
        return (X_sample,)

    def _sample_rows(self, X, Y, sample_shape, n_cross_validation_folds, seed):
        """
        Stratified uniform sampling of rows, according to the classes in Y.
        Ensures there are enough samples from each class in Y for cross
        validation.
        """
        np.random.seed(seed)
        if sample_shape[0] is None or X.shape[0] <= sample_shape[0]:
            X_sample, Y_sample = X, Y
        elif Y is None:
            row_indices = np.random.choice(
                X.shape[0], size=sample_shape[0], replace=False
            )
            X_sample, Y_sample = X.iloc[row_indices], Y
        else:
            row_indices = []
            Y_groupby = Y.groupby(Y)
            min_class_samples = n_cross_validation_folds
            n_classes = len(Y_groupby)
            for class_ in Y_groupby.groups:
                class_indices = Y_groupby.get_group(class_).index
                sample_ratio = (len(class_indices) - min_class_samples) / (
                    Y.shape[0] - min_class_samples*n_classes
                )
                sample_ratio = max(0, sample_ratio)
                n_samples = int(round(
                    (sample_shape[0] - min_class_samples*n_classes) * sample_ratio + min_class_samples
                ))
                replace = False
                # sample with replacement when data is limited
                if n_samples > len(class_indices):
                    replace = True
                class_row_indices = np.random.choice(
                    class_indices, size=n_samples, replace=replace
                )
                row_indices = np.append(row_indices, class_row_indices)
            n_sampled_rows = len(row_indices)
            # todo handle this case internally
            if n_sampled_rows != sample_shape[0]:
                raise Exception(f"sample_shape {sample_shape} sampled rows {n_sampled_rows}")
            X_sample, Y_sample = X.iloc[row_indices], Y.iloc[row_indices]
        return (X_sample, Y_sample)

    def _get_categorical_features_with_no_missing_values(
        self, X_sample, column_types
    ):
        categorical_features_with_no_missing_values = []
        for feature in X_sample.columns:
            if column_types[feature] == self.CATEGORICAL:
                no_nan_series = X_sample[feature].dropna(
                    axis=0, how='any'
                )
                categorical_features_with_no_missing_values.append(
                    no_nan_series
                )
        return (categorical_features_with_no_missing_values,)

    def _get_categorical_features_and_class_with_no_missing_values(
        self, X_sample, Y_sample, column_types
    ):
        categorical_features_and_class_with_no_missing_values = []
        for feature in X_sample.columns:
            if column_types[feature] == self.CATEGORICAL:
                df = pd.concat([X_sample[feature],Y_sample], axis=1).dropna(
                    axis=0, how='any'
                )
                categorical_features_and_class_with_no_missing_values.append(
                    (df[feature],df[Y_sample.name])
                )
        return (categorical_features_and_class_with_no_missing_values,)

    def _get_numeric_features_with_no_missing_values(
        self, X_sample, column_types
    ):
        numeric_features_with_no_missing_values = []
        for feature in X_sample.columns:
            if column_types[feature] == self.NUMERIC:
                no_nan_series = X_sample[feature].dropna(
                    axis=0, how='any'
                )
                numeric_features_with_no_missing_values.append(
                    no_nan_series
                )
        return (numeric_features_with_no_missing_values,)

    def _get_binned_numeric_features_with_no_missing_values(
        self, numeric_features_array
    ):
        binned_feature_array = [
            (
                pd.cut(feature,
                round(feature.shape[0]**(1./3.)))
            ) for feature in numeric_features_array
        ]
        return (binned_feature_array,)

    def _get_binned_numeric_features_and_class_with_no_missing_values(
        self, X_sample, Y_sample, column_types
    ):
        numeric_features_and_class_with_no_missing_values = []
        for feature in X_sample.columns:
            if column_types[feature] == self.NUMERIC:
                df = pd.concat([X_sample[feature],Y_sample], axis=1).dropna(
                    axis=0, how='any'
                )
                numeric_features_and_class_with_no_missing_values.append(
                    (df[feature],df[Y_sample.name])
                )
        binned_feature_class_array = [
            (
                pd.cut(feature_class_pair[0],
                round(feature_class_pair[0].shape[0]**(1./3.))),
                feature_class_pair[1]
            ) for feature_class_pair in numeric_features_and_class_with_no_missing_values
        ]
        return (binned_feature_class_array,)
