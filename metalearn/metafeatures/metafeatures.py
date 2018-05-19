import os
import math
import json
import time
import multiprocessing
from contextlib import redirect_stderr
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


def threadsafe_timeout_function(f, args, timeout):
    p = multiprocessing.Process(target=f, args=args)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()

class Metafeatures(object):
    """
    Computes metafeatures on a given tabular dataset (pandas.DataFrame) with
    categorical targets. These metafeatures are particularly useful for
    computing summary statistics on a dataset and for machine learning and
    meta-learning applications.
    """

    VALUE_NAME = 'value'
    TIME_NAME = 'time'
    TIMEOUT_BUFFER = .1
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"

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

    def compute(
        self, X: DataFrame, Y: Series, column_types: Dict[str, str] = None,
        metafeature_ids: List = None, sample_rows=True, sample_columns=True,
        seed=None, timeout=None
    ) -> DataFrame:
        """
        Parameters
        ----------
        X: pandas.DataFrame, the dataset features
        Y: pandas.Seris, the dataset targets
        metafeature_ids: list, the metafeatures to compute.
            default of None indicates to compute all metafeatures
        sample_rows: bool, whether to uniformly sample from the rows
        sample_columns: bool, whether to uniformly sample from the columns
        seed: int, the seed used to generate psuedo-random numbers.
            default is None, a seed will be generated randomly
        timeout: int, the maximum amount of wall time in seconds used to
            compute metafeatures

        Returns
        -------
        A dataframe containing one row and two columns for each metafeature:
        one for the value and one for the compute time of that metafeature
        value
        """
        self.computed_metafeatures = DataFrame()
        if timeout is None:
            self._compute(
                X, Y, column_types, metafeature_ids, sample_rows,
                sample_columns, seed
            )
        else:
            with redirect_stderr(io.StringIO()):
                threadsafe_timeout_function(
                    self._compute,
                    (
                        X, Y, column_types, metafeature_ids, sample_rows,
                        sample_columns, seed
                    ),
                    timeout-self.TIMEOUT_BUFFER
                )
        return self.computed_metafeatures

    def _compute(
        self, X, Y, column_types, metafeature_ids, sample_rows, sample_columns,
        seed
    ):
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_rows, sample_columns,
            seed
        )
        if column_types is None:
            column_types = self._infer_column_types(X, Y)
        if metafeature_ids is None:
            metafeature_ids = self.list_metafeatures()
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_rows, sample_columns,
            seed
        )

        X_raw = X
        X = X_raw.dropna(axis=1, how='all')
        self._set_random_seed(seed)
        self.resource_results_dict = {
            'XRaw': {self.VALUE_NAME: X_raw, self.TIME_NAME: 0.},
            'X': {self.VALUE_NAME: X, self.TIME_NAME: 0.},
            'Y': {self.VALUE_NAME: Y, self.TIME_NAME: 0.},
            'ColumnTypes': {self.VALUE_NAME: column_types, self.TIME_NAME: 0.},
            'SampleRowsFlag': {
                self.VALUE_NAME: sample_rows, self.TIME_NAME: 0.
            },
            'SampleColumnsFlag': {
                self.VALUE_NAME: sample_columns, self.TIME_NAME: 0.
            }
        }
        self._compute_metafeatures(metafeature_ids)

    def _set_random_seed(self, seed):
        if seed is None:
            self.seed = np.random.randint(2**32)
        else:
            self.seed = seed

    def _get_random_seed(self):
        return (self.seed + self.seed_offset,)

    def _validate_compute_arguments(
        self, X, Y, column_types, metafeature_ids, sample_rows, sample_columns,
        seed
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pandas.DataFrame')
        if not isinstance(Y, pd.Series):
            raise TypeError('Y must be of type pandas.Series')
        if column_types is not None:
            if len(column_types.keys()) != len(X.columns) + 1:
                raise ValueError(
                    "The number of column_types does not match the number of " +
                    "features plus the target"
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
            if column_types[Y.name] == self.NUMERIC:
                raise TypeError('Regression problems are not supported (target feature is numeric)')
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

    def _infer_column_types(self, X, Y):
        column_types = {}
        for col_name in X.columns:
            if dtype_is_numeric(X[col_name].dtype):
                column_types[col_name] = self.NUMERIC
            else:
                column_types[col_name] = self.CATEGORICAL
        if dtype_is_numeric(Y.dtype):
            column_types[Y.name] = self.NUMERIC
        else:
            column_types[Y.name] = self.CATEGORICAL
        return column_types

    def _compute_metafeatures(self, metafeature_ids):
        for metafeature_id in metafeature_ids:
            value, time_value = self._retrieve_resource(metafeature_id)
            row, col = 0, metafeature_id
            self.computed_metafeatures.at[row, col] = value
            col += "_Time"
            self.computed_metafeatures.at[row, col] = time_value

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
            feature_series = X_sample[feature]
            col = feature_series.as_matrix()
            dropped_nan_series = X_sampled_columns[feature].dropna(
                axis=0,how='any'
            )
            num_nan = np.sum(feature_series.isnull())
            np.random.seed(seed)
            col[feature_series.isnull()] = np.random.choice(
                dropped_nan_series, num_nan
            )
            if column_types[feature_series.name] == self.CATEGORICAL:
                feature_series = pd.get_dummies(feature_series)
            series_array.append(feature_series)
        return (pd.concat(series_array, axis=1, copy=False),)

    def _get_sample_of_columns(
        self, X, sample_columns, seed=42, max_columns=150
    ):
        if sample_columns and X.shape[1] > max_columns:
            np.random.seed(seed)
            column_indices = np.random.permutation(X.shape[1])[:max_columns]
            columns = X.columns[column_indices]
            return (X[columns],)
        else:
            return (X,)

    def _get_sample_of_rows(
        self, X, Y, sample_rows, seed=42, approximate_max_rows=150000,
        min_row_per_class=2
    ):
        if sample_rows == True and X.shape[0] > approximate_max_rows:
            samples = []
            total_rows = Y.shape[0]
            class_groupby = Y.groupby(Y)
            for group_key in class_groupby.groups:
                group = class_groupby.get_group(group_key).index
                num_to_sample = max(
                    math.floor(
                        float(group.shape[0]) / float(total_rows) *
                        approximate_max_rows
                    ), min_row_per_class
                )
                np.random.seed(seed)
                row_indices = np.random.permutation(group)[:num_to_sample]
                samples.append(row_indices)
            row_indices = np.concatenate(samples)
            return (X.iloc[row_indices], Y.iloc[row_indices])
        else:
            return (X, Y)

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

    def _get_numeric_features_and_class_with_no_missing_values(
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
        return (numeric_features_and_class_with_no_missing_values,)

    def _get_binned_numeric_features_and_class_with_no_missing_values(
        self, numeric_features_class_array
    ):
        binned_feature_class_array = [
            (
                pd.cut(feature_class_pair[0],
                round(feature_class_pair[0].shape[0]**(1./3.))),
                feature_class_pair[1]
            ) for feature_class_pair in numeric_features_class_array
        ]
        return (binned_feature_class_array,)
