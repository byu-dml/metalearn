import os
import math
import json
import time
import io
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedShuffleSplit

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

    VALUE_KEY = 'value'
    COMPUTE_TIME_KEY = 'compute_time'
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    NO_TARGETS = "NO_TARGETS"

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

    def list_metafeatures(self, group="all"):
        """
        Returns a list of metafeatures computable by the Metafeatures class.
        """
        if group == "all":
            return self.metafeatures_list
        elif group == "landmarking":
            return list(filter(
                lambda mf_id: "ErrRate" in mf_id or "Kappa" in mf_id,
                self.metafeatures_list
            ))
        else:
            raise ValueError(f"Unknown group {group}")

    def list_target_dependent_metafeatures(self):
        return list(filter(self._is_target_dependent, self.metafeatures_list))

    def compute(
        self, X: DataFrame, Y: Series = None,
        column_types: Dict[str, str] = None, metafeature_ids: List = None,
        sample_shape=None, seed=None, n_folds=2
    ) -> dict:
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
        n_folds: int, the number of cross validation folds used by the
            landmarking metafeatures. also affects the sample_shape validation

        Returns
        -------
        A dictionary mapping the metafeature id to another dictionary containing
        the `value` and `compute_time` (if requested) of the referencing
        metafeature. The value is typically a number, but can be a string
        indicating a reason why the value could not be computed.
        """
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
        )
        if column_types is None:
            column_types = self._infer_column_types(X, Y)
        if metafeature_ids is None:
            metafeature_ids = self.list_metafeatures()
        if sample_shape is None:
            sample_shape = (None, None)
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
        )

        self._set_seed(seed)

        # todo refactor to separate function
        self.resource_results_dict = {
            "XRaw": {
                self.VALUE_KEY: X,
                self.COMPUTE_TIME_KEY: 0.
            },
            "X": {
                self.VALUE_KEY: X.dropna(axis=1, how="all"),
                self.COMPUTE_TIME_KEY: 0.
            },
            "Y": {
                self.VALUE_KEY: Y,
                self.COMPUTE_TIME_KEY: 0.
            },
            "ColumnTypes": {
                self.VALUE_KEY: column_types,
                self.COMPUTE_TIME_KEY: 0.
            },
            "sample_shape": {
                self.VALUE_KEY: sample_shape,
                self.COMPUTE_TIME_KEY: 0.
            },
            "n_folds": {
                self.VALUE_KEY: n_folds,
                self.COMPUTE_TIME_KEY: 0.
            }
        }

        self.computed_metafeatures = {}
        if Y is None:
            # todo refactor to separate function, perhaps combine with _compute_metafeatures
            # set every target-dependent metafeature that was requested by the
            # user to "NO_TARGETS"
            copmutable_metafeature_ids = []
            for metafeature_id in metafeature_ids:
                if self._is_target_dependent(metafeature_id):
                    self.computed_metafeatures[metafeature_id] = {
                        self.VALUE_KEY: self.NO_TARGETS,
                        self.COMPUTE_TIME_KEY: None
                    }
                else:
                    copmutable_metafeature_ids.append(metafeature_id)
            metafeature_ids = copmutable_metafeature_ids

        self._compute_metafeatures(metafeature_ids)
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
        self, X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
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
        self._validate_sample_shape(
            X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
        )
        self._validate_n_folds(
            X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
        )

    def _validate_sample_shape(
        self, X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
    ):
        if not sample_shape is None:
            if not type(sample_shape) in [tuple, list]:
                raise ValueError(
                    "`sample_shape` must be of type `tuple` or `list`"
                )
            if len(sample_shape) != 2:
                raise ValueError("`sample_shape` must be of length 2")
            if not sample_shape[0] is None and sample_shape[0] < 1:
                raise ValueError("Cannot sample less than one row")
            if not sample_shape[1] is None and sample_shape[1] < 1:
                raise ValueError("Cannot sample less than 1 column")
            if not sample_shape[0] is None and not Y is None:
                min_samples = Y.unique().shape[0] * n_folds
                if sample_shape[0] < min_samples:
                    raise ValueError(
                        f"Cannot sample less than {min_samples} rows from Y"
                    )

    def _validate_n_folds(
        self, X, Y, column_types, metafeature_ids, sample_shape, seed, n_folds
    ):
        if not dtype_is_numeric(type(n_folds)) or (n_folds != int(n_folds)):
            raise ValueError(f"`n_folds` must be an integer, not {n_folds}")
        if n_folds < 2:
            raise ValueError(f"`n_folds` must be >= 2, but was {n_folds}")
        if not Y is None and metafeature_ids is not None:
            # when computing landmarking metafeatures, there must be at least
            # n_folds instances of each class of Y
            landmarking_mfs = self.list_metafeatures(group="landmarking")
            if len(list(filter(
                lambda mf_id: mf_id in landmarking_mfs,metafeature_ids
            ))):
                Y_grouped = Y.groupby(Y)
                for group_id, group in Y_grouped:
                    if group.shape[0] < n_folds:
                        raise ValueError(
                            "The minimum number of instances in each class of" +
                            f" Y is n_folds={n_folds}. Class {group_id} has " +
                            f"{group.shape[0]}."
                        )

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

    def _compute_metafeatures(self, metafeature_ids):
        for metafeature_id in metafeature_ids:
            value, compute_time = self._retrieve_resource(metafeature_id)
            self.computed_metafeatures[metafeature_id] = {
                self.VALUE_KEY: value,
                self.COMPUTE_TIME_KEY: compute_time
            }

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
                    self.VALUE_KEY: result,
                    self.COMPUTE_TIME_KEY: total_time
                }
        value = self.resource_results_dict[resource_name][self.VALUE_KEY]
        total_time = self.resource_results_dict[resource_name][self.COMPUTE_TIME_KEY]
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

    def _get_preprocessed_data(self, X_sample, X_sampled_columns, column_types, seed):
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

    def _sample_rows(self, X, Y, sample_shape, n_folds, seed):
        """
        Stratified uniform sampling of rows, according to the classes in Y.
        Ensures there are enough samples from each class in Y for cross
        validation.
        """
        if sample_shape[0] is None or X.shape[0] <= sample_shape[0]:
            X_sample, Y_sample = X, Y
        elif Y is None:
            np.random.seed(seed)
            row_indices = np.random.choice(
                X.shape[0], size=sample_shape[0], replace=False
            )
            X_sample, Y_sample = X.iloc[row_indices], Y
        else:
            drop_size = X.shape[0] - sample_shape[0]
            sample_size = sample_shape[0]
            sss = StratifiedShuffleSplit(
                n_splits=2, test_size=drop_size, train_size=sample_size, random_state=seed
            )
            row_indices, _ = next(sss.split(X, Y))
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
