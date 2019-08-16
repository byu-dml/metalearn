import os
import math
import json
import time
import io
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedShuffleSplit

from .resources import METAFEATURE_CONFIG
from .common_operations import *
from .simple_metafeatures import *
from .statistical_metafeatures import *
from .information_theoretic_metafeatures import *
from .landmarking_metafeatures import *
from .decision_tree_metafeatures import *
from .text_metafeatures import *


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
    TEXT = "TEXT"
    CATEGORICAL = "CATEGORICAL"
    NO_TARGETS = "NO_TARGETS"
    NUMERIC_TARGETS = "NUMERIC_TARGETS"
    TIMEOUT = "TIMEOUT"

    with open(METAFEATURE_CONFIG, 'r') as f:
        _metadata = json.load(f)
    IDS = list(_metadata["metafeatures"].keys())
    _resources_info = {}
    _resources_info.update(_metadata["resources"])
    _resources_info.update(_metadata["metafeatures"])

    @classmethod
    def list_metafeatures(cls, group="all"):
        """
        Returns a list of metafeatures computable by the Metafeatures class.
        """
        # todo make group for intractable metafeatures for wide datasets or
        # datasets with high cardinality categorical columns:
        # PredPCA1, PredPCA2, PredPCA3, PredEigen1, PredEigen2, PredEigen3,
        # PredDet, kNN1NErrRate, kNN1NKappa, LinearDiscriminantAnalysisKappa,
        # LinearDiscriminantAnalysisErrRate
        if group == "all":
            return copy.deepcopy(cls.IDS)
        elif group == "landmarking":
            return list(filter(
                lambda mf_id: "ErrRate" in mf_id or "Kappa" in mf_id, cls.IDS
            ))
        elif group == "target_dependent":
            return list(filter(
                cls._resource_is_target_dependent, cls.IDS
            ))
        else:
            raise ValueError(f"Unknown group {group}")

    def compute(
        self, X: DataFrame, Y: Series=None,
        column_types: Dict[str, str]=None, metafeature_ids: List=None,
        exclude: List=None, sample_shape=None, seed=None, n_folds=2,
        verbose=False, timeout=None
    ) -> dict:
        """
        Parameters
        ----------
        X: pandas.DataFrame, the dataset features
        Y: pandas.Series, the dataset targets
        column_types: Dict[str, str], dict from column name to column type as
            "NUMERIC" or "CATEGORICAL" or "TEXT", must include Y column
        metafeature_ids: list, the metafeatures to compute. default of None
            indicates to compute all metafeatures
        exclude: list, default None. The metafeatures to be excluded from computation.
            Must be None if metafeature_ids is not None.
        sample_shape: tuple, the shape of X after sampling (X,Y) uniformly.
            Default is (None, None), indicate not to sample rows or columns.
        seed: int, the seed used to generate pseudo-random numbers. when None
            is given, a seed will be generated pseudo-randomly. this can be
            used for reproducibility of metafeatures. a generated seed can be
            accessed through the 'seed' property, after calling this method.
        n_folds: int, the number of cross validation folds used by the
            landmarking metafeatures. also affects the sample_shape validation
        verbose: bool, default False. When True, prints the ID of each
            metafeature right before it is about to be computed.
        timeout: float, default None. If timeout is None, compute_metafeatures
            will be run to completion. Otherwise, execution will halt after
            approximately timeout seconds. Any metafeatures that have not been
            computed will be labeled 'TIMEOUT'.

        Returns
        -------
        A dictionary mapping the metafeature id to another dictionary containing
        the `value` and `compute_time` (if requested) of the referencing
        metafeature. The value is typically a number, but can be a string
        indicating a reason why the value could not be computed.
        """
        start_time = time.time()
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
            n_folds, verbose
        )
        if timeout is None:
            def check_time():
                pass
        else:
            def check_time():
                if time.time() - start_time > timeout:
                    raise TimeoutError()
        self._check_timeout = check_time

        if column_types is None:
            column_types = self._infer_column_types(X, Y)
        if metafeature_ids is None:
            metafeature_ids = self._get_metafeature_ids(exclude)
            exclude = None
        if sample_shape is None:
            sample_shape = (None, None)
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self._validate_compute_arguments(
            X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
            n_folds, verbose
        )

        self._init_resources(
            X, Y, column_types, sample_shape, seed, n_folds
        )

        computed_metafeatures = {name: {self.VALUE_KEY: self.TIMEOUT, self.COMPUTE_TIME_KEY: 0}
                                 for name in metafeature_ids}
        try:
            for metafeature_id in metafeature_ids:
                self._check_timeout()
                if verbose:
                    print(metafeature_id)
                if self._resource_is_target_dependent(metafeature_id) and (
                    Y is None or column_types[Y.name] == self.NUMERIC
                ):
                    if Y is None:
                        value = self.NO_TARGETS
                    else:
                        value = self.NUMERIC_TARGETS
                    compute_time = None
                else:
                    value, compute_time = self._get_resource(metafeature_id)

                computed_metafeatures[metafeature_id] = {
                    self.VALUE_KEY: value,
                    self.COMPUTE_TIME_KEY: compute_time
                }
        except TimeoutError:
            pass

        return computed_metafeatures

    def _init_resources(
        self, X, Y, column_types, sample_shape, seed, n_folds
    ):
        self._resources = {
            "X_raw": {
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
            "column_types": {
                self.VALUE_KEY: column_types,
                self.COMPUTE_TIME_KEY: 0.
            },
            "sample_shape": {
                self.VALUE_KEY: sample_shape,
                self.COMPUTE_TIME_KEY: 0.
            },
            "seed_base": {
                self.VALUE_KEY: seed,
                self.COMPUTE_TIME_KEY: 0.
            },
            "n_folds": {
                self.VALUE_KEY: n_folds,
                self.COMPUTE_TIME_KEY: 0.
            }
        }

    @classmethod
    def _resource_is_target_dependent(cls, resource_id):
        if resource_id=='Y':
            return True
        elif resource_id=='XSample':
            return False
        else:
            resource_info = cls._resources_info[resource_id]
            function = resource_info["function"]
            args = resource_info["arguments"]
            for parameter, argument in args.items():
                if (argument in cls._resources_info and
                    cls._resource_is_target_dependent(argument)
                ):
                    return True
            return False

    def _get_cv_seed(self, seed_base, seed_offset):
        return (seed_base + seed_offset,)

    def _validate_compute_arguments(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        for f in [
            self._validate_X, self._validate_Y, self._validate_column_types,
            self._validate_metafeature_ids, self._validate_sample_shape,
            self._validate_n_folds, self._validate_verbose
        ]:
            f(
                X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
                n_folds, verbose
            )

    def _validate_X(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pandas.DataFrame')
        if X.empty:
            raise ValueError('X must not be empty')

    def _validate_Y(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        if not isinstance(Y, pd.Series) and not Y is None:
            raise TypeError('Y must be of type pandas.Series')
        if Y is not None and Y.shape[0] != X.shape[0]:
            raise ValueError('Y must have the same number of rows as X')

    def _validate_column_types(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        if not column_types is None:
            invalid_column_types = {}
            columns = list(X.columns)
            if not Y is None:
                columns.append(Y.name)
            for col in columns:
                if col not in column_types:
                    raise ValueError(
                        f"Column type not specified for column {col}"
                    )
                col_type = column_types[col]
                # todo: add self.TEXT to check. Additionally add self.TEXT to all tests that check for column types
                if not col_type in [self.NUMERIC, self.CATEGORICAL, self.TEXT]:
                    invalid_column_types[col] = col_type
            if len(invalid_column_types) > 0:
                raise ValueError(
                    f"Invalid column types: {invalid_column_types}. Valid types " +
                    f"include {self.NUMERIC} and {self.CATEGORICAL} and {self.TEXT}."
                )

    def _validate_metafeature_ids(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        ids = None
        if metafeature_ids is not None and exclude is not None:
            raise ValueError("metafeature_ids and exclude cannot both be non-null")
        elif metafeature_ids is not None:
            ids = metafeature_ids
            list_label = 'requested'
        elif exclude is not None:
            ids = exclude
            list_label = 'excluded'
        if ids is not None:
            invalid_metafeature_ids = [
                mf for mf in ids if mf not in self._resources_info
            ]
            if len(invalid_metafeature_ids) > 0:
                raise ValueError(
                    'One or more {} metafeatures are not valid: {}'.
                    format(list_label, invalid_metafeature_ids)
                )

    def _validate_sample_shape(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
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
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        if not dtype_is_numeric(type(n_folds)) or (n_folds != int(n_folds)):
            raise ValueError(f"`n_folds` must be an integer, not {n_folds}")
        if n_folds < 2:
            raise ValueError(f"`n_folds` must be >= 2, but was {n_folds}")
        if (Y is not None and 
            column_types is not None and 
            column_types[Y.name] != self.NUMERIC and 
            metafeature_ids is not None):
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

    def _validate_verbose(
        self, X, Y, column_types, metafeature_ids, exclude, sample_shape, seed,
        n_folds, verbose
    ):
        if not type(verbose) is bool:
            raise ValueError("`verbose` must be of type bool.")

    # todo: intelligently infer TEXT data type
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
                # todo: get number of unique values in col_name, compute unique/total ratio. Use ratio to infer type

                column_types[Y.name] = self.CATEGORICAL
        return column_types

    def _get_metafeature_ids(self, exclude):
        if exclude is not None:
            return [mf for mf in self.list_metafeatures() if mf not in exclude]
        else:
            return self.list_metafeatures()

    def _get_resource(self, resource_id):
        self._check_timeout()
        if not resource_id in self._resources:
            resource_info = self._resources_info[resource_id]
            f_name = resource_info["function"]
            f = self._get_function(f_name)
            args, total_time = self._get_arguments(resource_id)
            return_resources = resource_info["returns"]
            start_timestamp = time.perf_counter()
            computed_resources = f(**args)
            compute_time = time.perf_counter() - start_timestamp
            total_time += compute_time
            for res_id, computed_resource in zip(
                return_resources, computed_resources
            ):
                self._resources[res_id] = {
                    self.VALUE_KEY: computed_resource,
                    self.COMPUTE_TIME_KEY: total_time
                }
        resource = self._resources[resource_id]
        return resource[self.VALUE_KEY], resource[self.COMPUTE_TIME_KEY]

    def _get_function(self, f_name):
        if f_name.startswith("self."):
            return getattr(self, f_name[len("self."):])
        else:
            return globals()[f_name]

    def _get_arguments(self, resource_id):
        resource_info = self._resources_info[resource_id]
        args = resource_info["arguments"]
        resolved_parameters = {}
        total_time = 0.0
        for parameter, argument in args.items():
            argument_type = type(argument)
            if parameter == "seed":
                seed_base, compute_time = self._get_resource("seed_base")
                argument += seed_base
            elif argument_type is str:
                if argument in self._resources_info:
                    argument, compute_time = self._get_resource(argument)
                else:
                    compute_time = 0
            elif dtype_is_numeric(argument_type):
                compute_time = 0
            else:
                raise Exception("unhandled argument type")
            resolved_parameters[parameter] = argument
            total_time += compute_time
        return (resolved_parameters, total_time)

    def _get_preprocessed_data(self, X_sample, X_sampled_columns, column_types, seed):
        series_array = []
        for feature in X_sample.columns:
            is_text = False
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
            elif column_types[feature_series.name] == self.TEXT:
                is_text = True
            if not is_text:
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

    def _sample_rows(self, X, Y, sample_shape, seed):
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

    def _get_text_features_with_no_missing_values(
				self, X_sample, column_types
		):
        text_features_with_no_missing_values = []
        for feature in X_sample.columns:
            if column_types[feature] == self.TEXT:
                no_nan_series = X_sample[feature].dropna(
					axis=0, how='any'
				)
                text_features_with_no_missing_values.append(
					no_nan_series
				)
        return (text_features_with_no_missing_values,)

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
                # renaming avoids name collisions and problems when y does not have a name
                df = pd.concat([X_sample[feature].rename('x'), Y_sample.rename('y')], axis=1)
                df.dropna(axis=0, how='any', inplace=True)
                numeric_features_and_class_with_no_missing_values.append(
                    (df['x'],df['y'])
                )
        binned_feature_class_array = [
            (
                pd.cut(feature_class_pair[0],
                round(feature_class_pair[0].shape[0]**(1./3.))),
                feature_class_pair[1]
            ) for feature_class_pair in numeric_features_and_class_with_no_missing_values
        ]
        return (binned_feature_class_array,)
