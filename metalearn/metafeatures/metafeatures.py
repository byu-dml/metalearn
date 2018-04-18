import os
import math
import json
import time
import multiprocessing

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .common_operations import *
from .simple_metafeatures import *
from .statistical_metafeatures import *
from .information_theoretic_metafeatures import *
from .landmarking_metafeatures import *


class Metafeatures(object):

    def __init__(self):
        self.dependencies_dict = {}
        self.function_dict = {}
        self.resource_results_dict = {}
        self.resource_info_dict = {}
        self.metafeatures_list = []
        self.target_name = 'target'
        self.value_name = 'value'
        self.time_name = 'time'
        self.metafeatures_info_file_name = os.path.realpath(__file__).rsplit(".", 1)[0]+".json"
        self.load_dicts(self.metafeatures_info_file_name)

    def load_dicts(self, file_name):
        with open(file_name, 'r') as json_file:
            json_dict = json.load(json_file)
            self.function_dict = json_dict['functions']
            json_metafeatures_dict = json_dict['metafeatures']
            json_resources_dict = json_dict['resources']
            self.metafeatures_list = list(json_metafeatures_dict.keys())
            combined_dict = {**json_metafeatures_dict, **json_resources_dict}
            for key in combined_dict:
                self.resource_info_dict[key] = combined_dict[key]

    def compute(self, X: DataFrame, Y: Series, metafeature_ids: list = None, sample_rows=True, sample_columns=True, seed=42, timeout=None) -> DataFrame:
        """
        Parameters
        ----------
        X: pandas.DataFrame, the features used to compute metafeatures
        Y: pandas.Seris, the targets for the features used to compute metafeatures
        metafeature_ids: list, the metafeatures to compute. default of None indicates to compute all metafeatures
        sample_rows: bool, whether to uniformly sample from the rows
        sample_columns: bool, whether to uniformly sample from the columns
        seed: int, the seed used to generate psuedo-random numbers
        timeout: int, the maximum amount of wall time used to compute metafeatures

        Returns
        -------
        A dataframe containing one row and two columns for each metafeature: one for the value and one for the compute time of that metafeature value
        """
        self.computed_metafeatures = DataFrame()
        if timeout is None:
            self._compute(X, Y, metafeature_ids, sample_rows, sample_columns, seed, timeout)
        else:
            p = multiprocessing.Process(target=self._compute, args=(X, Y, metafeature_ids, sample_rows, sample_columns, seed, timeout))
            p.start()
            p.join(timeout-2)
            if p.is_alive():
                # Terminate
                p.terminate()
                p.join()
        return self.computed_metafeatures

    def _compute(self, X, Y, metafeature_ids, sample_rows, sample_columns, seed, timeout):
        self._validate_compute_arguments(X, Y, metafeature_ids, sample_rows, sample_columns, seed, timeout)

        if metafeature_ids is None:
            metafeature_ids = self.list_metafeatures()

        X_raw = X
        X = X_raw.dropna(axis=1, how="all")
        self.seed = seed
        self.resource_results_dict['XRaw'] = {self.value_name: X_raw, self.time_name: 0.}
        self.resource_results_dict['X'] = {self.value_name: X, self.time_name: 0.}
        self.resource_results_dict['Y'] = {self.value_name: Y, self.time_name: 0.}
        self.resource_results_dict['SampleRowsFlag'] = {self.value_name: sample_rows, self.time_name: 0.}
        self.resource_results_dict['SampleColumnsFlag'] = {self.value_name: sample_columns, self.time_name: 0.}

        self._compute_metafeatures(metafeature_ids)

    def _validate_compute_arguments(self, X, Y, metafeature_ids, sample_rows, sample_columns, seed, timeout):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be of type pandas.DataFrame")
        if not isinstance(Y, pd.Series):
            raise TypeError("Y must be of type pandas.Series")
        if metafeature_ids is not None:
            invalid_metafeature_ids = [mf for mf in metafeature_ids if mf not in self.resource_info_dict]
            if len(invalid_metafeature_ids) > 0:
                raise ValueError("One or more requested metafeatures are not valid: {}".format(invalid_metafeature_ids))

    def list_metafeatures(self):
        return self.metafeatures_list

    def _compute_metafeatures(self, metafeature_ids):
        for metafeature_id in metafeature_ids:
            value, time_value = self._retrieve_resource(metafeature_id)
            self.computed_metafeatures.at[0,metafeature_id] = value
            self.computed_metafeatures.at[0,metafeature_id+'_Time'] = time_value

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

    def _retrieve_resource(self, resource_name):
        if resource_name not in self.resource_results_dict:
            retrieved_parameters, total_time = self._retrieve_parameters(resource_name)
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
                self.resource_results_dict[result_name] = {self.value_name: result, self.time_name: total_time}
        value = self.resource_results_dict[resource_name][self.value_name]
        total_time = self.resource_results_dict[resource_name][self.time_name]
        return (value, total_time)

    def _get_random_state(self):
        return (self.seed+self.seed_offset,)

    def _get_preprocessed_data(self, X_sample, X_sampled_columns, seed=42):
        series_array = []
        for feature in X_sample.columns:
            feature_series = X_sample[feature]
            col = feature_series.as_matrix()
            dropped_nan_series = X_sampled_columns[feature].dropna(axis=0,how='any')
            num_nan = np.sum(feature_series.isnull())
            np.random.seed(seed)
            col[feature_series.isnull()] = np.random.choice(dropped_nan_series, num_nan)
            if not dtype_is_numeric(feature_series.dtype):
                feature_series = pd.get_dummies(feature_series)
            series_array.append(feature_series)
        return (pd.concat(series_array, axis=1, copy=False),)

    def _get_sample_of_columns(self, X, sample_columns, seed=42, max_columns=150):
        if sample_columns and X.shape[1] > max_columns:
            np.random.seed(seed)
            column_indices = np.random.permutation(X.shape[1])[:max_columns]
            columns = X.columns[column_indices]
            return (X[columns],)
        else:
            return (X,)

    def _get_sample_of_rows(self, X, Y, sample_rows, seed=42, approximate_max_rows=150000, min_row_per_class=2):
        if sample_rows == True and X.shape[0] > approximate_max_rows:
            samples = []
            total_rows = Y.shape[0]
            class_groupby = Y.groupby(Y)
            for group_key in class_groupby.groups:
                group = class_groupby.get_group(group_key).index
                num_to_sample = max(math.floor(float(group.shape[0]) / float(total_rows) * approximate_max_rows), min_row_per_class)
                np.random.seed(seed)
                row_indices = np.random.permutation(group)[:num_to_sample]
                samples.append(row_indices)
            row_indices = np.concatenate(samples)
            return (X.iloc[row_indices], Y.iloc[row_indices])
        else:
            return (X, Y)

    def _get_nominal_features_and_class_with_no_missing_values(self, X_sample, Y_sample):
        nominal_features_and_class_with_no_missing_values = []
        numeric_features = get_numeric_features(X_sample)
        for feature in X_sample.columns:
            if feature not in numeric_features:
                df = pd.concat([X_sample[feature],Y_sample], axis=1).dropna(axis=0, how="any")
                nominal_features_and_class_with_no_missing_values.append((df[feature],df[Y_sample.name]))
        return (nominal_features_and_class_with_no_missing_values,)

    def _get_numeric_features_and_class_with_no_missing_values(self, X_sample, Y_sample):
        numeric_features_and_class_with_no_missing_values = []
        numeric_features = get_numeric_features(X_sample)
        for feature in numeric_features:
            df = pd.concat([X_sample[feature],Y_sample], axis=1).dropna(axis=0, how="any")
            numeric_features_and_class_with_no_missing_values.append((df[feature],df[Y_sample.name]))
        return (numeric_features_and_class_with_no_missing_values,)

    def _get_binned_numeric_features_and_class_with_no_missing_values(self, numeric_features_class_array):
        binned_feature_class_array = [(pd.cut(feature_class_pair[0], round(feature_class_pair[0].shape[0]**(1./3.))), feature_class_pair[1]) for feature_class_pair in numeric_features_class_array]
        return (binned_feature_class_array,)
