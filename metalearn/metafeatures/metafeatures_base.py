import time
import numpy as np
import pandas as pd
from pandas import DataFrame

class MetafeaturesBase(object):

    def __init__(self, function_dict=None, dependencies_dict=None):

        if dependencies_dict is None or function_dict is None:
            raise NotImplementedError("Child class must create and pass a dependencies_dict and a function_dict")

        self.dependencies_dict = dependencies_dict
        self.function_dict = function_dict
        self.metafeature_dict = {}
        self.target_name = 'target'
        self.value_name = 'value'
        self.time_name = 'time'

    def compute(self, dataframe: DataFrame, metafeatures: list = None) -> DataFrame:
        """
        Parameters
        ----------
        dataframe: The data for metafeatures to be computed on. The targets are contained in a column named 'target'
        metafeatures: A list of strings (metafeature names) to be computed
        Returns
        -------
        A dataframe containing one row and twice as many columns as requested metafeatures because <metafeature>_time columns will also be included
        """
        dataframe = dataframe.dropna(axis=1, how="all")
        X = dataframe.drop(self.target_name, axis=1)
        Y = dataframe[self.target_name]
        if metafeatures is None:
            metafeatures = self.list_metafeatures()
        return self._retrieve_metafeatures(metafeatures, X, Y)

    def list_metafeatures(self):
        return list(self.function_dict.keys())

    def _retrieve_metafeatures(self, metafeatures, X, Y):
        metafeature_frame = pd.DataFrame()
        for metafeature_name in metafeatures:
            value, time_value = self._retrieve_metafeature(metafeature_name, X, Y)
            metafeature_frame.at[0,metafeature_name] = value
            metafeature_frame.at[0,metafeature_name + '_Time'] = time_value
        return metafeature_frame

    def _retrieve_metafeature(self, metafeature_name, X, Y):
        total_time = 0.0
        dependencies = []
        dependency_metafeatures = self.dependencies_dict[metafeature_name]
        for dependency_metafeature in dependency_metafeatures:
            value, time_value = self._retrieve_metafeature(dependency_metafeature, X, Y)
            dependencies.append(value)
            total_time += time_value
        if not metafeature_name in self.metafeature_dict:
            if np.nan in dependencies:
                values = {metafeature_name: np.nan}
                total_time = np.nan
            else:
                start = time.time()
                values = self.function_dict[metafeature_name](X, Y, *dependencies)
                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time
            for key in values.keys():
                if values[key] == np.nan:
                    self.metafeature_dict[key] = {
                        self.value_name: values[key],
                        self.time_name: np.nan
                    }
                else:
                    self.metafeature_dict[key] = {
                        self.value_name: values[key],
                        self.time_name: total_time
                    }
        value = self.metafeature_dict[metafeature_name][self.value_name]
        total_time = self.metafeature_dict[metafeature_name][self.time_name]
        return (value, total_time)

    def _profile_distribution(self, data, label):
        """
        Compute the min, max, mean, and standard deviation of a vector

        Parameters
        ----------
        data: array of real values
        label: string with which the data will be associated in the returned dictionary

        Returns
        -------
        features = dictionary containing the min, max, mean, and standard deviation
        """
        # todo replace with pd.describe
        if len(data) == 0:
            return {
                'Min' + label: np.nan,
                'Max' + label: np.nan,
                'Mean' + label: np.nan,
                'Quartile1' + label: np.nan,
                'Quartile2' + label: np.nan,
                'Quartile3' + label: np.nan,
                'Stdev' + label: np.nan
            }

        features = {}

        features['Min' + label] = np.amin(data)
        features['Max' + label] = np.amax(data)
        features['Mean' + label] = np.mean(data)
        features['Quartile1' + label] = np.percentile(data, 0.25)
        features['Quartile2' + label] = np.percentile(data, 0.5)
        features['Quartile3' + label] = np.percentile(data, 0.75)

        ddof = 1 if len(data) > 1 else 0
        features['Stdev' + label] = np.std(data, axis = 0, ddof = ddof)

        return features

    def _get_numeric_features(self, dataframe):
        """
        Gets the names of the numeric attributes in the data.
        """
        numeric_columns = []
        for col_name, col_type in zip(dataframe.columns, dataframe.dtypes):
            if "int" in str(col_type) or "float" in str(col_type):
                numeric_columns.append(col_name)
        return numeric_columns

    def _preprocess_data(self, dataframe):
        series_array = []
        for feature in dataframe.columns:
            feature_series = dataframe[feature]
            col = feature_series.as_matrix()
            dropped_nan_series = feature_series.dropna(axis=0,how='any')
            num_nan = col.shape[0] - dropped_nan_series.shape[0]
            col[feature_series.isnull()] = np.random.choice(dropped_nan_series, num_nan)
            if not self._dtype_is_numeric(feature_series.dtype):
                feature_series = pd.get_dummies(feature_series)
            series_array.append(feature_series)
        preprocessed_dataframe = pd.concat(series_array, axis=1, copy=False)
        return preprocessed_dataframe

    def _dtype_is_numeric(self, dtype):
        return "int" in str(dtype) or "float" in str(dtype)