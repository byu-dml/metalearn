import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
        type_dict = dataframe.columns.to_series().groupby(dataframe.dtypes).groups
        for t in type_dict.keys():
            if ('int' in str(t) or 'float' in str(t)):
                numeric_columns.extend(list(type_dict[t]))
        return numeric_columns

    def _get_nominal_features(self, dataframe):
        """
        Gets the names of the nominal attributes in the data.
        """
        nominal_columns = []
        for col_name, col_type in zip(dataframe.columns, dataframe.dtypes):
            if not("int" in str(col_type) or "float" in str(col_type)):
                nominal_columns.append(col_name)
        return nominal_columns

    def _replace_nominal_column(self, col):
        """
        Returns a One Hot Encoded ndarray of col
        """
        labelledCol = LabelEncoder().fit_transform(col)
        labelledCol = labelledCol.reshape(labelledCol.shape[0],1)
        return OneHotEncoder().fit_transform(labelledCol).toarray()
