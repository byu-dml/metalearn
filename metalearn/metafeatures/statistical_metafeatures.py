import time
import math
import itertools

import numpy as np
import pandas as pd
# from scipy.stats import skew, kurtosis
from sklearn.cross_decomposition import CCA

from .metafeatures_base import MetafeaturesBase


class StatisticalMetafeatures(MetafeaturesBase):

    def __init__(self):
        
        function_dict = {
            'MeanMeansOfNumericFeatures': self._get_numeric_means,
            'StdevMeansOfNumericFeatures': self._get_numeric_means,
            'MinMeansOfNumericFeatures': self._get_numeric_means,
            'MaxMeansOfNumericFeatures': self._get_numeric_means,
            'Quartile1MeansOfNumericFeatures': self._get_numeric_means,
            'Quartile2MeansOfNumericFeatures': self._get_numeric_means,
            'Quartile3MeansOfNumericFeatures': self._get_numeric_means,
            'MeanStdDevOfNumericFeatures': self._get_numeric_stdev,
            'StdevStdDevOfNumericFeatures': self._get_numeric_stdev,
            'MinStdDevOfNumericFeatures': self._get_numeric_stdev,
            'MaxStdDevOfNumericFeatures': self._get_numeric_stdev,
            'Quartile1StdDevOfNumericFeatures': self._get_numeric_stdev,
            'Quartile2StdDevOfNumericFeatures': self._get_numeric_stdev,
            'Quartile3StdDevOfNumericFeatures': self._get_numeric_stdev,
            'MeanSkewnessOfNumericFeatures': self._get_numeric_skewness,
            'StdevSkewnessOfNumericFeatures': self._get_numeric_skewness,
            'MinSkewnessOfNumericFeatures': self._get_numeric_skewness,
            'MaxSkewnessOfNumericFeatures': self._get_numeric_skewness,
            'Quartile1SkewnessOfNumericFeatures': self._get_numeric_skewness,
            'Quartile2SkewnessOfNumericFeatures': self._get_numeric_skewness,
            'Quartile3SkewnessOfNumericFeatures': self._get_numeric_skewness,
            'MeanKurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'StdevKurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'MinKurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'MaxKurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'Quartile1KurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'Quartile2KurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'Quartile3KurtosisOfNumericFeatures': self._get_numeric_kurtosis,
            'MeanCanonicalCorrelation': self._get_correlations,
            'StdevCanonicalCorrelation': self._get_correlations,
            'MeanCanonicalCorrelationOfFeaturesSplitByClass': self._get_correlations_by_class,
            'StdevCanonicalCorrelationOfFeaturesSplitByClass': self._get_correlations_by_class
        }

        dependencies_dict = {
            'MeanMeansOfNumericFeatures': [],
            'StdevMeansOfNumericFeatures': [],
            'MinMeansOfNumericFeatures': [],
            'MaxMeansOfNumericFeatures': [],
            'Quartile1MeansOfNumericFeatures': [],
            'Quartile2MeansOfNumericFeatures': [],
            'Quartile3MeansOfNumericFeatures': [],
            'MeanStdDevOfNumericFeatures': [],
            'StdevStdDevOfNumericFeatures': [],
            'MinStdDevOfNumericFeatures': [],
            'MaxStdDevOfNumericFeatures': [],
            'Quartile1StdDevOfNumericFeatures': [],
            'Quartile2StdDevOfNumericFeatures': [],
            'Quartile3StdDevOfNumericFeatures': [],
            'MeanSkewnessOfNumericFeatures': [],
            'StdevSkewnessOfNumericFeatures': [],
            'MinSkewnessOfNumericFeatures': [],
            'MaxSkewnessOfNumericFeatures': [],
            'Quartile1SkewnessOfNumericFeatures': [],
            'Quartile2SkewnessOfNumericFeatures': [],
            'Quartile3SkewnessOfNumericFeatures': [],
            'MeanKurtosisOfNumericFeatures': [],
            'StdevKurtosisOfNumericFeatures': [],
            'MinKurtosisOfNumericFeatures': [],
            'MaxKurtosisOfNumericFeatures': [],
            'Quartile1KurtosisOfNumericFeatures': [],
            'Quartile2KurtosisOfNumericFeatures': [],
            'Quartile3KurtosisOfNumericFeatures': [],
            'MeanCanonicalCorrelation': [],
            'StdevCanonicalCorrelation': [],
            'MeanCanonicalCorrelationOfFeaturesSplitByClass': [],
            'StdevCanonicalCorrelationOfFeaturesSplitByClass': []          
        }

        super().__init__(function_dict, dependencies_dict)    

    def _get_numeric_means(self, X, Y):
        numeric_features = self._get_numeric_features(X)
        means = [X[feature].dropna(axis=0, how="any").mean() for feature in numeric_features]
        return self._profile_distribution(means, 'MeansOfNumericFeatures')

    def _get_numeric_stdev(self, X, Y):
        numeric_features = self._get_numeric_features(X)
        stdevs = [X[feature].dropna(axis=0, how="any").std() for feature in numeric_features]
        return self._profile_distribution(stdevs, 'StdDevOfNumericFeatures')

    def _get_numeric_skewness(self, X, Y):
        numeric_features = self._get_numeric_features(X)
        skews = [X[feature].dropna(axis=0, how="any").skew() for feature in numeric_features]
        return self._profile_distribution(skews, 'SkewnessOfNumericFeatures')

    def _get_numeric_kurtosis(self, X, Y):
        numeric_features = self._get_numeric_features(X)
        kurtosis = [X[feature].dropna(axis=0, how="any").kurtosis() for feature in numeric_features]
        return self._profile_distribution(kurtosis, 'KurtosisOfNumericFeatures')

    def _get_correlations(self, X, Y):
        correlations = self._get_canonical_correlations(X)        
        values_dict = self._profile_distribution(correlations, 'CanonicalCorrelation')
        return {
            'MeanCanonicalCorrelation': values_dict['MeanCanonicalCorrelation'],
            'StdevCanonicalCorrelation': values_dict['StdevCanonicalCorrelation']
        }

    def _get_correlations_by_class(self, X, Y):
        correlations = []
        XY = pd.concat([X,Y], axis=1)
        XY_grouped_by_class = XY.groupby(self.target_name)
        for label in Y.unique():                        
            group = XY_grouped_by_class.get_group(label).drop(self.target_name, axis=1)            
            correlations.extend(self._get_canonical_correlations(group))
        values_dict = self._profile_distribution(correlations, 'CanonicalCorrelationOfFeaturesSplitByClass')
        return {
            'MeanCanonicalCorrelationOfFeaturesSplitByClass': values_dict['MeanCanonicalCorrelationOfFeaturesSplitByClass'],
            'StdevCanonicalCorrelationOfFeaturesSplitByClass': values_dict['StdevCanonicalCorrelationOfFeaturesSplitByClass']
        }

    def _get_canonical_correlations(self, dataframe):
        '''
        computes the correlation coefficient between each distinct pairing of columns
        preprocessing note:
            any rows with missing values (in either paired column) are dropped for that pairing
            nominal columns are replaced with one-hot encoded columns
            any columns which have only one distinct value (after dropping missing values) are skipped
        returns a list of the pairwise canonical correlation coefficients
        '''

        def preprocess(series):
            array = series.as_matrix()
            if "int" not in str(series.dtype) and "float" not in str(series.dtype):
                array = self._replace_nominal_column(array)
            return array.astype(float).reshape(series.shape[0], -1)

        correlations = []
        skip_cols = set()
        for col_name_i, col_name_j in itertools.combinations(dataframe.columns, 2):
            if col_name_i in skip_cols or col_name_j in skip_cols:
                continue

            df_ij = dataframe[[col_name_i, col_name_j]].dropna(axis=0, how="any")
            col_i = preprocess(df_ij[col_name_i])
            col_j = preprocess(df_ij[col_name_j])

            if np.unique(col_i).shape[0] <= 1:
                skip_cols.add(col_name_i)
                continue
            if np.unique(col_j).shape[0] <= 1:
                skip_cols.add(col_name_j)
                continue
            cca = CCA(n_components=1).fit(col_i,col_j)
            c = cca.score(col_i, col_j)
            correlations.append(c)

        return correlations

    def _get_abs_cor(self, data, attributes):
        numAtt = len(data[0]) - 1
        if (numAtt > 1):
            classes = attributes[-1][1]
            sums = 0.0
            n = 0.0
            for label in classes:            
                for i in range(numAtt):
                    col_i_data_by_class = get_column_of_class(data, i, label)
                    if (not is_numeric(attributes[i])):
                        col_i_data_by_class = replace_nominal_column(col_i_data_by_class)
                    else:
                        col_i_data_by_class = col_i_data_by_class.reshape(col_i_data_by_class.shape[0], 1)
                    for j in range(numAtt):
                        col_j_data_by_class = get_column_of_class(data, j, label)
                        if (not is_numeric(attributes[j])):
                            col_j_data_by_class = replace_nominal_column(col_j_data_by_class)
                        else:
                            col_j_data_by_class = col_j_data_by_class.reshape(col_j_data_by_class.shape[0], 1)
                        cca = CCA(kernelcca = False, reg = 0., numCC = 1, verbose=False)
                        try:                        
                            cca.train([col_i_data_by_class.astype(float), col_j_data_by_class.astype(float)])                        
                            c = cca.cancorrs[0]
                        except:
                            continue
                        if (c):
                            sums += abs(c)
                            n += 1            
            if (n != 0):
                return sums / n
        return 0.0
