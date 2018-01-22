import time
import math

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from .metafeatures_base import MetafeaturesBase
from .rcca import CCA

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
        means = []
        numeric_features = self._get_numeric_features(X)
        for feature in numeric_features:
            means.append(np.mean(X[feature].as_matrix()))
        values_dict = self._profile_distribution(means, 'MeansOfNumericFeatures')        
        return {
            'MeanMeansOfNumericFeatures': values_dict['MeanMeansOfNumericFeatures'],
            'StdevMeansOfNumericFeatures': values_dict['StdevMeansOfNumericFeatures'],
            'MinMeansOfNumericFeatures': values_dict['MinMeansOfNumericFeatures'],
            'MaxMeansOfNumericFeatures': values_dict['MaxMeansOfNumericFeatures'],
            'Quartile1MeansOfNumericFeatures': values_dict['Quartile1MeansOfNumericFeatures'],
            'Quartile2MeansOfNumericFeatures': values_dict['Quartile2MeansOfNumericFeatures'],
            'Quartile3MeansOfNumericFeatures': values_dict['Quartile3MeansOfNumericFeatures']
        }

    def _get_numeric_stdev(self, X, Y):            
        stdevs = []
        numeric_features = self._get_numeric_features(X)
        for feature in numeric_features:
            stdevs.append(np.std(X[feature].as_matrix()))
        values_dict = self._profile_distribution(stdevs, 'StdDevOfNumericFeatures')
        return {
            'MeanStdDevOfNumericFeatures': values_dict['MeanStdDevOfNumericFeatures'],
            'StdevStdDevOfNumericFeatures': values_dict['StdevStdDevOfNumericFeatures'],
            'MinStdDevOfNumericFeatures': values_dict['MinStdDevOfNumericFeatures'],
            'MaxStdDevOfNumericFeatures': values_dict['MaxStdDevOfNumericFeatures'],
            'Quartile1StdDevOfNumericFeatures': values_dict['Quartile1StdDevOfNumericFeatures'],
            'Quartile2StdDevOfNumericFeatures': values_dict['Quartile2StdDevOfNumericFeatures'],
            'Quartile3StdDevOfNumericFeatures': values_dict['Quartile3StdDevOfNumericFeatures']
        }

    def _get_numeric_skewness(self, X, Y): 
        # suppress errors brought about by code in the scipy skew function    
        np.seterr(divide='ignore', invalid='ignore')            
        skw = []
        numeric_features = self._get_numeric_features(X)
        for feature in numeric_features:
            v = skew(X[feature].as_matrix())
            if ((v != None) and (not math.isnan(v))):
                skw.append(v)                
        values_dict = self._profile_distribution(skw, 'SkewnessOfNumericFeatures')
        return {
            'MeanSkewnessOfNumericFeatures': values_dict['MeanSkewnessOfNumericFeatures'],
            'StdevSkewnessOfNumericFeatures': values_dict['StdevSkewnessOfNumericFeatures'],
            'MinSkewnessOfNumericFeatures': values_dict['MinSkewnessOfNumericFeatures'],
            'MaxSkewnessOfNumericFeatures': values_dict['MaxSkewnessOfNumericFeatures'],
            'Quartile1SkewnessOfNumericFeatures': values_dict['Quartile1SkewnessOfNumericFeatures'],
            'Quartile2SkewnessOfNumericFeatures': values_dict['Quartile2SkewnessOfNumericFeatures'],
            'Quartile3SkewnessOfNumericFeatures': values_dict['Quartile3SkewnessOfNumericFeatures']
        }            

    def _get_numeric_kurtosis(self, X, Y):
        # suppress errors brought about by code in the scipy kurtosis function    
        np.seterr(divide='ignore', invalid='ignore')            
        kurt = []
        numeric_features = self._get_numeric_features(X)
        for feature in numeric_features:
            v = kurtosis(X[feature].as_matrix(), fisher = False)
            if ((v != None) and (not math.isnan(v))):
                kurt.append(v)
        values_dict = self._profile_distribution(kurt, 'KurtosisOfNumericFeatures')
        return {
            'MeanKurtosisOfNumericFeatures': values_dict['MeanKurtosisOfNumericFeatures'],
            'StdevKurtosisOfNumericFeatures': values_dict['StdevKurtosisOfNumericFeatures'],
            'MinKurtosisOfNumericFeatures': values_dict['MinKurtosisOfNumericFeatures'],
            'MaxKurtosisOfNumericFeatures': values_dict['MaxKurtosisOfNumericFeatures'],
            'Quartile1KurtosisOfNumericFeatures': values_dict['Quartile1KurtosisOfNumericFeatures'],
            'Quartile2KurtosisOfNumericFeatures': values_dict['Quartile2KurtosisOfNumericFeatures'],
            'Quartile3KurtosisOfNumericFeatures': values_dict['Quartile3KurtosisOfNumericFeatures']
        }

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
        correlations = []
        nominal_features = self._get_nominal_features(dataframe)
        for feature_i in dataframe.columns:
            col_i = dataframe[feature_i].as_matrix()
            if feature_i in nominal_features:                
                col_i = self._replace_nominal_column(col_i)
            col_i = col_i.reshape(col_i.shape[0], -1)
            for feature_j in dataframe.columns:                
                if feature_i != feature_j:                    
                    col_j = dataframe[feature_j].as_matrix()
                    if feature_j in nominal_features:
                        col_j = self._replace_nominal_column(col_j)
                    col_j = col_j.reshape(col_j.shape[0], -1)
                    cca = CCA(kernelcca = False, reg = 0., numCC = 1, verbose=False)                    
                    try:                        
                        cca.train([col_i.astype(float), col_j.astype(float)])                        
                        c = cca.cancorrs[0]
                    except:
                        continue                
                    if c:
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
