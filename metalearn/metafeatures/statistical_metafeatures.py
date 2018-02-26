import time
import math
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from .common_operations import *

warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning) # suppress sklearn warnings

def get_numeric_means(numeric_features_class_array):
    means = [feature_class_pair[0].mean() for feature_class_pair in numeric_features_class_array]
    return profile_distribution(means)

def get_numeric_stdev(numeric_features_class_array):
    stdevs = [feature_class_pair[0].std() for feature_class_pair in numeric_features_class_array]
    return profile_distribution(stdevs)

def get_numeric_skewness(numeric_features_class_array):
    skews = [feature_class_pair[0].skew() for feature_class_pair in numeric_features_class_array]
    return profile_distribution(skews)

def get_numeric_kurtosis(numeric_features_class_array):
    kurtoses = [feature_class_pair[0].kurtosis() for feature_class_pair in numeric_features_class_array]
    return profile_distribution(kurtoses)

def get_pca(X_preprocessed):    
    num_components = min(3, X_preprocessed.shape[1])
    pca_data = PCA(n_components=num_components)
    pca_data.fit_transform(X_preprocessed.as_matrix())
    pred_pca = pca_data.explained_variance_ratio_
    pred_eigen = pca_data.explained_variance_
    pred_det = np.linalg.det(pca_data.get_covariance())
    variance_percentages = [np.nan] * 3    
    for i in range(len(pred_pca)):
        variance_percentages[i] = pred_pca[i]
    eigenvalues = [np.nan] * 3
    for i in range(len(pred_eigen)):
        eigenvalues[i] = pred_eigen[i]
    return (variance_percentages[0], variance_percentages[1], variance_percentages[2], eigenvalues[0], eigenvalues[1], eigenvalues[2], pred_det)

def get_correlations(X_sample):
    correlations = get_canonical_correlations(X_sample)
    mean_correlation, stdev_correlation, _, _, _, _, _ = profile_distribution(correlations)
    return (mean_correlation, stdev_correlation)

def get_correlations_by_class(X_sample, Y_sample):
    correlations = []
    XY = pd.concat([X_sample,Y_sample], axis=1)
    XY_grouped_by_class = XY.groupby(Y_sample.name)
    for label in Y_sample.unique():
        group = XY_grouped_by_class.get_group(label).drop(Y_sample.name, axis=1)
        correlations.extend(get_canonical_correlations(group))
    mean_correlation, stdev_correlation, _, _, _, _, _ = profile_distribution(correlations)
    return (mean_correlation, stdev_correlation)

def get_canonical_correlations(dataframe):
    '''
    computes the correlation coefficient between each distinct pairing of columns
    preprocessing note:
        any rows with missing values (in either paired column) are dropped for that pairing
        nominal columns are replaced with one-hot encoded columns
        any columns which have only one distinct value (after dropping missing values) are skipped
    returns a list of the pairwise canonical correlation coefficients
    '''

    def preprocess(series):
        if not dtype_is_numeric(series.dtype):
            series = pd.get_dummies(series)
        array = series.as_matrix().reshape(series.shape[0], -1)
        return array

    numeric_features = get_numeric_features(dataframe)
    if len(numeric_features) < 2:
        return []
    dataframe = dataframe[numeric_features]
    
    correlations = []
    skip_cols = set()
    for col_name_i, col_name_j in itertools.combinations(dataframe.columns, 2):
        if col_name_i in skip_cols or col_name_j in skip_cols:
            correlations.append(0)
            continue

        df_ij = dataframe[[col_name_i, col_name_j]].dropna(axis=0, how="any")
        col_i = df_ij[col_name_i]
        col_j = df_ij[col_name_j]

        if np.unique(col_i).shape[0] <= 1:
            skip_cols.add(col_name_i)
            correlations.append(0)
            continue
        if np.unique(col_j).shape[0] <= 1:
            skip_cols.add(col_name_j)
            correlations.append(0)
            continue

        col_i = preprocess(col_i)
        col_j = preprocess(col_j)

        col_i_c, col_j_c = CCA(n_components=1).fit_transform(col_i,col_j)

        if np.unique(col_i_c).shape[0] <= 1 or np.unique(col_j_c).shape[0] <= 1:
            c = 0
        else:
            c = np.corrcoef(col_i_c.T, col_j_c.T)[0,1]
        correlations.append(c)

    return correlations
