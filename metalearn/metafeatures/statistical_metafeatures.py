import itertools

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from metalearn.metafeatures.common_operations import profile_distribution
from metalearn.metafeatures.base import build_resources_info, MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup
import metalearn.metafeatures.constants as consts


def get_numeric_means(numeric_features_array):
    means = [feature.mean() for feature in numeric_features_array]
    return profile_distribution(means)

get_numeric_means = MetafeatureComputer(
    get_numeric_means,
    [
        "MeanMeansOfNumericFeatures",
        "StdevMeansOfNumericFeatures",
        "SkewMeansOfNumericFeatures",
        "KurtosisMeansOfNumericFeatures",
        "MinMeansOfNumericFeatures",
        "Quartile1MeansOfNumericFeatures",
        "Quartile2MeansOfNumericFeatures",
        "Quartile3MeansOfNumericFeatures",
        "MaxMeansOfNumericFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.STATISTICAL],
    {
        "numeric_features_array": "NoNaNNumericFeatures"
    }
)


def get_numeric_stdev(numeric_features_array):
    stdevs = [feature.std() for feature in numeric_features_array]
    return profile_distribution(stdevs)

get_numeric_stdev = MetafeatureComputer(
    get_numeric_stdev,
    [
        "MeanStdDevOfNumericFeatures",
        "StdevStdDevOfNumericFeatures",
        "SkewStdDevOfNumericFeatures",
        "KurtosisStdDevOfNumericFeatures",
        "MinStdDevOfNumericFeatures",
        "Quartile1StdDevOfNumericFeatures",
        "Quartile2StdDevOfNumericFeatures",
        "Quartile3StdDevOfNumericFeatures",
        "MaxStdDevOfNumericFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.STATISTICAL],
    {
        "numeric_features_array": "NoNaNNumericFeatures"
    }
)


def get_numeric_skewness(numeric_features_array):
    skews = [feature.skew() for feature in numeric_features_array]
    return profile_distribution(skews)

get_numeric_skewness = MetafeatureComputer(
    get_numeric_skewness,
    [
        "MeanSkewnessOfNumericFeatures",
        "StdevSkewnessOfNumericFeatures",
        "SkewSkewnessOfNumericFeatures",
        "KurtosisSkewnessOfNumericFeatures",
        "MinSkewnessOfNumericFeatures",
        "Quartile1SkewnessOfNumericFeatures",
        "Quartile2SkewnessOfNumericFeatures",
        "Quartile3SkewnessOfNumericFeatures",
        "MaxSkewnessOfNumericFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.STATISTICAL],
    {
        "numeric_features_array": "NoNaNNumericFeatures"
    }
)


def get_numeric_kurtosis(numeric_features_array):
    kurtoses = [feature.kurtosis() for feature in numeric_features_array]
    return profile_distribution(kurtoses)

get_numeric_kurtosis = MetafeatureComputer(
    get_numeric_kurtosis,
    [
        "MeanKurtosisOfNumericFeatures",
        "StdevKurtosisOfNumericFeatures",
        "SkewKurtosisOfNumericFeatures",
        "KurtosisKurtosisOfNumericFeatures",
        "MinKurtosisOfNumericFeatures",
        "Quartile1KurtosisOfNumericFeatures",
        "Quartile2KurtosisOfNumericFeatures",
        "Quartile3KurtosisOfNumericFeatures",
        "MaxKurtosisOfNumericFeatures"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.STATISTICAL],
    {
        "numeric_features_array": "NoNaNNumericFeatures"
    }
)


def get_pca(X_preprocessed):
    num_components = min(3, X_preprocessed.shape[1])
    pca_data = PCA(n_components=num_components)
    pca_data.fit_transform(X_preprocessed.values)
    pred_pca = pca_data.explained_variance_ratio_
    pred_eigen = pca_data.explained_variance_
    pred_det = np.linalg.det(pca_data.get_covariance())
    variance_percentages = [0] * 3
    for i in range(len(pred_pca)):
        variance_percentages[i] = pred_pca[i]
    eigenvalues = [0] * 3
    for i in range(len(pred_eigen)):
        eigenvalues[i] = pred_eigen[i]
    return (variance_percentages[0], variance_percentages[1], variance_percentages[2], eigenvalues[0], eigenvalues[1], eigenvalues[2], pred_det)

get_pca = MetafeatureComputer(
    get_pca,
    [
        "PredPCA1",
        "PredPCA2",
        "PredPCA3",
        "PredEigen1",
        "PredEigen2",
        "PredEigen3",
        "PredDet"
    ],
    ProblemType.ANY,
    [MetafeatureGroup.STATISTICAL],
    {
        "X_preprocessed": "XPreprocessed"
    }
)


def get_correlations(X_sample, column_types):
    correlations = get_canonical_correlations(X_sample, column_types)
    profile_distribution(correlations)

def get_correlations_by_class(X_sample, Y_sample):
    correlations = []
    XY = pd.concat([X_sample,Y_sample], axis=1)
    XY_grouped_by_class = XY.groupby(Y_sample.name)
    for label in Y_sample.unique():
        group = XY_grouped_by_class.get_group(label).drop(Y_sample.name, axis=1)
        correlations.extend(get_canonical_correlations(group))
    return profile_distribution(correlations)

def get_canonical_correlations(dataframe, column_types):
    '''
    computes the correlation coefficient between each distinct pairing of columns
    preprocessing note:
        any rows with missing values (in either paired column) are dropped for that pairing
        categorical columns are replaced with one-hot encoded columns
        any columns which have only one distinct value (after dropping missing values) are skipped
    returns a list of the pairwise canonical correlation coefficients
    '''

    def preprocess(series):
        if column_types[series.name] == consts.CATEGORICAL:
            series = pd.get_dummies(series)
        array = series.values.reshape(series.shape[0], -1)
        return array

    if dataframe.shape[1] < 2:
        return []

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


"""
A list of all MetafeatureComputer
instances in this module.
"""
metafeatures_info = build_resources_info(
    get_numeric_means,
    get_numeric_stdev,
    get_numeric_skewness,
    get_numeric_kurtosis,
    get_pca
)
