import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from metalearn.metafeatures.base import build_resources_info, ResourceComputer
import metalearn.metafeatures.constants as consts


def get_X(X_raw):
    return X_raw.dropna(axis=1, how="all"),

get_X = ResourceComputer(get_X, ["X"])

def get_cv_seed(seed_base, seed_offset):
    return (seed_base + seed_offset,)

get_cv_seed = ResourceComputer(get_cv_seed, ["cv_seed"], {'seed_offset': 1})


def sample_columns(X, sample_shape, seed):
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

sample_columns = ResourceComputer(
    sample_columns,
    ["XSampledColumns"],
    { "seed": 2 }
)


def sample_rows(X, Y, sample_shape, seed):
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

sample_rows = ResourceComputer(
    sample_rows,
    ["XSample","YSample"],
    { "X": "XSampledColumns", "seed": 3 }
)


def get_preprocessed_data(X_sample, X_sampled_columns, column_types, seed):
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
        if column_types[feature_series.name] == consts.CATEGORICAL:
            feature_series = pd.get_dummies(feature_series)
        elif column_types[feature_series.name] == consts.TEXT:
            is_text = True
        if not is_text:
            series_array.append(feature_series)
    return (pd.concat(series_array, axis=1, copy=False),)

get_preprocessed_data = ResourceComputer(
    get_preprocessed_data,
    ["XPreprocessed"],
    {
        "X_sample": "XSample",
        "X_sampled_columns": "XSampledColumns",
        "seed": 4
    }
)


def get_categorical_features_with_no_missing_values(
    X_sample, column_types
):
    categorical_features_with_no_missing_values = []
    for feature in X_sample.columns:
        if column_types[feature] == consts.CATEGORICAL:
            no_nan_series = X_sample[feature].dropna(
                axis=0, how='any'
            )
            categorical_features_with_no_missing_values.append(
                no_nan_series
            )
    return (categorical_features_with_no_missing_values,)

get_categorical_features_with_no_missing_values = ResourceComputer(
    get_categorical_features_with_no_missing_values,
    ["NoNaNCategoricalFeatures"],
    { "X_sample": "XSample" }
)


def get_categorical_features_and_class_with_no_missing_values(
    X_sample, Y_sample, column_types
):
    categorical_features_and_class_with_no_missing_values = []
    for feature in X_sample.columns:
        if column_types[feature] == consts.CATEGORICAL:
            df = pd.concat([X_sample[feature],Y_sample], axis=1).dropna(
                axis=0, how='any'
            )
            categorical_features_and_class_with_no_missing_values.append(
                (df[feature],df[Y_sample.name])
            )
    return (categorical_features_and_class_with_no_missing_values,)

get_categorical_features_and_class_with_no_missing_values = ResourceComputer(
    get_categorical_features_and_class_with_no_missing_values,
    ["NoNaNCategoricalFeaturesAndClass"],
    {
        "X_sample": "XSample",
        "Y_sample": "YSample"
    }
)


def get_numeric_features_with_no_missing_values(
    X_sample, column_types
):
    numeric_features_with_no_missing_values = []
    for feature in X_sample.columns:
        if column_types[feature] == consts.NUMERIC:
            no_nan_series = X_sample[feature].dropna(
                axis=0, how='any'
            )
            numeric_features_with_no_missing_values.append(
                no_nan_series
            )
    return (numeric_features_with_no_missing_values,)

get_numeric_features_with_no_missing_values = ResourceComputer(
    get_numeric_features_with_no_missing_values,
    ["NoNaNNumericFeatures"],
    { "X_sample": "XSample" }
)


def get_binned_numeric_features_with_no_missing_values(
    numeric_features_array
):
    binned_feature_array = [
        (
            pd.cut(feature,
            round(feature.shape[0]**(1./3.)))
        ) for feature in numeric_features_array
    ]
    return (binned_feature_array,)

get_binned_numeric_features_with_no_missing_values = ResourceComputer(
    get_binned_numeric_features_with_no_missing_values,
    ["NoNaNBinnedNumericFeatures"],
    { "numeric_features_array": "NoNaNNumericFeatures" }
)


def get_binned_numeric_features_and_class_with_no_missing_values(
    X_sample, Y_sample, column_types
):
    numeric_features_and_class_with_no_missing_values = []
    for feature in X_sample.columns:
        if column_types[feature] == consts.NUMERIC:
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

get_binned_numeric_features_and_class_with_no_missing_values = ResourceComputer(
    get_binned_numeric_features_and_class_with_no_missing_values,
    ["NoNaNBinnedNumericFeaturesAndClass"],
    {
        "X_sample": "XSample",
        "Y_sample": "YSample"
    }
)


def get_text_features_with_no_missing_values(
    X_sample, column_types
):
    text_features_with_no_missing_values = []
    for feature in X_sample.columns:
        if column_types[feature] == consts.TEXT:
            no_nan_series = X_sample[feature].dropna(
                axis=0, how='any'
            )
            text_features_with_no_missing_values.append(
                no_nan_series
            )
    return (text_features_with_no_missing_values,)

get_text_features_with_no_missing_values = ResourceComputer(
    get_text_features_with_no_missing_values,
    ["NoNaNTextFeatures"],
    { "X_sample": "XSample" }
)

"""
A list of all ResourceComputer
instances in this module.
"""
resources_info = build_resources_info(
    get_X,
    get_cv_seed,
    sample_columns,
    sample_rows,
    get_preprocessed_data,
    get_categorical_features_with_no_missing_values,
    get_categorical_features_and_class_with_no_missing_values,
    get_numeric_features_with_no_missing_values,
    get_binned_numeric_features_with_no_missing_values,
    get_binned_numeric_features_and_class_with_no_missing_values,
    get_text_features_with_no_missing_values
)
