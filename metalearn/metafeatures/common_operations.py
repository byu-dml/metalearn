import numpy as np
import pandas as pd

def profile_distribution(data):
    """
    Compute the mean, standard deviation, min, quartile1, quartile2, quartile3, and max of a vector

    Parameters
    ----------
    data: array of real values

    Returns
    -------
    features = dictionary containing the min, max, mean, and standard deviation
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        ddof = 1 if len(data) > 1 else 0
        dist_mean = np.mean(data)
        dist_stdev = np.std(data, ddof=ddof)
        dist_min, dist_quartile1, dist_quartile2, dist_quartile3, dist_max = np.percentile(data, [0,25,50,75,100])
    return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_quartile2, dist_quartile3, dist_max)

def get_numeric_features(dataframe, column_types):
    return [feature for feature in dataframe.columns if column_types[feature] == "NUMERIC"]

def get_categorical_features(dataframe, column_types):
    return [feature for feature in dataframe.columns if column_types[feature] == "CATEGORICAL"]

def dtype_is_numeric(dtype):
    return "int" in str(dtype) or "float" in str(dtype)
