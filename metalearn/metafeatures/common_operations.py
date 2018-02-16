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
        dist_max = np.amax(data)
        dist_min, dist_quartile1, dist_quartile2, dist_quartile3 = np.percentile(data, [0,.25,.5,.75])
    return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_quartile2, dist_quartile3, dist_max)

def get_numeric_features(dataframe):
    """
    Gets the names of the numeric attributes in the data.
    """
    return [col_name for col_name, col_type in zip(dataframe.columns, dataframe.dtypes) if dtype_is_numeric(col_type)]

def dtype_is_numeric(dtype):
    return "int" in str(dtype) or "float" in str(dtype)
