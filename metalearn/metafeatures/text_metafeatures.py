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

def get_string_lengths_array_from_text_features(text_features_array):
	lengths = []
	for feature in text_features_array:
		lengths.append(feature.apply(len))
	return (lengths,)

def get_string_length_means(string_lengths_array):
	means = [feature.mean() for feature in string_lengths_array]
	return profile_distribution(means)

def get_string_length_stdev(string_lengths_array):
	stdevs = [feature.std() for feature in string_lengths_array]
	return profile_distribution(stdevs)

def get_string_length_skewness(string_lengths_array):
	skews = [feature.skew() for feature in string_lengths_array]
	return profile_distribution(skews)

def get_string_length_kurtosis(string_lengths_array):
	kurtoses = [feature.kurtosis() for feature in string_lengths_array]
	return profile_distribution(kurtoses)