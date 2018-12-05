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


def get_string_length(string):
	return len(string)

def get_string_lengths_array_from_text_features(text_features_array):
	lengths = []
	for feature in text_features_array:
		lengths.append(feature.apply(get_string_length))
	return lengths

def get_string_length_means(text_features_array):
	string_lengths_array = get_string_lengths_array_from_text_features(text_features_array)
	means = [feature.mean() for feature in string_lengths_array]
	return profile_distribution(means)

def get_string_length_stdev(text_features_array):
	string_lengths_array = get_string_lengths_array_from_text_features(text_features_array)
	stdevs = [feature.std() for feature in string_lengths_array]
	return profile_distribution(stdevs)


def get_string_length_skewness(text_features_array):
	string_lengths_array = get_string_lengths_array_from_text_features(text_features_array)
	skews = [feature.skew() for feature in string_lengths_array]
	return profile_distribution(skews)

def get_string_length_kurtosis(text_features_array):
	string_lengths_array = get_string_lengths_array_from_text_features(text_features_array)
	kurtoses = [feature.kurtosis() for feature in string_lengths_array]
	return profile_distribution(kurtoses)