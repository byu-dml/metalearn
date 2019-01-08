import time
import math
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import string

from .common_operations import *

"""
text_features_array is a list of pd.series objects
"""

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

def get_mfs_for_tokens_split_by_space(text_features_array, most_common_limit=10):
	tokens = {}
	alphanumeric_tokens = {}
	numeric_tokens = {}
	number_of_tokens = 0
	number_of_tokens_containing_numeric_char = 0

	# make dictionaries of token:count
	for feature in text_features_array:
		for text in feature:
			for token in text.split(" "):
				# update numeric_tokens
				if token.isdigit():
					if token not in numeric_tokens:
						numeric_tokens[token] = 0
					numeric_tokens[token] += 1
				# update alphanumeric_tokens
				elif token.isalnum():
					if token not in alphanumeric_tokens:
						alphanumeric_tokens[token] = 0
					alphanumeric_tokens[token] += 1

				# update number_of_tokens_containing_numeric_char
				if any(char.isdigit() for char in token):
					number_of_tokens_containing_numeric_char += 1

				# update tokens
				if token not in tokens:
					tokens[token] = 0
				tokens[token] += 1
				number_of_tokens += 1

	# get most_common_alphanumeric_tokens
	most_common_alphanumeric_tokens = []
	for token in sorted(alphanumeric_tokens, key=alphanumeric_tokens.get, reverse=True):
		most_common_alphanumeric_tokens.append(
			{
				"token": token,
				"count": alphanumeric_tokens[token],
				"ratio": alphanumeric_tokens[token]/len(tokens)
			}
		)
		if len(most_common_alphanumeric_tokens) == most_common_limit:
			break

	# get most_common_numeric_tokens
	most_common_numeric_tokens = []
	for token in sorted(numeric_tokens, key=numeric_tokens.get, reverse=True):
		most_common_numeric_tokens.append(
			{
				"token": token,
				"count": numeric_tokens[token],
				"ratio": numeric_tokens[token]/len(tokens)
			}
		)
		if len(most_common_numeric_tokens) == most_common_limit:
			break

	# get most_common_tokens
	most_common_tokens = []
	for token in sorted(tokens, key=tokens.get, reverse=True):
		most_common_tokens.append(
			{
				"token": token,
				"count": tokens[token],
				"ratio": tokens[token]/len(tokens)
			}
		)
		if len(most_common_tokens) == most_common_limit:
			break

	# compute number_of_distinct_tokens
	number_of_distinct_tokens = len(tokens)
	# compute ratio_of_distinct_tokens
	ratio_of_distinct_tokens = number_of_distinct_tokens / number_of_tokens
	# compute ratio_of_tokens_containing_numeric_char
	ratio_of_tokens_containing_numeric_char = number_of_tokens_containing_numeric_char / number_of_tokens

	return most_common_tokens, most_common_alphanumeric_tokens, most_common_numeric_tokens, number_of_tokens, number_of_distinct_tokens, number_of_tokens_containing_numeric_char, ratio_of_distinct_tokens, ratio_of_tokens_containing_numeric_char


