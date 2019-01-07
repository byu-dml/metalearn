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

def get_most_common_tokens_split_by_space(text_features_array, most_common_limit=10):
	tokens = {}
	alnum_tokens = {}
	num_tokens = {}
	# make dictionaries of token:count
	for feature in text_features_array:
		for text in feature:
			for token in text.split(" "):
				if token.isdigit():
					if token not in num_tokens:
						num_tokens[token] = 0
					num_tokens[token] += 1
				elif token.isalnum():
					if token not in alnum_tokens:
						alnum_tokens[token] = 0
					alnum_tokens[token] += 1

				if token not in tokens:
					tokens[token] = 0
				tokens[token] += 1

	# get most_common_alnum_tokens
	most_common_alnum_tokens = []
	for token in sorted(alnum_tokens, key=alnum_tokens.get, reverse=True):
		most_common_alnum_tokens.append(
			{
				"token": token,
				"count": alnum_tokens[token],
				"ratio": alnum_tokens[token]/len(tokens)
			}
		)
		if len(most_common_alnum_tokens) == most_common_limit:
			break

	# get most_common_num_tokens
	most_common_num_tokens = []
	for token in sorted(num_tokens, key=num_tokens.get, reverse=True):
		most_common_num_tokens.append(
			{
				"token": token,
				"count": num_tokens[token],
				"ratio": num_tokens[token]/len(tokens)
			}
		)
		if len(most_common_num_tokens) == most_common_limit:
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

	return most_common_tokens, most_common_alnum_tokens, most_common_num_tokens


def get_most_common_tokens_split_by_punctuation(text_features_array, most_common_limit=10):
	tokens = {}
	# make dictionaries of token:count
	for feature in text_features_array:
		for text in feature:
			# for token in re.split("[^\w^\s]", text):
			for token in text.split(string.punctuation):
				if token not in tokens:
					tokens[token] = 0
				tokens[token] += 1