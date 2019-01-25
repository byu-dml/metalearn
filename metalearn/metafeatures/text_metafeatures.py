from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd

from .common_operations import *

def get_string_lengths_array_from_text_features(text_features_array):
	lengths = [feature.apply(len) for feature in text_features_array]
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

def get_mfs_for_tokens_split_by_space(text_features_array, most_common_limit=10):

	def isnumeric(token):
		try:
			float(token)
		except ValueError:
			return False
		return True

	def contains_numeric(token):
		return any([char.isdigit() for char in token])

	def flatten_nested_list(nested_list):
		return list(chain.from_iterable(nested_list))

	def filter_and_aggregate(tokens_series, f):
		filtered_tokens = tokens_series.combine(tokens_series.apply(f), lambda tokens, mask: np.array(tokens)[mask.astype(bool)].tolist())
		return flatten_nested_list(filtered_tokens)

	isnumeric = np.vectorize(isnumeric, otypes=[object])
	isalnum = np.vectorize(str.isalnum, otypes=[object])
	contains_numeric = np.vectorize(contains_numeric, otypes=[object])

	tokens = []
	numeric_tokens = []
	alphanumeric_tokens = []
	contains_numeric_tokens = []

	for feature in text_features_array:
		feature_tokens = feature.apply(str.split)
		tokens.extend(flatten_nested_list(feature_tokens))
		numeric_tokens.extend(filter_and_aggregate(feature_tokens, isnumeric))
		alphanumeric_tokens.extend(filter_and_aggregate(feature_tokens, isalnum))
		contains_numeric_tokens.extend(filter_and_aggregate(feature_tokens, contains_numeric))

	token_counts = Counter(tokens)
	numeric_token_counts = Counter(numeric_tokens)
	alphanumeric_token_counts = Counter(alphanumeric_tokens)
	contains_numeric_token_counts = Counter(contains_numeric_tokens)

	number_of_tokens = len(tokens)
	number_of_distinct_tokens = len(token_counts)
	number_of_tokens_containing_numeric_char = len(contains_numeric_tokens)

	ratio_of_distinct_tokens = 0 if number_of_tokens == 0 else (number_of_distinct_tokens / number_of_tokens)
	ratio_of_tokens_containing_numeric_char = 0 if number_of_tokens == 0 else (number_of_tokens_containing_numeric_char / number_of_tokens)

	return number_of_tokens, number_of_distinct_tokens, number_of_tokens_containing_numeric_char, ratio_of_distinct_tokens, ratio_of_tokens_containing_numeric_char


	# todo: re-include these loops after deciding what to do with most_common_tokens,
	# todo: most_common_alphanumeric_tokens, and most_common_numeric_tokens
	# get most_common_alphanumeric_tokens
	# most_common_alphanumeric_tokens = []
	# for token in sorted(alphanumeric_token_counts, key=alphanumeric_token_counts.get, reverse=True):
	# 	most_common_alphanumeric_tokens.append(
	# 		{
	# 			"token": token,
	# 			"count": alphanumeric_token_counts[token],
	# 			"ratio": alphanumeric_token_counts[token]/len(token_counts)
	# 		}
	# 	)
	# 	if len(most_common_alphanumeric_tokens) == most_common_limit:
	# 		break
	#
	# # get most_common_numeric_tokens
	# most_common_numeric_tokens = []
	# for token in sorted(numeric_token_counts, key=numeric_token_counts.get, reverse=True):
	# 	most_common_numeric_tokens.append(
	# 		{
	# 			"token": token,
	# 			"count": numeric_token_counts[token],
	# 			"ratio": numeric_token_counts[token]/len(token_counts)
	# 		}
	# 	)
	# 	if len(most_common_numeric_tokens) == most_common_limit:
	# 		break
	#
	# # get most_common_tokens
	# most_common_tokens = []
	# for token in sorted(token_counts, key=token_counts.get, reverse=True):
	# 	most_common_token_counts.append(
	# 		{
	# 			"token": token,
	# 			"count": token_counts[token],
	# 			"ratio": token_counts[token]/len(token_counts)
	# 		}
	# 	)
	# 	if len(most_common_tokens) == most_common_limit:
	# 		break
