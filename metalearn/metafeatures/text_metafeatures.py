from collections import Counter
from itertools import chain

import numpy as np

from metalearn.metafeatures.common_operations import profile_distribution
from metalearn.metafeatures.base import build_resources_info, ResourceComputer, MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup


def get_string_lengths_array_from_text_features(text_features_array):
	lengths = [feature.apply(len) for feature in text_features_array]
	return (lengths,)

get_string_lengths_array_from_text_features = ResourceComputer(
	get_string_lengths_array_from_text_features,
	["ArrayOfStringLengthsOfTextFeatures"],
	{ "text_features_array": "NoNaNTextFeatures" }
)

def get_string_length_means(string_lengths_array):
	means = [feature.mean() for feature in string_lengths_array]
	return profile_distribution(means)

get_string_length_means = MetafeatureComputer(
	get_string_length_means,
	[
		"MeanMeansOfStringLengthOfTextFeatures",
		"StdevMeansOfStringLengthOfTextFeatures",
		"SkewMeansOfStringLengthOfTextFeatures",
		"KurtosisMeansOfStringLengthOfTextFeatures",
		"MinMeansOfStringLengthOfTextFeatures",
		"Quartile1MeansOfStringLengthOfTextFeatures",
		"Quartile2MeansOfStringLengthOfTextFeatures",
		"Quartile3MeansOfStringLengthOfTextFeatures",
		"MaxMeansOfStringLengthOfTextFeatures"
	],
	ProblemType.ANY,
	[MetafeatureGroup.TEXT],
	{
		"string_lengths_array": "ArrayOfStringLengthsOfTextFeatures"
	}
)


def get_string_length_stdev(string_lengths_array):
	stdevs = [feature.std() for feature in string_lengths_array]
	return profile_distribution(stdevs)

get_string_length_stdev = MetafeatureComputer(
	get_string_length_stdev,
	[
		"MeanStdDevOfStringLengthOfTextFeatures",
		"StdevStdDevOfStringLengthOfTextFeatures",
		"SkewStdDevOfStringLengthOfTextFeatures",
		"KurtosisStdDevOfStringLengthOfTextFeatures",
		"MinStdDevOfStringLengthOfTextFeatures",
		"Quartile1StdDevOfStringLengthOfTextFeatures",
		"Quartile2StdDevOfStringLengthOfTextFeatures",
		"Quartile3StdDevOfStringLengthOfTextFeatures",
		"MaxStdDevOfStringLengthOfTextFeatures"
	],
	ProblemType.ANY,
	[MetafeatureGroup.TEXT],
	{
		"string_lengths_array": "ArrayOfStringLengthsOfTextFeatures"
	}
)


def get_string_length_skewness(string_lengths_array):
	skews = [feature.skew() for feature in string_lengths_array]
	return profile_distribution(skews)

get_string_length_skewness = MetafeatureComputer(
	get_string_length_skewness,
	[
		"MeanSkewnessOfStringLengthOfTextFeatures",
		"StdevSkewnessOfStringLengthOfTextFeatures",
		"SkewSkewnessOfStringLengthOfTextFeatures",
		"KurtosisSkewnessOfStringLengthOfTextFeatures",
		"MinSkewnessOfStringLengthOfTextFeatures",
		"Quartile1SkewnessOfStringLengthOfTextFeatures",
		"Quartile2SkewnessOfStringLengthOfTextFeatures",
		"Quartile3SkewnessOfStringLengthOfTextFeatures",
		"MaxSkewnessOfStringLengthOfTextFeatures"
	],
	ProblemType.ANY,
	[MetafeatureGroup.TEXT],
	{
		"string_lengths_array": "ArrayOfStringLengthsOfTextFeatures"
	}
)


def get_string_length_kurtosis(string_lengths_array):
	kurtoses = [feature.kurtosis() for feature in string_lengths_array]
	return profile_distribution(kurtoses)

get_string_length_kurtosis = MetafeatureComputer(
	get_string_length_kurtosis,
	[
		"MeanKurtosisOfStringLengthOfTextFeatures",
		"StdevKurtosisOfStringLengthOfTextFeatures",
		"SkewKurtosisOfStringLengthOfTextFeatures",
		"KurtosisKurtosisOfStringLengthOfTextFeatures",
		"MinKurtosisOfStringLengthOfTextFeatures",
		"Quartile1KurtosisOfStringLengthOfTextFeatures",
		"Quartile2KurtosisOfStringLengthOfTextFeatures",
		"Quartile3KurtosisOfStringLengthOfTextFeatures",
		"MaxKurtosisOfStringLengthOfTextFeatures"
	],
	ProblemType.ANY,
	[MetafeatureGroup.TEXT],
	{
		"string_lengths_array": "ArrayOfStringLengthsOfTextFeatures"
	}
)


def get_mfs_for_tokens_split_by_space(text_features_array, most_common_limit):

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

get_mfs_for_tokens_split_by_space = MetafeatureComputer(
	get_mfs_for_tokens_split_by_space,
	[
		"NumberOfTokens",
		"NumberOfDistinctTokens",
		"NumberOfTokensContainingNumericChar",
		"RatioOfDistinctTokens",
		"RatioOfTokensContainingNumericChar"
	],
	ProblemType.ANY,
	[MetafeatureGroup.TEXT],
	{
		"text_features_array": "NoNaNTextFeatures",
		'most_common_limit': 10,
	}
)


"""
A list of all ResourceComputer
instances in this module.
"""
resources_info = build_resources_info(
	get_string_lengths_array_from_text_features
)

"""
A list of all MetafeatureComputer
instances in this module.
"""
metafeatures_info = build_resources_info(
    get_string_length_means,
	get_string_length_stdev,
	get_string_length_skewness,
	get_string_length_kurtosis,
	get_mfs_for_tokens_split_by_space
)
