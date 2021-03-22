import numpy as np

from sklearn.linear_model import Perceptron

from metalearn.metafeatures.common_operations import profile_distribution
from metalearn.metafeatures.base import build_resources_info, ResourceComputer, MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup


def get_fitted_perceptron(X, Y, seed, n_classes, frac):
    class_fraction = n_classes / len(X)

    if frac == 'all':
        frac = class_fraction
    elif frac == 'tenth':
        frac = min(max(0.9, class_fraction), 1 - class_fraction)
    elif frac == 'half':
        frac = min(max(0.5, class_fraction), 1 - class_fraction)
    elif frac == 'sqrt':
        frac = min(max(1 - (np.sqrt(len(X)) / len(X)), class_fraction), 1 - class_fraction)

    clf = Perceptron(random_state=seed, validation_fraction=frac, early_stopping=True, max_iter=1000, tol=1e-3)

    if n_classes == 1:
        # The `.fit` method cannot handle a dataset with 1 class, so we
        # give the model the theoretical weights it would learn on a single
        # class classification problem: zero weights (so as to predict class 0
        # every time).
        clf.coef_ = np.zeros(X.shape[1])
        clf.intercept_ = np.array([0])
        clf.n_iter_ = 0
    else:
        clf.fit(X, Y)

    return clf,


get_full_perceptron = ResourceComputer(
    computer=get_fitted_perceptron,
    returns=["FullPerceptron"],
    argmap={"X": "XPreprocessed", "Y": "YSample", "seed": 10, "n_classes": "NumberOfClasses", "frac": "all"}
)

get_one_tenth_perceptron = ResourceComputer(
    computer=get_fitted_perceptron,
    returns=["OneTenthPerceptron"],
    argmap={"X": "XPreprocessed", "Y": "YSample", "seed": 11, "n_classes": "NumberOfClasses", "frac": "tenth"}
)

get_one_half_perceptron = ResourceComputer(
    computer=get_fitted_perceptron,
    returns=["OneHalfPerceptron"],
    argmap={"X": "XPreprocessed", "Y": "YSample", "seed": 12, "n_classes": "NumberOfClasses", "frac": "half"}
)

get_sqrt_perceptron = ResourceComputer(
    computer=get_fitted_perceptron,
    returns=["SqrtPerceptron"],
    argmap={"X": "XPreprocessed", "Y": "YSample", "seed": 13, "n_classes": "NumberOfClasses", "frac": "sqrt"}
)


def get_perceptron_weights_sum(perceptron):
    weights_sum = np.sum(perceptron.coef_)
    return weights_sum,


get_full_perceptron_weights_sum = MetafeatureComputer(
    computer=get_perceptron_weights_sum,
    returns=["FullPerceptronWeightsSum"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "FullPerceptron"}
)

get_one_tenth_perceptron_weights_sum = MetafeatureComputer(
    computer=get_perceptron_weights_sum,
    returns=["OneTenthPerceptronWeightsSum"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneTenthPerceptron"}
)

get_one_half_perceptron_weights_sum = MetafeatureComputer(
    computer=get_perceptron_weights_sum,
    returns=["OneHalfPerceptronWeightsSum"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneHalfPerceptron"}
)

get_sqrt_perceptron_weights_sum = MetafeatureComputer(
    computer=get_perceptron_weights_sum,
    returns=["SqrtPerceptronWeightsSum"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "SqrtPerceptron"}
)


def get_perceptron_weights_dist(perceptron):
    weights = perceptron.coef_.flatten()
    return profile_distribution(weights)


get_full_perceptron_weights_dist = MetafeatureComputer(
    computer=get_perceptron_weights_dist,
    returns=["MeanFullPerceptronWeights",
             "StdevFullPerceptronWeights",
             "SkewFullPerceptronWeights",
             "KurtosisFullPerceptronWeights",
             "MinFullPerceptronWeights",
             "Quartile1FullPerceptronWeights",
             "Quartile2FullPerceptronWeights",
             "Quartile3FullPerceptronWeights",
             "MaxFullPerceptronWeights"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "FullPerceptron"}
)

get_one_tenth_perceptron_weights_dist = MetafeatureComputer(
    computer=get_perceptron_weights_dist,
    returns=["MeanOneTenthPerceptronWeights",
             "StdevOneTenthPerceptronWeights",
             "SkewOneTenthPerceptronWeights",
             "KurtosisOneTenthPerceptronWeights",
             "MinOneTenthPerceptronWeights",
             "Quartile1OneTenthPerceptronWeights",
             "Quartile2OneTenthPerceptronWeights",
             "Quartile3OneTenthPerceptronWeights",
             "MaxOneTenthPerceptronWeights"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneTenthPerceptron"}
)

get_one_half_perceptron_weights_dist = MetafeatureComputer(
    computer=get_perceptron_weights_dist,
    returns=["MeanOneHalfPerceptronWeights",
             "StdevOneHalfPerceptronWeights",
             "SkewOneHalfPerceptronWeights",
             "KurtosisOneHalfPerceptronWeights",
             "MinOneHalfPerceptronWeights",
             "Quartile1OneHalfPerceptronWeights",
             "Quartile2OneHalfPerceptronWeights",
             "Quartile3OneHalfPerceptronWeights",
             "MaxOneHalfPerceptronWeights"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneHalfPerceptron"}
)

get_sqrt_perceptron_weights_dist = MetafeatureComputer(
    computer=get_perceptron_weights_dist,
    returns=["MeanSqrtPerceptronWeights",
             "StdevSqrtPerceptronWeights",
             "SkewSqrtPerceptronWeights",
             "KurtosisSqrtPerceptronWeights",
             "MinSqrtPerceptronWeights",
             "Quartile1SqrtPerceptronWeights",
             "Quartile2SqrtPerceptronWeights",
             "Quartile3SqrtPerceptronWeights",
             "MaxSqrtPerceptronWeights"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "SqrtPerceptron"}
)


def get_perceptron_bias_dist(perceptron):
    intercepts = perceptron.intercept_
    return profile_distribution(intercepts)


get_full_perceptron_bias_dist = MetafeatureComputer(
    computer=get_perceptron_bias_dist,
    returns=["MeanFullPerceptronBias",
             "StdevFullPerceptronBias",
             "SkewFullPerceptronBias",
             "KurtosisFullPerceptronBias",
             "MinFullPerceptronBias",
             "Quartile1FullPerceptronBias",
             "Quartile2FullPerceptronBias",
             "Quartile3FullPerceptronBias",
             "MaxFullPerceptronBias"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "FullPerceptron"}
)

get_one_tenth_perceptron_bias_dist = MetafeatureComputer(
    computer=get_perceptron_bias_dist,
    returns=["MeanOneTenthPerceptronBias",
             "StdevOneTenthPerceptronBias",
             "SkewOneTenthPerceptronBias",
             "KurtosisOneTenthPerceptronBias",
             "MinOneTenthPerceptronBias",
             "Quartile1OneTenthPerceptronBias",
             "Quartile2OneTenthPerceptronBias",
             "Quartile3OneTenthPerceptronBias",
             "MaxOneTenthPerceptronBias"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneTenthPerceptron"}
)

get_one_half_perceptron_bias_dist = MetafeatureComputer(
    computer=get_perceptron_bias_dist,
    returns=["MeanOneHalfPerceptronBias",
             "StdevOneHalfPerceptronBias",
             "SkewOneHalfPerceptronBias",
             "KurtosisOneHalfPerceptronBias",
             "MinOneHalfPerceptronBias",
             "Quartile1OneHalfPerceptronBias",
             "Quartile2OneHalfPerceptronBias",
             "Quartile3OneHalfPerceptronBias",
             "MaxOneHalfPerceptronBias"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneHalfPerceptron"}
)

get_sqrt_perceptron_bias_dist = MetafeatureComputer(
    computer=get_perceptron_bias_dist,
    returns=["MeanSqrtPerceptronBias",
             "StdevSqrtPerceptronBias",
             "SkewSqrtPerceptronBias",
             "KurtosisSqrtPerceptronBias",
             "MinSqrtPerceptronBias",
             "Quartile1SqrtPerceptronBias",
             "Quartile2SqrtPerceptronBias",
             "Quartile3SqrtPerceptronBias",
             "MaxSqrtPerceptronBias"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "SqrtPerceptron"}
)


def get_perceptron_n_iters(perceptron):
    n_iter = perceptron.n_iter_
    return n_iter,


get_full_perceptron_n_iters = MetafeatureComputer(
    computer=get_perceptron_n_iters,
    returns=["FullPerceptronNIters"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "FullPerceptron"}
)

get_one_tenth_perceptron_n_iters = MetafeatureComputer(
    computer=get_perceptron_n_iters,
    returns=["OneTenthPerceptronNIters"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneTenthPerceptron"}
)

get_one_half_perceptron_n_iters = MetafeatureComputer(
    computer=get_perceptron_n_iters,
    returns=["OneHalfPerceptronNIters"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "OneHalfPerceptron"}
)

get_sqrt_perceptron_n_iters = MetafeatureComputer(
    computer=get_perceptron_n_iters,
    returns=["SqrtPerceptronNIters"],
    problem_type=ProblemType.CLASSIFICATION,
    groups=[MetafeatureGroup.MODEL_BASED, MetafeatureGroup.TARGET_DEPENDENT],
    argmap={"perceptron": "SqrtPerceptron"}
)

"""
A list of all ResourceComputer
instances in this module.
"""
resources_info = build_resources_info(
    get_full_perceptron,
    get_one_tenth_perceptron,
    get_one_half_perceptron,
    get_sqrt_perceptron
)

"""
A list of all MetafeatureComputer
instances in this module.
"""
metafeatures_info = build_resources_info(
    get_full_perceptron_weights_sum,
    get_one_tenth_perceptron_weights_sum,
    get_one_half_perceptron_weights_sum,
    get_sqrt_perceptron_weights_sum,
    get_full_perceptron_weights_dist,
    get_one_tenth_perceptron_weights_dist,
    get_one_half_perceptron_weights_dist,
    get_sqrt_perceptron_weights_dist,
    get_full_perceptron_bias_dist,
    get_one_tenth_perceptron_bias_dist,
    get_one_half_perceptron_bias_dist,
    get_sqrt_perceptron_bias_dist,
    get_full_perceptron_n_iters,
    get_one_tenth_perceptron_n_iters,
    get_one_half_perceptron_n_iters,
    get_sqrt_perceptron_n_iters
)
