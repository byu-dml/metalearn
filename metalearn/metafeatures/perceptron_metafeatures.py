import numpy as np
from sklearn.linear_model import Perceptron

from metalearn.metafeatures.common_operations import profile_distribution


def get_fitted_perceptron(X, Y, seed, n_classes, frac='all'):
    class_fraction = n_classes/len(X)

    if len(X) < n_classes * 2:
        raise ValueError(f'The number of instances in X must be at least 2 * n_classes')

    if frac == 'all':
        frac = class_fraction
    elif frac == 'tenth':
        frac = min(max(0.9, class_fraction), 1-class_fraction)
    elif frac == 'half':
        frac = min(max(0.5, class_fraction), 1-class_fraction)
    elif frac == 'sqrt':
        frac = min(max(1 - (np.sqrt(len(X)) / len(X)), class_fraction), 1-class_fraction)
    else:
        raise ValueError(f'{frac} is not a valid option. Must be one of ["all", "tenth", "half", "sqrt"]')

    cls = Perceptron(random_state=seed, validation_fraction=frac, early_stopping=True, max_iter=1000, tol=1e-3)
    cls.fit(X, Y)

    return cls,


def get_perceptron_weights_sum(perceptron):
    weights_sum = np.sum(perceptron.coef_)
    return weights_sum,


def get_perceptron_weights_dist(perceptron):
    weights = perceptron.coef_.flatten()
    return profile_distribution(weights)


def get_perceptron_n_iters(perceptron):
    return perceptron.n_iter_,
