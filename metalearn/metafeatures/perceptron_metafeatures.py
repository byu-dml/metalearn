import numpy as np

from sklearn.linear_model import Perceptron

from metalearn.metafeatures.common_operations import profile_distribution


def get_fitted_perceptron(X, Y, seed, n_classes, frac='all'):
    class_fraction = n_classes/len(X)

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


def get_perceptron_weights_sum(perceptron):
    weights_sum = np.sum(perceptron.coef_)
    return weights_sum,


def get_perceptron_weights_dist(perceptron):
    weights = perceptron.coef_.flatten()
    return profile_distribution(weights)


def get_perceptron_bias_dist(perceptron):
    intercepts = perceptron.intercept_
    return profile_distribution(intercepts)


def get_perceptron_n_iters(perceptron):
    n_iter = perceptron.n_iter_
    return n_iter,
