from __future__ import absolute_import
import numpy as np
import scipy.special as sp


def compute_nfa(data, inliers_threshold, n_samples, model, model_proba_fun):

    N = data.shape[0]
    k = np.sum(model.distances(data) < inliers_threshold) - n_samples
    proba = model_proba_fun(data, inliers_threshold, model)
    if k <= 0:
        nfa = np.inf
    else:
        nfa = log_betainc(k, N - k + 1, proba)
    n_tests = log_nchoosek(N, n_samples)
    return (nfa + n_tests) / np.log(10)


def random_line_probability(data, inliers_threshold, line):
    maxima = np.max(data, axis=1)
    minima = np.min(data, axis=1)
    print minima, maxima
    vec = maxima - minima
    area = np.prod(vec)
    dist = np.linalg.norm(vec)
    return dist * 2 * inliers_threshold / area


def random_circle_probability(data, inliers_threshold, circle):
    area = np.prod(np.max(data, axis=1) - np.min(data, axis=1))
    proba = np.pi * ((circle.radius + inliers_threshold) ** 2 -
                     (circle.radius + inliers_threshold) ** 2) / area;
    return min(proba, 1)


def log_nchoosek(n, k):
    return sp.gammaln(n + 1) - sp.gammaln(n - k + 1) - sp.gammaln(k + 1)


def log_betainc(a, b, x):

    if a <= 0.0 or b <= 0.0:
        raise ValueError('Bad a or b in function log_betainc')
    if x <= 0.0 or x > 1.0:
        raise ValueError('Bad x in function log_betainc')

    if x == 1.0:
        return 0
    logbt = sp.gammaln(a + b) - sp.gammaln(a) - sp.gammaln(b) +\
            a * np.log(x) + b * np.log(1.0 - x)

    if x < (a + 1.0) / (a + b + 2.0):
        # Use continued fraction directly
        return logbt + np.log(sp.betainc(x, a, b))
    else:
        # Factors in front of the continued fraction.
        bt = np.exp(logbt)
        # Use continued fraction after making the symmetry transformation.
        return np.log(1.0 - bt * sp.betainc(1.0 - x, b, a) / b)

