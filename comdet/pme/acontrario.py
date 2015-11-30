import numpy as np
import scipy.special as special
import itertools
import multipledispatch
import utils
import line
import circle


def compute_nfa(inliers, n_samples, instance_proba):
    n = inliers.shape[0]
    k = inliers.sum() - n_samples
    if k <= 0:
        nfa = np.inf
    else:
        nfa = log_betainc(k, n - k + 1, instance_proba)
    n_tests = log_nchoosek(n, n_samples)
    return (nfa + n_tests) / np.log(10)


def meaningful(data, model, inliers, inliers_threshold, epsilon):
    proba = random_probability(data, inliers_threshold, model)
    return compute_nfa(inliers, model.min_sample_size, proba) < epsilon


def filter_meaningful(meaningful_fun, mod_inliers_iter):
    return itertools.ifilter(meaningful_fun, mod_inliers_iter)


@multipledispatch.dispatch(np.ndarray, float, line.Line)
def random_probability(data, inliers_threshold, model):
    vec = np.max(data, axis=0) - np.min(data, axis=0)
    area = np.prod(vec)
    length = np.linalg.norm(vec)
    return length * 2 * inliers_threshold / area


@multipledispatch.dispatch(np.ndarray, float, circle.Circle)
def random_probability(data, inliers_threshold, model):
    area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
    ring_area = np.pi * ((model.radius + inliers_threshold) ** 2 -
                         (model.radius - inliers_threshold) ** 2)
    return min(ring_area / area, 1)


def log_nchoosek(n, k):
    return special.gammaln(n + 1) - special.gammaln(n - k + 1)\
           - special.gammaln(k + 1)


def log_betainc(a, b, x):
    if a <= 0.0 or b <= 0.0:
        raise ValueError('Bad a or b in function log_betainc')
    if x <= 0.0 or x > 1.0:
        raise ValueError('Bad x in function log_betainc')
    if x == 1.0:
        return 0

    logbt = special.gammaln(a + b) - special.gammaln(a) - special.gammaln(b) +\
            a * np.log(x) + b * np.log(1.0 - x)

    if x < (a + 1.0) / (a + b + 2.0):
        # Use continued fraction directly
        return logbt + np.log(special.betainc(a, b, x))
    else:
        # Factors in front of the continued fraction.
        bt = np.exp(logbt)
        # Use continued fraction after making the symmetry transformation.
        return np.log(1.0 - bt * special.betainc(b, a, 1.0 - x) / b)


def exclusion_principle(data, mod_inliers_list, inliers_threshold, epsilon):
    nfa_list = []
    for (mod, in_a) in mod_inliers_list:
        proba = random_probability(data, inliers_threshold, mod)
        nfa_list.append(compute_nfa(in_a, mod.min_sample_size, proba))
    idx = utils.argsort(nfa_list)

    out_list = []
    for i, pick in enumerate(idx):
        mod, in_a = mod_inliers_list[pick]
        if i == 0:
            out_list.append(pick)
            continue

        inliers = [in_b for k, (_, in_b) in enumerate(mod_inliers_list)
                   if idx.index(k) in out_list]
        inliers = map(lambda x: in_a - in_a.multiply(x).astype(bool), inliers)

        # proba = random_probability(data, inliers_threshold, mod)
        # print [compute_nfa(in_b, mod.min_sample_size, proba)
        #        for in_b in inliers]

        if all([meaningful(data, mod, in_b, inliers_threshold, epsilon)
                for in_b in inliers]):
            out_list.append(pick)

    return out_list
    # return idx
