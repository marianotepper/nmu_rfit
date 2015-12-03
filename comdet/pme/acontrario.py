import numpy as np
import scipy.sparse as sp
import scipy.special as special
import itertools
import multipledispatch
import heapq
import utils
import line
import circle


def meaningful(data, model, inliers, inliers_threshold, epsilon):
    return compute_nfa(data, model, inliers, inliers_threshold) < epsilon


def filter_meaningful(meaningful_fun, mod_inliers_iter):
    return itertools.ifilter(meaningful_fun, mod_inliers_iter)


def compute_nfa(data, model, inliers, inliers_threshold):
    probability = random_probability(data, inliers_threshold, model)
    return inner_compute_nfa(inliers, model.min_sample_size, probability)


def inner_compute_nfa(inliers, n_samples, instance_proba):
    n = inliers.shape[0]
    k = inliers.sum() - n_samples
    if k <= 0:
        pfa = np.inf
    else:
        pfa = log_betainc(k, n - k + 1, instance_proba)
    n_tests = log_nchoosek(n, n_samples)
    return (pfa + n_tests) / np.log(10)


@multipledispatch.dispatch(np.ndarray, float, line.Line)
def random_probability(data, inliers_threshold, model):
    vec = np.max(data, axis=0) - np.min(data, axis=0)
    area = np.prod(vec)
    length = np.linalg.norm(vec)
    return length * 2 * inliers_threshold / area


@multipledispatch.dispatch(np.ndarray, float, circle.Circle)
def random_probability(data, inliers_threshold, model):
    upper = np.maximum(np.max(data, axis=0), model.center + model.radius)
    lower = np.minimum(np.min(data, axis=0), model.center - model.radius)
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


class _DummyArray:
    def __init__(self, shape, nnz):
        self.shape = shape
        self.nnz = nnz

    def sum(self):
        return self.nnz


def multiscale_meaningful(data, model, epsilon, min_count=None, max_count=None,
                          max_thresh=None):
    dist = model.distances(data).tolist()
    heapq.heapify(dist)
    nnz = 0
    while dist:
        s = heapq.heappop(dist)
        nnz += 1
        if min_count is not None and nnz < min_count:
            continue
        if max_count is not None and nnz > max_count:
            break
        if max_thresh is not None and s > max_thresh:
            break
        inliers = _DummyArray((data.shape[0],), nnz)
        if meaningful(data, model, inliers, s, epsilon):
            return True
    return False


def best_nfa(data, model, inliers):
    included = sp.find(inliers)[0]
    if included.shape[0] == 0:
        return np.inf
    max_dist = model.distances(data[included, :]).max()
    nfa = compute_nfa(data, model, inliers, max_dist)
    return nfa


def exclusion_principle(data, mod_inliers_list, inliers_threshold, epsilon):
    nfa_list = [best_nfa(data, mod, in_a)
                for mod, in_a in mod_inliers_list
                if compute_nfa(data, mod, in_a, inliers_threshold) < epsilon]
    idx = utils.argsort(nfa_list)

    keep_list = list(idx)
    for i, pick in enumerate(idx):
        mod, in_a = mod_inliers_list[pick]
        if i == 0:
            continue

        in_list = [mod_inliers_list[k][1] for k in keep_list if k != pick]
        in_list = map(lambda x: in_a.multiply(x).astype(bool), in_list)
        inliers = in_a - reduce(lambda x, y: (x + y).astype(bool), in_list)

        if not multiscale_meaningful(data, mod, epsilon, max_count=inliers.nnz):
            keep_list.remove(pick)

    return keep_list
