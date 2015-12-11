import numpy as np
import scipy.sparse as sp
import scipy.special as special
import multipledispatch
import heapq
import operator
import utils
import line
import circle


def meaningful(value, upper_bound):
    return value < upper_bound


def false_alarms_number(data, model, inliers, inliers_threshold):
    probability = random_probability(data, inliers_threshold, model)
    return inner_number_false_positives(inliers, model.min_sample_size, probability)


def inner_number_false_positives(inliers, n_samples, instance_proba):
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
    # area = np.prod(upper - lower)
    area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
    # (a + b)**2 - (a - b)**2 == 4ab
    ring_area = np.pi * 4 * model.radius * inliers_threshold
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


def optimal(data, model, inliers):
    if sp.issparse(inliers):
        inliers = np.squeeze(inliers.toarray())
    dist = model.distances(data[inliers, :]).tolist()
    dist.sort()
    min_nfa = np.inf
    for k, s in enumerate(dist):
        if s < np.finfo(np.float32).resolution:
            continue
        dummy_inliers = _DummyArray((data.shape[0],), k+1)
        nfa = false_alarms_number(data, model, dummy_inliers, s)
        min_nfa = np.minimum(nfa, min_nfa)
    return min_nfa + np.log10(k+1)


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
        nfa = false_alarms_number(data, model, inliers, s)
        if meaningful(nfa, epsilon):
            return True
    return False


def uniformity(x, model, inliers):
    bins = np.linspace(-np.pi, np.pi, inliers.sum()+1)[1:]
    if sp.issparse(inliers):
        inliers = np.squeeze(inliers.toarray())
    vecs = x[inliers, :] - model.center
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    idx = np.searchsorted(bins, angles)
    bc = np.bincount(idx)
    n = inliers.sum()
    k = (bc <= 0).sum()
    if k == 0:
        proba = np.inf
    else:
        proba = log_betainc(n - k, k + 1, 1 - np.exp(-1))
    return np.log(n) + proba < 0


def exclusion_principle(data, mod_inliers_list, inliers_threshold, epsilon):
    # return range(len(mod_inliers_list))
    nfa_list = [(i, optimal(data, mod, in_a))
                for i, (mod, in_a) in enumerate(mod_inliers_list)]
    nfa_list = filter(lambda e: meaningful(e[1], epsilon), nfa_list)
    nfa_list = sorted(nfa_list, key=operator.itemgetter(1))
    print nfa_list
    idx = zip(*nfa_list)[0]

    # return idx

    keep_list = list(idx)
    print keep_list
    for pick in idx:
        mod, in_a = mod_inliers_list[pick]
        in_list = [mod_inliers_list[k][1] for k in keep_list if k < pick]
        if not in_list:
            continue
        in_list = map(lambda x: in_a.multiply(x).astype(bool), in_list)
        inliers = in_a - reduce(lambda x, y: (x + y).astype(bool), in_list)
        if not multiscale_meaningful(data, mod, epsilon, max_count=inliers.nnz):
            keep_list.remove(pick)

    print keep_list
    return keep_list

    # keep_list = []
    # while idx:
    #     max_nfa = -np.inf
    #     max_idx = None
    #     for i, pick in enumerate(idx):
    #         mod, in_a = mod_inliers_list[pick]
    #
    #         in_list = [mod_inliers_list[k][1] for k in idx if k != pick]
    #         if not in_list:
    #             break
    #         in_list = map(lambda x: in_a.multiply(x).astype(bool), in_list)
    #         inliers = in_a - reduce(lambda x, y: (x + y).astype(bool), in_list)
    #
    #         nfa = best_nfa(data, mod, inliers)
    #         if nfa > max_nfa:
    #             max_nfa = nfa
    #             max_idx = pick
    #
    #     print max_nfa, max_idx
    #     if max_nfa > epsilon:
    #         idx.remove(max_idx)
    #     else:
    #         break
    #
    # print idx
    # return idx
