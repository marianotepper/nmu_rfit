import numpy as np
import scipy.sparse as sp
import scipy.special as special
import abc
import heapq
import operator


def meaningful(value, epsilon):
    return value < epsilon


class BinomialNFA(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, inliers_threshold=None):
        n = self.data.shape[0]
        p = self._random_probability(model, inliers_threshold=inliers_threshold)
        pfa = log_binomial(n, n_inliers - model.min_sample_size, p)
        n_tests = log_nchoosek(n, model.min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon

    @abc.abstractmethod
    def _random_probability(self, model, inliers_threshold=None):
        pass


def log_binomial(n, k, instance_proba):
    if k <= 0:
        return np.inf
    else:
        return log_betainc(k, n - k + 1, instance_proba)


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


def optimal_nfa(ac_tester, model, inliers):
    if sp.issparse(inliers):
        inliers = np.squeeze(inliers.toarray())
    dist = model.distances(ac_tester.data[inliers, :]).tolist()
    dist.sort()
    min_nfa = np.inf
    for k, s in enumerate(dist):
        if s < np.finfo(np.float32).resolution:
            continue
        nfa = ac_tester.nfa(model, k+1, inliers_threshold=s)
        min_nfa = np.minimum(nfa, min_nfa)
    return min_nfa + np.log10(k+1)


def multiscale_meaningful(ac_tester, model, min_count=None, max_count=None,
                          max_thresh=None):
    dist = model.distances(ac_tester.data).tolist()
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
        if ac_tester.meaningful(model, nnz, inliers_threshold=s):
            return True
    return False


def exclusion_principle(ac_tester, mod_inliers_list):
    # return range(len(mod_inliers_list))
    nfa_list = [(i, optimal_nfa(ac_tester, mod, in_a))
                for i, (mod, in_a) in enumerate(mod_inliers_list)]
    nfa_list = filter(lambda e: meaningful(e[1], ac_tester.epsilon), nfa_list)
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
        if not multiscale_meaningful(mod, max_count=inliers.nnz):
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
