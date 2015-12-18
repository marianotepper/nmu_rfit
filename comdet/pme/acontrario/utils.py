import numpy as np
import scipy.sparse as sp
import scipy.special as special
import abc
import operator
import itertools


def ifilter(ac_tester, mig):
    def inner_meaningful((model, inliers)):
        return ac_tester.meaningful(model, inliers.sum())
    return itertools.ifilter(meaningful, mig)


def meaningful(value, epsilon):
    return value < epsilon


class BinomialNFA(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, data=None, inliers_threshold=None):
        n = self.data.shape[0]
        p = self._random_probability(model, data=data,
                                     inliers_threshold=inliers_threshold)
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
    if a <= 0.:
        raise ValueError('Bad a in function log_betainc')
    if b <= 0.:
        raise ValueError('Bad b in function log_betainc')
    if x <= 0. or x > 1.:
        raise ValueError('Bad x in function log_betainc')
    if x == 1.:
        return 0.

    logbt = special.gammaln(a + b) - special.gammaln(a) - special.gammaln(b) +\
            a * np.log(x) + b * np.log(1. - x)

    if x < (a + 1.) / (a + b + 2.):
        # Use continued fraction directly
        return logbt + np.log(special.betainc(a, b, x))
    else:
        # Factors in front of the continued fraction.
        bt = np.exp(logbt)
        # Use continued fraction after making the symmetry transformation.
        return np.log(1.0 - bt * special.betainc(b, a, 1. - x) / b)


def optimal_nfa(ac_tester, model, inliers, considered=None):
    if considered is None:
        data_considered = None
    else:
        if considered.sum() == inliers.sum():
            return -np.inf
        data_considered = ac_tester.data[considered, :]
    if sp.issparse(inliers):
        inliers = np.squeeze(inliers.toarray())
    dist = model.distances(ac_tester.data[inliers, :])
    if dist.size <= model.min_sample_size:
        return np.inf
    dist.sort()
    min_nfa = np.inf
    for k, s in enumerate(dist):
        if k < model.min_sample_size:
            continue
        if considered is not None and k + 1 >= data_considered.shape[0]:
            continue
        if s < np.finfo(np.float32).resolution:
            continue
        nfa = ac_tester.nfa(model, k + 1, inliers_threshold=s,
                            data=data_considered)
        min_nfa = np.minimum(nfa, min_nfa)
    return min_nfa + np.log10(len(dist) - model.min_sample_size)


def exclusion_principle(ac_tester, mod_inliers_list):
    nfa_list = [(i, optimal_nfa(ac_tester, mod, in_a))
                for i, (mod, in_a) in enumerate(mod_inliers_list)]
    nfa_list = filter(lambda e: meaningful(e[1], ac_tester.epsilon), nfa_list)
    nfa_list = sorted(nfa_list, key=operator.itemgetter(1))
    idx = zip(*nfa_list)[0]

    keep_list = list(idx)
    for pick in idx:
        mod, in_a = mod_inliers_list[pick]
        in_list = [mod_inliers_list[k][1] for k in keep_list if k < pick]
        if not in_list:
            continue
        excluded = reduce(lambda x, y: np.logical_or(x, y), in_list)
        considered = np.logical_not(excluded)
        inliers = in_a - np.logical_and(in_a, excluded)
        nfa = optimal_nfa(ac_tester, mod, inliers, considered=considered)
        if not meaningful(nfa, ac_tester.epsilon):
            keep_list.remove(pick)

    return keep_list

