import numpy as np
import scipy.special as special
import abc
import itertools


def ifilter(ac_tester, model_generator):
    def inner_meaningful(model):
        return ac_tester.meaningful(model)
    return itertools.ifilter(inner_meaningful, model_generator)


def meaningful(value, epsilon):
    return value < epsilon


class BinomialNFA(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, epsilon):
        self.data = data
        self.epsilon = epsilon

    def nfa(self, model, considered=None, inliers_threshold=None):
        if considered is None:
            data = self.data
        else:
            data = self.data[considered]
        if inliers_threshold is None:
            inliers_threshold = self.threshold(model)
        if considered is None:
            ratios = [2]
        else:
            ratios = [2, 3, 4]
        pfa = self._pfa(model, data, inliers_threshold, ratios=ratios)
        n_tests = log_nchoosek(len(self.data), model.min_sample_size)
        n_tests += np.log(len(ratios))
        # if considered is not None:
        #     print considered.sum(), n, k, p, pfa, n_tests, (pfa + n_tests) / np.log(10)
        return (pfa + n_tests) / np.log(10)

    def _pfa(self, model, data, inliers_threshold, ratios=[2]):
        pfa_list = []
        for r in ratios:
            n, k, p = self._binomial_params(model, data, inliers_threshold,
                                            ratio=r)
            if k <= 0:
                return np.inf
            if n == k:
                return -np.inf
            pfa_list.append(log_binomial(n, k, p))
        return min(pfa_list)

    def meaningful(self, model, considered=None, inliers_threshold=None):
        nfa = self.nfa(model, considered=considered,
                       inliers_threshold=inliers_threshold)
        return nfa < self.epsilon

    @abc.abstractmethod
    def threshold(self, model):
        pass

    @abc.abstractmethod
    def _binomial_params(self, model, data, inliers_threshold):
        pass

    def inliers(self, model, data=None, inliers_threshold=None):
        if data is None:
            data = self.data
        if inliers_threshold is None:
            inliers_threshold = self.threshold(model)
        return np.abs(model.distances(data)) <= inliers_threshold


class LocalNFA(BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LocalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold, ratio=2.):
        dist = model.distances(data)
        dist_abs = np.abs(dist)
        inliers = dist_abs <= inliers_threshold
        k = inliers.sum()
        outliers = dist_abs > inliers_threshold
        if outliers.sum() == 0:
            k -= model.min_sample_size
            return k, k, 1

        upper_threshold = np.maximum(inliers_threshold * (ratio + 1),
                                     np.min(dist_abs[outliers]))
        region1 = np.logical_and(dist >= -upper_threshold,
                                 dist < -inliers_threshold)
        region2 = np.logical_and(dist <= upper_threshold,
                                 dist > inliers_threshold)
        n1 = region1.sum()
        n2 = region2.sum()

        if n1 == 0 or n2 == 0:
            n = np.maximum(n1, n2) + k
            p = 1. / ratio
        else:
            n = n1 + n2 + k
            p = 1. / (ratio + 1)
        k -= model.min_sample_size
        return n, k, p

    def threshold(self, model):
        return self.inliers_threshold


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
