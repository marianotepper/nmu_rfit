import numpy as np
import scipy.special as special
from abc import ABCMeta, abstractmethod


class NFA(object):
    __metaclass__ = ABCMeta

    def __init__(self, epsilon, proba, min_sample_size):
        self.epsilon = epsilon
        self.proba = proba
        self.min_sample_size = min_sample_size

    def nfa(self, membership):
        return (self._pfa(membership) + self._n_tests(membership)) / np.log(10)

    @abstractmethod
    def _n_tests(self, membership):
        pass

    @abstractmethod
    def _pfa(self, membership):
        pass

    def meaningful(self, membership):
        return self.nfa(membership) < self.epsilon


class BinomialNFA(NFA):
    def __init__(self, epsilon, proba, min_sample_size):
        super(BinomialNFA, self).__init__(epsilon, proba, min_sample_size)

    def _n_tests(self, membership):
        return log_nchoosek(membership.size, self.min_sample_size)

    def _pfa(self, membership):
        n = membership.size - np.isnan(membership).sum() - self.min_sample_size
        k = (membership > 0).sum() - self.min_sample_size
        if k <= 0:
            return np.inf
        if n == k:
            return -np.inf
        return log_binomial(n, k, self.proba)


class ImageTransformNFA(NFA):
    def __init__(self, epsilon, proba, min_sample_size, n_tests_factor=1):
        super(ImageTransformNFA, self).__init__(epsilon, proba, min_sample_size)
        self.n_tests_factor = n_tests_factor

    def _n_tests(self, membership):
        k = np.maximum((membership > 0).sum(), self.min_sample_size)
        return (log_nchoosek(membership.size, k) +
                log_nchoosek(k, self.min_sample_size) +
                np.log(membership.size - self.min_sample_size) +
                np.log(self.n_tests_factor))

    def _pfa(self, membership):
        k = (membership > 0).sum() - self.min_sample_size
        if k <= 0:
            return np.inf
        proba = np.nanmax(membership) * self.proba
        return k * np.log(proba)


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


def log_gammainc(a, x):
    eps = np.finfo(np.float).eps
    fp_min = np.finfo(np.float).tiny / eps
    b = x + 1.0 - a
    c = 1.0 / fp_min
    d = 1.0 / b
    log_h = np.log(d)
    i = 1
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if np.abs(d) < fp_min:
            d = fp_min
        c = b + an / c
        if np.abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        log_h += np.log(delta)
        if np.abs(delta - 1.0) <= eps:
            break
        i += 1
    return -x + a * np.log(x) - special.gammaln(a) + log_h
