import numpy as np
import scipy.special as special
import abc


class BinomialNFA(object):
    def __init__(self, epsilon, proba):
        self.epsilon = epsilon
        self.proba = proba

    def nfa(self, membership, min_sample_size):
        pfa = self._pfa(membership, min_sample_size)
        n_tests = log_nchoosek(membership.size, min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def _pfa(self, membership, min_sample_size):
        n = membership.size - np.isnan(membership).sum() - min_sample_size
        k = (membership > 0).sum() - min_sample_size
        if k <= 0:
            return np.inf
        if n == k:
            return -np.inf
        return log_binomial(n, k, self.proba)

    def meaningful(self, membership, min_sample_size):
        nfa = self.nfa(membership, min_sample_size)
        return nfa < self.epsilon


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
