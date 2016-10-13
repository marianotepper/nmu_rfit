import numpy as np
from scipy.special import gammaln, smirnov
from scipy.stats import kstest


def log_nchoosek(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def concentration_nfa(membership, ms_size, trim=True):
    n = len(membership)
    membership = membership[membership > 0]
    if trim:
        ones = np.where(membership == 1)[0]
        membership = np.delete(membership, ones[-min(len(ones), ms_size):])
    if len(membership) > 0:
        d_min, _ = kstest(membership, 'uniform', alternative='less')
        pvalue = smirnov(len(membership), d_min)
    else:
        pvalue = 1.
    return (log_nchoosek(n, ms_size) + np.log(pvalue)) / np.log(10)


def meaningful(membership, mss, log_epsilon=0, trim=True):
    nfa = concentration_nfa(membership, mss, trim=trim)
    return nfa < log_epsilon
