import numpy as np
from scipy.special import gammaln, smirnov
from scipy.stats import kstest


def log_nchoosek(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def n_tests(n, ms_size, trim=False):
    if trim:
        n -= ms_size
    return log_nchoosek(n, ms_size) / np.log(10)


def concentration_pfa(membership, ms_size, trim=False):
    membership = membership[membership > 0]
    if trim:
        ones = np.where(membership > 1 - 1e-5)[0]
        membership = np.delete(membership, ones[-min(len(ones), ms_size):])
    if len(membership) > 1:
        d_min, _ = kstest(membership, 'uniform', alternative='less')
        pvalue = smirnov(len(membership), d_min)
    else:
        pvalue = 1.
    if pvalue == 0:
        return -300
    else:
        return np.log10(pvalue)


def concentration_nfa(membership, ms_size, trim=False):
    nt = n_tests(len(membership), ms_size, trim=trim)
    return nt + concentration_pfa(membership, ms_size, trim=trim)


def meaningful(membership, mss, log_epsilon=0, trim=False):
    nfa = concentration_nfa(membership, mss, trim=trim)
    return nfa < log_epsilon
