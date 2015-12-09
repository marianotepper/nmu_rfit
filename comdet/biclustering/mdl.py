import numpy as np
import scipy.sparse as sp
import multiprocessing
import functools
import multipledispatch
import comdet.biclustering.utils as utils


log2_2pi = np.log2(2 * np.pi)


@multipledispatch.dispatch(object)
def universal_bernoulli(arr):
    """
    universal codelength for an IID Bernoulli sequence .
    Uses the approximate closed form expression (using Stirling's
    formula) of the enumerative code for Bernoulli sequences (Cover'91)
    """
    n = reduce(lambda x, y: x * y, arr.shape)
    k = utils.count_nonzero(arr)
    return universal_bernoulli(n, k)


@multipledispatch.dispatch(int, int)
def universal_bernoulli(n, k):
    """
    universal codelength for an IID Bernoulli sequence .
    Uses the approximate closed form expression (using Stirling's
    formula) of the enumerative code for Bernoulli sequences (Cover'91)
    """
    if n == 0:
        return 0  # null matrix: nothing to encode
    code = 0.5 * np.log2(n)  # parameter cost
    if k == n:
        pass
    elif k > 1:
        def f(x):
            return (x + 0.5) * np.log2(x)
        code += 0.5 * log2_2pi + f(n) - f(n - k) - f(k)  # general case
    elif k == 1:
        code += np.log2(n)  # parameter cost + value of parameter
    return code


class OnlineMDL:

    def __init__(self):
        self.code_u = 0
        self.code_v = 0

    def add_rank1_approximation(self, remainder, u, v):
        self.code_u += universal_bernoulli(u)
        self.code_v += universal_bernoulli(v)
        code_diff = universal_bernoulli(remainder)
        return self.code_u + self.code_v + code_diff
