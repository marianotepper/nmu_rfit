import numpy as np
from scipy.sparse import issparse, spmatrix
from multiprocessing import Pool
from functools import partial
from multipledispatch import dispatch
from comdet.biclustering.utils import count_nonzero


def quantize(arr, q):
    """
    Quantizes a vector/matrix A to precision q.
    The resulting matrix has integer values which, when multiplied by q,
    give the quantized entries of A.
    The function returns integer so that they can be easily encoded later.
    """
    if q > 0.:
        return (arr / q).rint()
    else:
        return arr


log2_2pi = np.log2(2 * np.pi)


@dispatch(object)
def universal_bernoulli(arr):
    """
    universal codelength for an IID Bernoulli sequence .
    Uses the approximate closed form expression (using Stirling's
    formula) of the enumerative code for Bernoulli sequences (Cover'91)
    """
    n = reduce(lambda x, y: x * y, arr.shape)
    k = count_nonzero(arr)
    return universal_bernoulli(n, k)


@dispatch(int, int)
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


def universal_bernoulli_uniform(arr, q):
    """
    universal codelength for encoding a matrix with sparse
    entries, where the non-zeros are uniformly distributed in
    the subset of integers {-Q,-Q+1,...,-1,1,...,Q-1,Q}
    This is naturally described as the entries of A being
    Bernoulli-Uniform variables, which can be described with no loss
    in two parts: first the Bernoulli part using en enumerative code
    gives the locations of the non-zeros, then the non-zeros are
    described with ceil(log_2(Q))+1 bits each
    """
    if issparse(arr):
        k = arr.nnz
    else:
        k = np.count_nonzero(arr)
    code_ber = universal_bernoulli(arr)
    if k > 0 and q > 0:
        code_uni = k * (np.ceil(np.log2(q)) + 1)
    else:
        code_uni = 0
    return code_ber + code_uni


def codelength_fixed_precision(arr, u, v, q):
    """
    Compute the codelength of encoding a low rank binary matrix A
    as E,uq,vq where
    uq = [u]_q, u quantized uniformly to a precision of q
    vq = [v]_q
    E = [A - uq * vq']_1, where [1] is binary thresholding
    """
    uq = quantize(u, q)
    vq = quantize(v, q)
    max_valq = max((np.max(uq), np.max(vq)))
    diff_binary = quantize(arr - q * uq.dot(q * vq), q)
    code_diff = universal_bernoulli_uniform(diff_binary, np.max(diff_binary))
    code_u = universal_bernoulli_uniform(uq, max_valq)
    code_v = universal_bernoulli_uniform(vq, max_valq)
    return code_diff + code_u + code_v


def codelength(arr, u, v, limits=(-6, 1), pool_size=0):
    """
    Compute the codelength of encoding a low rank binary matrix A
    as E,uq,vq where
    uq = [u]_q, u quantized uniformly to a precision of q
    vq = [v]_q
    E = [A - uq * vq']_1, where [1] is binary thresholding
    """
    if pool_size > 0:
        pool = Pool(pool_size)
        map_fun = pool.map
    else:
        map_fun = map

    qtop = np.ceil(np.log2(np.max(arr.flat)))
    qs = np.power(2, np.arange(qtop + limits[0], qtop + limits[1]))
    cl_fp = partial(codelength_fixed_precision, arr, u, v)
    lengths = map_fun(cl_fp, qs)
    k, cl_min = min(enumerate(lengths), key=lambda e: e[1])
    if pool_size > 0:
        pool.close()

    return cl_min


def codelengths_binary(arr, u, v):
    """
    Compute the codelength of encoding a low rank binary matrix A
    as E,uq,vq where
    uq = [u]_q, u quantized uniformly to a precision of q
    vq = [v]_q
    E = [A - uq * vq']_1, where [1] is binary thresholding
    """
    code_u = universal_bernoulli(u)
    code_v = universal_bernoulli(v)
    code_diff = universal_bernoulli(arr - u.dot(v))
    return code_u, code_v, code_diff


class OnlineMDL:

    def __init__(self):
        self.u_nnz = 0
        self.u_count = 0
        self.v_nnz = 0
        self.v_count = 0

    def add_rank1_approximation(self, remainder, u, v):
        u_n = reduce(lambda x, y: x * y, u.shape)
        self.u_count += 1
        self.u_nnz += count_nonzero(u)
        code_u = universal_bernoulli(u_n * self.u_count, self.u_nnz)
        v_n = reduce(lambda x, y: x * y, v.shape)
        self.v_count += 1
        self.v_nnz += count_nonzero(v)
        code_v = universal_bernoulli(v_n * self.v_count, self.v_nnz)
        code_diff = universal_bernoulli(remainder)
        return code_u + code_v + code_diff