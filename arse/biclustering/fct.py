from __future__ import absolute_import
import numpy as np
import scipy.sparse as sp
import math
import scipy.linalg
from . import utils


def normalized_hadamard(n):
    return scipy.linalg.hadamard(n) / math.sqrt(n)


def create_blockdiagonal_matrix(func, n_blocks):
    diag = [utils.sparse(func()) for _ in range(n_blocks)]
    return sp.block_diag(diag)


def basis(k, m):
    rows = np.random.randint(k, size=(m,))
    cols = range(m)
    data = np.ones((m,))
    return utils.sparse((data, (rows, cols)))


def cauchy(m, block_side):
    def diag_cauchy():
        return np.diag(np.random.standard_cauchy((block_side,)))

    return create_blockdiagonal_matrix(diag_cauchy, m / block_side)


def spread_matrix(m, s):
    gs = np.vstack((normalized_hadamard(s), np.eye(s)))

    def create_block():
        return gs

    return create_blockdiagonal_matrix(create_block, m / s)


def fast_cauchy_transform(m, s, k):
    h = spread_matrix(m, s)
    d = cauchy(2 * m, 2 * s)
    b = basis(k, 2 * m)
    transform_mat = 4 * b.dot(d.dot(h))
    return transform_mat
