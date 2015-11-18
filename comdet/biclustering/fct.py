from __future__ import absolute_import
import numpy as np
from scipy.linalg import hadamard
from math import sqrt
from scipy.sparse import coo_matrix as sparse
from scipy.sparse import block_diag


def normalized_hadamard(n):
    return hadamard(n) / sqrt(n)


def create_blockdiagonal_matrix(func, blocksize, n_blocks):
    diag = [sparse(func()) for _ in range(n_blocks)]
    return block_diag(diag)


def basis(size):
    rows = np.random.randint(size[0], size=(size[1],))
    cols = range(size[1])
    data = np.ones((size[1],))
    return sparse((data, (rows, cols)))


def cauchy(m, block_side):
    block_size = (block_side, block_side)

    def diag_cauchy():
        return np.diag(np.random.standard_cauchy((block_side,)))

    return create_blockdiagonal_matrix(diag_cauchy, block_size, m / block_side)


def spread_matrix(m, s):

    gs = np.vstack((normalized_hadamard(s), np.eye(s)))

    def create_block():
        return gs

    block_size = (2*s, s)
    return create_blockdiagonal_matrix(create_block, block_size, m / s)


def fast_cauchy_transform(m, s, k):
    h = spread_matrix(m, s)
    d = cauchy(2*m, 2*s)
    b = basis((k, 2*m))
    transform_mat = 4 * b.dot(d.dot(h))
    return transform_mat
