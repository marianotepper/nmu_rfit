import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csc_matrix, spmatrix, find
from multipledispatch import dispatch


def sparse(*args, **kwargs):
    return csc_matrix(*args, **kwargs)
    # return coo_matrix(*args, **kwargs).tocsc()


@dispatch(spmatrix, spmatrix)
def relative_error(mat_ref, mat, ord='fro'):
    if ord == 'fro':
        ord = 2
    temp = mat_ref - mat
    num = norm(find(temp)[2], ord=ord)
    denom = norm(find(mat_ref)[2], ord=ord)
    return num / denom

@dispatch(np.ndarray, np.ndarray)
def relative_error(mat_ref, mat, ord='fro'):
    if ord == 0:
        def reshape(arr):
            return arr.flatten()
    else:
        def reshape(arr):
            return arr
    mat_ref_flat = reshape(mat_ref)
    num = norm(mat_ref_flat - reshape(mat), ord=ord)
    denom = norm(mat_ref_flat, ord=ord)
    return num / denom


def normalized_error(mat1, mat2, ord='fro'):
    if issparse(mat1) and issparse(mat2):
        if ord == 'fro':
            ord = 2
        norm1 = norm(mat1.data, ord=ord)
        norm2 = norm(mat2.data, ord=ord)
        diff = (mat1 / norm1) - (mat2 / norm2)
        res = norm(diff.data, ord=ord)
    else:
        if ord == 0:
            def reshape(arr):
                return arr.flatten()
        else:
            def reshape(arr):
                return arr
        norm1 = norm(mat1, ord=ord)
        norm2 = norm(mat2, ord=ord)
        diff = (mat1 / norm1) - (mat2 / norm2)
        res = norm(diff, ord=ord)
    return res


@dispatch(np.ndarray)
def count_nonzero(array):
    return np.count_nonzero(array)


@dispatch(spmatrix)
def count_nonzero(array):
    return array.nnz


@dispatch(np.ndarray)
def frobenius_norm(array):
    return np.linalg.norm(array)


@dispatch(spmatrix)
def frobenius_norm(array):
    return np.sqrt(array.multiply(array).sum())