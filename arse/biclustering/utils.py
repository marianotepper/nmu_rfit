import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import multipledispatch


def sparse(*args, **kwargs):
    return sp.csc_matrix(*args, **kwargs)


def find(mat):
    return sp.find(mat)


def issparse(mat):
    return sp.issparse(mat)


def solve(mat, b):
    if mat.shape[0] == mat.shape[1]:
        return spla.spsolve(mat, b)
    else:
        return spla.lsqr(mat, b)[0]


def sparsify(x, tol=1e-4, dtype=None):
    i, j, v = find(x)
    if v.size == 0:
        return sparse(x.shape)
    if tol is None:
        thresh = 0
    else:
        thresh = tol * v.max()
    mask = v > thresh
    return sparse((v[mask], (i[mask], j[mask])), shape=x.shape, dtype=dtype)



@multipledispatch.dispatch(sp.spmatrix, sp.spmatrix)
def relative_error(mat_ref, mat, ord='fro'):
    if ord == 'fro':
        ord = 2
    temp = mat_ref - mat
    num = np.linalg.norm(sp.find(temp)[2], ord=ord)
    denom = np.linalg.norm(sp.find(mat_ref)[2], ord=ord)
    return num / denom


@multipledispatch.dispatch(np.ndarray, np.ndarray)
def relative_error(mat_ref, mat, ord='fro'):
    if ord == 0:
        def reshape(arr):
            return arr.flatten()
    else:
        def reshape(arr):
            return arr
    mat_ref = reshape(mat_ref)
    num = np.linalg.norm(mat_ref - reshape(mat), ord=ord)
    denom = np.linalg.norm(mat_ref, ord=ord)
    return num / denom


def normalized_error(mat1, mat2, ord='fro'):
    if sp.issparse(mat1) and sp.issparse(mat2):
        if ord == 'fro':
            ord = 2
        norm1 = np.linalg.norm(mat1.data, ord=ord)
        norm2 = np.linalg.norm(mat2.data, ord=ord)
        diff = (mat1 / norm1) - (mat2 / norm2)
        res = np.linalg.norm(diff.data, ord=ord)
    else:
        if ord == 0:
            def reshape(arr):
                return arr.flatten()
        else:
            def reshape(arr):
                return arr
        norm1 = np.linalg.norm(mat1, ord=ord)
        norm2 = np.linalg.norm(mat2, ord=ord)
        diff = (mat1 / norm1) - (mat2 / norm2)
        res = np.linalg.norm(diff, ord=ord)
    return res


@multipledispatch.dispatch(np.ndarray)
def count_nonzero(array):
    return np.count_nonzero(array)


@multipledispatch.dispatch(sp.spmatrix)
def count_nonzero(array):
    return array.nnz


@multipledispatch.dispatch(np.ndarray)
def norm(array, ord=None):
    return np.linalg.norm(array, ord=ord)


@multipledispatch.dispatch(sp.spmatrix)
def norm(array, ord=None):
    return np.linalg.norm(sp.find(array)[2], ord=ord)


def svds(array, k):
    success = False
    tol = 0
    while not success:
        if tol > 1e-3:
            raise spla.ArpackNoConvergence('SVD failed to converge with '
                                           'tol={0}'.format(tol))
        try:
            u, s, vt = spla.svds(array, k, tol=tol)
            success = True
        except spla.ArpackNoConvergence:
            if tol == 0:
                tol = 1e-10
            else:
                tol *= 10
    return u, s, vt


class Downdater(object):
    def __init__(self, array):
        self.array = array
        self._array_lil = None

    def additive_downdate(self, u, v):
        self.array -= u.dot(v)
        self._array_lil = None

    def remove_columns(self, idx_cols):
        if self._array_lil is None:
            self._array_lil = self.array.tolil()
        self._array_lil[:, idx_cols] = 0
        self.array = self._array_lil.tocsc()

    def remove_rows(self, idx_rows):
        if self._array_lil is None:
            self._array_lil = self.array.tolil()
        self._array_lil[idx_rows, :] = 0
        self.array = self._array_lil.tocsc()
