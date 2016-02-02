from __future__ import absolute_import
import numpy as np
import multipledispatch
import scipy.sparse as sp
import collections
from . import fct
from . import utils


@multipledispatch.dispatch(sp.spmatrix)
def power_of_two_padding(mat, axis=0):
    new_shape = list(mat.shape)
    new_size = 2**np.ceil(np.log2(mat.shape[axis]))
    new_shape[axis] = new_size - new_shape[axis]
    new_shape = tuple(new_shape)
    if new_shape[axis] > 0:
        if axis == 0:
            stack = sp.vstack
        else:
            stack = sp.hstack
        return utils.sparse(stack((mat, utils.sparse(new_shape))))
    else:
        return mat


@multipledispatch.dispatch(np.ndarray)
def power_of_two_padding(mat, axis=0):
    new_size = 2**np.ceil(np.log2(mat.shape[axis]))
    if new_size > mat.shape[0]:
        shape = mat.shape
        shape[axis] = new_size
        return mat.resize(shape)
    else:
        return mat


def select_leverage_scores(projected_mat, s, original_dim_size, axis=0):
    leverage_scores = np.linalg.norm(projected_mat, 1, axis=1 - axis)
    leverage_scores = leverage_scores[:original_dim_size]
    sum_ls = np.sum(leverage_scores)
    if sum_ls == 0:
        return None
    # leverage_scores /= sum
    idx = leverage_scores.size - s
    selection = np.argpartition(leverage_scores, idx)[idx:]
    return selection


class OnlineColumnCompressor(object):
    def __init__(self, array, n_samples):
        super(OnlineColumnCompressor, self).__init__()
        self.n_samples = n_samples
        self.ncols_original = array.shape[1]

        mat = power_of_two_padding(array, axis=1)
        self.fct = fct.fast_cauchy_transform(mat.shape[1], n_samples, n_samples)
        self.downdater = utils.Downdater(mat)

    def compress(self):
        r_inv = utils.sparse(self._invert_r())
        projected_mat = r_inv.T.dot(self.downdater.array).toarray()
        selection = select_leverage_scores(projected_mat, self.n_samples,
                                           self.ncols_original, axis=1)
        return selection

    def _invert_r(self, rcond=1e-10):
        subsampled_mat = self.downdater.array.dot(self.fct.T).toarray()
        u, s, _ = np.linalg.svd(subsampled_mat, full_matrices=False)
        mask = np.abs(s) > rcond * np.max(s)
        return u[:, mask] / s[mask]

    def additive_downdate(self, u, v, apply_to_matrix=True):
        v_padded = power_of_two_padding(v, axis=1)
        self.downdater.additive_downdate(u, v_padded)

    def remove_columns(self, idx_cols):
        self.downdater.remove_columns(idx_cols)

    def remove_rows(self, idx_rows):
        self.downdater.remove_rows(idx_rows)
