from __future__ import absolute_import
import numpy as np
import multipledispatch
import scipy.sparse as sp
from . import  fct
from . import utils


@multipledispatch.dispatch(sp.spmatrix)
def power_of_two_padding(mat):
    new_shape = list(mat.shape)
    new_m = 2**np.ceil(np.log2(mat.shape[0]))
    new_shape[0] = new_m - new_shape[0]
    new_shape = tuple(new_shape)
    if new_shape[0] > 0:
        return utils.sparse(sp.vstack((mat, utils.sparse(new_shape))))
    else:
        return mat


@multipledispatch.dispatch(np.ndarray)
def power_of_two_padding(mat):
    new_shape = list(mat.shape)
    new_m = 2**np.ceil(np.log2(mat.shape[0]))
    new_shape[0] = new_m - new_shape[0]
    new_shape = tuple(new_shape)
    if new_shape[0] > 0:
        return np.vstack((mat, np.zeros(new_shape)))
    else:
        return mat


def invert_r(mat, rcond=1e-15):
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    mask = s > rcond * np.max(s)
    s = 1. / s[mask]
    vt = vt[mask, :]
    return vt.T * s


def select_leverage_scores(projected_mat, s, nrows_original):
    leverage_scores = np.linalg.norm(projected_mat, 1, axis=1)
    leverage_scores = leverage_scores[:nrows_original]
    sum_ls = np.sum(leverage_scores)
    if sum_ls == 0:
        return None
    # leverage_scores /= sum
    idx = leverage_scores.size - s
    selection = np.argpartition(leverage_scores, idx)[idx:]
    # print '#########', selection.size, np.min(leverage_scores[selection]), np.max(leverage_scores[selection])
    return selection


def compress(array, s, roc='rows'):
    if roc == 'cols':
        array = array.T
    nrows_original = array.shape[0]
    mat = power_of_two_padding(array)
    m = mat.shape[0]

    transform_mat = fct.fast_cauchy_transform(m, s, s)
    subsampled_mat = transform_mat.dot(mat).toarray()

    r_inv = invert_r(subsampled_mat)
    projected_mat = np.dot(mat, utils.sparse(r_inv)).toarray()
    selection = select_leverage_scores(projected_mat, s, nrows_original)

    mat2 = array.tocsr()
    mat2 = mat2[selection, :]
    # TODO scale rows that are not zero
    return mat2, selection


class Downdate:

    additive = 0
    column_removal = 1

    def __init__(self, type, param):
        self.type = type
        self.param = param


class OnlineColumnCompressor:

    def __init__(self, array, n_samples):
        self.n_samples = n_samples
        self.nrows_original = array.shape[0]

        self.mat = power_of_two_padding(array)
        self.mat_lil = self.mat.tolil()
        self.downdates = []

        self.transform_mat = fct.fast_cauchy_transform(self.mat.shape[0],
                                                       n_samples, n_samples)
        subsampled_mat = self.transform_mat.dot(self.mat).toarray()
        self.svd = utils.UpdatableSVD(subsampled_mat)

    def compress(self):
        self.apply_downdates()
        self.svd.trim()
        r_inv = OnlineColumnCompressor._invert_r(self.svd)
        projected_mat = self.mat.dot(utils.sparse(r_inv)).toarray()
        selection = select_leverage_scores(projected_mat, self.n_samples,
                                           self.nrows_original)
        # TODO scale nonzero rows
        return selection

    @staticmethod
    def _invert_r(svd, rcond=1e-10):
        mask = svd.s > rcond * np.max(svd.s)
        s = 1. / svd.s[mask]
        vt = svd.vt[mask, :]
        return vt.T * s

    def additive_downdate(self, u, v):
        u_padded = power_of_two_padding(u)
        u_subsampled = -self.transform_mat.dot(u_padded)
        self.svd.update(np.squeeze(u_subsampled.toarray()),
                        np.squeeze(v.toarray()))
        dd = Downdate(Downdate.additive, u_padded * v)
        self.downdates.append(dd)

    def remove_column(self, idx):
        self.svd.remove_column(idx)
        dd = Downdate(Downdate.column_removal, idx)
        self.downdates.append(dd)

    def remove_row(self, idx):
        v = self.mat[idx, :]
        if v.min() == 0 and v.max() == 0:
            return
        u = utils.sparse(([1], ([idx], [0])), shape=(self.nrows_original, 1))
        self.additive_downdate(u, v)

    def apply_downdates(self):
        if len(self.downdates) == 0:
            return
        current_type = self.downdates[0].type
        for dd in self.downdates:
            if dd.type != current_type:
                if dd.type == Downdate.column_removal:
                    self.mat_lil = self.mat.tolil()
                if dd.type == Downdate.additive:
                    self.mat = self.mat_lil.tocsc()
                current_type = dd.type

            if dd.type == current_type:
                if dd.type == Downdate.column_removal:
                    self.mat_lil[:, dd.param] = 0
                if dd.type == Downdate.additive:
                    self.mat = self.mat + dd.param

        if current_type == Downdate.column_removal:
            self.mat = self.mat_lil.tocsc()
        if current_type == Downdate.additive:
            self.mat_lil = self.mat.tolil()