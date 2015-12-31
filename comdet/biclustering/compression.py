from __future__ import absolute_import
import numpy as np
import multipledispatch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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


class Downdate:
    additive = 0
    column_removal = 1
    row_removal = 2

    def __init__(self, type, param):
        self.type = type
        self.param = param


class OnlineRowCompressor(object):
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
        if self.svd.s is None:
            return None
        self.apply_downdates()
        r_inv = self._invert_r()
        projected_mat = self.mat.dot(utils.sparse(r_inv)).toarray()
        selection = select_leverage_scores(projected_mat, self.n_samples,
                                           self.nrows_original)
        # TODO scale nonzero rows
        return selection

    def _invert_r(self, rcond=1e-10):
        mask = np.abs(self.svd.s) > rcond * np.max(self.svd.s)
        s = 1. / self.svd.s[mask]
        vt = self.svd.vt[mask, :]
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


class OnlineColumnCompressor(object):
    def __init__(self, array, n_samples):
        self.n_samples = n_samples
        self.ncols_original = array.shape[1]

        self.mat = power_of_two_padding(array, axis=1)
        self.mat_lil = self.mat.tolil()
        self.downdates = []

        self.transform_mat = fct.fast_cauchy_transform(self.mat.shape[1],
                                                       n_samples, n_samples)
        subsampled_mat = self.mat.dot(self.transform_mat.T).toarray()
        self.svd = utils.UpdatableSVD(subsampled_mat)

    def compress(self):
        if self.svd.s is None:
            selection = None
        else:
            self.apply_downdates()
            r_inv = utils.sparse(self._invert_r())
            projected_mat = r_inv.T.dot(self.mat).toarray()
            selection = select_leverage_scores(projected_mat, self.n_samples,
                                               self.ncols_original, axis=1)
        return selection

    def _invert_r(self, rcond=1e-10):
        mask = np.abs(self.svd.s) > rcond * np.max(self.svd.s)
        s = 1. / self.svd.s[mask]
        u = self.svd.u[:, mask]
        # u, s, vt = spla.svds(self.mat, self.n_samples)
        # mask = np.abs(s) > rcond * np.max(s)
        # s = 1. / s[mask]
        # u = u[:, mask]
        return u * s

    def additive_downdate(self, u, v, apply_to_matrix=True):
        v_padded = power_of_two_padding(v, axis=1)
        v_subsampled = -v_padded.dot(self.transform_mat.T)
        self.svd.update(np.squeeze(u.toarray()),
                        np.squeeze(v_subsampled.toarray()))
        if apply_to_matrix:
            dd = Downdate(Downdate.additive, u * v_padded)
            self.downdates.append(dd)

    def remove_column(self, idx):
        u = self.mat[:, idx]
        if u.min() == 0 and u.max() == 0:
            return
        v = utils.sparse(([1], ([0], [idx])), shape=(1, self.mat.shape[1]))
        self.additive_downdate(u, v, apply_to_matrix=False)
        dd = Downdate(Downdate.column_removal, idx)
        self.downdates.append(dd)

    def remove_row(self, idx):
        self.svd.remove_row(idx)
        dd = Downdate(Downdate.row_removal, idx)
        self.downdates.append(dd)

    def apply_downdates(self):
        if len(self.downdates) == 0:
            return
        current_type = self.downdates[0].type
        for dd in self.downdates:
            if dd.type != current_type:
                if dd.type == Downdate.row_removal:
                    self.mat_lil = self.mat.tolil()
                if dd.type == Downdate.column_removal:
                    self.mat_lil = self.mat.tolil()
                if dd.type == Downdate.additive:
                    self.mat = self.mat_lil.tocsc()
                current_type = dd.type

            if dd.type == current_type:
                if dd.type == Downdate.row_removal:
                    self.mat_lil[dd.param, :] = 0
                if dd.type == Downdate.column_removal:
                    self.mat_lil[:, dd.param] = 0
                if dd.type == Downdate.additive:
                    self.mat = self.mat + dd.param

        if current_type == Downdate.column_removal:
            self.mat = self.mat_lil.tocsc()
        if current_type == Downdate.additive:
            self.mat_lil = self.mat.tolil()