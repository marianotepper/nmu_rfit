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


def select_leverage_scores(projected_mat, s, axis=0):
    leverage_scores = np.linalg.norm(projected_mat, 1, axis=1 - axis)
    sum_ls = np.sum(leverage_scores)
    if sum_ls == 0:
        return None
    # leverage_scores /= sum
    idx = leverage_scores.size - s
    selection = np.argpartition(leverage_scores, idx)[idx:]
    return selection


def compress_columns(array, n_samples, rcond=1e-10):
    mat = power_of_two_padding(array, axis=1)
    transform = fct.fast_cauchy_transform(mat.shape[1], n_samples, n_samples)

    subsampled_mat = mat.dot(transform.T).toarray()
    u, s, _ = np.linalg.svd(subsampled_mat, full_matrices=False)
    mask = np.abs(s) > rcond * np.max(s)
    s[mask] = 1. / s[mask]
    s[np.logical_not(mask)] = 0
    r_inv = utils.sparse(u * s)

    projected_mat = r_inv.T.dot(array).toarray()
    selection = select_leverage_scores(projected_mat, n_samples, axis=1)
    if selection is None or n_samples > selection.size:
        print(selection is None, array.nnz, array.size)
        return array
    else:
        return array[:, selection]


def compress_rows(array, n_samples, rcond=1e-10):
    mat = power_of_two_padding(array, axis=0)
    transform = fct.fast_cauchy_transform(mat.shape[0], n_samples, n_samples)

    subsampled_mat = transform.dot(mat).toarray()
    u, s, v = np.linalg.svd(subsampled_mat, full_matrices=False)
    mask = np.abs(s) > rcond * np.max(s)
    s[mask] = 1. / s[mask]
    s[np.logical_not(mask)] = 0

    r_inv = utils.sparse(v.T * s)
    projected_mat = array.dot(r_inv).toarray()
    selection = select_leverage_scores(projected_mat, n_samples, axis=0)
    if selection is None or n_samples > selection.size:
        return array
    else:
        return array[selection, :]
