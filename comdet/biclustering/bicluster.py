from __future__ import absolute_import
import numpy as np
from . import utils
from . import nmf
from . import mdl
from . import deflation


def bicluster(deflator, n=None, share_elements=True):
    if n is None:
        n_iters = deflator.array.shape[1]
        online_mdl = mdl.OnlineMDL()
    else:
        n_iters = n

    bic_list = []
    total_codelength = []
    for _ in range(n_iters):
        if deflator.array.nnz == 0:
            break

        u, v = single_bicluster(deflator)

        if u.nnz <= 1 or v.nnz <= 1:
            break

        bic_list.append((u, v))

        idx_v = utils.find(v)[1]
        deflator.remove_columns(idx_v)
        if not share_elements:
            idx_u = utils.find(u)[0]
            deflator.remove_rows(idx_u)

        if n is None:
            cl = online_mdl.add_rank1_approximation(deflator.array, u, v)
            total_codelength.append(cl)

    if n is None and bic_list:
        cut_point = np.argmin(np.array(total_codelength))
        bic_list = bic_list[:cut_point+1]

    return bic_list


def single_bicluster(deflator):
    try:
        array = deflator.compressed_array
    except deflation.DeflationError:
        array = deflator.array

    u, v = nmf.nmf_robust_multiplicative(array, 1)
    u = utils.binarize(u)
    idx_u, _, vals_u = utils.find(u)

    array_crop, u_crop, v_init = crop_left(deflator.array, idx_u)
    v = nmf.nmf_robust_admm(array_crop, u_init=u_crop, v_init=v_init,
                            update='right')
    v = utils.binarize(v)
    idx_v = utils.find(v)[1]

    array_crop, u_init, v_crop = crop_right(deflator.array, idx_v)
    u = nmf.nmf_robust_admm(array_crop, u_init=u_init, v_init=v_crop,
                            update='left')
    u = utils.binarize(u)
    return u, v


def crop_left(array, idx_u):
    array_cropped = utils.sparse(array[idx_u, :])
    u_cropped = utils.sparse(np.ones((idx_u.size, 1)))
    v_init = u_cropped.T.dot(array_cropped)
    arr_sum = array_cropped.sum(axis=0)
    v_init[arr_sum > 0] /= arr_sum[arr_sum > 0]
    return array_cropped, u_cropped, v_init


def crop_right(array, idx_v):
    array_cropped = utils.sparse(array[:, idx_v])
    v_cropped = utils.sparse(np.ones((1, idx_v.size)))
    u_init = array_cropped.dot(v_cropped.T)
    arr_sum = array_cropped.sum(axis=1)
    u_init[arr_sum > 0] /= arr_sum[arr_sum > 0]
    return array_cropped, u_init, v_cropped
