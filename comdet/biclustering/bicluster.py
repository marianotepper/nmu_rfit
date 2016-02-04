from __future__ import absolute_import
import numpy as np
from . import utils
from . import nmf
from . import mdl
from . import compression


def bicluster(array, n=None, share_elements=True, comp_level=None):
    if n is None:
        n_iters = array.shape[1]
        online_mdl = mdl.OnlineMDL()
    else:
        n_iters = n

    downdater = utils.Downdater(array)

    bic_list = []
    total_codelength = []
    for i in range(n_iters):
        if downdater.array.nnz == 0:
            break

        print(i)

        u, v = single_bicluster(downdater.array, comp_level=comp_level)

        if u.nnz <= 1 or v.nnz <= 1:
            break

        bic_list.append((u, v))

        idx_v = utils.find(v)[1]
        downdater.remove_columns(idx_v)
        if not share_elements:
            idx_u = utils.find(u)[0]
            downdater.remove_rows(idx_u)

        if n is None:
            cl = online_mdl.add_rank1_approximation(downdater.array, u, v)
            total_codelength.append(cl)

    if n is None and bic_list:
        cut_point = np.argmin(np.array(total_codelength))
        bic_list = bic_list[:cut_point+1]

    return bic_list


def single_bicluster(array, comp_level=None):
    if comp_level is not None:
        array_nmf = compression.compress_columns(array, comp_level)
    else:
        array_nmf = array
    u, v = nmf.nmf_robust_multiplicative(array_nmf, 1)
    u = utils.binarize(u)

    idx_u = utils.find(u)[0]
    array_crop, u_crop, v_init = crop_left(array, idx_u, comp_level=comp_level)
    v = nmf.nmf_robust_admm(array_crop, u_init=u_crop, v_init=v_init,
                            update='right')
    v = utils.binarize(v)

    idx_v = utils.find(v)[1]
    array_crop, u_init, v_crop = crop_right(array, idx_v, comp_level=comp_level)
    u = nmf.nmf_robust_admm(array_crop, u_init=u_init, v_init=v_crop,
                            update='left')
    u = utils.binarize(u)
    return u, v


def crop_left(array, idx_u, comp_level=None):
    array_crop = utils.sparse(array[idx_u, :])

    if comp_level is not None and comp_level < array_crop.shape[0]:
        array_crop = compression.compress_rows(array_crop, comp_level)

    u_crop = utils.sparse(np.ones((array_crop.shape[0], 1)))
    v_init = u_crop.T.dot(array_crop)
    arr_sum = array_crop.sum(axis=0)
    v_init[arr_sum > 0] /= arr_sum[arr_sum > 0]
    return array_crop, u_crop, v_init


def crop_right(array, idx_v, comp_level=None):
    array_crop = utils.sparse(array[:, idx_v])

    if comp_level is not None and comp_level < array_crop.shape[1]:
        array_crop = compression.compress_columns(array_crop, comp_level)

    v_crop = utils.sparse(np.ones((1, array_crop.shape[1])))
    u_init = array_crop.dot(v_crop.T)
    arr_sum = array_crop.sum(axis=1)
    u_init[arr_sum > 0] /= arr_sum[arr_sum > 0]
    return array_crop, u_init, v_crop
