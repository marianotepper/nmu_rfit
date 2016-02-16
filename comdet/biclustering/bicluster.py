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
    for _ in range(n_iters):
        if downdater.array.nnz == 0:
            break

        u, v = single_bicluster(downdater.array, comp_level=comp_level)

        if v.nnz <= 1:
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

    u, _ = nmf.nmf_robust_multiplicative(array_nmf, 1)
    u = utils.binarize(u)

    if u.nnz == 0:
        return u, utils.sparse((1, array.shape[1]))

    v = update_right(array, u, comp_level=comp_level)

    if v.nnz == 0:
        return u, v

    u = update_left(array, v, comp_level=comp_level)

    return u, v


def update_right(array, u, comp_level=None):
    idx_u = utils.find(u)[0]
    array_crop = utils.sparse(array[idx_u, :])

    if comp_level is not None and comp_level < array_crop.shape[0]:
        array_crop = compression.compress_rows(array_crop, comp_level)

    u_crop = utils.sparse(np.ones((array_crop.shape[0], 1)))
    v_init = u_crop.T.dot(array_crop)
    v_init = (v_init / array_crop.shape[0]).rint()
    v_init = utils.binarize(v_init)

    v = nmf.nmf_robust_admm(array_crop, u_init=u_crop, v_init=v_init,
                            update='right')
    return utils.binarize(v)


def update_left(array, v, comp_level=None):
    idx_v = utils.find(v)[1]
    array_crop = utils.sparse(array[:, idx_v])

    if comp_level is not None and comp_level < array_crop.shape[1]:
        array_crop = compression.compress_columns(array_crop, comp_level)

    v_crop = utils.sparse(np.ones((1, array_crop.shape[1])))
    u_init = array_crop.dot(v_crop.T)
    u_init = (u_init / array_crop.shape[1]).rint()
    u_init = utils.binarize(u_init)

    u = nmf.nmf_robust_admm(array_crop, u_init=u_init, v_init=v_crop,
                            update='left')
    return utils.binarize(u)
