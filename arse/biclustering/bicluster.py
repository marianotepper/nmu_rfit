from __future__ import absolute_import
import numpy as np
from . import utils
from . import nmf
from . import mdl
from . import compression
import timeit


def bicluster(array, n=None, share_elements=True, comp_level=None):
    if n is None:
        n_iters = array.shape[1]
        online_mdl = mdl.OnlineMDL()
    else:
        n_iters = n

    downdater = utils.Downdater(array)

    total_time = 0
    bic_list = []
    total_codelength = []
    for i in range(n_iters):
        if downdater.array.nnz == 0:
            break

        t = timeit.default_timer()
        u, v = single_bicluster(downdater.array, comp_level=comp_level)
        total_time += timeit.default_timer() - t
        print i, u.nnz, v.nnz

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

    print total_time

    if n is None and bic_list:
        cut_point = np.argmin(np.array(total_codelength))
        bic_list = bic_list[:cut_point+1]

    return bic_list


def single_bicluster(array, comp_level=None):
    if comp_level is not None and comp_level < array.shape[1]:
        selection = compression.compress_columns(array, comp_level)
        if selection is not None:
            array_nmf = array[:, selection]
    else:
        array_nmf = array

    u, _ = nmf.nmf_robust_multiplicative(array_nmf, 1)
    u = utils.sparsify(u)

    if u.nnz == 0:
        return u, utils.sparse((1, array.shape[1]))

    v = update_right(array, u, comp_level=comp_level)

    if v.nnz == 0:
        return u, v

    u = update_left(array, v, comp_level=comp_level)

    return utils.sparsify(u, dtype=bool), utils.sparsify(v, dtype=bool)


def update_right(array, u, comp_level=None):
    idx_u, _, vals_u = utils.find(u)
    array_crop = utils.sparse(array[idx_u, :])

    if comp_level is not None and comp_level < array_crop.shape[0]:
        selection = compression.compress_rows(array_crop, comp_level)
        if selection is not None:
            array_crop = array_crop[selection, :]
            vals_u = vals_u[selection]

    v = nmf.nmf_robust_admm(array_crop, u_init=vals_u, update='right', tol=1e-2)
    return utils.sparsify(v)


def update_left(array, v, comp_level=None):
    _, idx_v, vals_v = utils.find(v)
    array_crop = utils.sparse(array[:, idx_v])

    if comp_level is not None and comp_level < array_crop.shape[1]:
        selection = compression.compress_columns(array_crop, comp_level)
        if selection is not None:
            array_crop = array_crop[:, selection]
            vals_v = vals_v[selection]

    u = nmf.nmf_robust_admm(array_crop, v_init=vals_v, update='left', tol=1e-2)
    return utils.sparsify(u)
