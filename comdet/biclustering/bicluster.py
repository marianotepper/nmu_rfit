from __future__ import absolute_import
import numpy as np
from . import utils
from . import nmf
from . import mdl
from . import deflation


def single_bicluster(deflator):
    try:
        u, v = nmf.nmf_robust_multiplicative(deflator.compressed_array, 1)
        # u, v = nmf.nmf_robust_rank1(deflator.array_compressed)
        u = utils.binarize(u)
        idx_u = utils.find(u)[0]

        array_cropped = deflator.array[idx_u, :]
        u_cropped = utils.sparse(np.ones((idx_u.size, 1)))
        v_init = u_cropped.T.dot(array_cropped)
        v_init /= v_init.max()
        v = nmf.nmf_robust_admm_v(array_cropped, u_cropped, v_init)
        v = utils.binarize(v)
        idx_v = utils.find(v)[1]

        array_cropped = deflator.array[:, idx_v]
        v_cropped = utils.sparse(np.ones((1, idx_v.size)))
        u_init = array_cropped.dot(v_cropped.T)
        u_init /= u_init.max()
        u = nmf.nmf_robust_admm_u(array_cropped, u_init, v_cropped)
        u = utils.binarize(u)

    except deflation.DeflationError:
        u, v = nmf.nmf_robust_admm(deflator.array)
        u = utils.binarize(u)
        v = utils.binarize(v)
    return u, v


def bicluster(deflator, n=None, share_elements=True):
    bic_list = []
    total_codelength = []

    if n is None:
        n_iters = deflator.array.shape[1]
        online_mdl = mdl.OnlineMDL()
    else:
        n_iters = n

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

    if not bic_list:
        return bic_list
    if n is None:
        total_codelength = np.array(total_codelength)
        cut_point = np.argmin(total_codelength)
        return bic_list[:cut_point+1]
    else:
        return bic_list