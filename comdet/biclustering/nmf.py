from __future__ import absolute_import
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from . import utils
from . import mdl


def nmf_robust_rank1(array, lambda_u=1, lambda_v=1, lambda_e=1, u_init=None,
                     v_init=None, max_iter=5e2):
    if u_init is None and v_init is None:
        x, s, y = spla.svds(array, 1)
        s = np.sqrt(s)
        x = s * np.abs(x)
        y = s * np.abs(y)
    else:
        x = u_init.copy()
        y = v_init.copy()

    x = utils.sparse(x)
    y = utils.sparse(y)
    u = x
    v = y
    e = utils.sparse(array.shape)
    gamma_u = utils.sparse(u.shape)
    gamma_v = utils.sparse(v.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for _ in range(int(max_iter)):
        temp = array - e
        num_x = (lambda_e * temp.dot(y.T) + lambda_u * u - gamma_u +
                 gamma_e.dot(y.T))
        denom_x = y.dot(y.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x

        num_y = (lambda_e * x.T.dot(temp) + lambda_v * v - gamma_v +
                 x.T.dot(gamma_e))
        denom_y = x.T.dot(x).toarray()[0, 0] + lambda_v
        y = num_y / denom_y

        u = projection_positive(x + gamma_u / lambda_u)
        v = projection_positive(y + gamma_v / lambda_v)
        gamma_u += lambda_u * (x - u)
        gamma_v += lambda_v * (y - v)

        xy = x.dot(y)
        temp = array - xy
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, xy + e))
        if error[-1] < 1e-4:
            break
    return u, v


def nmf_robust_rank1_u(array, u_init, v, lambda_u=1, lambda_e=1, max_iter=5e2):
    u = u_init
    e = utils.sparse(array.shape)
    gamma_u = utils.sparse(u.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for _ in range(int(max_iter)):
        temp = array - e
        num_x = (lambda_e * temp.dot(v.T) + lambda_u * u - gamma_u +
                 gamma_e.dot(v.T))
        denom_x = v.dot(v.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x

        u = projection_positive(x + gamma_u / lambda_u)
        gamma_u += lambda_u * (x - u)

        xv = x.dot(v)
        temp = array - xv
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, xv + e))
        if error[-1] < 1e-4:
            break
    return u


def nmf_robust_rank1_v(array, u, v_init, lambda_v=1, lambda_e=1, max_iter=5e2):
    v = v_init
    e = utils.sparse(array.shape)
    gamma_v = utils.sparse(v.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for _ in range(int(max_iter)):
        temp = array - e
        num_y = lambda_e * u.T.dot(temp) + lambda_v * v - gamma_v +\
                u.T.dot(gamma_e)
        denom_y = u.T.dot(u).toarray()[0, 0] + lambda_v
        y = num_y / denom_y

        v = projection_positive(y + gamma_v / lambda_v)
        gamma_v += lambda_v * (y - v)

        uy = u.dot(y)
        temp = array - uy
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, uy + e))
        if error[-1] < 1e-4:
            break

    return v


def projection_positive(x):
    (i, j, data) = sp.find(x)
    mask = data > 0
    i = i[mask]
    j = j[mask]
    data = data[mask]
    return utils.sparse((data, (i, j)), shape=x.shape)


def shrinkage(t, alpha):
    if sp.issparse(t):
        i, j, x = sp.find(t)
        mask = np.abs(x) > alpha
        i = i[mask]
        j = j[mask]
        x = x[mask]
        s = np.sign(x) * (np.abs(x) - alpha)
        f = utils.sparse((s, (i, j)), shape=t.shape)
    else:
        f = t.sign() * (t.abs() - alpha).maximum(0)
    return f


def binarize(x):
    i, j, v = sp.find(x)
    mask = v > (1e-4 * v.max())
    return utils.sparse((v[mask], (i[mask], j[mask])), shape=x.shape,
                        dtype=bool)


def bicluster(deflator, n=None, share_points=True):
    if n is None:
        n = deflator.array.shape[1]

    bic_list = []
    online_mdl = mdl.OnlineMDL()
    total_codelength = []

    for _ in range(n):
        if deflator.array.nnz == 0:
            break

        try:
            if deflator.n_samples > deflator.selection.size:
                raise ValueError('Fewer active rows than compression rate')

            u, v = nmf_robust_rank1(deflator.array_compressed)
            u = binarize(u)
            v = binarize(v)
            idx_u = sp.find(u)[0]

            array_cropped = deflator.array[idx_u, :]
            u_cropped = utils.sparse(np.ones((idx_u.size, 1)))
            _, idx_v, v_data = sp.find(v)
            v_init = utils.sparse((v_data, (np.zeros_like(idx_v),
                                            deflator.selection[idx_v])),
                                  shape=(1, deflator.array.shape[1]))
            v = nmf_robust_rank1_v(array_cropped, u_cropped, v_init)

        except(AttributeError, ValueError):
            u, v = nmf_robust_rank1(deflator.array)
            u = binarize(u)

        v = binarize(v)
        idx_v = sp.find(v)[1]
        bic_list.append((u, v))

        deflator.remove_columns(idx_v)
        if not share_points:
            idx_u = sp.find(u)[0]
            deflator.remove_rows(idx_u)

        if n is not None:
            cl = online_mdl.add_rank1_approximation(deflator.array, u, v)
            total_codelength.append(cl)

    print [bic[0].nnz * bic[1].nnz for bic in bic_list]

    if n is not None:
        total_codelength = np.array(total_codelength)
        cut_point = np.argmin(total_codelength)

    return bic_list[:cut_point+1]
