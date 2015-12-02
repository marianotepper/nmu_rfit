import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import comdet.biclustering.utils as utils
import comdet.biclustering.mdl as mdl
# import matplotlib.pyplot as plt


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
        num_x = (lambda_e * temp.dot(y.T) + lambda_u * u - gamma_u +\
                 gamma_e.dot(y.T))
        denom_x = y.dot(y.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x

        num_y = (lambda_e * x.T.dot(temp) + lambda_v * v - gamma_v +\
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
        num_x = (lambda_e * temp.dot(v.T) + lambda_u * u - gamma_u +\
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
        num_y = (lambda_e * u.T.dot(temp) + lambda_v * v - gamma_v +\
                 u.T.dot(gamma_e))
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
        (i, j, x) = sp.find(t)
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
    x.data[:] = 1
    x.astype(bool)
    return x


def bicluster(deflator, n=None, share_points=True):
    if n is None:
        n = deflator.array.shape[1]

    bic_list = []
    online_mdl = mdl.OnlineMDL()
    total_codelength = []

    for k in range(n):
        if deflator.array.nnz == 0:
            break

        try:
            active_rows = np.unique(sp.find(deflator.array_compressed)[0])
            if deflator.n_samples > active_rows.shape[0]:
                raise ValueError('Fewer active rows than compression rate')
            # print k, active_rows.shape, np.sort(deflator.selection[active_rows]), deflator.n_samples
            u, v = nmf_robust_rank1(deflator.array_compressed)

            idx_v = sp.find(v)[1]
            array_cropped = deflator.array[:, idx_v]
            v_cropped = utils.sparse(np.ones((1, idx_v.size)))

            idx_u, _, u_data = sp.find(u)
            u_init = utils.sparse((u_data, (deflator.selection[idx_u],
                                            np.zeros_like(idx_u))),
                                  shape=(deflator.array.shape[0], 1))
            u = nmf_robust_rank1_u(array_cropped, u_init, v_cropped)
            # print sp.find(u)[0], '\n---', idx_v, idx_v.shape
        except(AttributeError, ValueError):
            # print k, np.unique(sp.find(deflator.array_compressed)[0]).shape, deflator.n_samples
            u, v = nmf_robust_rank1(deflator.array)
            idx_v = sp.find(v)[1]

        u = binarize(u)
        v = binarize(v)
        bic_list.append((u, v))

        deflator.remove_columns(idx_v)
        if not share_points:
            idx_u = sp.find(u)[0]
            deflator.remove_rows(idx_u)

        if n is not None:
            cl = online_mdl.add_rank1_approximation(deflator.array, u, v)
            total_codelength.append(cl)

    if n is not None:
        total_codelength = np.array(total_codelength)
        cut_point = np.argmin(total_codelength)
        # plt.figure()
        # plt.plot(total_codelength)
        # plt.plot([cut_point], total_codelength[cut_point], marker='o', color='r')

    return bic_list[:cut_point+1]
