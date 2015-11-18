import numpy as np
import math
from scipy.sparse.linalg import svds
from scipy.sparse import issparse, find
from comdet.biclustering.utils import relative_error, sparse
import comdet.biclustering.mdl as mdl
import matplotlib.pyplot as plt


def nmf_robust_rank1(array, lambda_u=1, lambda_v=1, lambda_e=1, u_init=None,
                     v_init=None, max_iter=1e2):

    if u_init is None and v_init is None:
        x, s, y = svds(array, 1)
        s = math.sqrt(s)
        x = s * np.abs(x)
        y = s * np.abs(y)
    else:
        x = u_init.copy()
        y = v_init.copy()

    x = sparse(x)
    y = sparse(y)
    x = sparsify(x)
    y = sparsify(y)
    u = x
    v = y
    e = sparse(array.shape)
    gamma_u = sparse(u.shape)
    gamma_v = sparse(v.shape)
    gamma_e = sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        temp = array - e
        num_x = lambda_e * temp.dot(y.T) + lambda_u * u - gamma_u +\
                gamma_e.dot(y.T)
        denom_x = y.dot(y.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x
        x = sparsify(x)

        num_y = lambda_e * x.T.dot(temp) + lambda_v * v - gamma_v +\
                x.T.dot(gamma_e)
        denom_y = x.T.dot(x).toarray()[0, 0] + lambda_v
        y = num_y / denom_y
        y = sparsify(y)

        u = projection_positive(x + gamma_u / lambda_u)
        v = projection_positive(y + gamma_v / lambda_v)
        if u.nnz == 0 or v.nnz == 0:
            x = -x
            y = -y
            u = projection_positive(x / lambda_u)
            v = projection_positive(y / lambda_v)
            gamma_u = sparse(u.shape)
            gamma_v = sparse(v.shape)
            gamma_e = sparse(e.shape)
        # u = sparsify(u)
        # v = sparsify(v)

        gamma_u = gamma_u + lambda_u * (x - u)
        gamma_v = gamma_v + lambda_v * (y - v)

        temp = array - x.dot(y)
        e = shrinkage(temp + gamma_e / lambda_e, 1 / lambda_e)
        gamma_e = gamma_e + lambda_e * (temp - e)

        error.append(relative_error(array, temp))
        if i > max_iter / 2 and math.fabs(error[i] - error[i-1]) < 1e-6:
            break

    return u, v


def nmf_robust_rank1_u(array, u_init, v, lambda_u=1, lambda_e=1, max_iter=1e2):

    x = u_init.copy()
    x = sparse(x)
    x = sparsify(x)
    u = u_init
    e = sparse(array.shape)
    gamma_u = sparse(u.shape)
    gamma_e = sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        temp = array - e
        num_x = lambda_e * temp.dot(v.T) + lambda_u * u - gamma_u +\
                gamma_e.dot(v.T)
        denom_x = v.dot(v.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x
        x = sparsify(x)

        u = projection_positive(x + gamma_u / lambda_u)
        gamma_u = gamma_u + lambda_u * (x - u)

        temp = array - u.dot(v)
        e = shrinkage(temp + gamma_e / lambda_e, 1 / lambda_e)
        gamma_e = gamma_e + lambda_e * (temp - e)

        error.append(relative_error(array, temp))
        if i > max_iter / 2 and math.fabs(error[i] - error[i-1]) < 1e-6:
            break

    return u


def nmf_robust_rank1_v(array, u, v_init, lambda_v=1, lambda_e=1, max_iter=1e2):

    v = v_init
    e = sparse(array.shape)
    gamma_v = sparse(v.shape)
    gamma_e = sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        temp = array - e
        num_y = lambda_e * u.T.dot(temp) + lambda_v * v - gamma_v +\
                u.T.dot(gamma_e)
        denom_y = u.T.dot(u).toarray()[0, 0] + lambda_v
        y = num_y / denom_y
        y = sparsify(y)

        v = projection_positive(y + gamma_v / lambda_v)
        gamma_v = gamma_v + lambda_v * (y - v)

        temp = array - u.dot(y)
        e = shrinkage(temp + gamma_e / lambda_e, 1 / lambda_e)
        gamma_e = gamma_e + lambda_e * (temp - e)

        error.append(relative_error(array, temp))
        if i > max_iter / 2 and math.fabs(error[i] - error[i-1]) < 1e-6:
            break

    return v


def projection_positive(x):
    (i, j, data) = find(x)
    mask = data > 0
    i = i[mask]
    j = j[mask]
    data = data[mask]
    return sparse((data, (i, j)), shape=x.shape)


def shrinkage(t, alpha):
    if issparse(t):
        (i, j, x) = find(t)
        mask = np.abs(x) > alpha
        i = i[mask]
        j = j[mask]
        x = x[mask]
        s = np.sign(x) * (np.abs(x) - alpha)
        f = sparse((s, (i, j)), shape=t.shape)
    else:
        f = t.sign() * (t.abs() - alpha).maximum(0)
    return f


def sparsify(x, threshold=1e-1):
    # return x
    (i, j, data) = find(x)
    data_abs = np.abs(data)
    mask = data_abs > threshold * np.max(data_abs)
    i = i[mask]
    j = j[mask]
    data = data[mask]
    return sparse((data, (i, j)), shape=x.shape)


def binarize(x):
    x.data[:] = 1
    return x


def bicluster(online_deflator, n=None, share_points=True):

    if n is None:
        n = online_deflator.array.shape[1]

    rows = []
    cols = []
    online_mdl = mdl.OnlineMDL()
    total_codelength = []

    for k in range(n):
        if online_deflator.array.nnz == 0:
            break

        try:
            selection = online_deflator.selection
            u, v = nmf_robust_rank1(online_deflator.array_compressed)

            idx_v = find(v)[1]
            array_cropped = online_deflator.array[:, idx_v]
            v_cropped = sparse(np.ones((1, idx_v.size)))

            idx_u, _, u_data = find(u)
            u_init = sparse((u_data, (online_deflator.selection[idx_u],
                                      np.zeros_like(idx_u))),
                            shape=(online_deflator.array.shape[0], 1))

            u = nmf_robust_rank1_u(array_cropped, u_init, v_cropped, max_iter=10)
        except AttributeError:
            u, v = nmf_robust_rank1(online_deflator.array)
            idx_v = find(v)[1]

        rows.append(binarize(u))
        cols.append(binarize(v))

        print k, idx_v.shape, idx_v, selection, find(u)[0].shape
        online_deflator.remove_columns(idx_v)
        if not share_points:
            idx_u = find(u)[0]
            online_deflator.remove_rows(idx_u)

        if n is not None:
            cl = online_mdl.add_rank1_approximation(online_deflator.array, u, v)
            total_codelength.append(cl)

        plt.matshow(online_deflator.array.toarray())

    if n is not None:
        total_codelength = np.array(total_codelength)
        cut_point = np.argmin(total_codelength)

        plt.figure()
        plt.plot(total_codelength)
        plt.plot([cut_point], total_codelength[cut_point], marker='o', color='r')

    return rows[:cut_point+1], cols[:cut_point+1]
