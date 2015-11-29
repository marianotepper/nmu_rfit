import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import sklearn.decomposition.nmf as nmf
import comdet.biclustering.nls as nls
import comdet.biclustering.utils as utils
import comdet.biclustering.mdl as mdl
import matplotlib.pyplot as plt


def nmf_robust(array, n_components=1, lambda_e=1., max_iter=500):
    e = utils.sparse(array.shape)
    gamma_e = utils.sparse(e.shape)

    learner = nmf.NMF(n_components=n_components, init='nndsvd', max_iter=500)
    u = learner.fit_transform(array - e)
    v = learner.components_

    learner = nmf.NMF(n_components=n_components, init='custom', max_iter=500)

    error = []
    for i in range(int(max_iter)):
        u_sp = utils.sparse(u)
        v_sp = utils.sparse(v)
        temp = array - u_sp.dot(v_sp)
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        u = learner.fit_transform(array - e, W=u, H=v)
        v = learner.components_

        error.append(learner.reconstruction_err_)
        if i > 0 and np.fabs(error[i] - error[i-1]) < 1e-6:
            break
    return utils.sparse(u), utils.sparse(v)


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
    for i in range(int(max_iter)):
        temp = array - e
        num_x = lambda_e * temp.dot(y.T) + lambda_u * u - gamma_u +\
                gamma_e.dot(y.T)
        denom_x = y.dot(y.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x

        num_y = lambda_e * x.T.dot(temp) + lambda_v * v - gamma_v +\
                x.T.dot(gamma_e)
        denom_y = x.T.dot(x).toarray()[0, 0] + lambda_v
        y = num_y / denom_y

        u = projection_positive(x + gamma_u / lambda_u)
        v = projection_positive(y + gamma_v / lambda_v)
        if u.nnz == 0 or v.nnz == 0:
            x = -x
            y = -y
            u = projection_positive(x / lambda_u)
            v = projection_positive(y / lambda_v)
            gamma_u = utils.sparse(u.shape)
            gamma_v = utils.sparse(v.shape)
            gamma_e = utils.sparse(e.shape)

        gamma_u += lambda_u * (x - u)
        gamma_v += lambda_v * (y - v)

        temp = array - x.dot(y)
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, temp))
        if i > 10 and all([np.fabs(error[i] - error[i-k]) < 1e-4
                           for k in range(1, 11)]):
            break
        if i > max_iter / 2 and np.fabs(error[i] - error[i-1]) < 1e-4:
            break
    return u, v


def nls(array, u_init, v):
    return nmf.non_negative_factorization(array, W=u_init, H=v, init='custom',
                                          update_H=False)[0]


def nmf_robust_u(array, u_init, v, lambda_e=1, max_iter=500):
    u = u_init.toarray()
    v_np = v.toarray()

    e = utils.sparse(array.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        u = nls(array - e, u, v_np)
        u_sp = utils.sparse(u)
        temp = array - u_sp.dot(v)
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(learner.reconstruction_err_)
        if i > 0 and np.fabs(error[i] - error[i-1]) < 1e-6:
            break
    return utils.sparse(u), utils.sparse(v)


def nmf_robust_rank1_u(array, u_init, v, lambda_u=1, lambda_e=1, max_iter=5e2):

    u = u_init
    e = utils.sparse(array.shape)
    gamma_u = utils.sparse(u.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        temp = array - e
        num_x = lambda_e * temp.dot(v.T) + lambda_u * u - gamma_u +\
                gamma_e.dot(v.T)
        denom_x = v.dot(v.T).toarray()[0, 0] + lambda_u
        x = num_x / denom_x

        u = projection_positive(x + gamma_u / lambda_u)
        gamma_u += lambda_u * (x - u)

        temp = array - u.dot(v)
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, temp))
        if i > 10 and all([np.fabs(error[i] - error[i-k]) < 1e-4
                           for k in range(1, 11)]):
            break
        if i > max_iter / 2 and np.fabs(error[i] - error[i-1]) < 1e-4:
            break

    return u


def nmf_robust_rank1_v(array, u, v_init, lambda_v=1, lambda_e=1, max_iter=5e2):

    v = v_init
    e = utils.sparse(array.shape)
    gamma_v = utils.sparse(v.shape)
    gamma_e = utils.sparse(e.shape)

    error = []
    for i in range(int(max_iter)):
        temp = array - e
        num_y = lambda_e * u.T.dot(temp) + lambda_v * v - gamma_v +\
                u.T.dot(gamma_e)
        denom_y = u.T.dot(u).toarray()[0, 0] + lambda_v
        y = num_y / denom_y

        v = projection_positive(y + gamma_v / lambda_v)
        gamma_v += lambda_v * (y - v)

        temp = array - u.dot(y)
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, temp))
        if i > 10 and all([np.fabs(error[i] - error[i-k]) < 1e-4
                           for k in range(1, 11)]):
            break
        if i > max_iter / 2 and np.fabs(error[i] - error[i-1]) < 1e-4:
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


def bicluster(online_deflator, n=None, share_points=True):
    if n is None:
        n = online_deflator.array.shape[1]

    rows = []
    cols = []
    online_mdl = mdl.OnlineMDL()
    total_codelength = []

    for k in range(n):
        print k
        if online_deflator.array.nnz == 0:
            break

        try:
            u, v = nmf_robust(online_deflator.array_compressed)

            idx_v = sp.find(v)[1]
            array_cropped = online_deflator.array[:, idx_v]
            v_cropped = utils.sparse(np.ones((1, idx_v.size)))

            idx_u, _, u_data = sp.find(u)
            u_init = utils.sparse((u_data, (online_deflator.selection[idx_u],
                                            np.zeros_like(idx_u))),
                                  shape=(online_deflator.array.shape[0], 1))
            # u = nmf_robust_rank1_u(array_cropped, u_init, v_cropped,
            #                        max_iter=10)
            u = nmf_robust_u(array_cropped, u_init, v_cropped)

        except AttributeError:
            u, v = nmf_robust(online_deflator.array, n_components=1)
            # u1, v1 = nmf_robust_rank1(online_deflator.array)
            idx_v = sp.find(v)[1]

        u = binarize(u)
        v = binarize(v)

        rows.append(u)
        cols.append(v)

        online_deflator.remove_columns(idx_v)
        if not share_points:
            idx_u = sp.find(u)[0]
            online_deflator.remove_rows(idx_u)

        if n is not None:
            cl = online_mdl.add_rank1_approximation(online_deflator.array, u, v)
            total_codelength.append(cl)

    if n is not None:
        total_codelength = np.array(total_codelength)
        cut_point = np.argmin(total_codelength)
        plt.figure()
        plt.plot(total_codelength)
        plt.plot([cut_point], total_codelength[cut_point], marker='o', color='r')

    return rows[:cut_point+1], cols[:cut_point+1]
