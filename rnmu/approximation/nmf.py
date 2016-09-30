from __future__ import absolute_import
import numpy as np
from . import utils


def nmf_robust_multiplicative(array, r, u_init=None, v_init=None, min_iter=20,
                              max_iter=1e2, update='both', tol=1e-4,
                              relative_tol=1e-4):
    if u_init is None and v_init is None:
        u, v = _nmf_initialize(array, r)
    else:
        if utils.issparse(u_init):
            u_init = u_init.toarray()
        u = np.copy(u_init)
        if utils.issparse(v_init):
            v_init = v_init.toarray()
        v = np.copy(v_init)

    if utils.issparse(array):
        array = array.toarray()

    delta2 = np.finfo(np.float).eps ** 2

    error = []
    for _ in range(int(max_iter)):
        weights = array - np.dot(u, v)
        weights = np.power(np.power(weights, 2) + delta2, -0.5)
        aw = array * weights
        if update == 'both' or update == 'left':
            u = _mul_left_update(aw, u, v, weights)
        if update == 'both' or update == 'right':
            v = _mul_right_update(aw, u, v, weights)

        error.append(utils.relative_error(array, np.dot(u, v), ord=1))
        if len(error) >= min_iter:
            if error[-1] < tol:
                break
            if abs(error[-2] - error[-1]) < relative_tol * error[-2]:
                break

    if update == 'left':
        return u
    if update == 'right':
        return v
    return u, v


def _mul_left_update(aw, u, v, weights):
    eps = np.finfo(np.float).eps
    uvw = np.dot(u, v) * weights
    u *= np.dot(aw, v.T) / np.maximum(np.dot(uvw, v.T), eps)
    return u


def _mul_right_update(aw, u, v, weights):
    eps = np.finfo(np.float).eps
    uvw = np.dot(u, v) * weights
    v *= np.dot(u.T, aw) / np.maximum(np.dot(u.T, uvw), eps)
    return v


def _nmf_initialize(array, r):
    x, s, y = utils.svds(array, r)
    m = x.max() + 1e-16
    x /= m
    s = s[:, np.newaxis] * m
    y = s * np.abs(y)
    return x, y


def nmf_robust_admm(array, lambda_e=1, u_init=None, v_init=None, max_iter=5e2,
                    tol=1e-3):
    """
    Restricted to the rank 1 case.
    :param array:
    :param lambda_e:
    :param u_init:
    :param v_init:
    :param max_iter:
    :param tol:
    :return:
    """
    if u_init is None and v_init is None:
        u, v = _nmf_initialize(array, 1)
    else:
        u = u_init.copy()
        v = v_init.copy()

    e = array - u.dot(v)
    e = projection_positive(e)
    if utils.issparse(array):
        gamma_e = utils.sparse(e.shape)
    else:
        gamma_e = np.zeros(e.shape)

    error = []
    for k in range(int(max_iter)):
        print k
        temp = array - e
        u = projection_positive((lambda_e * temp + gamma_e).dot(v.T))
        u /= u.max() + 1e-16
        v = u.T.dot(lambda_e * temp + gamma_e)
        v /= (u.T.dot(u))[0, 0]
        v = projection_positive(v)

        uv = u.dot(v)
        temp = array - uv
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        # e = projection_positive(e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, uv + e))
        if error[-1] < tol:
            break

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.log10(error))

    return u, v


def projection_positive(x):
    if utils.issparse(x):
        (i, j, data) = utils.find(x)
        mask = data > 0
        i = i[mask]
        j = j[mask]
        data = data[mask]
        return utils.sparse((data, (i, j)), shape=x.shape)
    else:
        return np.maximum(x, 0)


def shrinkage(t, alpha):
    if utils.issparse(t):
        i, j, x = utils.find(t)
        mask = np.abs(x) > alpha
        i = i[mask]
        j = j[mask]
        x = x[mask]
        s = np.sign(x) * (np.abs(x) - alpha)
        f = utils.sparse((s, (i, j)), shape=t.shape)
    else:
        f = np.maximum(np.sign(t) * (np.abs(t) - alpha), 0)
    return f
