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
    s = np.atleast_2d(np.sqrt(s))
    x = np.abs(x) * s
    y = s.T * np.abs(y)
    return x, y


def nmf_robust_admm(array, update='both', lambda_u=1, lambda_v=1, lambda_e=1,
                    u_init=None, v_init=None, max_iter=5e2, tol=1e-3):
    """
    Restricted to the rank 1 case.
    :param array:
    :param update:
    :param lambda_u:
    :param lambda_v:
    :param lambda_e:
    :param u_init:
    :param v_init:
    :param max_iter:
    :param tol:
    :return:
    """
    if u_init is None and v_init is None:
        x, y = _nmf_initialize(array, 1)
    else:
        if update == 'both':
            x = u_init.copy()
            y = v_init.copy()
        if update == 'left':
            x = utils.solve(array.T, v_init)
            x = utils.sparse(x[:, np.newaxis])
            y = utils.sparse(v_init[np.newaxis, :])
        if update == 'right':
            y = utils.solve(array, u_init)
            x = utils.sparse(u_init[:, np.newaxis])
            y = utils.sparse(y[np.newaxis, :])

    if update == 'both' or update == 'left':
        if utils.issparse(array):
            x = utils.sparse(x)
            gamma_u = utils.sparse(x.shape)
        else:
            gamma_u = np.zeros(x.shape)
        u = x
    if update == 'both' or update == 'right':
        if utils.issparse(array):
            y = utils.sparse(y)
            gamma_v = utils.sparse(y.shape)
        else:
            gamma_v = np.zeros(y.shape)
        v = y

    e = array - x.dot(y)
    if utils.issparse(array):
        gamma_e = utils.sparse(e.shape)
    else:
        gamma_e = np.zeros(e.shape)

    error = []
    for kk in range(int(max_iter)):
        temp = array - e
        if update == 'both' or update == 'left':
            x, u, gamma_u = _admm_left_update(temp, u, y, lambda_u, gamma_u,
                                              lambda_e, gamma_e)
        if update == 'both' or update == 'right':
            y, v, gamma_v = _admm_right_update(temp, x, v, lambda_v, gamma_v,
                                               lambda_e, gamma_e)

        xy = x.dot(y)
        temp = array - xy
        e = shrinkage(temp + gamma_e / lambda_e, 1. / lambda_e)
        gamma_e += lambda_e * (temp - e)

        error.append(utils.relative_error(array, xy + e))
        if error[-1] < tol:
            break

    if update == 'left':
        return u
    if update == 'right':
        return v
    return u, v


def _admm_left_update(mat, u, y, lambda_u, gamma_u, lambda_e, gamma_e):
    x = lambda_e * mat.dot(y.T) + lambda_u * u - gamma_u + gamma_e.dot(y.T)
    x /= (y.dot(y.T))[0, 0] + lambda_u
    u = projection_positive(x + gamma_u / lambda_u)
    gamma_u += lambda_u * (x - u)
    return x, u, gamma_u


def _admm_right_update(mat, x, v, lambda_v, gamma_v, lambda_e, gamma_e):
    y = lambda_e * x.T.dot(mat) + lambda_v * v - gamma_v + x.T.dot(gamma_e)
    y /= (x.T.dot(x))[0, 0] + lambda_v
    v = projection_positive(y + gamma_v / lambda_v)
    gamma_v += lambda_v * (y - v)
    return y, v, gamma_v


def projection_positive(x):
    if utils.issparse(x):
        (i, j, data) = utils.find(x)
        mask = data > 0
        i = i[mask]
        j = j[mask]
        data = data[mask]
        return utils.sparse((data, (i, j)), shape=x.shape)
    else:
        x[x < 0] = 0
        return x


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
