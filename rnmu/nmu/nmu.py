import numpy as np
import scipy.sparse.linalg as sp_linalg


def recursive_nmu(array, r=None, max_iter=5e2, tol=1e-3, downdate='minus',
                  init='svd', refine_v=False):
    if r is None:
        r = min(array.shape)

    array = array.copy()
    factors = []
    for k in range(r):
        u, v = nmu_admm(array, max_iter, tol, init=init)
        if refine_v:
            u, v = nmu_admm(array, max_iter, tol, init=u)
        if np.count_nonzero(u) == 0 or np.count_nonzero(v) == 0:
            break
        factors.append((u, v))
        if k == r - 1:
            continue
        if downdate == 'minus':
            array = np.maximum(0, array - np.dot(u, v))
        if downdate == 'hard-col' or downdate == 'hard-both':
            array[:, np.squeeze(v > 0)] = 0
        if downdate == 'hard-row' or downdate == 'hard-both':
            array[np.squeeze(u > 0), :] = 0
        if array.max() == 0:
            break

    return factors


def nmu(array, max_iter=5e2, tol=1e-3, init='svd', ret_errors=False):
    u, v = _nmu_initialize(array, init=init)
    u_old = u.copy()
    v_old = v.copy()
    mu = 0

    # Alternating optimization
    error_u = []
    error_v = []
    for k in range(int(max_iter)):
        # updating mu:
        if np.any(u > 0) and np.any(v > 0):
            remainder = array - u.dot(v)
            mu = np.maximum(0, mu - remainder / (k + 1))
        else:
            mu /= 2
            u = u_old
            v = v_old

        u_old = u.copy()
        v_old = v.copy()
        # updating u, v:
        aux = array - mu
        u = np.maximum(0, aux.dot(v.T))
        u = np.maximum(0, u)
        umax = u.max()
        if umax == 0:
            v[:] = 0
        else:
            u /= umax
            v = u.T.dot(aux) / u.T.dot(u)
            v = np.maximum(0, v)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    if ret_errors:
        return u, v, error_u, error_v
    else:
        return u, v


def nmu_admm(array, max_iter=5e2, tol=1e-3, init='svd', ret_errors=False):
    u, v = _nmu_initialize(array, init=init)

    sigma = 1.
    gamma_r = np.zeros(array.shape)
    remainder = np.maximum(0, array - u.dot(v))

    # Alternating optimization
    error_u = []
    error_v = []
    for _ in range(int(max_iter)):
        u_old = u.copy()
        v_old = v.copy()
        # updating u, v:
        aux = array - remainder + gamma_r / sigma

        if isinstance(init, basestring):
            u = aux.dot(v.T)
            u = np.maximum(0, u)
            umax = u.max()
            if umax <= 1e-10:
                u[:] = 0
                v[:] = 0
                break
            u /= umax

        v = u.T.dot(aux) / u.T.dot(u)
        v = np.maximum(0, v)

        temp = array - u.dot(v)
        remainder = (temp + gamma_r / sigma)
        remainder = np.maximum(0, remainder)
        gamma_r += sigma * (temp - remainder)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    if ret_errors:
        return u, v, error_u, error_v
    else:
        return u, v


def _nmu_initialize(array, init):
    if isinstance(init, np.ndarray):
        x = init.copy()
        y = x.T.dot(array) / np.dot(x.T, x)
        m = np.max(x)
        if m > 0:
            x /= m
        y *= m
    elif init == 'max':
        idx = np.argmax(np.sum(array, axis=0))
        x = array[:, idx][:, np.newaxis]
        y = x.T.dot(array) / np.dot(x.T, x)
    elif init == 'svd':
        x, s, y = sp_linalg.svds(array, 1)
        y *= s[0]
        if np.all(x <= 1e-10) and np.all(y <= 1e-10):
            x *= -1
            y *= -1
    else:
        raise ValueError('Unknown initialization method')

    m = np.max(x)
    if m > 0:
        x /= m
    y *= m

    return x, y
