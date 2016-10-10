import numpy as np


def recursive_nmu(array, r=1, max_iter=5e2, tol=1e-3, downdate='minus'):
    array = array.copy()
    factors = []
    for _ in range(r):
        u, v = nmu_admm(array, max_iter, tol)
        factors.append((u, v))
        if downdate == 'minus':
            array = np.maximum(0, array - np.dot(u, v))
        if downdate == 'hard_col' or downdate == 'hard':
            array[:, np.squeeze(v > 0)] = 0
        if downdate == 'hard_row' or downdate == 'hard':
            array[np.squeeze(u > 0), :] = 0
        if array.max() == 0:
            break

    return factors


def nmu(array, max_iter, tol):
    u, v = _nmu_initialize(array)
    u_old = u.copy()
    v_old = v.copy()
    # Initialization of Lagrangian variable lambda
    mu = 0

    # Alternating optimization
    error_u = []
    error_v = []
    for k in range(int(max_iter)):
        # updating mu:
        if np.any(u > 0) and np.any(v > 0):
            remainder = array - np.dot(u, v)
            mu = np.maximum(0, mu - remainder / (k + 1))
        else:
            mu /= 2
            u = u_old
            v = v_old

        for _ in range(2):
            u_old = u.copy()
            v_old = v.copy()
            # updating u, v:
            aux = array - mu
            u = np.maximum(0, aux.dot(v.T))
            u /= np.max(u) + 1e-16
            v = np.maximum(0, np.dot(u.T, aux) / np.dot(u.T, u))

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    return u, v


def nmu_admm(array, max_iter, tol):
    u, v = _nmu_initialize(array)

    # Initialization of Lagrangian variables lambda_r, gamma_r
    gamma_r = np.zeros(array.shape)
    remainder = np.maximum(0, array - np.dot(u, v))

    # Alternating optimization
    error_u = []
    error_v = []
    for k in range(int(max_iter)):
        u_old = u.copy()
        v_old = v.copy()
        # updating u, v:
        aux = array - remainder

        u = (aux + gamma_r).dot(v.T)
        u = np.maximum(0, u)
        u /= np.max(u) + 1e-16

        v = np.dot(u.T, aux + gamma_r) / np.dot(u.T, u)
        v = np.maximum(0, v)

        temp = array - np.dot(u, v)
        remainder = (temp + gamma_r)
        remainder = np.maximum(0, remainder)
        gamma_r += (temp - remainder)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    return u, v


def _nmu_initialize(array):
    idx = np.argmax(np.sum(array, axis=0))
    x = array[:, idx][:, np.newaxis]
    m = np.max(x) + 1e-16
    x /= m
    y = m * x.T.dot(array) / np.dot(x.T, x)
    return x, y
