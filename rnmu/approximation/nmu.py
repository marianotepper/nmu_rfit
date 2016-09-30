import numpy as np
from sklearn.utils.extmath import randomized_svd
from . import nmf

# % recursive Lagrangian NMU algorithm
# %
# % cf. "Using Underapproximations for Sparse Nonnegative Matrix Factorization",
# % N. Gillis and F. Glineur, Pattern Recognition 43 (4), pp. 1676-1687, 2010.
# % and
# % "Dimensionality Reduction, Classification, and Spectral Mixture Analysis
# % using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
# % Optical Engineering 50, 027001, February 2011.
# %
# % website. http://sites.google.com/site/nicolasgillis/
# %
# %
# % [x,y] = recursiveNMU(M,r,Cnorm,maxiter)
# %
# % Input.
# %   M              : (m x n) matrix to factorize.
# %   r              : factorization rank, default = 1.
# %   Cnorm          : Choice of the norm 1 or 2, default = 2.
# %   maxiter        : number of iterations, default = 40.
# %
# % Output.
# %   (U,V) : solution, UV^T "<=" M, U >= 0, V >= 0   ("." - approximately)


def recursive_nmu(array, r=1, max_iter=None):
    array = array.copy()
    factors = []
    for k in range(r):
        print(k)
        if max_iter is None:
            u, v = nmu_admm(array)
            # u, v = nmf.nmf_robust_admm(array, tol=1e-6)
        else:
            u, v = nmu_admm(array, max_iter=max_iter)
        factors.append((u, v))

        import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(np.maximum(0, np.dot(u, v) - array), interpolation='none', cmap=plt.get_cmap('Blues'))
        # plt.figure()
        # plt.imshow(np.dot(u, v), interpolation='none', cmap=plt.get_cmap('Blues'))
        # plt.title('uv')
        #
        # print('nnz u', np.count_nonzero(u))
        #
        # plt.figure()
        # plt.subplot(121)
        # plt.stem(u)
        # plt.title('u')
        # plt.subplot(122)
        # plt.stem(v.T)
        # plt.title('v')

        plt.figure()
        plt.imshow(np.dot(u, u.T), interpolation='none')

        print np.min(np.dot(u, v)), np.max(np.dot(u, v))
        print np.min(array - np.dot(u, v)), np.max(array - np.dot(u, v))

        array = np.maximum(0, array - np.dot(u, v))

    return factors


def nmu(array, max_iter=5e2, tol=1e-3):
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
            print 'entra'
            mu /= 2
            u = u_old
            v = v_old

        print 'mM:', np.min(mu), np.max(mu), np.min(remainder), np.max(remainder)
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


def nmu_admm(array, max_iter=5e2, tol=1e-3):
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
