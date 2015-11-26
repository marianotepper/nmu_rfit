import timeit

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import matplotlib.pyplot as plt

import comdet.biclustering.svd as svd


def test_compare():
    m = 20
    n = 10
    x = np.random.rand(m, n)
    u, s, vt = svd(x, 5, random_state=0)
    print u.shape, s.shape, vt.shape
    print s
    a = -u[:, 0] * s[0]
    b = vt[0, :]

    u_udt, s_udt, vt_udt = svd.update_svd(u, s, vt, a, b)
    print u_udt.shape, s_udt.shape, vt_udt.shape
    print s_udt

    print 'hola', a.shape, b.shape
    x2 = x + np.outer(a, b)
    u2, s2, vt2 = svd(x2, 5, random_state=0)
    print u2.shape, s2.shape, vt2.shape
    print s2

    print vt[1, :]
    print vt2[0, :]


def binary_matrix_generator(m, r, step=50, seed=0):
    np.random.seed(0)
    x = (np.abs(np.random.rand(m, r)) > 0.9).astype(np.float)
    mat = np.hstack((np.atleast_2d(x[:, i]).T for i in range(x.shape[1])
                     for _ in range(step)))

    np.random.seed(seed)
    noise = np.random.uniform(size=mat.shape)
    mask = np.abs(noise) > 0.97
    for i in range(0, r*step, step):
        mask[:, i:i+1] = False
    # mat[mask] = 1 - mat[mask]
    plt.matshow(mat)
    return mat


def test():
    m = 1000
    r = 1000
    x = binary_matrix_generator(m, r)

    t = timeit.default_timer()
    u, s, vt = sp_linalg.svds(sp.coo_matrix(x), 1)
    print timeit.default_timer() - t
    plt.figure()
    plt.plot(vt[0, :], 'r')

    a = -x[:, 20]
    b = np.zeros((x.shape[1],))
    b[:50] = 1
    x2 = x + np.outer(a, b)

    t = timeit.default_timer()
    svd_online = svd.SVD(x2)
    svd_online.update(a, b)
    print timeit.default_timer() - t
    print svd_online.u.shape, svd_online.s.shape, svd_online.vt.shape
    print svd_online.s

    # a = -x[:, 20]
    # b = np.zeros((x.shape[1],))
    # b[:50] = 1
    # u_udt, s_udt, vt_udt = update_svd(u_udt, s_udt, vt_udt, a, b)

    u2, s2, vt2 = sp_linalg.svds(sp.coo_matrix(x2), 20)
    print u2.shape, s2.shape, vt2.shape
    print np.flipud(s)

    # print vt[1, :]
    # print vt2[0, :]

    plt.figure()
    plt.plot(vt[0, :], 'r')
    plt.plot(-svd_online.vt[0, :], 'g')
    plt.plot(vt2[0, :])


def test_sequential():
    m = 1000
    r = 1000
    x_orig = binary_matrix_generator(m, r)
    x_orig = sp.csc_matrix(x_orig)

    x = x_orig.copy()
    t = timeit.default_timer()
    svd_online = svd.SVD(x)
    for k, i in enumerate(range(0, x.shape[1], 50)):
        print i, timeit.default_timer() - t
        a = -x[:, i]
        b = np.zeros((x.shape[1],))
        b[i:i+50] = 1
        svd_online.update(np.squeeze(a.toarray()), b)
    print timeit.default_timer() - t


def create_block_matrix(m, r, step=50):
    block_size = m / r
    block = np.ones((block_size, step))
    mat = sp.block_diag([block for _ in range(r)]).toarray()
    plt.matshow(mat)
    return mat


if __name__ == '__main__':

    # test_compare()
    test()
    # test_sequential()
    # test_sequential2()
    plt.show()
