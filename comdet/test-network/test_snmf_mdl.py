from __future__ import absolute_import, print_function
import timeit
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
from functools import partial
import comdet.biclustering.utils as bic_utils
import comdet.snmf.snpa as snpa


def binary_matrix_generator(seed):
    np.random.seed(0)
    m = 200
    r = 5
    step = 50
    x = (np.abs(np.random.rand(m, r)) > 0.8).astype(np.float)
    mat = np.hstack((np.atleast_2d(x[:, i]).T for i in range(x.shape[1])
                     for _ in range(step)))
    noiseless_mat = np.copy(mat)
    np.random.seed(seed)
    noise = np.random.uniform(size=mat.shape)
    mask = np.abs(noise) > 0.97
    for i in range(0, r*step, step):
        mask[:, i:i+1] = False
    mat[mask] = 1 - mat[mask]
    return mat, noiseless_mat, r


def test(seed, snmf, matrix_generator):

    cmap = sns.cubehelix_palette(256, start=2.5, rot=0, dark=0.15, light=1,
                                 reverse=False, as_cmap=True)

    mat, noiseless_mat, r = matrix_generator(seed)

    def test_selection(selection):
        t = timeit.default_timer()
        cols, weights, mdl_codelengths = snmf(mat, normalize=True,
                                              selection=selection)
        time = timeit.default_timer() - t
        print(selection, cols)

        noiseless_weights = snpa.compute_weights(noiseless_mat, cols, func)

        rec = np.dot(mat[:, cols], weights)
        noiseless_rec = np.dot(mat[:, cols], noiseless_weights)

        plt.figure()
        plt.plot(mdl_codelengths, 'bo-')

        res = {'error': bic_utils.relative_error(mat, rec),
               'error_noiseless': bic_utils.relative_error(noiseless_mat, noiseless_rec),
               'time': time}
        return res

    res_snpa = test_selection('standard')
    res_mt = test_selection('mt')
    return res_snpa, res_mt


def test_binary_multiple(trials, snmf, matrix_generator):

    res_snpa = []
    res_mt = []
    for s in range(trials):
        r_s = test(s, snmf, matrix_generator)
        res_snpa.append(r_s[0])
        res_mt.append(r_s[1])

    funcs = [np.mean, np.std, np.median, np.min, np.max]
    fmt = ['{' + str(i) + ':1.3e} ' for i in range(len(funcs))]
    fmt = reduce(str.__add__, fmt)

    def print_stat(key):
        stat_std = np.array([r[key] for r in res_snpa])
        stat_mt = np.array([r[key] for r in res_mt])
        print(key + ' ST: ' + fmt.format(*[f(stat_std) for f in funcs]))
        print(key + ' MT: ' + fmt.format(*[f(stat_mt) for f in funcs]))

    print_stat('error')
    print_stat('error_noiseless')
    print_stat('time')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac

    repetitions = 1

    def run(name, method, func, pool_size, matrix_generator):
        print(name)
        f = partial(snpa.snmf, method=method, func=func,
                    pool_size=pool_size)
        test_binary_multiple(repetitions, f, matrix_generator)

    # func = snpa.frobenius_loss
    # run('SPA Frobenius', 'spa', func, 0, gaussian_matrix_generator)
    # run('SNPA Frobenius', 'snpa', func, 0, gaussian_matrix_generator)
    # # # run('SPA Frobenius parallel', 'spa', func, 4, gaussian_matrix_generator)
    # run('SNPA Frobenius parallel', 'snpa', func, 4, gaussian_matrix_generator)

    func = snpa.robust_loss
    run('SPA robust', 'spa', func, 0, binary_matrix_generator)
    run('SNPA robust', 'snpa', func, 0, binary_matrix_generator)
    # run('SPA robust parallel', 'spa', func, 4, binary_matrix_generator)
    # run('SNPA robust parallel', 'snpa', func, 4, binary_matrix_generator)

    plt.show()