from __future__ import absolute_import, print_function
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.sparse.linalg as sp_linalg
import seaborn.apionly as sns
import sys
import timeit
import rnmu.nmu as nmu
import rnmu.test.utils as test_utils


def plot_approximation(dir_name, filename, ncols=1):
    f = scipy.io.loadmat('../data/' + filename + '.mat')
    mat = f['A']

    if filename.find('mon') != -1:
        years_step = 10
        x_labels_pos = np.arange(0, mat.shape[1], 12 * years_step)
        x_labels_names = ['{}/{}'.format(1 + i % 12, 1948 + i / 12)
                          for i in x_labels_pos]
    elif filename.find('day') != -1:
        years_step = 10
        years = np.cumsum(np.insert(f['years'], 0, 0))
        x_labels_pos = years[::years_step]
        x_labels_names = ['01/01/{}'.format(1948 + i / 12)
                          for i in range(len(x_labels_pos))]
    else:
        x_labels_pos = None

    t = timeit.default_timer()
    factors = nmu.recursive_nmu(mat, r=ncols, tol=1e-5)
    t = timeit.default_timer() - t
    print('time {:.2f}'.format(t))

    approx = 0
    for i, (lf, rf) in enumerate(factors):
        approx += lf.dot(rf)
        err = np.linalg.norm(approx - mat) ** 2 / np.linalg.norm(mat) ** 2
        print('ncols {}, error {:.4f}, energy {:.4f}'.format(i+1, err, 1 - err))

    test_name = dir_name + filename

    for k in range(0, ncols):
        shape = (73, 144)
        img = np.squeeze(factors[k][0]).reshape(shape)
        img = np.roll(np.flipud(img), shape[1] / 2, axis=1)

        plt.figure()
        ax = plt.axes(projection=ccrs.Robinson())
        ax.imshow(img, vmin=0, vmax=1, cmap='RdYlBu_r',
                  transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()

        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r')
        sm._A = []
        cb = plt.colorbar(sm, shrink=0.5)

        cb.set_ticks([0, 1])

        plt.savefig(test_name + '_comp{}_left.png'.format(k+1),
                    dpi=150, bbox_inches='tight')
        plt.close()

        with sns.axes_style("whitegrid"):
            plt.figure()
            sns.regplot(np.arange(mat.shape[1]), np.squeeze(factors[k][1]),
                        scatter_kws={'s': 10, 'facecolor': '#1f78b4'},
                        line_kws={'color': '#e41a1c'},
                        order=1, ci=None, truncate=True)
            if x_labels_pos is not None:
                plt.xticks(x_labels_pos, x_labels_names, rotation=45,
                           size='x-large')
            for item in plt.yticks()[1]:
                item.set_fontsize('x-large')
            # plt.title('Component {}'.format(k + 1))
            plt.savefig(test_name + '_comp{}_right.pdf'.format(k + 1),
                        dpi=150, bbox_inches='tight')
            plt.close()


def plot_comparison(dir_name, filename):
    f = scipy.io.loadmat('../data/' + filename + '.mat')
    mat = f['A']

    u_svd, s, v_svd = sp_linalg.svds(mat, k=1)
    u_lag, v_lag = nmu.nmu(mat, max_iter=5e2, tol=1e-5)
    u_adm, v_adm = nmu.nmu_admm(mat, max_iter=5e2, tol=1e-5)

    def plot_hist(diff, title):
        plt.hist(diff.flatten(), bins=100, normed=True,
                 histtype='stepfilled', color='#a6cee3', edgecolor='#1f78b4')
        bbox = plt.ylim()
        plt.plot([0, 0], [bbox[0], bbox[1]], color='#e41a1c', linewidth=2,
                 linestyle='--')
        plt.ylim(bbox)
        bbox = plt.xlim()
        locs = np.round([0, bbox[0], bbox[-1]]).astype(np.int)
        plt.xticks(locs, ['{}'.format(x) for x in locs])
        plt.yticks([])
        plt.title(title)

    diff_svd = mat - s[0] * u_svd.dot(v_svd)
    diff_lag = mat - u_lag.dot(v_lag)
    diff_adm = mat - u_adm.dot(v_adm)

    with sns.axes_style("white"):
        fig = plt.figure(figsize=(7, 3))

        plt.subplot(131)
        plot_hist(diff_svd, 'SVD/NMF')

        plt.subplot(132)
        plot_hist(diff_lag, 'NMU - LR')

        plt.subplot(133)
        plot_hist(diff_adm, 'NMU - ADMM')

        fig.set_tight_layout(True)
        fig.savefig(dir_name + filename + '_comp_hist.pdf',
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_errors(dir_name, filename):
    f = scipy.io.loadmat('../data/' + filename + '.mat')
    mat = f['A']

    u, v, err_u1, err_v1 = nmu.nmu_admm(mat, max_iter=5e2, tol=1e-8,
                                        ret_errors=True)
    mat -= u.dot(v)
    print(mat.min(), mat.max())
    mat = np.maximum(mat, 0)
    _, _, err_u2, err_v2 = nmu.nmu_admm(mat, max_iter=5e2, tol=1e-8,
                                        ret_errors=True)

    with sns.axes_style("whitegrid"):
        plt.figure()
        plt.semilogy(err_u1, linewidth=2, color='#e41a1c')
        plt.semilogy(err_v1, linewidth=2, color='#377eb8')
        plt.semilogy(err_u2, linewidth=2, color='#e41a1c', linestyle='--')
        plt.semilogy(err_v2, linewidth=2, color='#377eb8', linestyle='--')
        for item in plt.xticks()[1]:
            item.set_fontsize('x-large')
        for item in plt.yticks()[1]:
            item.set_fontsize('x-large')
        plt.legend(['Left factor 1', 'Right factor 1',
                    'Left factor 2', 'Right factor 2'],
                   prop={'size': 'x-large'})
        plt.tight_layout()
        plt.savefig(dir_name + filename + '_convergence.pdf',
                    dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    dir_name = '../results/climate/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    logger = test_utils.Logger(dir_name + 'test.txt')
    sys.stdout = logger

    plot_approximation(dir_name, 'air_mon', ncols=5)
    # plot_approximation(dir_name, 'air_day', ncols=5)

    plot_comparison(dir_name, 'air_mon')
    plot_errors(dir_name, 'air_mon')

    sys.stdout = logger.stdout
    logger.close()

    plt.show()
