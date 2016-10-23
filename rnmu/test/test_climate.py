from __future__ import absolute_import, print_function
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import seaborn.apionly as sns
import sys
import timeit
import rnmu.nmu as nmu
import rnmu.test.utils as test_utils


def run(dir_name, filename, ncols=1):
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
    factors = nmu.recursive_nmu(mat, r=ncols)
    t = timeit.default_timer() - t
    print('time {:.2f}'.format(t))

    approx = 0
    for i, (lf, rf) in enumerate(factors):
        approx += lf.dot(rf)
        err = np.linalg.norm(approx - mat) ** 2 / np.linalg.norm(mat) ** 2
        print('ncols {}, error {:.4f}, energy {:.4f}'.format(i+1, err, 1 - err))

    test_name = dir_name + filename

    shape = (73, 144)
    lats = np.linspace(90, -90, shape[0])
    lons = np.linspace(0, 360, shape[1])

    for k in range(0, ncols):
        img = np.squeeze(factors[k][0]).reshape(shape)
        img = np.roll(np.flipud(img), shape[1] / 2, axis=1)

        plt.figure()
        ax = plt.axes(projection=ccrs.Robinson())
        ax.imshow(img, vmin=0, vmax=1, cmap='RdYlBu_r',
                  transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        plt.savefig(test_name + 'comp{}_left.png'.format(k+1),
                    dpi=150, bbox_inches='tight', pad_inches=0)

        with sns.axes_style("darkgrid"):
            plt.figure()
            sns.regplot(np.arange(mat.shape[1]), np.squeeze(factors[k][1]),
                        scatter_kws={'s': 10}, line_kws={'color':'r'},
                        order=1, ci=None, truncate=True)
            if x_labels_pos is not None:
                plt.xticks(x_labels_pos, x_labels_names, rotation=45)
            # plt.title('Component {}'.format(k + 1))
            plt.savefig(test_name + 'comp{}_right.pdf'.format(k + 1),
                        dpi=150, bbox_inches='tight', pad_inches=0)

        plt.close('all')


if __name__ == '__main__':
    dir_name = '../results/climate/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    logger = test_utils.Logger(dir_name + 'test.txt')
    sys.stdout = logger

    run(dir_name, 'air_mon', ncols=5)
    run(dir_name, 'air_day', ncols=5)

    sys.stdout = logger.stdout
    logger.close()

    plt.show()
