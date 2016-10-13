from __future__ import absolute_import, print_function
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
import scipy.io
import re
import timeit
import rnmu.test.utils as test_utils
import rnmu.pme.detection as detection
import rnmu.pme.multigs as multigs
import rnmu.pme.sampling as sampling
import rnmu.pme.line as line
import rnmu.pme.circle as circle


def base_plot(x):
    x_lim = (x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
    y_lim = (x[:, 1].min() - 0.1, x[:, 1].max() + 0.1)
    delta_x = x_lim[1] - x_lim[0]
    delta_y = y_lim[1] - y_lim[0]
    min_delta = min([delta_x, delta_y])
    delta_x /= min_delta
    delta_y /= min_delta
    fig_size = (4 * delta_x, 4 * delta_y)

    plt.figure(figsize=fig_size)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.scatter(x[:, 0], x[:, 1], c='w', marker='o', s=10)


def plot_models(x, models, palette):
    base_plot(x)
    for mod, color in zip(models, palette):
        mod.plot(color=color, linewidth=5, alpha=0.5)


def plot_original_models(x, original_models, bic_list, palette):
    base_plot(x)
    for i, (_, rf) in enumerate(bic_list):
        for j in np.nonzero(rf)[1]:
            original_models[j].plot(color=palette[i], alpha=0.5 * rf[:, j])


def plot_final_biclusters(x, bic_list, palette):
    base_plot(x)
    for (lf, rf), color in zip(bic_list, palette):
        sel = np.squeeze(lf > 0)
        color = np.array(sel.sum() * [color])
        color = np.append(color, lf[sel], axis=1)
        plt.scatter(x[sel, 0], x[sel, 1], c=color, marker='o', s=10,
                    edgecolors='none')


def ground_truth(data, n_groups, group_size=50):
    gt_groups = []
    for i in range(n_groups):
        g = np.zeros((len(data),), dtype=bool)
        g[i * group_size:(i+1) * group_size] = True
        gt_groups.append(g)

    return gt_groups


def test(ransac_gen, x, sigma, name=None, gt_groups=None, palette='Set1'):
    t = timeit.default_timer()
    pref_mat, orig_models, models, bics = detection.run(ransac_gen, x, sigma)
    t1 = timeit.default_timer() - t
    print('Total time:', t1)

    base_plot(x)
    if name is not None:
        plt.savefig(name + '_data.pdf', dpi=600)

    plt.figure()
    detection.plot(pref_mat)
    if name is not None:
        plt.savefig(name + '_pref_mat.pdf', dpi=600)

    palette = sns.color_palette(palette, len(bics))

    plt.figure()
    detection.plot(bics, palette=palette)
    if name is not None:
        plt.savefig(name + '_pref_mat_bic.pdf', dpi=600)

    plot_models(x, models, palette=palette)
    if name is not None:
        plt.savefig(name + '_final_models.pdf', dpi=600)

    plot_final_biclusters(x, bics, palette=palette)

    plot_original_models(x, orig_models, bics, palette)
    # if name is not None:
    #     plt.savefig(name + '_bundles.pdf', dpi=600)

    bc_groups = [bic[0] for bic in bics]
    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)

    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def run(types):
    # Sampling ratio with respect to the number of elements
    sampling_factor = 50
    sigma = 0.05

    config = {'Star': line.Line,
              'Stairs': line.Line,
              'Circles': circle.Circle,
              }

    dir_name = 'test_2d'
    if dir_name is not None:
        output_dir = '../results/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir += '{0}/'.format(dir_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    logger = test_utils.Logger(output_dir + 'test_2d.txt')
    sys.stdout = logger

    stats_list = []
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')
    for example in mat.keys():
        for c in types:
            if example.find(c) == 0:
                ex_type = c
                break
        else:
            continue

        model_class = config[ex_type]
        data = mat[example].T

        n_samples = data.shape[0] * sampling_factor
        # generator = multigs.ModelGenerator(model_class, n_samples, batch=10,
        #                                    h_ratio=.1)
        sampler = sampling.UniformSampler(n_samples)
        generator = sampling.ModelGenerator(model_class, sampler)

        match = re.match(ex_type + '[0-9]*_', example)
        try:
            match = re.match('[0-9]+', match.group()[len(ex_type):])
            n_groups = int(match.group())
        except AttributeError:
            n_groups = 4
        gt_groups = ground_truth(data, n_groups)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        output_prefix = output_dir + example
        res = test(generator, data, sigma, name=output_prefix,
                   gt_groups=gt_groups)
        stats_list.append(res)

        print('-'*40)
        # break
        plt.close('all')

    # reg_list, comp_list = zip(*stats_list)
    #
    # print('Statistics of regular bi-clustering')
    # test_utils.compute_stats(reg_list)
    # print('Statistics of compressed bi-clustering')
    # test_utils.compute_stats(comp_list)
    # print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    run(['Star', 'Circles', 'Stairs'])
    # run(['Stairs'])
    # run(['Circles'])


if __name__ == '__main__':
    run_all()
    plt.show()
