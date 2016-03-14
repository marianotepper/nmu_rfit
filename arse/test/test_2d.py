from __future__ import absolute_import, print_function
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import scipy.sparse as sp
import numpy as np
import scipy.io
import re
import timeit
import arse.biclustering as bc
import arse.test.utils as test_utils
import arse.pme.membership as membership
import arse.pme.preference as pref
import arse.pme.sampling as sampling
import arse.pme.line as line
import arse.pme.circle as circle
import arse.pme.acontrario as ac


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


def plot_models(models, palette=None, **kwargs):
    if palette is not None and 'color' in kwargs:
        raise RuntimeError('Cannot specify palette and color simultaneously.')
    for i, mod in enumerate(models):
        if palette is not None:
            kwargs['color'] = palette[i]
        mod.plot(**kwargs)


def plot_final_models(x, models, palette):
    base_plot(x)
    plot_models(models, palette=palette, linewidth=5, alpha=0.5)


def plot_original_models(x, original_models, right_factors, palette):
    base_plot(x)
    for i, rf in enumerate(right_factors):
        plot_models([original_models[j] for j in sp.find(rf)[1]],
                    color=palette[i], alpha=0.5)


def ground_truth(data, n_groups, group_size=50, model_class=None,
                 thresholder=None):
    gt_groups = []
    for i in range(n_groups):
        g = np.zeros((len(data),), dtype=bool)
        g[i * group_size:(i+1) * group_size] = True
        if model_class is None and thresholder is None:
            gt_groups.append(g)
        else:
            model = model_class(data=data[g])
            inliers = thresholder.membership(model, data) > 0
            gt_groups.append(inliers)

    return gt_groups


def run_biclustering(model_class, x, original_models, pref_matrix, comp_level,
                     thresholder, ac_tester, gt_groups, output_prefix,
                     palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, x, thresholder, ac_tester,
                                        bic_list)

    palette = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    plot_final_models(x, models, palette=palette)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    plot_original_models(x, original_models, [bic[1] for bic in bic_list],
                         palette)
    plt.savefig(output_prefix + '_bundles.pdf', dpi=600)

    bc_groups = [bic[0] for bic in bic_list]
    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)

    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def test(model_class, x, name, ransac_gen, thresholder, ac_tester, gt_groups,
         dir_name=None):
    print(name, x.shape)

    output_dir = '../results/'
    if dir_name is not None:
        output_dir += '{0}/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_prefix = output_dir + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(ransac_gen,
                                                            thresholder,
                                                            ac_tester)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix,
                                              'orig_models': orig_models})
    with open(output_prefix + '.pickle', 'wb') as handle:
        pickle.dump(pref_matrix, handle)
        pickle.dump(orig_models, handle)
    with open(output_prefix + '.pickle', 'rb') as handle:
        pref_matrix = pickle.load(handle)
        orig_models = pickle.load(handle)

    print('Preference matrix size:', pref_matrix.shape)

    base_plot(x)
    plot_models(orig_models, alpha=0.2)
    plt.savefig(output_prefix + '_original_models.pdf', dpi=600)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    compression_level = None
    stats_reg = run_biclustering(model_class, x, orig_models, pref_matrix,
                                 compression_level, thresholder, ac_tester,
                                 gt_groups, output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 32
    stats_comp = run_biclustering(model_class, x, orig_models, pref_matrix,
                                  compression_level, thresholder, ac_tester,
                                  gt_groups, output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def run(restimate_gt=False):
    log_file = 'test_2d_{0}.txt'
    if restimate_gt:
        log_file = log_file.format('restimate_gt')
    else:
        log_file = log_file.format('given_gt')
    # RANSAC parameter
    inliers_threshold = 0.015

    logger = test_utils.Logger(log_file)
    sys.stdout = logger

    # RANSAC parameter
    sampling_factor = 10
    # a contrario test parameters
    epsilon = 0.
    local_ratio = 3.

    config = {'Star': line.Line,
              'Stairs': line.Line,
              'Circles': circle.Circle,
              }

    stats_list = []
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')
    for example in mat.keys():
        for c in config:
            if example.find(c) == 0:
                ex_type = c
                break
        else:
            continue

        model_class = config[ex_type]
        data = mat[example].T

        min_sample_size = model_class().min_sample_size
        n_samples = data.shape[0] * sampling_factor * min_sample_size

        sampler = sampling.UniformSampler(n_samples)
        generator = sampling.ModelGenerator(model_class, data, sampler)

        proba = 1. / local_ratio
        ac_tester = ac.BinomialNFA(epsilon, proba, min_sample_size)
        thresholder = membership.LocalThresholder(inliers_threshold,
                                                  ratio=local_ratio)

        match = re.match(ex_type + '[0-9]*_', example)
        try:
            match = re.match('[0-9]+', match.group()[len(ex_type):])
            n_groups = int(match.group())
        except AttributeError:
            n_groups = 4
        if restimate_gt:
            gt_groups = ground_truth(data, n_groups, model_class=model_class,
                                     thresholder=thresholder)
        else:
            gt_groups = ground_truth(data, n_groups)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        output_prefix = example
        if restimate_gt:
            dir_name = 'test_2d_restimate_gt'
        else:
            dir_name = 'test_2d_given_gt'

        res = test(model_class, data, output_prefix, generator, thresholder,
                   ac_tester, gt_groups, dir_name=dir_name)
        stats_list.append(res)

        print('-'*40)
        plt.close('all')

    reg_list, comp_list = zip(*stats_list)

    print('Statistics of regular bi-clustering')
    test_utils.compute_stats(reg_list)
    print('Statistics of compressed bi-clustering')
    test_utils.compute_stats(comp_list)
    print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    # run(restimate_gt=False)
    run(restimate_gt=True)


if __name__ == '__main__':
    run_all()
    plt.show()
