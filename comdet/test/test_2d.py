from __future__ import absolute_import, print_function
import sys
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import scipy.sparse as sp
import numpy as np
import scipy.io
import re
import timeit
import comdet.biclustering as bc
import comdet.test.utils as test_utils
import comdet.pme.preference as pref
import comdet.pme.sampling as sampling
import comdet.pme.line as line
import comdet.pme.circle as circle
import comdet.pme.acontrario as ac


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


def ground_truth(n_elements, n_groups=5, group_size=50):
    gt_groups = []
    for i in range(n_groups):
        v = np.zeros((n_elements,), dtype=bool)
        v[i * group_size:(i+1) * group_size] = True
        gt_groups.append(v)
    return gt_groups


def run_biclustering(model_class, x, original_models, pref_matrix, deflator,
                     ac_tester, gt_groups, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(deflator)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, x, ac_tester, bic_list)

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


def test(model_class, x, name, ransac_gen, ac_tester, gt_groups):
    print(name, x.shape)

    output_prefix = '../results/' + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(x.shape[0],
                                                            ransac_gen,
                                                            ac_tester)
    print('Preference matrix size:', pref_matrix.shape)

    base_plot(x)
    plot_models(orig_models, alpha=0.2)
    plt.savefig(output_prefix + '_original_models.pdf', dpi=600)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    deflator = bc.deflation.Deflator(pref_matrix)
    stats_reg = run_biclustering(model_class, x, orig_models, pref_matrix,
                                 deflator, ac_tester, gt_groups,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 128
    deflator = bc.deflation.L1CompressedDeflator(pref_matrix, compression_level)
    stats_comp = run_biclustering(model_class, x, orig_models, pref_matrix,
                                  deflator, ac_tester, gt_groups,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def run():
    sys.stdout = test_utils.Logger("test_2d.txt")

    sampling_factor = 5
    inliers_threshold = 0.03
    epsilon = 0

    configuration = {'Star': (line.Line, sampling.UniformSampler(),
                               ac.LocalNFA),
                     'Stairs': (line.Line, sampling.UniformSampler(),
                                ac.LocalNFA),
                     'Circles': (circle.Circle, sampling.UniformSampler(),
                                 ac.circle.LocalNFA),
                     }

    stats_list = []
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')
    for example in mat.keys():
        exp_type = None
        for c in configuration:
            if example.find(c) == 0:
                exp_type = c
                break
        else:
            if exp_type is None:
                continue

        model_class, sampler, ac_tester_class = configuration[exp_type]
        data = mat[example].T

        sampler.n_samples = data.shape[0] * sampling_factor *\
                            model_class().min_sample_size
        ransac_gen = sampling.ModelGenerator(model_class, data, sampler)
        ac_tester = ac_tester_class(data, epsilon, inliers_threshold)

        match = re.match(exp_type + '[0-9]*_', example)
        try:
            match = re.match('[0-9]+', match.group()[len(exp_type):])
            n_groups = int(match.group())
        except AttributeError:
            n_groups = 4
        gt_groups = ground_truth(data.shape[0], n_groups=n_groups,
                                 group_size=50)

        print('-'*40)
        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        res = test(model_class, data, example, ransac_gen, ac_tester, gt_groups)
        stats_list.append(res)

        plt.close('all')

    reg_list, comp_list = zip(*stats_list)

    print('Statistics of regular bi-clustering')
    test_utils.print_stats(reg_list)
    print('Statistics of compressed bi-clustering')
    test_utils.print_stats(comp_list)

    plt.show()

if __name__ == '__main__':
    run()