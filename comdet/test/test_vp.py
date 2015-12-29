from __future__ import absolute_import, print_function
import sys
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import PIL
import numpy as np
import timeit
import os
import collections
import comdet.biclustering as bc
import comdet.test.utils as test_utils
import comdet.pme.preference as pref
import comdet.pme.sampling as sampling
import comdet.pme.lsd as lsd
import comdet.pme.vanishing as vp
import comdet.pme.line as line
import comdet.pme.acontrario as ac


def base_plot(segments=[]):
    plt.figure()
    plt.axis('off')
    plt.imshow(gray_image, cmap='gray', alpha=.5)
    for seg in segments:
        seg.plot(c='k', linewidth=1)


def plot_models(models, palette=None, **kwargs):
    if palette is not None and 'color' in kwargs:
        raise RuntimeError('Cannot specify palette and color simultaneously.')
    for i, mod in enumerate(models):
        if palette is not None:
            kwargs['color'] = palette[i]
        mod.plot(**kwargs)


def plot_final_models(x, mod_inliers, palette):
    base_plot()
    sz_ratio = 1.5
    plt.xlim((1 - sz_ratio) * gray_image.size[0], sz_ratio * gray_image.size[0])
    plt.ylim(sz_ratio * gray_image.size[1],(1 - sz_ratio) * gray_image.size[1])

    all_inliers = []
    for ((mod, inliers), color) in zip(mod_inliers, palette):
        all_inliers.append(inliers)
        segments = x[inliers]
        if mod.point[2] != 0:
            mod.plot(color=color, linewidth=1)
        for seg in segments:
            if mod.point[2] != 0:
                midpoint = (seg.p_a + seg.p_b) / 2
                new_seg = lsd.Segment(midpoint, mod.point)
                new_seg.plot(c=color, linewidth=.2, alpha=.3)
            else:
                seg_line = line.Line(np.vstack((seg.p_a[:2], seg.p_b[:2])))
                seg_line.plot(c=color, linewidth=.2, alpha=.3)
            seg.plot(c=color, linewidth=1)
    remaining_segs = np.logical_not(reduce(np.logical_or, all_inliers))
    for seg in x[remaining_segs]:
        seg.plot(c='k', linewidth=1)


def ground_truth(n_elements, n_groups=5, group_size=50):
    gt_groups = []
    for i in range(n_groups):
        v = np.zeros((n_elements,), dtype=bool)
        v[i * group_size:(i+1) * group_size] = True
        gt_groups.append(v)
    return gt_groups


TestStats = collections.namedtuple("VPStats", ['time'])


def run_biclustering(x, original_models, pref_matrix, deflator, ac_tester,
                     gt_groups, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(deflator, n=5)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(vp.VanishingPoint, x, ac_tester,
                                        bic_list)

    palette = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    mod_inliers_list = [(mod, ac_tester.inliers(mod)) for mod in models]
    plot_final_models(x, mod_inliers_list, palette=palette)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    # test_utils.compute_measures(gt_groups, [bic[0] for bic in bic_list])
    return TestStats(time=t1)


def test(x, name, ransac_gen, ac_tester, gt_groups):
    print(name, len(x))

    output_prefix = '../results/vp/'
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    output_prefix += name
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    output_prefix += '/' + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(len(x), ransac_gen,
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
    stats_reg = run_biclustering(x, orig_models, pref_matrix, deflator,
                                 ac_tester, gt_groups,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 128
    deflator = bc.deflation.L1CompressedDeflator(pref_matrix, compression_level)
    stats_comp = run_biclustering(x, orig_models, pref_matrix, deflator,
                                  ac_tester, gt_groups,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def print_stats(stats):
    time_str = 'Time. mean: {0}, std: {1}, median: {2}'
    times = [s.time for s in stats]
    print(time_str.format(np.mean(times), np.std(times), np.median(times)))


if __name__ == '__main__':
    sys.stdout = test_utils.Logger("test_vp.txt")

    sampling_factor = 5
    inliers_threshold = 2 * np.pi * 0.01
    epsilon = 0

    dir_name = '/Users/mariano/Documents/datasets/YorkUrbanDB/'

    stats_list = []
    for i, example in enumerate(os.listdir(dir_name)):
        # if example != 'P1020171':
        #     continue
        # if example != 'P1020829':
        #     continue
        # if example != 'P1020826':
        #     continue
        if not os.path.isdir(dir_name + example):
            continue
        img_name = dir_name + '{0}/{0}.jpg'.format(example)
        gray_image = PIL.Image.open(img_name).convert('L')
        segments = lsd.compute(gray_image)
        segments = np.array(segments)

        ac_tester = ac.LocalNFA(segments, epsilon, inliers_threshold)
        sampler = sampling.AdaptiveSampler(int(len(segments) * sampling_factor))
        ransac_gen = sampling.ModelGenerator(vp.VanishingPoint, segments,
                                             sampler)

        # gt_groups = ground_truth(data.shape[0], n_groups=n_groups,
        #                          group_size=50)
        gt_groups = None

        print('-'*40)
        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        res = test(segments, example, ransac_gen, ac_tester, gt_groups)
        stats_list.append(res)

        plt.close('all')

    reg_list, comp_list = zip(*stats_list)

    print('Statistics of regular bi-clustering')
    print_stats(reg_list)
    print('Statistics of compressed bi-clustering')
    print_stats(comp_list)

    plt.show()
