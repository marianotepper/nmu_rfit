from __future__ import absolute_import, print_function
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
import scipy.io
import timeit
import os
import collections
import comdet.biclustering as bc
import comdet.test.utils as test_utils
import comdet.pme.preference as pref
import comdet.pme.line as line
import comdet.pme.sampling as sampling
import comdet.pme.lsd as lsd
import comdet.pme.vanishing as vp
import comdet.pme.acontrario as ac


def base_plot(image, segments=[]):
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap='gray', alpha=.5)
    for seg in segments:
        seg.plot(c='k', linewidth=1)


def plot_models(models, palette=None, **kwargs):
    if palette is not None and 'color' in kwargs:
        raise RuntimeError('Cannot specify palette and color simultaneously.')
    for i, mod in enumerate(models):
        if palette is not None:
            kwargs['color'] = palette[i]
        mod.plot(**kwargs)


def plot_final_models(image, x, mod_inliers, palette):
    base_plot(image)
    sz_ratio = 1.5
    plt.xlim((1 - sz_ratio) * image.size[0], sz_ratio * image.size[0])
    plt.ylim(sz_ratio * image.size[1],(1 - sz_ratio) * image.size[1])

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


def ground_truth(association, gt_segments, lsd_segments, threshold):
    gt_groups = []
    for v in np.unique(association):
        p = vp.VanishingPoint(data=gt_segments[association == v])
        inliers = np.abs(p.distances(lsd_segments)) <= threshold
        gt_groups.append(inliers)
    return gt_groups


def run_biclustering(image, x, original_models, pref_matrix, deflator, ac_tester,
                     output_prefix, gt_groups=None, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(deflator)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(vp.VanishingPoint, x, ac_tester,
                                        bic_list)

    palette = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    mod_inliers_list = [(mod, ac_tester.inliers(mod)) for mod in models]
    plot_final_models(image, x, mod_inliers_list, palette)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    if gt_groups is not None:
        bc_groups = [bic[0] for bic in bic_list]
        gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)
        return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)
    else:
        return dict(time=t1)


def test(image, x, res_dir_name, name, ransac_gen, ac_tester, gt_groups=None):
    print(name, len(x))

    output_prefix = '../results/' + res_dir_name + '/'
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    output_prefix += name
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    output_prefix += '/' + name

    base_plot(image, x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(len(x), ransac_gen,
                                                            ac_tester)
    print('Preference matrix size:', pref_matrix.shape)

    base_plot(image, x)
    plot_models(orig_models, alpha=0.2)
    plt.savefig(output_prefix + '_original_models.pdf', dpi=600)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    deflator = bc.deflation.Deflator(pref_matrix)
    stats_reg = run_biclustering(image, x, orig_models, pref_matrix, deflator,
                                 ac_tester, output_prefix + '_bic_reg',
                                 gt_groups=gt_groups)

    print('Running compressed bi-clustering')
    compression_level = 32
    deflator = bc.deflation.L1CompressedDeflator(pref_matrix, compression_level)
    stats_comp = run_biclustering(image, x, orig_models, pref_matrix, deflator,
                                  ac_tester, output_prefix + '_bic_comp',
                                  gt_groups=gt_groups)

    return stats_reg, stats_comp


def evaluate_york(res_dir_name, run_with_lsd=False):
    log_filename = res_dir_name + '.txt'
    sampling_factor = 10
    inliers_threshold = np.pi * 1e-2
    epsilon = 0

    sys.stdout = test_utils.Logger(log_filename)
    dir_name = '/Users/mariano/Documents/datasets/YorkUrbanDB/'

    stats_list = []
    for i, example in enumerate(os.listdir(dir_name)):
        if not os.path.isdir(dir_name + example):
            continue
        img_name = dir_name + '{0}/{0}.jpg'.format(example)
        gray_image = PIL.Image.open(img_name).convert('L')

        gt_name = dir_name + '{0}/{0}LinesAndVP.mat'.format(example)
        mat = scipy.io.loadmat(gt_name)
        gt_lines = mat['lines']
        gt_segments = [lsd.Segment(gt_lines[k, :], gt_lines[k + 1, :])
                    for k in range(0, len(gt_lines), 2)]
        gt_segments = np.array(gt_segments)
        gt_association = np.squeeze(mat['vp_association'])

        if run_with_lsd:
            segments = lsd.compute(gray_image)
            segments = np.array(segments)
            gt_groups = ground_truth(gt_association, gt_segments, segments,
                                     inliers_threshold)
        else:
            segments = gt_segments
            gt_groups = [gt_association == v for v in np.unique(gt_association)]

        ac_tester = ac.LocalNFA(segments, epsilon, inliers_threshold)
        sampler = sampling.AdaptiveSampler(int(len(segments) * sampling_factor))
        ransac_gen = sampling.ModelGenerator(vp.VanishingPoint, segments,
                                             sampler)

        print('-'*40)
        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        res = test(gray_image, segments, res_dir_name, example, ransac_gen,
                   ac_tester, gt_groups=gt_groups)
        stats_list.append(res)

        plt.close('all')

    reg_list, comp_list = zip(*stats_list)

    print('Statistics of regular bi-clustering')
    test_utils.print_stats(reg_list)
    print('Statistics of compressed bi-clustering')
    test_utils.print_stats(comp_list)


def run_all():
    evaluate_york('test_vp_lsd', run_with_lsd=True)
    # evaluate_york('test_vp_gt', run_with_lsd=False)


if __name__ == '__main__':
    run_all()
    plt.show()
