from __future__ import absolute_import, print_function
import sys
import os
import PIL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
import scipy.spatial.distance as distance
import numpy as np
import scipy.io
import pickle
import timeit
import arse.biclustering as bc
import arse.test.utils as test_utils
import arse.pme.preference as pref
import arse.pme.sampling as sampling
import arse.pme.homography as homography
import arse.pme.acontrario as ac


def load(path, tol=1e-5):
    data = scipy.io.loadmat(path)

    x = data['data'].T
    gt = np.squeeze(data['label'])

    # remove repeated points
    m = x.shape[0]
    dist = distance.squareform(distance.pdist(x)) + np.triu(np.ones((m, m)), 0)
    mask = np.all(dist >= tol, axis=1)
    gt = gt[mask]
    x = x[mask, :]

    # sort in reverse order (inliers first, ourliers last)
    inv_order = np.argsort(gt)[::-1]
    gt = gt[inv_order]
    x = x[inv_order, :]

    data['data'] = x
    data['label'] = gt
    return data


def base_plot(data):
    def inner_plot_img(pos, img):
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.hold(True)
        plt.imshow(gray_image, cmap='gray')
        plt.scatter(pos[:, 0], pos[:, 1], c='w', marker='o', s=10)
        plt.axis('off')

    x = data['data']
    plt.figure()
    plt.subplot(121)
    inner_plot_img(x[:, 0:2], data['img1'])
    plt.subplot(122)
    inner_plot_img(x[:, 3:5], data['img2'])


def plot_models(data, groups, palette, s=10, marker='o'):
    def inner_plot_img(pos, img):
        plt.hold(True)
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.imshow(gray_image, cmap='gray')
        for g, color in zip(groups, palette):
            plt.scatter(pos[g, 0], pos[g, 1], c=color, edgecolors='face',
                        marker=marker, s=s)
        plt.axis('off')

    x = data['data']
    plt.figure()
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0)

    plt.subplot(gs[0])
    inner_plot_img(x[:, 0:2], data['img1'])
    plt.subplot(gs[1])
    inner_plot_img(x[:, 3:5], data['img2'])


def ground_truth(labels):
    gt_groups = []
    for i in np.unique(labels):
        gt_groups.append(labels == i)
    return gt_groups


def run_biclustering(model_class, data, original_models, pref_matrix, comp_level,
                     ac_tester, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, data['data'], ac_tester,
                                        bic_list)

    colors = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=colors)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    bc_groups = [np.squeeze(bic[0].toarray()) for bic in bic_list]

    plot_models(data, bc_groups, palette=colors)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    inliers = reduce(lambda x, y: np.logical_or(x, y), bc_groups)
    bc_groups.append(np.logical_not(inliers))
    gt_groups = ground_truth(data['label'])
    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)

    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def test(model_class, data, name, ransac_gen, ac_tester):
    x = data['data']
    print(name, x.shape)

    output_prefix = '../results/' + name

    base_plot(data)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    gt_groups = ground_truth(data['label'])
    gt_colors = sns.color_palette('Set1', len(gt_groups))
    plot_models(data, gt_groups, palette=gt_colors)
    plt.savefig(output_prefix + '_gt10.pdf', dpi=600)
    plot_models(data, gt_groups, palette=gt_colors, s=.1, marker='.')
    plt.savefig(output_prefix + '_gt1.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(x.shape[0],
                                                            ransac_gen,
                                                            ac_tester)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix})
    with open(output_prefix + '.pickle', 'wb') as handle:
        pickle.dump(pref_matrix, handle)
        pickle.dump(orig_models, handle)
    with open(output_prefix + '.pickle', 'rb') as handle:
        pref_matrix = pickle.load(handle)
        orig_models = pickle.load(handle)

    print('Preference matrix size:', pref_matrix.shape)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    compression_level = None
    stats_reg = run_biclustering(model_class, data, orig_models, pref_matrix,
                                 compression_level, ac_tester,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 32
    stats_comp = run_biclustering(model_class, data, orig_models, pref_matrix,
                                  compression_level, ac_tester,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def run_all():
    logger = test_utils.Logger("test_homography.txt")
    sys.stdout = logger

    inliers_threshold = 3.
    sampling_factor = int(2e4)
    epsilon = 0

    path = '../data/adelaidermf/'

    filenames = []
    for (_, _, fn) in os.walk(path):
        filenames.extend(fn)
        break

    stats_list = []
    for example in filenames:
        data = load(path + example)
        x = data['data']

        n_samples = x.shape[0] * sampling_factor
        sampler = sampling.AdaptiveSampler(n_samples)
        ransac_gen = sampling.ModelGenerator(homography.Homography, x, sampler)
        ac_tester = ac.LocalNFA(x, epsilon, inliers_threshold, ratio=30)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        prefix = example[:-4]
        res = test(homography.Homography, data, prefix, ransac_gen, ac_tester)
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


if __name__ == '__main__':
    run_all()
    plt.show()
