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
import arse.pme.multigs as multigs
import arse.pme.membership as membership
import arse.pme.homography as homography
import arse.pme.fundamental as fundamental
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

    # sort in reverse order (inliers first, outliers last)
    inv_order = np.argsort(gt)[::-1]
    gt = gt[inv_order]
    x = x[inv_order, :]

    x[:, 0:2] -= np.array(data['img1'].shape[:2], dtype=np.float) / 2
    x[:, 3:5] -= np.array(data['img2'].shape[:2], dtype=np.float) / 2

    data['data'] = x
    data['label'] = gt
    return data


def base_plot(data):
    def inner_plot_img(pos, img):
        pos_rc = pos + np.array(img.shape[:2], dtype=np.float) / 2
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.hold(True)
        plt.imshow(gray_image, cmap='gray')
        plt.scatter(pos_rc[:, 0], pos_rc[:, 1], c='w', marker='o', s=10)
        plt.axis('off')

    x = data['data']
    plt.figure()
    plt.subplot(121)
    inner_plot_img(x[:, 0:2], data['img1'])
    plt.subplot(122)
    inner_plot_img(x[:, 3:5], data['img2'])


def plot_models(data, groups, palette, s=10, marker='o'):
    def inner_plot_img(pos, img):
        pos_rc = pos + np.array(img.shape[:2], dtype=np.float) / 2
        plt.hold(True)
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.imshow(gray_image, cmap='gray', interpolation='none')
        for g, color in zip(groups, palette):
            plt.scatter(pos_rc[g, 0], pos_rc[g, 1], c=color, edgecolors='face',
                        marker=marker, s=s)

        # labels = ['{0}'.format(i) for i in range(pos.shape[0])]
        # for label, x, y in zip(labels, pos_rc[:, 0], pos_rc[:, 1]):
        #     plt.annotate(label, xy=(x, y), xytext=(-10, 10), size=3,
        #                  textcoords='offset points', ha='right', va='bottom',
        #                  bbox=dict(boxstyle='round, pad=0.5', fc='yellow',
        #                            alpha=0.5),
        #                  arrowprops=dict(arrowstyle='->', linewidth=.5,
        #                                  color='yellow',
        #                                  connectionstyle='arc3,rad=0'))
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


def run_biclustering(model_class, data, pref_matrix, comp_level, thresholder,
                     ac_tester, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    bic_list = test_utils.clean(model_class, data['data'], thresholder,
                                ac_tester, bic_list, share_elements=False)[1]

    colors = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=colors)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    bc_groups = [np.squeeze(bic[0].toarray()) for bic in bic_list]

    plot_models(data, bc_groups, palette=colors)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    inliers = reduce(lambda a, b: np.logical_or(a, b), bc_groups)
    bc_groups.append(np.logical_not(inliers))
    gt_groups = ground_truth(data['label'])
    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)

    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def test(model_class, data, name, ransac_gen, thresholder, ac_tester,
         dir_name):
    x = data['data']
    print(name, x.shape)

    output_dir = '../results/{0}/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_prefix = output_dir + name

    base_plot(data)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    gt_groups = ground_truth(data['label'])
    gt_colors = sns.color_palette('Set1', len(gt_groups) - 1)
    gt_colors.insert(0, [1., 1., 1.])
    plot_models(data, gt_groups, palette=gt_colors)
    plt.savefig(output_prefix + '_gt10.pdf', dpi=600)
    plot_models(data, gt_groups, palette=gt_colors, s=.1, marker='.')
    plt.savefig(output_prefix + '_gt1.pdf', dpi=600)

    pref_matrix, _ = pref.build_preference_matrix(ransac_gen, thresholder,
                                                  ac_tester)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix})
    with open(output_prefix + '.pickle', 'wb') as handle:
        pickle.dump(pref_matrix, handle)
    with open(output_prefix + '.pickle', 'rb') as handle:
        pref_matrix = pickle.load(handle)

    print('Preference matrix size:', pref_matrix.shape)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    compression_level = None
    stats_reg = run_biclustering(model_class, data, pref_matrix,
                                 compression_level, thresholder, ac_tester,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 32
    stats_comp = run_biclustering(model_class, data, pref_matrix,
                                  compression_level, thresholder, ac_tester,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def run(transformation, inliers_threshold):
    logger = test_utils.Logger('test_{0}_{1:.0e}.txt'.format(transformation,
                                                             inliers_threshold))
    sys.stdout = logger

    n_samples = 2000
    epsilon = 0

    path = '../data/adelaidermf/{0}/'.format(transformation)

    filenames = []
    for (_, _, fn) in os.walk(path):
        filenames.extend(fn)
        break

    stats_list = []
    for example in filenames:
        # if example != 'biscuit.mat':
        #     continue
        # if example != 'biscuitbookbox.mat':
        #     continue
        # if example != 'breadcartoychips.mat':
        #     continue
        # if example != 'boardgame.mat':
        #     continue

        data = load(path + example)

        if transformation == 'homography':
            model_class = homography.Homography
            nfa_proba = np.pi / np.prod(data['img2'].shape[:2])
        else:
            model_class = fundamental.Fundamental
            img_size = data['img2'].shape[:2]
            nfa_proba = (2. * np.linalg.norm(img_size) / np.prod(img_size))

        generator = multigs.ModelGenerator(model_class, data['data'], n_samples)
        min_sample_size = model_class().min_sample_size
        ac_tester = ac.ImageTransformNFA(epsilon, nfa_proba, min_sample_size)
        thresholder = membership.GlobalThresholder(inliers_threshold)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        prefix = example[:-4]
        dir_name = '{0}_{1:.0e}'.format(transformation, inliers_threshold)

        res = test(model_class, data, prefix, generator, thresholder, ac_tester,
                   dir_name)
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
    for thresh in np.power(np.arange(.5, 4, .5), 2):
        run('homography', thresh)
    for thresh in np.arange(2.5e-3, 2.51e-2, 2.5e-3):
        run('fundamental', thresh)

if __name__ == '__main__':
    run_all()
    plt.show()
