from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
import scipy.spatial.distance as distance
import numpy as np
import os
import PIL
import scipy.io
import sys
import timeit
import rnmu.pme.detection as detection
import rnmu.pme.fundamental as fundamental
import rnmu.pme.homography as homography
import rnmu.pme.multigs as multigs
import rnmu.pme.sampling as sampling
import rnmu.test.utils as test_utils


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

        for g, c in zip(groups, palette):
            colors = np.repeat(np.atleast_2d(c), len(g), axis=0)
            if colors.shape[1] == 3:
                colors = np.append(colors, g[:, np.newaxis], axis=1)
            if colors.shape[1] == 4:
                colors[:, 3] = g
            plt.scatter(pos_rc[:, 0], pos_rc[:, 1], c=colors,
                        edgecolors='face', marker=marker, s=s)

        plt.axis('off')

    palette = sns.color_palette(palette, len(groups))

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


def test(ransac_gen, data, sigma, name=None, palette='Set1'):
    t = timeit.default_timer()
    pref_mat, _, _, bics = detection.run(ransac_gen, data['data'], sigma,
                                         overlaps=False, pre_eps=3)
    t1 = timeit.default_timer() - t
    print('Total time:', t1)

    gt_groups = ground_truth(data['label'])
    gt_colors = sns.color_palette(palette, len(gt_groups) - 1)
    gt_colors.insert(0, [1., 1., 1.])
    plot_models(data, gt_groups, palette=gt_colors)
    if name is not None:
        plt.savefig(name + '_gt10.pdf', dpi=600, bbox_inches='tight',
                    pad_inches=0)
    plot_models(data, gt_groups, palette=gt_colors, s=.1, marker='.')
    if name is not None:
        plt.savefig(name + '_gt1.pdf', dpi=600, bbox_inches='tight',
                    pad_inches=0)

    plt.figure()
    detection.plot(pref_mat)
    if name is not None:
        plt.savefig(name + '_pref_mat.png', dpi=600, bbox_inches='tight',
                    pad_inches=0)

    plt.figure()
    detection.plot(bics, palette=palette)
    if name is not None:
        plt.savefig(name + '_pref_mat_bic.png', dpi=600, bbox_inches='tight',
                    pad_inches=0)

    bc_groups = [bic[0].flatten() for bic in bics]

    plot_models(data, bc_groups, palette=palette)
    if name is not None:
        plt.savefig(name + '_final_models.pdf', dpi=600, bbox_inches='tight',
                    pad_inches=0)

    if bc_groups:
        thresh = 0.5
        bc_groups = [(g >= thresh).astype(dtype=float) for g in bc_groups]
        outliers = np.sum(np.vstack(bc_groups), axis=0) == 0
    else:
        outliers = np.ones((len(data['data']),))
    bc_groups.append(outliers.astype(dtype=float))

    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)
    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def run(transformation, sigma, sampling_type='multigs', n_samples=5000):
    dir_name = '{0}_{1}'.format(transformation, sigma)
    output_dir = '../results/{0}/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger = test_utils.Logger(output_dir + 'test.txt')
    sys.stdout = logger

    path = '../data/adelaidermf/{0}/'.format(transformation)

    filenames = []
    for (_, _, fn) in os.walk(path):
        filenames.extend(fn)
        break

    stats_list = []
    for i, example in enumerate(filenames):
        print(example)
        # if example != 'johnsonb.mat':
        #     continue
        # if example != 'bonhall.mat':
        #     continue
        # if example != 'breadcartoychips.mat':
        #     continue
        # if example != 'breadtoy.mat':
        #     continue
        # if example != 'neem.mat':
        #     continue

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)

        data = load(path + example)

        if transformation == 'homography':
            model_class = homography.Homography
        else:
            model_class = fundamental.Fundamental

        if sampling_type == 'multigs':
            generator = multigs.ModelGenerator(model_class, n_samples,
                                               seed=seed)
        elif sampling_type == 'uniform':
            sampler = sampling.UniformSampler(n_samples, seed=seed)
            generator = sampling.ModelGenerator(model_class, sampler)
        else:
            raise RuntimeError('Unknown sampling method')

        output_prefix = output_dir + example[:-4]
        res = test(generator, data, sigma, name=output_prefix)
        stats_list.append(res)

        print('-'*40)
        plt.close('all')

    print('Statistics')
    test_utils.compute_stats(stats_list)
    print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    # run('homography', 7.5)
    # run('fundamental', 8.)
    for thresh in np.arange(5, 10.5, .5):
        run('homography', thresh)
    for thresh in np.arange(5, 10.5, .5):
        run('fundamental', thresh)


if __name__ == '__main__':
    run_all()
    plt.show()
