from __future__ import absolute_import, print_function
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
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


def load(path):
    data = scipy.io.loadmat(path)
    x = data['data'].T
    gt = np.squeeze(data['label'])

    # sort in reverse order (inliers first, outliers last)
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
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.imshow(gray_image, cmap='gray', interpolation='none')

        for g, c in zip(groups, palette):
            colors = np.repeat(np.atleast_2d(c), len(g), axis=0)
            if colors.shape[1] == 3:
                colors = np.append(colors, g[:, np.newaxis], axis=1)
            if colors.shape[1] == 4:
                colors[:, 3] = g
            plt.scatter(pos[:, 0], pos[:, 1], c=colors,
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
                                         overlaps=False)
    t1 = timeit.default_timer() - t
    print('Total time:', t1)

    if name is not None:
        scipy.io.savemat(name + '.mat', {'pref_mat': pref_mat})

    gt_groups = ground_truth(data['label'])
    gt_colors = sns.color_palette(palette, len(gt_groups) - 1)
    gt_colors.insert(0, [1., 1., 1.])
    plot_models(data, gt_groups, palette=gt_colors)
    if name is not None:
        plt.savefig(name + '_gt10.pdf', dpi=600, bbox_inches='tight',
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

    bc_groups = [b[0].flatten() for b in bics]

    plot_models(data, bc_groups, palette=palette)
    if name is not None:
        plt.savefig(name + '_final_models.pdf', dpi=600, bbox_inches='tight',
                    pad_inches=0)

    if bc_groups:
        bc_groups = [(g > 0).astype(dtype=float) for g in bc_groups]
        outliers = np.sum(np.vstack(bc_groups), axis=0) == 0
    else:
        outliers = np.ones((len(data['data']),))
    bc_groups.append(outliers.astype(dtype=float))

    stats = test_utils.compute_measures(gt_groups, bc_groups)
    stats['time'] = t1
    return stats


def run(transformation, sigma, sampling_type='multigs', n_samples=5000,
        name_prefix=None, test_examples=None):
    if name_prefix is None:
        dir_name = '{}_{}'.format(transformation, sigma)
    else:
        dir_name = '{}_{}_{}'.format(name_prefix, transformation, sigma)
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
        if test_examples is not None and example[:-4] not in test_examples:
            continue
        # if example != 'dinobooks.mat':
        #     continue

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)

        data = load(path + example)

        if transformation == 'homography':
            model_class = homography.Homography
        elif transformation == 'fundamental':
            model_class = fundamental.Fundamental
        else:
            raise RuntimeError('Unknown transformation')

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
        # break

    print('Statistics')
    test_utils.compute_stats(stats_list)
    print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def plot_results(transformation):
    res_dir = '../results'

    _, dir_sigmas, _ = os.walk(res_dir).next()
    dir_sigmas = [ds for ds in dir_sigmas if ds.find(transformation) == 0]
    sigmas = [float(ds[len(transformation) + 1:]) for ds in dir_sigmas]
    idx_sigmas = np.argsort(sigmas)
    sigmas = [sigmas[i] for i in idx_sigmas]
    dir_sigmas = [dir_sigmas[i] for i in idx_sigmas]

    sigma_miss_err = {}
    sigma_times = {'PM': {}, 'NMU': {}, 'TOTAL': {}}
    example_miss_err = {}
    res_files = ['{}/{}/test.txt'.format(res_dir, ds) for ds in dir_sigmas]

    # Very crude parser, do not change console printing output
    # or this will break
    for s, rf in zip(sigmas, res_files):
        with open(rf, 'r') as file_contents:
            sigma_miss_err[s] = []
            sigma_times['PM'][s] = []
            sigma_times['NMU'][s] = []
            sigma_times['TOTAL'][s] = []
            for i, line in enumerate(file_contents):
                if line.find('Statistics') == 0:
                    break
                if i % 10 == 0:
                    example = line[:-5]
                if i % 10 == 3:
                    t = float(line.split()[4])
                    sigma_times['PM'][s].append(t)
                if i % 10 == 4:
                    t = float(line.split()[2])
                    sigma_times['NMU'][s].append(t)
                if i % 10 == 7:
                    t = float(line.split()[2])
                    sigma_times['TOTAL'][s].append(t)
                if i % 10 == 8:
                    pr = 100 * float(line.split()[3][:-1])
                    if example not in example_miss_err:
                        example_miss_err[example] = []
                    example_miss_err[example].append(pr)
                    sigma_miss_err[s].append(pr)

    def sort_dict(d):
        return collections.OrderedDict(sorted(d.items()))

    example_miss_err = sort_dict(example_miss_err)
    sigma_miss_err = sort_dict(sigma_miss_err)
    sigma_times['PM'] = sort_dict(sigma_times['PM'])
    sigma_times['NMU'] = sort_dict(sigma_times['NMU'])
    sigma_times['TOTAL'] = sort_dict(sigma_times['TOTAL'])

    def round2(vals, decimals=2):
        return np.round(vals, decimals=decimals)

    print('Misclassification error')
    for key in sigma_miss_err:
        values = np.array(sigma_miss_err[key])
        stats = (key, round2(np.mean(values)),
                 round2(np.median(values)),
                 round2(np.std(values, ddof=1)))
        fmt_str = 'sigma: {}\tmean: {}\tmedian: {}\tstd: {}'
        print(fmt_str.format(*stats))
        # print('\t', values)

    with sns.axes_style("whitegrid"):
        values = sigma_miss_err.values()
        max_val = max([max(sl) for sl in values])

        plt.figure()
        sns.boxplot(data=values, color='.95', whis=100)
        sns.stripplot(data=values, jitter=True)
        sigmas_text = ['{:.2f}'.format(s) for s in sigmas]
        plt.xticks(range(len(sigmas)), sigmas_text, size='x-large')
        yticks = [yt for yt in plt.yticks()[0] if yt >= 0]
        plt.yticks(yticks, size='x-large')
        plt.xlabel(r'$\sigma$', size='x-large')
        plt.ylabel('Misclassification error (%)', size='x-large')
        ylim = plt.ylim()
        plt.ylim((-2, 10 * np.ceil(max_val / 10)))
        plt.tight_layout()
        plt.savefig('{}/{}_result.pdf'.format(res_dir, transformation),
                    bbox_inches='tight')

    print('Time')
    for key in sigma_miss_err:
        mean_PM = round2(np.mean(np.array(sigma_times['PM'][key])))
        mean_NMU = round2(np.mean((np.array(sigma_times['NMU'][key]))))
        mean_total = round2(np.mean((np.array(sigma_times['TOTAL'][key]))))
        stats = (key, mean_total,
                 round2(mean_PM / mean_total),
                 round2(mean_NMU / mean_total))
        fmt_str = 'sigma: {}\tTOTAL: {}\tRATIO PM: {}\tRATIO PM: {}'
        print(fmt_str.format(*stats))


if __name__ == '__main__':
    # Parameters with best results
    run('homography', 4.33)
    run('fundamental', 4.67)

    # for sigma_unadjusted in np.arange(5, 10.5, .5):
    #     sigma = np.round(sigma_unadjusted / 1.5, decimals=2)
    #     run('homography', sigma)
    # for sigma_unadjusted in np.arange(5, 10.5, .5):
    #     sigma = np.round(sigma_unadjusted / 1.5, decimals=2)
    #     run('fundamental', sigma)

    # plot_results('homography')
    # plot_results('fundamental')

    # These tests need code modification (comment testing) to run properly
    # run('fundamental', 4.67, n_samples=500, name_prefix='notesting',
    #     test_examples=['boardgame'])
    # run('homography', 4.33, n_samples=500, name_prefix='notesting',
    #     test_examples=['johnsonb'])

    plt.show()
