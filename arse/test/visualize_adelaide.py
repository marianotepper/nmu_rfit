from __future__ import absolute_import, print_function
import PIL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
import scipy.spatial.distance as distance
import numpy as np
import scipy.io


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


def plot_models(data, selection=[], s=10, marker='o'):
    def inner_plot_img(pos, img):
        pos_rc = pos + np.array(img.shape[:2], dtype=np.float) / 2
        plt.hold(True)
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.imshow(gray_image, cmap='gray', interpolation='none')
        for g, color in zip(groups, palette):
            plt.scatter(pos_rc[g, 0], pos_rc[g, 1], c=color, edgecolors='face',
                        marker=marker, s=s)

        labels = ['{0}'.format(i) for i in selection]
        pos_rc = pos_rc[selection, :]
        for label, x, y in zip(labels, pos_rc[:, 0], pos_rc[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(-10, 10), size=3,
                         textcoords='offset points', ha='right', va='bottom',
                         bbox=dict(boxstyle='round, pad=0.5', fc='yellow',
                                   alpha=0.5),
                         arrowprops=dict(arrowstyle='->', linewidth=.5,
                                         color='yellow',
                                         connectionstyle='arc3,rad=0'))
        plt.axis('off')

    groups = ground_truth(data['label'])
    palette = sns.color_palette('Set1', len(groups) - 1)
    palette.insert(0, [1., 1., 1.])

    x = data['data']

    if not selection:
        selection = np.arange(x.shape[0])

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


def plot(transformation, example, selection=[]):
    path = '../data/adelaidermf/{0}/{1}.mat'.format(transformation, example)
    data = load(path)
    print(path, data['data'].shape)

    # plot_models(data, selection=selection)
    # plt.savefig(example + '_gt10.pdf')
    # plt.savefig(example + '_gt10.svg')
    # plot_models(data, selection=selection, s=.1, marker='.')
    # plt.savefig(example + '_gt1.pdf')
    # plt.savefig(example + '_gt1.svg')

if __name__ == '__main__':
    plot('homography', 'barrsmith', [0, 8, 11, 12, 20])
    plot('fundamental', 'boardgame', [0, 1, 2, 3, 7, 12, 13, 14, 15, 16, 58, 70,
                                     71, 77, 98, 106])

    plt.show()
