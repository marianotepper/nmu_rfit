from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import scipy.io
import scipy.stats
import seaborn.apionly as sns
from rnmu.pme.line import Line


def n_tests(n):
    return n * (n - 1.) / 2


def nfa(line, data, sigma, n_bins=10):
    proj, loc, u, x0 = line.project(data)
    dists = np.linalg.norm(proj - data, axis=1) / sigma
    # to_keep = dists <= 3
    membership = np.exp(-(dists ** 2))

    lsp = np.linspace(loc.min(), loc.max(), num=n_bins, endpoint=False)
    labels = np.searchsorted(lsp, loc, side='right')
    # counts = np.array([membership[np.logical_and(labels == k, to_keep)].sum()
    #                    for k in range(1, n_bins + 1)])
    counts = np.array([membership[labels == k].sum()
                       for k in range(1, n_bins + 1)])
    total_count = counts.sum()

    bin_proba = n_bins ** -1
    expected = total_count.sum() * bin_proba
    bin_deviation = (counts - expected) / np.sqrt(expected)

    # chi_val = (bin_deviation ** 2).sum()
    # print(chi_val, scipy.stats.chi2.cdf(chi_val, n_bins - 1))

    # print(bin_deviation)

    probas = scipy.stats.halfnorm.sf(np.abs(bin_deviation), scale=1 - bin_proba)
    probas.sort()
    probas = probas[1:]
    # print(probas)
    print('sigma:', sigma, 'PFA:', probas.prod(), 'NFA:', n_tests(len(data)) * probas.prod())

    # bin_deviation[bin_deviation >= 0] = 0
    # stat = scipy.stats.halfnorm.sf(np.abs(bin_deviation), scale=1 - bin_proba)
    # print(stat)
    # print('PFA:', stat.prod(), 'NFA:', n_tests(len(data)) * stat.prod())
    # stat = scipy.stats.halfnorm.sf(np.abs(bin_deviation), scale=1 - bin_proba)
    # print(stat)
    # print((stat).prod())

    # plt.figure()
    # plt.stem(bin_deviation)


def plot_projection(data, mss, sigma, subplots):
    line = Line(data[mss, :])
    proj, loc, u, x0 = line.project(data)

    dists = line.distances(data) / sigma
    membership = np.exp(-np.power(dists, 2))

    nfa(line, data, sigma)

    plt.subplot(subplots[0])

    x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
    y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
    bbox = np.vstack((x_lim, y_lim)).T
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.scatter(data[:, 0], data[:, 1], c='w', s=10)
    plt.scatter(data[mss, 0], data[mss, 1], c='r', s=10)
    plt.scatter(data[100, 0], data[100, 1], c='c', s=10)

    plot_soft_line(line, sigma, bbox, n_levels=10, color='r', alpha=0.8)

    plt.axis('equal')

    plt.subplot(subplots[1])
    for s in np.linspace(loc.min(), loc.max(), num=11, endpoint=True):
        plt.scatter(s, 1, c='r', marker='|', s=100)
        plt.scatter(s, 1.5, c='r', marker='|', s=100)
        plt.scatter(s, 2, c='r', marker='|', s=100)

    c = plt_colors.ColorConverter().to_rgba('b')
    color = np.array(len(data) * [c])
    plt.scatter(loc, np.ones_like(loc), color=color, marker='o', s=10)

    members = np.linalg.norm(proj - data, axis=1) < 3 * sigma
    loc_members = loc[members]
    plt.scatter(loc_members, 1.5 * np.ones_like(loc_members), color=color,
                marker='o', s=10)

    color = np.array(len(data) * [c])
    color[:, 3] = membership
    plt.scatter(loc, 2 * np.ones_like(loc), color=color, marker='o', s=10)


def plot_soft_line(ax, line, sigma, box, n_levels=10, color='r', alpha=0.8):
    x_min, x_max = box
    xi, yi = np.mgrid[slice(x_min[0], x_max[0], .001),
                      slice(x_min[1], x_max[1], .001)]
    pos = np.vstack((xi.flatten(), yi.flatten())).T
    dists = line.distances(pos)
    alphas = scipy.stats.norm.pdf(dists, loc=0, scale=sigma)
    alphas /= alphas.max()
    alphas = alphas.reshape(xi.shape)

    levels = np.linspace(1e-2, 1, num=n_levels, endpoint=True)
    c = plt_colors.ColorConverter().to_rgba(color)
    colors = np.tile(c, (n_levels, 1))
    colors[:, 3] = levels

    ax.contourf(xi, yi, alphas, levels=levels, colors=colors, antialiased=True)


def beta_derivative(x, a, b):
    fx = scipy.stats.beta.cdf(x, a, b)
    fprime = scipy.stats.beta.pdf(x, a, b)
    # fprime = -fx * ((a + b - 2) * x - a - 1) / ((x-1) * x)
    p = fx - fprime * x
    return fprime, p


def plot_orthogonal_projection(data, mss, sigma, axes):
    line = Line(data[mss, :])

    dists = line.distances(data) / sigma
    membership = np.exp(-(dists ** 2))

    if axes is not None:
        x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
        y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
        bbox = np.vstack((x_lim, y_lim)).T

        ax = axes[0]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.scatter(data[:, 0], data[:, 1], c='w', s=10)
        ax.scatter(data[mss, 0], data[mss, 1], c='r', s=10)
        plot_soft_line(ax, line, sigma, bbox, n_levels=10, color='r', alpha=0.8)
        ax.set_aspect('equal', adjustable='box')

    cutoff = 3
    idx = membership > np.exp(-(cutoff ** 2))
    membership = membership[idx]
    membership = np.insert(membership, 0, 0)

    pvalue = scipy.stats.kstest(membership, 'uniform', alternative='less')[1]

    if axes is not None:
        membership.sort()

        n = len(membership)
        acc = np.arange(n, dtype=np.float)
        acc /= n

        below = membership > acc

        idx = np.where(below)[0]
        starts = np.setdiff1d(idx - 1, idx)
        ends = np.setdiff1d(idx + 1, idx)

        def get_crossing(i):
            if i < 0 or i + 2 >= n:
                return None
            sl = slice(i, i + 2)
            x1 = membership[sl]
            y2 = acc[sl]

            pts2 = np.vstack((x1, y2)).T
            crossing = np.cross([-1, 1, 0], Line(pts2).eq)
            crossing /= crossing[2]
            return crossing

        membership_all_runs = []
        acc_all_runs = []
        for s, e in zip(starts, ends):
            run = slice(s + 1, e)
            membership_run = membership[run]
            acc_run = acc[run]

            crossing = get_crossing(s)
            if crossing is not None:
                membership_run = np.insert(membership_run, 0, crossing[0])
                acc_run = np.insert(acc_run, 0, crossing[1])

            crossing = get_crossing(e - 1)
            if crossing is not None:
                membership_run = np.append(membership_run, crossing[0])
                acc_run = np.append(acc_run, crossing[1])

            membership_all_runs = np.hstack(
                (membership_all_runs, membership_run))
            acc_all_runs = np.hstack((acc_all_runs, acc_run))

        ax = axes[1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.plot(membership, acc, 'k-', linewidth=2)
        ax.plot([0, 1], [0, 1], 'g-', linewidth=2)

        ax.fill_between(membership_all_runs, membership_all_runs, acc_all_runs,
                         color='g', alpha=0.5)

        arg_Dmin = np.argmax(membership - acc)
        pt = membership[arg_Dmin]
        ax.plot([pt, pt], [pt, acc[arg_Dmin]], 'r-', linewidth = 2)

        ax.set_aspect('equal', adjustable='box')

        # ax.set_title('p-value: {:.2e}'.format(pvalue))
        ax.set_title('NFA: {:.2e}'.format(n_tests(n) * pvalue))

    return pvalue


def main(sigma=0.02):
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')

    # data = mat['Stairs4_S00075_O60'].T
    # data = mat['Star5_S00075_O50'].T
    data = 2 * np.random.rand(500, 2) - 1
    n = len(data)
    print(n, n_tests(n))

    random_sample = np.random.randint(n, size=2)
    # random_sample = [1, 30]
    print(random_sample)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_orthogonal_projection(data, random_sample, sigma, axes)

    # i = 0
    # for j in range(0, 50):
    #     if i == j:
    #         continue
    #     random_sample = [i, j]
    #     plt.figure()
    #     plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


def paper_figure(sigma=0.02):
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')

    data = mat['Stairs4_S00075_O60'].T
    n = len(data)
    print(n, n_tests(n))

    fig, axes = plt.subplots(nrows=2, ncols=4)

    plot_orthogonal_projection(data, [3, 10], sigma, axes[:, 0])
    plot_orthogonal_projection(data, [60, 92], sigma, axes[:, 1])
    plot_orthogonal_projection(data, [4, 180], sigma, axes[:, 2])
    plot_orthogonal_projection(data, [250, 260], sigma, axes[:, 3])

    fig.tight_layout(pad=0, h_pad=-3, w_pad=-1)


def half_layout(sigma=0.02):
    data = np.random.rand(500, 2)
    data[:, 0] /= 2
    data = np.append(data, np.array([[1, 0], [.5, 0], [.5, 1]]), axis=0)
    n = len(data)
    print(data.shape, n_tests(n))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    plot_orthogonal_projection(data, [-1, -2], sigma, axes)

    # i = 0
    # for j in range(0, 50):
    #     if i == j:
    #         continue
    #     random_sample = [i, j]
    #     plt.figure()
    #     plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


if __name__ == '__main__':
    main()
    # half_layout()
    # paper_figure()
    plt.show()
