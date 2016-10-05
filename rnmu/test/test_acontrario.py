from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import scipy.io
import scipy.stats
import seaborn.apionly as sns
from rnmu.pme.line import Line
import rnmu.pme.stats as stats


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


def n_tests(n):
    return n * (n - 1.) / 2


def plot_projection(data, mss, sigma, axes):
    line = Line(data[mss, :])
    # n = len(data)

    proj, loc, u, x0 = line.project(data)
    dists = np.linalg.norm(proj - data, axis=1) / sigma
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
    loc = loc[idx]
    # loc.sort()
    # loc = np.diff(loc)
    loc = (loc - loc.min()) / (loc.max() - loc.min())

    # pvalue = stats.weighted_kstest(loc, membership, 'uniform', alternative='two-sided')[1]
    pvalue = scipy.stats.kstest(loc, 'uniform', alternative='two-sided')[1]

    if axes is not None:
        idx = np.argsort(loc)
        vals = loc[idx]
        vals = np.insert(vals, 0, 0)
        hist = np.insert(membership[idx], 0, 0)
        hist = np.cumsum(hist)
        hist /= hist[-1]

        ax = axes[1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.plot(vals, hist, 'k-', linewidth=2)
        ax.plot([0, 1], [0, 1], 'g-', linewidth=2)

        # ax.fill_between(hist, hist, vals, color='g', alpha=0.3)

        # arg_Dmin = np.argmax(membership - acc)
        # pt = membership[arg_Dmin]
        # ax.plot([pt, pt], [pt, d], 'r-', linewidth = 2)

        ax.set_aspect('equal', adjustable='box')

        ax.set_title('p-value: {:.2e}'.format(pvalue))
        # ax.set_title('NFA: {:.2e}'.format(n_tests(n) * pvalue))

    return pvalue


def plot_orthogonal_projection(data, mss, sigma, axes):
    line = Line(data[mss, :])

    dists = line.distances(data) / sigma
    membership = np.exp(-(dists ** 2))

    nfa = stats.concentration_nfa(membership)

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


    membership.sort()
    membership = np.insert(membership, 0, 0)

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

    ax.fill_between(membership_all_runs, membership_all_runs, acc_all_runs,
                     color='g', alpha=0.3)

    arg_Dmin = np.argmax(membership - acc)
    pt = membership[arg_Dmin]
    ax.plot([pt, pt], [pt, acc[arg_Dmin]], 'r-', linewidth=2)

    ax.plot(membership, acc, 'k-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'g-', linewidth=2)

    ax.set_aspect('equal', adjustable='box')

    ax.set_title('NFA: {:.2e}'.format(nfa))


def main(sigma=0.02, transversal=True):
    if transversal:
        fun = plot_orthogonal_projection
    else:
        fun = plot_projection

    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')

    # data = mat['Stairs4_S00075_O60'].T
    # data = mat['Star5_S00075_O50'].T
    data = np.random.rand(500, 2)
    n = len(data)
    print(n, n_tests(n))

    random_sample = np.random.randint(n, size=2)
    # random_sample = [1, 30]
    print(random_sample)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fun(data, random_sample, sigma, axes)

    # i = 0
    # for j in range(0, 50):
    #     if i == j:
    #         continue
    #     random_sample = [i, j]
    #     plt.figure()
    #     plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


def paper_figure(sigma=0.02, transversal=True):
    if transversal:
        fun = plot_orthogonal_projection
    else:
        fun = plot_projection

    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')

    data = mat['Stairs4_S00075_O60'].T
    n = len(data)
    print(n, n_tests(n))

    fig, axes = plt.subplots(nrows=2, ncols=4)

    fun(data, [3, 10], sigma, axes[:, 0])
    fun(data, [60, 92], sigma, axes[:, 1])
    fun(data, [4, 180], sigma, axes[:, 2])
    fun(data, [250, 260], sigma, axes[:, 3])

    fig.tight_layout(pad=0, h_pad=-3, w_pad=-1)


def half_layout(sigma=0.02, add_line=False, transversal=True):
    if transversal:
        fun = plot_orthogonal_projection
    else:
        fun = plot_projection

    data = np.random.rand(1000, 2)
    data[:, 0] /= 2
    data = np.append(data, np.array([[1, 0]]), axis=0)

    if add_line:
        width = 1. / 50
        line = np.random.rand(50, 2)
        line[:, 1] *= width
        line = np.append(line, np.array([[0, width/2], [1, width/2]]), axis=0)
        theta = np.radians(30)
        rot = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
        line = line.dot(rot)
        line[:, 1] += 0.2

        data = np.append(data, line, axis=0)
    else:
        data = np.append(data, np.array([[.5, 0], [.5, 1]]), axis=0)


    n = len(data)
    print(data.shape, n_tests(n))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    sample = [-1, -2]
    fun(data, sample, sigma, axes)

    # i = 0
    # for j in range(0, 50):
    #     if i == j:
    #         continue
    #     random_sample = [i, j]
    #     plt.figure()
    #     plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


def dashed_layout(sigma=0.02, transversal=True):
    if transversal:
        fun = plot_orthogonal_projection
    else:
        fun = plot_projection

    n = 20
    data = np.hstack((np.random.rand(n, 1), np.zeros((n, 1))))
    # n_parts = 5
    # data = []
    # for i in range(n_parts):
    #     data.append(np.random.rand(n, 2) + [2*i, 0])
    # data = np.vstack(data)
    # print(data.shape)
    width = 0
    data[:, 1] *= width
    data = np.append(data, np.array([[0, width/2],
                                     [1, width/2]]), axis=0)

    n = len(data)
    print(data.shape, n_tests(n))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    sample = [-1, -2]
    fun(data, sample, sigma, axes)

    # i = 0
    # for j in range(0, 50):
    #     if i == j:
    #         continue
    #     random_sample = [i, j]
    #     plt.figure()
    #     plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


if __name__ == '__main__':
    transversal = False
    # main(transversal=transversal)
    # half_layout(transversal=transversal)
    # half_layout(add_line=True, transversal=transversal)
    # paper_figure(transversal=transversal)
    dashed_layout(transversal=False)
    plt.show()
