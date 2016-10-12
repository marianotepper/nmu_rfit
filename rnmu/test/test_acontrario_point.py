from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import scipy.io
import scipy.stats
from rnmu.pme.point import Point
from rnmu.pme.line import Line
import rnmu.pme.stats as stats


def plot_soft_point(ax, point, sigma, box, n_levels=64, color='r', alpha=0.8):
    x_min, x_max = box
    xi, yi = np.mgrid[slice(x_min[0], x_max[0], .001),
                      slice(x_min[1], x_max[1], .001)]
    pos = np.vstack((xi.flatten(), yi.flatten())).T
    dists = point.distances(pos)
    alphas = scipy.stats.norm.pdf(dists, loc=0, scale=sigma)
    alphas /= alphas.max()
    alphas = alphas.reshape(xi.shape)

    levels = np.linspace(1e-2, 1, num=n_levels, endpoint=True)
    c = plt_colors.ColorConverter().to_rgba(color)
    colors = np.tile(c, (n_levels, 1))
    colors[:, 3] = levels

    ax.contourf(xi, yi, alphas, levels=levels, colors=colors, antialiased=True)


def plot_orthogonal_projection(data, mss, sigma, cutoff, axes=None):
    point = Point(data[mss, :])

    dists = point.distances(data) / sigma
    membership = np.exp(-(dists ** 2))
    idx = membership > np.exp(-(cutoff ** 2))
    membership = membership[idx]

    nfa = stats.concentration_nfa(membership, len(mss))

    if axes is not None:
        x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
        y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
        bbox = np.vstack((x_lim, y_lim)).T

        ax = axes[0]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.scatter(data[:, 0], data[:, 1], c='w', s=10)
        ax.scatter(data[mss, 0], data[mss, 1], c='r', s=10)
        plot_soft_point(ax, point, sigma, bbox, n_levels=10, color='r', alpha=0.8)
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

    return nfa


def main(sigma=0.07, cutoff=1.5):
    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print('seed:', seed)
    np.random.seed(seed)

    # data = mat['Stairs4_S00075_O60'].T
    # data = mat['Star5_S00075_O50'].T
    noise = np.random.rand(950, 2)
    cl = np.random.rand(50, 2) * 0.1 + 0.5
    data = np.append(noise, cl, axis=0)

    n = len(data)
    print(n)

    # random_sample = np.random.randint(n, size=1)
    # print(random_sample)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    plot_orthogonal_projection(data, [300], sigma, cutoff, axes[:, 0])
    plot_orthogonal_projection(data, [976], sigma, cutoff, axes[:, 1])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
    y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.scatter(data[:, 0], data[:, 1], c='w', s=10)

    for i in range(len(data)):
        nfa = plot_orthogonal_projection(data, [i], sigma, cutoff)
        if nfa < 0:
            print(i, nfa)
            ax.scatter(data[i, 0], data[i, 1], c='r', s=20)

    ax.set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    main()
    plt.show()
