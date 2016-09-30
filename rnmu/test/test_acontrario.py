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

    plt.axis('equal')


def plot_soft_line(line, sigma, box, n_levels=10, color='r', alpha=0.8):
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

    plt.contourf(xi, yi, alphas, levels=levels, colors=colors, antialiased=True)


def beta_derivative(x, a, b):
    fx = scipy.stats.beta.cdf(x, a, b)
    fprime = scipy.stats.beta.pdf(x, a, b)
    # fprime = -fx * ((a + b - 2) * x - a - 1) / ((x-1) * x)
    p = fx - fprime * x
    return fprime, p


def plot_orthogonal_projection(data, mss, sigma, subplots):
    line = Line(data[mss, :])

    dists = line.distances(data) / sigma
    membership = np.exp(-np.power(dists, 2))

    plt.subplot(subplots[0])

    x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
    y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
    bbox = np.vstack((x_lim, y_lim)).T
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.scatter(data[:, 0], data[:, 1], c='w', s=10)
    plt.scatter(data[mss, 0], data[mss, 1], c='r', s=10)
    plot_soft_line(line, sigma, bbox, n_levels=10, color='r', alpha=0.8)

    plt.axis('equal')

    idx = np.logical_and(membership > 0, membership < 1)
    est_values = membership[idx]

    mean = np.mean(est_values)
    var = np.var(est_values, ddof=1)
    a = mean * (mean * (1 - mean) / var - 1)
    b = (1 - mean) * (mean * (1 - mean) / var - 1)
    # print(a, b)

    hist, bin_edges = np.histogram(membership, bins=100, range=(0, 1),
                                   density=True)
    hist = np.cumsum(hist * np.diff(bin_edges))
    x = bin_edges[:-1] + np.diff(bin_edges) / 2
    center = np.logical_and(x > 0.2, x < 0.8)
    x = x[center]
    hist_center = hist[center]
    slope = Line(np.vstack((x, hist_center)).T)

    plt.subplot(subplots[1])
    plt.hist(membership, bins=100, normed=True, cumulative=True, histtype='step', edgecolor='k', linewidth=2)
    # vals = np.linspace(0, 1, num=100)
    # plt.plot(vals, scipy.stats.beta.cdf(vals, a, b), 'g-', linewidth=2)
    # fprime, p = beta_derivative(0.5, a, b)
    # plt.plot([0, 1], [p, fprime + p], 'r--', linewidth=2)
    # plt.scatter(0.5, scipy.stats.beta.cdf(0.5, a, b), c='k')
    # slope.plot(color='r', linewidth=2)
    plt.plot([0, 1], [ hist[0], 1])
    plt.xlim((0, 1))
    plt.ylim((hist[0], 1))
    # one_crossing = -(slope.eq[0] + slope.eq[2]) / slope.eq[1]
    # plt.title('{}'.format(one_crossing))


def main():
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')
    print(mat.keys())

    data = mat['Stairs4_S00075_O60'].T
    # data = mat['Star5_S00075_O50'].T
    # data = 2 * np.random.rand(500, 2) - 1
    # noise = 2 * np.random.rand(500, 2) - 1
    # data = np.vstack([data, noise])
    n = len(data)
    print(n, n_tests(n))

    # plt.figure()

    random_sample = np.random.randint(n, size=2)
    # random_sample = [66, 494]
    # random_sample = [283r, 473]
    # random_sample = [380, 147]
    # random_sample = [36, 392]
    # random_sample = [60, 80]
    print(random_sample)

    # sigma = 0.03
    # plot_projection(data, [0, 40], sigma, [241, 242])
    # plot_projection(data, random_sample, sigma, [245, 246])
    #
    sigma = 0.02
    # plot_projection(data, [0, 40], sigma, [243, 244])
    # plot_projection(data, random_sample, sigma, [247, 248])

    # plot_orthogonal_projection(data, [3, 10], sigma, [221, 222])
    # plot_orthogonal_projection(data, random_sample, sigma, [223, 224])

    for j in range(40, 60):
        random_sample = [0, j]
        plt.figure()
        plot_orthogonal_projection(data, random_sample, sigma, [121, 122])


if __name__ == '__main__':
    # print(np.exp(-.5**2))
    main()
    plt.show()
