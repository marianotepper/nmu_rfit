from __future__ import absolute_import, print_function
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import timeit
import seaborn.apionly as sns
import rnmu.pme.approximation as approximation
from rnmu.pme.stats import meaningful


def run(ransac_gen, data, sigma, cutoff=3, downdate='hard-col', overlaps=True):
    t = timeit.default_timer()
    pref_matrix, orig_models = build_preference_matrix(ransac_gen, data, sigma,
                                                       cutoff)
    t1 = timeit.default_timer() - t
    print('Preference matrix size:', pref_matrix.shape)
    print('Preference matrix computation time:', t1)

    if pref_matrix.size == 0:
        return pref_matrix, orig_models, [], []

    t = timeit.default_timer()
    bics = approximation.recursive_nmu(pref_matrix, downdate=downdate)
    t1 = timeit.default_timer() - t
    print('NMU time:', t1)

    print('Biclusters:', len(bics))

    models, bics = clean(ransac_gen.model_class, data, sigma, cutoff, overlaps,
                         bics)

    print('Refined biclusters:', len(models))

    return pref_matrix, orig_models, models, bics


def build_preference_matrix(ransac_gen, elements, sigma, cutoff):
    ransac_gen.elements = elements
    pref_matrix = []
    original_models = []
    for model in ransac_gen:
        mem = membership(model, elements, sigma, cutoff)
        if meaningful(mem, model.min_sample_size):
            pref_matrix.append(mem)
            original_models.append(model)

    pref_matrix = np.array(pref_matrix).T
    return pref_matrix, original_models


def membership(model, data, sigma, cutoff):
    dists = model.distances(data) / sigma
    sim = np.exp(-(dists ** 2))
    sim[dists > cutoff] = 0
    return sim


def clean(model_class, data, sigma, cutoff, overlaps, bics):
    min_sample_size = model_class().min_sample_size

    bics_final = []
    models = []
    for lf, rf in bics:
        if np.count_nonzero(rf) <= 1 or np.count_nonzero(lf) < min_sample_size:
            continue
        inliers = np.squeeze(lf)
        mod = model_class(data, weights=inliers)
        mem = membership(mod, data, sigma, cutoff)
        if meaningful(mem, mod.min_sample_size):
            models.append(mod)
            bics_final.append((mem[:, np.newaxis], rf))

    if not bics_final:
        return [], []

    models, bics_final = _eliminate_redundancy(bics_final, models)

    if not bics_final:
        return [], []

    if not overlaps:
        _solve_intersections(data, bics_final, models)

    return models, bics_final


def _eliminate_redundancy(bics, models):
    left_factors = np.concatenate(zip(*bics)[0], axis=1)
    r = left_factors.T.dot(left_factors)
    idx = np.unique(np.argmax(r, axis=0))
    models = [models[i] for i in idx]
    bics_final = [bics[i] for i in idx]
    return models, bics_final


def _solve_intersections(data, bics, models):
    left_factors = np.concatenate(zip(*bics)[0], axis=1)
    intersection = np.sum(left_factors > 0, axis=1) > 1
    dists = [mod.distances(data[intersection, :]) for mod in models]
    idx = np.argmin(np.vstack(dists), axis=0)
    for i, bics in enumerate(bics):
        bics[0][intersection] = idx[:, np.newaxis] == i


def plot(array_or_bic_list, palette='Set1'):
    plt.hold(True)
    try:
        plt.imshow(array_or_bic_list, interpolation='none', cmap=cm.gray_r)
    except TypeError:
        palette = sns.color_palette(palette, len(array_or_bic_list))

        for (u, v), c in zip(array_or_bic_list, palette):
            colors = [np.array([1., 1., 1., 0])] +\
                     sns.light_palette(c, n_colors=63)
            cmap = mpl_colors.ListedColormap(colors)
            plt.imshow(u.dot(v), interpolation='none', cmap=cmap)
        plt.tick_params(which='both',  # both major and minor ticks are affected
                        bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labelleft='off')
        plt.axis('image')
