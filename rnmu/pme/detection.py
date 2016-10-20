from __future__ import absolute_import, print_function
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import timeit
import seaborn.apionly as sns
import rnmu.pme.approximation as approximation
from rnmu.pme.clique import max_independent_set
from rnmu.pme.stats import meaningful


def run(ransac_gen, data, sigma, cutoff=3, downdate='hard-col', overlaps=True):
    t = timeit.default_timer()
    pref_matrix, orig_models = _build_preference_matrix(ransac_gen, data, sigma,
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

    models, bics = _clean(ransac_gen.model_class, data, sigma, cutoff, overlaps,
                          bics)

    print('Refined biclusters:', len(models))

    return pref_matrix, orig_models, models, bics


def _build_preference_matrix(ransac_gen, elements, sigma, cutoff):
    ransac_gen.elements = elements
    pref_matrix = []
    original_models = []
    for i, model in enumerate(ransac_gen):
        mem = _membership(model, elements, sigma, cutoff)
        if meaningful(mem, model.min_sample_size, trim=False):
            pref_matrix.append(mem)
            original_models.append(model)

    pref_matrix = np.array(pref_matrix).T
    return pref_matrix, original_models


def _membership(model, data, sigma, cutoff):
    dists = model.distances(data) / sigma
    sim = np.exp(-(dists ** 2))
    sim[dists > cutoff] = 0
    return sim


def _clean(model_class, data, sigma, cutoff, overlaps, bics):
    ms_size = model_class().min_sample_size
    bics_final = []
    models = []
    for lf, rf in bics:
        if np.count_nonzero(rf) <= 1 or np.count_nonzero(lf) < ms_size:
            continue
        inliers = np.squeeze(lf)
        mod = model_class(data, weights=inliers)
        mem = _membership(mod, data, sigma, cutoff)
        if meaningful(mem, mod.min_sample_size):
            models.append(mod)
            bics_final.append((mem[:, np.newaxis], rf))

    if not bics_final:
        return [], []

    idx_keep = _eliminate_redundancy(bics_final)
    models = _select(models, idx_keep)
    bics_final = _select(bics_final, idx_keep)

    if not bics_final:
        return [], []

    if not overlaps:
        _solve_intersections(bics_final)

    return models, bics_final


def _eliminate_redundancy(bics):
    left_factors = np.concatenate(zip(*bics)[0], axis=1)
    r = left_factors.T.dot(left_factors)
    norm_bics = np.linalg.norm(left_factors, axis=0)
    r /= np.outer(norm_bics, norm_bics)
    iset = max_independent_set(r > 0.9)
    return iset


def _select(values, idx):
    return [values[i] for i in idx]


def _solve_intersections(bics):
    left_factors = np.concatenate(zip(*bics)[0], axis=1)
    idx = np.argmax(left_factors, axis=1)
    for i, bics in enumerate(bics):
        bics[0][idx != i] = 0


def plot(array_or_bic_list, palette='Set1'):
    def get_cmap(base_color, n_colors=256):
        colors = [np.array([1., 1., 1., 0])] + \
                 sns.light_palette(base_color, n_colors=n_colors - 1)
        return mpl_colors.ListedColormap(colors)

    try:
        plt.imshow(array_or_bic_list, interpolation='none', cmap=get_cmap('k'))
    except TypeError:
        palette = sns.color_palette(palette, len(array_or_bic_list))
        plt.hold(True)
        for (u, v), c in zip(array_or_bic_list, palette):
            plt.imshow(u.dot(v), interpolation='none', cmap=get_cmap(c))

    plt.tick_params(which='both',  # both major and minor ticks are affected
                    bottom='off', top='off', left='off', right='off',
                    labelbottom='off', labelleft='off')
    plt.axis('image')
