from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import timeit
import rnmu.pme.approximation as approximation
from rnmu.pme.stats import meaningful, concentration_nfa


def run(ransac_gen, data, sigma, n_models=10):
    pref_matrix, orig_models = build_preference_matrix(ransac_gen, data, sigma)
    print('Preference matrix size:', pref_matrix.shape)

    t = timeit.default_timer()
    bic_list = approximation.recursive_nmu(pref_matrix, r=n_models)
    t1 = timeit.default_timer() - t
    print('NMU Time:', t1)

    print('Biclusters:', len(bic_list))

    models, bic_list = clean(ransac_gen.model_class, data, sigma, bic_list)

    print('Models', len(models))

    return pref_matrix, orig_models, models, bic_list


def build_preference_matrix(ransac_gen, elements, sigma):
    ransac_gen.elements = elements
    pref_matrix = []
    original_models = []
    for i, model in enumerate(ransac_gen):
        mem = membership(model, elements, sigma)
        if meaningful(mem, model.min_sample_size):
            pref_matrix.append(mem)
            original_models.append(model)

    pref_matrix = np.array(pref_matrix).T
    return pref_matrix, original_models


def membership(model, data, sigma, cutoff=3):
    dists = model.distances(data) / sigma
    sim = np.exp(-np.power(dists, 2))
    sim[dists > 3] = 0
    return sim


def plot(array, palette='Blues'):
    plt.imshow(array, interpolation='none', cmap=plt.get_cmap(palette))
    plt.tick_params(
        which='both',  # both major and minor ticks are affected
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    plt.axis('image')


def clean(model_class, data, sigma, bic_list):
    min_sample_size = model_class().min_sample_size
    bic_list = [bic for bic in bic_list
                if np.count_nonzero(bic[1]) >= 1 and
                np.count_nonzero(bic[0]) >= min_sample_size]

    bic_list_final = []
    model_list = []
    for lf, rf in bic_list:
        inliers = np.squeeze(lf)
        mod = model_class(data, weights=inliers)
        mem = membership(mod, data, sigma)
        if meaningful(mem, mod.min_sample_size):
            model_list.append(mod)
            bic_list_final.append((lf, rf))

    return model_list, bic_list_final


def keep_meaningful(inliers_list, model_list, bic_list):
    z_list = zip(inliers_list, model_list, bic_list)
    z_list = filter(lambda e: meaningful(e[0], e[1].min_sample_size), z_list)
    if z_list:
        return zip(*z_list)
    else:
        return [], [], []
