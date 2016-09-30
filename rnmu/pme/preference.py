from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np


def build_preference_matrix(ransac_gen, elements):
    pref_matrix = np.zeros((elements.shape[0], ransac_gen.n_samples))
    ransac_gen.elements = elements
    original_models = []
    # idx = []
    for i, model in enumerate(ransac_gen):
        mem = membership(model, elements)
        # if mem.sum() < 80:
        #     continue
        # idx.append(i)
        pref_matrix[:, i] = mem
        original_models.append(model)

    # pref_matrix = pref_matrix[:, idx]

    plt.figure()
    plt.stem(pref_matrix.sum(axis=0))

    return pref_matrix, original_models


def membership(model, data):
    dists = model.distances(data) / 0.01
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


# def plot(bic_list, palette='Blues'):
#     plt.hold(True)
#
#     if bic_list:
#         palette = sns.color_palette(palette, len(bic_list))
#
#     for (u, v), c in zip(bic_list, palette):
#         uv = u.dot(v).astype(int)
#         white = mpl_colors.colorConverter.to_rgba('w', alpha=0)
#         c = mpl_colors.colorConverter.to_rgba(c, alpha=1)
#         cmap = mpl_colors.ListedColormap([white, c])
#         try:
#             plt.imshow(uv, interpolation='none', cmap=cmap)
#         except TypeError:
#             plt.imshow(uv.toarray(), interpolation='none', cmap=cmap)
#
#     plt.tick_params(
#         which='both',  # both major and minor ticks are affected
#         bottom='off',
#         top='off',
#         left='off',
#         right='off',
#         labelbottom='off',
#         labelleft='off')
#     plt.axis('image')