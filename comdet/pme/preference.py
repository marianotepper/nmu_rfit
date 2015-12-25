from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import seaborn.apionly as sns
import numpy as np
import scipy.sparse as sp
import comdet.pme.acontrario as ac


def create_preference_matrix(n_rows):
    return sp.csc_matrix((n_rows, 0))


def add_col(preference_matrix, in_column, value=1):
    col_shape = (preference_matrix.shape[0], 1)
    col_idx = np.where(in_column)[0]
    data_shape = (len(col_idx),)
    column = sp.csc_matrix((value * np.ones(data_shape),
                           (col_idx, np.zeros(data_shape))),
                           col_shape)
    if preference_matrix.shape[1] > 0:
        preference_matrix = sp.hstack([preference_matrix, column])
    else:
        preference_matrix = column
    return preference_matrix


def plot(array, bic_list=[], palette='Set1'):
    white = mpl_colors.colorConverter.to_rgba('w', alpha=1)
    black = mpl_colors.colorConverter.to_rgba('k', alpha=1)

    cmap = mpl_colors.ListedColormap([white, black])
    plt.hold(True)
    try:
        plt.imshow(array, interpolation='none', cmap=cmap)
    except TypeError:
        plt.imshow(array.toarray(), interpolation='none', cmap=cmap)

    if bic_list:
        palette = sns.color_palette(palette, len(bic_list))

    for (u, v), c in zip(bic_list, palette):
        uv = u.dot(v).astype(int)
        white = mpl_colors.colorConverter.to_rgba('w', alpha=0)
        c = mpl_colors.colorConverter.to_rgba(c, alpha=1)
        cmap = mpl_colors.ListedColormap([white, c])
        try:
            plt.imshow(uv, interpolation='none', cmap=cmap)
        except TypeError:
            plt.imshow(uv.toarray(), interpolation='none', cmap=cmap)

    plt.tick_params(
        which='both',  # both major and minor ticks are affected
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    plt.axis('image')


def build_preference_matrix(n_elements, ransac_gen, ac_tester):
    pref_matrix = create_preference_matrix(n_elements)
    original_models = []
    for model in ac.ifilter(ac_tester, ransac_gen):
        pref_matrix = add_col(pref_matrix, ac_tester.inliers(model))
        original_models.append(model)
        # If sampler allows it, bias sampling towards points with
        # 'low participation' in the preference matrix. If not, this
        #  has now effect.
        distribution = np.squeeze(np.asarray(pref_matrix.sum(axis=1)))
        ransac_gen.sampler.distribution = distribution

    return pref_matrix, original_models
