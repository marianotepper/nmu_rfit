from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns
import numpy as np
import scipy.sparse
import comdet.biclustering.utils as utils


def create_preference_matrix(n_rows):
    return utils.sparse((n_rows, 0))


def add_col(preference_matrix, in_column, value=1):
    col_shape = (preference_matrix.shape[0], 1)
    col_idx = np.where(in_column)[0]
    data_shape = (len(col_idx),)
    column = utils.sparse((value * np.ones(data_shape),
                           (col_idx, np.zeros(data_shape))),
                          col_shape)
    if preference_matrix.shape[1] > 0:
        preference_matrix = scipy.sparse.hstack([preference_matrix, column])
    else:
        preference_matrix = column
    return preference_matrix


def plot_preference_matrix(array, bands=[], offset=0, labels=None, title=None):
    try:
        n_colors = int(np.max(array))
    except TypeError:
        array = array.toarray()
        n_colors = int(np.max(array))
    palette = sns.cubehelix_palette(n_colors + 1, start=2, rot=0, dark=0.15,
                                    light=1)
    cmap = colors.ListedColormap(palette, N=n_colors + 1)
    plt.imshow(array, interpolation='none', cmap=cmap)
    count = 0
    for k in bands:
        count += k + offset
        plt.plot([count - 0.5] * 2, [-0.5, array.shape[0] - 0.5], 'k')
    plt.tick_params(
        which='both',  # both major and minor ticks are affected
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    plt.axis('image')

    if n_colors > 1:
        cmap = colors.ListedColormap(palette[1:], N=3)
        locs = np.linspace(1, n_colors, n_colors)
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(1, n_colors + 1)
        cb = plt.colorbar(mappable, drawedges=True)
        cb.set_ticks(locs + 0.5)
        if labels is not None:
            cb.set_ticklabels(labels)
        if title is not None:
            cb.ax.set_title(title, loc='left')
        cb.ax.tick_params(left='off', right='off')