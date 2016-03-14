from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import seaborn.apionly as sns
import numpy as np
import scipy.sparse as sp
import arse.pme.acontrario as ac


class PreferenceMatrix(object):
    def __init__(self, n_rows):
        self.mat = sp.csc_matrix((n_rows, 0))

    def add_col(self, in_column):
        in_column = (in_column > 0).astype(np.float)
        column = sp.csc_matrix(in_column[:, np.newaxis])
        if self.mat.shape[1] > 0:
            self.mat = sp.hstack([self.mat, column])
        else:
            self.mat = column


def build_preference_matrix(ransac_gen, thresholder, ac_tester):
    pref_matrix = PreferenceMatrix(ransac_gen.elements.shape[0])
    original_models = []
    for i, model in enumerate(ac.ifilter(ransac_gen, thresholder, ac_tester)):
        membership = thresholder.membership(model, ransac_gen.elements)
        pref_matrix.add_col(membership)
        original_models.append(model)

    return pref_matrix.mat, original_models


def plot(array, bic_list=[], palette='Set1'):
    plt.hold(True)
    try:
        plt.imshow(array, interpolation='none', cmap=cm.gray_r)
    except TypeError:
        plt.imshow(array.toarray(), interpolation='none', cmap=cm.gray_r)

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
