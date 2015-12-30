from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn.apionly as sns
import numpy as np
import scipy.sparse as sp
import timeit
import os
import comdet.pme.preference as pref
import comdet.biclustering as bc
import comdet.test.utils as test_utils


def base_plot(x, size=2, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='w', marker='o', s=size)

    lower = x.min(axis=0)
    upper = x.max(axis=0)
    med = np.zeros((3,))
    for i in range(x.shape[1]):
        med[i] = np.median(np.unique(x[:, i]))
    max_diff = np.max(upper - lower) / 2
    ax.set_xlim(med[0] - max_diff, med[0] + max_diff)
    ax.set_ylim(med[1] - max_diff, med[1] + max_diff)
    ax.set_zlim(med[2] - max_diff, med[2] + max_diff)
    ax.view_init(elev=10.)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if filename is not None:
        plt.savefig(filename + '.pdf', dpi=600)
    return fig, ax


def plot_final_models(data, mod_inliers_list, palette, filename=None,
                      show_data=True, save_animation=False):
    if show_data:
        size = 2
    else:
        size = 0
    fig, ax = base_plot(data, size=size)
    for (mod, inliers), color in zip(mod_inliers_list, palette):
        lower = data[inliers, :].min(axis=0)
        upper = data[inliers, :].max(axis=0)
        limits = zip(lower, upper)
        mod.plot(ax, limits=limits, color=color, linewidth=5, alpha=0.7)

    if filename is not None:
        plt.savefig(filename + '.pdf', dpi=600)

    if save_animation:
        def init():
            return

        def animate(k):
            if k < 360:
                ax.view_init(elev=10., azim=k)
            elif k < 360 + 85:
                ax.view_init(elev=k-360+10., azim=360)
            else:
                ax.view_init(elev=85., azim=k-85)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=805, interval=10)
        anim.save(filename + '.mp4', fps=30, dpi=150,
                  extra_args=['-vcodec', 'libx264'])


def plot_original_models(x, original_models, bic_list, palette, filename=None):
    fig, ax = base_plot(x, size=5)
    for i, (lf, rf) in enumerate(bic_list):
        inliers = sp.find(lf)[0]
        lower = x[inliers, :].min(axis=0)
        upper = x[inliers, :].max(axis=0)
        limits = [(lower[k], upper[k]) for k in range(x.shape[1])]
        for j in sp.find(rf)[1]:
            original_models[j].plot(ax, limits=limits, color=palette[i],
                                    alpha=0.5)

    if filename is not None:
        plt.savefig(filename + '.pdf', dpi=600)


def run_biclustering(model_class, x, original_models, pref_matrix, deflator,
                     ac_tester, output_prefix, plotter=None, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(deflator)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, x, ac_tester, bic_list)

    palette = sns.color_palette(palette, len(bic_list), desat=.5)

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    mod_inliers_list = [(mod, ac_tester.inliers(mod)) for mod in models]

    filename = output_prefix + '_final_models'
    plot_final_models(x, mod_inliers_list, palette, filename=filename,
                      save_animation=True)
    plot_final_models(x, mod_inliers_list, palette, show_data=False,
                      filename=filename + '_clean', save_animation=True)

    if plotter is not None:
        if not os.path.exists(output_prefix):
            os.mkdir(output_prefix)
        plotter.dirname_out = output_prefix + '/'
        plotter.plot(mod_inliers_list, palette, show_data=False)


def test(model_class, x, name, ransac_gen, ac_tester, plotter=None,
         run_regular=True):
    print(name, x.shape)

    output_prefix = '../results/' + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(x.shape[0],
                                                            ransac_gen,
                                                            ac_tester)
    print('Preference matrix size:', pref_matrix.shape)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running compressed bi-clustering')
    compression_level = 128
    deflator = bc.deflation.L1CompressedDeflator(pref_matrix, compression_level)
    run_biclustering(model_class, x, orig_models, pref_matrix, deflator,
                     ac_tester, output_prefix + '_bic_comp',
                     plotter=plotter)

    if run_regular:
        print('Running regular bi-clustering')
        deflator = bc.deflation.Deflator(pref_matrix)
        run_biclustering(model_class, x, orig_models, pref_matrix, deflator,
                         ac_tester, output_prefix + '_bic_reg')
