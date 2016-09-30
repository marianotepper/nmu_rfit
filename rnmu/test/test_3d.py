from __future__ import absolute_import, print_function
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn.apionly as sns
import numpy as np
import scipy.sparse as sp
import timeit
import scipy.io
import re
import rnmu.pme.preference as pref
import rnmu.approximation as bc
import rnmu.test.utils as test_utils


class BasePlotter(object):
    def __init__(self, data, normalize_axes=True):
        self.data = data
        self.filename_prefix_out = None
        self.normalize_axes = normalize_axes

    def base_plot(self, size=2, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c='w',
                   marker='o', s=size)

        if self.normalize_axes:
            lower = self.data.min(axis=0)
            upper = self.data.max(axis=0)
            med = (upper + lower) / 2
            max_diff = np.max(upper - lower) / 2
            max_diff *= 1.01
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

    def plot_final_models(self, mod_inliers_list, palette, filename=None,
                          show_data=True, save_animation=False):
        if show_data:
            size = 2
        else:
            size = 0
        fig, ax = self.base_plot(size=size)
        for (mod, inliers), color in zip(mod_inliers_list, palette):
            inliers = np.squeeze(inliers.toarray())
            lower = self.data[inliers, :].min(axis=0)
            upper = self.data[inliers, :].max(axis=0)
            limits = zip(lower, upper)
            mod.plot(ax, limits=limits, color=color, linewidth=5, alpha=0.7)

        if filename is not None:
            plt.savefig(filename + '.pdf', dpi=600)

        if save_animation:
            BasePlotter.save_animation(fig, ax, filename)

    def plot_original_models(self, original_models, bic_list, palette,
                             filename=None):
        fig, ax = self.base_plot(size=5)
        for i, (lf, rf) in enumerate(bic_list):
            inliers = sp.find(lf)[0]
            lower = self.data[inliers, :].min(axis=0)
            upper = self.data[inliers, :].max(axis=0)
            limits = [(lower[k], upper[k]) for k in range(self.data.shape[1])]
            for j in sp.find(rf)[1]:
                original_models[j].plot(ax, limits=limits, color=palette[i],
                                        alpha=0.5)

        if filename is not None:
            plt.savefig(filename + '.pdf', dpi=600)

    @staticmethod
    def save_animation(fig, ax, filename):
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


def run_biclustering(model_class, x, pref_matrix, comp_level, thresholder,
                     ac_tester, output_prefix, plotter=None, palette='Set1',
                     save_animation=True):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, x, thresholder, ac_tester,
                                        bic_list, check_overlap=True)
    bic_groups = [bic[0] for bic in bic_list]

    palette = sns.color_palette(palette, len(bic_list), desat=.5)

    mod_inliers_list = zip(models, bic_groups)

    filename = output_prefix + '_final_models'
    plotter.plot_final_models(mod_inliers_list, palette, filename=filename,
                              save_animation=save_animation)
    plotter.plot_final_models(mod_inliers_list, palette, show_data=False,
                              filename=filename + '_clean',
                              save_animation=save_animation)

    try:
        special_plot = plotter.special_plot
    except AttributeError:
        pass
    else:
        plotter.filename_prefix_out = output_prefix
        special_plot(mod_inliers_list, palette)


def test(model_class, x, name, ransac_gen, thresholder, ac_tester,
         compression_level=32, plotter=None, run_regular=True,
         save_animation=True):
    print(name, x.shape)

    output_dir = '../results/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_prefix = output_dir + name

    if plotter is None:
        plotter = BasePlotter(x)

    plotter.base_plot()
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(ransac_gen,
                                                            thresholder,
                                                            ac_tester)
    print('Preference matrix size:', pref_matrix.shape)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix})

    print('Running compressed bi-clustering')
    run_biclustering(model_class, x, pref_matrix, compression_level,
                     thresholder, ac_tester, output_prefix + '_bic_comp',
                     plotter=plotter, save_animation=save_animation)

    if run_regular:
        print('Running regular bi-clustering')
        compression_level = None
        run_biclustering(model_class, x, pref_matrix, compression_level,
                         thresholder, ac_tester, output_prefix + '_bic_reg',
                         plotter=plotter, save_animation=save_animation)


def plot_times(log_filenames, output_filename, relative=False, col_width=0.35):
    pattern_size = 'Preference matrix size: \((?P<m>.+), (?P<n>.+)\)'
    pref_sizes = []
    comp_time = []
    reg_time = []
    for fn in log_filenames:
        with open(fn, 'r') as f:
            s = str(f.read())
            m = re.search(pattern_size, s).groupdict()
            pref_sizes.append((int(m['m']), int(m['n'])))

            def retrieve_time(pat_type):
                pattern_time = 'Running {0} bi-clustering\nTime: (?P<time>.+)'
                m = re.search(pattern_time.format(pat_type), s)
                if m is None:
                    return np.nan
                else:
                    return float(m.groupdict()['time'])

            comp_time.append(retrieve_time('compressed'))
            reg_time.append(retrieve_time('regular'))

    idx = np.arange(len(log_filenames))
    rse_time = np.array(reg_time)
    arse_time = np.array(comp_time)
    if relative:
        ps = np.array([np.prod(s) for s in pref_sizes])
        rse_time = rse_time / ps
        arse_time = arse_time / ps

    colors = sns.color_palette('Set1', n_colors=2)

    with sns.axes_style('whitegrid'):
        plt.figure()
        plt.yscale('log')

        bars = plt.bar(idx, rse_time, col_width, linewidth=0, color=colors[0])
        bars.set_label('RSE')
        bars = plt.bar(idx + col_width, arse_time, col_width, linewidth=0,
                       color=colors[1])
        bars.set_label('ARSE')

        plt.xticks(idx + col_width, pref_sizes, horizontalalignment='center',
                   fontsize='16')
        _, labels = plt.yticks()
        for l in labels:
            l.set_fontsize(16)

        plt.xlabel('Preference matrix size', fontsize='16')
        if relative:
            plt.ylabel('Time / size', fontsize='16')
            loc = 'upper right'
        else:
            plt.ylabel('Time (s)', fontsize='16')
            loc = 'upper left'

        plt.legend(ncol=2, fontsize='16', loc=loc)

        plt.xlim(-col_width, idx.size + col_width / 2)
        plt.tight_layout()

        if relative:
            output_filename += '_relative'
        output_filename += '.pdf'
        plt.savefig(output_filename, dpi=600)
