from __future__ import absolute_import, print_function
import itertools
import sys
import struct
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import comdet.pme.plane as plane
import comdet.pme.sampling as sampling
import comdet.pme.acontrario as ac
import comdet.test.utils as utils
import comdet.test.test_3d as test_3d
import comdet.test.utils as test_utils


class RangePlotter(test_3d.BasePlotter):
    def __init__(self, data, width, height, dirname_out=None,
                 save_animation=True):
        super(RangePlotter, self).__init__(data, normalize_axes=False)
        self.width = width
        self.height = height
        self.save_animation = save_animation

    def surface_plot(self, cmap=None, facecolors=None, filename=None):
        img_data = np.reshape(self.data, (self.width, self.height, 3))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(img_data[:, :, 0], img_data[:, :, 1], img_data[:, :, 2],
                        facecolors=facecolors, cmap=cmap, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, alpha=0.5)
        if filename is not None:
            plt.savefig(filename, dpi=600)
            if self.save_animation:
                test_3d.BasePlotter.save_animation(fig, ax, filename)

    def special_plot(self, mod_inliers_list, palette):
        membership = np.zeros((self.width * self.height, 3))
        for (mod, inliers), color in zip(mod_inliers_list, palette):
            membership[inliers, :] = color
        membership = np.reshape(membership, (self.width, self.height, 3))
        plt.figure()
        plt.imshow(membership, interpolation='none')
        if self.filename_prefix_out is not None:
            filename_prefix = self.filename_prefix_out + 'final_models'
            plt.savefig(filename_prefix + '_2d.pdf',
                        dpi=600)

        self.surface_plot(facecolors=membership,
                          filename=filename_prefix + '_3d.pdf')


def read_rasterfile(filename, subsampling=None):
    with open(filename, 'rb') as f:
        header = struct.unpack('>' + 'i' * 8, f.read(32))
        if header[3] == 1:
            raise NotImplementedError('Cannot read rasterfile with depth=1.')
        length = header[4]
        range_values = struct.unpack('>' + 'B' * length, f.read(length))
        width = header[1]
        height = header[2]
        range_values = np.reshape(range_values, (width, height))

        if subsampling is None:
            range_values = np.reshape(range_values,
                                      (width, height)).astype(np.float)
        else:
            img = PIL.Image.fromarray(range_values.astype(np.float))
            img = img.resize((width / subsampling, height / subsampling))
            width = img.size[0]
            height = img.size[1]
            range_values = np.array(img)

        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        data = np.dstack((x, y, range_values))
        data = np.reshape(data, (width * height, 3))
        return data, width, height


def ground_truth(gt_seg, data=None, ac_tester=None):
    association = np.ravel(gt_seg[:, 2])
    if data is None or ac_tester is None:
        return [association == v for v in np.unique(association)]
    else:
        gt_groups = []
        for v in np.unique(association):
            inliers = association == v
            if inliers.sum() < 3:
                continue
            model = plane.Plane(data=data[inliers])
            if ac_tester.meaningful(model):
                gt_groups.append(inliers)
        return gt_groups


def run(compression_level, subsampling):
    logger = utils.Logger('abw_cl{0}_s{1}.txt'.format(compression_level,
                                                          subsampling))
    sys.stdout = logger

    inliers_threshold = 1
    epsilon = 0
    sampling_factor = 5  # fraction of the total number of elements

    dirname = '../data/ABW-TRAIN-IMAGES/'
    stats_list = []
    for idx in range(10):
        if idx != 2:
            continue
        filename = 'abw.train.{0}'.format(idx)

        output_prefix = '../results/' + filename

        data, width, height = read_rasterfile(dirname + filename + '.range',
                                              subsampling=subsampling)

        n_samples = int(data.shape[0] * sampling_factor)
        sampler = sampling.AdaptiveSampler(n_samples)
        ransac_gen = sampling.ModelGenerator(plane.Plane, data, sampler)
        ac_tester = ac.LocalNFA(data, epsilon, inliers_threshold)
        plotter = RangePlotter(data, width, height, save_animation=False)

        gt_seg, _, _ = read_rasterfile(dirname + filename + '.gt-seg',
                                       subsampling=subsampling)
        gt_groups = ground_truth(gt_seg, data=data, ac_tester=ac_tester)

        plt.figure()
        plt.imshow(np.reshape(data[:, 2], (width, height)), cmap=cm.RdYlBu_r,
                   interpolation='none')
        plt.savefig(output_prefix + '_data_2d.pdf', dpi=600)

        plotter.surface_plot(cmap=cm.RdYlBu_r,
                             filename=output_prefix + '_data_3d.pdf')

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        output_prefix = filename + '_cl{0}_s{1}'.format(compression_level,
                                                        subsampling)
        res = test_3d.test(plane.Plane, data, output_prefix, ransac_gen,
                           ac_tester, plotter=plotter, run_regular=False,
                           compression_level=compression_level,
                           gt_groups=gt_groups, save_animation=False)
        stats_list.append(res)

        print('-'*40)
        plt.close('all')

    print('Statistics of compressed bi-clustering')
    test_utils.print_stats(stats_list)
    print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    # compresion_levels = [8, 16, 32, 64, 128]
    # subsamplings = [20, 10, 5, 2, 1]
    compresion_levels = [8]
    subsamplings = [10]
    for s, cl in itertools.product(subsamplings, compresion_levels):
        run(cl, s)


if __name__ == '__main__':
    run_all()
    plt.show()