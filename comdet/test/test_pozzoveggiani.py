from __future__ import absolute_import
import os
import sys
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import scipy.io
import comdet.pme.plane as plane
import comdet.pme.sampling as sampling
import comdet.pme.acontrario as ac
import comdet.test.utils as utils
import comdet.test.test_3d as test_3d


class Projector(object):
    def __init__(self, data, visibility, proj_mat, dirname_in,
                 dirname_out=None):
        self.data = data
        self.visibility = visibility
        self.proj_mat = proj_mat
        self.dirname_in = dirname_in
        self.dirname_out = dirname_out

    def _project(self, points, k):
        n = points.shape[0]
        data_homogeneous = np.hstack((points, np.ones((n, 1))))
        img_data = data_homogeneous.dot(self.proj_mat[:, :, k].T)
        img_data /= np.atleast_2d(img_data[:, 2]).T
        return img_data

    def plot(self, mod_inliers_list, palette, show_data=True):

        for i, filename in enumerate(os.listdir(self.dirname_in)):
            plt.figure()
            self.inner_plot(mod_inliers_list, palette, filename,
                            show_data=show_data)
            plt.close()

    def inner_plot(self, mod_inliers_list, palette, filename, show_data=True):
        try:
            idx = int(filename[-7:-4])
            k = idx - 1
        except ValueError:
            return
        if np.any(np.isnan(self.proj_mat[:, :, k])):
            return

        img = PIL.Image.open(self.dirname_in + filename).convert('L')

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.hold(True)

        for (mod, inliers), color in zip(mod_inliers_list, palette):
            visible = np.logical_and(self.visibility[:, k], inliers)
            if visible.sum() < 3:
                continue

            img_data = self._project(self.data[visible, :], k)
            plt.scatter(img_data[:, 0], img_data[:, 1], c='w')

            lower = self.data[visible, :].min(axis=0)
            upper = self.data[visible, :].max(axis=0)
            limits = [(lower[i], upper[i]) for i in range(self.data.shape[1])]
            points = mod.plot_points(limits[0], limits[1], limits[2])
            if not points:
                continue
            points = np.array(points)
            img_points = self._project(points, k)
            plt.fill(img_points[:, 0], img_points[:, 1], color=color, alpha=0.5)

        if self.dirname_out is not None:
            plt.savefig(self.dirname_out + filename + '.pdf', dpi=600)


def run(subsampling=1):
    sys.stdout = utils.Logger('pozzoveggiani_s{0}.txt'.format(subsampling))
    inliers_threshold = 0.5
    sigma = 1
    epsilon = 0

    name = 'PozzoVeggiani'
    dirname = '../data/PozzoVeggiani/'

    mat = scipy.io.loadmat(dirname + 'Results.mat')
    data = mat['Points'].T
    proj_mat = mat['Pmat']
    visibility = mat['Visibility']

    # Removing far away points for display
    keep = reduce(np.logical_and, [data[:, 0] > -10, data[:, 0] < 20,
                                   data[:, 2] > 10, data[:, 2] < 45])
    data = data[keep, :]
    visibility = visibility[keep, :]
    # Re-order dimensions and invert vertical direction to get upright data
    data[:, 1] *= -1
    data = np.take(data, [0, 2, 1], axis=1)
    proj_mat[:, 1, :] *= -1
    proj_mat = np.take(proj_mat, [0, 2, 1, 3], axis=1)

    # subsample the input points
    points_considered = np.arange(0, data.shape[0], subsampling)
    data = data[points_considered, :]
    visibility = visibility[points_considered, :]

    n_samples = data.shape[0] * 2
    sampler = sampling.GaussianLocalSampler(sigma, n_samples)
    ransac_gen = sampling.ModelGenerator(plane.Plane, data, sampler)
    ac_tester = ac.LocalNFA(data, epsilon, inliers_threshold)

    projector = Projector(data, visibility, proj_mat, dirname)

    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print 'seed:', seed
    np.random.seed(seed)

    output_prefix = name + '_n{0}'.format(data.shape[0])
    test_3d.test(plane.Plane, data, output_prefix, ransac_gen, ac_tester,
                 plotter=projector)

    plt.close('all')


if __name__ == '__main__':
    run(subsampling=10)
    run(subsampling=5)
    run(subsampling=2)
    run(subsampling=1)
    # plt.show()
