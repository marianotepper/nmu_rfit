from __future__ import absolute_import, print_function
import os
import sys
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import scipy.io
import arse.pme.plane as plane
import arse.pme.sampling as sampling
import arse.pme.acontrario as ac
import arse.test.utils as utils
import arse.test.test_3d as test_3d


class Projector(test_3d.BasePlotter):
    def __init__(self, data, visibility, proj_mat, dirname_in):
        super(Projector, self).__init__(data)
        self.visibility = visibility
        self.proj_mat = proj_mat
        self.dirname_in = dirname_in

    def _project(self, points, k):
        n = points.shape[0]
        data_homogeneous = np.hstack((points, np.ones((n, 1))))
        img_data = data_homogeneous.dot(self.proj_mat[:, :, k].T)
        img_data /= np.atleast_2d(img_data[:, 2]).T
        return img_data

    def special_plot(self, mod_inliers_list, palette):
        if not os.path.exists(self.filename_prefix_out):
            os.mkdir(self.filename_prefix_out)
        self.filename_prefix_out += '/'

        for i, filename in enumerate(os.listdir(self.dirname_in)):
            plt.figure()
            self.inner_plot(mod_inliers_list, palette, filename)
            plt.close()

    def inner_plot(self, mod_inliers_list, palette, filename):
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
            inliers = np.squeeze(inliers.toarray())
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

        if self.filename_prefix_out is not None:
            plt.savefig(self.filename_prefix_out + filename + '.pdf', dpi=600)


def run(subsampling=1, inliers_threshold=0.2):
    logger = utils.Logger('pozzoveggiani_s{0}.txt'.format(subsampling))
    sys.stdout = logger

    sigma = 1
    epsilon = 0

    name = 'PozzoVeggiani'
    dirname = '../data/' + name + '/'

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

    n_samples = data.shape[0]
    sampler = sampling.GaussianLocalSampler(sigma, n_samples)
    ransac_gen = sampling.ModelGenerator(plane.Plane, data, sampler)
    ac_tester = ac.LocalNFA(data, epsilon, inliers_threshold)

    projector = Projector(data, visibility, proj_mat, dirname)

    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print('seed:', seed)
    np.random.seed(seed)

    output_prefix = name + '_n{0}'.format(data.shape[0])
    test_3d.test(plane.Plane, data, output_prefix, ransac_gen, ac_tester,
                 plotter=projector, run_regular=True)

    plt.close('all')

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    run(subsampling=10, inliers_threshold=0.2)
    run(subsampling=5, inliers_threshold=0.2)
    run(subsampling=2, inliers_threshold=0.2)
    run(subsampling=1, inliers_threshold=0.2)


if __name__ == '__main__':
    run_all()
    plt.show()
