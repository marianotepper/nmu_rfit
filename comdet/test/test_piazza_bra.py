from __future__ import absolute_import, print_function
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


def run(subsampling=1, inliers_threshold=0.2):
    logger = utils.Logger('pozzoveggiani_s{0}.txt'.format(subsampling))
    sys.stdout = logger

    sigma = 1
    epsilon = 0

    name = 'Piazza_Bra'
    dirname = '../data/' + name + '/'

    mat = scipy.io.loadmat(dirname + 'Results.mat')
    data = mat['Points']

    # subsample the input points
    points_considered = np.arange(0, data.shape[0], subsampling)
    data = data[points_considered, :]

    n_samples = data.shape[0] * 5
    sampler = sampling.GaussianLocalSampler(sigma, n_samples)
    # sampler = sampling.AdaptiveSampler(n_samples)
    ransac_gen = sampling.ModelGenerator(plane.Plane, data, sampler)
    ac_tester = ac.LocalNFA(data, epsilon, inliers_threshold)

    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print('seed:', seed)
    np.random.seed(seed)

    output_prefix = name + '_n{0}'.format(data.shape[0])
    test_3d.test(plane.Plane, data, output_prefix, ransac_gen, ac_tester,
                 run_regular=True)

    plt.close('all')

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    run(subsampling=10, inliers_threshold=0.1)
    run(subsampling=5, inliers_threshold=0.1)
    run(subsampling=2, inliers_threshold=0.1)
    run(subsampling=1, inliers_threshold=0.1)


if __name__ == '__main__':
    run_all()
    plt.show()
