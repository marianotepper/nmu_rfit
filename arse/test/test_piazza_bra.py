from __future__ import absolute_import, print_function
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import arse.pme.plane as plane
import arse.pme.sampling as sampling
import arse.pme.membership as membership
import arse.pme.acontrario as ac
import arse.test.utils as utils
import arse.test.test_3d as test_3d


def run(subsampling=1, inliers_threshold=0.2):
    logger = utils.Logger('piazza_bra_s{0}.txt'.format(subsampling))
    sys.stdout = logger

    sigma = 1
    epsilon = 0
    local_ratio = 3.

    name = 'Piazza_Bra'
    dirname = '../data/' + name + '/'

    mat = scipy.io.loadmat(dirname + 'Samantha_Bra.mat')
    data = mat['Points']

    # subsample the input points
    points_considered = np.arange(0, data.shape[0], subsampling)
    data = data[points_considered, :]

    n_samples = data.shape[0] * 5
    sampler = sampling.GaussianLocalSampler(sigma, n_samples)
    ransac_gen = sampling.ModelGenerator(plane.Plane, data, sampler)
    thresholder = membership.LocalThresholder(inliers_threshold,
                                              ratio=local_ratio)
    min_sample_size = plane.Plane().min_sample_size
    ac_tester = ac.BinomialNFA(epsilon, 1. / local_ratio, min_sample_size)

    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print('seed:', seed)
    np.random.seed(seed)

    output_prefix = name + '_n{0}'.format(data.shape[0])
    test_3d.test(plane.Plane, data, output_prefix, ransac_gen, thresholder,
                 ac_tester, run_regular=True)

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
