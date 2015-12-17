from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import comdet.test.test_3d as test_3d
import comdet.pme.plane as plane
import comdet.pme.sampling as sampling
import comdet.pme.acontrario.plane as ac_plane

n_samples = int(1e4)
inliers_threshold = 0.5
epsilon = 0

name = 'PozzoVeggiani'
dirname = '../data/PozzoVeggiani/'
mat = scipy.io.loadmat(dirname + 'Results.mat')

data = mat['Points'].T
proj_mat = mat['Pmat']
visibility = mat['Visibility']

# Removing outliers
keep = reduce(np.logical_and, [data[:, 0] > -10, data[:, 0] < 20,
                               data[:, 2] > 0, data[:, 2] < 45])
data = data[keep, :]
visibility = visibility[keep, :]
# Re-ordering dimensions
data[:, 1] *= -1
data = np.take(data, [0, 2, 1], axis=1)
proj_mat[:, 1, :] *= -1
proj_mat = np.take(proj_mat, [0, 2, 1, 3], axis=1)

# TODO remove later on
n = data.shape[0]
data = data[np.arange(0, n, 10), :]
visibility = visibility[np.arange(0, n, 10), :]

# sampler = sampling.UniformSampler(n_samples)
# ac_tester = ac_plane.GlobalNFA(data, epsilon, inliers_threshold)
sampler = sampling.GaussianLocalSampler(0.5, n_samples)
ac_tester = ac_plane.LocalNFA(data, epsilon, inliers_threshold)

ransac_gen = sampling.ransac_generator(plane.Plane, data, sampler,
                                       inliers_threshold)

projector = test_3d.Projector(data, visibility, proj_mat, dirname, None)

print '-'*40
seed = 0
# seed = np.random.randint(0, np.iinfo(np.uint32).max)
print 'seed:', seed
np.random.seed(seed)

test_3d.test(plane.Plane, data, name, ransac_gen, ac_tester,
             projector=projector)

plt.show()
