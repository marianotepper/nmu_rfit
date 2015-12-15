from __future__ import absolute_import
import numpy as np
import scipy.spatial.distance as distance


class UniformSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def generate(self, x, min_sample_size):
        n_elements = x.shape[0]
        for i in range(self.n_samples):
            while True:
                sample = np.random.randint(0, n_elements, size=min_sample_size)
                if np.unique(sample).size == min_sample_size:
                    break
            yield sample


class GaussianLocalSampler(object):
    def __init__(self, sigma, n_samples=None):
        self.n_samples = n_samples
        # p(x[i] | x[j]) = exp(-(dist(x[i], x[j])) / sigma)
        self.var = sigma ** 2
        self.distribution = None

    def generate(self, x, min_sample_size):
        n_elements = x.shape[0]
        self.distribution = np.zeros((n_elements,))
        for _ in range(self.n_samples):
            bins = np.cumsum(self.distribution.max() - self.distribution)
            bins /= bins[-1]
            rnd = np.random.random()
            j = np.searchsorted(bins, rnd)
            # j = np.random.randint(0, n_elements)
            dists = distance.cdist(x, np.atleast_2d(x[j, :]), 'euclidean')
            bins = np.cumsum(np.exp(-(dists ** 2) / self.var))
            bins /= bins[-1]
            while True:
                rnd = np.random.random((min_sample_size - 1,))
                sample = np.searchsorted(bins, rnd)
                sample = np.hstack((sample, [j]))
                if np.unique(sample).size == min_sample_size:
                    break
            self.distribution[sample] += 1
            yield sample


def model_distance_generator(model_class, elements, sampler):
    samples = sampler.generate(elements, model_class().min_sample_size)
    for s in samples:
        ms_set = np.take(elements, s, axis=0)
        model = model_class()
        model.fit(ms_set)
        yield model, model.distances(elements)


def inliers_generator(mdg, threshold):
    for model, dist in mdg:
        yield model, dist <= threshold


def ransac_generator(model_class, elements, sampler, inliers_threshold):
    mdg = model_distance_generator(model_class, elements, sampler)
    return inliers_generator(mdg, inliers_threshold)


# if __name__ == '__main__':
#     x = np.random.rand(100, 2)
#     sampler = GaussianLocalSampler(0.1)
#     list(sampler.generate(x, 1, 2))
