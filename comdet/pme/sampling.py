from __future__ import absolute_import
import numpy as np
import scipy.spatial.distance as distance
import itertools


class SimpleSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        for _ in range(self.n_samples):
            while True:
                sample = np.random.randint(0, n_elements, size=min_sample_size)
                if np.unique(sample).size == min_sample_size:
                    break
            yield sample


class UniformSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        distribution = np.zeros((n_elements,))
        for _ in range(self.n_samples):
            bins = np.cumsum(distribution.max() - distribution)
            if bins[-1] > 0:
                rnd = np.random.randint(0, bins[-1], size=min_sample_size)
                sample = np.searchsorted(bins, rnd)
            else:
                sample = np.array([], dtype=np.int)
            unique_sample = np.unique(sample)
            if unique_sample.size < min_sample_size:
                remainder = min_sample_size - unique_sample.size
                comp = np.random.randint(0, n_elements, size=remainder)
                sample = np.append(sample, [comp])
            distribution[sample] += 1
            yield sample


class GaussianLocalSampler(object):
    def __init__(self, sigma, n_samples=None):
        self.n_samples = n_samples
        # p(x[i] | x[j]) = exp(-(dist(x[i], x[j])) / sigma)
        self.var = sigma ** 2

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        distribution = np.zeros((n_elements,))
        counter_samples = 0
        counter_total = 0
        while (counter_samples < self.n_samples and
               counter_total < self.n_samples * 100):
            bins = np.cumsum(distribution.max() - distribution)
            bins /= bins[-1]
            rnd = np.random.random()
            j = np.searchsorted(bins, rnd)
            dists = distance.cdist(x, np.atleast_2d(x[j]), 'euclidean')
            bins = np.cumsum(np.exp(-(dists ** 2) / self.var))
            bins /= bins[-1]

            success = False
            for _ in range(100):
                rnd = np.random.random((min_sample_size - 1,))
                sample = np.searchsorted(bins, rnd)
                sample = np.append(sample, [j])
                if np.unique(sample).size == min_sample_size:
                    success = True
                    break

            if success:
                distribution[sample] += 1
                counter_samples += 1
                yield sample
            counter_total += 1


def model_generator(model_class, elements, sampler):
    samples = sampler.generate(elements, model_class().min_sample_size)
    for s in samples:
        ms_set = np.take(elements, s, axis=0)
        model = model_class()
        model.fit(ms_set)
        yield model


def inliers(model, elements, threshold):
    return model.distances(elements) <= threshold


def inliers_generator(mg, elements, threshold):
    return itertools.imap(lambda m: (m, inliers(m, elements, threshold)), mg)


def ransac_generator(model_class, elements, sampler, inliers_threshold):
    mg = model_generator(model_class, elements, sampler)
    return inliers_generator(mg, elements, inliers_threshold)


# if __name__ == '__main__':
#     x = np.random.rand(100, 2)
#     sampler = GaussianLocalSampler(0.1)
#     list(sampler.generate(x, 1, 2))
