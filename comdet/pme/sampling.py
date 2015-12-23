from __future__ import absolute_import
import numpy as np
import scipy.spatial.distance as distance
import itertools


class UniformSampler(object):
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


class ModelGenerator(object):
    def __init__(self, model_class, elements, sampler):
        self.model_class = model_class
        self.elements = elements
        self.sampler = sampler

    def __iter__(self):
        def generate(s):
            ms_set = np.take(self.elements, s, axis=0)
            return self.model_class(ms_set)
        samples = self.sampler.generate(self.elements,
                                        self.model_class().min_sample_size)
        return itertools.imap(generate, samples)

