from __future__ import absolute_import
import numpy as np
import scipy.spatial.distance as distance
import itertools


class UniformSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        for _ in range(self.n_samples):
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False)
            yield sample


class AdaptiveSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        self.distribution = None

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        self.distribution = np.zeros((n_elements,))
        for _ in range(self.n_samples):
            dist_max = self.distribution.max()
            if dist_max > 0:
                probas = dist_max - self.distribution
            else:
                probas = np.ones((n_elements,))
            probas /= probas.sum()
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False, p=probas)
            yield sample


class GaussianLocalSampler(object):
    def __init__(self, sigma, n_samples=None):
        self.n_samples = n_samples
        # p(x[i] | x[j]) = exp(-(dist(x[i], x[j])) / sigma)
        self.var = sigma ** 2
        self.distribution = None

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        self.distribution = np.zeros((n_elements,))
        for _ in range(self.n_samples):
            dist_max = self.distribution.max()
            if dist_max > 0:
                probas = dist_max - self.distribution
            else:
                probas = np.ones((n_elements,))
            probas /= probas.sum()
            while True:
                j = np.random.choice(all_elems, p=probas)
                dists = distance.cdist(x, np.atleast_2d(x[j]), 'euclidean')
                bins = np.squeeze(np.exp(-(dists ** 2) / self.var))
                bins /= bins.sum()
                if np.count_nonzero(bins) >= min_sample_size:
                    break
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False, p=bins)
            yield sample


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

