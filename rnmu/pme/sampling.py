from __future__ import absolute_import
import numpy as np
import scipy.spatial.distance as distance
import itertools
import collections


class SampleSet(collections.MutableSet):
    def __init__(self):
        self._dict = {}
        self._len = 0

    def __contains__(self, sample):
        try:
            return reduce(lambda d, k: d[k], sample, self._dict)
        except KeyError:
            return False

    def __len__(self):
        return self._len

    def add(self, sample):
        d = self._dict
        for i, s in enumerate(sample):
            if i == len(sample) - 1:
                d[s] = True
                continue
            if s not in d:
                d[s] = {}
            d = d[s]
        self._len += 1

    def discard(self, sample):
        pass

    def __iter__(self):
        pass


class UniformSampler(object):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        self.sample_set = SampleSet()

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        for _ in range(self.n_samples):
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False)
            if sample not in self.sample_set:
                self.sample_set.add(sample)
                yield sample


class GaussianLocalSampler(object):
    def __init__(self, sigma, n_samples=None):
        self.n_samples = n_samples
        # p(x[i] | x[j]) = exp(-(dist(x[i], x[j])) / sigma)
        self.var = sigma ** 2
        self.sample_set = SampleSet()

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        for _ in range(self.n_samples):
            while True:
                j = np.random.choice(all_elems)
                dists = distance.cdist(x, np.atleast_2d(x[j]), 'sqeuclidean')
                bins = np.squeeze(np.exp(-dists / self.var))
                bins /= bins.sum()
                if np.count_nonzero(bins) >= min_sample_size:
                    break
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False, p=bins)
            if sample not in self.sample_set:
                self.sample_set.add(sample)
                yield sample


class ModelGenerator(object):
    def __init__(self, model_class, sampler):
        self.model_class = model_class
        self.elements = None
        self.sampler = sampler

    @property
    def n_samples(self):
        return self.sampler.n_samples

    def __iter__(self):
        def generate(s):
            ms_set = np.take(self.elements, s, axis=0)
            return self.model_class(ms_set)
        samples = self.sampler.generate(self.elements,
                                        self.model_class().min_sample_size)
        return itertools.imap(generate, samples)
