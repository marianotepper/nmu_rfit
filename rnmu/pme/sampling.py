from __future__ import absolute_import
import numpy as np
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
    def __init__(self, n_samples=None, seed=None):
        self.n_samples = n_samples
        self.sample_set = SampleSet()
        if seed is not None:
            np.random.seed(seed)

    def generate(self, x, min_sample_size):
        n_elements = len(x)
        all_elems = np.arange(n_elements)
        for _ in range(self.n_samples):
            sample = np.random.choice(all_elems, size=min_sample_size,
                                      replace=False)
            if sample not in self.sample_set:
                self.sample_set.add(sample)
                yield sample


class ModelGenerator(object):
    def __init__(self, model_class, sampler):
        self._sampler = sampler
        self.model_class = model_class
        self.elements = None

    @property
    def n_samples(self):
        return self._sampler.n_samples

    def __iter__(self):
        def generate(s):
            ms_set = np.take(self.elements, s, axis=0)
            return self.model_class(ms_set)
        samples = self._sampler.generate(self.elements,
                                         self.model_class().min_sample_size)
        return itertools.imap(generate, samples)
