from __future__ import absolute_import
import numpy as np
import itertools
from . import sampling


class ModelGenerator(object):
    def __init__(self, model_class, n_samples, batch=10, h_ratio=.1):
        self.model_class = model_class
        self.elements = None
        self.n_samples = n_samples
        self.batch = batch
        self.h_ratio = h_ratio
        self.bias = None
        self.sample_set = sampling.SampleSet()

    def __iter__(self):
        def generate(s):
            ms_set = np.take(self.elements, s, axis=0)
            return self.model_class(ms_set)

        n_elements = self.elements.shape[0]
        min_sample_size = self.model_class().min_sample_size

        sampler = sampling.UniformSampler(min(self.batch, self.n_samples))
        mss_samples = sampler.generate(self.elements, min_sample_size)

        residuals = None
        for m in itertools.imap(generate, mss_samples):
            yield m
            residuals = self._add_model(residuals, m)

        all_elems = np.arange(n_elements)
        for i in range(self.n_samples - self.batch):
            if i % self.batch == 0:
                ranking = self._rank(residuals)

            probas = self.bias
            sample = []
            for j in range(min_sample_size):
                k = np.random.choice(all_elems, p=probas)
                sample.append(k)
                pk = ModelGenerator._compute_probabilities(ranking, k)
                if j == 0:
                    probas = pk
                else:
                    probas *= pk
                probas /= probas.sum()

            if sample not in self.sample_set:
                self.sample_set.add(sample)
                m = generate(sample)
                yield m
                residuals = self._add_model(residuals, m)

    def _add_model(self, residuals, m):
        dist = np.atleast_2d(m.distances(self.elements)).T
        if residuals is None:
            return dist
        else:
            return np.append(residuals, dist, axis=1)

    def _rank(self, residuals):
        h = int(np.ceil(residuals.shape[1] * self.h_ratio))
        ranking = np.argpartition(residuals, h, axis=1)
        return ranking[:, :h]

    @staticmethod
    def _compute_probabilities(ranking, k):
        def intersect_ratio(a, b):
            return float(np.intersect1d(a, b, assume_unique=True).size) / a.size
        p = np.apply_along_axis(intersect_ratio, 1, ranking, ranking[k, :])
        p[k] = 0
        return p

    def apply_distribution(self, distribution):
        try:
            dist_max = distribution.max()
            if dist_max > 0:
                self.bias = dist_max - distribution
                self.bias /= self.bias.sum()
            else:
                self.bias = None

        except AttributeError:
            pass
