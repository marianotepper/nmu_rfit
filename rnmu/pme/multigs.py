from __future__ import absolute_import
import numpy as np
import itertools
from .sampling import UniformSampler, SampleSet


class ModelGenerator(object):
    def __init__(self, model_class, n_samples, batch=10, h_ratio=.1, seed=None):
        self.model_class = model_class
        self.elements = None
        self._n_samples = n_samples
        self._batch = batch
        self._h_ratio = h_ratio
        self._bias = None
        self._sample_set = SampleSet()
        self._uni_sampler = UniformSampler(n_samples=min(batch, n_samples),
                                           seed=seed)
        if seed is not None:
            np.random.seed(seed)

    def __iter__(self):
        def generate(s):
            ms_set = np.take(self.elements, s, axis=0)
            return self.model_class(ms_set)

        n_elements = self.elements.shape[0]
        min_sample_size = self.model_class().min_sample_size

        mss_samples = self._uni_sampler.generate(self.elements, min_sample_size)

        residuals = None
        for m in itertools.imap(generate, mss_samples):
            yield m
            residuals = self._add_model(residuals, m)

        all_elems = np.arange(n_elements)
        for i in range(self._n_samples - self._batch):
            if i % self._batch == 0:
                ranking = self._rank(residuals)

            probas = self._bias
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

            if sample not in self._sample_set:
                self._sample_set.add(sample)
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
        h = int(np.ceil(residuals.shape[1] * self._h_ratio))
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
