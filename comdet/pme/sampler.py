from __future__ import absolute_import
import numpy as np


def random_sampler(n_elements, n_samples, min_sample_size):
    for _ in range(n_samples):
        while True:
            sample = np.random.randint(0, n_elements, size=min_sample_size)
            if np.unique(sample).size == min_sample_size:
                break
        yield sample


def model_distance_generator(model_class, elements, n_samples):

    rs = random_sampler(len(elements), n_samples, model_class().min_sample_size)
    for sample in rs:
        ms_set = np.take(elements, sample, axis=0)
        model = model_class()
        model.fit(ms_set)
        yield model, model.distances(elements)


def inliers_generator(mdg, threshold):
    for model, dist in mdg:
        yield model, dist < threshold
