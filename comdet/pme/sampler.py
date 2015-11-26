from __future__ import absolute_import
import numpy as np


def random_sampler(n_elements, n_samples, min_sample_size):
    for _ in range(n_samples):
        while True:
            sample = np.random.randint(0, n_elements, size=min_sample_size)
            if np.unique(sample).size == min_sample_size:
                yield sample


def model_distance_generator(model_class, elements, n_samples):
    model = model_class()
    rs = random_sampler(len(elements), n_samples, model.min_sample_size())
    for sample in rs:
        ms_set = np.take(elements, sample, axis=0)
        model.fit(ms_set)
        yield model.distances(elements).shape


def test():
    from comdet.pme.line import Line
    x = np.random.rand(100, 2)
    mdg = model_distance_generator(Line, x, 5000)
    for dist in mdg:
        print dist


if __name__ == '__main__':
    test()