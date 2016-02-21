import numpy as np
from . import nfa


class GlobalNFA(nfa.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def threshold(self, model):
        return self.inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold):
        inliers_mask = self.inliers(model, data, inliers_threshold)
        volume = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
        _, s = model.project(data)
        area = np.prod(s.max(axis=0) - s.min(axis=0))
        p = area * 2 * inliers_threshold / volume
        k = inliers_mask.sum() - model.min_sample_size
        n = len(data) - model.min_sample_size
        return n, k, p
