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
        area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
        _, s = model.project(data)
        length = s.max() - s.min()
        p = length * 2 * inliers_threshold / area
        return len(data), inliers_mask.sum(), p
