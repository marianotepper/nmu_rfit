import numpy as np
import comdet.pme.acontrario.utils as utils


class GlobalNFA(utils.BinomialNFA):
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
        return len(data), inliers_mask.sum(), p


class LocalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LocalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold):
        dist = model.distances(data)
        upper_threshold = np.maximum(inliers_threshold * 3,
                                     np.min(dist[dist > inliers_threshold]))
        n = self.inner_inliers(dist, upper_threshold).sum()
        k = self.inner_inliers(dist, inliers_threshold).sum()
        p = inliers_threshold / upper_threshold
        return n, k, p

    def threshold(self, model):
        return self.inliers_threshold
