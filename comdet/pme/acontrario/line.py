import numpy as np
import comdet.pme.acontrario.utils as utils


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def threshold(self, model):
        return self.inliers_threshold

    def _total_and_probability(self, model, data, inliers_threshold):
        area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
        _, s = model.project(data)
        length = s.max() - s.min()
        return length * 2 * inliers_threshold / area


class LocalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LocalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def _total_and_probability(self, model, data, inliers_threshold):
        dist = model.distances(data)
        upper_threshold = np.maximum(inliers_threshold * 3,
                                     np.min(dist[dist > inliers_threshold]))
        region_mask = dist <= upper_threshold
        p = inliers_threshold / upper_threshold
        return region_mask.sum(), p

    def threshold(self, model):
        return self.inliers_threshold