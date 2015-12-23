import numpy as np
import comdet.pme.acontrario.utils as utils


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold
        self.area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))

    def threshold(self, model):
        return self.inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold):
        inliers_mask = self.inliers(model, data, inliers_threshold)
        # (a + b)**2 - (a - b)**2 == 4ab
        ring_area = np.pi * 4 * model.radius * inliers_threshold
        p = min(ring_area / self.area, 1)
        return len(data), inliers_mask.sum(), p


class LocalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LocalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold):
        if model.radius <= inliers_threshold:
            raise ValueError('Inliers threshold too big for this circle')

        dist = model.distances(data)
        upper_threshold = np.maximum(inliers_threshold * 3,
                                     np.min(dist[dist > inliers_threshold]))
        n = self.inner_inliers(dist, upper_threshold).sum()
        k = self.inner_inliers(dist, inliers_threshold).sum()

        if model.radius < upper_threshold:
            p = 4 * model.radius * inliers_threshold
            p /= (model.radius + upper_threshold) ** 2
        else:
            p = inliers_threshold / upper_threshold

        return n, k, p

    def threshold(self, model):
        return self.inliers_threshold
