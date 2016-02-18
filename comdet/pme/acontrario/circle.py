import numpy as np
from . import nfa


class GlobalNFA(nfa.BinomialNFA):
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


class LocalNFA(nfa.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LocalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold

    def _binomial_params(self, model, data, inliers_threshold, ratio=2):
        if model.radius <= inliers_threshold:
            raise ValueError('Inliers threshold too big for this circle')

        dist = model.distances(data)
        dist_abs = np.abs(dist)
        inliers = dist_abs <= inliers_threshold
        k = inliers.sum()
        outliers = dist_abs > inliers_threshold
        if outliers.sum() == 0:
            return k, k, 1

        min_dist = np.min(dist_abs[dist_abs > inliers_threshold])
        upper_threshold = np.maximum(inliers_threshold * (ratio + 1), min_dist)
        region_in = np.logical_and(dist >= -upper_threshold,
                                   dist < -inliers_threshold)
        region_out = np.logical_and(dist <= upper_threshold,
                                    dist > inliers_threshold)
        n_in = region_in.sum()
        n_out = region_out.sum()
        n = k
        ring_inliers = ring_area(model.radius + inliers_threshold,
                                 model.radius - inliers_threshold)
        if n_in == 0:
            ring_in = 0
        else:
            n += n_in
            if model.radius < upper_threshold:
                ring_in = ring_area(model.radius - inliers_threshold, 0)
            else:
                ring_in = ring_area(model.radius - inliers_threshold,
                                    model.radius - upper_threshold)
        if n_out == 0:
            ring_out = 0
        else:
            n += n_out
            ring_out = ring_area(model.radius + upper_threshold,
                                    model.radius + inliers_threshold)
        p = ring_inliers / (ring_inliers + ring_in + ring_out)
        return n, k, p

    def threshold(self, model):
        return self.inliers_threshold


def ring_area(upper, lower):
    return np.pi * (upper ** 2 - lower ** 2)
