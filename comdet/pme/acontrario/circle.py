import numpy as np
import comdet.pme.acontrario.utils as utils


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon)
        self.inliers_threshold = inliers_threshold
        self.area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))

    def threshold(self, model):
        return self.inliers_threshold

    def _random_probability(self, model, data=None, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        # (a + b)**2 - (a - b)**2 == 4ab
        ring_area = np.pi * 4 * model.radius * inliers_threshold
        return min(ring_area / self.area, 1)


class LocalNFA(object):
    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, data=None, inliers_threshold=None):
        if data is None:
            data = self.data
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold

        if model.radius < inliers_threshold:
            return np.inf

        upper_threshold = inliers_threshold * 3
        region_mask = model.distances(data) <= upper_threshold

        p = inliers_threshold / upper_threshold
        k = n_inliers - model.min_sample_size
        pfa = utils.log_binomial(region_mask.sum(), k, p)
        n_tests = utils.log_nchoosek(data.shape[0], model.min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon

    def threshold(self, model):
        return self.inliers_threshold
