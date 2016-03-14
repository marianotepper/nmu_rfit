import numpy as np
from abc import ABCMeta, abstractmethod


class GlobalThresholder(object):
    def __init__(self, inliers_threshold):
        self.inliers_threshold = inliers_threshold

    def membership(self, model, data):
        return global_membership(model.distances(data), self.inliers_threshold)


class LocalThresholder(object):
    def __init__(self, inliers_threshold, ratio=3.):
        self.inliers_threshold = inliers_threshold
        self.ratio = ratio

    def membership(self, model, data):
        return local_membership(model.distances(data), self.inliers_threshold,
                                self.ratio)


class AdaptiveThresholder(object):
    __metaclass__ = ABCMeta

    def __init__(self, tester, threshold_list):
        self.tester = tester
        self.threshold_list = threshold_list

    def membership(self, model, data):
        distances = model.distances(data)

        best_nfa = self.tester.epsilon
        best_membership = None
        for thresh in self.threshold_list:
            membership = self._inner_membership(distances, thresh)
            nfa = self.tester.nfa(membership)
            if nfa < best_nfa:
                best_nfa = nfa
                best_membership = membership

        if best_membership is None:
            best_membership = distances.copy()
            best_membership[:] = np.nan

        return best_membership

    @abstractmethod
    def _inner_membership(self, distances, inliers_threshold):
        pass


class AdaptiveGlobalThresholder(AdaptiveThresholder):
    def __init__(self, tester, threshold_list):
        super(AdaptiveGlobalThresholder, self).__init__(tester, threshold_list)

    def _inner_membership(self, distances, inliers_threshold):
        return global_membership(distances, inliers_threshold)


class AdaptiveLocalThresholder(object):
    def __init__(self, tester, threshold_list, ratio=3.):
        super(AdaptiveGlobalThresholder, self).__init__(tester, threshold_list)
        self.ratio = ratio

    def _inner_membership(self, distances, inliers_threshold):
        return local_membership(distances, inliers_threshold, self.ratio)


def global_membership(distances, inliers_threshold):
    return (distances <= inliers_threshold) * inliers_threshold


def local_membership(distances, inliers_threshold, ratio):
    membership = distances <= inliers_threshold

    if np.all(membership):
        return membership * inliers_threshold

    outliers = np.logical_not(membership)
    upper_threshold = inliers_threshold * ratio
    upper_threshold = np.maximum(upper_threshold, np.min(distances[outliers]))
    upper_threshold = np.minimum(upper_threshold, np.max(distances[outliers]))

    membership *= inliers_threshold
    membership[distances > upper_threshold] = np.nan
    return membership
