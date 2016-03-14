import numpy as np


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


def global_membership(distances, inliers_threshold):
    return (distances <= inliers_threshold).astype(np.float) * inliers_threshold


def local_membership(distances, inliers_threshold, ratio):
    membership = distances <= inliers_threshold

    if np.all(membership):
        return membership.astype(np.float) * inliers_threshold

    outliers = np.logical_not(membership)
    upper_threshold = inliers_threshold * ratio
    upper_threshold = np.maximum(upper_threshold, np.min(distances[outliers]))
    upper_threshold = np.minimum(upper_threshold, np.max(distances[outliers]))

    membership = membership.astype(np.float) * inliers_threshold
    membership[distances > upper_threshold] = np.nan
    return membership
